"""
eval_policy_proximity_guidance.py — Inference-time proximity-guided policy evaluation.

When the arm comes near an obstacle the policy action is corrected before
execution.  Activation distance: SDF_MARGIN from the arm surface in "sdf"
mode (the default mode; all SDF_* defaults live in differentiable_proximity.py
only); GUIDANCE_THRESHOLD (default 5 cm from link centers) in the legacy modes.
Modes (GUIDANCE_MODE):

"chunk" (default) — CAPE-style trajectory refinement
----------------------------------------------------
Client-side approximation of guidance-inside-denoising (CAPE, Yang et al. 2025,
arXiv:2511.22773).  When the action chunk arrives from the model server, EVERY
waypoint is FK-evaluated against the scene; waypoints whose links fall within
threshold are pushed along the joint-space repulsion gradient

    dq_k += gain · (1 − dist/threshold) · Jᵀ n̂_away

over GUIDANCE_CHUNK_ITERS refinement passes, with a temporal box filter
(GUIDANCE_CHUNK_SMOOTH) standing in for CAPE's re-denoising smoothing.
This perturbs the whole planned trajectory away from obstacles instead of
fighting individual executed actions.

"cbf" — safety filter
---------------------
Enforces a velocity barrier on the policy's own commanded step
dq = action − q_measured for each near-obstacle link:

    approach_speed = n̂ᵀ J dq  ≤  alpha · (dist − margin)

Violating components are projected out analytically (closed-form single-
constraint QP), closest link first.  The filter never adds motion of its own:
moving away from an obstacle is never penalised, so it cannot fight the policy
or inject oscillations.

"repulse" (legacy) — Jacobian repulsion
---------------------------------------
1. Per-link proximity distances via two-phase AABB + trimesh.
2. Top _TOP_K_DANGER closest links below threshold, weighted 1/dist.
3. Per arm: weighted-average repulsion direction away from obstacles,
   dq = DLS-pinv(J) @ d_repulsion, normalised, scaled by
   gain × (1 − dist_closest/threshold), added to the action.
4. Intra-chunk carry-forward accumulator with 0.9/step decay.

Tuning
------
Override defaults via environment variables:
    GUIDANCE_MODE                 default "sdf"  ("chunk" | "cbf" | "repulse")
    SDF_MARGIN           metres   (sdf: standoff from arm surface; default
                                   in differentiable_proximity.py)
    GUIDANCE_THRESHOLD   metres   default 0.05   (legacy modes: activation distance)
    GUIDANCE_CHUNK_GAIN  radians  default 0.05   (chunk: push per waypoint/pass)
    GUIDANCE_CHUNK_ITERS          default 2      (chunk: refinement passes)
    GUIDANCE_CHUNK_SMOOTH         default 5      (chunk: box-filter width)
    GUIDANCE_CHUNK_STRIDE         default 2      (chunk: evaluate every Nth waypoint)
    GUIDANCE_GAIN        radians  default 0.20   (repulse: max correction/step)
    GUIDANCE_CBF_ALPHA            default 0.2    (cbf: approach speed factor)
    GUIDANCE_CBF_MARGIN  metres   default 0.005  (cbf: hard standoff distance)
    EVAL_TEST_NUM                 default 100    (episodes to evaluate)
    EVAL_PROXIMITY_GUIDANCE  1    (set to 0 to run as plain eval, no guidance)

Usage: identical to eval_policy_client.py
    python script/eval_policy_proximity_guidance.py \\
        --config policy/pi05/deploy_policy.yml \\
        --port 9999 \\
        task_name move_book_onto_table \\
        task_config bench_demo_study_clean ...
"""

import sys
import os
import subprocess
import socket
import json
import time
import traceback
import yaml
import argparse
import importlib
import base64
import hashlib

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from script.bench_script.setup_paths import setup_paths
setup_paths()

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions


# ── Guidance hyper-parameters (override via env vars) ─────────────────────
_PROXIMITY_THRESHOLD = float(os.environ.get("GUIDANCE_THRESHOLD", 0.05))
_GUIDANCE_GAIN       = float(os.environ.get("GUIDANCE_GAIN",       0.20))
_TOP_K_DANGER        = 3       # top-N closest (link, obstacle) pairs used
_JACOBIAN_EPS        = 1e-4   # finite-difference step for numerical Jacobian

_GUIDANCE_ENABLED = os.environ.get("EVAL_PROXIMITY_GUIDANCE", "1").lower() not in ("0", "false", "no")

# Guidance mode:
#   "sdf"     — (default) differentiable ground-truth proximity cost: analytic
#               FK + CuRobo collision spheres + per-episode SDF voxel grid;
#               the whole chunk is optimized with Adam (see
#               differentiable_proximity.py).  Falls back to "chunk" if the
#               module fails to build (missing torch, FK validation, ...).
#   "chunk"   — CAPE-style: refine ALL waypoints of each received action chunk
#               with finite-difference Jᵀ repulsion + temporal smoothing.
#   "cbf"     — safety filter: remove the obstacle-approaching component of the
#               policy's own commanded motion (minimally invasive, per step).
#   "repulse" — legacy: add a fixed-magnitude repulsion dq from the obstacle.
_GUIDANCE_MODE = os.environ.get("GUIDANCE_MODE", "sdf").lower()

# CBF parameters: per-step allowed approach speed toward an obstacle is
#   alpha * (dist - margin)   [metres/step]
# At dist == margin all approach is forbidden; far away the constraint is loose.
_CBF_ALPHA  = float(os.environ.get("GUIDANCE_CBF_ALPHA",  0.2))
_CBF_MARGIN = float(os.environ.get("GUIDANCE_CBF_MARGIN", 0.005))

# Chunk-mode parameters (CAPE analogues):
#   gain   — push magnitude per waypoint per refinement pass [rad] (CAPE λ)
#   iters  — refinement passes over the chunk (CAPE's iterative expansion)
#   smooth — box-filter width over per-waypoint corrections (stands in for
#            CAPE's re-denoising, which smooths the perturbed trajectory)
#   stride — evaluate every Nth waypoint; smoothing interpolates the gaps
#            (halves the FK/distance cost at stride 2)
_CHUNK_GAIN   = float(os.environ.get("GUIDANCE_CHUNK_GAIN",   0.05))
_CHUNK_ITERS  = int(os.environ.get("GUIDANCE_CHUNK_ITERS",  2))
_CHUNK_SMOOTH = int(os.environ.get("GUIDANCE_CHUNK_SMOOTH", 5))
_CHUNK_STRIDE = int(os.environ.get("GUIDANCE_CHUNK_STRIDE", 2))


# ── Proximity distance computation (from collect_rollout_proximity_client) ─

_TOP_K_OBSTACLES       = 3
_MESH_REFINE_THRESHOLD = 0.3
_N_AABB_CANDIDATES     = 10

# Only the 16 policy-controlled arm links (fl/fr, joints 1-8).
_ARM_LINK_NAMES = frozenset({
    'fl_link1', 'fl_link2', 'fl_link3', 'fl_link4',
    'fl_link5', 'fl_link6', 'fl_link7', 'fl_link8',
    'fr_link1', 'fr_link2', 'fr_link3', 'fr_link4',
    'fr_link5', 'fr_link6', 'fr_link7', 'fr_link8',
})

# Distal gripper links used as end-effector positions for grasp detection.
_EE_LINK_NAMES = frozenset({'fl_link8', 'fr_link8'})

# ── Per-episode guidance diagnostics ────────────────────────────────────────
# Set to a fresh dict before each rollout, None otherwise.
_episode_log: dict | None = None

def _reset_episode_log():
    global _episode_log
    _episode_log = {
        'trigger_counts': {},   # obstacle_name -> count
        'link_counts':    {},   # link_name -> count
        'min_dists':      {},   # obstacle_name -> min dist seen (m)
        'actors_dumped':  False,
    }

def _log_guidance_trigger(link_name: str, obstacle_name: str, dist_m: float):
    if _episode_log is None:
        return
    _episode_log['trigger_counts'][obstacle_name] = \
        _episode_log['trigger_counts'].get(obstacle_name, 0) + 1
    _episode_log['link_counts'][link_name] = \
        _episode_log['link_counts'].get(link_name, 0) + 1
    prev = _episode_log['min_dists'].get(obstacle_name, 9.9)
    if dist_m < prev:
        _episode_log['min_dists'][obstacle_name] = dist_m

def _log_actor_names(task_env, exclude_names=()):
    """Print scene actor names + active exclusions once per episode."""
    if _episode_log is None or _episode_log.get('actors_dumped'):
        return
    _episode_log['actors_dumped'] = True
    names = sorted(a.get_name() for a in task_env.scene.get_all_actors())
    print(f"\n[diag] scene actors: {names}")
    print(f"[diag] excluded from guidance: {sorted(exclude_names)}\n")

def _print_episode_guidance_summary(task_env, guided: bool, succ: bool):
    if _episode_log is None:
        return
    total = sum(_episode_log['trigger_counts'].values())
    cm    = getattr(task_env, 'collision_metrics', {})
    col   = int(sum(cm.values())) if cm else 0

    tag = "guided" if guided else "noguid"
    result_str = "\033[92mSUCC\033[0m" if succ else "\033[91mFAIL\033[0m"
    print(f"\n\033[36m[ep diag | {tag} | {result_str}]  "
          f"guidance_steps={total}  collisions={col}\033[0m")

    if total > 0:
        # Top obstacles by trigger count
        by_count = sorted(_episode_log['trigger_counts'].items(),
                          key=lambda x: -x[1])
        print("  obstacles triggered (name: steps, min_dist):")
        for name, cnt in by_count[:10]:
            md = _episode_log['min_dists'].get(name, 9.9)
            print(f"    {name:<30s}  steps={cnt:4d}  min_dist={md*100:.1f}cm")
        # Top links
        by_link = sorted(_episode_log['link_counts'].items(), key=lambda x: -x[1])
        print("  links firing guidance:")
        for lname, cnt in by_link:
            print(f"    {lname:<14s}  steps={cnt:4d}")

    if cm:
        print("  collision_metrics:", cm)
    print()


def _compute_exclusion_names(task_env):
    """Actor names that must NOT be treated as guidance obstacles.

    Static: furniture/floor + task target objects declared by the env +
    task containers (box picked from / placed into, destination object) —
    structural proximity the task REQUIRES; repelling from them blocks the
    grasp/place entirely (empty_box: 042_wooden_box was the top trigger and
    guided success dropped to 0%).

    Dynamic: a target object in contact with a gripper link (currently held).
    Non-target objects in contact with the gripper are NOT excluded.
    """
    _exclude_names = frozenset({"ground", "table", "wall"}) | frozenset(
        getattr(task_env, 'target_object_names', set())
    )

    # The env's own structural-furniture set (kitchen: basket_right,
    # cabinet_wall_filler, ...) — the env itself doesn't count collisions
    # with these, so guidance must not repel from them either.
    _exclude_names = _exclude_names | frozenset(
        getattr(task_env, 'furniture_names', ()) or ())

    # Robot furniture: the arm rest stand (e.g. '094_rest') is what the arms
    # park ON — guidance must not fight the home pose.
    try:
        for _a in task_env.scene.get_all_actors():
            _n = _a.get_name()
            if _n.endswith('_rest') or _n == 'rest':
                _exclude_names = _exclude_names | {_n}
    except Exception:
        pass

    for _attr in ('box', 'des_obj'):
        _container = getattr(task_env, _attr, None)
        if _container is None:
            continue
        for _cand in (getattr(_container, 'actor', None), _container):
            if _cand is None:
                continue
            try:
                _exclude_names = _exclude_names | {_cand.get_name()}
                break
            except Exception:
                continue

    target_names = frozenset(getattr(task_env, 'target_object_names', set()))
    try:
        for contact in task_env.scene.get_contacts():
            n0 = contact.bodies[0].entity.name
            n1 = contact.bodies[1].entity.name
            if n0 in _EE_LINK_NAMES and n1 in target_names:
                _exclude_names = _exclude_names | {n1}
            elif n1 in _EE_LINK_NAMES and n0 in target_names:
                _exclude_names = _exclude_names | {n0}
    except Exception:
        pass

    return _exclude_names


def _compute_link_distances(task_env, mesh_refine=True):
    """Two-phase distance from each arm link center to the top-K closest obstacles.

    Phase 1 — vectorised AABB (fast O(M) pre-filter).
    Phase 2 — trimesh refinement for close candidates (uses _proximity_mesh_cache
               if present; falls back to AABB when the cache is absent, which is
               the common case during plain evaluation).
               Pass mesh_refine=False to skip phase 2 entirely — chunk-mode
               guidance evaluates ~100 virtual configurations per chunk and
               trimesh queries make that minutes instead of seconds; AABB
               distance + direction is accurate enough for repulsion gradients.

    Returns:
        {link_name: {'dists': (K,) float32, 'deltas': (K, 3) float32}}
        Sorted ascending by distance.  delta points FROM link center TOWARD
        nearest obstacle surface point.  Missing slots padded with dist=99/delta=0.
    """
    robot = task_env.robot
    all_links = list(robot.left_entity.get_links())
    if robot.right_entity is not robot.left_entity:
        all_links += list(robot.right_entity.get_links())
    links = [lk for lk in all_links if lk.get_name() in _ARM_LINK_NAMES]
    if not links:
        return {}

    robot_entities = {robot.left_entity, robot.right_entity}
    mesh_cache     = getattr(task_env, '_proximity_mesh_cache', {})

    _exclude_names = _compute_exclusion_names(task_env)

    _log_actor_names(task_env, _exclude_names)

    obstacle_actors = []
    for actor in task_env.scene.get_all_actors():
        name = actor.get_name()
        if name in _exclude_names:
            continue
        for comp in actor.get_components():
            if hasattr(comp, 'get_global_aabb_fast'):
                try:
                    aabb = comp.get_global_aabb_fast()
                    obstacle_actors.append((
                        actor, name,
                        np.array(aabb[0], dtype=np.float32),
                        np.array(aabb[1], dtype=np.float32),
                    ))
                except RuntimeError:
                    pass
                break

    aabb_only_mins, aabb_only_maxs = [], []
    for art in task_env.scene.get_all_articulations():
        if art in robot_entities:
            continue
        for art_link in art.get_links():
            if not art_link.collision_shapes:
                continue
            for comp in art_link.entity.get_components():
                if hasattr(comp, 'get_global_aabb_fast'):
                    try:
                        aabb = comp.get_global_aabb_fast()
                        aabb_only_mins.append(np.array(aabb[0], dtype=np.float32))
                        aabb_only_maxs.append(np.array(aabb[1], dtype=np.float32))
                    except RuntimeError:
                        pass
                    break

    K = _TOP_K_OBSTACLES
    _pad = lambda: (np.full(K, 99.0, dtype=np.float32), np.zeros((K, 3), dtype=np.float32))

    if not obstacle_actors and not aabb_only_mins:
        return {lk.get_name(): {'dists': _pad()[0], 'deltas': _pad()[1]}
                for lk in links if lk.collision_shapes}

    n_actor  = len(obstacle_actors)
    all_mins = np.array([o[2] for o in obstacle_actors] + aabb_only_mins, dtype=np.float32)
    all_maxs = np.array([o[3] for o in obstacle_actors] + aabb_only_maxs, dtype=np.float32)
    M = len(all_mins)
    N_CAND = min(_N_AABB_CANDIDATES, M)

    result = {}
    for lk in links:
        if not lk.collision_shapes:
            continue
        try:
            link_center = np.array(lk.get_pose().p, dtype=np.float64)
        except Exception:
            continue

        lc32      = link_center.astype(np.float32)
        clamped   = np.clip(lc32, all_mins, all_maxs)
        delta_aabb = clamped - lc32
        dists_aabb = np.linalg.norm(delta_aabb, axis=1)

        cand_idx     = np.argsort(dists_aabb)[:N_CAND]
        final_dists  = dists_aabb[cand_idx].copy().astype(np.float64)
        final_deltas = delta_aabb[cand_idx].copy().astype(np.float64)

        for ci, oi in enumerate(cand_idx):
            if not mesh_refine or dists_aabb[oi] >= _MESH_REFINE_THRESHOLD:
                break
            if oi >= n_actor:
                continue
            actor, name, _, _ = obstacle_actors[oi]
            if name not in mesh_cache:
                continue
            try:
                mesh     = mesh_cache[name]
                T        = actor.get_pose().to_transformation_matrix()
                T_inv    = np.linalg.inv(T)
                local_pt = T_inv[:3, :3] @ link_center + T_inv[:3, 3]
                closest_pts, mesh_dists, _ = trimesh.proximity.closest_point(mesh, [local_pt])
                closest_world = T[:3, :3] @ closest_pts[0] + T[:3, 3]
                final_dists[ci]  = float(mesh_dists[0])
                final_deltas[ci] = closest_world - link_center
            except Exception:
                pass

        # Track obstacle names parallel to cand_idx so we can log which object triggered.
        cand_names = []
        for oi in cand_idx:
            cand_names.append(obstacle_actors[oi][1] if oi < n_actor else '<articulation>')

        order      = np.argsort(final_dists)[:K]
        top_dists  = final_dists[order].astype(np.float32)
        top_deltas = final_deltas[order].astype(np.float32)
        top_names  = [cand_names[i] for i in order[:K]]

        if len(top_dists) < K:
            pad = K - len(top_dists)
            top_dists  = np.concatenate([top_dists,  np.full(pad, 99.0, dtype=np.float32)])
            top_deltas = np.concatenate([top_deltas, np.zeros((pad, 3), dtype=np.float32)])
            top_names += ['<pad>'] * pad

        result[lk.get_name()] = {'dists': top_dists, 'deltas': top_deltas,
                                  'obstacles': top_names}

    return result


# ── Jacobian helpers ────────────────────────────────────────────────────────

def _get_arm_dof_indices(entity, robot, arm='left'):
    """DOF indices (within entity.get_active_joints()) that belong to the arm.

    Uses robot.left/right_arm_joints (SAPIEN joint objects with get_name()) when
    available — this is robust regardless of joint ordering in the articulation.
    Falls back to a positional heuristic when the name lookup fails.
    """
    target_joints = getattr(robot, f'{arm}_arm_joints', None)
    if target_joints is not None:
        try:
            arm_names  = {j.get_name() for j in target_joints}
            active     = entity.get_active_joints()
            indices    = [i for i, j in enumerate(active) if j.get_name() in arm_names]
            if indices:
                return indices
        except Exception:
            pass

    # Fallback heuristic: arm joints occupy the first 6 DOFs of their entity.
    # For a shared (dual_arm_embodied) entity the right arm starts at offset n//2.
    n_dofs = len(entity.get_qpos())
    if arm == 'left' or robot.right_entity is not robot.left_entity:
        return list(range(min(6, n_dofs)))
    else:
        offset = n_dofs // 2
        return list(range(offset, min(offset + 6, n_dofs)))


def _compute_link_jacobian(entity, link, dof_indices):
    """Numerical position Jacobian J (3, len(dof_indices)) via finite differences.

    Perturbs each joint in dof_indices by _JACOBIAN_EPS, reads the link pose
    (pure FK — no scene.step() needed), then restores qpos.  The drive targets
    are NOT perturbed so no physics artefact bleeds into the next step.

    Returns None on error.
    """
    try:
        q0   = entity.get_qpos().copy()
        pos0 = np.array(link.get_pose().p, dtype=np.float64)
    except Exception:
        return None

    J = np.zeros((3, len(dof_indices)), dtype=np.float64)
    try:
        for ci, ji in enumerate(dof_indices):
            q_pert     = q0.copy()
            q_pert[ji] += _JACOBIAN_EPS
            entity.set_qpos(q_pert)
            pos_pert   = np.array(link.get_pose().p, dtype=np.float64)
            J[:, ci]   = (pos_pert - pos0) / _JACOBIAN_EPS
    finally:
        entity.set_qpos(q0)   # always restore, even on partial failure

    return J


# ── Main guidance function ──────────────────────────────────────────────────

def compute_proximity_correction(action, task_env,
                                  threshold=_PROXIMITY_THRESHOLD,
                                  gain=_GUIDANCE_GAIN):
    """Return a corrected policy action with proximity-repulsion blended in.

    action: (14,) float32 — policy output:
        [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]

    Correction is applied only to the 6 revolute arm joints of each arm
    (indices 0:6 for left, 7:13 for right).  Gripper joints are untouched.

    Returns the corrected action (same dtype/shape as input).
    """
    link_dists = _compute_link_distances(task_env)
    if not link_dists:
        return action

    # Per-link aggregation: weight each obstacle's delta by its scalar distance, then sum.
    # This collapses the K per-obstacle vectors into one representative direction per link.
    # The link's minimum distance (dists[0], already sorted ascending) gates the threshold.
    per_link = []
    for link_name, data in link_dists.items():
        dists     = data['dists']      # (K,) ascending
        deltas    = data['deltas']     # (K, 3)
        obstacles = data.get('obstacles', ['?'] * len(dists))
        min_dist  = float(dists[0])
        if min_dist >= threshold:
            continue                              # link is safe — skip
        agg_delta = np.zeros(3, dtype=np.float64)
        for k in range(len(dists)):
            if dists[k] < 99.0:                  # skip padding sentinel entries
                agg_delta += float(dists[k]) * deltas[k].astype(np.float64)
        closest_obstacle = obstacles[0] if obstacles else '?'
        per_link.append((link_name, min_dist, agg_delta, closest_obstacle))

    if not per_link:
        return action

    # Global top-3 links by minimum distance (closest link first)
    per_link.sort(key=lambda e: e[1])
    top_k = per_link[:_TOP_K_DANGER]

    # Inverse-distance weights across top-3 links (closer link → stronger repulsion)
    raw_w   = np.array([1.0 / max(e[1], 1e-6) for e in top_k])
    weights = raw_w / raw_w.sum()

    # Split by arm prefix
    left_entries  = [(e, w) for e, w in zip(top_k, weights) if e[0].startswith('fl_')]
    right_entries = [(e, w) for e, w in zip(top_k, weights) if e[0].startswith('fr_')]

    robot     = task_env.robot
    corrected = action.copy()

    for arm_entries, arm_name, entity, act_slice in [
        (left_entries,  'left',  robot.left_entity,  slice(0, 6)),
        (right_entries, 'right', robot.right_entity, slice(7, 13)),
    ]:
        if not arm_entries:
            continue

        # Weighted-average of per-link aggregated deltas (unit-normalised per link)
        d_avg = np.zeros(3, dtype=np.float64)
        for entry, w in arm_entries:
            _, min_dist, agg_delta = entry[0], entry[1], entry[2]
            n = np.linalg.norm(agg_delta)
            if n > 1e-8:
                d_avg += w * (agg_delta / n)     # unit direction per link, then weight

        d_avg_norm = np.linalg.norm(d_avg)
        if d_avg_norm < 1e-8:
            continue

        # Repulsion = opposite of weighted-average delta (delta points link→obstacle)
        d_repulsion = -d_avg / d_avg_norm   # unit vector, away from obstacle

        # Closest link/distance in this arm group (tuple: link_name, min_dist, agg_delta, obstacle)
        closest_entry   = min(arm_entries, key=lambda x: x[0][1])
        closest_name    = closest_entry[0][0]   # link_name
        dist_closest    = closest_entry[0][1]   # min_dist
        closest_obstacle = closest_entry[0][3]  # obstacle actor name

        # Resolve SAPIEN link object
        link_map    = {lk.get_name(): lk for lk in entity.get_links()}
        target_link = link_map.get(closest_name)
        if target_link is None:
            continue

        # DOF indices for this arm's revolute joints
        dof_indices = _get_arm_dof_indices(entity, robot, arm=arm_name)
        if not dof_indices:
            continue

        # Numerical Jacobian for the closest link
        J = _compute_link_jacobian(entity, target_link, dof_indices)
        if J is None:
            continue

        # Damped least-squares pseudo-inverse — prevents blow-up near singularities.
        # Plain pinv amplifies noise when J has small singular values; DLS caps it.
        _DLS_DAMPING = 0.05
        J_pinv = J.T @ np.linalg.inv(J @ J.T + _DLS_DAMPING**2 * np.eye(3))
        dq     = J_pinv @ d_repulsion       # (n_arm_dof,) — unnormalised

        # Scale: linear ramp from 0 at threshold to gain at dist=0
        scale    = gain * max(0.0, 1.0 - dist_closest / threshold)
        dq_norm  = np.linalg.norm(dq)
        if dq_norm > 1e-8:
            dq = (dq / dq_norm) * scale

        corrected[act_slice] = (corrected[act_slice] + dq.astype(np.float32))

        _log_guidance_trigger(closest_name, closest_obstacle, dist_closest)

        step = getattr(task_env, 'take_action_cnt', -1)
        active_joints = entity.get_active_joints()
        joint_names = [active_joints[i].get_name() for i in dof_indices]
        dq_parts = "  ".join(f"{joint_names[ci]}={dq[ci]:+.4f}" for ci in range(len(dq)))
        print(f"[guidance t={step:4d}] {arm_name:5s}  dist={dist_closest*100:.1f}cm  "
              f"link={closest_name}  obstacle={closest_obstacle}  scale={scale:.4f}\n"
              f"           joints: {dq_parts}")

    return corrected


# ── CBF-style safety filter (GUIDANCE_MODE=cbf, default) ───────────────────

def compute_cbf_correction(action, task_env,
                            threshold=_PROXIMITY_THRESHOLD,
                            alpha=_CBF_ALPHA,
                            margin=_CBF_MARGIN):
    """Minimally-invasive safety filter on the policy's commanded joint step.

    Unlike the legacy repulsion mode, this NEVER adds motion of its own.  For
    each near-obstacle link it enforces a velocity barrier on the policy's own
    commanded step  dq = action - q_measured:

        approach_speed = n̂ᵀ J dq  ≤  alpha · (dist − margin)

    where n̂ is the unit vector link→obstacle.  When the policy's step violates
    the constraint, the violating component is projected out analytically
    (single-constraint QP solution):

        dq ← dq − ((n̂ᵀJdq − b) / ‖Jᵀn̂‖²) · Jᵀn̂      with b = alpha·(dist−margin)

    Constraints are applied sequentially, closest link first (top-K).  Moving
    away from an obstacle is never penalised, so the filter cannot fight the
    policy or inject oscillations — the failure mode observed with repulsion.

    action: (14,) float32 — absolute joint position targets:
        [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
    Returns the corrected action (gripper joints untouched).
    """
    link_dists = _compute_link_distances(task_env)
    if not link_dists:
        return action

    # Collect dangerous (link, dist, n̂, obstacle) entries below threshold.
    danger = []
    for link_name, data in link_dists.items():
        dist = float(data['dists'][0])
        if dist >= threshold:
            continue
        delta = data['deltas'][0].astype(np.float64)
        n = np.linalg.norm(delta)
        if n < 1e-8:
            continue
        obstacles = data.get('obstacles', ['?'])
        danger.append((link_name, dist, delta / n, obstacles[0]))

    if not danger:
        return action

    danger.sort(key=lambda e: e[1])
    danger = danger[:_TOP_K_DANGER]

    robot     = task_env.robot
    corrected = np.asarray(action, dtype=np.float32).copy()

    for arm_name, entity, act_slice, prefix in [
        ('left',  robot.left_entity,  slice(0, 6),  'fl_'),
        ('right', robot.right_entity, slice(7, 13), 'fr_'),
    ]:
        arm_danger = [d for d in danger if d[0].startswith(prefix)]
        if not arm_danger:
            continue

        dof_indices = _get_arm_dof_indices(entity, robot, arm=arm_name)
        if not dof_indices:
            continue

        q_current = np.asarray(entity.get_qpos(), dtype=np.float64)[dof_indices]
        dq        = corrected[act_slice].astype(np.float64) - q_current

        link_map = {lk.get_name(): lk for lk in entity.get_links()}
        step     = getattr(task_env, 'take_action_cnt', -1)

        for link_name, dist, n_hat, obstacle_name in arm_danger:
            target_link = link_map.get(link_name)
            if target_link is None:
                continue
            J = _compute_link_jacobian(entity, target_link, dof_indices)
            if J is None:
                continue

            a = J.T @ n_hat                       # ∂(approach speed)/∂dq
            approach = float(n_hat @ (J @ dq))    # m/step toward obstacle
            allowed  = alpha * max(0.0, dist - margin)
            if approach <= allowed:
                continue                          # constraint satisfied — no-op

            a_sq = float(a @ a)
            if a_sq < 1e-10:
                continue
            lam = (approach - allowed) / a_sq
            dq  = dq - lam * a

            _log_guidance_trigger(link_name, obstacle_name, dist)
            print(f"[cbf t={step:4d}] {arm_name:5s}  dist={dist*100:.1f}cm  "
                  f"link={link_name}  obstacle={obstacle_name}  "
                  f"approach={approach*100:.2f}cm/step → allowed={allowed*100:.2f}cm/step")

        corrected[act_slice] = (q_current + dq).astype(np.float32)

    return corrected


# ── CAPE-style chunk guidance (GUIDANCE_MODE=chunk, default) ────────────────

def compute_chunk_guidance(chunk, task_env,
                            threshold=_PROXIMITY_THRESHOLD,
                            gain=_CHUNK_GAIN,
                            iters=_CHUNK_ITERS,
                            smooth=_CHUNK_SMOOTH,
                            stride=_CHUNK_STRIDE):
    """Refine a whole action chunk with repulsion gradients (client-side CAPE).

    Following CAPE (Yang et al. 2025), guidance is applied to the *planned
    trajectory* rather than only the currently executed action: every waypoint
    of the chunk is FK-evaluated against the scene, and waypoints whose links
    come within `threshold` of an obstacle are pushed along the joint-space
    steepest-descent direction of the hinge distance cost:

        dq_k += gain · (1 − dist/threshold) · Jᵀ n̂_away

    Jᵀ is the gradient chain rule (not a pseudo-inverse) — no singularity
    blow-up.  CAPE relies on re-denoising to keep the perturbed trajectory
    smooth and feasible; we approximate that with a temporal box filter over
    the per-waypoint corrections, repeated for `iters` refinement passes.

    chunk: (N, 14) float32 absolute joint position targets.  Gripper columns
    (6, 13) are never modified.  Returns the corrected chunk.
    """
    chunk = np.asarray(chunk, dtype=np.float32).copy()
    if chunk.ndim != 2 or chunk.shape[1] < 14:
        return chunk

    robot = task_env.robot
    left_e, right_e = robot.left_entity, robot.right_entity
    left_dofs  = _get_arm_dof_indices(left_e,  robot, 'left')
    right_dofs = _get_arm_dof_indices(right_e, robot, 'right')
    if not left_dofs or not right_dofs:
        return chunk

    # Save the real sim state; all FK below is virtual and restored afterwards.
    saved = [(left_e, np.asarray(left_e.get_qpos()).copy())]
    if right_e is not left_e:
        saved.append((right_e, np.asarray(right_e.get_qpos()).copy()))

    step      = getattr(task_env, 'take_action_cnt', -1)
    n_wp      = len(chunk)
    touched   = set()
    max_dq    = 0.0
    link_hits = {}

    try:
        for it in range(iters):
            corrections   = np.zeros_like(chunk)
            any_this_iter = False

            for k in range(0, n_wp, max(1, stride)):
                wp = chunk[k]
                # Virtual FK to the waypoint configuration (arm joints only;
                # gripper/other dofs keep their real current values).
                qL = saved[0][1].copy()
                qL[left_dofs] = wp[0:6]
                if right_e is left_e:
                    qL[right_dofs] = wp[7:13]
                    left_e.set_qpos(qL)
                else:
                    left_e.set_qpos(qL)
                    qR = saved[1][1].copy()
                    qR[right_dofs] = wp[7:13]
                    right_e.set_qpos(qR)

                # AABB-only: trimesh refinement over ~100 virtual configs per
                # chunk costs minutes; AABB direction is enough for repulsion.
                link_dists = _compute_link_distances(task_env, mesh_refine=False)
                if not link_dists:
                    continue

                danger = []
                for link_name, data in link_dists.items():
                    dist = float(data['dists'][0])
                    if dist >= threshold:
                        continue
                    delta = data['deltas'][0].astype(np.float64)
                    nrm   = np.linalg.norm(delta)
                    if nrm < 1e-8:
                        continue
                    obstacles = data.get('obstacles', ['?'])
                    danger.append((link_name, dist, delta / nrm, obstacles[0]))
                if not danger:
                    continue
                danger.sort(key=lambda e: e[1])
                danger = danger[:_TOP_K_DANGER]

                for entity, act_slice, dofs, prefix in [
                    (left_e,  slice(0, 6),  left_dofs,  'fl_'),
                    (right_e, slice(7, 13), right_dofs, 'fr_'),
                ]:
                    arm_danger = [d for d in danger if d[0].startswith(prefix)]
                    if not arm_danger:
                        continue
                    link_map = {lk.get_name(): lk for lk in entity.get_links()}
                    dq = np.zeros(len(dofs), dtype=np.float64)
                    for link_name, dist, n_hat, obstacle_name in arm_danger:
                        target_link = link_map.get(link_name)
                        if target_link is None:
                            continue
                        J = _compute_link_jacobian(entity, target_link, dofs)
                        if J is None:
                            continue
                        a   = J.T @ (-n_hat)        # joint-space push away
                        a_n = np.linalg.norm(a)
                        if a_n < 1e-10:
                            continue
                        w   = 1.0 - dist / threshold
                        dq += gain * w * (a / a_n)
                        if it == 0:
                            _log_guidance_trigger(link_name, obstacle_name, dist)
                            link_hits[link_name] = link_hits.get(link_name, 0) + 1

                    dq_n = np.linalg.norm(dq)
                    if dq_n < 1e-10:
                        continue
                    if dq_n > gain:                 # cap combined multi-link push
                        dq *= gain / dq_n
                    corrections[k, act_slice] = dq.astype(np.float32)
                    touched.add(k)
                    any_this_iter = True

            if not any_this_iter:
                break

            # Temporal smoothing — stands in for CAPE's re-denoising pass.
            if smooth > 1 and n_wp > 1:
                kernel = np.ones(smooth, dtype=np.float32) / smooth
                for j in list(range(0, 6)) + list(range(7, 13)):
                    corrections[:, j] = np.convolve(corrections[:, j], kernel, mode='same')

            chunk += corrections
            max_dq = max(max_dq, float(np.abs(corrections).max()))
    finally:
        for ent, q in saved:
            ent.set_qpos(q)

    if touched:
        hits = "  ".join(f"{n}×{c}" for n, c in
                         sorted(link_hits.items(), key=lambda x: -x[1])[:4])
        print(f"[chunk t={step:4d}] guided waypoints: {len(touched)}/{n_wp}  "
              f"max|dq|={max_dq:.4f} rad  links: {hits}")

    return chunk


# ── Differentiable SDF guidance (GUIDANCE_MODE=sdf, default) ────────────────
# Per-episode ProximityCost instance — built lazily on the first chunk of each
# rollout (the scene must be fully set up), torn down via _reset_sdf_state().

_SDF_STATE = {'cost': None, 'failed': False}


def _reset_sdf_state():
    _SDF_STATE['cost']     = None
    _SDF_STATE['failed']   = False
    _SDF_STATE['events']   = 0
    _SDF_STATE['predlog']  = None   # per-episode JSONL of predictions/actuals
    _SDF_STATE['obs_hash'] = None   # cumulative obs hash (determinism probe)
    _SDF_STATE['mode']     = ''     # 'guided' | 'noguid' (predlog filename tag)
    # Per-rollout CONTACT counters — the primary safety metric: the goal is
    # to avoid touching things at all, not merely avoid displacing them.
    _SDF_STATE['contact_steps'] = 0      # steps with any robot↔clutter contact
    _SDF_STATE['force_steps']   = 0      # steps with impulse > 1e-3
    _SDF_STATE['objects']       = set()  # distinct clutter objects touched


def _sdf_event_dir():
    """Debug output dir from SDF_DEBUG_VIZ, or None when disabled."""
    dbg = os.environ.get("SDF_DEBUG_VIZ", "")
    if not dbg or dbg.lower() in ("0", "false", "no"):
        return None
    return dbg if dbg.lower() not in ("1", "true", "yes") else "sdf_debug"


def _sdf_predlog(entry):
    """Append one JSON line to the per-episode prediction log."""
    path = _SDF_STATE.get('predlog')
    if path is None:
        os.makedirs("sdf_debug", exist_ok=True)
        tag  = _SDF_STATE.get('mode') or 'run'
        path = (f"sdf_debug/predlog_{datetime.now().strftime('%H%M%S')}"
                f"_{tag}.jsonl")
        _SDF_STATE['predlog'] = path
        print(f"[sdf] prediction log: {path}")
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _sdf_record_actual(task_env):
    """Per-step ground truth for prediction validation (unguided rollouts).

    Records the robot's REAL minimum clearance at every executed step; when a
    previously predicted collision's execution step arrives, also exports a
    solid snapshot of the actual pose (pairs with the sdf_event_pred_* files
    to answer: did the FK-lookahead prediction come true?).
    """
    cost = _SDF_STATE.get('cost')
    if cost is None:
        return
    try:
        t = getattr(task_env, 'take_action_cnt', -1)
        clear = cost._current_clearance()
        _sdf_predlog({"type": "actual", "t": int(t),
                      "clear": round(float(clear), 4)})

        # Independent physics ground truth: REAL contacts between the robot
        # (or its held target) and non-excluded clutter, straight from the
        # engine — catches gentle grazes the displacement-based collision
        # metrics never see.  This is what "collision" means; the clearance
        # above is only the model's opinion.
        try:
            excl = getattr(cost, '_exclude_names', frozenset())
            obstacle_names = {a.get_name()
                              for a in task_env.scene.get_all_actors()} - set(excl)
            targets  = frozenset(getattr(task_env, 'target_object_names', set()))
            robotish = _ARM_LINK_NAMES | targets
            pairs = []
            for c in task_env.scene.get_contacts():
                e0, e1 = c.bodies[0].entity, c.bodies[1].entity
                for ea, eb in ((e0, e1), (e1, e0)):
                    a, b = ea.name, eb.name
                    if a in robotish and b in obstacle_names:
                        imp = 0.0
                        try:
                            imp = float(sum(np.linalg.norm(p.impulse)
                                            for p in c.points))
                        except Exception:
                            pass
                        # obstacle identity as name#per_scene_id — names are
                        # duplicated in cluttered scenes
                        bid = getattr(eb, 'per_scene_id', '')
                        pairs.append([a, f"{b}#{bid}", round(imp, 5)])
                        break
            if pairs:
                _sdf_predlog({"type": "contact", "t": int(t), "pairs": pairs})
                _SDF_STATE['contact_steps'] = \
                    _SDF_STATE.get('contact_steps', 0) + 1
                if any(p[2] > 1e-3 for p in pairs):
                    _SDF_STATE['force_steps'] = \
                        _SDF_STATE.get('force_steps', 0) + 1
                _SDF_STATE.setdefault('objects', set()).update(
                    p[1] for p in pairs)
        except Exception:
            pass

    except Exception:
        pass


def _sdf_get_cost(task_env):
    """Lazily build (and cache) the per-episode ProximityCost."""
    if _SDF_STATE['cost'] is None:
        from differentiable_proximity import ProximityCost
        robot = task_env.robot
        ldofs = _get_arm_dof_indices(robot.left_entity,  robot, 'left')
        rdofs = _get_arm_dof_indices(robot.right_entity, robot, 'right')
        excl  = _compute_exclusion_names(task_env)
        _log_actor_names(task_env, excl)
        cost = ProximityCost(task_env, ldofs, rdofs, excl)
        print(f"[sdf] ready: fk_err=({cost.fk_err_left*1000:.2f}, "
              f"{cost.fk_err_right*1000:.2f}) mm  sdf={cost.sdf_info}")
        _SDF_STATE['cost'] = cost
    return _SDF_STATE['cost']


def _sdf_guided_chunk(chunk, task_env, threshold, optimize=True):
    """FK-lookahead on the chunk against the differentiable proximity cost.

    optimize=True  — guidance: optimize the chunk and return the corrected one.
    optimize=False — passive prediction (unguided rollouts): never modify the
                     chunk; log the predicted clearance and export event
                     snapshots so predictions can be validated against the
                     actual rollout.

    Falls back to compute_chunk_guidance (finite-difference CAPE mode) if the
    differentiable module cannot be built (missing torch, FK mismatch, ...).
    """
    if _SDF_STATE['failed']:
        if not optimize:
            return chunk
        return compute_chunk_guidance(chunk, task_env, threshold=threshold)

    try:
        cost = _sdf_get_cost(task_env)
        cost.maybe_rebuild(task_env)

        step = getattr(task_env, 'take_action_cnt', -1)
        # Determinism probe: with paired noise (seed_rng) the per-chunk action
        # hashes of guided/unguided rollouts must match pairwise until guidance
        # first modifies something.  The first mismatching chunk marks the
        # fork; the obs hash at that boundary attributes it — obs differs ⇒
        # env/render nondeterminism; obs same but actions differ ⇒ server.
        _h = hashlib.sha1(np.ascontiguousarray(
            np.asarray(chunk, dtype=np.float32)).tobytes()).hexdigest()[:10]
        _oh = _SDF_STATE.get('obs_hash')
        _ot = f"  obs={_oh['h'].hexdigest()[:10]}" if _oh else ""
        print(f"[sdf] chunk@t{step:<4d} act={_h}{_ot}")
        event_dir = _sdf_event_dir() \
            if _SDF_STATE.get('events', 0) < 8 else None

        if not optimize:
            # ── Passive prediction ────────────────────────────────────────
            ev = cost.evaluate_chunk(np.asarray(chunk, dtype=np.float32))
            if ev['clear0'] < -0.10 and cost.sdf_info.get('signed', True):
                cost.force_unsigned(task_env)
                ev = cost.evaluate_chunk(np.asarray(chunk, dtype=np.float32))
            w = ev['worst']
            _sdf_predlog({"type": "pred", "t": int(step),
                          "prox": round(ev['prox0'], 4),
                          "clear": round(ev['clear0'], 4), "worst": w})
            if ev['clear0'] < 0.0:
                print(f"[sdf-pred t={step:4d}] PREDICTED collision: "
                      f"clear={ev['clear0']*100:.1f}cm  "
                      f"worst={w['link']}@k{w['waypoint']} {w['pos']}")
                if event_dir:
                    cost._export_event(event_dir, f"pred_t{step}",
                                       ev['centers0'], ev['clr_t'],
                                       worst_k=w['waypoint'])
                    _SDF_STATE['events'] = _SDF_STATE.get('events', 0) + 1
            return chunk

        # ── Guidance ──────────────────────────────────────────────────────
        out, info = cost.optimize_chunk(np.asarray(chunk, dtype=np.float32),
                                        event_dir=event_dir,
                                        event_tag=f"t{step}")
        if info.get('event_written'):
            _SDF_STATE['events'] = _SDF_STATE.get('events', 0) + 1

        # A commanded configuration >10 cm "inside" an obstacle is physically
        # implausible — flipped mesh normals corrupting the sign of the field
        # in a region the build-time sanity check didn't cover.  Switch this
        # episode to unsigned distances and redo the chunk.
        if info['clear0'] < -0.10 and cost.sdf_info.get('signed', True):
            w = info.get('worst', {})
            print(f"\033[33m[sdf] implausible {info['clear0']*100:.0f} cm "
                  f"penetration at {w.get('link')}@k{w.get('waypoint')} "
                  f"{w.get('pos')} — rebuilding grid unsigned\033[0m")
            cost.force_unsigned(task_env)
            out, info = cost.optimize_chunk(np.asarray(chunk, dtype=np.float32))

        # Symmetric prediction log for the guided arm (clear_after shows what
        # the optimizer bought back).
        _sdf_predlog({"type": "pred", "t": int(step),
                      "prox": round(info['prox0'], 4),
                      "clear": round(info['clear0'], 4),
                      "clear_after": round(info['clear1'], 4),
                      "worst": info.get('worst')})

        if not info['skipped']:
            _log_guidance_trigger('sdf', 'sdf_cost', max(info['clear0'], 0.0))
            w = info.get('worst', {})
            print(f"[sdf t={step:4d}] prox {info['prox0']:.4f}→{info['prox1']:.4f}  "
                  f"clearance {info['clear0']*100:.1f}→{info['clear1']*100:.1f}cm  "
                  f"max|dq|={info['max_dq']:.4f} rad  ({info['ms']:.0f} ms)  "
                  f"worst={w.get('link','?')}@k{w.get('waypoint','?')} {w.get('pos','')}")
        return out

    except Exception as e:
        print(f"\033[31m[sdf] disabled for this run ({type(e).__name__}: {e}) — "
              f"falling back to finite-difference chunk guidance\033[0m")
        traceback.print_exc()
        _SDF_STATE['failed'] = True
        if not optimize:
            return chunk
        return compute_chunk_guidance(chunk, task_env, threshold=threshold)


# ── Serialisation helpers ───────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            dtype = str(obj.dtype) if obj.dtype not in (
                np.float32, np.float64, np.int32, np.int64) else str(obj.dtype)
            return {'__numpy_array__': True,
                    'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                    'dtype': dtype, 'shape': obj.shape}
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    def object_hook(dct):
        if '__numpy_array__' in dct:
            return np.frombuffer(
                base64.b64decode(dct['data']), dtype=dct['dtype']
            ).reshape(dct['shape'])
        return dct
    return json.loads(json_str, object_hook=object_hook)


# ── ModelClient ─────────────────────────────────────────────────────────────

class ModelClient:
    _MIRRORED_DEFAULTS = {"observation_window": None}

    def __init__(self, host='localhost', port=9999, timeout=30, **mirrored):
        self.host = host; self.port = port; self.timeout = timeout; self.sock = None
        for k, v in self._MIRRORED_DEFAULTS.items():
            object.__setattr__(self, k, v)
        for k, v in mirrored.items():
            object.__setattr__(self, k, v)
        self._connect()

    def _connect(self):
        for attempt in range(1000):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                if self.sock: self.sock.close()
                print(f"⚠️  Connection attempt {attempt+1} failed: {e}. Retrying in 5s...")
                time.sleep(5)
        raise ConnectionError("Failed to connect to model server after 1000 attempts")

    def _send_recv(self, data):
        json_data = numpy_to_json(data).encode('utf-8')
        self.sock.sendall(len(json_data).to_bytes(4, 'big'))
        self.sock.sendall(json_data)
        return self._recv_response()

    def _recv_response(self):
        size = int.from_bytes(self.sock.recv(4), 'big')
        chunks, received = [], 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk: raise ConnectionError("Incomplete response")
            chunks.append(chunk); received += len(chunk)
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        if 'res' not in response:
            err = response.get('error', '<no error field>')
            tb  = response.get('traceback', '')
            raise RuntimeError(f"Server error during RPC '{func_name}':\n{err}\n{tb}")
        return response['res']

    def __getattr__(self, name):
        if name.startswith("_") or name in {"host", "port", "timeout", "sock"}:
            raise AttributeError(name)
        def _proxy(*args, **kwargs):
            obs = None if not args else (args[0] if len(args) == 1 else {"_args": list(args)})
            result = self.call(func_name=name, obs=obs)
            if name in ("reset_obsrvationwindows", "reset_model"):
                object.__setattr__(self, "observation_window", None)
            elif name == "update_observation_window":
                object.__setattr__(self, "observation_window", True)
            return result
        return _proxy

    def close(self):
        if self.sock:
            try: self.sock.close()
            except: pass
            finally: self.sock = None; print("🔌 Connection closed")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


# ── Env / policy helpers ────────────────────────────────────────────────────

def class_decorator(task_name):
    envs_module = None
    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        for mod_path in [f"bench_envs.{task_name}", f"bench_envs.study.{task_name}",
                         f"bench_envs.office.{task_name}", f"bench_envs.kitchenl.{task_name}",
                         f"bench_envs.kitchens.{task_name}"]:
            try:
                envs_module = importlib.import_module(mod_path); break
            except ModuleNotFoundError:
                continue
    if envs_module is None:
        envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(envs_module, task_name)()
    except AttributeError:
        raise SystemExit("No such task")


def eval_function_decorator(policy_name, model_name):
    return getattr(importlib.import_module(policy_name), model_name)


def get_camera_config(camera_type):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "../task_config/_camera_config.yml")
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg[camera_type]


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


# ── Evaluation loop ─────────────────────────────────────────────────────────

def eval_policy(task_name, TASK_ENV, args, model, st_seed,
                test_num=100, video_size=None, instruction_type="seen",
                seed_list=None,
                threshold=_PROXIMITY_THRESHOLD, gain=_GUIDANCE_GAIN,
                guidance_enabled=_GUIDANCE_ENABLED,
                compare=False):
    """Evaluate the policy.

    compare=True: for each valid seed run guided then unguided back-to-back on
    the SAME seed so results are directly comparable.  Prints a per-episode
    table and running totals for both conditions.
    """
    print(f"\033[34mTask: {task_name}  |  Policy: {args['policy_name']}\033[0m")
    if _GUIDANCE_MODE == "sdf":
        # activation = SDF_MARGIN from the arm SURFACE; legacy threshold unused.
        # Import the resolved knob values — defaults live only in that module.
        try:
            import differentiable_proximity as _dp
            _params = (f"activation={_dp._MARGIN*100:.1f}cm-from-surface  "
                       f"steps={_dp._STEPS}  lr={_dp._LR}  max_dq={_dp._MAX_DQ}")
        except Exception as _e:   # module unusable → runtime falls back to chunk
            _params = f"(differentiable_proximity unavailable: {_e})"
        _mode_desc = f"mode={_GUIDANCE_MODE}  {_params}"
    else:
        if _GUIDANCE_MODE == "chunk":
            _params = (f"gain={_CHUNK_GAIN}  iters={_CHUNK_ITERS}  "
                       f"smooth={_CHUNK_SMOOTH}  stride={_CHUNK_STRIDE}")
        elif _GUIDANCE_MODE == "cbf":
            _params = f"alpha={_CBF_ALPHA}  margin={_CBF_MARGIN*100:.1f}cm"
        else:
            _params = f"gain={gain:.4f}"
        _mode_desc = f"mode={_GUIDANCE_MODE}  threshold={threshold*100:.1f}cm  {_params}"
    if compare:
        print(f"\033[33mCOMPARE mode: guided vs. unguided  {_mode_desc}\033[0m")
    elif guidance_enabled:
        print(f"\033[33mProximity guidance ON  {_mode_desc}\033[0m")
    else:
        print("\033[33mProximity guidance OFF\033[0m")

    eval_func        = eval_function_decorator(args["policy_name"], "eval")
    clear_cache_freq = args["clear_cache_freq"]
    args["eval_mode"] = True

    pi0_step = getattr(model, 'pi0_step', 50)

    # ── Inner helper: run one rollout, return (success, collision_count) ──────
    def _run_rollout(now_id, now_seed, episode_info, guided, video_suffix=""):
        _reset_episode_log()
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)

        ep_info_list = [episode_info["info"]]
        results      = generate_episode_descriptions(task_name, ep_info_list, test_num)
        instruction  = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        if TASK_ENV.eval_video_path is not None and video_size is not None:
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pixel_format", "rgb24",
                 "-video_size", video_size, "-framerate", "15", "-i", "-",
                 "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                 f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}{video_suffix}.mp4"],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        model.reset_model()
        try:
            # Identical sampling noise for both rollouts of a paired seed —
            # otherwise policy stochasticity masquerades as a guidance effect.
            model.seed_rng(now_seed)
        except Exception as e:
            print(f"\033[33m[warn] seed_rng failed ({type(e).__name__}: {e}) "
                  f"— rollouts use UNPAIRED sampling noise; per-seed "
                  f"comparisons are confounded by policy variance\033[0m")

        sdf_passive = (not guided) and _GUIDANCE_MODE == "sdf"
        _orig_ta = TASK_ENV.take_action

        if sdf_passive:
            # Passive FK-lookahead on the UNGUIDED rollout: predict collisions
            # from each chunk (never modify it) and record the actual per-step
            # clearance, so the prediction's validity can be checked offline.
            _reset_sdf_state()
            _SDF_STATE['mode'] = 'noguid'
            _orig_ga = model.get_action
            def _monitor_get_action(*a, _orig=_orig_ga, **kw):
                chunk = _orig(*a, **kw)
                return _sdf_guided_chunk(chunk, TASK_ENV, threshold,
                                         optimize=False)
            model.get_action = _monitor_get_action

            def _passive_take_action(action, _orig=_orig_ta):
                _sdf_record_actual(TASK_ENV)
                return _orig(action)
            TASK_ENV.take_action = _passive_take_action

        if guided:
            if _GUIDANCE_MODE == "sdf":
                # Differentiable proximity cost: optimize the whole chunk on
                # arrival (Adam on FK+SDF objective); per-step execution only
                # RECORDS contacts/clearance (same evidence as unguided —
                # contact-for-contact comparison).
                _reset_sdf_state()
                _SDF_STATE['mode'] = 'guided'
                _orig_ga = model.get_action
                def _guided_get_action(*a, _orig=_orig_ga, **kw):
                    chunk = _orig(*a, **kw)
                    return _sdf_guided_chunk(chunk, TASK_ENV, threshold)
                model.get_action = _guided_get_action

                def _guided_take_action(action, _orig=_orig_ta):
                    _sdf_record_actual(TASK_ENV)
                    return _orig(action)
            elif _GUIDANCE_MODE == "chunk":
                # CAPE-style: refine the whole chunk when it arrives from the
                # server; per-step execution stays untouched.  The instance
                # attribute shadows ModelClient.__getattr__'s socket proxy.
                _orig_ga = model.get_action
                def _guided_get_action(*a, _orig=_orig_ga, **kw):
                    chunk = _orig(*a, **kw)
                    return compute_chunk_guidance(chunk, TASK_ENV,
                                                  threshold=threshold)
                model.get_action = _guided_get_action
                _guided_take_action = None
            elif _GUIDANCE_MODE == "cbf":
                # Safety filter on the policy's own step — recomputed from the
                # measured qpos every step, so no accumulator is needed.
                def _guided_take_action(action, _orig=_orig_ta):
                    action    = np.asarray(action, dtype=np.float32)
                    corrected = compute_cbf_correction(
                        action, TASK_ENV, threshold=threshold)
                    return _orig(corrected)
            else:
                # Legacy repulsion mode with intra-chunk carry-forward.
                _state = {'accum': np.zeros(14, dtype=np.float32), 'chunk': -1}

                def _guided_take_action(action, _orig=_orig_ta):
                    action   = np.asarray(action, dtype=np.float32)
                    t        = getattr(TASK_ENV, 'take_action_cnt', 0)
                    chunk_id = t // pi0_step
                    if chunk_id != _state['chunk']:
                        _state['accum'] = np.zeros(14, dtype=np.float32)
                        _state['chunk'] = chunk_id
                    action_adj = action + _state['accum']
                    corrected  = compute_proximity_correction(
                        action_adj, TASK_ENV, threshold=threshold, gain=gain)
                    step_dq = corrected - action_adj
                    _state['accum'] = (_state['accum'] + step_dq) * 0.9
                    return _orig(corrected)

            if _guided_take_action is not None:
                TASK_ENV.take_action = _guided_take_action

        if _GUIDANCE_MODE == "sdf" and (guided or sdf_passive):
            # Cumulative hash of everything the policy sees (images + state),
            # reported next to each chunk's action hash by the determinism
            # probe in _sdf_guided_chunk.
            _orig_uow = model.update_observation_window
            _obs_h = {'h': hashlib.sha1()}

            def _hashed_uow(img_arr, state, _orig=_orig_uow):
                try:
                    for _im in img_arr:
                        _obs_h['h'].update(np.ascontiguousarray(_im).tobytes())
                    _obs_h['h'].update(np.ascontiguousarray(
                        np.asarray(state, dtype=np.float64)).tobytes())
                except Exception:
                    pass
                return _orig(img_arr, state)

            model.update_observation_window = _hashed_uow
            _SDF_STATE['obs_hash'] = _obs_h

        succ = False
        sdf_contacts = {}
        try:
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break
        finally:
            TASK_ENV.take_action = _orig_ta
            if (guided and _GUIDANCE_MODE in ("chunk", "sdf")) or sdf_passive:
                try:
                    del model.get_action   # restore __getattr__ proxy
                except AttributeError:
                    pass
            if _GUIDANCE_MODE == "sdf":
                try:
                    del model.update_observation_window
                except AttributeError:
                    pass
                sdf_contacts = {
                    'contact_steps': _SDF_STATE.get('contact_steps', 0),
                    'force_steps':   _SDF_STATE.get('force_steps', 0),
                    'objects': sorted(_SDF_STATE.get('objects') or []),
                }
                _reset_sdf_state()     # free the per-episode SDF grid

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        cm  = getattr(TASK_ENV, 'collision_metrics', {})
        col = int(sum(cm.values())) if cm else 0

        _print_episode_guidance_summary(TASK_ENV, guided=guided, succ=succ)
        if sdf_contacts:
            print(f"  contacts: steps={sdf_contacts['contact_steps']}  "
                  f"force_steps={sdf_contacts['force_steps']}  "
                  f"objects={sdf_contacts['objects']}")

        # Free the SAPIEN render cache every rollout. The d15 clutter + depth
        # renderer leaks GPU memory across scene setups; without this the sim
        # OOMs (svulkan2/OptiX cudaErrorMemoryAllocation) after ~30 episodes.
        # Matches eval_policy_client.py / collect_rollout_proximity_client.py,
        # which clear at clear_cache_freq (1 for d15).
        TASK_ENV.close_env(clear_cache=True)
        return succ, col, sdf_contacts

    # ── Main loop ─────────────────────────────────────────────────────────────
    TASK_ENV.test_num = 0
    now_seed          = st_seed
    succ_seed         = 0
    now_id            = 0

    # Replay mode: evaluate exactly these seeds (paired comparison against a
    # previous run) — the expert pre-check never skips a listed seed.
    replay = seed_list is not None
    if replay:
        test_num  = len(seed_list)
        seed_iter = iter(seed_list)

    suc_guided  = 0;  col_guided  = 0
    suc_noguid  = 0;  col_noguid  = 0
    episode_records = []   # per-episode {seed, success, collisions} for post-hoc metrics

    while succ_seed < test_num:
        args["render_freq"] = 0
        if replay:
            now_seed = next(seed_iter)

        # Expert pre-roll: builds episode_info (instruction source); in scan
        # mode it also decides whether the seed is skipped.
        episode_info = None
        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            if episode_info is None:
                episode_info = getattr(TASK_ENV, "info", {"info": {}})
            # clear_cache=True: the pre-roll renders a full scene for every
            # seed (including ones later skipped), which also leaks GPU memory.
            TASK_ENV.close_env(clear_cache=True)
            expert_ok = TASK_ENV.plan_success and TASK_ENV.check_success()
        except UnStableError:
            TASK_ENV.close_env(clear_cache=True); expert_ok = False
        except Exception:
            TASK_ENV.close_env(clear_cache=True); expert_ok = False

        if not expert_ok:
            if replay:
                print(f"\033[91m[replay] expert pre-roll failed on fixed seed "
                      f"{now_seed} — running rollout anyway\033[0m")
                if episode_info is None:
                    episode_info = getattr(TASK_ENV, "info", {"info": {}})
            else:
                now_seed += 1; continue

        succ_seed += 1
        ep_n = TASK_ENV.test_num

        if compare:
            # ── Guided rollout ───────────────────────────────────────────
            succ_g, col_g, con_g = _run_rollout(now_id, now_seed, episode_info,
                                                guided=True,
                                                video_suffix="_guided")
            TASK_ENV.test_num = ep_n   # keep test_num stable for video naming

            # ── Unguided rollout (same seed) ─────────────────────────────
            succ_u, col_u, con_u = _run_rollout(now_id, now_seed, episode_info,
                                                guided=False,
                                                video_suffix="_noguid")

            suc_guided += int(succ_g);  col_guided += col_g
            suc_noguid += int(succ_u);  col_noguid += col_u
            episode_records.append({"seed": int(now_seed),
                                    "guided":  {"success": bool(succ_g),
                                                "collisions": int(col_g), **con_g},
                                    "unguided": {"success": bool(succ_u),
                                                 "collisions": int(col_u), **con_u}})
            n = succ_seed

            g_mark = "\033[92m✓\033[0m" if succ_g else "\033[91m✗\033[0m"
            u_mark = "\033[92m✓\033[0m" if succ_u else "\033[91m✗\033[0m"
            print(
                f"\033[90mseed {now_seed}\033[0m  ep {ep_n:3d}  "
                f"guided={g_mark} col={col_g} "
                f"fcon={con_g.get('force_steps', '?')}   "
                f"noguid={u_mark} col={col_u} "
                f"fcon={con_u.get('force_steps', '?')}"
            )
            print(
                f"  Running  guided  {suc_guided}/{n} "
                f"({100*suc_guided/n:.0f}%)  col/ep={col_guided/n:.2f}   |  "
                f"noguid  {suc_noguid}/{n} "
                f"({100*suc_noguid/n:.0f}%)  col/ep={col_noguid/n:.2f}"
            )

        else:
            # ── Single rollout mode (original behaviour) ──────────────────
            succ, col, con = _run_rollout(now_id, now_seed, episode_info,
                                          guided=guidance_enabled)
            n = succ_seed
            if guidance_enabled:
                suc_guided += int(succ);  col_guided += col
            else:
                suc_noguid += int(succ);  col_noguid += col
            episode_records.append({"seed": int(now_seed),
                                    "success": bool(succ),
                                    "collisions": int(col), **con})

            mark = "\033[92mSuccess!\033[0m" if succ else "\033[91mFail!\033[0m"
            suc_total = suc_guided if guidance_enabled else suc_noguid
            print(mark)
            print(
                f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | "
                f"\033[92m{args['task_config']}\033[0m\n"
                f"Success rate: \033[96m{suc_total}/{n}\033[0m => "
                f"\033[95m{100*suc_total/n:.1f}%\033[0m  "
                f"col/ep={col_guided/n if guidance_enabled else col_noguid/n:.2f}  "
                f"seed={now_seed}\n"
            )

        now_id  += 1
        TASK_ENV.test_num += 1
        if succ_seed % clear_cache_freq == 0:
            import gc; gc.collect()
        now_seed += 1

    return now_seed, suc_guided, suc_noguid, col_guided, col_noguid, episode_records


# ── Entry point ─────────────────────────────────────────────────────────────

def main(usr_args):
    current_time  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name     = usr_args["task_name"]
    task_config   = usr_args["task_config"]
    ckpt_setting  = usr_args["ckpt_setting"]
    policy_name   = usr_args["policy_name"]
    instruction_type = usr_args.get("instruction_type", "seen")
    port          = usr_args["port"]

    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        cfg_path = f"{os.getenv('BENCH_ROOT')}/bench_task_config/{task_config}.yml"
    else:
        cfg_path = f"./task_config/{task_config}.yml"
    with open(cfg_path, "r") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args.update({"task_name": task_name, "task_config": task_config,
                 "ckpt_setting": ckpt_setting})

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(et):
        path = _embodiment_types[et]["file_path"]
        if path is None: raise ValueError(f"missing embodiment file for {et}")
        return path

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type      = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"]  = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"]  = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"]   = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment must have 1 or 3 entries")

    args["left_embodiment_config"]  = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}"
                    f"/proximity_guided/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    if args.get("eval_video_log"):
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size    = f"{camera_config['w']}x{camera_config['h']}"
        args["eval_video_save_dir"] = save_dir

    TASK_ENV = class_decorator(task_name)
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"]  = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed      = usr_args.get("seed") or 0
    st_seed   = 100000 * (1 + seed)
    test_num  = int(os.environ.get("EVAL_TEST_NUM", 100))

    # EVAL_SEED_LIST="100000,100002,..." — replay exactly these seeds
    seed_list = None
    if os.environ.get("EVAL_SEED_LIST"):
        seed_list = [int(s) for s in os.environ["EVAL_SEED_LIST"].split(",") if s.strip()]
        test_num  = len(seed_list)
        print(f"[replay] fixed seed list ({test_num} seeds): {seed_list}")

    threshold = float(usr_args.get("guidance_threshold", _PROXIMITY_THRESHOLD))
    gain      = float(usr_args.get("guidance_gain",      _GUIDANCE_GAIN))
    guidance  = _GUIDANCE_ENABLED

    model_mirrored = {}
    if "pi0_step" in usr_args:
        model_mirrored["pi0_step"] = usr_args["pi0_step"]
    model = ModelClient(port=port, **model_mirrored)

    compare = bool(usr_args.get("compare", False))

    _, suc_g, suc_u, col_g, col_u, episode_records = eval_policy(
        task_name, TASK_ENV, args, model, st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        seed_list=seed_list,
        threshold=threshold,
        gain=gain,
        guidance_enabled=guidance,
        compare=compare,
    )

    # Per-episode records → JSONL (post-hoc metric computation)
    with open(save_dir / "episodes.jsonl", "w") as f:
        for rec in episode_records:
            f.write(json.dumps(rec) + "\n")

    file_path = save_dir / "_result.txt"
    with open(file_path, "w") as f:
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Threshold: {threshold*100:.1f} cm\n")
        f.write(f"Gain: {gain}\n")
        f.write(f"Instruction type: {instruction_type}\n\n")
        if compare:
            # ── CONTACT metrics (primary): the goal is to avoid touching ──
            fc_g = sum(1 for r in episode_records
                       if r["guided"].get("force_steps", 0) > 0)
            fc_u = sum(1 for r in episode_records
                       if r["unguided"].get("force_steps", 0) > 0)
            fs_g = sum(r["guided"].get("force_steps", 0) for r in episode_records)
            fs_u = sum(r["unguided"].get("force_steps", 0) for r in episode_records)
            cs_g = sum(r["guided"].get("contact_steps", 0) for r in episode_records)
            cs_u = sum(r["unguided"].get("contact_steps", 0) for r in episode_records)
            f.write(f"=== CONTACT (primary, n={test_num}) ===\n")
            f.write(f"Guided   force-contact eps: {fc_g}/{test_num}  "
                    f"force_steps/ep={fs_g/test_num:.1f}  "
                    f"contact_steps/ep={cs_g/test_num:.1f}\n")
            f.write(f"No-guide force-contact eps: {fc_u}/{test_num}  "
                    f"force_steps/ep={fs_u/test_num:.1f}  "
                    f"contact_steps/ep={cs_u/test_num:.1f}\n\n")
            hard_g = sum(1 for r in episode_records
                         if r["guided"]["success"] and r["guided"]["collisions"] == 0)
            hard_u = sum(1 for r in episode_records
                         if r["unguided"]["success"] and r["unguided"]["collisions"] == 0)
            coll_ep_g = sum(1 for r in episode_records if r["guided"]["collisions"] > 0)
            coll_ep_u = sum(1 for r in episode_records if r["unguided"]["collisions"] > 0)
            f.write(f"=== COMPARISON (n={test_num}) ===\n")
            f.write(f"Guided   success: {suc_g}/{test_num} = {suc_g/test_num*100:.1f}%  "
                    f"hard={hard_g}/{test_num} = {hard_g/test_num*100:.1f}%  "
                    f"col_rate={coll_ep_g/test_num*100:.1f}%  col/ep={col_g/test_num:.2f}\n")
            f.write(f"No-guide success: {suc_u}/{test_num} = {suc_u/test_num*100:.1f}%  "
                    f"hard={hard_u}/{test_num} = {hard_u/test_num*100:.1f}%  "
                    f"col_rate={coll_ep_u/test_num*100:.1f}%  col/ep={col_u/test_num:.2f}\n")
        else:
            suc_num = suc_g if guidance else suc_u
            col_num = col_g if guidance else col_u
            hard_num = sum(1 for r in episode_records
                           if r["success"] and r["collisions"] == 0)
            coll_eps = sum(1 for r in episode_records if r["collisions"] > 0)
            f.write(f"Guidance: {'ON' if guidance else 'OFF'}\n")
            f.write(f"Success rate:      {suc_num}/{test_num} = {suc_num/test_num*100:.1f}%\n")
            f.write(f"Hard success rate: {hard_num}/{test_num} = {hard_num/test_num*100:.1f}%"
                    f"  (success with zero collisions)\n")
            f.write(f"Collision rate:    {coll_eps}/{test_num} = {coll_eps/test_num*100:.1f}%"
                    f"  (episodes with >=1 collision)\n")
            f.write(f"Collisions total: {col_num}  col/ep={col_num/test_num:.2f}\n")

    print(f"\nDone. Results saved to {file_path}")


def parse_args_and_config():
    parser = argparse.ArgumentParser(
        description="Proximity-guided policy evaluation")
    parser.add_argument("--port",         type=int, required=True)
    parser.add_argument("--config",       type=str, required=True)
    parser.add_argument("--task_name",    type=str, default=None)
    parser.add_argument("--task_config",  type=str, default=None)
    parser.add_argument("--ckpt_setting", type=str, default=None)
    parser.add_argument("--guidance_threshold", type=float, default=None,
                        help="Legacy-mode activation distance in metres (default 0.05)")
    parser.add_argument("--guidance_gain", type=float, default=None,
                        help="Repulse mode: max joint correction per step in radians (default 0.20)")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Run guided and unguided rollouts back-to-back on the same seed")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["port"] = args.port

    if args.task_name    is not None: config["task_name"]    = args.task_name
    if args.task_config  is not None: config["task_config"]  = args.task_config
    if args.ckpt_setting is not None: config["ckpt_setting"] = args.ckpt_setting
    if args.guidance_threshold is not None:
        config["guidance_threshold"] = args.guidance_threshold
    if args.guidance_gain is not None:
        config["guidance_gain"] = args.guidance_gain
    if args.compare:
        config["compare"] = True

    if args.overrides:
        for i in range(0, len(args.overrides), 2):
            key = args.overrides[i].lstrip("--")
            try:    value = eval(args.overrides[i + 1])
            except: value = args.overrides[i + 1]
            config[key] = value

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()
    main(usr_args)
