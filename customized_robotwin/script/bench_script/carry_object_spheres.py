#!/usr/bin/env python3
"""P0 of the carry-leg seed (Phase C): what the held object looks like to the planner.

The carry-leg clearance grid has to be labelled with the object ATTACHED -- an IK solution that
is collision-free for a free gripper can be in collision once a 20 cm bottle is hanging off it
(already confirmed in the expert: "beside_box flips from IK-feasible to infeasible once attached
with this same chained qpos"). This module supplies the held-object model for that sweep, plus
the offline geometry analysis that decided HOW to supply it.

THE ANSWER, in one line: do not model the held object -- copy the model curobo already built.

`planner.attach_object()` voxelizes the real collision mesh into spheres on every motion_gen
(`attach_external_objects_to_robot(..., surface_sphere_radius, voxelize_method="ray")`) and writes
them, in the attached_object LINK frame, into that motion_gen's kinematics config. Both the
motion_gen and the grid's IKSolver are built from the same robot yml, which declares the
`attached_object` link with 60 sphere slots (all placeholders at centre [0,0,0], radius 0.001 --
see assets/embodiments/aloha-agilex/collision_aloha_{left,right}.yml). So the tensor read off one
can be written verbatim onto the other:

    spheres = attached_spheres_from_planner(planner)     # after the expert's own attach_object()
    apply_attached_spheres(ik, spheres)                  # the grid's collision-ON IK solver
    ...                                                  # label_volume / warm field, attached
    detach_attached_spheres(ik)                          # in a finally

That makes the grid's held-object model EXACTLY the planner's, with no bounding sphere, no
sphere-line fit, and -- the part that actually mattered -- no re-derivation of the frame chain
from the ee target pose to the attached_object link. curobo computed that from FK at the attach
joint state; we just carry the result across.

WHY NOT A NOMINAL MODEL (the approximation the plan originally assumed). Two measurements, both
reproducible with --analyze:

  1. The object's pose in the ee frame is IDENTICAL for all 8 contact points -- origin
     [0.12, 0, -0.1212] m, long axis exactly +Z, spread 0.0 deg. The 8 contact points are pure
     yaw variants about the bottle's own symmetry axis, so which one is grasped is irrelevant.
     A nominal model would have been exact here.

  2. But `choose_best_pose` does not stop at the contact point. `create_target_pose_list` rotates
     the grasp pose about the EE's own local Y (axis_type="target") through the contact point,
     ROTATE_NUM=10 steps over rotate_lim=[0,1] rad. In the ee frame that tilts the bottle by up to
     0.9 rad = 51.6 deg, and swings its origin from x=0.120 to x=0.215. A fixed nominal model is
     therefore wrong by up to half a right angle, in a direction not known until the grasp is
     chosen.

Point 2 is what kills the nominal. It costs nothing to avoid: the carry seed is built ONCE per
episode, after a grasp has already succeeded, so the exact spheres are simply available by then.
(The cacheability argument for a nominal model applied to the candidate SEARCH; the carry leg
never runs inside it.)

Usage:
    python carry_object_spheres.py --analyze     # reproduce both measurements + figure
    python carry_object_spheres.py --selftest    # no GPU, no scene, no assets beyond the model json
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

from lib.scene_constants import TARGET_ID, TARGET_MODEL

# The mesh->gripper convention from _base_task.get_grasp_pose: the contact frame is post-multiplied
# by this before the -0.12 m back-off along the gripper's approach axis. Duplicated here (rather
# than imported) so the offline analysis runs without importing the whole sapien task stack; the
# --selftest asserts it still matches _base_task.py.
GRASP_CONV = np.array([[0, 0, 1, 0],
                       [-1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, 0, 1]], dtype=float)
# get_grasp_pose's fixed back-off from the contact point along the gripper's local +X.
GRASP_BACKOFF = 0.12
ATTACH_LINK = "attached_object"


# --------------------------------------------------------------- runtime: the exact sphere copy

def attached_spheres_from_planner(planner, link_name: str = ATTACH_LINK):
    """The held-object spheres curobo built, read off `planner`'s motion_gen. Returns a CLONED
    (n, 4) tensor [x, y, z, r] in the attached_object link frame, or None if nothing is attached.

    Must be called AFTER the expert's own `planner.attach_object(...)` -- that is what voxelizes
    the mesh and populates these slots. Returns None otherwise, rather than silently handing the
    carry grid a model of nothing.

    "Nothing attached" is detected from the sphere CENTRES, not the radii. Two resting states
    both have to read as unattached, and a radius test only catches one of them:
      - after `detach_object`: centres 0, radius -100  (caught by either test)
      - freshly loaded from the robot yml: 60 placeholder slots at centre [0,0,0] with radius
        **0.001**, which is POSITIVE -- and 0.001 is also CUROBO_ATTACH_SPHERE_RADIUS, the
        surface-sphere radius a real attach uses, so radius alone cannot separate them.
    What does separate them is that a real attach spreads spheres over the object's surface,
    ~0.12 m from the link origin for this target, while every placeholder sits exactly at it.

    Cloned because the returned tensor aliases the live kinematics buffer: the expert calls
    detach_object() on its own schedule, which would zero it underneath a grid build."""
    mgs = list(planner._active_motion_gens())
    if not mgs:
        return None
    cfg = mgs[0].kinematics.kinematics_config
    spheres = cfg.get_link_spheres(link_name).clone()
    a = np.asarray(spheres.cpu().numpy() if hasattr(spheres, "cpu") else spheres, dtype=float)
    live = a[a[:, 3] > 0]
    if live.shape[0] == 0 or float(np.abs(live[:, :3]).max()) <= 1e-6:
        return None                      # disabled, or unpopulated placeholders at the origin
    return spheres


def apply_attached_spheres(ik_solver, spheres, link_name: str = ATTACH_LINK) -> bool:
    """Write `spheres` (as returned above) onto an IKSolver so its collision check sees the held
    object. Returns False when there is nothing to apply, so the caller can decide whether an
    unattached grid is acceptable rather than getting a silently wrong one.

    Shape must match the link's slot count exactly (curobo log_errors otherwise) -- it always does
    when both solvers come from the same robot yml, which is the only supported case here."""
    if spheres is None:
        return False
    ik_solver.kinematics.kinematics_config.attach_object(
        sphere_radius=None, sphere_tensor=spheres, link_name=link_name)
    return True


def detach_attached_spheres(ik_solver, link_name: str = ATTACH_LINK) -> None:
    """Undo apply_attached_spheres. Call in a `finally` -- the grid's IK solver may be reused for
    a later unattached sweep, and a leftover attached object would silently shrink its FREE set."""
    ik_solver.kinematics.kinematics_config.detach_object(link_name=link_name)


def carry_sphere_extent(spheres) -> float:
    """How far the held object reaches from the attach link origin: max(|centre| + radius) over
    the live spheres, in metres.

    This is the ONLY number the clearance metric needs beyond the attached IK sweep. Feasibility
    is already handled -- with the spheres applied, a FREE label means the arm AND the object fit.
    What is not handled is eps*, the widest-path objective, which is computed from a scalar
    obstacle EDT that still measures to a point gripper. Inflating that EDT by this extent keeps
    eps* interpretable on the carry leg ("widest corridor that fits the loaded gripper"). It stays
    a conservative scalar: the object is not a ball, so a rotation-aware measure would allow more.
    """
    if spheres is None:
        return 0.0
    # via numpy so this works on a torch tensor or a plain array (the selftest uses the latter)
    a = np.asarray(spheres.cpu().numpy() if hasattr(spheres, "cpu") else spheres, dtype=float)
    live = a[a[:, 3] > 0]
    if live.shape[0] == 0:
        return 0.0
    return float((np.linalg.norm(live[:, :3], axis=1) + live[:, 3]).max())


# ------------------------------------------------------- offline: object pose in the ee frame

def load_model_data(model_dir: Path, model_id: int) -> dict:
    with open(model_dir / f"model_data{model_id}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def object_in_ee(model_data: dict, contact_point_id: int = 0, theta: float = 0.0) -> np.ndarray:
    """4x4 transform taking OBJECT-local coordinates to the EE frame, for the grasp at
    `contact_point_id` after `theta` radians of choose_best_pose's rotation search.

    Mirrors the real chain exactly:
      contact frame C  (model_data, translation scaled by the actor scale -- Actor.get_point does
                        `local_matrix[:3,3] *= scale` and leaves rotation alone)
      ee frame      G = C @ GRASP_CONV @ translate(-GRASP_BACKOFF along local +X)   (get_grasp_pose)
      rotation search: G is rotated by theta about ITS OWN local Y, through the contact point
                       (create_target_pose_list -> rotate_along_axis with axis_type="target")
    and returns inv(G). Everything is expressed in the object's own frame, so the object's world
    pose cancels -- this transform does not depend on where the object is on the table."""
    import transforms3d as t3d
    scale = np.asarray(model_data["scale"], dtype=float)
    C = np.asarray(model_data["contact_points_pose"][contact_point_id], dtype=float).copy()
    C[:3, 3] *= scale
    back = np.eye(4)
    back[0, 3] = -GRASP_BACKOFF
    G = C @ GRASP_CONV @ back
    if theta:
        axis = G[:3, :3] @ np.array([0.0, 1.0, 0.0])       # axis_type="target" -> ee local Y
        R = t3d.axangles.axangle2mat(axis, theta)
        pivot = C[:3, 3]                                    # rotation is about the contact point
        G = np.block([[R @ G[:3, :3], (R @ (G[:3, 3] - pivot) + pivot).reshape(3, 1)],
                      [np.zeros((1, 3)), np.ones((1, 1))]])
    return np.linalg.inv(G)


def object_long_axis(model_data: dict) -> np.ndarray:
    """Unit vector along the object's longest bounding-box axis, in object-local coordinates."""
    ext = np.asarray(model_data["extents"], dtype=float)
    a = np.zeros(3)
    a[int(np.argmax(ext))] = 1.0
    return a


def rotation_search_thetas(rotate_lim=(0.0, 1.0), rotate_num: int = 10) -> np.ndarray:
    """The thetas create_target_pose_list actually enumerates: rotate_step*i + lim[0], i in
    range(ROTATE_NUM). rotate_lim comes from the embodiment config (aloha-agilex: [0, 1] rad),
    ROTATE_NUM from envs/_GLOBAL_CONFIGS.py (10)."""
    step = (rotate_lim[1] - rotate_lim[0]) / rotate_num
    return np.array([step * i + rotate_lim[0] for i in range(rotate_num)], dtype=float)


def mesh_points_in_ee(mesh_path: Path, model_data: dict, oie: np.ndarray, max_pts: int = 4000):
    """Collision-mesh vertices (scaled) mapped into the ee frame. Returns None if trimesh or the
    mesh is unavailable -- the numeric analysis does not depend on it, only the figure does."""
    try:
        import trimesh
    except ImportError:
        return None
    if not Path(mesh_path).exists():
        return None
    m = trimesh.load(str(mesh_path), force="mesh")
    V = np.asarray(m.vertices, dtype=float) * np.asarray(model_data["scale"], dtype=float)
    if V.shape[0] > max_pts:                      # figure only; a subsample is plenty
        V = V[np.random.default_rng(0).choice(V.shape[0], max_pts, replace=False)]
    return (oie[:3, :3] @ V.T).T + oie[:3, 3]


# ------------------------------------------------------------------------------- the analysis

def analyze(model_dir: Path, model_id: int, out_dir: Path, figure: bool = True) -> dict:
    """Reproduce both measurements and write summary.json (+ a figure). Returns the summary."""
    md = load_model_data(model_dir, model_id)
    scale = np.asarray(md["scale"], dtype=float)
    extents_m = np.asarray(md["extents"], dtype=float) * scale
    axis_local = object_long_axis(md)
    n_cp = len(md["contact_points_pose"])

    # --- measurement 1: does the contact point choice move the object in the ee frame?
    per_cp = []
    for i in range(n_cp):
        oie = object_in_ee(md, i, 0.0)
        per_cp.append({"contact_point_id": i,
                       "origin_in_ee": (oie[:3, 3]).tolist(),
                       "long_axis_in_ee": (oie[:3, :3] @ axis_local).tolist()})
    A = np.array([c["long_axis_in_ee"] for c in per_cp])
    O = np.array([c["origin_in_ee"] for c in per_cp])
    cp_axis_spread_deg = float(np.degrees(np.arccos(np.clip((A @ A.T).min(), -1.0, 1.0))))
    cp_origin_spread_m = float(np.linalg.norm(O - O.mean(axis=0), axis=1).max())

    # --- measurement 2: what does the rotation search do to it?
    thetas = rotation_search_thetas()
    per_theta = []
    for th in thetas:
        oie = object_in_ee(md, 0, float(th))
        ax = oie[:3, :3] @ axis_local
        per_theta.append({"theta_rad": float(th),
                          "tilt_deg": float(np.degrees(np.arccos(np.clip(ax @ np.array([0, 0, 1.0]), -1, 1)))),
                          "origin_in_ee": (oie[:3, 3]).tolist(),
                          "long_axis_in_ee": ax.tolist()})
    max_tilt = max(p["tilt_deg"] for p in per_theta)

    summary = {
        "model": f"{model_dir.name}/id{model_id}",
        "scale": scale.tolist(),
        "extents_m": extents_m.tolist(),
        "long_axis_local": axis_local.tolist(),
        "length_m": float(extents_m.max()),
        "cross_section_radius_m": float(np.sort(extents_m)[-2] / 2.0),
        "n_contact_points": n_cp,
        "contact_point_axis_spread_deg": cp_axis_spread_deg,
        "contact_point_origin_spread_m": cp_origin_spread_m,
        "contact_point_invariant": bool(cp_axis_spread_deg < 1e-6 and cp_origin_spread_m < 1e-6),
        "rotation_search_thetas_rad": thetas.tolist(),
        "rotation_search_max_tilt_deg": max_tilt,
        "per_contact_point": per_cp,
        "per_theta": per_theta,
        "verdict": ("contact point is irrelevant, but the rotation search tilts the object by up to "
                    f"{max_tilt:.1f} deg -- a nominal held-object model cannot be fixed ahead of the "
                    "grasp, so copy curobo's own attached spheres instead"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if figure:
        _figure(md, model_dir / "collision" / f"base{model_id}.glb", summary,
                out_dir / "carry_object_in_ee.png")
    return summary


def _figure(md, mesh_path, summary, png_path):
    """Two panels, both in the EE frame (gripper at the origin, +X = the approach direction):
    where the object sits at theta=0, and how the rotation search moves it."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[carry-spheres] matplotlib unavailable -> skipping figure")
        return
    axis_local = object_long_axis(md)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    def draw(ax, theta, color, label, alpha=0.35):
        oie = object_in_ee(md, 0, theta)
        P = mesh_points_in_ee(mesh_path, md, oie)
        if P is not None:
            ax.scatter(P[:, 0], P[:, 2], s=1, color=color, alpha=alpha, linewidths=0, label=label)
        else:                                    # no mesh -> draw the axis segment instead
            o = oie[:3, 3]
            a = oie[:3, :3] @ axis_local
            L = summary["length_m"]
            ax.plot([o[0], o[0] + a[0] * L], [o[2], o[2] + a[2] * L], c=color, lw=3,
                    alpha=0.8, label=label)

    ax = axes[0]
    draw(ax, 0.0, "tab:blue", "held object", alpha=0.5)
    ax.plot(0, 0, marker="+", ms=18, mew=3, c="k")
    ax.annotate("gripper\n(ee frame origin)", (0, 0), textcoords="offset points", xytext=(6, -34),
                fontsize=9, ha="left")
    o = summary["per_theta"][0]["origin_in_ee"]
    ax.plot(o[0], o[2], marker="o", ms=7, mfc="none", mec="k", mew=1.5)
    ax.text(0.03, 0.06,
            f"object origin  x={o[0]:.3f}  z={o[2]:.3f} m\n"
            f"hangs {abs(o[2])*100:.1f} cm below the gripper,\n"
            f"{o[0]*100:.1f} cm out along the approach axis",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.9))
    ax.set_title(f"Where the held object sits, seen from the gripper\n"
                 f"{summary['model']}  --  {summary['length_m']*100:.1f} cm long, "
                 f"{summary['cross_section_radius_m']*200:.1f} cm across", fontsize=11)

    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for k, p in enumerate(summary["per_theta"]):
        draw(ax, p["theta_rad"], cmap(k / max(1, len(summary["per_theta"]) - 1)),
             f"{p['theta_rad']:.1f} rad" if k in (0, len(summary["per_theta"]) - 1) else None,
             alpha=0.25)
    ax.plot(0, 0, marker="+", ms=18, mew=3, c="k")
    ax.set_title("choose_best_pose's rotation search moves it\n"
                 f"10 candidate grasps span {summary['rotation_search_max_tilt_deg']:.1f} deg of tilt;"
                 " the winner is unknown until planning", fontsize=11)

    for ax in axes:
        ax.set_xlabel("ee +X  (approach direction), m")
        ax.set_ylabel("ee +Z, m")
        ax.axhline(0, lw=0.5, c="0.7"); ax.axvline(0, lw=0.5, c="0.7")
        ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("P0: the held object in the gripper's frame  --  why the carry grid copies "
                 "curobo's attached spheres instead of assuming a shape", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"saved figure -> {png_path}")


# ------------------------------------------------------------------------------------ selftest

def _selftest() -> int:
    """Pure geometry -- no GPU, no sapien, no curobo."""
    repo = Path(__file__).resolve().parents[2]
    md_path = repo / "assets" / "objects" / "001_bottle"
    md = load_model_data(md_path, 9)

    # the duplicated convention must still match the live one in _base_task.py
    base = (repo / "envs" / "_base_task.py").read_text(encoding="utf-8")
    assert "[[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0]" in base, "GRASP_CONV drifted from _base_task"
    assert "np.array([-0.12 - pre_dis, 0, 0])" in base, "GRASP_BACKOFF drifted from _base_task"

    # measurement 1: contact-point invariance
    axis_local = object_long_axis(md)
    axes = [object_in_ee(md, i, 0.0)[:3, :3] @ axis_local for i in range(len(md["contact_points_pose"]))]
    A = np.array(axes)
    assert np.degrees(np.arccos(np.clip((A @ A.T).min(), -1, 1))) < 1e-6, "contact points disagree"
    assert np.allclose(A[0], [0, 0, 1], atol=1e-6), f"expected +Z long axis, got {A[0]}"

    # measurement 2: the rotation search tilts it, and the tilt equals theta
    for th in rotation_search_thetas():
        ax = object_in_ee(md, 0, float(th))[:3, :3] @ axis_local
        tilt = np.degrees(np.arccos(np.clip(ax @ np.array([0, 0, 1.0]), -1, 1)))
        assert abs(tilt - np.degrees(th)) < 1e-6, f"tilt {tilt} != theta {np.degrees(th)}"

    # the transform must not depend on where the object is in the world
    scale = np.asarray(md["scale"], float)
    C = np.asarray(md["contact_points_pose"][0], float).copy(); C[:3, 3] *= scale
    M = np.eye(4); M[:3, 3] = [0.37, -0.11, 0.83]            # arbitrary world placement
    back = np.eye(4); back[0, 3] = -GRASP_BACKOFF
    G_world = M @ C @ GRASP_CONV @ back
    assert np.allclose(np.linalg.inv(G_world) @ M, object_in_ee(md, 0, 0.0), atol=1e-9), \
        "object_in_ee depends on the world pose -- it must cancel"

    # extents sanity: a ~20 cm bottle
    ext = np.asarray(md["extents"], float) * scale
    assert 0.15 < ext.max() < 0.30, f"unexpected object length {ext.max()}"

    _selftest_transfer()
    print("[selftest] ALL PASS (grasp-convention drift / contact-point invariance / "
          "rotation-search tilt / world-pose cancellation / extents / sphere transfer)")
    return 0


def _selftest_transfer() -> None:
    """The motion_gen -> IKSolver sphere transfer, against stubs that mimic curobo's API.

    Worth stubbing rather than leaving to the first GPU run: the failure this guards is silent.
    If attached_spheres_from_planner returned the placeholder slots instead of None, the carry
    grid would be labelled for a robot holding nothing, quietly promise routes the bottle cannot
    take, and show up only as unexplained trajopt failures much later."""
    # numpy stand-in for the torch tensor curobo really returns: .clone() is the only tensor-ism
    # attached_spheres_from_planner uses, and the clone is load-bearing (see below).
    class _Arr(np.ndarray):
        def clone(self):
            return np.array(self).view(_Arr)

    def arr(rows):
        return np.asarray(rows, dtype=float).view(_Arr)

    class _Cfg:
        def __init__(self, spheres):
            self.s = {ATTACH_LINK: np.asarray(spheres, dtype=float).view(_Arr)}

        def get_link_spheres(self, link_name):
            return self.s[link_name]

        def attach_object(self, sphere_radius=None, sphere_tensor=None, link_name=ATTACH_LINK):
            cur = self.s[link_name]
            if sphere_radius is not None:
                cur[:, 3] = sphere_radius
            if sphere_tensor is not None:
                assert sphere_tensor.shape == cur.shape, "shape mismatch curobo would log_error on"
                cur[:, :] = sphere_tensor
            return True

        def detach_object(self, link_name=ATTACH_LINK):
            self.s[link_name][:, 3] = -100.0
            self.s[link_name][:, :3] = 0.0
            return True

    class _Holder:
        def __init__(self, cfg):
            self.kinematics = type("K", (), {"kinematics_config": cfg})()

    class _Planner:
        def __init__(self, cfg):
            self._mg = _Holder(cfg)

        def _active_motion_gens(self):
            return [self._mg]

    n = 8
    # NOTE the placeholder radius: 0.001 from the robot yml, POSITIVE, and equal to
    # CUROBO_ATTACH_SPHERE_RADIUS. A radius-only "is anything attached" test passes this and
    # would label the whole carry grid as if the gripper were empty. That bug existed and this
    # case is what caught it -- keep the 0.001 exactly as the yml has it.
    placeholders = arr([[0.0, 0.0, 0.0, 0.001]] * n)        # fresh from the robot yml
    detached = arr([[0.0, 0.0, 0.0, -100.0]] * n)           # after curobo's detach_object
    # a real attach: surface spheres of the same 0.001 radius, spread over the held object
    attached = arr([[0.12, 0.0, -0.02 * i, 0.001] for i in range(n)])

    # nothing attached -> None, in BOTH resting states, so an unattached grid is never built
    for state in (placeholders, detached):
        assert attached_spheres_from_planner(_Planner(_Cfg(state.copy()))) is None, \
            "unattached slots must read as None, not as a model of nothing"

    # attached -> exact round trip onto a separate solver
    src_cfg = _Cfg(attached.copy())
    spheres = attached_spheres_from_planner(_Planner(src_cfg))
    assert spheres is not None and np.allclose(spheres, attached), "read-back changed the spheres"
    dst_cfg = _Cfg(placeholders.copy())
    dst = _Holder(dst_cfg)
    assert apply_attached_spheres(dst, spheres) is True
    assert np.allclose(dst_cfg.get_link_spheres(ATTACH_LINK), attached), "transfer lost data"

    # the returned tensor must be a CLONE: the expert detaches on its own schedule, and an
    # alias would be zeroed underneath a grid build already in progress
    src_cfg.detach_object()
    assert np.allclose(spheres, attached), "returned tensor aliased the live buffer"

    # detach on the destination disables collision again (a reused solver must not stay loaded)
    detach_attached_spheres(dst)
    assert float(dst_cfg.get_link_spheres(ATTACH_LINK)[:, 3].max()) <= 0.0, "detach left spheres live"

    # applying nothing is a no-op that reports it, so the caller can choose to refuse
    assert apply_attached_spheres(dst, None) is False

    # extent = furthest point of the held object from the attach origin
    _far = float(np.linalg.norm([0.12, 0.0, -0.02 * (n - 1)])) + 0.001
    assert abs(carry_sphere_extent(attached) - _far) < 1e-9, \
        "extent must be max(|centre| + radius)"
    assert carry_sphere_extent(None) == 0.0
    assert carry_sphere_extent(detached) == 0.0, "disabled spheres must not count toward extent"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--analyze", action="store_true", help="run the geometry analysis (default)")
    ap.add_argument("--model-dir", default=None, help="assets/objects/<model> (default: the bench target)")
    ap.add_argument("--model-id", type=int, default=None)
    ap.add_argument("--out-dir", default=None,
                    help="default: scripts/validation/results/carry_spheres/<stamp>/")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    repo = Path(__file__).resolve().parents[2]
    # Default to the same shared target identity used by the rollout task.
    model, mid = TARGET_MODEL, TARGET_ID
    model_dir = Path(args.model_dir) if args.model_dir else repo / "assets" / "objects" / model
    model_id = args.model_id if args.model_id is not None else mid

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        repo.parent / "scripts" / "validation" / "results" / "carry_spheres" / stamp)

    t0 = time.perf_counter()
    summary = analyze(model_dir, model_id, out_dir, figure=not args.no_figure)
    seconds = time.perf_counter() - t0
    with open(out_dir / "timings.json", "w", encoding="utf-8") as f:
        json.dump({"analyze_seconds": seconds, "stamp": stamp,
                   "model": summary["model"]}, f, indent=2)

    print(f"\n{summary['model']}: {summary['length_m']*100:.1f} cm long, "
          f"{summary['cross_section_radius_m']*200:.1f} cm across (scale {summary['scale'][0]})")
    print(f"  contact points ({summary['n_contact_points']}): axis spread "
          f"{summary['contact_point_axis_spread_deg']:.3g} deg, origin spread "
          f"{summary['contact_point_origin_spread_m']:.3g} m "
          f"-> {'INVARIANT' if summary['contact_point_invariant'] else 'VARIES'}")
    print(f"  rotation search: tilts the object up to "
          f"{summary['rotation_search_max_tilt_deg']:.1f} deg")
    print(f"  verdict: {summary['verdict']}")
    print(f"\nwrote -> {out_dir}  ({seconds:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
