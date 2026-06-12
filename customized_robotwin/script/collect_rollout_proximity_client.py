"""
collect_rollout_proximity_client.py — proximity-aware policy rollout data collection.

Saves episodes (success and/or failure depending on COLLECT_SELECTIVE_SAVE) with:
  - Per-step proximity distance (min_dist) and direction vectors written to HDF5.
  - All four cameras: countertop, right, left, head.
  - Output goes to rollout_data/{task}/{env_type}/.

HDF5 structure per episode — matches collect_data.py standard exactly, with extra keys marked:

    Standard keys (identical to collect_data.py):
        observation/{cam}/rgb          JPEG  (T,)
        observation/{cam}/depth        PNG   (T,)     when data_type.depth=true
        joint_action/left_arm          f32   (T, 6)
        joint_action/left_gripper      f32   (T,)
        joint_action/right_arm         f32   (T, 6)
        joint_action/right_gripper     f32   (T,)
        joint_action/vector            f32   (T, 14)
        endpose/left_endpose           f32   (T, 7)
        endpose/left_gripper           f32   (T,)
        endpose/right_endpose          f32   (T, 7)
        endpose/right_gripper          f32   (T,)
        proximity/{part}/min_dist      f32   (T,)
        proximity/{part}/delta         f32   (T, 3)

    Extra keys (not in standard):
        action/left_arm                f32   (T, 6)   policy command at obs[t]
        action/left_gripper            f32   (T,)
        action/right_arm               f32   (T, 6)
        action/right_gripper           f32   (T,)
        action/vector                  f32   (T, 14)
        label                          int8  scalar   1=success / 0=failure
        attrs: instruction, episode, success, collision

Usage (via shell script):
    bash policy/pi05/collect_rollout.sh <task> <task_config> <train_config> \\
        <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]

Env vars:
    COLLECT_NUM              — episodes to collect (default 100)
    COLLECT_START_SEED       — starting seed (default 100000*(1+seed))
    COLLECT_EXPERT_CHECK     — if set, run expert CuRobo check before each episode (filters invalid seeds; slower)
    COLLECT_BRANCH_NUM       — CuRobo branches per failure (default 0 = simple rollout)
    COLLECT_SELECTIVE_SAVE   — if set, use selective save: primary saved only on fail/collision,
                               branches saved only on success. Default: save everything.
"""

import sys
import os

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from script.bench_script.setup_paths import setup_paths
setup_paths()

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import socket
import json
import subprocess
import h5py
import cv2
import time
import traceback
import yaml
import argparse
import importlib
import base64

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from generate_episode_instructions import generate_episode_descriptions


# ── Numpy ↔ JSON serialization ────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            dtype = str(obj.dtype) if obj.dtype not in (np.float32, np.float64, np.int32, np.int64) else str(obj.dtype)
            return {'__numpy_array__': True, 'data': base64.b64encode(obj.tobytes()).decode('ascii'),
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
            return np.frombuffer(base64.b64decode(dct['data']), dtype=dct['dtype']).reshape(dct['shape'])
        return dct
    return json.loads(json_str, object_hook=object_hook)


# ── ModelClient ────────────────────────────────────────────────────────────

class ModelClient:
    _MIRRORED_DEFAULTS = {"observation_window": None}

    def __init__(self, host='localhost', port=9999, timeout=30, **mirrored):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
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
                print(f"⚠️ Connection attempt {attempt+1} failed: {e}. Retrying in 5s...")
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
        return self._send_recv({"cmd": func_name, "obs": obs})['res']

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


# ── Helpers ────────────────────────────────────────────────────────────────

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


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../task_config/_camera_config.yml")
    with open(camera_config_path, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg[camera_type]


# ── Frame builder — shared by both formats ────────────────────────────────

_TOP_K_OBSTACLES       = 3     # obstacles to record per link
_CUROBO_SAVE_FREQ      = 15   # 250Hz physics / 15 ≈ 17fps — matches pi05 take_action rate
_MESH_REFINE_THRESHOLD = 0.3  # only do trimesh refinement when AABB dist < this (m)
_N_AABB_CANDIDATES     = 10   # top-N AABB candidates to consider for trimesh refinement

# Only track the 16 policy-controlled arm links (fl/fr, joints 1-8).
# lr/rr arms are passive (∂pos/∂policy_joints = 0); cameras and base/wheel links excluded.
_ARM_LINK_NAMES = frozenset({
    'fl_link1', 'fl_link2', 'fl_link3', 'fl_link4',
    'fl_link5', 'fl_link6', 'fl_link7', 'fl_link8',
    'fr_link1', 'fr_link2', 'fr_link3', 'fr_link4',
    'fr_link5', 'fr_link6', 'fr_link7', 'fr_link8',
})

def _compute_link_distances(task_env):
    """Two-phase distance from each robot link center to the top-K closest obstacles.

    Phase 1 — vectorized AABB:
        Batch point-to-AABB for every obstacle in the scene.  Fast O(M) pre-filter
        that identifies the closest candidates.

    Phase 2 — trimesh refinement:
        For any AABB candidate within _MESH_REFINE_THRESHOLD, refine with
        trimesh.proximity.closest_point on the actor's actual collision mesh
        (if present in task_env._proximity_mesh_cache).  This is the same two-phase
        approach used by _compute_proximity_step in _bench_base_task.py and gives the
        true surface-to-surface distance instead of the conservative AABB gap.
        Falls back to AABB when no mesh is cached (articulations, uncached actors).

    Excluded from obstacle set:
        ground / table / wall — arm legitimately sweeps near these
        task target objects   — intentionally touched during grasping

    Returns:
        {link_name: {'dists': (K,) float32, 'deltas': (K, 3) float32}}
        Sorted ascending by distance.  Padded with dist=99 / delta=0 for missing slots.
        delta points FROM the link center TOWARD the nearest surface point.
    """
    robot = task_env.robot
    links = list(robot.left_entity.get_links())
    if robot.right_entity is not robot.left_entity:
        links += list(robot.right_entity.get_links())
    links = [lk for lk in links if lk.get_name() in _ARM_LINK_NAMES]
    if not links:
        return {}

    robot_entities = {robot.left_entity, robot.right_entity}
    mesh_cache     = getattr(task_env, '_proximity_mesh_cache', {})

    _exclude_names = frozenset({"ground", "table", "wall"}) | frozenset(
        getattr(task_env, 'target_object_names', set())
    )

    # ── Build obstacle list ───────────────────────────────────────────────
    # Actors: static scene objects — may have mesh cache entry for trimesh refinement.
    # obstacle_actors: list of (actor, name, aabb_min(3,), aabb_max(3,))
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

    # Non-robot articulations: no mesh cache available, AABB-only.
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

    # Pre-build flat AABB arrays for vectorized batch — actors first, then AABB-only.
    n_actor = len(obstacle_actors)
    all_mins = np.array(
        [o[2] for o in obstacle_actors] + aabb_only_mins, dtype=np.float32)  # (M, 3)
    all_maxs = np.array(
        [o[3] for o in obstacle_actors] + aabb_only_maxs, dtype=np.float32)  # (M, 3)
    M = len(all_mins)
    N_CAND = min(_N_AABB_CANDIDATES, M)

    result = {}
    for lk in links:
        if not lk.collision_shapes:
            continue
        try:
            link_center = np.array(lk.get_pose().p, dtype=np.float64)  # float64 for trimesh
        except Exception:
            continue

        lc32 = link_center.astype(np.float32)

        # Phase 1: vectorized point-to-AABB for all obstacles
        clamped    = np.clip(lc32, all_mins, all_maxs)   # (M, 3)
        delta_aabb = clamped - lc32                        # (M, 3) toward obstacle surface
        dists_aabb = np.linalg.norm(delta_aabb, axis=1)   # (M,)

        cand_idx    = np.argsort(dists_aabb)[:N_CAND]
        final_dists  = dists_aabb[cand_idx].copy().astype(np.float64)
        final_deltas = delta_aabb[cand_idx].copy().astype(np.float64)

        # Phase 2: trimesh refinement for actor candidates within threshold
        for ci, oi in enumerate(cand_idx):
            if dists_aabb[oi] >= _MESH_REFINE_THRESHOLD:
                break  # sorted ascending — nothing further qualifies
            if oi >= n_actor:
                continue  # AABB-only obstacle, no mesh
            actor, name, _, _ = obstacle_actors[oi]
            if name not in mesh_cache:
                continue
            try:
                mesh  = mesh_cache[name]
                T     = actor.get_pose().to_transformation_matrix()
                T_inv = np.linalg.inv(T)
                local_pt = T_inv[:3, :3] @ link_center + T_inv[:3, 3]
                closest_pts, mesh_dists, _ = trimesh.proximity.closest_point(mesh, [local_pt])
                closest_world = T[:3, :3] @ closest_pts[0] + T[:3, 3]
                final_dists[ci]  = float(mesh_dists[0])
                final_deltas[ci] = closest_world - link_center
            except Exception:
                pass  # keep AABB result on any error

        # Re-sort after refinement (trimesh may change ranking), keep top-K
        order      = np.argsort(final_dists)[:K]
        top_dists  = final_dists[order].astype(np.float32)
        top_deltas = final_deltas[order].astype(np.float32)

        if len(top_dists) < K:
            pad = K - len(top_dists)
            top_dists  = np.concatenate([top_dists,  np.full(pad,       99.0, dtype=np.float32)])
            top_deltas = np.concatenate([top_deltas, np.zeros((pad, 3),       dtype=np.float32)])

        result[lk.get_name()] = {'dists': top_dists, 'deltas': top_deltas}

    return result


def _obs_to_frame(obs, state, action=None, task_env=None):
    """Build a buffer frame dict including depth, endpose, proximity, pcd, link_dists, and action."""
    _CAMS = ('countertop_camera', 'right_camera', 'left_camera', 'head_camera')
    frame = {
        'obs':       {cam: obs['observation'][cam]['rgb'].copy()
                      for cam in _CAMS if cam in obs['observation']},
        'state':     state.copy(),
        'proximity': obs.get('proximity', {}),
    }
    depth_frames = {
        cam: obs['observation'][cam]['depth'].copy()
        for cam in _CAMS
        if cam in obs['observation'] and 'depth' in obs['observation'][cam]
    }
    if depth_frames:
        frame['depth'] = depth_frames
    endpose = obs.get('endpose', {})
    if endpose:
        frame['endpose'] = {k: np.array(v, dtype=np.float32) for k, v in endpose.items()}
    if action is not None:
        frame['action'] = np.asarray(action, dtype=np.float32).copy()

    # Per-link distances via simulator AABB — independent of point cloud
    if task_env is not None:
        frame['link_dists'] = _compute_link_distances(task_env)

    # Per-step scene state: object poses keyed by per_scene_id + articulation
    # joint states.  Objects MOVE mid-episode (uniquely per run) — this is the
    # generator the relabel pass needs to keep clearance labels correct after
    # displacement (scene/seed_N stores only the INITIAL state).
    if task_env is not None:
        try:
            frame['object_poses'] = {
                int(a.per_scene_id): np.concatenate(
                    [np.asarray(a.get_pose().p, dtype=np.float32),
                     np.asarray(a.get_pose().q, dtype=np.float32)])
                for a in task_env.scene.get_all_actors()
            }
            robot = task_env.robot
            _robot_ents = {robot.left_entity, robot.right_entity}
            frame['articulation_qpos'] = {
                j: np.asarray(art.get_qpos(), dtype=np.float32).ravel()
                for j, art in enumerate(task_env.scene.get_all_articulations())
                if art not in _robot_ents
            }
        except Exception:
            pass

    return frame


# ── PROXIMITY: HDF5 writer with label + proximity datasets ────────────────

def save_proximity_hdf5(buffer, save_dir, ep_idx, instruction, success, collision, collector="pi05",
                        extra_attrs=None, filename=None):
    """Write one rollout episode to HDF5.

    Always saved regardless of success/failure.
    Matches the standard collect_data.py HDF5 schema exactly, with three extra keys:
        action/*, label, and attrs (instruction, success, collision).

    Structure (standard keys):
        observation/{cam}/rgb          — JPEG-encoded frames   (T,)
        observation/{cam}/depth        — PNG-encoded depth     (T,)   when available
        joint_action/left_arm          — float32               (T, 6)
        joint_action/left_gripper      — float32               (T,)
        joint_action/right_arm         — float32               (T, 6)
        joint_action/right_gripper     — float32               (T,)
        joint_action/vector            — float32               (T, 14)
        endpose/left_endpose           — float32               (T, 7)
        endpose/left_gripper           — float32               (T,)
        endpose/right_endpose          — float32               (T, 7)
        endpose/right_gripper          — float32               (T,)
        proximity/{part}/min_dist      — float32               (T,)
        proximity/{part}/delta         — float32               (T, 3)

    Extra keys (not in standard):
        action/left_arm                — float32               (T, 6)   policy command
        action/left_gripper            — float32               (T,)
        action/right_arm               — float32               (T, 6)
        action/right_gripper           — float32               (T,)
        action/vector                  — float32               (T, 14)
        label                          — int8 scalar           1=success / 0=failure
        attrs: instruction, episode, success, collision
    """
    # Auto-detect next available index so multiple runs into the same folder continue cleanly
    data_path = Path(save_dir)
    if filename is not None:
        hdf5_path = data_path / filename          # caller-controlled (overwrites on rerun)
    else:
        existing  = [int(p.stem.split('_')[1]) for p in data_path.glob("episode_*.hdf5")
                     if p.stem.split('_')[1].isdigit()]
        file_idx  = max(existing, default=-1) + 1
        hdf5_path = data_path / f"episode_{file_idx}.hdf5"
    cam_names      = list(buffer[0]['obs'].keys())
    cam_images     = {cam: [] for cam in cam_names}
    cam_depths     = {}
    states         = []   # joint_action/vector shape (T, 14)
    actions        = []   # policy action shape (T, 14)
    endpose_accum  = {}
    prox_accum     = {}

    for frame in buffer:
        for cam in cam_names:
            cam_images[cam].append(frame['obs'][cam])
        states.append(frame['state'])
        if frame.get('action') is not None:
            actions.append(frame['action'])
        for k, v in frame.get('endpose', {}).items():
            endpose_accum.setdefault(k, []).append(v)
        for part, vals in frame.get('proximity', {}).items():
            if part not in prox_accum:
                prox_accum[part] = {'min_dist': [], 'delta': []}
            prox_accum[part]['min_dist'].append(vals['min_dist'])
            prox_accum[part]['delta'].append(vals['delta'])
        for cam, depth_arr in frame.get('depth', {}).items():
            cam_depths.setdefault(cam, []).append(depth_arr)

    states_arr  = np.array(states,  dtype=np.float32)   # (T, 14)
    actions_arr = np.array(actions, dtype=np.float32) if actions else None  # (T, 14)

    with h5py.File(hdf5_path, 'w') as f:
        f.attrs['instruction'] = np.bytes_(instruction)
        f.attrs['episode']     = ep_idx
        f.attrs['success']     = success
        f.attrs['collision']   = collision
        f.attrs['collector']   = np.bytes_(collector)   # 'pi05' | 'curobo' | 'curobo_unaware' | 'stitched'
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = np.bytes_(v) if isinstance(v, str) else v

        # ── observation (matches standard key name) ───────────────────────
        obs_grp = f.create_group('observation')

        for cam in cam_names:
            encoded, max_len = [], 0
            for img in cam_images[cam]:
                _, buf = cv2.imencode('.jpg', img)
                data = buf.tobytes()
                encoded.append(data)
                max_len = max(max_len, len(data))
            padded = [d.ljust(max_len, b'\0') for d in encoded]
            obs_grp.create_dataset(f'{cam}/rgb', data=padded, dtype=f'S{max_len}')

        for cam, depths in cam_depths.items():
            encoded, max_len = [], 0
            for d in depths:
                d_u16 = np.asarray(d).clip(0, 65535).astype(np.uint16)
                _, buf = cv2.imencode('.png', d_u16)
                data = buf.tobytes()
                encoded.append(data)
                max_len = max(max_len, len(data))
            padded = [d.ljust(max_len, b'\0') for d in encoded]
            obs_grp.create_dataset(f'{cam}/depth', data=padded, dtype=f'S{max_len}')

        # ── joint_action (matches standard split structure) ───────────────
        ja_grp = f.create_group('joint_action')
        ja_grp.create_dataset('left_arm',      data=states_arr[:, 0:6])
        ja_grp.create_dataset('left_gripper',  data=states_arr[:, 6])
        ja_grp.create_dataset('right_arm',     data=states_arr[:, 7:13])
        ja_grp.create_dataset('right_gripper', data=states_arr[:, 13])
        ja_grp.create_dataset('vector',        data=states_arr)

        # ── endpose (same structure as standard) ─────────────────────────
        if endpose_accum:
            ep_grp = f.create_group('endpose')
            for k, vals in endpose_accum.items():
                ep_grp.create_dataset(k, data=np.array(vals, dtype=np.float32))

        # ── proximity (same structure as standard) ───────────────────────
        if prox_accum:
            prox_grp = f.create_group('proximity')
            for part, arrays in prox_accum.items():
                pg = prox_grp.create_group(part)
                pg.create_dataset('min_dist', data=np.array(arrays['min_dist'], dtype=np.float32))
                pg.create_dataset('delta',    data=np.array(arrays['delta'],    dtype=np.float32))


        # ── per-link top-K distances (dist + delta) ──────────────────────────
        ld_list  = [fr.get('link_dists', {}) for fr in buffer]
        first_ld = next((ld for ld in ld_list if ld), {})
        if first_ld:
            ln    = list(first_ld.keys())
            T_buf = len(buffer)
            K     = _TOP_K_OBSTACLES
            dist_arr  = np.full((T_buf, len(ln), K),    99.0, dtype=np.float32)
            delta_arr = np.zeros((T_buf, len(ln), K, 3),      dtype=np.float32)
            for t, ld in enumerate(ld_list):
                for j, name in enumerate(ln):
                    if name in ld:
                        dist_arr[t, j]  = ld[name]['dists']    # (K,)
                        delta_arr[t, j] = ld[name]['deltas']   # (K, 3)
            ld_grp = f.create_group('link_distances')
            ld_grp.create_dataset('dist',  data=dist_arr)   # (T, N_links, K)
            ld_grp.create_dataset('delta', data=delta_arr)  # (T, N_links, K, 3)
            ld_grp.attrs['link_names'] = [n.encode() for n in ln]
            ld_grp.attrs['top_k']      = K

        # ── per-step collision flag ──────────────────────────────────────
        coll_steps = np.array([fr.get('collision_step', False) for fr in buffer], dtype=bool)
        f.create_dataset('collision_per_step', data=coll_steps)

        # ── extra keys (not in standard) ─────────────────────────────────
        if actions_arr is not None:
            ac_grp = f.create_group('action')
            ac_grp.create_dataset('left_arm',      data=actions_arr[:, 0:6])
            ac_grp.create_dataset('left_gripper',  data=actions_arr[:, 6])
            ac_grp.create_dataset('right_arm',     data=actions_arr[:, 7:13])
            ac_grp.create_dataset('right_gripper', data=actions_arr[:, 13])
            ac_grp.create_dataset('vector',        data=actions_arr)

        f.create_dataset('label', data=np.int8(1 if success else 0))

        # ── per-step scene state (object pose track + articulation qpos) ──
        op = [fr.get('object_poses') for fr in buffer]
        if op and all(o is not None for o in op):
            ids = sorted(op[0].keys())
            if all(sorted(o.keys()) == ids for o in op):
                grp = f.create_group('object_poses')
                grp.create_dataset('ids', data=np.asarray(ids, dtype=np.int32))
                grp.create_dataset('pose', data=np.stack(
                    [np.stack([o[i] for i in ids]) for o in op]))   # (T, n, 7) p+q(wxyz)
        aq = [fr.get('articulation_qpos') for fr in buffer]
        if aq and all(a is not None for a in aq) and aq[0]:
            grp = f.create_group('articulation_qpos')
            for j in sorted(aq[0].keys()):
                try:
                    grp.create_dataset(f'art_{j}', data=np.stack([a[j] for a in aq]))
                except Exception:
                    pass

    return hdf5_path  # callers derive video names / index entries from the path


# ── Scene state snapshot / restore ────────────────────────────────────────

def _snapshot_state(TASK_ENV):
    robot = TASK_ENV.robot
    _robot_entities = {robot.left_entity, robot.right_entity}
    snap = {
        'take_action_cnt': TASK_ENV.take_action_cnt,
        'left_qpos': robot.left_entity.get_qpos().copy(),
        'left_qvel': robot.left_entity.get_qvel().copy(),
        'left_ee_pose':  robot.get_left_ee_pose(),
        'right_ee_pose': robot.get_right_ee_pose(),
        # Keyed by per_scene_id — actor NAMES are duplicated in cluttered
        # scenes (three '108_block', ...) and a name-keyed map silently
        # restores all duplicates onto one entity.
        'actors': [(a.per_scene_id, a.get_pose())
                   for a in TASK_ENV.scene.get_all_actors()],
        # Articulations: keyed by index in the (robot-filtered) enumeration —
        # stable within an episode, immune to duplicate names.
        'articulations': [
            (i, art.get_qpos().copy(), art.get_qvel().copy())
            for i, art in enumerate(
                art for art in TASK_ENV.scene.get_all_articulations()
                if art not in _robot_entities)
        ],
    }
    if robot.right_entity is not robot.left_entity:
        snap['right_qpos'] = robot.right_entity.get_qpos().copy()
        snap['right_qvel'] = robot.right_entity.get_qvel().copy()
    return snap


def _restore_state(TASK_ENV, snap):
    import sapien.physx as _physx

    robot = TASK_ENV.robot
    robot.left_entity.set_qpos(snap['left_qpos'])
    robot.left_entity.set_qvel(snap['left_qvel'])
    if robot.right_entity is not robot.left_entity:
        robot.right_entity.set_qpos(snap['right_qpos'])
        robot.right_entity.set_qvel(snap['right_qvel'])

    for entity in set([robot.left_entity, robot.right_entity]):
        qpos = entity.get_qpos()
        for i, joint in enumerate(entity.get_active_joints()):
            if i < len(qpos):
                joint.set_drive_target(qpos[i])
                joint.set_drive_velocity_target(0.0)

    actor_map = {a.per_scene_id: a for a in TASK_ENV.scene.get_all_actors()}
    for sid, pose in snap['actors']:
        actor = actor_map.get(sid)
        if actor is None:
            continue
        try:
            moved = False
            for comp in actor.get_components():
                if isinstance(comp, _physx.PhysxRigidDynamicComponent):
                    comp.set_entity_pose(pose)
                    comp.set_linear_velocity([0, 0, 0])
                    comp.set_angular_velocity([0, 0, 0])
                    moved = True
                    break
            if not moved:
                actor.set_pose(pose)
        except Exception as e:
            print(f"[restore_state] Warning: could not restore actor #{sid}: {e}")

    _robot_entities = {TASK_ENV.robot.left_entity, TASK_ENV.robot.right_entity}
    arts = [art for art in TASK_ENV.scene.get_all_articulations()
            if art not in _robot_entities]
    for i, qpos, qvel in snap.get('articulations', []):
        if i >= len(arts):
            continue
        try:
            arts[i].set_qpos(qpos)
            arts[i].set_qvel(qvel)
        except Exception as e:
            print(f"[restore_state] Warning: could not restore articulation [{i}]: {e}")

    TASK_ENV.take_action_cnt = snap['take_action_cnt']
    TASK_ENV.eval_success = False
    if hasattr(TASK_ENV, '_init_collision_metrics'):
        TASK_ENV._init_collision_metrics()
    if hasattr(TASK_ENV, '_snapshot_static_object_poses'):
        TASK_ENV._snapshot_static_object_poses()

    try:
        vals = TASK_ENV.robot.get_normal_real_gripper_val()
        TASK_ENV.robot.left_gripper_val  = vals[0]
        TASK_ENV.robot.right_gripper_val = vals[1]
    except Exception:
        pass

    TASK_ENV.scene.step()


# ── Shared video helpers ───────────────────────────────────────────────────

_VIDEO_CAMS = ('countertop_camera', 'demo_camera', 'head_camera',
               'right_camera', 'left_camera')
_VIDEO_FPS  = 10.0


def _pick_video_frame(obs_dict):
    for cam in _VIDEO_CAMS:
        if cam in obs_dict:
            return obs_dict[cam]['rgb'].copy()
    return None


def _save_video(video_frames, ep_ref, video_dir):
    """ep_ref: hdf5 path (video named after its stem) or a bare episode index."""
    if not video_frames or video_dir is None:
        return
    name = Path(ep_ref).stem if isinstance(ep_ref, (Path, str)) else f"episode{ep_ref}"
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    try:
        from envs.utils.images_to_video import images_to_video
        images_to_video(np.stack(video_frames),
                        str(Path(video_dir) / f"{name}.mp4"),
                        fps=_VIDEO_FPS)
    except Exception as e:
        print(f"[video] {name} failed: {e}")


# ── CuRobo collision-avoidance escape ─────────────────────────────────────

def _curobo_escape(TASK_ENV, n_steps, encode_obs_fn, buffer, model,
                   video_buf=None):
    robot = TASK_ENV.robot

    if not hasattr(robot, 'left_planner'):
        print("[curobo] No planner on robot — skipping branch.")
        return False

    if not hasattr(TASK_ENV, 'get_curobo_target'):
        print("[curobo] Task has no get_curobo_target() — skipping branch.")
        return False

    n_l = len(robot.left_arm_joints)
    n_r = len(robot.right_arm_joints)

    left_target, right_target = TASK_ENV.get_curobo_target()
    if left_target is None and right_target is None:
        print("[curobo] Task returned no subgoal targets — skipping branch.")
        return False

    _has_update_world = hasattr(TASK_ENV, 'update_world')
    _exclude_after    = _has_update_world and getattr(TASK_ENV, 'enable_collision_metrics', False)
    if _has_update_world:
        TASK_ENV.update_world()
        print("[curobo] World updated with full obstacles for collision-free planning.")

    left_pos = right_pos = None
    left_result = right_result = None
    if left_target is not None:
        left_result = robot.left_plan_path(left_target)
        if left_result.get('status') == 'Success':
            left_pos = left_result['position']
        else:
            print(f"[curobo] Left arm planning failed: {left_result.get('status')}")

    if right_target is not None:
        right_result = robot.right_plan_path(right_target)
        if right_result.get('status') == 'Success':
            right_pos = right_result['position']
        else:
            print(f"[curobo] Right arm planning failed: {right_result.get('status')}")

    if left_pos is None and right_pos is None:
        if _exclude_after:
            TASK_ENV.update_world(exclude_obstacles=True)
        return False

    if _exclude_after:
        TASK_ENV.update_world(exclude_obstacles=True)
        print("[curobo] World restored to exclude obstacles for collision metrics.")

    # PROXIMITY: initial obs includes proximity
    obs0 = TASK_ENV.get_obs()
    rgb0, state0 = encode_obs_fn(obs0)
    buffer.append(_obs_to_frame(obs0, state0, task_env=TASK_ENV))
    model.set_language(TASK_ENV.get_instruction())
    model.update_observation_window(rgb0, state0)

    plan_len = min(len(p) for p in (left_pos, right_pos) if p is not None)
    n_exec   = min(n_steps, plan_len)
    right_vel = right_result.get('velocity') if right_result else None
    left_vel  = left_result.get('velocity')  if left_result  else None

    obs_every = max(1, n_exec // 30)
    _collision_active = (getattr(TASK_ENV, 'enable_collision_metrics', False)
                         and hasattr(TASK_ENV, 'robot_link_names'))

    for i in range(n_exec):
        if TASK_ENV.eval_success:
            break

        if left_pos is not None:
            robot.set_arm_joints(left_pos[i],
                                 left_vel[i] if left_vel is not None else np.zeros(n_l),
                                 'left')
        if right_pos is not None:
            robot.set_arm_joints(right_pos[i],
                                 right_vel[i] if right_vel is not None else np.zeros(n_r),
                                 'right')

        if _collision_active:
            TASK_ENV._snapshot_static_object_poses()
        TASK_ENV.scene.step()
        TASK_ENV.take_action_cnt += 1
        if _collision_active:
            TASK_ENV.check_collisions()

        if i % obs_every == 0 or i == n_exec - 1:
            obs_n = TASK_ENV.get_obs()
            rgb_n, state_n = encode_obs_fn(obs_n)
            model.update_observation_window(rgb_n, state_n)
            # PROXIMITY: capture proximity at each recorded CuRobo step
            buffer.append(_obs_to_frame(obs_n, state_n, task_env=TASK_ENV))
            if video_buf is not None:
                frame = _pick_video_frame(obs_n.get('observation', {}))
                if frame is not None:
                    video_buf.append(frame)

    for entity in {robot.left_entity, robot.right_entity}:
        qpos = entity.get_qpos()
        entity.set_qvel(np.zeros_like(entity.get_qvel()))
        for j, joint in enumerate(entity.get_active_joints()):
            if j < len(qpos):
                joint.set_drive_target(qpos[j])
                joint.set_drive_velocity_target(0.0)
    TASK_ENV.scene.step()

    print(f"[curobo] Executed {n_exec}/{plan_len} steps of collision-free plan.")
    return True


# ── Simple rollout (FIXED to capture every frame) ─────────────────────────

def collect_rollouts(task_name, TASK_ENV, args, model, st_seed,
                     collect_num=100, save_dir=None, instruction_type="seen",
                     video_size=None, save_seed_fn=None):
    save_dir = Path(save_dir)
    data_dir = save_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    eval_func = eval_function_decorator(args["policy_name"], "eval")
    args["eval_mode"] = True
    clear_cache_freq = args["clear_cache_freq"]

    from policy.pi05.deploy_policy import encode_obs as _encode_obs

    now_seed = st_seed
    ep_idx = 0
    succ_count = 0
    collision_count = 0
    episode_log = []
    _ec_raw      = args.get('expert_check', os.environ.get('COLLECT_EXPERT_CHECK', ''))
    expert_check = _ec_raw is True or (isinstance(_ec_raw, str) and _ec_raw.lower() not in ('', '0', 'false', 'no', 'none'))
    _ss_raw       = args.get('collect_selective_save', os.environ.get('COLLECT_SELECTIVE_SAVE', ''))
    selective_save = _ss_raw is True or (isinstance(_ss_raw, str) and _ss_raw.lower() not in ('', '0', 'false', 'no', 'none'))

    print(f"\033[34mTask: {task_name}  |  Proximity collection  |  {collect_num} episodes"
          f"{'  |  expert check' if expert_check else ''}"
          f"{'  |  selective save' if selective_save else ''}\033[0m")

    while ep_idx < collect_num:
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                if episode_info is None:
                    episode_info = getattr(TASK_ENV, "info", {"info": {}})
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env(); now_seed += 1; continue
            except Exception:
                TASK_ENV.close_env(); now_seed += 1; continue

            if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
                now_seed += 1; continue

        TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=now_seed, is_test=True, **args)
        if not expert_check:
            episode_info = getattr(TASK_ENV, "info", {"info": {}})

        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, collect_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        model.reset_model()
        TASK_ENV._contrastive_buffer = []

        # Patch take_action: capture (obs_before_action, action) so each frame
        # pairs the policy's input with its command — no orphaned initial frame.
        _orig_take_action = TASK_ENV.take_action
        def _record_take_action(action):
            action = np.asarray(action, dtype=np.float32)
            if TASK_ENV._contrastive_buffer is not None:
                _obs = TASK_ENV.get_obs()           # obs the policy saw
                _, _state = _encode_obs(_obs)
                TASK_ENV._contrastive_buffer.append(_obs_to_frame(_obs, _state, action=action, task_env=TASK_ENV))
            return _orig_take_action(action)
        TASK_ENV.take_action = _record_take_action

        succ = False
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break

        TASK_ENV.take_action = _orig_take_action

        is_collision = False
        col_metrics = {}
        if hasattr(TASK_ENV, 'get_collision_metrics'):
            col_metrics = TASK_ENV.get_collision_metrics()
            is_collision = col_metrics.get("is_collision", False)
            if is_collision:
                collision_count += 1

        n_frames = len(TASK_ENV._contrastive_buffer)
        _should_save = (not selective_save) or succ
        file_idx = None
        if n_frames >= 2 and _should_save:
            file_idx = save_proximity_hdf5(
                TASK_ENV._contrastive_buffer, data_dir,
                ep_idx, instruction, succ, is_collision, collector="pi05",
            )
        # (simple mode has no separate video save)
        TASK_ENV._contrastive_buffer = None

        label_str = "\033[92mSuccess\033[0m" if succ else "\033[91mFail\033[0m"
        col_detail = _col_detail_str(col_metrics)
        if _should_save:
            _tag = Path(file_idx).stem if file_idx is not None else f"{ep_idx:04d}"
            print(f"  ep{_tag} seed={now_seed} {label_str}  frames={n_frames}{col_detail}")
        else:
            print(f"  seed={now_seed} {label_str}  frames={n_frames}  skipped (selective)")

        if succ:
            succ_count += 1

        if _should_save:
            episode_log.append(_make_log_entry(ep_idx, now_seed, n_frames, succ,
                                               is_collision, instruction, col_metrics))
            with open(save_dir / "collect_summary.json", "w") as _f:
                json.dump(episode_log, _f, indent=2)
            ep_idx += 1
        TASK_ENV.close_env(clear_cache=(ep_idx % clear_cache_freq == 0))
        now_seed += 1
        if save_seed_fn:
            save_seed_fn(now_seed)

        if ep_idx > 0:
            print(f"  → {ep_idx}/{collect_num} collected | "
                  f"SR \033[95m{succ_count/ep_idx*100:.1f}%\033[0m | "
                  f"CR \033[95m{collision_count/ep_idx*100:.1f}%\033[0m")

    with open(save_dir / "collect_summary.json", "w") as f:
        json.dump(episode_log, f, indent=2)

    return episode_log


# ── Branching rollout collection ───────────────────────────────────────────

def collect_rollouts_branching(task_name, TASK_ENV, args, model, st_seed,
                               collect_num=100, save_dir=None, instruction_type="seen",
                               video_size=None, save_seed_fn=None):
    def _p(key, env_var, default, cast=str):
        return cast(args.get(key, os.environ.get(env_var, default)))

    n_branches = _p('collect_branch_num', 'COLLECT_BRANCH_NUM', 0, int)
    _ec_raw      = args.get('expert_check', os.environ.get('COLLECT_EXPERT_CHECK', ''))
    expert_check = _ec_raw is True or (isinstance(_ec_raw, str) and _ec_raw.lower() not in ('', '0', 'false', 'no', 'none'))
    _ss_raw       = args.get('collect_selective_save', os.environ.get('COLLECT_SELECTIVE_SAVE', ''))
    selective_save = _ss_raw is True or (isinstance(_ss_raw, str) and _ss_raw.lower() not in ('', '0', 'false', 'no', 'none'))
    save_dir = Path(save_dir)
    data_dir = save_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    eval_func = eval_function_decorator(args["policy_name"], "eval")
    args["eval_mode"] = True
    clear_cache_freq = args["clear_cache_freq"]
    from policy.pi05.deploy_policy import encode_obs as _encode_obs

    now_seed        = st_seed
    ep_idx          = 0
    succ_count      = 0
    collision_count = 0
    episode_log     = []

    print(f"\033[34mTask: {task_name}  |  Proximity branching  |  {n_branches} branches per failure"
          f"{'  |  selective save' if selective_save else ''}\033[0m")

    _video_dir = save_dir / "videos" if video_size is not None else None

    def _run_curobo_branch(ep_num, seed):
        args["enable_collision_metrics"] = True
        try:
            TASK_ENV.setup_demo(now_ep_num=ep_num, seed=seed, is_test=True, **args)
            TASK_ENV.plan_success = True
            if hasattr(TASK_ENV, 'update_world'):
                TASK_ENV.update_world()
        except Exception:
            return False, 0, [], {}, []

        buf = []
        video_frames = []
        _orig_ta           = TASK_ENV.take_action
        _orig_tda          = TASK_ENV.take_dense_action
        _orig_take_picture = TASK_ENV._take_picture
        _orig_save_freq    = TASK_ENV.save_freq

        _VIDEO_STEP_FREQ = 25

        def _video_take_picture():
            obs = TASK_ENV.get_obs()
            frame = _pick_video_frame(obs.get('observation', {}))
            if frame is not None:
                video_frames.append(frame)

        def _rec_obs(action=None):
            obs = TASK_ENV.get_obs()
            _, st = _encode_obs(obs)
            buf.append(_obs_to_frame(obs, st, action=action, task_env=TASK_ENV))

        def _rec_ta(action):
            _rec_obs(action=action)
            return _orig_ta(action)

        def _rec_tda(control_seq, save_freq=-1):
            _rec_obs()
            return _orig_tda(control_seq, save_freq=save_freq)

        TASK_ENV.take_action       = _rec_ta
        TASK_ENV.take_dense_action = _rec_tda
        TASK_ENV._take_picture     = _video_take_picture
        TASK_ENV.save_freq         = _VIDEO_STEP_FREQ

        ok = False
        try:
            TASK_ENV.play_once()
            ok = TASK_ENV.plan_success and TASK_ENV.check_success()
        except Exception as e:
            print(f"  play_once() error: {e}")
        finally:
            TASK_ENV.take_action       = _orig_ta
            TASK_ENV.take_dense_action = _orig_tda
            TASK_ENV._take_picture     = _orig_take_picture
            TASK_ENV.save_freq         = _orig_save_freq

        col = {}
        if hasattr(TASK_ENV, 'get_collision_metrics'):
            col = TASK_ENV.get_collision_metrics()
        return ok, len(buf), buf, col, video_frames

    while ep_idx < collect_num:
        args["render_freq"] = 0

        if expert_check:
            print(f"#### Seed value {now_seed} ####")
            _result_check = _run_curobo_branch(ep_idx, now_seed)
            if not isinstance(_result_check, tuple) or len(_result_check) != 5:
                raise RuntimeError(f"[BUG] _run_curobo_branch returned unexpected type")
            ok_check, _, _, _, _ = _result_check
            TASK_ENV.close_env()
            if not ok_check:
                print(f"[curobo-check] seed={now_seed} skipped — play_once() failed.")
                now_seed += 1
                continue
            print(f"[curobo-check] seed={now_seed} OK.")

        episode_info = getattr(TASK_ENV, "info", {"info": {}})
        results = generate_episode_descriptions(task_name, [episode_info["info"]], 1)
        instruction = np.random.choice(results[0][instruction_type])

        args["enable_collision_metrics"] = True
        TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=now_seed, is_test=True, **args)
        TASK_ENV.eval_video_path = None
        TASK_ENV.set_instruction(instruction=instruction)

        model.reset_model()
        TASK_ENV._contrastive_buffer = []

        _prev_col_total       = 0
        succ                  = False
        _orig_ta              = TASK_ENV.take_action
        _step_cnt             = [0]
        _primary_video_frames = []

        def _primary_take_action(action):
            nonlocal _prev_col_total
            if TASK_ENV._contrastive_buffer is not None:
                _obs = TASK_ENV.get_obs()
                _, _state = _encode_obs(_obs)
                TASK_ENV._contrastive_buffer.append(
                    _obs_to_frame(_obs, _state, action=action, task_env=TASK_ENV))
                frame = _pick_video_frame(_obs.get('observation', {}))
                if frame is not None:
                    _primary_video_frames.append(frame)
            result = _orig_ta(action)
            _step_cnt[0] += 1
            if _first_collision_step is None and hasattr(TASK_ENV, 'get_collision_metrics'):
                total = TASK_ENV.get_collision_metrics().get("total_collision_count", 0)
                if total > _prev_col_total:
                    _collision_info[0] = _step_cnt[0]
                    print(f"[collision] step={_step_cnt[0]}  metrics={TASK_ENV.get_collision_metrics()}")
                _prev_col_total = total
            return result

        _collision_info = [None]
        _first_collision_step = None
        TASK_ENV.take_action = _primary_take_action

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break

        TASK_ENV.take_action = _orig_ta

        is_collision = False
        col_metrics  = {}
        if hasattr(TASK_ENV, 'get_collision_metrics'):
            col_metrics  = TASK_ENV.get_collision_metrics()
            is_collision = col_metrics.get("is_collision", False)
            if is_collision:
                collision_count += 1

        n_frames = len(TASK_ENV._contrastive_buffer)
        _primary_had_collision = _collision_info[0] is not None
        _save_primary = (not selective_save) or (not succ) or _primary_had_collision

        file_idx = None
        if n_frames >= 2 and _save_primary:
            file_idx = save_proximity_hdf5(TASK_ENV._contrastive_buffer, data_dir,
                                           ep_idx, instruction, succ, is_collision, collector="pi05")
        if file_idx is not None:
            _save_video(_primary_video_frames, file_idx, _video_dir)
        TASK_ENV._contrastive_buffer = None

        _res = "\033[92mSuccess\033[0m" if succ else "\033[91mFail\033[0m"
        _col_step_str = f"  collision_step={_collision_info[0]}" if _collision_info[0] is not None else ""
        if _save_primary:
            _tag = Path(file_idx).stem if file_idx is not None else f"{ep_idx:04d}"
            print(f"  ep{_tag} seed={now_seed} [primary]  {_res}  "
                  f"frames={n_frames}{_col_step_str}{_col_detail_str(col_metrics)}")
            episode_log.append(_make_log_entry(ep_idx, now_seed, n_frames, succ,
                                               is_collision, instruction, col_metrics))
            with open(save_dir / "collect_summary.json", "w") as _f:
                json.dump(episode_log, _f, indent=2)
            if succ: succ_count += 1
            ep_idx += 1
        else:
            print(f"  seed={now_seed} [primary]  {_res}  frames={n_frames}  skipped (selective)")

        if (not succ or _primary_had_collision) and n_branches > 0:
            print(f"Number of Branches: {n_branches}  "
                  f"(primary {'collision' if _primary_had_collision else 'failure'})")
            for b in range(n_branches):
                if ep_idx >= collect_num:
                    break

                print(f"[branch {b+1:02d}]")
                TASK_ENV.close_env()

                _result_branch = _run_curobo_branch(ep_idx, now_seed)
                if not isinstance(_result_branch, tuple) or len(_result_branch) != 5:
                    raise RuntimeError(f"[BUG] _run_curobo_branch returned unexpected type")
                succ_b, n_frames_b, buf_b, col_b, vframes_b = _result_branch
                is_col_b = col_b.get("is_collision", False)
                if is_col_b: collision_count += 1

                _res_b = "\033[92mSuccess\033[0m" if succ_b else "\033[91mFail\033[0m"
                _save_branch = (not selective_save) or succ_b
                if n_frames_b >= 2 and _save_branch:
                    file_idx_b = save_proximity_hdf5(buf_b, data_dir,
                                                     ep_idx, instruction, succ_b, is_col_b, collector="curobo")
                    _save_video(vframes_b, file_idx_b, _video_dir)
                    print(f"  ep{ep_idx:04d} seed={now_seed} "
                          f"[branch {b+1:02d}/{n_branches}]  {_res_b}  "
                          f"frames={n_frames_b}{_col_detail_str(col_b)}")
                    episode_log.append(_make_log_entry(ep_idx, now_seed, n_frames_b,
                                                       succ_b, is_col_b, instruction, col_b))
                    with open(save_dir / "collect_summary.json", "w") as _f:
                        json.dump(episode_log, _f, indent=2)
                    if succ_b: succ_count += 1
                    ep_idx += 1
                elif n_frames_b < 2:
                    print(f"  [branch {b+1:02d}/{n_branches}] discarded — {n_frames_b} frames")
                else:
                    print(f"  [branch {b+1:02d}/{n_branches}] {_res_b}  skipped (selective)")

        TASK_ENV.close_env(clear_cache=(ep_idx % clear_cache_freq == 0))
        now_seed += 1
        if save_seed_fn:
            save_seed_fn(now_seed)

        if ep_idx > 0:
            print(f"  → {ep_idx}/{collect_num} collected | "
                  f"SR \033[95m{succ_count/ep_idx*100:.1f}%\033[0m | "
                  f"CR \033[95m{collision_count/ep_idx*100:.1f}%\033[0m")

    with open(save_dir / "collect_summary.json", "w") as f:
        json.dump(episode_log, f, indent=2)

    return episode_log


# ── Stitched rollout collection ────────────────────────────────────────────

def collect_rollouts_stitched(task_name, TASK_ENV, args, model, st_seed,
                               collect_num=100, save_dir=None, instruction_type="seen",
                               video_size=None, save_seed_fn=None):
    def _p(key, env_var, default, cast=str):
        return cast(args.get(key, os.environ.get(env_var, default)))

    lookback     = _p('collect_stitch_lookback',     'COLLECT_STITCH_LOOKBACK',     5,   int)
    curobo_steps = _p('collect_stitch_curobo_steps', 'COLLECT_STITCH_CUROBO_STEPS', 100, int)
    _ss_raw       = args.get('collect_selective_save', os.environ.get('COLLECT_SELECTIVE_SAVE', ''))
    selective_save = _ss_raw is True or (isinstance(_ss_raw, str) and _ss_raw.lower() not in ('', '0', 'false', 'no', 'none'))

    save_dir = Path(save_dir)
    data_dir = save_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _video_dir = save_dir / "videos" if video_size is not None else None

    eval_func        = eval_function_decorator(args["policy_name"], "eval")
    args["eval_mode"] = True
    clear_cache_freq = args["clear_cache_freq"]
    from policy.pi05.deploy_policy import encode_obs as _encode_obs

    now_seed        = st_seed
    ep_idx          = 0
    succ_count      = 0
    collision_count = 0
    episode_log     = []

    print(f"\033[34mTask: {task_name}  |  Proximity stitched (pi05→CuRobo→pi05)  |  lookback={lookback}"
          f"{'  |  selective save' if selective_save else ''}\033[0m")

    while ep_idx < collect_num:
        args["render_freq"] = 0

        args["enable_collision_metrics"] = True
        TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=now_seed, is_test=True, **args)
        TASK_ENV.eval_video_path = None

        episode_info = getattr(TASK_ENV, "info", {"info": {}})
        results      = generate_episode_descriptions(task_name, [episode_info["info"]], 1)
        instruction  = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        model.reset_model()

        _col_step   = [None]
        _prev_total = [0]
        _step_cnt   = [0]
        _orig_ta    = TASK_ENV.take_action

        def _detect_ta(action):
            result = _orig_ta(action)
            _step_cnt[0] += 1
            if _col_step[0] is None and hasattr(TASK_ENV, 'get_collision_metrics'):
                total = TASK_ENV.get_collision_metrics().get("total_collision_count", 0)
                if total > _prev_total[0]:
                    _col_step[0] = _step_cnt[0]
                    print(f"[stitch-detect] collision at step={_step_cnt[0]}")
                _prev_total[0] = total
            return result

        TASK_ENV.take_action = _detect_ta
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            eval_func(TASK_ENV, model, TASK_ENV.get_obs())
            if TASK_ENV.eval_success:
                break
        TASK_ENV.take_action = _orig_ta
        TASK_ENV.close_env()

        collision_step = _col_step[0]
        if collision_step is None:
            now_seed += 1
            continue

        stitch_at = max(1, collision_step - lookback)
        print(f"  seed={now_seed}  collision_step={collision_step}  stitch_at={stitch_at}")

        args["enable_collision_metrics"] = True
        TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=now_seed, is_test=True, **args)
        TASK_ENV.eval_video_path = None
        if hasattr(TASK_ENV, 'update_world'):
            TASK_ENV.update_world()
        TASK_ENV.set_instruction(instruction=instruction)

        model.reset_model()
        TASK_ENV._contrastive_buffer = []
        video_frames = []
        _orig_ta2 = TASK_ENV.take_action

        def _record_ta(action):
            # Capture obs BEFORE action (policy's input paired with its command)
            if TASK_ENV._contrastive_buffer is not None:
                obs = TASK_ENV.get_obs()
                _, state = _encode_obs(obs)
                TASK_ENV._contrastive_buffer.append(_obs_to_frame(obs, state, action=action, task_env=TASK_ENV))
                frame = _pick_video_frame(obs.get('observation', {}))
                if frame is not None:
                    video_frames.append(frame)
            return _orig_ta2(action)

        TASK_ENV.take_action = _record_ta
        while TASK_ENV.take_action_cnt < stitch_at and TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            eval_func(TASK_ENV, model, TASK_ENV.get_obs())
            if TASK_ENV.eval_success:
                break
        TASK_ENV.take_action = _orig_ta2

        succ = TASK_ENV.eval_success

        if not succ:
            ok_curobo = _curobo_escape(TASK_ENV, curobo_steps, _encode_obs,
                                       TASK_ENV._contrastive_buffer, model,
                                       video_buf=video_frames)
            if not ok_curobo:
                print(f"  [stitch] CuRobo escape failed — skipping seed={now_seed}")
                TASK_ENV._contrastive_buffer = None
                TASK_ENV.close_env()
                now_seed += 1
                continue

            TASK_ENV.take_action = _record_ta
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                eval_func(TASK_ENV, model, TASK_ENV.get_obs())
                if TASK_ENV.eval_success:
                    succ = True
                    break
            TASK_ENV.take_action = _orig_ta2

        is_collision = False
        col_metrics  = {}
        if hasattr(TASK_ENV, 'get_collision_metrics'):
            col_metrics  = TASK_ENV.get_collision_metrics()
            is_collision = col_metrics.get("is_collision", False)
            if is_collision:
                collision_count += 1

        n_frames = len(TASK_ENV._contrastive_buffer)
        _should_save = (not selective_save) or succ
        file_idx = None
        if n_frames >= 2 and _should_save:
            file_idx = save_proximity_hdf5(TASK_ENV._contrastive_buffer, data_dir,
                                           ep_idx, instruction, succ, is_collision, collector="stitched")
            _save_video(video_frames, file_idx, _video_dir)

        TASK_ENV._contrastive_buffer = None

        _res = "\033[92mSuccess\033[0m" if succ else "\033[91mFail\033[0m"
        if _should_save:
            _tag = Path(file_idx).stem if file_idx is not None else f"{ep_idx:04d}"
            print(f"  ep{_tag} seed={now_seed} [stitched]  {_res}  "
                  f"frames={n_frames}{_col_detail_str(col_metrics)}")
            episode_log.append(_make_log_entry(ep_idx, now_seed, n_frames, succ,
                                               is_collision, instruction, col_metrics))
            with open(save_dir / "collect_summary.json", "w") as _f:
                json.dump(episode_log, _f, indent=2)
            if succ:
                succ_count += 1
            ep_idx += 1
        else:
            print(f"  seed={now_seed} [stitched]  {_res}  frames={n_frames}  skipped (selective)")

        TASK_ENV.close_env(clear_cache=(ep_idx % clear_cache_freq == 0))
        now_seed += 1
        if save_seed_fn:
            save_seed_fn(now_seed)

        if ep_idx > 0:
            sr = succ_count / ep_idx * 100
            cr = collision_count / ep_idx * 100
            print(f"  → {ep_idx}/{collect_num} collected | "
                  f"SR \033[95m{sr:.1f}%\033[0m | CR \033[95m{cr:.1f}%\033[0m")

    with open(save_dir / "collect_summary.json", "w") as f:
        json.dump(episode_log, f, indent=2)

    return episode_log


# ── Shared helpers ─────────────────────────────────────────────────────────

def _append_index(save_dir, entry):
    """One line per saved episode — the dataset's primary index (see
    data_generation_documentation.md)."""
    with open(Path(save_dir) / "index.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def _run_curobo_leg(TASK_ENV, args, now_seed, ep_num, exclude_clutter,
                    instruction_type, encode_obs, instruction=None,
                    metric_paths=None, post_setup=None):
    """One CuRobo planner episode on now_seed.

    exclude_clutter=False → clutter in the planner collision world
                            (collision-aware expert, dataset source A);
    exclude_clutter=True  → planner ignores clutter
                            (collision-unaware, dataset source C).
    Returns dict(buf, vframes, ok, col, is_col, instruction) or None when
    setup fails.  Caller saves the HDF5 and closes the env.
    """
    tag = "curobo-unaware" if exclude_clutter else "curobo"
    args["enable_collision_metrics"] = True
    try:
        TASK_ENV.setup_demo(now_ep_num=ep_num, seed=now_seed, is_test=True, **args)
        TASK_ENV.eval_video_path = None
        TASK_ENV.plan_success = True
        if hasattr(TASK_ENV, 'update_world'):
            if exclude_clutter:
                TASK_ENV.update_world(exclude_obstacles=True)
            else:
                TASK_ENV.update_world()
    except Exception as e:
        print(f"  seed={now_seed} [{tag}] setup failed: {e}")
        TASK_ENV.close_env()
        return None

    if post_setup is not None:
        post_setup(TASK_ENV)

    if instruction is None:
        episode_info = getattr(TASK_ENV, "info", {"info": {}})
        results      = generate_episode_descriptions(
            args["task_name"], [episode_info["info"]], 1)
        instruction  = np.random.choice(
            results[0].get(instruction_type, results[0].get("seen", [""])))

    buf, vframes = [], []
    _orig_ta, _orig_tda = TASK_ENV.take_action, TASK_ENV.take_dense_action
    _orig_pic, _orig_sf = TASK_ENV._take_picture, TASK_ENV.save_freq
    _last_col = [0]

    def _capture():
        obs = TASK_ENV.get_obs()
        _, st = encode_obs(obs)
        buf.append(_obs_to_frame(obs, st, action=None, task_env=TASK_ENV))
        if hasattr(TASK_ENV, 'get_collision_metrics'):
            cur = TASK_ENV.get_collision_metrics().get("total_collision_count", 0)
            buf[-1]['collision_step'] = (cur > _last_col[0])
            _last_col[0] = cur
        else:
            buf[-1]['collision_step'] = False
        fr = _pick_video_frame(obs.get('observation', {}))
        if fr is not None: vframes.append(fr)

    TASK_ENV.take_action       = lambda action, _o=_orig_ta: _o(action)
    TASK_ENV.take_dense_action = lambda cs, *_, _o=_orig_tda: _o(cs, save_freq=_CUROBO_SAVE_FREQ)
    TASK_ENV._take_picture     = _capture
    TASK_ENV.save_freq         = _CUROBO_SAVE_FREQ

    if metric_paths is not None and hasattr(TASK_ENV, 'start_metric_streams'):
        TASK_ENV.start_metric_streams(*[str(p) for p in metric_paths])

    ok = False
    try:
        TASK_ENV.play_once()
        ok = TASK_ENV.plan_success and TASK_ENV.check_success()
    except Exception as e:
        print(f"  [{tag}] play_once() error: {e}")
    finally:
        if hasattr(TASK_ENV, 'stop_metric_streams'):
            TASK_ENV.stop_metric_streams()
        TASK_ENV.take_action       = _orig_ta
        TASK_ENV.take_dense_action = _orig_tda
        TASK_ENV._take_picture     = _orig_pic
        TASK_ENV.save_freq         = _orig_sf

    col, is_col = {}, False
    if hasattr(TASK_ENV, 'get_collision_metrics'):
        col    = TASK_ENV.get_collision_metrics()
        is_col = col.get("is_collision", False)
    return {"buf": buf, "vframes": vframes, "ok": ok, "col": col,
            "is_col": is_col, "instruction": instruction}


def collect_rollouts_paired(task_name, TASK_ENV, args, model, st_seed,
                            collect_num=10, save_dir=None, instruction_type="seen",
                            video_size=None, save_seed_fn=None):
    """Triplet collection on each seed (see data_generation_documentation.md):

        A. CuRobo, clutter IN the collision world  (planner_collision_aware)
        B. pi05 policy rollout                     (pi05_base)
        C. CuRobo, clutter NOT in the world        (planner_collision_unaware)

    A runs first; seeds where it produces no trajectory are skipped entirely,
    guaranteeing every saved triplet has a valid expert rollout.  B and C are
    always saved regardless of outcome.  Per seed this also exports the scene
    record (scene/seed_N/) and per-episode contact/collision jsonl streams
    (metrics/), plus one fk_basis.npz per dataset.
    collect_num = number of seeds → up to collect_num*3 HDF5 files.

    Layout: one folder per source (episode_seed<N>.hdf5 + .info.json each):
        <save_dir>/curobo_collision_free/   planner_collision_aware
        <save_dir>/pi05_rollout/            pi05_base
        <save_dir>/curobo_collision/        planner_collision_unaware
    """
    save_dir = Path(save_dir)
    _SRC_DIRS = {
        "planner_collision_aware":   save_dir / "curobo_collision_free",
        "pi05_base":                 save_dir / "pi05_rollout",
        "planner_collision_unaware": save_dir / "curobo_collision",
    }
    for _d in _SRC_DIRS.values():
        _d.mkdir(parents=True, exist_ok=True)
    _video_dir = save_dir / "videos" if video_size is not None else None

    eval_func        = eval_function_decorator(args["policy_name"], "eval")
    args["eval_mode"] = True
    clear_cache_freq = args["clear_cache_freq"]
    from policy.pi05.deploy_policy import encode_obs as _encode_obs

    now_seed    = st_seed
    seed_idx    = 0
    episode_log = []
    pi05_succ = pi05_coll = curobo_succ = curobo_coll = 0
    # STRICT TRIPLETS (default): legs B & C run only on seeds where the
    # collision-aware expert (leg A) succeeded — all three sources always share
    # the same seed/scene.  Opt-in COLLECT_ON_EXPERT_FAIL=1 additionally
    # collects pi05 + collision-unaware legs on expert-blocked seeds (the
    # contact-rich scenes), flagged and not counted toward collect_num.
    _on_expert_fail = os.environ.get("COLLECT_ON_EXPERT_FAIL", "0").lower() \
        in ("1", "true", "yes")
    attempts, max_attempts = 0, max(collect_num * 20, 20)

    print(f"\033[34mTask: {task_name}  |  Paired (CuRobo first → pi05)  |  {collect_num} seeds\033[0m")

    while seed_idx < collect_num:
        attempts += 1
        if attempts > max_attempts:
            print(f"\033[31m[paired] giving up after {max_attempts} seed attempts "
                  f"({seed_idx}/{collect_num} valid triplets)\033[0m")
            break
        args["render_freq"] = 0
        scene_dir   = save_dir / "scene" / f"seed_{now_seed}"
        fk_path     = save_dir / "fk_basis.npz"
        _scene_rel  = f"scene/seed_{now_seed}"

        def _export_scene_once(env, _sd=scene_dir, _fk=fk_path, _seed=now_seed):
            try:
                from differentiable_proximity import export_scene, serialize_fk_and_spheres
                _sd.mkdir(parents=True, exist_ok=True)
                h = export_scene(env, str(_sd))
                if not _fk.exists():
                    serialize_fk_and_spheres(env, str(_fk))
                print(f"  seed={_seed} scene exported ({h[:10]})")
            except Exception as e:
                print(f"  seed={_seed} scene export FAILED: {e}")

        def _mpaths(source, _seed=now_seed):
            md = _SRC_DIRS[source] / "metrics"
            md.mkdir(parents=True, exist_ok=True)
            return (md / f"seed{_seed}_contacts.jsonl",
                    md / f"seed{_seed}_collisions.jsonl")

        def _discard_seed_artifacts(_sd=scene_dir, _seed=now_seed):
            """Skipped seeds leave no trace: scene export + metric streams are
            written at setup, before the planner can fail — remove them."""
            import shutil
            if _sd.exists():
                shutil.rmtree(_sd, ignore_errors=True)
            for d in _SRC_DIRS.values():
                for p in (d / "metrics").glob(f"seed{_seed}_*"):
                    try:
                        p.unlink()
                    except OSError:
                        pass

        def _index_entry(file_path, source, ep, succ, is_col, n_frames,
                         col_metrics=None):
            entry = {
                "file": str(Path(file_path).relative_to(save_dir)),
                "seed": int(now_seed),
                "source": source, "episode": int(ep),
                "success": bool(succ), "collision": bool(is_col),
                "n_frames": int(n_frames), "scene_dir": _scene_rel,
                "contacts": str((_SRC_DIRS[source] / "metrics" /
                                 f"seed{now_seed}_contacts.jsonl").relative_to(save_dir)),
                "collisions": str((_SRC_DIRS[source] / "metrics" /
                                   f"seed{now_seed}_collisions.jsonl").relative_to(save_dir)),
                "instruction": str(instruction),
            }
            _append_index(save_dir, entry)
            # Per-episode info json, written IMMEDIATELY after the episode so a
            # killed run loses nothing (collect_summary.json only flushes per seed).
            info = dict(entry)
            if col_metrics:
                info["collision_metrics"] = col_metrics
            with open(Path(file_path).with_suffix(".info.json"), "w") as fo:
                json.dump(info, fo, indent=1, default=str)

        # ── Leg A: collision-AWARE CuRobo (skip seed if no trajectory) ───────
        legA = _run_curobo_leg(TASK_ENV, args, now_seed, seed_idx * 3,
                               exclude_clutter=False,
                               instruction_type=instruction_type,
                               encode_obs=_encode_obs,
                               metric_paths=_mpaths("planner_collision_aware"),
                               post_setup=_export_scene_once)
        if legA is None:
            _discard_seed_artifacts()
            now_seed += 1
            if save_seed_fn: save_seed_fn(now_seed)
            continue
        buf_c, ok_c      = legA["buf"], legA["ok"]
        instruction      = legA["instruction"]
        col_c, is_col_c  = legA["col"], legA["is_col"]
        expert_ok = len(buf_c) >= 2 and ok_c

        if expert_ok:
            col_frames_c  = [i for i, fr in enumerate(buf_c) if fr.get('collision_step', False)]
            n_col_steps_c = len(col_frames_c)
            file_c = save_proximity_hdf5(buf_c, _SRC_DIRS["planner_collision_aware"],
                                         seed_idx * 3, instruction, ok_c, is_col_c,
                                         collector="curobo",
                                         extra_attrs={"seed": int(now_seed),
                                                      "source": "planner_collision_aware",
                                                      "scene_dir": _scene_rel},
                                         filename=f"episode_seed{now_seed}.hdf5")
            if _video_dir and legA["vframes"]:
                _save_video(legA["vframes"], file_c, _SRC_DIRS["planner_collision_aware"] / "videos")
            TASK_ENV.close_env(clear_cache=False)
            _index_entry(file_c, "planner_collision_aware", seed_idx * 3, ok_c, is_col_c,
                         len(buf_c), col_metrics=col_c)

            _r1 = "\033[92mSuccess\033[0m" if ok_c else "\033[91mFail\033[0m"
            _col_ts_c = f"  col_timesteps={col_frames_c}" if col_frames_c else ""
            print(f"  seed={now_seed} [curobo] {_r1}  frames={len(buf_c)}  col_steps={n_col_steps_c}{_col_detail_str(col_c)}{_col_ts_c}")
            if ok_c:    curobo_succ += 1
            if is_col_c: curobo_coll += 1
        else:
            reason = "no trajectory" if len(buf_c) < 2 else f"Fail  frames={len(buf_c)}"
            TASK_ENV.close_env(clear_cache=False)
            if not _on_expert_fail:
                print(f"  seed={now_seed} [curobo] {reason} — skipping seed")
                _discard_seed_artifacts()
                now_seed += 1
                if save_seed_fn: save_seed_fn(now_seed)
                continue
            # Expert-blocked scene: exactly where the collision-unaware planner
            # WILL plow through clutter — collect legs B & C anyway (flagged).
            print(f"  seed={now_seed} [curobo] {reason} — expert-blocked scene: "
                  f"collecting pi05 + collision-unaware legs (contact-rich)")
            if len(buf_c) >= 2:   # failed expert attempt is still data — save flagged
                file_c = save_proximity_hdf5(buf_c, _SRC_DIRS["planner_collision_aware"],
                                             seed_idx * 3, instruction, False, is_col_c,
                                             collector="curobo",
                                             extra_attrs={"seed": int(now_seed),
                                                          "source": "planner_collision_aware",
                                                          "scene_dir": _scene_rel,
                                                          "expert_blocked": True},
                                             filename=f"episode_seed{now_seed}.hdf5")
                _index_entry(file_c, "planner_collision_aware", seed_idx * 3, False,
                             is_col_c, len(buf_c), col_metrics=col_c)

        # ── Leg B: pi05 rollout (same seed, always save) ──────────────────────
        try:
            TASK_ENV.setup_demo(now_ep_num=seed_idx * 3 + 1, seed=now_seed, is_test=True, **args)
            TASK_ENV.eval_video_path = None   # disable native ffmpeg recording
        except Exception as e:
            print(f"  seed={now_seed} [pi05] setup failed: {e} — saving curobo only")
            TASK_ENV.close_env()
            episode_log.append(_make_log_entry(seed_idx * 3, now_seed, len(buf_c),
                                               ok_c, is_col_c, instruction, col_c))
            with open(save_dir / "collect_summary.json", "w") as _f:
                json.dump(episode_log, _f, indent=2)
            now_seed += 1; seed_idx += 1
            if save_seed_fn: save_seed_fn(now_seed)
            continue

        TASK_ENV.set_instruction(instruction=instruction)
        model.reset_model()
        TASK_ENV._contrastive_buffer = []
        if hasattr(TASK_ENV, 'start_metric_streams'):
            TASK_ENV.start_metric_streams(*[str(p) for p in _mpaths("pi05_base")])

        _orig_ta = TASK_ENV.take_action
        succ_pi05 = False
        vframes_p = []
        _last_col_count = [0]  # cumulative collision count from previous action step

        def _rec_pi05(action, _orig=_orig_ta):
            action = np.asarray(action, dtype=np.float32)
            if TASK_ENV._contrastive_buffer is not None:
                _obs = TASK_ENV.get_obs()
                _, _state = _encode_obs(_obs)
                TASK_ENV._contrastive_buffer.append(
                    _obs_to_frame(_obs, _state, action=action, task_env=TASK_ENV))
                fr = _pick_video_frame(_obs.get('observation', {}))
                if fr is not None: vframes_p.append(fr)
            result = _orig(action)
            # Tag the frame with whether any collision occurred during this action step.
            # We compare cumulative total_collision_count before vs after take_action —
            # this catches collisions in any sub-step, not just the last one (filtered_contacts_for_log
            # is reset each sub-step so only the last sub-step's contacts would be visible there).
            if TASK_ENV._contrastive_buffer and hasattr(TASK_ENV, 'get_collision_metrics'):
                cur_count = TASK_ENV.get_collision_metrics().get("total_collision_count", 0)
                TASK_ENV._contrastive_buffer[-1]['collision_step'] = (cur_count > _last_col_count[0])
                _last_col_count[0] = cur_count
            return result

        TASK_ENV.take_action = _rec_pi05
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            eval_func(TASK_ENV, model, TASK_ENV.get_obs())
            if TASK_ENV.eval_success:
                succ_pi05 = True; break
        TASK_ENV.take_action = _orig_ta
        if hasattr(TASK_ENV, 'stop_metric_streams'):
            TASK_ENV.stop_metric_streams()

        col_pi05 = {}; is_col_pi05 = False
        if hasattr(TASK_ENV, 'get_collision_metrics'):
            col_pi05    = TASK_ENV.get_collision_metrics()
            is_col_pi05 = col_pi05.get("is_collision", False)

        n_pi05 = len(TASK_ENV._contrastive_buffer)
        col_frames_pi05  = [i for i, fr in enumerate(TASK_ENV._contrastive_buffer) if fr.get('collision_step', False)]
        n_col_steps_pi05 = len(col_frames_pi05)
        file_p = None
        if n_pi05 >= 2:
            file_p = save_proximity_hdf5(
                TASK_ENV._contrastive_buffer, _SRC_DIRS["pi05_base"],
                seed_idx * 3 + 1, instruction, succ_pi05, is_col_pi05, collector="pi05",
                extra_attrs={"seed": int(now_seed), "source": "pi05_base",
                             "scene_dir": _scene_rel},
                filename=f"episode_seed{now_seed}.hdf5")
        if _video_dir and vframes_p and file_p is not None:
            _save_video(vframes_p, file_p, _SRC_DIRS["pi05_base"] / "videos")
        TASK_ENV._contrastive_buffer = None
        TASK_ENV.close_env(clear_cache=(seed_idx % clear_cache_freq == 0))
        if file_p is not None:
            _index_entry(file_p, "pi05_base", seed_idx * 3 + 1, succ_pi05, is_col_pi05,
                         n_pi05, col_metrics=col_pi05)

        _r0 = "\033[92mSuccess\033[0m" if succ_pi05 else "\033[91mFail\033[0m"
        _col_ts_p = f"  col_timesteps={col_frames_pi05}" if col_frames_pi05 else ""
        print(f"  seed={now_seed} [pi05]   {_r0}  frames={n_pi05}  col_steps={n_col_steps_pi05}{_col_detail_str(col_pi05)}{_col_ts_p}")
        if succ_pi05:    pi05_succ += 1
        if is_col_pi05:  pi05_coll += 1

        # ── Leg C: collision-UNAWARE CuRobo (same seed, planner ignores clutter;
        #    contact-rich negatives — always saved, never gates the seed) ──────
        legC = _run_curobo_leg(TASK_ENV, args, now_seed, seed_idx * 3 + 2,
                               exclude_clutter=True,
                               instruction_type=instruction_type,
                               encode_obs=_encode_obs,
                               instruction=instruction,
                               metric_paths=_mpaths("planner_collision_unaware"))
        ok_u = is_col_u = False; n_u = 0; col_u = {}
        if legC is not None:
            buf_u, ok_u = legC["buf"], legC["ok"]
            col_u, is_col_u = legC["col"], legC["is_col"]
            n_u = len(buf_u)
            if n_u >= 2:
                file_u = save_proximity_hdf5(buf_u, _SRC_DIRS["planner_collision_unaware"],
                                             seed_idx * 3 + 2, instruction, ok_u, is_col_u,
                                             collector="curobo_unaware",
                                             extra_attrs={"seed": int(now_seed),
                                                          "source": "planner_collision_unaware",
                                                          "scene_dir": _scene_rel},
                                             filename=f"episode_seed{now_seed}.hdf5")
                if _video_dir and legC["vframes"]:
                    _save_video(legC["vframes"], file_u, _SRC_DIRS["planner_collision_unaware"] / "videos")
                _index_entry(file_u, "planner_collision_unaware", seed_idx * 3 + 2,
                             ok_u, is_col_u, n_u, col_metrics=col_u)
            TASK_ENV.close_env(clear_cache=False)
            _r2 = "\033[92mSuccess\033[0m" if ok_u else "\033[91mFail\033[0m"
            print(f"  seed={now_seed} [curobo-unaware] {_r2}  frames={n_u}{_col_detail_str(col_u)}")

        episode_log.append(_make_log_entry(seed_idx * 3,     now_seed, len(buf_c),  ok_c,      is_col_c,    instruction, col_c))
        episode_log.append(_make_log_entry(seed_idx * 3 + 1, now_seed, n_pi05,      succ_pi05, is_col_pi05, instruction, col_pi05))
        episode_log.append(_make_log_entry(seed_idx * 3 + 2, now_seed, n_u,         ok_u,      is_col_u,    instruction, col_u))
        with open(save_dir / "collect_summary.json", "w") as _f:
            json.dump(episode_log, _f, indent=2)

        now_seed += 1
        if expert_ok:
            seed_idx += 1
        if save_seed_fn: save_seed_fn(now_seed)
        if seed_idx > 0:
            print(f"  → {seed_idx}/{collect_num} seeds | "
                  f"curobo SR \033[95m{curobo_succ/seed_idx*100:.1f}%\033[0m  "
                  f"pi05 SR \033[95m{pi05_succ/seed_idx*100:.1f}%\033[0m")

    with open(save_dir / "collect_summary.json", "w") as f:
        json.dump(episode_log, f, indent=2)
    return episode_log


def _col_detail_str(col_metrics):
    if not col_metrics.get("is_collision"):
        return ""
    parts = []
    if col_metrics.get("robot_to_furniture_names"):
        parts.append("furn=" + ",".join(col_metrics["robot_to_furniture_names"]))
    if col_metrics.get("robot_to_static_object_names"):
        parts.append("obj=" + ",".join(col_metrics["robot_to_static_object_names"]))
    if col_metrics.get("target_to_static_object_names"):
        parts.append("tgt=" + ",".join(col_metrics["target_to_static_object_names"]))
    return f"  \033[91mCollision: {' | '.join(parts)}\033[0m"


def _make_log_entry(ep_idx, seed, n_frames, succ, is_collision, instruction, col_metrics):
    return {
        "episode":     int(ep_idx),
        "seed":        int(seed),
        "n_frames":    int(n_frames),
        "success":     bool(succ),
        "label":       1 if succ else 0,
        "collision":   bool(is_collision),
        "instruction": instruction,
        **{k: (int(col_metrics[k]) if k in col_metrics and not isinstance(col_metrics[k], list)
               else list(col_metrics.get(k, v)))
           for k, v in {
               "robot_to_furniture": 0, "robot_to_static_object": 0,
               "target_to_static_object": 0,
               "robot_to_furniture_names": [], "robot_to_static_object_names": [],
               "target_to_static_object_names": [],
           }.items()},
    }


# ── Entry point ────────────────────────────────────────────────────────────

def main(usr_args):
    task_name    = usr_args["task_name"]
    task_config  = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name  = usr_args["policy_name"]
    instruction_type = usr_args.get("instruction_type", "seen")
    port         = usr_args["port"]
    collect_num  = int(os.environ.get("COLLECT_NUM", usr_args.get("collect_num", 100)))

    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        cfg_path = f"{os.getenv('BENCH_ROOT')}/bench_task_config/{task_config}.yml"
    else:
        cfg_path = f"./task_config/{task_config}.yml"
    with open(cfg_path, "r") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args.update({"task_name": task_name, "task_config": task_config, "ckpt_setting": ckpt_setting})

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(et):
        path = _embodiment_types[et]["file_path"]
        if path is None: raise ValueError(f"missing embodiment file for {et}")
        return path

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment must have 1 or 3 entries")

    args["left_embodiment_config"]  = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    for k in ("collect_branch_num", "collect_branch_lookback", "collect_branch_curobo_steps",
              "collect_stitch_lookback", "collect_stitch_curobo_steps", "collect_mode",
              "expert_check", "action_noise_var", "collect_num"):
        if k in usr_args and k not in args:
            args[k] = usr_args[k]

    env_type = "clean" if "clean" in task_config else "cluttered"
    # Paired (triplet) collection goes to its own dataset root, NOT rollout_data.
    _mode_early = os.environ.get("COLLECT_MODE", usr_args.get("collect_mode", args.get("collect_mode", "")))
    _root = "collision_dataset" if _mode_early == "paired" else "rollout_data"
    save_dir = Path(f"{_root}/{task_name}/{env_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    if args.get("eval_video_log"):
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = f"{camera_config['w']}x{camera_config['h']}"
        args["eval_video_save_dir"] = save_dir / "videos"
        (save_dir / "videos").mkdir(parents=True, exist_ok=True)

    TASK_ENV = class_decorator(task_name)
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"]  = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args.get("seed", 0)
    # Seed state per config in the env_type folder (so d6–d15 each keep their own seed)
    _seed_state_file = save_dir / f"seed_state_{task_config}.txt"
    if os.environ.get("COLLECT_START_SEED"):
        st_seed = int(os.environ["COLLECT_START_SEED"])
    elif _seed_state_file.exists():
        st_seed = int(_seed_state_file.read_text().strip())
        print(f"\033[33mResuming from seed {st_seed} (loaded from {_seed_state_file})\033[0m")
    else:
        st_seed = 100000 + (1 + seed)

    model_mirrored = {}
    if "pi0_step" in usr_args:
        model_mirrored["pi0_step"] = usr_args["pi0_step"]
    model = ModelClient(port=port, **model_mirrored)

    def _save_seed(seed):
        _seed_state_file.write_text(str(seed))

    # Env vars take priority over YAML config so shell scripts can override the defaults
    _branch_num   = int(os.environ.get("COLLECT_BRANCH_NUM", args.get("collect_branch_num", 0)))
    _collect_mode = os.environ.get("COLLECT_MODE", args.get("collect_mode", ""))
    if _collect_mode == "stitched":
        _collect_fn = collect_rollouts_stitched
    elif _collect_mode == "paired":
        _collect_fn = collect_rollouts_paired
    elif _branch_num > 0:
        _collect_fn = collect_rollouts_branching
    else:
        _collect_fn = collect_rollouts
    episode_log = _collect_fn(
        task_name, TASK_ENV, args, model, st_seed,
        collect_num=collect_num,
        save_dir=str(save_dir),
        instruction_type=instruction_type,
        video_size=video_size,
        save_seed_fn=_save_seed,
    )

    succ_count = sum(1 for e in episode_log if e["success"])
    coll_count = sum(1 for e in episode_log if e["collision"])
    n = len(episode_log)
    print(f"\nDone. {n} episodes saved to {save_dir}")
    print(f"SR: {succ_count/n*100:.1f}%  |  CR: {coll_count/n*100:.1f}%")


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["port"] = args.port

    if args.overrides:
        pairs = args.overrides
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            try: value = eval(pairs[i + 1])
            except: value = pairs[i + 1]
            config[key] = value

    return config


if __name__ == "__main__":
    usr_args = parse_args_and_config()
    main(usr_args)
