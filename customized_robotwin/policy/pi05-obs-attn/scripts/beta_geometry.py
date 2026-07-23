# ------------------------------------------------------------------------------
# Shared geometry for the beta_t proximity-weighting pipeline (offline).
#
# Forward kinematics of the aloha-agilex dual-ARX5 arms, per-episode base->world
# transform recovery, robot body point sampling, point->AABB distance, and
# world->image projection. Used by scripts/precompute_beta_weights.py.
#
# Ported from SafeVLA scripts/beta_geometry.py into RoboPRO so the obstacle-
# attention pipeline is self-contained (no SafeVLA dependency at data time).
# ------------------------------------------------------------------------------
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

# Actors that are scenery or the robot itself (mirrors the LeRobot converter).
DEFAULT_EXCLUDE_NAMES = ("ground", "wall", "table", "floor", "robot")

# RoboPRO ships the aloha-agilex URDF under both customized_robotwin/assets and
# benchmark/assets after `download_assets`. Prefer the closer policy-tree copy.
_POLICY_ROOT = Path(__file__).resolve().parents[1]  # pi05-obs-attn/
_ROBOPRO_ROOT = _POLICY_ROOT.parents[2]  # RoboPRO/
_URDF_CANDIDATES = (
    _ROBOPRO_ROOT
    / "customized_robotwin"
    / "assets"
    / "embodiments"
    / "aloha-agilex"
    / "urdf"
    / "arx5_description_isaac.urdf",
    _ROBOPRO_ROOT
    / "benchmark"
    / "assets"
    / "embodiments"
    / "aloha-agilex"
    / "urdf"
    / "arx5_description_isaac.urdf",
)


def default_urdf() -> str:
    for p in _URDF_CANDIDATES:
        if p.is_file():
            return str(p)
    # Fall back to the preferred relative location so the error message is useful.
    return str(_URDF_CANDIDATES[0])


DEFAULT_URDF = default_urdf()

# Kinematic chains we sample points along (both arms, base -> gripper).
ARM_LINK_CHAINS = {
    "left": ["fl_base_link"] + [f"fl_link{i}" for i in range(1, 9)],
    "right": ["fr_base_link"] + [f"fr_link{i}" for i in range(1, 9)],
}
EEF_LINKS = {"left": "fl_link6", "right": "fr_link6"}
ARM_JOINTS = {
    "left": [f"fl_joint{i}" for i in range(1, 7)],
    "right": [f"fr_joint{i}" for i in range(1, 7)],
}


# ------------------------------- io / scene ----------------------------------
def episode_idx(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else -1


def scene_episode(scene_info_path: str, ep_idx: int) -> dict:
    return json.load(open(scene_info_path))[f"episode_{ep_idx}"]


def scene_info_path_for(h5_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(h5_path)), "scene_info.json")


def obstacle_ids(scene_ep: dict, exclude_names=DEFAULT_EXCLUDE_NAMES, exclude_target: bool = True) -> np.ndarray:
    """Non-robot, non-scenery actor ids (obstacles). Optionally drop the task
    target/destination so proximity weighting emphasizes *non-target* objects."""
    id_map = scene_ep["actor_id_map"]
    keep = [int(k) for k, name in id_map.items() if not any(name.startswith(p) for p in exclude_names)]
    if exclude_target:
        roles = scene_ep.get("role_names", {}) or {}
        drop = {int(roles[k]) for k in ("target_id", "destination_id") if roles.get(k) is not None}
        keep = [k for k in keep if k not in drop]
    return np.asarray(sorted(keep), dtype=np.int64)


def id_to_name(scene_ep: dict) -> dict:
    return {int(k): v for k, v in scene_ep["actor_id_map"].items()}


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """Batch convert wxyz quaternions [..., 4] -> rotation matrices [..., 3, 3].

    Matches sapien / transforms3d convention (w, x, y, z).
    """
    q = np.asarray(q, dtype=np.float64)
    lead = q.shape[:-1]
    q = q.reshape(-1, 4)
    # Normalize to guard against tiny numerical drift in logged poses.
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = q.T
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R.reshape(*lead, 3, 3)


def load_episode_arrays(h5_path: str) -> dict:
    """Load joint/endpose arrays + per-frame oriented actor boxes.

    Current RoboPRO format (``envs/_base_task.py``) stores object-aligned boxes:
      ``obb_center`` [T,N,3], ``obb_half`` [T,N,3], ``obb_quat`` [T,N,4] (wxyz).

    Older demos may still have world-frame AABBs (``aabb_min``/``aabb_max``); those
    are converted to an equivalent OBB with identity orientation.
    """
    with h5py.File(h5_path, "r") as f:
        out = dict(
            LA=f["joint_action/left_arm"][()],  # [T,6]
            RA=f["joint_action/right_arm"][()],  # [T,6]
            LEP=f["endpose/left_endpose"][:, :3],  # [T,3] world
            REP=f["endpose/right_endpose"][:, :3],
            bb_id=f["actor_bbox/id"][()],  # [T,N]
        )
        g = f["actor_bbox"]
        if "obb_center" in g and "obb_half" in g and "obb_quat" in g:
            out["obb_center"] = g["obb_center"][()].astype(np.float64)
            out["obb_half"] = g["obb_half"][()].astype(np.float64)
            out["obb_quat"] = g["obb_quat"][()].astype(np.float64)
        elif "aabb_min" in g and "aabb_max" in g:
            # Legacy axis-aligned format -> degenerate OBB (identity quat).
            mn = g["aabb_min"][()].astype(np.float64)
            mx = g["aabb_max"][()].astype(np.float64)
            out["obb_center"] = 0.5 * (mn + mx)
            out["obb_half"] = 0.5 * (mx - mn)
            out["obb_quat"] = np.zeros((*mn.shape[:-1], 4), dtype=np.float64)
            out["obb_quat"][..., 0] = 1.0
        else:
            raise KeyError(
                f"{h5_path}: actor_bbox missing OBB fields "
                f"(obb_center/obb_half/obb_quat) and legacy aabb_min/aabb_max; "
                f"got keys {list(g.keys())}"
            )
        return out


def load_camera_frame(h5_path: str, frame: int, cam: str = "countertop_camera"):
    """Decoded RGB [H,W,3] uint8 + extrinsic_cv [3,4] (w2c) + intrinsic_cv [3,3]."""
    with h5py.File(h5_path, "r") as f:
        buf = np.frombuffer(f[f"observation/{cam}/rgb"][frame], dtype=np.uint8)
        rgb = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        ext = f[f"observation/{cam}/extrinsic_cv"][frame].astype(np.float64)  # [3,4]
        K = f[f"observation/{cam}/intrinsic_cv"][frame].astype(np.float64)  # [3,3]
    return rgb, ext, K


# --------------------------------- kinematics --------------------------------
def load_urdf(urdf_path: str = DEFAULT_URDF):
    try:
        import yourdfpy
    except ImportError as e:
        raise ImportError(
            "yourdfpy is required for beta_t FK precompute; install with "
            "`uv add yourdfpy` / `pip install yourdfpy`."
        ) from e
    path = os.path.expanduser(urdf_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"URDF not found at {path}. Download RoboPRO assets "
            "(benchmark/assets/embodiments or customized_robotwin/assets/embodiments)."
        )
    return yourdfpy.URDF.load(path, load_meshes=False, build_scene_graph=True)


def fk_link_origins(urdf, la: np.ndarray, ra: np.ndarray, links) -> dict:
    """Set the 12 arm joints for one frame; return {link: base-frame origin}."""
    cfg = {}
    for i in range(6):
        cfg[ARM_JOINTS["left"][i]] = float(la[i])
        cfg[ARM_JOINTS["right"][i]] = float(ra[i])
    urdf.update_cfg(cfg)
    return {lk: urdf.get_transform(lk, "base_link")[:3, 3].copy() for lk in links}


def sample_body_points(origins: dict, samples_per_link: int) -> np.ndarray:
    """Robot point cloud (base frame): link origins + interpolated points along
    each consecutive link segment of both arm chains. [P,3]."""
    pts = []
    for chain in ARM_LINK_CHAINS.values():
        for a, b in zip(chain[:-1], chain[1:], strict=False):
            pa, pb = origins[a], origins[b]
            pts.append(pa)
            for s in range(1, samples_per_link + 1):
                f = s / (samples_per_link + 1)
                pts.append(pa * (1 - f) + pb * f)
        pts.append(origins[chain[-1]])
    return np.asarray(pts, dtype=np.float64)


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid (R, t) minimizing ||R P + t - Q||. Returns R, t, per-point residual."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = Q.mean(0) - R @ P.mean(0)
    res = np.linalg.norm((P @ R.T + t) - Q, axis=1)
    return R, t, res


def solve_base_to_world(urdf, arr: dict):
    """Per-episode base->world (R, t) from FK-EEF (base frame) vs endpose (world),
    across all frames + both arms. Returns R, t, residual array."""
    P, Q = [], []
    T = arr["LA"].shape[0]
    for ti in range(T):
        o = fk_link_origins(urdf, arr["LA"][ti], arr["RA"][ti], [EEF_LINKS["left"], EEF_LINKS["right"]])
        P.append(o[EEF_LINKS["left"]])
        Q.append(arr["LEP"][ti])
        P.append(o[EEF_LINKS["right"]])
        Q.append(arr["REP"][ti])
    return kabsch(np.asarray(P), np.asarray(Q))


def robot_world_points(urdf, la, ra, R, t, samples_per_link: int):
    """Robot body point cloud in WORLD frame for one frame. [P,3]."""
    links = ARM_LINK_CHAINS["left"] + ARM_LINK_CHAINS["right"]
    origins = fk_link_origins(urdf, la, ra, links)
    pts_base = sample_body_points(origins, samples_per_link)
    return pts_base @ R.T + t


def robot_world_points_sequence(urdf, arr, R, t, samples_per_link):
    """World-frame robot points for every frame. [T, P, 3]."""
    T = arr["LA"].shape[0]
    return np.stack(
        [
            robot_world_points(urdf, arr["LA"][ti], arr["RA"][ti], R, t, samples_per_link)
            for ti in range(T)
        ]
    )


def moving_point_mask(pts_seq: np.ndarray, motion_thresh: float) -> np.ndarray:
    """Boolean [P] mask of body points that actually MOVE over the episode
    (max displacement from their mean > motion_thresh, metres). Drops static
    links — arm-base / shoulder mounts — so persistently-near-but-immobile
    structure doesn't flag far obstacles. Falls back to all points if none move."""
    dev = np.linalg.norm(pts_seq - pts_seq.mean(axis=0, keepdims=True), axis=2)  # [T,P]
    moving = dev.max(axis=0) > motion_thresh  # [P]
    return moving if moving.any() else np.ones(pts_seq.shape[1], dtype=bool)


# --------------------------------- distances ---------------------------------
def point_aabb_dist(pts: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Min distance from a set of points to each world-frame AABB.
    pts [P,3], mins/maxs [K,3] -> [K] (0 if any point is inside the box)."""
    lo = np.clip(mins[None] - pts[:, None], 0.0, None)  # [P,K,3]
    hi = np.clip(pts[:, None] - maxs[None], 0.0, None)  # [P,K,3]
    d = np.linalg.norm(lo + hi, axis=2)  # [P,K]
    return d.min(axis=0)  # [K]


def point_obb_dist(
    pts: np.ndarray,
    centers: np.ndarray,
    halves: np.ndarray,
    quats: np.ndarray,
) -> np.ndarray:
    """Min distance from a set of world points to each oriented box.

    pts [P,3], centers/halves [K,3], quats [K,4] (wxyz) -> [K].
    Distance is 0 if any point lies inside the box. The box frame is the object's
    own pose: local = R^T (p - center), then clamp |local| - half.
    """
    R = _quat_wxyz_to_mat(quats)  # [K,3,3], local -> world
    # world -> local: (p - c) @ R  (since R columns are local axes in world)
    delta = pts[:, None, :] - centers[None, :, :]  # [P,K,3]
    local = np.einsum("pkj,kji->pki", delta, R)  # [P,K,3]
    outside = np.clip(np.abs(local) - halves[None], 0.0, None)  # [P,K,3]
    d = np.linalg.norm(outside, axis=2)  # [P,K]
    return d.min(axis=0)  # [K]


def obstacle_columns(bb_id: np.ndarray, obj_ids: np.ndarray, frame: int) -> np.ndarray:
    """Column index of each obstacle id in the per-frame bbox table (-1 if absent)."""
    idmap = {int(a): i for i, a in enumerate(bb_id[frame])}
    return np.array([idmap.get(int(k), -1) for k in obj_ids])


# --------------------------------- projection --------------------------------
def project_world_to_image(pts_world: np.ndarray, ext_cv: np.ndarray, K: np.ndarray):
    """World points [N,3] -> (pixels [N,2], depth [N]) via OpenCV w2c extrinsic + K.
    Depth <= 0 means behind the camera (pixel still returned, caller should mask)."""
    N = pts_world.shape[0]
    Xw = np.concatenate([pts_world, np.ones((N, 1))], axis=1)  # [N,4]
    Xc = Xw @ ext_cv.T  # [N,3] camera frame
    z = Xc[:, 2]
    uv = Xc @ K.T  # [N,3]
    uv = uv[:, :2] / np.clip(z[:, None], 1e-6, None)
    return uv, z


def obb_corners(center: np.ndarray, half: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """8 world-frame corners [8,3] of an oriented box (center/half [3], quat wxyz [4])."""
    signs = np.array(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
    )
    local = signs * half[None]
    R = _quat_wxyz_to_mat(quat)  # [3,3]
    return local @ R.T + center[None]


def aabb_corners(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """8 corners [8,3] of an AABB given min/max [3] (legacy helper)."""
    return obb_corners(0.5 * (mins + maxs), 0.5 * (maxs - mins), np.array([1.0, 0.0, 0.0, 0.0]))


AABB_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]
OBB_EDGES = AABB_EDGES
