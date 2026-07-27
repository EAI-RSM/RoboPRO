#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Precompute time-dependent proximity weights (beta_t) for the obstacle-attention
# target, from simulator ground truth + forward kinematics.
# ------------------------------------------------------------------------------
"""
beta_t upweights the obstacle heatmap of any *non-target* obstacle that comes
within a distance threshold of the robot body/end-effectors inside a centered
temporal window. The localized contact anchor is selected from that same window,
so its position and proximity weight refer to the same closest approach. Both
are functions of the robot
proprioceptive state (arm joint angles -> FK), object-aligned obstacle boxes
(``actor_bbox/{obb_center,obb_half,obb_quat}``), the window width W, and the
distance threshold d_thresh:

    d(k, tau)  = min over robot body points of point->OBB distance to obstacle k
    d_min(k,t) = min over tau in [t - W/2, t + W/2]
    beta_k(t)  = 1 + gain * relu(1 - d_min(k,t) / d_thresh)          (continuous ramp)

We cache a small per-episode table ``beta[T, K]`` (K = number of non-target
obstacle actor ids, column-aligned with ``obj_ids``). At LeRobot conversion time
``convert_aloha_data_to_lerobot_robotwin.py grounding`` paints these per-object
weights onto the segmentation pixels to form ``observation.mask.beta``.

Self-contained in RoboPRO (ported from SafeVLA). Needs ``yourdfpy`` + the ARX5
URDF under RoboPRO assets (see scripts/beta_geometry.py DEFAULT_URDF).

Usage (from pi05-obs-attn/):

    uv run python scripts/precompute_beta_weights.py \
        --data-root /path/to/robopro-grounding-demo
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

# Allow `python scripts/precompute_beta_weights.py` from the policy root.
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import beta_geometry as bg  # noqa: E402


def _stage_target_ids(cfg_dir: str, ep_idx: int) -> np.ndarray | None:
    """Resolve per-frame target ids from an optional masking sidecar."""
    path = Path(cfg_dir) / "masking" / f"episode{ep_idx}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        data = json.load(f)
    stages = {int(stage["stage"]): int(stage["target_id"]) for stage in data.get("stages", [])}
    frame_stage = data.get("frame_stage")
    if not stages or not frame_stage:
        return None
    return np.asarray([stages.get(int(stage), -1) for stage in frame_stage], dtype=np.int64)


def _stage_destination_ids(cfg_dir: str, ep_idx: int) -> np.ndarray | None:
    """Resolve per-frame object destination ids from an optional masking sidecar."""
    path = Path(cfg_dir) / "masking" / f"episode{ep_idx}.json"
    if not path.is_file():
        return None
    with path.open() as f:
        data = json.load(f)
    stages = {
        int(stage["stage"]): int(stage["bin"]["bin_id"])
        for stage in data.get("stages", [])
        if (stage.get("bin") or {}).get("bin_id") is not None
    }
    frame_stage = data.get("frame_stage")
    if not stages or not frame_stage:
        return None
    return np.asarray([stages.get(int(stage), -1) for stage in frame_stage], dtype=np.int64)


def infer_target_grasp(
    arr: dict,
    target_id: int,
    *,
    closed_threshold: float = 0.5,
    target_motion_thresh: float = 0.005,
) -> dict[str, np.ndarray]:
    """Infer one demonstrated grasp anchor in target-local OBB coordinates.

    A carried-object interval (closed gripper + target displaced from its initial
    pose) is preferred. The arm with the most stable target-relative gripper
    position wins. If no carried interval exists, use the closest closed-gripper
    approach. This uses only fields already present in RoboPRO expert HDF5s.
    """
    num_frames = arr["LA"].shape[0]
    invalid = {
        "target_contact_local": np.full(3, np.nan, dtype=np.float32),
        "target_contact_valid": np.zeros((), dtype=bool),
        "target_contact_arm": np.asarray(-1, dtype=np.int8),
        "target_contact_frame": np.asarray(-1, dtype=np.int32),
        "target_contact_distance": np.asarray(np.inf, dtype=np.float32),
        "target_contact_confidence": np.asarray(0.0, dtype=np.float32),
    }
    if target_id < 0:
        return invalid

    cols = np.array(
        [
            {int(a): j for j, a in enumerate(arr["bb_id"][ti])}.get(int(target_id), -1)
            for ti in range(num_frames)
        ],
        dtype=np.int64,
    )
    valid = cols >= 0
    if not valid.any():
        return invalid

    safe_cols = np.maximum(cols, 0)
    centers = arr["obb_center"][np.arange(num_frames), safe_cols]
    halves = arr["obb_half"][np.arange(num_frames), safe_cols]
    quats = arr["obb_quat"][np.arange(num_frames), safe_cols]
    first = int(np.flatnonzero(valid)[0])
    moved = np.linalg.norm(centers - centers[first], axis=-1) > target_motion_thresh

    candidates = []
    for arm_idx, (eep, grip) in enumerate(((arr["LEP"], arr["LG"]), (arr["REP"], arr["RG"]))):
        local = bg.world_to_obb_normalized(eep, centers, halves, quats)
        dist = np.full(num_frames, np.inf, dtype=np.float64)
        for ti in np.flatnonzero(valid):
            dist[ti] = bg.point_obb_dist(
                eep[ti : ti + 1],
                centers[ti : ti + 1],
                halves[ti : ti + 1],
                quats[ti : ti + 1],
            )[0]

        closed = np.asarray(grip) <= closed_threshold
        carried = valid & closed & moved
        use = carried if carried.sum() >= 3 else (valid & closed)
        if not use.any():
            continue
        idx = np.flatnonzero(use)
        if carried.sum() >= 3:
            anchor = np.median(local[idx], axis=0)
            residual = np.linalg.norm(local[idx] - anchor, axis=-1)
            representative = int(idx[np.argmin(residual)])
            spread = float(np.median(residual))
        else:
            representative = int(idx[np.argmin(dist[idx])])
            anchor = local[representative]
            spread = 1.0
        distance = float(dist[representative])
        score = distance + 0.02 * spread
        candidates.append((score, arm_idx, representative, anchor, distance, spread))

    if not candidates:
        return invalid
    _, arm_idx, frame, anchor, distance, spread = min(candidates, key=lambda x: x[0])
    confidence = float(np.exp(-distance / 0.05) / (1.0 + spread))
    return {
        "target_contact_local": np.asarray(anchor, dtype=np.float32),
        "target_contact_valid": np.ones((), dtype=bool),
        "target_contact_arm": np.asarray(arm_idx, dtype=np.int8),
        "target_contact_frame": np.asarray(frame, dtype=np.int32),
        "target_contact_distance": np.asarray(distance, dtype=np.float32),
        "target_contact_confidence": np.asarray(confidence, dtype=np.float32),
    }


def select_centered_contacts(
    distance: np.ndarray,
    instantaneous_local: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select each obstacle's nearest anchor in a centered temporal window."""
    num_frames, num_objects = distance.shape
    contact_local = np.full((num_frames, num_objects, 3), np.nan, dtype=np.float32)
    contact_time_offset = np.full((num_frames, num_objects), -1, dtype=np.int32)
    contact_distance = np.full((num_frames, num_objects), np.inf, dtype=np.float32)
    half = window // 2
    for ti in range(num_frames):
        lo = max(0, ti - half)
        hi = min(num_frames, ti + half + 1)
        if num_objects == 0:
            continue
        indices = np.argmin(distance[lo:hi], axis=0)
        chosen_t = lo + indices
        finite = np.isfinite(distance[chosen_t, np.arange(num_objects)])
        columns = np.flatnonzero(finite)
        contact_local[ti, finite] = instantaneous_local[chosen_t[finite], columns]
        contact_time_offset[ti, finite] = chosen_t[finite] - ti
        contact_distance[ti, finite] = distance[chosen_t[finite], columns]
    return contact_local, contact_time_offset, contact_distance


def compute_episode_beta(
    urdf,
    h5_path,
    obj_ids,
    window,
    d_thresh,
    gain,
    samples_per_link,
    motion_thresh,
    band,
    *,
    scene_ep: dict | None = None,
    contact_window: int | None = None,
    grasp_closed_threshold: float = 0.5,
    target_motion_threshold: float = 0.005,
    target_ids: np.ndarray | None = None,
    return_contacts: bool = False,
):
    arr = bg.load_episode_arrays(h5_path)
    T = arr["LA"].shape[0]
    K = len(obj_ids)
    R, t, res = bg.solve_base_to_world(urdf, arr)
    if res.mean() > 0.01:  # 1 cm — FK / base-parked assumption broke
        values = (None, None, f"SKIP Kabsch residual mean={res.mean()*1e3:.1f}mm too high")
        return (*values[:2], None, values[2]) if return_contacts else values

    # robot body points for all frames, then keep only the MOVING ones so that
    # static arm-base/shoulder links near back furniture don't flag far obstacles.
    pts_seq = bg.robot_world_points_sequence(urdf, arr, R, t, samples_per_link)  # [T,P,3]
    moving = bg.moving_point_mask(pts_seq, motion_thresh)
    pts_seq = pts_seq[:, moving]  # [T,P',3]

    # per-frame instantaneous distance d[t, k] (min over moving robot points)
    # Uses object-aligned OBBs (center/half/quat), not inflated world AABBs.
    d = np.full((T, K), np.inf, dtype=np.float64)
    instantaneous_local = np.full((T, K, 3), np.nan, dtype=np.float64)
    for ti in range(T):
        cols = bg.obstacle_columns(arr["bb_id"], obj_ids, ti)
        valid = cols >= 0
        if not valid.any():
            continue
        c = cols[valid]
        distance, closest_local, _ = bg.point_obb_closest(
            pts_seq[ti],
            arr["obb_center"][ti, c],
            arr["obb_half"][ti, c],
            arr["obb_quat"][ti, c],
        )
        d[ti, valid] = distance
        instantaneous_local[ti, valid] = bg.obb_local_normalize(
            closest_local, arr["obb_half"][ti, c]
        )

    # Preserve the original centered-window beta for object-mask baselines.
    spatial_window = window if contact_window is None else contact_window
    half = window // 2
    d_min = np.empty_like(d)
    for ti in range(T):
        lo, hi = max(0, ti - half), min(T, ti + half + 1)
        d_min[ti] = d[lo:hi].min(axis=0)

    # within threshold: ramp 1 -> 1+gain as d_min -> 0. Beyond the threshold,
    # instead of a hard 1 -> 0 cliff, decay linearly to 0 over a soft `band`
    # (meters): so an obstacle hovering near d_thresh fades smoothly rather than
    # flickering full<->zero frame-to-frame. band=0 recovers the hard cliff.
    def proximity_weights(min_distance):
        ramp = np.clip(1.0 - min_distance / d_thresh, 0.0, None)
        band_decay = 1.0 - np.clip((min_distance - d_thresh) / max(band, 1e-9), 0.0, 1.0)
        return np.where(min_distance < d_thresh, 1.0 + gain * ramp, band_decay).astype(np.float32)

    beta = proximity_weights(d_min)
    if spatial_window == window:
        contact_beta = beta.copy()
    else:
        contact_half = spatial_window // 2
        contact_d_min = np.empty_like(d)
        for ti in range(T):
            lo, hi = max(0, ti - contact_half), min(T, ti + contact_half + 1)
            contact_d_min[ti] = d[lo:hi].min(axis=0)
        contact_beta = proximity_weights(contact_d_min)

    # Spatial contact anchor uses the same centered closest-approach window.
    # The anchor is stored in the obstacle's local frame at the selected
    # time, allowing conversion to render it at the obstacle's current pose.
    contact_local, contact_time_offset, contact_distance = select_centered_contacts(
        d, instantaneous_local, spatial_window
    )

    roles = (scene_ep or {}).get("role_names", {}) or {}
    default_target_id = int(roles["target_id"]) if roles.get("target_id") is not None else -1
    if target_ids is None:
        target_ids = np.full(T, default_target_id, dtype=np.int64)
    else:
        target_ids = np.asarray(target_ids, dtype=np.int64)
        if target_ids.shape[0] < T:
            target_ids = np.pad(target_ids, (0, T - target_ids.shape[0]), constant_values=default_target_id)
        target_ids = target_ids[:T]

    target_contact_local = np.full((T, 3), np.nan, dtype=np.float32)
    target_contact_valid = np.zeros(T, dtype=bool)
    target_contact_arm = np.full(T, -1, dtype=np.int8)
    target_contact_frame = np.full(T, -1, dtype=np.int32)
    target_contact_distance = np.full(T, np.inf, dtype=np.float32)
    target_contact_confidence = np.zeros(T, dtype=np.float32)
    for target_id in np.unique(target_ids[target_ids >= 0]):
        grasp = infer_target_grasp(
            arr,
            int(target_id),
            closed_threshold=grasp_closed_threshold,
            target_motion_thresh=target_motion_threshold,
        )
        select = target_ids == target_id
        target_contact_local[select] = grasp["target_contact_local"]
        target_contact_valid[select] = grasp["target_contact_valid"]
        target_contact_arm[select] = grasp["target_contact_arm"]
        target_contact_frame[select] = grasp["target_contact_frame"]
        target_contact_distance[select] = grasp["target_contact_distance"]
        target_contact_confidence[select] = grasp["target_contact_confidence"]

    contacts = {
        "obstacle_contact_local": contact_local,
        "obstacle_contact_time_offset": contact_time_offset,
        "obstacle_contact_distance": contact_distance,
        "obstacle_contact_valid": np.isfinite(contact_distance),
        "contact_beta": contact_beta,
        "contact_window": np.asarray(spatial_window, dtype=np.int32),
        "target_contact_id": target_ids,
        "target_contact_local": target_contact_local,
        "target_contact_valid": target_contact_valid,
        "target_contact_arm": target_contact_arm,
        "target_contact_frame": target_contact_frame,
        "target_contact_distance": target_contact_distance,
        "target_contact_confidence": target_contact_confidence,
    }
    # Retain raw pre-window distances for diagnostics and offline analysis.
    beta_max = float(beta.max()) if beta.size else 0.0
    result = (
        beta,
        d.astype(np.float32),
        (
            f"OK T={T} K={K} res={res.mean()*1e3:.2f}mm moving_pts={int(moving.sum())}/{moving.size} "
            f"beta_max={beta_max:.2f} frac_within={float((beta > 1).mean()) if beta.size else 0.0:.2f} "
            f"frac_band={float(((beta > 0) & (beta < 1)).mean()) if beta.size else 0.0:.2f} "
            f"frac_zero={float((beta == 0).mean()) if beta.size else 0.0:.2f} "
            f"target_contact={float(contacts['target_contact_valid'].mean()):.2f}"
        ),
    )
    if return_contacts:
        return result[0], result[1], contacts, result[2]
    return result


def _iter_config_dirs(
    root: str,
    *,
    domains: list[str] | None = None,
    tasks: list[str] | None = None,
    configs: list[str] | None = None,
) -> list[str]:
    """Find config dirs that contain both ``data/`` and ``scene_info.json``.

    Supports ``mzxuan/robopro_expert`` layout ``<domain>/<task>/<config>/`` and the
    flatter demo layout ``<task>/<config>/``.
    """
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune heavy non-episode trees early.
        dirnames[:] = [d for d in dirnames if d not in {"_traj_data", "export", "scene", "video", "masking", "beta_weights"}]
        if "scene_info.json" not in filenames or "data" not in dirnames:
            continue
        data_dir = os.path.join(dirpath, "data")
        if not any(f.endswith(".hdf5") for f in os.listdir(data_dir)):
            continue
        rel = os.path.relpath(dirpath, root)
        parts = Path(rel).parts
        # Filter by optional domain/task/config path components (from the end).
        # parts may be (domain, task, config) or (task, config).
        if configs is not None and (not parts or parts[-1] not in configs):
            continue
        if tasks is not None:
            task_name = parts[-2] if len(parts) >= 2 else None
            if task_name not in tasks:
                continue
        if domains is not None:
            domain_name = parts[0] if len(parts) >= 3 else None
            if domain_name not in domains:
                continue
        found.append(dirpath)
    return sorted(found)


def precompute_beta_weights(
    data_root: str,
    *,
    domains: list[str] | None = None,
    tasks: list[str] | None = None,
    configs: list[str] | None = None,
    urdf_path: str | None = None,
    window: int = 15,
    threshold: float = 0.15,
    gain: float = 3.0,
    band: float = 0.05,
    motion_thresh: float = 0.02,
    samples_per_link: int = 2,
    exclude_target: bool = True,
    contact_window: int | None = None,
    grasp_closed_threshold: float = 0.5,
    target_motion_threshold: float = 0.005,
    out_subdir: str = "beta_weights",
    beta_root: str | None = None,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Walk expert HDF5 trees and write per-episode beta caches.

    If ``beta_root`` is given, ``data_root`` is treated as read-only: caches are
    written to ``<beta_root>/<domain>/<task>/<config>/<out_subdir>/`` instead of
    next to the source HDF5s. Otherwise they land in ``<config>/<out_subdir>/``.

    Returns ``(n_ok, n_skip)``.
    """
    root = os.path.expanduser(data_root)
    beta_base = os.path.expanduser(beta_root) if beta_root else None
    urdf = bg.load_urdf(urdf_path or bg.DEFAULT_URDF)

    n_ok = n_skip = 0
    for cfg_dir in _iter_config_dirs(root, domains=domains, tasks=tasks, configs=configs):
        data_dir = os.path.join(cfg_dir, "data")
        scene_path = os.path.join(cfg_dir, "scene_info.json")
        rel = os.path.relpath(cfg_dir, root)
        out_dir = os.path.join(beta_base, rel, out_subdir) if beta_base else os.path.join(cfg_dir, out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        for fn in sorted((f for f in os.listdir(data_dir) if f.endswith(".hdf5")), key=bg.episode_idx):
            h5_path = os.path.join(data_dir, fn)
            out_path = os.path.join(out_dir, fn.replace(".hdf5", ".npz"))
            if os.path.isfile(out_path) and not overwrite:
                with np.load(out_path) as existing:
                    if all(
                        key in existing
                        for key in (
                            "obstacle_contact_local",
                            "target_contact_local",
                            "target_contact_id",
                            "contact_beta",
                            "contact_window",
                        )
                    ):
                        print(f"[skip-exists] {rel}/{fn}")
                        continue
                print(f"[upgrade-cache] {rel}/{fn}")
            scene_ep = bg.scene_episode(scene_path, bg.episode_idx(h5_path))
            obj_ids = bg.obstacle_ids(scene_ep, exclude_target=exclude_target)
            target_ids = _stage_target_ids(cfg_dir, bg.episode_idx(h5_path))
            destination_ids = _stage_destination_ids(cfg_dir, bg.episode_idx(h5_path))
            if exclude_target and target_ids is not None:
                obj_ids = np.setdiff1d(obj_ids, np.unique(target_ids[target_ids >= 0]))
            if exclude_target and destination_ids is not None:
                obj_ids = np.setdiff1d(obj_ids, np.unique(destination_ids[destination_ids >= 0]))
            beta, dist, contacts, report = compute_episode_beta(
                urdf,
                h5_path,
                obj_ids,
                window,
                threshold,
                gain,
                samples_per_link,
                motion_thresh,
                band,
                scene_ep=scene_ep,
                contact_window=contact_window,
                grasp_closed_threshold=grasp_closed_threshold,
                target_motion_threshold=target_motion_threshold,
                target_ids=target_ids,
                return_contacts=True,
            )
            print(f"[{'ok  ' if beta is not None else 'skip'}] {rel}/{fn}: {report}")
            if beta is None:
                n_skip += 1
                continue
            np.savez(
                out_path,
                beta=beta,
                dist=dist,
                obj_ids=obj_ids.astype(np.int64),
                window=window,
                d_thresh=threshold,
                gain=gain,
                band=band,
                motion_thresh=motion_thresh,
                samples_per_link=samples_per_link,
                grasp_closed_threshold=grasp_closed_threshold,
                target_motion_threshold=target_motion_threshold,
                **contacts,
            )
            n_ok += 1

    print(f"\n[done] {n_ok} episodes cached / {n_skip} skipped -> */{out_subdir}/*.npz")
    return n_ok, n_skip


def main() -> None:
    ap = argparse.ArgumentParser("precompute beta_t proximity weights (RoboPRO self-contained)")
    ap.add_argument(
        "--data-root",
        required=True,
        help="root of mzxuan/robopro_expert (or similar): "
        "<domain>/<task>/<config>/data/episode*.hdf5 (+ scene_info.json)",
    )
    ap.add_argument("--domains", nargs="*", default=None, help="e.g. kitchenl office")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--urdf", default=None, help=f"default: {bg.DEFAULT_URDF}")
    ap.add_argument("--window", type=int, default=15, help="centered temporal window width (frames)")
    ap.add_argument("--threshold", type=float, default=0.15, help="distance threshold (m)")
    ap.add_argument("--gain", type=float, default=3.0, help="max extra weight at contact")
    ap.add_argument(
        "--band",
        type=float,
        default=0.05,
        help="soft-gate band (m): beyond the threshold beta decays 1->0 "
        "linearly over this distance instead of a hard cliff (0 = hard cliff)",
    )
    ap.add_argument(
        "--motion-thresh",
        type=float,
        default=0.02,
        help="min per-point displacement (m) to count a robot link as 'moving'",
    )
    ap.add_argument("--samples-per-link", type=int, default=2)
    ap.add_argument(
        "--contact-window",
        type=int,
        default=None,
        help="centered window used to select spatial contact anchors (default: --window)",
    )
    ap.add_argument("--grasp-closed-threshold", type=float, default=0.5)
    ap.add_argument("--target-motion-threshold", type=float, default=0.005, help="metres")
    ap.add_argument(
        "--exclude-target",
        action="store_true",
        default=True,
        help="drop target/destination ids (default on)",
    )
    ap.add_argument("--include-target", dest="exclude_target", action="store_false")
    ap.add_argument("--out-subdir", default="beta_weights")
    ap.add_argument(
        "--beta-root",
        default=None,
        help="write caches under <beta-root>/<domain>/<task>/<config>/<out-subdir>/ "
        "instead of next to the HDF5s; keeps --data-root read-only",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    precompute_beta_weights(
        args.data_root,
        domains=args.domains,
        tasks=args.tasks,
        configs=args.configs,
        urdf_path=args.urdf,
        window=args.window,
        threshold=args.threshold,
        gain=args.gain,
        band=args.band,
        motion_thresh=args.motion_thresh,
        samples_per_link=args.samples_per_link,
        exclude_target=args.exclude_target,
        contact_window=args.contact_window,
        grasp_closed_threshold=args.grasp_closed_threshold,
        target_motion_threshold=args.target_motion_threshold,
        out_subdir=args.out_subdir,
        beta_root=args.beta_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
