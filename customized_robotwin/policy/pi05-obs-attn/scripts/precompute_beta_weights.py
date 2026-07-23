#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Precompute time-dependent proximity weights (beta_t) for the obstacle-attention
# target, from simulator ground truth + forward kinematics.
# ------------------------------------------------------------------------------
"""
beta_t upweights the obstacle heatmap of any *non-target* obstacle that comes
within a distance threshold of the robot body/end-effectors at any point inside
a centered temporal window around the frame. It is a function of the robot
proprioceptive state (arm joint angles -> FK), object-aligned obstacle boxes
(``actor_bbox/{obb_center,obb_half,obb_quat}``), the window width W, and the
distance threshold d_thresh:

    d(k, tau)  = min over robot body points of point->OBB distance to obstacle k
    d_min(k,t) = min over tau in [t - W/2, t + W/2] of d(k, tau)     (centered)
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
import os
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/precompute_beta_weights.py` from the policy root.
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import beta_geometry as bg  # noqa: E402


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
):
    arr = bg.load_episode_arrays(h5_path)
    T = arr["LA"].shape[0]
    K = len(obj_ids)
    if K == 0:
        z = np.zeros((T, 0), dtype=np.float32)
        return np.ones((T, 0), dtype=np.float32), z, f"OK T={T} K=0 (no non-target obstacles)"

    R, t, res = bg.solve_base_to_world(urdf, arr)
    if res.mean() > 0.01:  # 1 cm — FK / base-parked assumption broke
        return None, None, f"SKIP Kabsch residual mean={res.mean()*1e3:.1f}mm too high"

    # robot body points for all frames, then keep only the MOVING ones so that
    # static arm-base/shoulder links near back furniture don't flag far obstacles.
    pts_seq = bg.robot_world_points_sequence(urdf, arr, R, t, samples_per_link)  # [T,P,3]
    moving = bg.moving_point_mask(pts_seq, motion_thresh)
    pts_seq = pts_seq[:, moving]  # [T,P',3]

    # per-frame instantaneous distance d[t, k] (min over moving robot points)
    # Uses object-aligned OBBs (center/half/quat), not inflated world AABBs.
    d = np.full((T, K), np.inf, dtype=np.float64)
    for ti in range(T):
        cols = bg.obstacle_columns(arr["bb_id"], obj_ids, ti)
        valid = cols >= 0
        if not valid.any():
            continue
        c = cols[valid]
        d[ti, valid] = bg.point_obb_dist(
            pts_seq[ti],
            arr["obb_center"][ti, c],
            arr["obb_half"][ti, c],
            arr["obb_quat"][ti, c],
        )

    # centered temporal window min, then continuous ramp
    half = window // 2
    d_min = np.empty_like(d)
    for ti in range(T):
        lo, hi = max(0, ti - half), min(T, ti + half + 1)
        d_min[ti] = d[lo:hi].min(axis=0)
    # within threshold: ramp 1 -> 1+gain as d_min -> 0. Beyond the threshold,
    # instead of a hard 1 -> 0 cliff, decay linearly to 0 over a soft `band`
    # (meters): so an obstacle hovering near d_thresh fades smoothly rather than
    # flickering full<->zero frame-to-frame. band=0 recovers the hard cliff.
    ramp = np.clip(1.0 - d_min / d_thresh, 0.0, None)
    band_decay = 1.0 - np.clip((d_min - d_thresh) / max(band, 1e-9), 0.0, 1.0)
    beta = np.where(d_min < d_thresh, 1.0 + gain * ramp, band_decay).astype(np.float32)
    # also return the raw per-frame min distance (pre-window) so interactive tools
    # can recompute beta live for any window / threshold / gain / band.
    return (
        beta,
        d.astype(np.float32),
        (
            f"OK T={T} K={K} res={res.mean()*1e3:.2f}mm moving_pts={int(moving.sum())}/{moving.size} "
            f"beta_max={beta.max():.2f} frac_within={float((beta > 1).mean()):.2f} "
            f"frac_band={float(((beta > 0) & (beta < 1)).mean()):.2f} "
            f"frac_zero={float((beta == 0).mean()):.2f}"
        ),
    )


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
    out_subdir: str = "beta_weights",
    overwrite: bool = False,
) -> tuple[int, int]:
    """Walk expert HDF5 trees and write per-episode beta caches.

    Returns ``(n_ok, n_skip)``.
    """
    root = os.path.expanduser(data_root)
    urdf = bg.load_urdf(urdf_path or bg.DEFAULT_URDF)

    n_ok = n_skip = 0
    for cfg_dir in _iter_config_dirs(root, domains=domains, tasks=tasks, configs=configs):
        data_dir = os.path.join(cfg_dir, "data")
        scene_path = os.path.join(cfg_dir, "scene_info.json")
        out_dir = os.path.join(cfg_dir, out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        rel = os.path.relpath(cfg_dir, root)
        for fn in sorted((f for f in os.listdir(data_dir) if f.endswith(".hdf5")), key=bg.episode_idx):
            h5_path = os.path.join(data_dir, fn)
            out_path = os.path.join(out_dir, fn.replace(".hdf5", ".npz"))
            if os.path.isfile(out_path) and not overwrite:
                print(f"[skip-exists] {rel}/{fn}")
                continue
            scene_ep = bg.scene_episode(scene_path, bg.episode_idx(h5_path))
            obj_ids = bg.obstacle_ids(scene_ep, exclude_target=exclude_target)
            beta, dist, report = compute_episode_beta(
                urdf,
                h5_path,
                obj_ids,
                window,
                threshold,
                gain,
                samples_per_link,
                motion_thresh,
                band,
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
    ap.add_argument("--window", type=int, default=15, help="temporal window width (frames)")
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
        "--exclude-target",
        action="store_true",
        default=True,
        help="drop target/destination ids (default on)",
    )
    ap.add_argument("--include-target", dest="exclude_target", action="store_false")
    ap.add_argument("--out-subdir", default="beta_weights")
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
        out_subdir=args.out_subdir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
