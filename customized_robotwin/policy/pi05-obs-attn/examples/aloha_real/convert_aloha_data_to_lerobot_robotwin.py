"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>

Obstacle-attention variant — primary source is the HuggingFace raw expert dump
``mzxuan/robopro_expert`` (layout ``<domain>/<task>/<config>/data/episodeN.hdf5`` +
``scene_info.json``; actor boxes are object-aligned ``obb_*``). Downloads e.g.:

    huggingface-cli download mzxuan/robopro_expert --repo-type dataset \\
        --local-dir /path/to/robopro_expert

Then bake obstacle/beta/target/dest masks into a LeRobot dataset. The existing
obstacle/target fields are contact-localized from demonstrated motion; beta
proximity weights and object-local anchors come from
``scripts/precompute_beta_weights.py``. Pass ``--precompute-beta`` to generate or
upgrade those caches:

    uv run python scripts/precompute_beta_weights.py --data-root /path/to/robopro_expert

    uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py grounding \\
        --raw-dir /path/to/robopro_expert --repo-id local/robopro_expert \\
        --precompute-beta --attention-mask-mode contact \\
        --contact-audit-dir ./contact_audit
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import sys
import fnmatch

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import beta_geometry as _bg  # noqa: E402


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]

    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors), ),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                data = np.frombuffer(data, np.uint8)
                # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 解码为彩色图像
                imgs_array.append(cv2.imdecode(data, cv2.IMREAD_COLOR))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]
        # add prompt
        dir_path = os.path.dirname(ep_path)
        json_Path = f"{dir_path}/instructions.json"

        with open(json_Path, 'r') as f_instr:
            instruction_dict = json.load(f_instr)
            instructions = instruction_dict['instructions']
            instruction = np.random.choice(instructions)
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]
            dataset.add_frame(frame)
        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
        # download_raw(raw_dir, repo_id=raw_repo_id)
    hdf5_files = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            file_path = os.path.join(root, filename)
            hdf5_files.append(file_path)

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    # dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


# ---------------------------------------------------------------------------
# Obstacle-attention: read robopro-grounding-demo raw HDF5 directly and bake
# obstacle/beta/target/dest masks. Logic matches the SafeVLA attacher, but all
# tooling lives in this policy tree (see scripts/precompute_beta_weights.py).
# ---------------------------------------------------------------------------

# Actors that are scenery or the robot itself; everything else is an obstacle.
DEFAULT_EXCLUDE_NAMES = ("ground", "wall", "table", "floor", "robot")

# Native head/countertop camera view is fed to the model as base_0_rgb, which the
# model resizes to this resolution with aspect-preserving padding. We bake masks
# to the same geometry so they stay pixel-aligned with the SigLIP tokens.
MASK_RESOLUTION = (224, 224)


def _grounding_episode_idx(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits) if digits else -1


def _object_actor_ids(scene_info_path: str, ep_idx: int, exclude_names, exclude_roles: bool = True) -> np.ndarray:
    """Actor ids counted as obstacles: every named actor whose name does not start
    with an excluded prefix, minus the task target/destination when exclude_roles."""
    ep = json.load(open(scene_info_path))[f"episode_{ep_idx}"]
    id_map = ep["actor_id_map"]
    keep = [int(k) for k, name in id_map.items() if not any(name.startswith(p) for p in exclude_names)]
    if exclude_roles:
        roles = ep.get("role_names", {}) or {}
        drop = {int(roles[k]) for k in ("target_id", "destination_id") if roles.get(k) is not None}
        keep = [k for k in keep if k not in drop]
    return np.asarray(sorted(keep), dtype=np.int64)


def _episode_role_ids(scene_info_path: str, ep_idx: int) -> tuple[int, int]:
    ep = json.load(open(scene_info_path))[f"episode_{ep_idx}"]
    roles = ep.get("role_names", {}) or {}
    target_id = int(roles["target_id"]) if roles.get("target_id") is not None else -1
    dest_id = int(roles["destination_id"]) if roles.get("destination_id") is not None else -1
    return target_id, dest_id


def _stage_role_ids(cfg_dir: str, ep_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-frame target/object-destination ids from RoboPRO masking sidecars."""
    path = Path(cfg_dir) / "masking" / f"episode{ep_idx}.json"
    if not path.is_file():
        return None, None
    with path.open() as f:
        data = json.load(f)
    target_by_stage = {
        int(stage["stage"]): int(stage["target_id"])
        for stage in data.get("stages", [])
        if stage.get("target_id") is not None
    }
    dest_by_stage = {
        int(stage["stage"]): int(stage["bin"]["bin_id"])
        for stage in data.get("stages", [])
        if (stage.get("bin") or {}).get("bin_id") is not None
    }
    frame_stage = data.get("frame_stage")
    if not frame_stage:
        return None, None
    target = np.asarray([target_by_stage.get(int(s), -1) for s in frame_stage], dtype=np.int64)
    dest = np.asarray([dest_by_stage.get(int(s), -1) for s in frame_stage], dtype=np.int64)
    return target, dest


def _resize_with_pad_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """numpy/cv2 equivalent of openpi image_tools.resize_with_pad (nearest, pad=0).

    Matches the aspect-preserving + center-pad geometry the model applies to
    base_0_rgb so a baked mask lines up with the RGB tokens exactly.
    """
    cur_h, cur_w = mask.shape[:2]
    ratio = max(cur_h / height, cur_w / width)
    resized_h = max(1, int(cur_h / ratio))
    resized_w = max(1, int(cur_w / ratio))
    resized = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    pad_h0 = (height - resized_h) // 2
    pad_h1 = height - resized_h - pad_h0
    pad_w0 = (width - resized_w) // 2
    pad_w1 = width - resized_w - pad_w0
    return np.pad(resized, ((pad_h0, pad_h1), (pad_w0, pad_w1)), constant_values=0.0)


def _resize_rgb_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Aspect-preserving RGB resize matching `_resize_with_pad_nearest` geometry."""
    cur_h, cur_w = image.shape[:2]
    ratio = max(cur_h / height, cur_w / width)
    resized_h = max(1, int(cur_h / ratio))
    resized_w = max(1, int(cur_w / ratio))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((height, width, 3), dtype=resized.dtype)
    y0, x0 = (height - resized_h) // 2, (width - resized_w) // 2
    out[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return out


def _decode_seg_frame(seg_ds, frame_idx: int) -> np.ndarray:
    """Decode a PNG-encoded uint16 actor-id map for one frame -> [H, W] uint16."""
    buf = np.frombuffer(seg_ds[frame_idx], dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _decode_rgb_frame(rgb_ds, frame_idx: int) -> np.ndarray:
    buf = np.frombuffer(rgb_ds[frame_idx], dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _bbox_frame(raw_h5: h5py.File, frame_idx: int):
    """Return actor + articulation-link OBBs for one frame, accepting legacy AABBs."""

    def read_group(group):
        ids = group["id"][frame_idx]
        if all(k in group for k in ("obb_center", "obb_half", "obb_quat")):
            return (
                ids,
                group["obb_center"][frame_idx],
                group["obb_half"][frame_idx],
                group["obb_quat"][frame_idx],
            )
        mins, maxs = group["aabb_min"][frame_idx], group["aabb_max"][frame_idx]
        centers = 0.5 * (mins + maxs)
        halves = 0.5 * (maxs - mins)
        quats = np.zeros((*centers.shape[:-1], 4), dtype=np.float32)
        quats[..., 0] = 1.0
        return ids, centers, halves, quats

    tables = [read_group(raw_h5["actor_bbox"])]
    if "link_bbox" in raw_h5 and "id" in raw_h5["link_bbox"]:
        tables.append(read_group(raw_h5["link_bbox"]))
    return tuple(np.concatenate([table[i] for table in tables], axis=0) for i in range(4))


@dataclasses.dataclass
class _BetaCache:
    """Per-episode proximity weights and object-local contact anchors."""

    beta: np.ndarray | None  # [T, K] or None
    cols: dict | None  # actor_id -> column
    contact_beta: np.ndarray | None = None  # [T,K], centered-window contact risk
    obstacle_contact_local: np.ndarray | None = None  # [T,K,3], normalized OBB-local
    obstacle_contact_valid: np.ndarray | None = None  # [T,K]
    target_contact_id: np.ndarray | None = None  # [T], supports stage-resolved targets
    target_contact_local: np.ndarray | None = None  # [T,3] (legacy caches may use [3])
    target_contact_valid: np.ndarray | bool = False  # [T] or scalar

    @classmethod
    def load(
        cls,
        cfg_dir: str,
        ep_filename: str,
        beta_subdir: str = "beta_weights",
        beta_root: str | None = None,
        raw_dir: str | None = None,
    ) -> "_BetaCache":
        # When beta_root is set the raw tree is read-only and caches live under a
        # mirrored <beta_root>/<domain>/<task>/<config>/ layout; otherwise they sit
        # next to the HDF5s in <config>/<beta_subdir>/.
        if beta_root is not None:
            rel = os.path.relpath(cfg_dir, os.path.expanduser(str(raw_dir))) if raw_dir else os.path.basename(cfg_dir)
            beta_dir = os.path.join(os.path.expanduser(beta_root), rel, beta_subdir)
        else:
            beta_dir = os.path.join(cfg_dir, beta_subdir)
        beta_path = os.path.join(beta_dir, ep_filename.replace(".hdf5", ".npz"))
        if os.path.isfile(beta_path):
            with np.load(beta_path) as bz:
                return cls(
                    beta=bz["beta"].copy(),
                    cols={int(a): j for j, a in enumerate(bz["obj_ids"])},
                    contact_beta=bz["contact_beta"].copy() if "contact_beta" in bz else None,
                    obstacle_contact_local=(
                        bz["obstacle_contact_local"].copy() if "obstacle_contact_local" in bz else None
                    ),
                    obstacle_contact_valid=(
                        bz["obstacle_contact_valid"].copy() if "obstacle_contact_valid" in bz else None
                    ),
                    target_contact_id=(
                        bz["target_contact_id"].copy() if "target_contact_id" in bz else None
                    ),
                    target_contact_local=(
                        bz["target_contact_local"].copy() if "target_contact_local" in bz else None
                    ),
                    target_contact_valid=(
                        bz["target_contact_valid"].copy() if "target_contact_valid" in bz else False
                    ),
                )
        return cls(beta=None, cols=None)


def _contact_seed_mask(
    seg: np.ndarray,
    actor_id: int,
    uv: np.ndarray,
    *,
    radius_px: int,
    max_snap_px: float,
) -> np.ndarray | None:
    """Snap a projected contact point to actor pixels and splat an actor-clipped disk."""
    ys, xs = np.nonzero(seg == actor_id)
    if xs.size == 0 or not np.all(np.isfinite(uv)):
        return None
    d2 = (xs.astype(np.float64) - float(uv[0])) ** 2 + (ys.astype(np.float64) - float(uv[1])) ** 2
    nearest = int(np.argmin(d2))
    if d2[nearest] > max_snap_px**2:
        return None
    cx, cy = int(xs[nearest]), int(ys[nearest])
    yy, xx = np.ogrid[: seg.shape[0], : seg.shape[1]]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
    return (disk & (seg == actor_id)).astype(np.float32)


def _project_local_anchor(
    normalized_local: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    quat: np.ndarray,
    ext_cv: np.ndarray,
    intrinsic_cv: np.ndarray,
) -> np.ndarray | None:
    world = _bg.obb_normalized_to_world(normalized_local, center, half, quat).reshape(1, 3)
    uv, depth = _bg.project_world_to_image(world, ext_cv, intrinsic_cv)
    return uv[0] if depth[0] > 1e-6 else None


def _build_masks_for_frame(
    seg: np.ndarray,
    obj_ids: np.ndarray,
    target_id: int,
    dest_id: int,
    beta: _BetaCache,
    frame_idx: int,
    *,
    bbox_ids: np.ndarray | None = None,
    bbox_centers: np.ndarray | None = None,
    bbox_halves: np.ndarray | None = None,
    bbox_quats: np.ndarray | None = None,
    ext_cv: np.ndarray | None = None,
    intrinsic_cv: np.ndarray | None = None,
    attention_mask_mode: Literal["contact", "object"] = "contact",
    contact_radius_px: int = 8,
    max_snap_px: float = 48.0,
    stats: dict[str, int] | None = None,
) -> dict[str, np.ndarray]:
    """Build obstacle/beta/target/dest masks for one frame, resized to the model
    geometry (resize_with_pad, nearest). Returns float32 [H, W] masks."""
    object_obstacle = np.isin(seg, obj_ids).astype(np.float32)
    obstacle = object_obstacle.copy()

    beta_field = np.ones(seg.shape, dtype=np.float32)
    beta_values = (
        beta.contact_beta
        if attention_mask_mode == "contact" and beta.contact_beta is not None
        else beta.beta
    )
    if beta_values is not None and frame_idx < beta_values.shape[0]:
        row = beta_values[frame_idx]
        for aid, col in beta.cols.items():
            w = float(row[col])
            if w != 1.0:
                beta_field[seg == aid] = w

    def role_mask(aid: int) -> np.ndarray:
        return (seg == aid).astype(np.float32) if aid >= 0 else np.zeros(seg.shape, np.float32)

    if beta.target_contact_id is not None and frame_idx < beta.target_contact_id.shape[0]:
        cached_target_id = int(beta.target_contact_id[frame_idx])
        if cached_target_id >= 0:
            target_id = cached_target_id
    target = role_mask(target_id)
    geometry_ready = all(
        x is not None
        for x in (bbox_ids, bbox_centers, bbox_halves, bbox_quats, ext_cv, intrinsic_cv)
    )
    id_to_col = {int(a): i for i, a in enumerate(bbox_ids)} if geometry_ready else {}

    if attention_mask_mode == "contact":
        localized_obstacle = np.zeros(seg.shape, dtype=np.float32)
        for aid in obj_ids:
            actor_mask = role_mask(int(aid))
            contact = None
            cache_col = beta.cols.get(int(aid)) if beta.cols is not None else None
            bbox_col = id_to_col.get(int(aid))
            cache_valid = (
                cache_col is not None
                and beta.obstacle_contact_local is not None
                and frame_idx < beta.obstacle_contact_local.shape[0]
                and (
                    beta.obstacle_contact_valid is None
                    or bool(beta.obstacle_contact_valid[frame_idx, cache_col])
                )
            )
            if geometry_ready and bbox_col is not None and cache_valid:
                uv = _project_local_anchor(
                    beta.obstacle_contact_local[frame_idx, cache_col],
                    bbox_centers[bbox_col],
                    bbox_halves[bbox_col],
                    bbox_quats[bbox_col],
                    ext_cv,
                    intrinsic_cv,
                )
                if uv is not None:
                    contact = _contact_seed_mask(
                        seg, int(aid), uv, radius_px=contact_radius_px, max_snap_px=max_snap_px
                    )
            if contact is None:
                localized_obstacle = np.maximum(localized_obstacle, actor_mask)
                if stats is not None and actor_mask.any():
                    stats["obstacle_fallback"] = stats.get("obstacle_fallback", 0) + 1
            else:
                localized_obstacle = np.maximum(localized_obstacle, contact)
                if stats is not None:
                    stats["obstacle_contact"] = stats.get("obstacle_contact", 0) + 1
        obstacle = localized_obstacle

        target_contact = None
        target_col = id_to_col.get(target_id)
        target_local = beta.target_contact_local
        if target_local is not None and target_local.ndim == 2:
            target_local = target_local[frame_idx] if frame_idx < target_local.shape[0] else None
        target_valid = beta.target_contact_valid
        if isinstance(target_valid, np.ndarray):
            if target_valid.ndim == 0:
                target_valid = bool(target_valid)
            else:
                target_valid = bool(target_valid[frame_idx]) if frame_idx < target_valid.shape[0] else False
        if (
            geometry_ready
            and target_col is not None
            and target_valid
            and target_local is not None
        ):
            uv = _project_local_anchor(
                target_local,
                bbox_centers[target_col],
                bbox_halves[target_col],
                bbox_quats[target_col],
                ext_cv,
                intrinsic_cv,
            )
            if uv is not None:
                target_contact = _contact_seed_mask(
                    seg, target_id, uv, radius_px=contact_radius_px, max_snap_px=max_snap_px
                )
        if target_contact is not None:
            target = target_contact
            if stats is not None:
                stats["target_contact"] = stats.get("target_contact", 0) + 1
        elif target.any() and stats is not None:
            stats["target_fallback"] = stats.get("target_fallback", 0) + 1

    out = {
        "obstacle": obstacle,
        "beta": beta_field,
        "target": target,
        "dest": role_mask(dest_id),
    }
    return {k: _resize_with_pad_nearest(v, *MASK_RESOLUTION) for k, v in out.items()}


def _grounding_state_action(raw_h5: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Match process_data.py: state = [left_arm(6), left_gripper, right_arm(6), right_gripper];
    obs frames = 0..T-2, action[i] = state[i+1]."""
    left_arm = raw_h5["/joint_action/left_arm"][()]
    left_grip = raw_h5["/joint_action/left_gripper"][()]
    right_arm = raw_h5["/joint_action/right_arm"][()]
    right_grip = raw_h5["/joint_action/right_gripper"][()]
    T = left_arm.shape[0]
    states = np.concatenate(
        [left_arm, left_grip[:, None], right_arm, right_grip[:, None]], axis=1
    ).astype(np.float32)  # [T, 14]
    obs_states = states[:-1]  # [T-1, 14]
    actions = states[1:]  # [T-1, 14]
    return obs_states, actions


def create_empty_grounding_dataset(
    repo_id: str,
    robot_type: str = "aloha",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle",
        "left_wrist_rotate", "left_gripper", "right_waist", "right_shoulder", "right_elbow",
        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
    ]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
        "action": {"dtype": "float32", "shape": (len(motors),), "names": [motors]},
    }
    for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    # Obstacle-attention masks: float32, stored (not video-compressed) to preserve
    # exact beta_t values. Shape is (1, H, W) mirroring the channel-first image layout.
    for role in ("obstacle", "beta", "target", "dest"):
        features[f"observation.mask.{role}"] = {
            "dtype": "float32",
            "shape": (1, *MASK_RESOLUTION),
            "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _find_grounding_episodes(raw_dir: Path) -> list[str]:
    """Find <task>/<config>/data/episodeN.hdf5 files under raw_dir."""
    files = []
    for root, _, filenames in os.walk(raw_dir):
        if os.path.basename(root) != "data":
            continue
        for fn in fnmatch.filter(filenames, "episode*.hdf5"):
            files.append(os.path.join(root, fn))
    return sorted(files)


def _read_instruction(cfg_dir: str, ep_idx: int, desc_type: str = "seen") -> str:
    """Language for an episode.

    Preference order (covers both older demos and ``mzxuan/robopro_expert``):
      1. ``instructions/episode{N}.json`` sidecar (``seen`` / ``instructions`` lists)
      2. ``scene_info.json`` -> ``episode_N.language_perturbation.instruction``
      3. empty string
    """
    ins_path = os.path.join(cfg_dir, "instructions", f"episode{ep_idx}.json")
    if os.path.isfile(ins_path):
        with open(ins_path, "r") as f:
            d = json.load(f)
        instructions = d.get(desc_type) or d.get("instructions") or []
        if instructions:
            return str(np.random.choice(instructions))

    scene_path = os.path.join(cfg_dir, "scene_info.json")
    if os.path.isfile(scene_path):
        ep = json.load(open(scene_path)).get(f"episode_{ep_idx}", {})
        lang = (ep.get("language_perturbation") or {}).get("instruction")
        if lang:
            return str(lang)
    return ""


def _ensure_beta_weights(
    raw_dir: Path, beta_subdir: str, exclude_target: bool, beta_root: str | None = None
) -> None:
    """Run in-repo FK precompute so the per-episode beta caches exist.

    With ``beta_root`` set, ``raw_dir`` is treated as read-only and caches are
    written under ``<beta_root>/<domain>/<task>/<config>/<beta_subdir>/``; without
    it they land next to the HDF5s in ``<config>/<beta_subdir>/``.
    """
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from precompute_beta_weights import precompute_beta_weights  # noqa: PLC0415

    dest = beta_root if beta_root else raw_dir
    print(f"[beta] precomputing proximity weights (raw={raw_dir}, out under {dest}) -> */{beta_subdir}/")
    precompute_beta_weights(
        str(raw_dir),
        out_subdir=beta_subdir,
        beta_root=beta_root,
        exclude_target=exclude_target,
        overwrite=False,
    )


def port_grounding(
    raw_dir: Path,
    repo_id: str,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    seg_exclude_names: tuple[str, ...] = DEFAULT_EXCLUDE_NAMES,
    mask_exclude_target: bool = True,
    seg_camera: str = "countertop_camera",
    left_camera: str = "left_camera",
    right_camera: str = "right_camera",
    beta_subdir: str = "beta_weights",
    beta_root: str | None = None,
    precompute_beta: bool = False,
    attention_mask_mode: Literal["contact", "object"] = "contact",
    contact_radius_px: int = 8,
    max_contact_snap_px: float = 48.0,
    max_contact_fallback_fraction: float = 0.8,
    strict_contact_fallback: bool = False,
    contact_audit_dir: str | None = None,
    contact_audit_frames_per_episode: int = 3,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert raw RoboPRO expert HDF5 (e.g. ``mzxuan/robopro_expert``) to LeRobot
    with baked obstacle/beta/target/dest masks from countertop-camera segmentation.
    In ``contact`` mode, obstacle/target masks contain localized inferred contact
    regions; ``object`` mode reproduces the original full-object masks.

    Expected layout under ``raw_dir``:
      ``<domain>/<task>/<config>/data/episodeN.hdf5`` + ``scene_info.json``
    (also accepts the flatter ``<task>/<config>/...`` demo layout).

    If ``precompute_beta`` is True, runs ``scripts/precompute_beta_weights.py`` first
    (skipping episodes that already have a cache). Without a cache, beta masks are
    all-ones (no proximity weighting).

    ``beta_root`` keeps ``raw_dir`` read-only: beta caches are written to and read
    from ``<beta_root>/<domain>/<task>/<config>/<beta_subdir>/`` (mirroring the raw
    layout) instead of next to the source HDF5s. Leave it None for legacy behavior.
    """
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    hdf5_files = _find_grounding_episodes(raw_dir)
    if not hdf5_files:
        raise ValueError(
            f"No **/data/episode*.hdf5 found under {raw_dir}. "
            "Expected mzxuan/robopro_expert layout: <domain>/<task>/<config>/data/."
        )

    if precompute_beta:
        _ensure_beta_weights(raw_dir, beta_subdir, exclude_target=mask_exclude_target, beta_root=beta_root)

    dataset = create_empty_grounding_dataset(repo_id, dataset_config=dataset_config)

    ep_indices = episodes if episodes is not None else range(len(hdf5_files))
    for ep_idx in tqdm.tqdm(ep_indices):
        ep_path = hdf5_files[ep_idx]
        cfg_dir = os.path.dirname(os.path.dirname(ep_path))  # .../<config>
        file_ep_idx = _grounding_episode_idx(ep_path)
        scene_info_path = os.path.join(cfg_dir, "scene_info.json")

        obj_ids = _object_actor_ids(
            scene_info_path, file_ep_idx, seg_exclude_names, exclude_roles=mask_exclude_target
        )
        target_id, dest_id = _episode_role_ids(scene_info_path, file_ep_idx)
        stage_target_ids, stage_dest_ids = _stage_role_ids(cfg_dir, file_ep_idx)
        if target_id < 0 and stage_target_ids is not None and np.any(stage_target_ids >= 0):
            target_id = int(stage_target_ids[stage_target_ids >= 0][0])
        if dest_id < 0 and stage_dest_ids is not None and np.any(stage_dest_ids >= 0):
            dest_id = int(stage_dest_ids[stage_dest_ids >= 0][0])
        if mask_exclude_target:
            role_ids = [
                ids[ids >= 0]
                for ids in (stage_target_ids, stage_dest_ids)
                if ids is not None
            ]
            if role_ids:
                obj_ids = np.setdiff1d(obj_ids, np.unique(np.concatenate(role_ids)))
        beta = _BetaCache.load(
            cfg_dir, os.path.basename(ep_path), beta_subdir, beta_root=beta_root, raw_dir=str(raw_dir)
        )
        if mask_exclude_target and beta.target_contact_id is not None:
            obj_ids = np.setdiff1d(
                obj_ids, np.unique(beta.target_contact_id[beta.target_contact_id >= 0])
            )
        instruction = _read_instruction(cfg_dir, file_ep_idx)
        contact_stats: dict[str, int] = {}

        with h5py.File(ep_path, "r") as raw_h5:
            obs_states, actions = _grounding_state_action(raw_h5)
            num_frames = obs_states.shape[0]
            audit_frames = (
                set(np.linspace(0, num_frames - 1, contact_audit_frames_per_episode, dtype=int))
                if contact_audit_dir and contact_audit_frames_per_episode > 0
                else set()
            )
            seg_ds = raw_h5[f"observation/{seg_camera}/actor_segmentation"]
            cam_high_ds = raw_h5[f"observation/{seg_camera}/rgb"]
            cam_left_ds = raw_h5[f"observation/{left_camera}/rgb"]
            cam_right_ds = raw_h5[f"observation/{right_camera}/rgb"]

            for i in range(num_frames):
                # RGB: BGR->RGB, resize to (640,480) matching process_data.py, then CHW.
                def _rgb(ds, idx):
                    img = _decode_rgb_frame(ds, idx)  # BGR [H, W, 3]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (640, 480))
                    return np.transpose(img, (2, 0, 1))  # CHW uint8

                seg = _decode_seg_frame(seg_ds, i)
                bbox_ids = bbox_centers = bbox_halves = bbox_quats = ext_cv = intrinsic_cv = None
                if attention_mask_mode == "contact":
                    try:
                        bbox_ids, bbox_centers, bbox_halves, bbox_quats = _bbox_frame(raw_h5, i)
                        ext_cv = raw_h5[f"observation/{seg_camera}/extrinsic_cv"][i]
                        intrinsic_cv = raw_h5[f"observation/{seg_camera}/intrinsic_cv"][i]
                    except KeyError:
                        # Old episodes remain convertible via the full-object fallback.
                        pass
                masks = _build_masks_for_frame(
                    seg,
                    obj_ids,
                    int(stage_target_ids[i]) if stage_target_ids is not None and i < len(stage_target_ids) else target_id,
                    int(stage_dest_ids[i]) if stage_dest_ids is not None and i < len(stage_dest_ids) else dest_id,
                    beta,
                    i,
                    bbox_ids=bbox_ids,
                    bbox_centers=bbox_centers,
                    bbox_halves=bbox_halves,
                    bbox_quats=bbox_quats,
                    ext_cv=ext_cv,
                    intrinsic_cv=intrinsic_cv,
                    attention_mask_mode=attention_mask_mode,
                    contact_radius_px=contact_radius_px,
                    max_snap_px=max_contact_snap_px,
                    stats=contact_stats,
                )
                if any(not np.all(np.isfinite(mask)) for mask in masks.values()):
                    raise ValueError(f"{ep_path} frame {i}: non-finite attention mask")
                if i in audit_frames:
                    rgb_native = cv2.cvtColor(_decode_rgb_frame(cam_high_ds, i), cv2.COLOR_BGR2RGB)
                    overlay = _resize_rgb_with_pad(rgb_native, *MASK_RESOLUTION).astype(np.float32)
                    obstacle_alpha = np.clip(masks["obstacle"], 0.0, 1.0)[..., None]
                    target_alpha = np.clip(masks["target"], 0.0, 1.0)[..., None]
                    overlay = overlay * (1.0 - 0.55 * np.maximum(obstacle_alpha, target_alpha))
                    overlay += obstacle_alpha * np.array([140.0, 0.0, 0.0])
                    overlay += target_alpha * np.array([0.0, 140.0, 0.0])
                    audit_root = Path(contact_audit_dir).expanduser()
                    audit_root.mkdir(parents=True, exist_ok=True)
                    rel_cfg = Path(os.path.relpath(cfg_dir, raw_dir))
                    audit_path = audit_root / ("__".join(rel_cfg.parts) + f"__ep{file_ep_idx}_f{i:04d}.png")
                    cv2.imwrite(
                        str(audit_path),
                        cv2.cvtColor(np.clip(overlay, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    )

                frame = {
                    "observation.state": obs_states[i],
                    "action": actions[i],
                    "task": instruction,
                    "observation.images.cam_high": _rgb(cam_high_ds, i),
                    "observation.images.cam_left_wrist": _rgb(cam_left_ds, i),
                    "observation.images.cam_right_wrist": _rgb(cam_right_ds, i),
                }
                for role, m in masks.items():
                    frame[f"observation.mask.{role}"] = m[None].astype(np.float32)  # (1, H, W)
                dataset.add_frame(frame)

        if attention_mask_mode == "contact":
            print(f"[contact-audit] {os.path.relpath(ep_path, raw_dir)}: {contact_stats}")
            for role in ("obstacle", "target"):
                good = contact_stats.get(f"{role}_contact", 0)
                fallback = contact_stats.get(f"{role}_fallback", 0)
                total = good + fallback
                fraction = fallback / total if total else 0.0
                if total and fraction > max_contact_fallback_fraction:
                    # High fallback = this episode's contacts couldn't be projected onto
                    # the seg camera, so its obstacle/target masks are full-object (still
                    # beta-weighted) rather than localized. By default warn and keep the
                    # episode; pass --strict-contact-fallback to abort instead (useful for
                    # catching a systemic cache/projection/camera regression early).
                    message = (
                        f"{os.path.relpath(ep_path, raw_dir)}: {role} contact fallback fraction "
                        f"{fraction:.1%} exceeds {max_contact_fallback_fraction:.1%}; using "
                        "full-object masks for this episode (inspect cache/projection/camera "
                        "if widespread)"
                    )
                    if strict_contact_fallback:
                        raise ValueError(message + " [--strict-contact-fallback]")
                    print(f"[warn] {message}", file=sys.stderr)
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "aloha": port_aloha,
            "grounding": port_grounding,
        }
    )
