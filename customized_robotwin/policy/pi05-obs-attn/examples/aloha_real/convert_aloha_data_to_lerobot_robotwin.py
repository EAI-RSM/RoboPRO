"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>

Obstacle-attention variant — primary source is the HuggingFace raw expert dump
``mzxuan/robopro_expert`` (layout ``<domain>/<task>/<config>/data/episodeN.hdf5`` +
``scene_info.json``; actor boxes are object-aligned ``obb_*``). Downloads e.g.:

    huggingface-cli download mzxuan/robopro_expert --repo-type dataset \\
        --local-dir /path/to/robopro_expert

Then bake obstacle/beta/target/dest masks into a LeRobot dataset. Beta proximity
weights come from in-repo ``scripts/precompute_beta_weights.py``; pass
``--precompute-beta`` to run that first if caches are missing:

    uv run python scripts/precompute_beta_weights.py --data-root /path/to/robopro_expert

    uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py grounding \\
        --raw-dir /path/to/robopro_expert --repo-id local/robopro_expert \\
        --precompute-beta
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


def _decode_seg_frame(seg_ds, frame_idx: int) -> np.ndarray:
    """Decode a PNG-encoded uint16 actor-id map for one frame -> [H, W] uint16."""
    buf = np.frombuffer(seg_ds[frame_idx], dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _decode_rgb_frame(rgb_ds, frame_idx: int) -> np.ndarray:
    buf = np.frombuffer(rgb_ds[frame_idx], dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@dataclasses.dataclass
class _BetaCache:
    """Per-episode beta_t proximity weights from scripts/precompute_beta_weights.py."""

    beta: np.ndarray | None  # [T, K] or None
    cols: dict | None  # actor_id -> column

    @classmethod
    def load(cls, cfg_dir: str, ep_filename: str, beta_subdir: str = "beta_weights") -> "_BetaCache":
        beta_path = os.path.join(cfg_dir, beta_subdir, ep_filename.replace(".hdf5", ".npz"))
        if os.path.isfile(beta_path):
            bz = np.load(beta_path)
            return cls(beta=bz["beta"], cols={int(a): j for j, a in enumerate(bz["obj_ids"])})
        return cls(beta=None, cols=None)


def _build_masks_for_frame(
    seg: np.ndarray,
    obj_ids: np.ndarray,
    target_id: int,
    dest_id: int,
    beta: _BetaCache,
    frame_idx: int,
) -> dict[str, np.ndarray]:
    """Build obstacle/beta/target/dest masks for one frame, resized to the model
    geometry (resize_with_pad, nearest). Returns float32 [H, W] masks."""
    obstacle = np.isin(seg, obj_ids).astype(np.float32)

    beta_field = np.ones(seg.shape, dtype=np.float32)
    if beta.beta is not None and frame_idx < beta.beta.shape[0]:
        row = beta.beta[frame_idx]
        for aid, col in beta.cols.items():
            w = float(row[col])
            if w != 1.0:
                beta_field[seg == aid] = w

    def role_mask(aid: int) -> np.ndarray:
        return (seg == aid).astype(np.float32) if aid >= 0 else np.zeros(seg.shape, np.float32)

    out = {
        "obstacle": obstacle,
        "beta": beta_field,
        "target": role_mask(target_id),
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


def _ensure_beta_weights(raw_dir: Path, beta_subdir: str, exclude_target: bool) -> None:
    """Run in-repo FK precompute so `<config>/<beta_subdir>/episodeN.npz` exists."""
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from precompute_beta_weights import precompute_beta_weights  # noqa: PLC0415

    print(f"[beta] precomputing proximity weights under {raw_dir} -> */{beta_subdir}/")
    precompute_beta_weights(
        str(raw_dir),
        out_subdir=beta_subdir,
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
    precompute_beta: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert raw RoboPRO expert HDF5 (e.g. ``mzxuan/robopro_expert``) to LeRobot
    with baked obstacle/beta/target/dest masks from countertop-camera segmentation.

    Expected layout under ``raw_dir``:
      ``<domain>/<task>/<config>/data/episodeN.hdf5`` + ``scene_info.json``
    (also accepts the flatter ``<task>/<config>/...`` demo layout).

    If ``precompute_beta`` is True, runs ``scripts/precompute_beta_weights.py`` first
    (skipping episodes that already have a cache). Without a cache, beta masks are
    all-ones (no proximity weighting).
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
        _ensure_beta_weights(raw_dir, beta_subdir, exclude_target=mask_exclude_target)

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
        beta = _BetaCache.load(cfg_dir, os.path.basename(ep_path), beta_subdir)
        instruction = _read_instruction(cfg_dir, file_ep_idx)

        with h5py.File(ep_path, "r") as raw_h5:
            obs_states, actions = _grounding_state_action(raw_h5)
            num_frames = obs_states.shape[0]
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
                masks = _build_masks_for_frame(seg, obj_ids, target_id, dest_id, beta, i)

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
