"""Convert the collision dataset -> a LeRobot dataset for adjoint-matching (AM).

Minimal AM dataset: the ORIGINAL successful demonstrations (BC anchor) PLUS a
per-frame ``obstacle_points`` column carrying that episode's scene obstacle
surface points (static, world frame, excluded-tags removed, voxel-downsampled,
prefiltered to the arm workspace, padded to a fixed M with a far sentinel).

The AM trainer keeps the base policy doing BC on these original actions while a
LoRA "fast" field is trained by adjoint matching against the differentiable
proximity reward evaluated on ``obstacle_points`` (collision_proximity.py).

Columns:
    observation.state        original measured qpos (T,14)
    observation.images.*      cam_high / cam_left_wrist / cam_right_wrist
    action                    original next-step joint config (BC target)
    obstacle_points           (M*3,) flattened per-episode obstacle points
    task                      language instruction

Usage (pi05 uv env, from customized_robotwin/policy/pi05/):
    uv run scripts/convert_collision_am_to_lerobot.py \\
        --root ../../collision_dataset --repo-id local/kitchenl_d15_am \\
        [--M 512] [--margin 0.03] [--fps 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import jax.numpy as jnp
import numpy as np
import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_actions_jax import load_obstacle_points, prefilter_points, voxel_downsample  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openpi.training.collision_fk import make_sphere_fn  # noqa: E402

CAM_MAP = {"head_camera": "cam_high", "left_camera": "cam_left_wrist", "right_camera": "cam_right_wrist"}
MOTORS = [
    "left_waist", "left_shoulder", "left_elbow",
    "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow",
    "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]
DEFAULT_WRONG_GRASP = [
    ("put_bottle_in_basket", 100010), ("put_bottle_in_fridge", 100030),
    ("put_bottle_in_fridge", 100061), ("put_sauce_can_in_basket", 100051),
]
FAR = 1e6   # padding sentinel: far points never become the nearest -> zero reward contribution


def pad_or_subsample(P: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    """-> (M,3): subsample if too many, pad with a far sentinel if too few."""
    if len(P) >= M:
        idx = rng.choice(len(P), M, replace=False)
        return P[idx].astype(np.float32)
    out = np.full((M, 3), FAR, np.float32)
    out[: len(P)] = P
    return out


def episode_obstacle_points(ep_path: Path, scene_root: Path, sphere_fn, margin, M, rng) -> np.ndarray:
    with h5py.File(ep_path, "r") as f:
        q0 = f["joint_action/vector"][:].astype(np.float32)
        sa = f.attrs.get("scene_dir", "")
        sa = sa.decode() if isinstance(sa, bytes) else str(sa)
        seed = int(f.attrs.get("seed", -1))
    scene_dir = (scene_root / sa) if sa else (scene_root / "scene" / f"seed_{seed}")
    if not (scene_dir / "scene.npz").exists():
        scene_dir = scene_root / "scene" / f"seed_{seed}"
    if not (scene_dir / "scene.npz").exists():
        # Scene not exported for this seed (e.g. expert-blocked) — keep the episode
        # for BC but with no obstacles (all-far padding ⇒ zero proximity reward).
        return pad_or_subsample(np.empty((0, 3), np.float32), M, rng)
    P = load_obstacle_points(scene_dir, {"target", "container", "furniture", "skipped"})
    if len(P):
        centers0 = np.asarray(sphere_fn(jnp.asarray(q0)))
        buffer = float(0.30 + margin)
        P = prefilter_points(P, centers0, buffer)
        P = voxel_downsample(P, 0.02)
    return pad_or_subsample(P, M, rng)


def load_demo(ep_path: Path):
    with h5py.File(ep_path, "r") as f:
        state = f["joint_action/vector"][:].astype(np.float32)
        raw = f.attrs.get("instruction", b"")
        instr = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
        success = bool(f.attrs.get("success", False))
        images = {}
        for hcam, lcam in CAM_MAP.items():
            key = f"observation/{hcam}/rgb"
            if key in f:
                frames = []
                for data in f[key][:]:
                    arr = np.frombuffer(bytes(data).rstrip(b"\x00"), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (640, 480)) if img is not None else np.zeros((480, 640, 3), np.uint8)
                    frames.append(img)
                images[lcam] = frames
    return state, images, instr, success


def create_empty_dataset(repo_id: str, fps: int, M: int) -> LeRobotDataset:
    dest = HF_LEROBOT_HOME / repo_id
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "obstacle_points": {"dtype": "float32", "shape": (M * 3,), "names": None},
    }
    for lcam in CAM_MAP.values():
        features[f"observation.images.{lcam}"] = {
            "dtype": "image", "shape": (3, 480, 640), "names": ["channels", "height", "width"]}
    return LeRobotDataset.create(repo_id=repo_id, fps=fps, robot_type="aloha",
                                 features=features, use_videos=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--repo-id", type=str, required=True)
    ap.add_argument("--source", default="pi05_rollout")
    ap.add_argument("--M", type=int, default=512, help="obstacle points per episode (pad/subsample)")
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--wrong-grasp-json", type=str, default=None,
                    help="JSON {'wrong_grasp':[{task,seed},...]} to exclude (default: built-in list)")
    ap.add_argument("--require-success", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.wrong_grasp_json:
        wg = json.load(open(args.wrong_grasp_json))["wrong_grasp"]
        exclude = {(e["task"], int(e["seed"])) for e in wg}
    else:
        exclude = set(DEFAULT_WRONG_GRASP)
    print(f"Excluding {len(exclude)} wrong-grasp episodes")
    rng = np.random.default_rng(0)
    tasks = args.tasks or sorted(p.name for p in args.root.iterdir() if p.is_dir())
    dataset = create_empty_dataset(args.repo_id, args.fps, args.M)
    n_used = n_skip = 0
    for task in tasks:
        fk_basis = args.root / task / "cluttered" / "fk_basis.npz"
        leg = args.root / task / "cluttered" / args.source
        if not (leg.is_dir() and fk_basis.exists()):
            continue
        sphere_fn, _ = make_sphere_fn(str(fk_basis))
        task_root = args.root / task / "cluttered"
        for ep in sorted(leg.glob("episode_seed*.hdf5"), key=lambda p: int(p.stem.split("seed")[1])):
            seed = int(ep.stem.split("seed")[1])
            if (task, seed) in exclude:
                n_skip += 1
                continue
            state, images, instr, success = load_demo(ep)
            if (args.require_success and not success) or len(state) < 2:
                n_skip += 1
                continue
            pts = episode_obstacle_points(ep, task_root, sphere_fn, args.margin, args.M, rng).reshape(-1)
            actions = np.concatenate([state[1:], state[-1:]], axis=0)
            for t in range(len(state)):
                frame = {
                    "observation.state": state[t],
                    "action": actions[t],
                    "obstacle_points": pts,            # same per-episode points each frame
                    "task": instr,
                }
                for lcam, frames in images.items():
                    if t < len(frames):
                        frame[f"observation.images.{lcam}"] = frames[t].transpose(2, 0, 1)
                dataset.add_frame(frame)
            dataset.save_episode()
            n_used += 1
            if args.limit and n_used >= args.limit:
                break
        if args.limit and n_used >= args.limit:
            break

    if hasattr(dataset, "consolidate"):
        dataset.consolidate(run_compute_stats=True)
    print(f"\nDone. episodes_used={n_used} skipped={n_skip} frames={dataset.num_frames} M={args.M}")
    print(f"Saved to {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    main()
