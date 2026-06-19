"""Convert collision-RELABELED episodes -> a plain LeRobot dataset for guided BC.

Companion to relabel_actions_jax.py.  Reads the ``<task>/cluttered/<source>_safe``
legs (episodes carrying ``joint_action/vector_safe``) and writes a STANDARD pi05
BC dataset — no advantage, no link_distances, just observation/action — so you
can train with the ordinary training script.  This is the "offline-relabel +
regular BC" path (distinct from convert_collision_to_lerobot.py, which builds
the in-loop collision-critic dataset).

Guided-BC convention (mirrors convert_rollout_to_lerobot.py):
    observation.state[t] = joint_action/vector[t]        (original — matches images)
    action[t]            = joint_action/vector_safe[t+1] (collision-free next config)
The loader windows ``action`` into the H-step chunk of safe next configs, so the
policy learns collision-free motion from the real observed situations.

Usage (pi05 uv env, from customized_robotwin/policy/pi05/):
    # 1) relabel each task first:
    #    uv run scripts/relabel_actions_jax.py <root>/<task>/cluttered
    # 2) convert all relabeled legs into one dataset:
    uv run scripts/convert_relabeled_to_lerobot.py \\
        --root ../../collision_dataset --source pi05_rollout_safe \\
        --repo-id local/kitchenl_d15_collision_safe [--fps 50] [--require-success]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

CAM_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
MOTORS = [
    "left_waist", "left_shoulder", "left_elbow",
    "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate", "left_gripper",
    "right_waist", "right_shoulder", "right_elbow",
    "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper",
]

# Wrong-object-grasp episodes (task, seed) — excluded (same list as convert_collision_to_lerobot).
DEFAULT_WRONG_GRASP = [
    ("put_bottle_in_basket", 100010),
    ("put_bottle_in_fridge", 100030),
    ("put_bottle_in_fridge", 100061),
    ("put_sauce_can_in_basket", 100051),
]


def create_empty_dataset(repo_id: str, fps: int) -> LeRobotDataset:
    dest = HF_LEROBOT_HOME / repo_id
    if dest.exists():
        shutil.rmtree(dest)
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (len(MOTORS),), "names": [MOTORS]},
    }
    for cam in CAM_MAP.values():
        features[f"observation.images.{cam}"] = {
            "dtype": "image", "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id, fps=fps, robot_type="aloha", features=features, use_videos=False,
    )


def load_episode(ep_path: Path):
    with h5py.File(ep_path, "r") as f:
        if "joint_action/vector_safe" not in f:
            raise KeyError(f"{ep_path.name} missing joint_action/vector_safe — run relabel first.")
        state = np.asarray(f["joint_action/vector"][:], dtype=np.float32)        # original (T,14)
        safe = np.asarray(f["joint_action/vector_safe"][:], dtype=np.float32)    # safe traj (T,14)
        raw = f.attrs.get("instruction", b"")
        instr = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
        success = bool(f.attrs.get("success", True))
        images = {}
        for hcam, lcam in CAM_MAP.items():
            key = f"observation/{hcam}/rgb"
            if key in f:
                frames = []
                for data in f[key][:]:
                    arr = np.frombuffer(bytes(data).rstrip(b"\x00"), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    img = (cv2.resize(img, (640, 480)) if img is not None
                           else np.zeros((480, 640, 3), dtype=np.uint8))
                    frames.append(img)
                images[lcam] = frames
    return state, safe, images, instr, success


def main():
    ap = argparse.ArgumentParser(description="Relabeled collision HDF5 -> plain LeRobot (guided BC)")
    ap.add_argument("--root", type=Path, required=True, help="collision_dataset root")
    ap.add_argument("--repo-id", type=str, required=True)
    ap.add_argument("--source", default="pi05_rollout_safe", help="relabeled leg subdir name")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task names (default: all)")
    ap.add_argument("--wrong-grasp-json", type=str, default=None)
    ap.add_argument("--require-success", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="convert at most N episodes (0=all)")
    args = ap.parse_args()

    if args.wrong_grasp_json:
        wg = json.load(open(args.wrong_grasp_json))["wrong_grasp"]
        exclude = {(e["task"], int(e["seed"])) for e in wg}
    else:
        exclude = set(DEFAULT_WRONG_GRASP)
    print(f"Excluding {len(exclude)} wrong-grasp episodes: {sorted(exclude)}")

    tasks = args.tasks or sorted(p.name for p in args.root.iterdir() if p.is_dir())
    dataset = create_empty_dataset(args.repo_id, args.fps)

    n_used = n_skip = 0
    for task in tasks:
        leg = args.root / task / "cluttered" / args.source
        if not leg.is_dir():
            continue
        for ep_path in sorted(leg.glob("episode_seed*.hdf5"),
                              key=lambda p: int(p.stem.split("seed")[1])):
            seed = int(ep_path.stem.split("seed")[1])
            if (task, seed) in exclude:
                n_skip += 1
                continue
            state, safe, images, instr, success = load_episode(ep_path)
            if args.require_success and not success:
                n_skip += 1
                continue
            T = len(state)
            if T < 2:
                n_skip += 1
                continue
            actions = np.concatenate([safe[1:], safe[-1:]], axis=0)   # safe next-step target
            for t in range(T):
                frame = {
                    "observation.state": state[t],     # original obs (matches images)
                    "action": actions[t],              # safe next config
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
    print(f"\nDone. episodes_used={n_used}  skipped={n_skip}  frames={dataset.num_frames}")
    print(f"Saved to {HF_LEROBOT_HOME / args.repo_id}")


if __name__ == "__main__":
    main()
