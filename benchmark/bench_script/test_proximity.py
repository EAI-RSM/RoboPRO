"""
Test script for proximity (SDF) tracking: load a bench env with proximity_tracking enabled,
run play_once(), save video + per-step log + episode-min summary.

USAGE:
    cd customized_robotwin
    source set_env.sh
    python $BENCH_ROOT/bench_script/test_proximity.py <task_name> <task_config> [options]

EXAMPLES:
    python $BENCH_ROOT/bench_script/test_proximity.py put_bottle_in_fridge bench_demo_kitchenl_clean --bench-subdir kitchenl
    python $BENCH_ROOT/bench_script/test_proximity.py put_mouse_on_pad bench_demo_clean --bench-subdir office --seed 42
    python $BENCH_ROOT/bench_script/test_proximity.py move_cup bench_demo_clean --bench-subdir study --num-episodes 3

NOTE: The task config may include proximity_tracking, or use --parts to set robot_parts at CLI level.

OUTPUTS (default dir: benchmark/proximity_test/):
    - <prefix>_ep<N>.mp4                        : Video per episode
    - <prefix>_ep<N>_proximity_log.json         : Per-action proximity values
    - <prefix>_ep<N>_proximity_metrics.json     : Episode-minimum summary
"""
import sys
import os
import argparse
import importlib
import yaml
import json
from pathlib import Path
import numpy as np


def _resolve_repo_paths():
    script_path = Path(__file__).resolve()
    default_bench_root = script_path.parent.parent
    workspace_root = default_bench_root.parent

    def _valid_bench_root(p):
        return (p / "bench_envs").is_dir() and (p / "bench_task_config").is_dir()

    def _valid_robotwin_root(p):
        return (p / "envs").is_dir() and (p / "task_config" / "_embodiment_config.yml").exists()

    bench_root = None
    for c in [os.environ.get("BENCH_ROOT"), default_bench_root]:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if _valid_bench_root(p):
            bench_root = p
            break

    robotwin_root = None
    for c in [os.environ.get("ROBOTWIN_ROOT"), workspace_root / "customized_robotwin"]:
        if not c:
            continue
        p = Path(c).expanduser().resolve()
        if _valid_robotwin_root(p):
            robotwin_root = p
            break

    if bench_root is None or robotwin_root is None:
        raise SystemExit(
            "Could not resolve BENCH_ROOT/customized_robotwin. "
            "Expected benchmark/ and customized_robotwin/ to be sibling folders."
        )

    os.environ["BENCH_ROOT"] = str(bench_root)
    os.environ["ROBOTWIN_ROOT"] = str(robotwin_root)
    return bench_root, robotwin_root


bench_root, robotwin_root = _resolve_repo_paths()

for p in [robotwin_root, bench_root]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

os.chdir(robotwin_root)

from envs import CONFIGS_PATH


def _extract_task_class(envs_module, task_name):
    try:
        return getattr(envs_module, task_name)
    except AttributeError:
        from envs._base_task import Base_Task
        for name in dir(envs_module):
            obj = getattr(envs_module, name)
            if isinstance(obj, type) and issubclass(obj, Base_Task) and obj is not Base_Task:
                return obj
        raise SystemExit(f"No task class found in {envs_module.__name__}")


def get_env_class(task_name, bench_subdir=None):
    BENCH_SUBDIRS = ["office", "study", "kitchenl", "kitchens"]

    if bench_subdir:
        try:
            m = importlib.import_module(f"bench_envs.{bench_subdir}.{task_name}")
            return _extract_task_class(m, task_name)
        except ModuleNotFoundError:
            raise SystemExit(f"Task '{task_name}' not found in bench_envs.{bench_subdir}")

    try:
        m = importlib.import_module(f"bench_envs.{task_name}")
        return _extract_task_class(m, task_name)
    except ModuleNotFoundError:
        pass

    for subdir in BENCH_SUBDIRS:
        try:
            m = importlib.import_module(f"bench_envs.{subdir}.{task_name}")
            return _extract_task_class(m, task_name)
        except ModuleNotFoundError:
            continue

    try:
        m = importlib.import_module(f"envs.{task_name}")
        return _extract_task_class(m, task_name)
    except ModuleNotFoundError:
        raise SystemExit(f"No task class found for '{task_name}' in bench_envs or envs")


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def patch_take_picture(env, proximity_log, frame_list, class_compute_proximity):
    """
    Replace env._take_picture with a version that:
    - calls _compute_proximity_step() at every save_freq tick (bypasses save_data gate)
    - captures a camera frame at the same tick

    This is the correct granularity: _take_picture fires at every save_freq step
    throughout the full trajectory, giving a complete per-step time series.
    """
    def patched_take_picture():
        # Proximity
        prox = class_compute_proximity(env)
        if prox:
            entry = {
                part: {
                    "top_k": [
                        {
                            "dist":  float(vals["top_k_dist"][i]),
                            "delta": vals["top_k_delta"][i].tolist(),
                            "name":  vals["top_k_names"][i],
                        }
                        for i in range(len(vals["top_k_names"]))
                    ]
                }
                for part, vals in prox.items()
            }
            proximity_log.append(entry)

        # Frame capture
        try:
            env.cameras.update_picture()
            rgb_dict = env.cameras.get_rgb()
            cam_name = next(
                (c for c in ("demo_camera", "countertop_camera", "head_camera") if c in rgb_dict), None
            )
            if cam_name is not None:
                img = rgb_dict[cam_name]["rgb"].copy()
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                frame_list.append(img)
        except Exception:
            pass

    env._take_picture = patched_take_picture


def run_episode(env, cfg, ep_idx, output_dir, run_prefix, class_compute_proximity):
    ep_cfg = dict(cfg)
    ep_cfg["now_ep_num"] = ep_idx
    ep_cfg["seed"] = cfg["seed"] + ep_idx

    print(f"\n{'='*60}")
    print(f"Episode {ep_idx}  (seed={ep_cfg['seed']})")
    print(f"{'='*60}")

    env.setup_demo(**ep_cfg)

    if not getattr(env, "_proximity_enabled", False):
        raise SystemExit(
            "Proximity tracking was not initialized — ensure the task inherits from "
            "a scene base task that supports proximity_tracking."
        )

    proximity_log = []
    frame_list = []
    patch_take_picture(env, proximity_log, frame_list, class_compute_proximity)

    env.play_once()
    print(f"Episode {ep_idx} done. Steps captured: {len(proximity_log)}, frames: {len(frame_list)}")

    metrics = env.get_proximity_metrics()
    metrics["task_success"] = bool(env.check_success())
    metrics["episode"] = ep_idx
    metrics["seed"] = ep_cfg["seed"]

    ep_prefix = f"{run_prefix}_ep{ep_idx}"

    log_path = output_dir / f"{ep_prefix}_proximity_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(proximity_log), f, indent=2)
    print(f"Saved log: {log_path} ({len(proximity_log)} steps)")

    metrics_path = output_dir / f"{ep_prefix}_proximity_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(metrics), f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    print(f"\n--- Episode {ep_idx} Proximity Summary ---")
    for part, vals in metrics.items():
        if isinstance(vals, dict) and "min_dist" in vals:
            dist = vals["min_dist"]
            delta = vals["delta"]
            name = vals.get("closest_name", "")
            print(f"  {part}: min_dist={dist:.4f} m  closest='{name}'  delta={[f'{v:.4f}' for v in delta]}")
        else:
            print(f"  {part}: {vals}")

    if proximity_log:
        parts = list(proximity_log[0].keys())
        n_show = min(10, len(proximity_log))
        print(f"\n  Per-step proximity (first {n_show} of {len(proximity_log)}, top-1 shown):")
        header = f"  {'step':>6}" + "".join(f"  {p+':dist[0]':>16}  {p+':name[0]':>20}" for p in parts)
        print(header)
        for i, entry in enumerate(proximity_log[:n_show]):
            row = f"  {i:>6}"
            for p in parts:
                top_k = entry.get(p, {}).get("top_k", [{}])
                d = top_k[0].get("dist", -1.0) if top_k else -1.0
                n = top_k[0].get("name", "")  if top_k else ""
                row += f"  {d:>16.4f}  {n:>20}"
            print(row)
        if len(proximity_log) > n_show:
            print(f"  ... ({len(proximity_log) - n_show} more in log file)")

    # Debug frames: 20 evenly-sampled steps, each annotated with distance + closest name
    if frame_list and proximity_log:
        n_samples = min(20, len(frame_list))
        indices = np.linspace(0, len(frame_list) - 1, n_samples, dtype=int)
        frames_dir = output_dir / f"{ep_prefix}_frames"
        frames_dir.mkdir(exist_ok=True)
        import cv2
        parts_in_log = [k for k in proximity_log[0].keys()]
        for rank, idx in enumerate(indices):
            img = frame_list[idx].copy()
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            # OpenCV expects BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Build annotation lines
            lines = [f"step {idx}/{len(frame_list)-1}"]
            prox = proximity_log[idx] if idx < len(proximity_log) else {}
            for p in parts_in_log:
                top_k = prox.get(p, {}).get("top_k", [])
                for rank, obj in enumerate(top_k):
                    d = obj.get("dist", -1.0)
                    name = obj.get("name", "")
                    delta = obj.get("delta", [0.0, 0.0, 0.0])
                    lines.append(f"{p}[{rank}]: {d:.3f}m -> {name}")
                    lines.append(f"  d=({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})")
            # Draw semi-transparent background box
            line_h, pad = 22, 6
            box_h = len(lines) * line_h + pad * 2
            box_w = max(len(l) for l in lines) * 10 + pad * 2
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0, img_bgr)
            for i, line in enumerate(lines):
                y = 5 + pad + (i + 1) * line_h - 4
                cv2.putText(img_bgr, line, (5 + pad, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 1, cv2.LINE_AA)
            frame_path = frames_dir / f"frame_{rank:02d}_step{idx:04d}.png"
            cv2.imwrite(str(frame_path), img_bgr)
        print(f"Saved {n_samples} debug frames: {frames_dir}/")

    video_path = output_dir / f"{ep_prefix}.mp4"
    if frame_list:
        try:
            import imageio
            writer = imageio.get_writer(str(video_path), fps=10)
            for frame in frame_list:
                writer.append_data(frame)
            writer.close()
            print(f"Saved video: {video_path} ({len(frame_list)} frames)")
        except Exception as e:
            print(f"Video save failed: {e}")

    env.close_env()
    return metrics


def main():
    if os.getenv("ROBOTWIN_BENCH_TASK") != "bench":
        os.environ["ROBOTWIN_BENCH_TASK"] = "bench"

    parser = argparse.ArgumentParser(description="Test proximity (SDF) tracking")
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to run (seeds: seed, seed+1, ...)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--render-freq", type=int, default=0)
    parser.add_argument("--bench-subdir", type=str, default=None)
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--parts", type=str, default=None,
                        help="Comma-separated robot parts, e.g. left_ee,right_ee")
    parser.add_argument("--aabb-threshold", type=float, default=None)
    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    run_prefix = f"{task_name}_{task_config}"
    output_dir = Path(args.output_dir) if args.output_dir else bench_root / "test_output" / "proximity"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = bench_root / "bench_task_config" / f"{task_config}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = args.render_freq
    cfg["need_plan"] = True
    cfg["save_data"] = False
    cfg["seed"] = args.seed
    if args.scene_id is not None:
        cfg["scene_id"] = args.scene_id

    prox_cfg = cfg.setdefault("proximity_tracking", {})
    prox_cfg["enabled"] = True
    if args.parts:
        prox_cfg["robot_parts"] = [p.strip() for p in args.parts.split(",")]
    elif "robot_parts" not in prox_cfg:
        prox_cfg["robot_parts"] = ["left_ee", "right_ee"]
    if args.aabb_threshold is not None:
        prox_cfg["aabb_threshold"] = args.aabb_threshold
    elif "aabb_threshold" not in prox_cfg:
        prox_cfg["aabb_threshold"] = 1.0

    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = _embodiment_types[name]["file_path"]
        if robot_file is None:
            raise SystemExit("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])

    print(f"Task: {task_name}  Config: {task_config}  Episodes: {args.num_episodes}")
    print(f"Proximity: {prox_cfg}")

    env_class = get_env_class(task_name, bench_subdir=args.bench_subdir)
    env = env_class()

    # Capture the unpatched class method once so multi-episode re-setup doesn't break the reference
    class_compute_proximity = type(env)._compute_proximity_step

    all_metrics = []
    for ep_idx in range(args.num_episodes):
        metrics = run_episode(env, cfg, ep_idx, output_dir, run_prefix, class_compute_proximity)
        all_metrics.append(metrics)

    if args.num_episodes > 1:
        print(f"\n{'='*60}")
        print(f"All-episode summary ({args.num_episodes} episodes)")
        print(f"{'='*60}")
        parts = [k for k, v in all_metrics[0].items() if isinstance(v, dict) and "min_dist" in v]
        for part in parts:
            dists = [m[part]["min_dist"] for m in all_metrics if part in m and m[part]["min_dist"] >= 0]
            if dists:
                print(f"  {part}: min={min(dists):.4f}  max={max(dists):.4f}  mean={sum(dists)/len(dists):.4f}  (over {len(dists)} episodes)")
        successes = [m.get("task_success", False) for m in all_metrics]
        print(f"  task_success: {sum(successes)}/{len(successes)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
