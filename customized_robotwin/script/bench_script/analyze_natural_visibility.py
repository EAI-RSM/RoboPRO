"""
Phase 1 (#30): natural visibility distribution / rejection-sampling baseline.

Sweeps seeds x obstacle_density for an office task and measures the t=0
countertop `visible_fraction` of the target (no rollout), reusing the Phase 0
primitive (#28). Produces:

  - a JSONL of per-(seed, density) records,
  - per-density histograms of visible_fraction with the bucket boundaries marked,
  - a per-bucket readout (natural mass / acceptance rate) classifying each bucket
    as selectable-by-rejection vs rare/empty -> needs-occluder.

DENOMINATOR CHOICE (documented):
  visible_fraction = visible_target_px(scene at density D) / full_target_px(clean scene),
  both on the countertop camera, for the SAME seed. The denominator is captured
  from a clean build (`cluttered_table=False`, density 0). This is valid because
  the target pose is set in load_actors() *before* any clutter is generated and
  np.random is seeded first, so the target sits at the same pose for every density
  of a given seed (verified by asserting pose equality). The clean count still
  includes any static-furniture / gripper occlusion present at t=0, so the
  fraction isolates the marginal effect of table clutter.

USAGE (run from the benchmark folder):
    cd benchmark
    source set_env.sh
    export ROBOTWIN_BENCH_TASK=bench
    python script/bench_script/analyze_natural_visibility.py \
        put_mouse_on_pad --bench-subdir office \
        --base-config bench_demo_office_d8 \
        --seed-start 0 --num-seeds 50 \
        --densities 0,4,8,12,15 \
        --out-dir ./visibility_phase1

    # Re-plot from an existing sweep without rebuilding scenes:
    python script/bench_script/analyze_natural_visibility.py --plot-only \
        --out-dir ./visibility_phase1
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

from setup_paths import setup_paths
setup_paths()

bench_root = Path(os.environ["BENCH_ROOT"])
robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
os.chdir(robotwin_root)

import yaml
from envs import CONFIGS_PATH
from visualize_task_scene import get_env_class, get_embodiment_config

CAMERA = "countertop_camera"
TARGET_ATTRS = ("target_obj", "target", "mouse")


def _resolve_target(env):
    for attr in TARGET_ATTRS:
        if hasattr(env, attr):
            return getattr(env, attr)
    raise SystemExit("Could not find a target actor attribute on the env")


def build_cfg(task_name, base_config, seed, dr_overrides):
    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        config_path = bench_root / "bench_task_config" / f"{base_config}.yml"
    else:
        config_path = Path(f"./task_config/{base_config}.yml")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = 0
    cfg["now_ep_num"] = 0
    cfg["seed"] = int(seed)
    cfg["need_plan"] = False          # no planning/rollout for a t=0 measurement
    cfg["save_data"] = False

    cfg.setdefault("domain_randomization", {})
    cfg["domain_randomization"].update(dr_overrides)

    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(name):
        robot_file = _embodiment_types[name]["file_path"]
        if robot_file is None:
            raise SystemExit("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])
    return cfg


# Denominator: clean (no table clutter), density 0.
DR_CLEAN = {"cluttered_table": False, "obstacle_density": 0, "clean_background_rate": 0}


def dr_dense(density):
    return {"cluttered_table": True, "obstacle_density": int(density), "clean_background_rate": 0}


def run_sweep(args):
    env_class = get_env_class(args.task_name, bench_subdir=args.bench_subdir)
    env = env_class()  # reuse one instance across builds (collect_data pattern)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    densities = [int(d) for d in args.densities.split(",")]

    print(f"task={args.task_name} base={args.base_config} camera={CAMERA}")
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  densities={densities}")
    print(f"writing -> {jsonl_path}\n")

    with open(jsonl_path, "w") as fout:
        for si, seed in enumerate(seeds):
            # --- denominator: clean build ---
            env.setup_demo(**build_cfg(args.task_name, args.base_config, seed, DR_CLEAN))
            target = _resolve_target(env)
            clean_pose = np.array(target.actor.get_pose().p)
            full_px = env.capture_target_pixel_count(target, camera_name=CAMERA)
            env.close_env()

            if full_px <= 0:
                print(f"[seed {seed}] WARNING: full_target_px=0 on clean scene; skipping seed.")
                continue

            for density in densities:
                env.setup_demo(**build_cfg(args.task_name, args.base_config, seed, dr_dense(density)))
                target = _resolve_target(env)
                # guard: target pose must match the clean build for the denominator to be valid
                pose_ok = bool(np.allclose(np.array(target.actor.get_pose().p), clean_pose, atol=1e-4))
                res = env.measure_target_visibility(
                    target, camera_name=CAMERA, denominator=full_px,
                    heavy_threshold=args.heavy_threshold,
                )
                env.close_env()

                rec = {
                    "seed": int(seed),
                    "density": int(density),
                    "full_px": int(full_px),
                    "visible_px": int(res["visible_pixel_count"]),
                    "visible_fraction": float(res["visible_fraction"]),
                    "bucket": res["bucket"],
                    "in_fov": bool(res["in_fov"]),
                    "pose_match": pose_ok,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            print(f"[{si+1}/{len(seeds)}] seed {seed}: full_px={full_px} done")

    env.close_env()
    print(f"\nsweep complete -> {jsonl_path}")
    return jsonl_path


def load_records(out_dir):
    jsonl_path = Path(out_dir) / "records.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"No records at {jsonl_path}; run the sweep first.")
    return [json.loads(l) for l in open(jsonl_path) if l.strip()]


def analyze(out_dir, heavy_threshold, selectable_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = load_records(out_dir)
    if any(not r.get("pose_match", True) for r in recs):
        n = sum(1 for r in recs if not r.get("pose_match", True))
        print(f"WARNING: {n} records had target pose != clean build; their fractions are suspect.")

    densities = sorted({r["density"] for r in recs})
    buckets = ["not_visible", "heavily_occluded", "visible"]

    # ---- histograms (one subplot per density) ----
    fig, axes = plt.subplots(1, len(densities), figsize=(4 * len(densities), 3.5), squeeze=False)
    for ax, d in zip(axes[0], densities):
        fr = [r["visible_fraction"] for r in recs if r["density"] == d]
        ax.hist(fr, bins=np.linspace(0, 1, 21), color="#4C72B0", edgecolor="white")
        ax.axvline(0.0, color="red", ls="--", lw=1)
        ax.axvline(heavy_threshold, color="orange", ls="--", lw=1)
        ax.set_title(f"density={d}  (n={len(fr)})")
        ax.set_xlabel("visible_fraction")
        ax.set_xlim(-0.02, 1.02)
    axes[0][0].set_ylabel("seed count")
    fig.suptitle(f"Natural visibility distribution (countertop)  "
                 f"red=not_visible | orange=heavy<{heavy_threshold}")
    fig.tight_layout()
    hist_path = Path(out_dir) / "histograms.png"
    fig.savefig(hist_path, dpi=120)
    print(f"saved {hist_path}")

    # ---- per-bucket readout ----
    summary = {}
    print("\n================ Per-bucket natural mass (acceptance rate) ================")
    header = f"{'density':>8} | " + " | ".join(f"{b:>16}" for b in buckets) + " |  needs-occluder"
    print(header)
    print("-" * len(header))
    for d in densities:
        sub = [r for r in recs if r["density"] == d]
        n = len(sub)
        row = {"n": n, "buckets": {}}
        cells = []
        needs = []
        for b in buckets:
            c = sum(1 for r in sub if r["bucket"] == b)
            frac = c / n if n else 0.0
            row["buckets"][b] = {"count": c, "fraction": frac,
                                 "status": ("selectable" if frac >= selectable_threshold
                                            else "rare" if frac > 0 else "empty")}
            cells.append(f"{c:>4} ({frac:>5.1%})")
            if b in ("not_visible", "heavily_occluded") and frac < selectable_threshold:
                needs.append(b)
        summary[str(d)] = row
        print(f"{d:>8} | " + " | ".join(f"{c:>16}" for c in cells) + f" |  {', '.join(needs) or '-'}")

    print(f"\nselectable threshold = {selectable_threshold:.0%} of seeds; "
          f"buckets below it among the occluded extremes need the occluder.")
    summary_path = Path(out_dir) / "bucket_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"heavy_threshold": heavy_threshold,
                   "selectable_threshold": selectable_threshold,
                   "per_density": summary}, f, indent=2)
    print(f"saved {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name", nargs="?", default="put_mouse_on_pad")
    ap.add_argument("--bench-subdir", default="office")
    ap.add_argument("--base-config", default="bench_demo_office_d8")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--densities", default="0,4,8,12,15")
    ap.add_argument("--heavy-threshold", type=float, default=0.25)
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="./visibility_phase1")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip the sweep; just (re)build figures from records.jsonl.")
    args = ap.parse_args()

    if not args.plot_only:
        run_sweep(args)
    analyze(args.out_dir, args.heavy_threshold, args.selectable_threshold)


if __name__ == "__main__":
    main()
