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
        --densities 8,12,15 --bins 20 \
        --out-dir ./visibility_phase1

    # Re-plot from an existing sweep without rebuilding scenes:
    python script/bench_script/analyze_natural_visibility.py --plot-only \
        --out-dir ./visibility_phase1
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

import numpy as np

from setup_paths import setup_paths
setup_paths()

from lib.scene_build import (
    DR_CLEAN, build_cfg, get_embodiment_config, get_env_class,
)
from lib.visibility import (
    BUCKET_BOUNDARIES, BUCKET_COLORS, BUCKET_ORDER, CAMERA, TARGET_ATTRS,
    _bucket_of, _resolve_target, analyze, analyze_rollout, load_records,
    run_rollout, save_overlay,
)
# these analysis scripts are bench-only; default the task mode so build_cfg looks
# under bench_task_config/ (explicit ROBOTWIN_BENCH_TASK still wins)
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

bench_root = Path(os.environ["BENCH_ROOT"])
robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
os.chdir(robotwin_root)

import yaml
from envs import CONFIGS_PATH


# Bucket taxonomy (must match DEFAULT_VISIBILITY_BUCKETS in envs/_base_task.py):
#   not_visible        frac == 0
#   heavily_occluded   0    < frac < 0.20
#   mostly_occluded    0.20 <= frac < 0.5
#   partially_occluded 0.5  <= frac < 0.9
#   fully_visible      0.9  <= frac






# Denominator: clean (no table clutter), density 0.


def dr_dense(density, tall_only=False, handcrafted=False):
    dr = {"cluttered_table": True, "obstacle_density": int(density), "clean_background_rate": 0}
    if tall_only:
        dr["obstacle_tall_only"] = True
    if handcrafted:
        dr["obstacle_handcrafted"] = True
    return dr








def run_sweep(args):
    env_class = get_env_class(args.task_name, bench_subdir=args.bench_subdir)
    env = env_class()  # reuse one instance across builds (collect_data pattern)

    out_dir = effective_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    densities = [int(d) for d in args.densities.split(",")]
    rollout = getattr(args, "rollout", False)
    ep_counter = 0   # unique episode id per rollout (-> video/episode{N}.mp4)

    print(f"task={args.task_name} base={args.base_config} camera={CAMERA}")
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  densities={densities}  rollout={rollout}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        for si, seed in enumerate(seeds):
            # --- denominator: clean build (skip seed if the build is unstable) ---
            try:
                env.setup_demo(**build_cfg(args.task_name, args.base_config, seed, DR_CLEAN))
                target = _resolve_target(env)
                clean_pose = np.array(target.actor.get_pose().p)
                res_clean = env.measure_target_visibility(target, camera_name=CAMERA)
                full_px = res_clean["visible_pixel_count"]
                if args.save_images:
                    save_overlay(env, res_clean["mask"], img_dir / f"seed{seed:04d}_clean.png",
                                 f"seed {seed} CLEAN (denominator)  full_px={full_px}")
            except Exception as e:
                print(f"[seed {seed}] clean build failed ({type(e).__name__}: {e}); skipping seed")
                safe_close()
                continue
            safe_close()

            if full_px <= 0:
                print(f"[seed {seed}] WARNING: full_target_px=0 on clean scene; skipping seed.")
                continue

            for density in densities:
                try:
                    env.setup_demo(**build_cfg(args.task_name, args.base_config, seed,
                                               dr_dense(density, args.tall_only, args.handcrafted)))
                    target = _resolve_target(env)
                    # guard: target pose must match the clean build for the denominator to be valid
                    pose_ok = bool(np.allclose(np.array(target.actor.get_pose().p), clean_pose, atol=1e-4))
                    res = env.measure_target_visibility(
                        target, camera_name=CAMERA, denominator=full_px,
                    )  # default DEFAULT_VISIBILITY_BUCKETS taxonomy
                    if args.save_images:
                        tag = "hand" if args.handcrafted else "tall" if args.tall_only else "mix"
                        save_overlay(
                            env, res["mask"],
                            img_dir / f"seed{seed:04d}_d{density:02d}_{tag}.png",
                            f"seed {seed} d={density} {tag}  vis_px={res['visible_pixel_count']} "
                            f"full={full_px} frac={res['visible_fraction']:.3f} "
                            f"{res['bucket']} pose_match={pose_ok}",
                        )
                except Exception as e:
                    print(f"[seed {seed} d{density}] build failed ({type(e).__name__}: {e}); skipping")
                    safe_close()
                    continue
                safe_close()

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
                # --- expert curobo rollout on the same scene (video + success) ---
                if rollout:
                    success = run_rollout(env, args.task_name, args.base_config, seed,
                                          dr_dense(density, args.tall_only, args.handcrafted),
                                          out_dir, ep_counter)
                    rec["rollout_success"] = bool(success)
                    rec["rollout_ep"] = ep_counter
                    rec["rollout_video"] = f"video/episode{ep_counter}.mp4"
                    print(f"    seed {seed} d={density} {res['bucket']}: "
                          f"rollout {'SUCCESS' if success else 'FAIL'} -> episode{ep_counter}.mp4")
                    ep_counter += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            print(f"[{si+1}/{len(seeds)}] seed {seed}: full_px={full_px} done")

    env.close_env()
    print(f"\nsweep complete -> {jsonl_path}")
    return jsonl_path






def effective_out_dir(args):
    """When --rollout is on, write everything to <out-dir>_rollout so the rollout
    run never collides with a measurement-only run of the same out-dir."""
    suffix = "_rollout" if getattr(args, "rollout", False) else ""
    return Path(str(args.out_dir).rstrip("/") + suffix)








def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name", nargs="?", default="put_mouse_on_pad")
    ap.add_argument("--bench-subdir", default="office")
    ap.add_argument("--base-config", default="bench_demo_office_d8")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--densities", default="30")
    ap.add_argument("--bins", type=int, default=20, help="equal-width histogram bins over [0,1]")
    ap.add_argument("--tall-only", action="store_true",
                    help="place table clutter from the tall obstacle pool only (diagnostic)")
    ap.add_argument("--handcrafted", action="store_true",
                    help="restrict table clutter to the handcrafted tallest/widest set (diagnostic)")
    ap.add_argument("--save-images", action="store_true",
                    help="save per-run countertop overlays (RGB | visible mouse pixels in red) to <out-dir>/images/")
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="../scripts/validation/results/visibility_phase1")
    ap.add_argument("--rollout", action="store_true",
                    help="run an expert curobo rollout per scene (writes to <out-dir>_rollout, "
                         "saves videos, and adds success-only distribution + P(success) per bucket)")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip the sweep; just (re)build figures from records.jsonl.")
    args = ap.parse_args()

    if not args.plot_only:
        run_sweep(args)
    out_dir = effective_out_dir(args)
    if args.rollout:
        analyze_rollout(out_dir, args.bins, args.selectable_threshold)
    else:
        analyze(out_dir, args.bins, args.selectable_threshold)


if __name__ == "__main__":
    main()
