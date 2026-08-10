"""
Phase 2 part 1 (#35): deterministic occluder-ring spawn + visibility distribution.

Places the target (001_bottle id 9) in the upper third of the reachable region (in
front of the back furniture) and spawns a RING of `--num-occluders` tall, skinny olive-oil
occluders (029_olive-oil id 3, scale 1.0) equally spaced (2*pi/n apart) on a circle of
radius `--offsets` around the target -- n=1 is a single box directly in front (-y, the
robot/camera side). No clutter, no binary search. Then measures the t=0 countertop
`visible_fraction` distribution across seeds, reusing the Phase 1 harness (measurement +
plotting).

visible_fraction = visible_target_px(with occluder) / full_target_px(no occluder),
same seed (target pose is fixed before the occluder is added).

OUTPUT LAYOUT: every run writes to its OWN timestamped folder under a single
dedicated results folder, grouped by run type, so re-running never clobbers a
previous run:
    <out-dir>/<type>/<YYYYmmdd-HHMMSS>/
where <type> is 'rollout' vs 'no_rollout' by default (override with --run-type,
e.g. a planner name like 'hamid' / 'baseline'). --plot-only re-reads an existing
timestamped folder, so pass its full path as --out-dir.

USAGE (from the benchmark folder):
    cd benchmark
    source set_env.sh
    export ROBOTWIN_BENCH_TASK=bench
    python script/bench_script/analyze_occluder_visibility.py \
        --seed-start 0 --num-seeds 50 --offsets 0.10 --bins 20
    # -> ../scripts/validation/results/occluder_visibility/no_rollout/<timestamp>/

    # narrow-region sweep (a few offsets) to see how distance affects occlusion:
    python script/bench_script/analyze_occluder_visibility.py \
        --num-seeds 40 --offsets 0.07,0.10,0.13

    # re-plot only (point at an existing timestamped run folder):
    python script/bench_script/analyze_occluder_visibility.py --plot-only \
        --out-dir ../scripts/validation/results/occluder_visibility/no_rollout/<timestamp>
"""
import os
import sys
import json
import time
import argparse
import contextlib
from datetime import datetime
from pathlib import Path

import numpy as np

from setup_paths import setup_paths
setup_paths()

# --occluder-asset has to be resolved BEFORE lib.scene_constants is imported below. The task
# mixins do `from lib.scene_constants import *`, which COPIES each constant into their own
# module namespace, so rebinding the constants after import would not reach them. Pre-scanning
# argv here and handing the value to the env var the table reads is the only way a CLI flag can
# drive it. The flag is still registered with argparse further down so --help documents it and a
# typo is rejected there; OCCLUDER_ASSET=... in the environment works identically (that is what
# run_approach_mode_ab.sh inherits), and the flag wins when both are given.
for _i, _a in enumerate(sys.argv):
    if _a.startswith("--occluder-asset="):
        os.environ["OCCLUDER_ASSET"] = _a.split("=", 1)[1]
    elif _a == "--occluder-asset" and _i + 1 < len(sys.argv):
        os.environ["OCCLUDER_ASSET"] = sys.argv[_i + 1]

from lib.occluder_ring import (
    draw_ring_config, occluder_ring_xy, parse_count_choices, parse_offset_specs,
)
from lib.planning_tuning import *  # noqa: F403
from lib.run_io import _Tee, _prune_empty_topdirs, effective_out_dir
from lib.scene_build import DR_CLEAN, build_cfg, dr_measure
from lib.scene_constants import *
from lib.visibility import (
    CAMERA, _resolve_target, analyze, analyze_rollout, run_rollout, save_overlay,
)
from task.occluder_task import make_occluder_task
# these analysis scripts are bench-only; default the task mode so build_cfg looks
# under bench_task_config/ (explicit ROBOTWIN_BENCH_TASK still wins)
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
os.chdir(robotwin_root)

def run(args):
    out_dir = effective_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    offset_specs = parse_offset_specs(args.offsets)
    count_choices = parse_count_choices(args.num_occluders)
    clutter_densities = [int(d) for d in args.clutter_densities.split(",")]
    rollout = getattr(args, "rollout", False)
    ep_counter = 0   # unique episode id per rollout (-> video/episode{N}.mp4)

    env = make_occluder_task()()
    spec_desc = ", ".join(f"{lo:.2f}" if lo == hi else f"{lo:.2f}-{hi:.2f}"
                          for lo, hi, _ in offset_specs)
    print(f"seeds from {args.seed_start}, want {args.num_seeds} STABLE  offsets=[{spec_desc}]  "
          f"n_occluders={count_choices}  ring_rotation="
          f"{'random' if args.random_ring_rotation else 'fixed(front)'}  "
          f"occluder_asset={OCCLUDER_ASSET} ({OCCLUDER_MODEL} id{OCCLUDER_ID})  "
          f"clutter_densities={clutter_densities}  rollout={rollout}  camera={CAMERA}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        # Draw seeds until we have args.num_seeds FULLY-USABLE seeds -> num_seeds complete
        # trajectory sets. "Fully usable" means EVERY build the seed needs succeeds and is
        # stable: the clean denominator build AND every (offset x clutter_density) measurement
        # build (target upright, occluder upright per check_stable, occluder clear of the pad).
        # If ANY is rejected (UnStableError from a toppled bottle/milk box, a too-close-to-pad
        # occluder, or any build error) the WHOLE seed is discarded and the next seed is drawn
        # in its place -- nothing partial is written and produced is not incremented. So
        # `--num-seeds N` always yields N seeds, each with the full (offset, density) grid
        # rolled out (== N rollouts for a single offset+density run), never fewer with silent
        # holes. Two passes per seed: (1) build+measure + stability gate, buffering; (2) only
        # if the whole seed passed, run rollouts and commit records. (Ported from the testbench.)
        produced = 0
        draw = args.seed_start          # incrementing seed to draw from (rejected ones skipped)
        max_draws = args.num_seeds * 20 + 50   # safety cap: don't loop forever if builds keep failing
        while produced < args.num_seeds and (draw - args.seed_start) < max_draws:
            seed = draw
            draw += 1
            # --- denominator: same scene, NO occluder ---
            env.spawn_occluder = False
            try:
                env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed, DR_CLEAN))
                target = _resolve_target(env)
                clean_pose = np.array(target.actor.get_pose().p)
                res_clean = env.measure_target_visibility(target, camera_name=CAMERA)
                full_px = res_clean["visible_pixel_count"]
                if args.save_images:
                    save_overlay(env, res_clean["mask"], img_dir / f"seed{seed:04d}_clean.png",
                                 f"seed {seed} CLEAN (denominator)  full_px={full_px}")
            except Exception as e:
                print(f"[seed {seed}] clean build failed/rejected ({type(e).__name__}: {e}); "
                      f"drawing another seed")
                safe_close()
                continue
            safe_close()
            if full_px <= 0:
                print(f"[seed {seed}] full_target_px=0 on clean scene; drawing another seed.")
                continue

            # ---- Pass 1: build + measure every (offset, density); gate stability ----
            seed_items = []       # buffered per-build results, committed only if the seed passes
            seed_ok = True
            for spec in offset_specs:
                if not seed_ok:
                    break
                off = spec[2]                  # nominal label for this spec (midpoint of a range)
                # per-build coin flip: drop the occluder with prob no_occluder_prob
                # (keyed on seed+offset so the decision is shared across densities)
                show = bool(np.random.default_rng(int(seed) * 1000 + int(round(off * 100))).random()
                            >= args.no_occluder_prob)
                # The formation (rotation, count, per-occluder radii) is drawn ONCE here and
                # then re-asserted verbatim on the env for both the measurement build and the
                # rollout build -- re-deriving it anywhere would risk measuring one scene and
                # rolling out another.
                angle0, n_occ, radii = draw_ring_config(seed, spec, count_choices,
                                                        args.random_ring_rotation)
                # Reject scenes where ANY occluder that WILL spawn (off-table ring positions
                # are dropped by the same xlim/ylim filter load_actors uses) lands too close
                # to / on the destination pad. Rejects the WHOLE seed (redraw) so the
                # trajectory count is still met.
                if show:
                    ring_xys = occluder_ring_xy(clean_pose[0], clean_pose[1], radii, n_occ,
                                                angle0, xlim=TABLE_XLIM, ylim=TABLE_YLIM)
                    pad_dist = min((float(np.linalg.norm(np.array(xy) - np.array(PAD_XY)))
                                    for xy in ring_xys), default=float("inf"))
                    if pad_dist < OCC_PAD_MIN_DIST:
                        print(f"[seed {seed}] a ring occluder@off={off:.2f} is {pad_dist:.3f}m from "
                              f"pad (< {OCC_PAD_MIN_DIST:.3f}m); rejecting seed, drawing another.")
                        seed_ok = False
                        break
                env.spawn_occluder = show
                env.occluder_offset = off
                env.num_occluders = n_occ
                env.occluder_angle0 = angle0
                env.occluder_radii = list(radii)
                for cd in clutter_densities:
                    try:
                        env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed,
                                                   dr_measure(cd)))
                        target = _resolve_target(env)
                        pose_ok = bool(np.allclose(np.array(target.actor.get_pose().p), clean_pose, atol=1e-4))
                        res = env.measure_target_visibility(target, camera_name=CAMERA, denominator=full_px)
                        if args.save_images:
                            tag = "occ" if show else "noocc"
                            save_overlay(
                                env, res["mask"],
                                img_dir / f"seed{seed:04d}_off{off:.2f}_cd{cd:02d}_{tag}.png",
                                f"seed {seed} off={off:.2f} {tag} clut={cd}  "
                                f"vis_px={res['visible_pixel_count']} full={full_px} "
                                f"frac={res['visible_fraction']:.3f} {res['bucket']} pose_match={pose_ok}",
                            )
                    except Exception as e:
                        # A toppled bottle/milk box (check_stable -> UnStableError) or any build
                        # error rejects the ENTIRE seed so we redraw and still hit num_seeds.
                        print(f"[seed {seed} off{off:.2f} cd{cd}] build failed/unstable "
                              f"({type(e).__name__}: {e}); rejecting seed, drawing another.")
                        safe_close()
                        seed_ok = False
                        break
                    safe_close()

                    # "offset" stays the spec's NOMINAL value so --group-by offset buckets a
                    # range into one group; occluder_radii/angle0 carry what was actually
                    # spawned, which is the truth for anything reading the geometry back.
                    rec = {"seed": int(seed), "offset": float(off), "full_px": int(full_px),
                           "visible_px": int(res["visible_pixel_count"]),
                           "visible_fraction": float(res["visible_fraction"]),
                           "bucket": res["bucket"], "in_fov": bool(res["in_fov"]),
                           "pose_match": pose_ok, "occluder_shown": show,
                           "num_occluders": int(n_occ) if show else 0,
                           "occluder_radii": ([round(r, 4) for r in radii] if show else []),
                           "occluder_angle0": (round(float(angle0), 4) if show else None),
                           # WHICH object the occluders are. Stamped because milk_box and
                           # olive_oil runs are otherwise indistinguishable in records.jsonl,
                           # and their footprints differ by 2x -- pooling them would be wrong.
                           "occluder_asset": OCCLUDER_ASSET,
                           "clutter_density": int(cd)}
                    seed_items.append({"off": off, "cd": cd, "show": show,
                                       "n_occ": n_occ, "angle0": angle0, "radii": list(radii),
                                       "rec": rec, "res": res})

            if not seed_ok or not seed_items:
                continue   # rejected (unstable / pad-blocked / produced nothing) -> next seed

            # ---- Pass 2: seed accepted -> roll out and commit all its records ----
            for item in seed_items:
                off, cd, show, rec, res = item["off"], item["cd"], item["show"], item["rec"], item["res"]
                # --- expert curobo rollout on the same scene (video + success) ---
                # Re-assert spawn_occluder / occluder_offset so run_rollout's own build
                # reproduces this item's occluder placement (env is shared across items).
                if rollout:
                    env.spawn_occluder = show
                    env.occluder_offset = off
                    env.num_occluders = item["n_occ"]
                    env.occluder_angle0 = item["angle0"]
                    env.occluder_radii = list(item["radii"])
                    # One log file per rollout under <out-dir>/log/episode{N}.log: the
                    # usual stdout/stderr (play_once + run_rollout diagnostics) tee'd to
                    # both console and file, plus the expert's per-phase timing appended.
                    log_dir = out_dir / "log"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    # utf-8 (not the platform-default ASCII) so the rollout's emoji log
                    # lines ("Video is saved...") encode cleanly into the file.
                    rollout_log = open(log_dir / f"episode{ep_counter}.log", "w",
                                       encoding="utf-8")
                    rollout_log.write(
                        f"# rollout episode{ep_counter}  seed={seed} offset={off:.3f} "
                        f"clutter_density={cd} occluder_shown={show} "
                        f"num_occluders={item['n_occ'] if show else 0} "
                        f"radii={[round(r, 3) for r in item['radii']] if show else []} "
                        f"angle0={item['angle0']:.3f}  "
                        f"bucket_measured={res['bucket']}\n\n")
                    rollout_log.flush()
                    # expose this episode's output folder + number to the seed-route visuals
                    # (_get_approach_seed writes into out_dir/seed_route_visuals/episode<N>_<arm>/)
                    env._rollout_out_dir = out_dir
                    env._rollout_ep = ep_counter
                    _t_rollout = time.perf_counter()
                    with contextlib.redirect_stdout(_Tee(sys.stdout, rollout_log)), \
                         contextlib.redirect_stderr(_Tee(sys.stderr, rollout_log)):
                        rollout_result = run_rollout(env, "put_mouse_on_pad", args.base_config, seed,
                                                     dr_measure(cd), out_dir, ep_counter)
                    success = bool(rollout_result["success"])
                    rec["rollout_success"] = success
                    # Wall-clock for THIS rollout (successes and failures alike -- a failure
                    # costs time too). Lets the summary report usable-samples/hour, which is
                    # what actually matters when the seed buys success rate but costs a
                    # clearance-metric build per scene.
                    rec["rollout_seconds"] = float(time.perf_counter() - _t_rollout)
                    rec["attached_trajectory_slowdown"] = ATTACHED_TRAJECTORY_SLOWDOWN
                    rec["rollout_ep"] = ep_counter
                    artifact_info = rollout_result.get("artifact_info") or {}
                    rec["rollout_bucket"] = artifact_info.get("bucket", "success" if success else "fail")
                    rec["rollout_video"] = artifact_info.get(
                        "video_relpath", f"{rec['rollout_bucket']}/video/episode{ep_counter}.mp4"
                    )
                    rec["rollout_data"] = artifact_info.get(
                        "hdf5_relpath", f"{rec['rollout_bucket']}/data/episode{ep_counter}.hdf5"
                    )
                    # Structured failure/candidate metadata set by play_once (see
                    # its checkpoint()/_select_pick_place_candidate) -- survives
                    # run_rollout's internal close_env() since it's just plain
                    # attributes on the (reused) env object. None on success.
                    rec["rollout_failure_stage"] = getattr(env, "rollout_failure_stage", None)
                    rec["rollout_failure_reason"] = getattr(env, "rollout_failure_reason", None)
                    rec["rollout_candidate_info"] = getattr(env, "rollout_candidate_info", None)
                    rec["rollout_grasp_attempts"] = getattr(env, "rollout_grasp_attempts", None)
                    rec["rollout_retention_checks"] = getattr(env, "rollout_retention_checks", None)
                    rec["rollout_descent_slices"] = getattr(env, "rollout_descent_slices", None)
                    # ROBOPRO Phase 4 (APPROACH_MODE A/B). approach_mode is stamped on every
                    # record so a results folder is self-describing and the summary can refuse
                    # to compare cells that were not actually run in different modes.
                    rec["approach_mode"] = env._approach_mode()
                    # Same reason: the backward leg is an independent knob, so a cell is only
                    # fully identified by BOTH modes. Without this, a scripted-placement run
                    # and a direct-placement run are indistinguishable in the results folder.
                    rec["placement_mode"] = env._placement_mode()
                    rec["rollout_plan_effort"] = getattr(env, "rollout_plan_effort", None)
                    rec["rollout_seed_stats"] = getattr(env, "rollout_seed_stats", None)
                    # Physics collision metrics for THIS episode (accumulated by
                    # check_collisions during play_once; reset per build by
                    # _init_collision_metrics; survives close_env like the fields
                    # above). This is the direct clutter-avoidance measurement:
                    # robot_to_static_object / target_to_static_object count clutter
                    # the arm / held object hit, with the object names. Empty/zero
                    # when enable_collision_metrics is off.
                    try:
                        rec["collision_metrics"] = env.get_collision_metrics()
                    except Exception:
                        rec["collision_metrics"] = None
                    cm = rec["collision_metrics"] or {}
                    summary = (f"    seed {seed} off={off:.2f} cd={cd} {res['bucket']}: "
                               f"rollout {'SUCCESS' if success else 'FAIL'} "
                               f"collisions={cm.get('total_collision_count', '?')} "
                               f"(clutter={cm.get('robot_to_static_object', '?')}"
                               f"+{cm.get('target_to_static_object', '?')}) -> "
                               f"{rec['rollout_bucket']}/episode{ep_counter}")
                    print(summary)
                    # append the expert's per-phase timing, then close the per-rollout log
                    stage_times = getattr(env, "rollout_stage_times", None) or []
                    rollout_log.write(summary + "\n")
                    rollout_log.write("\n# expert phase timings (wall-clock seconds per phase)\n")
                    total_s = 0.0
                    for stg in stage_times:
                        rollout_log.write(f"  {stg['stage']:<34} {stg['seconds']:8.3f}s\n")
                        total_s += stg["seconds"]
                    rollout_log.write(f"  {'TOTAL':<34} {total_s:8.3f}s\n")
                    rollout_log.close()
                    # top-level data/ and video/ are emptied by the success/fail bucketing
                    _prune_empty_topdirs(out_dir)
                    ep_counter += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            produced += 1
            print(f"[{produced}/{args.num_seeds}] seed {seed}: full_px={full_px} done")

        if produced < args.num_seeds:
            print(f"WARNING: only {produced}/{args.num_seeds} stable scenes after "
                  f"{draw - args.seed_start} seeds drawn (hit safety cap)")

    safe_close()
    print(f"\nsweep complete -> {jsonl_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--offsets", default="0.2",
                    help="occluder radius/radii in m (target at centre); one figure group per "
                         "token. A token is either a FIXED value (0.20 -- every occluder at that "
                         "radius, the original behaviour) or a RANGE (0.10-0.25 -- each occluder "
                         "independently draws its own radius from that interval, so one scene can "
                         "mix near and far occluders). e.g. 0.15,0.20,0.25 or 0.10-0.25")
    ap.add_argument("--num-occluders", default="1",
                    help="how many occluders to spawn, equally spaced in angle. A single value "
                         "(1 = the original single box in front) or a comma-separated menu "
                         "(2,3,4,5) from which each scene draws one, for a variety of "
                         "configurations.")
    ap.add_argument("--random-ring-rotation", action=argparse.BooleanOptionalAction, default=False,
                    help="rotate the whole formation by a random theta in [0, 2pi) per scene, so "
                         "the occluders are not always anchored with one directly in front of the "
                         "target. Off by default (theta=0), which keeps the historical layout.")
    # Registered so --help documents it and a typo is rejected; the VALUE is consumed by the
    # pre-argv-scan at the top of this file, because lib.scene_constants must already know the
    # asset by the time it is imported. Both spellings are handled there.
    ap.add_argument("--occluder-asset", default=os.environ.get("OCCLUDER_ASSET", "olive_oil"),
                    choices=["olive_oil", "milk_box"],
                    help="which object the occluders are made of. olive_oil (default) = the "
                         "current ring benchmark, 029_olive-oil id3, round ~0.08 footprint. "
                         "milk_box = the ORIGINAL pre-July scene's obstacle, 038_milk-box id2, "
                         "carton ~0.11x0.122 -- noticeably bigger. To rebuild that original "
                         "scene exactly, pair it with --num-occluders 1 --no-random-ring-rotation "
                         "(and --offsets 0.2 for the historical distance), which places the single "
                         "box directly in front (-y) of the target as the old hardcoded scene did.")
    ap.add_argument("--clutter-densities", default="0",
                    help="table clutter density/densities to sweep in the measurement scene, "
                         "comma-separated (0=off). e.g. 0,8,15")
    ap.add_argument("--group-by", default="offset", choices=["offset", "clutter_density"],
                    help="analysis grouping variable for the bucket/histogram figures")
    ap.add_argument("--no-occluder-prob", type=float, default=0.2,
                    help="probability the olive-oil occluder is NOT spawned for a build (default 0.2)")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="../scripts/validation/results/occluder_visibility",
                    help="dedicated results folder for this script; each run lands in its own "
                         "<out-dir>/<type>/<timestamp>/ subfolder (type = --run-type)")
    ap.add_argument("--run-type", default=None,
                    help="name of the <out-dir>/<type>/ subfolder grouping this run "
                         "(default: 'rollout' when --rollout else 'no_rollout'). Use to label "
                         "variants, e.g. a planner name like 'hamid' or 'baseline'.")
    # Tri-state: default None -> resolved after parsing to "on unless --rollout".
    # No-rollout runs save the per-scene overlay PNGs by default (they're the run's
    # primary artifact); --rollout runs default to video-only (run_rollout always
    # saves the episode mp4) and skip the stills. Pass --save-images / --no-save-images
    # to force either way regardless of --rollout.
    ap.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=None,
                    help="save per-scene overlay PNGs (default: on without --rollout, off with it)")
    ap.add_argument("--rollout", action="store_true",
                    help="run an expert curobo rollout per scene (goes under the 'rollout' type "
                         "folder, saves videos, and adds success-only distribution + P(success) per bucket)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    # Resolve the tri-state --save-images default: on for no-rollout runs (overlay PNGs
    # are their main artifact), off for --rollout runs (the episode video is). An explicit
    # --save-images / --no-save-images on the command line wins over this.
    if args.save_images is None:
        args.save_images = not args.rollout

    # One folder per run: <out-dir>/<type>/<timestamp>, so re-running never clobbers
    # previous results. --plot-only re-reads an existing run, so leave args.out-dir
    # exactly as the user pointed it (it should already be a .../<type>/<timestamp> dir).
    if not args.plot_only:
        run_type = args.run_type or ("rollout" if args.rollout else "no_rollout")
        args.out_dir = str(Path(args.out_dir) / run_type / datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(f"[run] writing to {args.out_dir}")

    if not args.plot_only:
        run(args)
    group_label = {"offset": "occluder offset (m)",
                   "clutter_density": "table clutter density"}[args.group_by]
    out_dir = effective_out_dir(args)
    analyze_kwargs = dict(group_key=args.group_by, group_label=group_label,
                          suptitle="Visibility with one olive-oil occluder (countertop)",
                          bar_title=f"Bucket proportions vs {group_label}")
    if args.rollout:
        analyze_rollout(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)
    else:
        analyze(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)


if __name__ == "__main__":
    main()
