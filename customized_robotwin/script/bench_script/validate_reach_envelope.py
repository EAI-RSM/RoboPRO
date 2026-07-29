#!/usr/bin/env python3
"""Correctness gate for the reach envelope (produced by reach_envelope.py, consumed by
clearance_metric_3d.py --reach-envelope). Validates either mask: --reach-mode occupancy (Tier 2,
default) or sphere (Tier 1).

The prune claim: every cell the mask marks is unreachable for ANY grasp pose, so it can be forced
BEYOND with no IK solve. This script verifies that claim DIRECTLY on a real scene:

  1. build the scene + arm exactly as the metric does,
  2. LOAD the exact prune mask a run would apply (occupancy or sphere) on the metric grid,
  3. solve IK on ONLY the PRUNED cells (invert the mask),
  4. assert every one comes back BEYOND (unreachable ignoring the scene, label != BEYOND == reach).

Zero pruned cells reachable -> PASS (the prune never cuts a reachable cell, so it can never drop a
FREE cell). Any pruned cell reachable -> FAIL: raise --occ-mc-safety/--occ-samples (occupancy) or
--reach-margin (sphere) in reach_envelope.py, regenerate, re-validate. Solving only the pruned cells
tests the claim directly and costs ~the pruned fraction, not a full sweep. High --ik-seeds so a hard-
but-reachable pose isn't missed (a miss would be a false PASS).

Grid MUST match the occupancy artifact (defaults = the metric/producer grid). Run reach_envelope.py
FIRST. USAGE:
    python validate_reach_envelope.py --seed 1 --arm right
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from clearance_metric_3d import make_occluder_task, select_arm
from lib.ik_grid import build_grid
from lib.labeling import BEYOND, label_volume, load_reach_envelope
from lib.run_io import CLEARANCE_RESULTS_DIR as RESULTS_DIR, Timings
from lib.scene_build import DR_CLEAN, build_cfg

VAL_DIR = RESULTS_DIR.parent / "reach_envelope_validation"


def diagnose_violations(out_dir, args, xs, ys, zs, prune_mask, bad):
    """On FAIL, characterise the reachable-but-pruned cells so we know the CAUSE:
      - shallow depth into the pruned region + scattered near the workspace edge => dilation shortfall
        (uniform joint-space sampling under-covers the reach boundary) -> just raise --occ-mc-safety;
      - deep and/or a coherent one-sided band => the occupancy is shifted vs where the metric reads it
        (centre C / transform M off) -> a real bug, not a knob.
    Prints depth stats + a suggested --occ-mc-safety, the violation bounding box vs the arm base, and a
    top-down figure (red = pruned-yet-reachable) whose PATTERN settles which case it is."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import distance_transform_edt
    res, zres = float(args.res), float(args.zres)

    depth = distance_transform_edt(prune_mask, sampling=(zres, res, res))   # dist of each pruned cell to nearest KEPT
    sh = depth[bad]                                                         # how deep each violation sits in the prune
    apath = Path(args.reach_cache_dir) / f"reach_envelope_{args.arm}.npz"
    d = np.load(apath)
    grip = float(d["gripper_offset"])
    mc = float(d["occ_mc_safety"]) if "occ_mc_safety" in d.files else 0.0
    bw = np.asarray(d["base_world"], dtype=float)
    suggest = mc + float(sh.max()) + res                                   # +1 cell headroom
    # is it a thin boundary rind (all violations sit ~1 cell outside the kept region) or a deep hole?
    cell = max(res, zres)
    surface_frac = float((sh <= 1.5 * cell).mean())                        # fraction within ~1 cell of kept
    print(f"[val] violation depth into pruned region (m): median={np.median(sh):.3f} "
          f"p90={np.percentile(sh, 90):.3f} max={sh.max():.3f}  (dilation now = grip {grip:.3f} + mc {mc:.3f})")
    print(f"[val] {100 * surface_frac:.0f}% of violations sit within ~1 cell of the kept region "
          f"(=> boundary rind, not an interior hole)")
    if sh.max() <= 3 * cell:
        print(f"[val] VERDICT: thin boundary shell (max depth {sh.max():.3f}m <= 3 cells). NOT a transform "
              f"bug -- the dilated occupancy just under-reaches the true reach surface. Fix: regenerate "
              f"with reach_envelope.py --occ-mc-safety {suggest:.3f} (and/or more --occ-samples).")
    else:
        print(f"[val] VERDICT: deep violations (max depth {sh.max():.3f}m > 3 cells) -- a coherent region is "
              f"uncovered; check the side view for a one-sided shift (transform) vs a sampling hole.")

    zi, yi, xi = np.nonzero(bad)
    wx, wy, wz = xs[xi], ys[yi], zs[zi]
    print(f"[val] violation bbox: x[{wx.min():.2f},{wx.max():.2f}] y[{wy.min():.2f},{wy.max():.2f}] "
          f"z[{wz.min():.2f},{wz.max():.2f}]  (arm-base xy = {np.round(bw[:2], 2)})")

    # TWO views: top-down (z-collapsed, where a boundary rind LOOKS like interior overlap) and a side
    # x-z elevation (y-collapsed), where a boundary rind reads correctly as a thin skin on the blob.
    fig, (axT, axS) = plt.subplots(1, 2, figsize=(15, 6.5))
    extent_xy = [xs.min(), xs.max(), ys.min(), ys.max()]
    axT.imshow(np.where((~prune_mask).any(axis=0), 1.0, 0.0), origin="lower", extent=extent_xy,
               cmap="Greens", vmin=0, vmax=1.5, aspect="equal")
    yb, xb = np.nonzero(bad.any(axis=0))
    axT.scatter(xs[xb], ys[yb], s=7, c="red", label=f"pruned-yet-reachable ({int(bad.sum())})")
    axT.plot(bw[0], bw[1], "*", color="gold", ms=16, mec="k", label="arm base xy")
    axT.legend(loc="upper right", fontsize=8); axT.set_xlabel("x (m)"); axT.set_ylabel("y (m)")
    axT.set_title("Top-down (z-collapsed): red on green = boundary at top/bottom z")

    extent_xz = [xs.min(), xs.max(), zs.min(), zs.max()]
    axS.imshow(np.where((~prune_mask).any(axis=1), 1.0, 0.0), origin="lower", extent=extent_xz,
               cmap="Greens", vmin=0, vmax=1.5, aspect="auto")
    zb2, xb2 = np.nonzero(bad.any(axis=1))                                  # collapse over y -> (z, x)
    axS.scatter(xs[xb2], zs[zb2], s=7, c="red", label="pruned-yet-reachable")
    axS.axhline(bw[2], color="gold", ls="--", lw=1.5, label="shoulder z")
    axS.legend(loc="upper right", fontsize=8); axS.set_xlabel("x (m)"); axS.set_ylabel("z (m)")
    axS.set_title("Side x-z (y-collapsed): thin red skin on green = boundary rind")
    fig.suptitle(f"Occupancy violations  |  seed {args.seed}, arm {args.arm}   "
                 f"(green = kept, red = reachable-but-pruned)", fontsize=12)
    stem = f"violations_seed{args.seed}_{args.arm}"
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_dir / f"{stem}.png", dpi=130); plt.close(fig)
    print(f"[val] wrote {stem}.png  (thin red skin on the blob => boundary rind; a solid one-sided slab => transform)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--offset", type=float, default=0.2)
    ap.add_argument("--num-occluders", type=int, default=1)
    ap.add_argument("--occluder-angle0", type=float, default=0.0)
    ap.add_argument("--no-occluder", dest="occluder", action="store_false")
    ap.set_defaults(occluder=True)
    ap.add_argument("--arm", choices=["left", "right", "auto"], default="auto")
    ap.add_argument("--topdown", action="store_true")
    ap.add_argument("--zmin", type=float, default=0.78)
    ap.add_argument("--zmax", type=float, default=1.4)
    ap.add_argument("--zres", type=float, default=0.03,
                    help="MUST match the occupancy artifact's grid (default = the metric/producer grid)")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.35)
    ap.add_argument("--ymax", type=float, default=0.35)
    ap.add_argument("--res", type=float, default=0.01,
                    help="MUST match the occupancy artifact's grid (default = the metric/producer grid)")
    ap.add_argument("--ik-seeds", type=int, default=60, help="use a HIGH seed count so 'reachable' is not "
                    "under-counted (a missed reachable cell would give a falsely-passing bound)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--reach-mode", choices=["occupancy", "sphere"], default="occupancy",
                    help="which precomputed mask to validate: occupancy (Tier 2) or sphere (Tier 1)")
    ap.add_argument("--reach-cache-dir", default=str(RESULTS_DIR / "_reach_cache"),
                    help="where reach_envelope.py stored the artifact being validated")
    ap.add_argument("--out-dir", default=str(VAL_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[val] writing to {out_dir}")
    tm = Timings()

    with tm.section("scene_setup"):
        env = make_occluder_task()()
        env.spawn_occluder = args.occluder
        env.occluder_offset = args.offset
        env.num_occluders = args.num_occluders
        env.occluder_angle0 = args.occluder_angle0
        env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, args.seed, DR_CLEAN))

    with tm.section("grid_build"):
        xs, ys, zs, XX, YY = build_grid(args)
    print(f"[val] grid: {XX.size} cells/slice x {len(zs)} slices = {XX.size * len(zs)} voxels")

    with tm.section("select_arm"):
        arm, planner, grasp_q, grasp_pose, ik = select_arm(env, args)
    args.arm = arm

    # LOAD the exact prune mask a run would apply (occupancy or sphere); errors if artifact missing
    prune_mask = load_reach_envelope(args.reach_cache_dir, arm, xs, ys, zs, XX, YY, mode=args.reach_mode)
    n_prune = int(prune_mask.sum())
    if n_prune == 0:
        print("[val] the mask prunes nothing -> nothing to validate"); tm.save(out_dir); return

    # Solve IK on ONLY the PRUNED cells (invert the mask): they must ALL come back BEYOND. This tests
    # the exact claim "pruned => unreachable" directly, and costs only ~the pruned fraction (not a full
    # sweep). free_only=False so the OFF (no-world) pass runs -> label != BEYOND == kinematic reach
    # ignoring the scene, which is what the prune must never contain (a scene-free-reachable cell could
    # be FREE under some occluder). High --ik-seeds so we don't under-count reach (a miss = false pass).
    with tm.section("solve_pruned_cells"):
        label, _q, _off = label_volume(env, planner, ik, arm, XX, YY, zs, grasp_q, args.chunk,
                                       num_seeds=args.ik_seeds, free_only=False, prune_mask=~prune_mask)

    bad = label != BEYOND                                          # a solved (=pruned) cell that is reachable
    passed = int(bad.sum()) == 0
    result = {
        "PASS": passed,
        "mode": args.reach_mode,
        "pruned_cells_solved": n_prune,                           # the cells we IK-tested (the prune set)
        "pruned_cells_reachable": int(bad.sum()),                 # MUST be 0 for a safe prune
        "voxels_total": int(label.size),
        "ik_saved_pct": round(100 * n_prune / label.size, 2),
        "config": {"seed": args.seed, "arm": arm, "offset": args.offset, "res": args.res,
                   "zmin": args.zmin, "zmax": args.zmax, "zres": args.zres, "ik_seeds": args.ik_seeds},
    }
    (out_dir / "validation.json").write_text(json.dumps(result, indent=2))

    print("[val] ------------------------------------------------------------")
    if passed:
        print(f"[val] PASS ({args.reach_mode}): all {n_prune:,} pruned voxels are unreachable "
              f"({100 * n_prune / label.size:.1f}% of the grid IK-skipped) -> safe.")
    else:
        print(f"[val] FAIL ({args.reach_mode}): {int(bad.sum())} of {n_prune:,} pruned cell(s) are actually "
              f"reachable -> the mask is too aggressive.")
        diagnose_violations(out_dir, args, xs, ys, zs, prune_mask, bad)
    print(f"[val] wrote validation.json -> {out_dir}")
    tm.save(out_dir)

    try:
        env.close_env()
    except Exception:
        pass


if __name__ == "__main__":
    main()
