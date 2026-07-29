#!/usr/bin/env python3
"""Producer for the pose-independent REACH ENVELOPE (Tier-1 speedup for clearance_metric_3d.py).

Run this ONCE per robot. It computes, per arm, the maximum distance the gripper control point can
ever be from the arm base -- over the ENTIRE joint box, hence over every grasp orientation -- via a
forward-kinematics Monte-Carlo. Any grid cell farther than that radius is unreachable NO MATTER THE
POSE, so the actual metric runs (clearance_metric_3d.py --reach-envelope) can skip its IK solve.

Why this is a separate file: the envelope depends ONLY on the robot + arm base pose. It does not
depend on the scene, seed, occluder, target, or grasp orientation. So it is genuinely a compute-once
artifact -- there is no reason for an actual run to ever recompute it. This script produces the
artifact (cached forever) and the eliminated-vs-kept image; clearance_metric_3d.py only LOADS it.

Strictness: sampling is UNIFORM over the raw joint limits (not ik.sample_configs, which would reject
self-colliding seeds). Including infeasible configs can only push R_max OUT, never in, so the radius
is an upper bound on reach -> a pruned cell is provably unreachable, never a false cut.

OUTPUTS (per arm):
  <cache-dir>/reach_envelope_<arm>.npz   the artifact runs load   (R_max, base_world, reach_radius, ...)
  <cache-dir>/reach_radius_<arm>.json    the raw FK radius cache   (so re-runs skip the FK)
  <out-dir>/<timestamp>/reach_envelope_simple_<arm>.png   the clean eliminated-vs-kept image

USAGE (env sourced + ROBOTWIN_BENCH_TASK=bench):
    python reach_envelope.py --arms both
    #   re-render the image on a different grid without recomputing FK: just re-run (R_max is cached)
    #   force a fresh FK radius:  add --reach-recompute
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import torch
import transforms3d as t3d
from scipy.ndimage import distance_transform_edt

# reachability_map import pulls in torch + the env stack and selects the Agg backend (headless-safe).
from clearance_metric_3d import LABEL_COLORS, make_occluder_task, select_arm
from lib.ik_grid import _build_ik_solver_no_world, build_grid
from lib.labeling import BEYOND, FREE, geometric_envelope
from lib.run_io import CLEARANCE_RESULTS_DIR as RESULTS_DIR, Timings
from lib.scene_build import DR_CLEAN, build_cfg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DEFAULT_CACHE = RESULTS_DIR / "_reach_cache"
DEFAULT_OUT = RESULTS_DIR.parent / "reach_envelope"


def compute_reach_radius(ik, n_samples=2_000_000, batch=100_000):
    """Max distance of the EE control point from the arm base over the ENTIRE joint box, via
    uniform-in-limits forward-kinematics Monte-Carlo (see module docstring for why this is a strict
    upper bound on reach). Returns R_max (m, EE in the base/kinematics frame)."""
    lim = ik.kinematics.get_joint_limits().position.detach()     # [dof, 2] = (min, max) per joint
    if lim.shape[0] == 2 and lim.shape[1] != 2:                   # tolerate a [2, dof] layout
        lim = lim.transpose(0, 1).contiguous()
    lo, hi = lim[:, 0], lim[:, 1]
    dof = int(lo.shape[0])
    device, dtype = lo.device, lo.dtype
    rmax, done = 0.0, 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        q = lo + (hi - lo) * torch.rand(b, dof, device=device, dtype=dtype)
        pos = ik.fk(q).ee_position                               # (b, 3) EE in the base/kinematics frame
        rmax = max(rmax, float(pos.norm(dim=1).max().item()))
        done += b
        del q, pos
    torch.cuda.empty_cache()
    return rmax


def reach_radius_cached(ik, planner, arm, cache_dir, n_samples, recompute=False):
    """R_max for this arm, cached to reach_radius_<arm>.json (depends only on the robot model, valid
    forever). Cache key = robot config yml + sample budget; recompute=True forces a rebuild."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"reach_radius_{arm}.json"
    key = {"yml": str(planner.yml_path), "n_samples": int(n_samples)}
    if cache.exists() and not recompute:
        prev = json.loads(cache.read_text())
        if prev.get("key") == key:
            print(f"[reach-env] loaded cached R_max={prev['R_max']:.4f}m for {arm}  ({cache})")
            return float(prev["R_max"])
    print(f"[reach-env] computing R_max for {arm} via {n_samples:,} FK samples ...")
    R_max = compute_reach_radius(ik, n_samples=n_samples)
    cache.write_text(json.dumps({"key": key, "R_max": R_max}, indent=2))
    print(f"[reach-env] R_max={R_max:.4f}m  (cached -> {cache})")
    return R_max


def plot_envelope_simple(out_dir, arm, XX, YY, zs, envelope, reach_radius):
    """Minimal 2-colour top-down: cells eliminated for EVERY pose (grey) vs every other cell (green),
    and nothing else on the axes. Just the distinction between the fully-eliminated set and the rest."""
    always = envelope.all(axis=0)                     # eliminated at every height = gone no matter what
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = ListedColormap([LABEL_COLORS[FREE], LABEL_COLORS[BEYOND]])
    ax.imshow(np.where(always, 1.0, 0.0), origin="lower", extent=extent, cmap=cmap,
              vmin=0, vmax=1, aspect="equal")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=LABEL_COLORS[BEYOND], label="eliminated (unreachable, any pose)"),
                       Patch(color=LABEL_COLORS[FREE], label="kept")], loc="upper right", fontsize=9)
    n_elim, n_tot = int(always.sum()), always.size
    ax.set_title(f"Fully-eliminated cells  |  arm {arm}\n"
                 f"{n_elim}/{n_tot} ({100 * n_elim / n_tot:.0f}%) eliminated for any pose "
                 f"(eff radius {reach_radius:.2f}m)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    stem = f"reach_envelope_simple_{arm}"
    fig.tight_layout(); fig.savefig(Path(out_dir) / f"{stem}.png", dpi=130); plt.close(fig)
    print(f"[reach-env] wrote {stem}.png  ({100 * n_elim / n_tot:.0f}% of the plan eliminated)")


def world_reach_transform(planner, arm):
    """Map an FK ee_position (kinematics/base frame) to the WORLD endlink point: X = C + M @ p_ee,
    with C the shoulder centre and M = R_base @ R_rot^T. This is the exact inverse of the
    _world_gripper_to_curobo frame chain (world->base, +frame_bias, +small per-arm yaw R_rot), so a
    sampled FK point lands where the metric's grid actually reads it. Returns (C (3,), M (3,3))."""
    R_base = t3d.quaternions.quat2mat(np.asarray(planner.robot_origion_pose.q, dtype=float))
    fb_vec = np.asarray(planner.frame_bias, dtype=float)
    C = np.asarray(planner.robot_origion_pose.p, dtype=float) - R_base @ fb_vec        # shoulder centre
    R_rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02 if arm == "left" else -0.01)     # same as _world_gripper_to_curobo
    M = R_base @ R_rot.T
    return C, M


def build_occupancy(ik_nw, C, M, xs, ys, zs, gripper_offset, mc_safety, n_samples,
                    batch=32_000, rejection_ratio=8):
    """POSE-INDEPENDENT reachable-workspace occupancy (Tier 2). Samples FEASIBLE (self-collision-free,
    WORLD-free) joint configs, forward-kinematics each to a world endlink point (X = C + M @ p_ee),
    and marks the grid voxel it lands in. Unioned over all configs, this is the arm's reachable ENDLINK
    set at SOME orientation -- a superset of the metric's reachable set at any single grasp_q, hence
    safe to prune the complement of. Reuses the FK sampler; no IK per grid cell.

    Two dilations make the prune a strict superset of reachable GRIPPER cells:
      - gripper_offset: the grid cell is the gripper, offset from the endlink by ~this much; a gripper
        is reachable-at-some-orientation iff it is within gripper_offset of a reachable endlink, so we
        grow the occupancy by that radius (exact Euclidean ball via anisotropic EDT).
      - mc_safety: extra slack covering Monte-Carlo boundary gaps (a truly reachable voxel that no
        sample happened to hit). Over-dilation only prunes LESS -> always safe.

    ik_nw must be a NO-WORLD IK solver (self-collision only) so the occupancy is scene-independent
    (never excludes a config just because THIS occluder blocks it). `batch` x `rejection_ratio` is the
    number of configs curobo collision-checks in ONE shot -- keep the product modest (~256k) or the
    self-collision check OOMs (curobo's default rejection_ratio is 50, so a big batch explodes).
    Returns (prune_mask (nz, ny, nx) bool True=eliminated, n_feasible int, raw_occ_frac float)."""
    nx, ny, nz = len(xs), len(ys), len(zs)
    res = float(xs[1] - xs[0]) if nx > 1 else 1.0
    zres = float(zs[1] - zs[0]) if nz > 1 else 1.0
    x0, y0, z0 = float(xs[0]), float(ys[0]), float(zs[0])
    occ = np.zeros((nz, ny, nx), dtype=bool)
    got, stalls = 0, 0
    while got < n_samples and stalls < 20:
        b = min(batch, n_samples - got)
        q = ik_nw.sample_configs(b, rejection_ratio=rejection_ratio)   # feasible = self-collision-free
        if q is None or q.shape[0] == 0:
            stalls += 1
            continue
        pe = ik_nw.fk(q).ee_position.detach().cpu().numpy()      # (m, 3) EE in kinematics frame
        X = pe @ M.T + C                                         # (m, 3) world endlink points
        ix = np.round((X[:, 0] - x0) / res).astype(np.int64)
        iy = np.round((X[:, 1] - y0) / res).astype(np.int64)
        iz = np.round((X[:, 2] - z0) / zres).astype(np.int64)
        m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        occ[iz[m], iy[m], ix[m]] = True                         # points off-grid are beyond it anyway
        got += int(q.shape[0])
        del q, pe
        torch.cuda.empty_cache()                             # release the rejection-sampling scratch each batch
    if stalls >= 20:
        print(f"[reach-env] WARN occupancy sampling stalled at {got:,}/{n_samples:,} feasible configs "
              f"(rejection too tight?); mask built from what was collected")
    # grow the endlink occupancy to a reachable-GRIPPER superset (true Euclidean ball, res != zres)
    dist = distance_transform_edt(~occ, sampling=(zres, res, res))
    reachable = dist <= (gripper_offset + mc_safety)
    return ~reachable, got, float(occ.mean())


def produce_for_arm(env, args, arm, cache_dir, out_dir):
    """Compute + persist the envelope artifact for one arm, and render its eliminated-vs-kept image.
    Stores BOTH the sphere fallback (grid-independent: shoulder centre + radius) and the Tier-2
    occupancy prune mask (grid-dependent: the real reachable-workspace shape)."""
    args.arm = arm
    _arm, planner, grasp_q, grasp_pose, ik = select_arm(env, args)   # forced arm -> that planner + IK
    del ik                                          # the world IK isn't needed; reach is scene-independent
    torch.cuda.empty_cache()
    ik_nw = _build_ik_solver_no_world(planner)      # self-collision only -> scene-independent reachability

    R_max = reach_radius_cached(ik_nw, planner, arm, cache_dir, args.reach_samples,
                                recompute=args.reach_recompute)

    # Sphere fallback centre: the arm's KINEMATIC ROOT (shoulder), which R_max is measured from -- NOT
    # robot_origion_pose (the floor origin). They differ by frame_bias, which SHIFTS the centre (it must
    # never be added to the radius; that inflates the ball ~2x and prunes nothing). See world_reach_transform.
    C, M = world_reach_transform(planner, arm)
    base_world = C
    reach_radius = R_max + args.gripper_offset + args.reach_margin      # gripper cell is endlink +/- offset

    # sanity: the target's real grasp pose (known reachable) must sit inside the radius, and within
    # R_max+grip of the centre (a reachable point can't exceed max reach -- if it does, the centre is off).
    if grasp_pose is not None:
        dg = float(np.linalg.norm(np.asarray(grasp_pose[:3], dtype=float) - base_world))
        ok = "OK" if dg <= reach_radius else "WARN OUTSIDE -> centre/margin wrong!"
        flag = "" if dg <= R_max + args.gripper_offset + 1e-6 else "  [!! > R_max+grip: centre off]"
        print(f"[reach-env] {arm}: R_max={R_max:.3f} +grip {args.gripper_offset:.3f} "
              f"+margin {args.reach_margin:.3f} -> radius={reach_radius:.3f}m  "
              f"centre(shoulder)={np.round(base_world, 3)}  grasp-dist={dg:.3f} ({ok}){flag}")

    # Tier 2: the occupancy prune mask on the metric grid (the real reachable-workspace shape)
    xs, ys, zs, XX, YY = build_grid(args)
    prune, n_feas, occ_frac = build_occupancy(ik_nw, C, M, xs, ys, zs, args.gripper_offset,
                                              args.occ_mc_safety, args.occ_samples,
                                              batch=args.occ_batch, rejection_ratio=args.occ_rejection)
    sphere_prune = geometric_envelope(XX, YY, zs, base_world, reach_radius)   # for the comparison print
    print(f"[reach-env] {arm}: occupancy from {n_feas:,} feasible configs (raw occ {100 * occ_frac:.1f}%)"
          f" -> PRUNE {100 * prune.mean():.1f}%   (vs sphere {100 * sphere_prune.mean():.1f}%)")

    # the artifact runs load: sphere fallback (grid-independent) + occupancy mask + its grid axes
    artifact = Path(cache_dir) / f"reach_envelope_{arm}.npz"
    np.savez(artifact, R_max=np.float64(R_max), base_world=base_world,
             reach_radius=np.float64(reach_radius), gripper_offset=np.float64(args.gripper_offset),
             frame_bias=np.asarray(planner.frame_bias, dtype=float),
             robot_origin=np.asarray(planner.robot_origion_pose.p, dtype=float),
             reach_margin=np.float64(args.reach_margin), n_samples=np.int64(args.reach_samples),
             occupancy_prune=prune, xs=xs, ys=ys, zs=zs,
             occ_mc_safety=np.float64(args.occ_mc_safety), n_feasible=np.int64(n_feas),
             yml=str(planner.yml_path))
    print(f"[reach-env] wrote artifact {artifact}  (sphere + occupancy)")

    plot_envelope_simple(out_dir, arm, XX, YY, zs, prune, reach_radius)   # image = the occupancy prune

    del ik_nw
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", choices=["left", "right", "both"], default="both",
                    help="which arm(s) to precompute the envelope for")
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed", type=int, default=1, help="scene seed (only used to build the robot; the "
                    "envelope itself is scene-independent)")
    ap.add_argument("--no-occluder", dest="occluder", action="store_false")
    ap.set_defaults(occluder=False)   # occluders are irrelevant to the envelope; skip them
    ap.add_argument("--offset", type=float, default=0.2)
    ap.add_argument("--num-occluders", type=int, default=1)
    ap.add_argument("--occluder-angle0", type=float, default=0.0)
    ap.add_argument("--topdown", action="store_true")
    ap.add_argument("--reach-samples", type=int, default=2_000_000,
                    help="FK samples for the max-reach radius (uniform over joint limits; cheap, cached)")
    ap.add_argument("--occ-samples", type=int, default=1_500_000,
                    help="FEASIBLE (self-collision-free) FK samples voxelised into the Tier-2 occupancy "
                         "mask; more = fewer Monte-Carlo gaps in the reachable-workspace shape")
    ap.add_argument("--occ-batch", type=int, default=32_000,
                    help="feasible configs requested per sample_configs call; LOWER this if you OOM "
                         "(curobo collision-checks occ-batch x occ-rejection configs at once)")
    ap.add_argument("--occ-rejection", type=int, default=8,
                    help="oversampling factor inside sample_configs (curobo default 50 OOMs a big batch); "
                         "occ-batch x occ-rejection is the real GPU batch -- keep the product ~256k")
    ap.add_argument("--occ-mc-safety", type=float, default=0.11,
                    help="extra dilation (m) on the occupancy beyond the gripper offset, covering the "
                         "Monte-Carlo boundary rind (uniform joint sampling under-covers the fully-"
                         "extended reach edge; measured ~0.075m gap on aloha-agilex). Larger only prunes "
                         "LESS (always safe). Validate after changing embodiment/grid.")
    ap.add_argument("--reach-margin", type=float, default=0.05,
                    help="safety slack (m) added to the prune radius so a reachable cell is never cut")
    ap.add_argument("--gripper-offset", type=float, default=0.12,
                    help="endlink->gripper control-point shift (m); added to the reach radius since the "
                         "grid cell is the gripper while FK measures the endlink (see _world_gripper_to_curobo)")
    ap.add_argument("--reach-recompute", action="store_true",
                    help="force-recompute the FK radius (otherwise the cached per-arm value is reused)")
    # demo-image grid (must match the metric's grid for the picture to match the runs)
    ap.add_argument("--xmin", type=float, default=-0.6); ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.35); ap.add_argument("--ymax", type=float, default=0.35)
    ap.add_argument("--res", type=float, default=0.01)
    ap.add_argument("--zmin", type=float, default=0.78); ap.add_argument("--zmax", type=float, default=1.4)
    ap.add_argument("--zres", type=float, default=0.03)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="stable location for the per-arm envelope artifact the runs load")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help="image location; each run lands in its own <out-dir>/<timestamp>/ subfolder")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"[reach-env] artifacts -> {args.cache_dir}   images -> {out_dir}")
    tm = Timings()

    with tm.section("scene_setup"):
        env = make_occluder_task()()
        env.spawn_occluder = args.occluder
        env.occluder_offset = args.offset
        env.num_occluders = args.num_occluders
        env.occluder_angle0 = args.occluder_angle0
        env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, args.seed, DR_CLEAN))

    arms = ["left", "right"] if args.arms == "both" else [args.arms]
    for arm in arms:
        with tm.section(f"envelope_{arm}"):
            produce_for_arm(env, args, arm, args.cache_dir, out_dir)

    tm.save(out_dir)
    print(f"[reach-env] done. Actual runs now use:  python clearance_metric_3d.py --reach-envelope")
    try:
        env.close_env()
    except Exception:
        pass


if __name__ == "__main__":
    main()
