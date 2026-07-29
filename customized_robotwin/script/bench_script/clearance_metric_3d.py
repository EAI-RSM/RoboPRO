#!/usr/bin/env python3
"""
Scene-difficulty (clearance) metric via curobo batched IK -- 2.5D / z-stack (IN PROGRESS).

This is the sandbox that will grow into the 3D generalisation of clearance_metric.py (the
frozen 2D single-slice tool). It is built up in phases; right now it is still the 2D pipeline
PLUS the Phase-0 gate diagnostic described below.

PHASE 0 (current): does the joint-space edge gate even make sense? The 2.5D widest-path must
only union adjacent FREE cells whose IK configs are close (so the arm can traverse the edge
without a branch jump). That gate is only trustworthy if curobo's per-cell IK solution varies
SMOOTHLY across the FREE region, with sharp jumps only at real config-space seams. Here we
plumb the IK joint config q through the ON sweep (curobo already computes it -- we just stop
discarding it) and, on the current single z-slice, measure the joint jump across every adjacent
FREE-cell pair. If the jump distribution is a smooth bulk with a clean gap up to a few seams,
the gate is meaningful and the gap tells us --gate-tau; if it's a broad smear, IK is branch-
hopping on seed noise and we must warm-start before the gate can be trusted.

Later phases: (1) stack the grid to z, (2) 3D occluder EDT, (3) 26-conn gated widest-path DSU,
(4) held-object collision hook.

This file reuses the scene / grid / IK machinery from reachability_map.py verbatim
(that stays the visualization tool); here we only add the metric on top.

OCCLUDER GEOMETRY (--occ-shape, default mesh): the clearance field and every figure measure against
the TRUE posed collision mesh, re-cut at each z, so the bottle tapers and its neck is thin. The old
solid -- the widest footprint extruded to the cap -- is still available as --occ-shape extruded, but
it invents up to ~4 cm of phantom obstacle around the neck (wider than the gripper), so it makes
climb-over routes look tighter than they are. eps* differs between the two.

USAGE (from the benchmark folder, env sourced + ROBOTWIN_BENCH_TASK=bench):
    python clearance_metric_3d.py --seed 1 --offset 0.2 --arm right --zmin 0.78 --zmax 1.4 --zres 0.03
    #   add --warm-start to also propagate a continuity branch field per slice (slower, one extra IK pass/slice)
    #   add --occ-shape extruded to reproduce the pre-2026-07-24 (prism-occluder) eps* numbers
"""

import argparse
import json
import os
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from setup_paths import setup_paths
setup_paths()

import torch
from lib.continuity import (
    _NEIGH8, _pick_nearest, _wrap_linf, warm_start_branches, warm_start_branches_3d,
)
from lib.ik_grid import (
    _build_ik_solver, _build_ik_solver_no_world, _solve_grid, _solve_grid_q,
    _solve_grid_q_multi, build_grid, grasp_orientation,
)
from lib.labeling import (
    BEYOND, FREE, LABEL_NAMES, OBSTACLE, geometric_envelope, label_volume,
    load_reach_envelope,
)
from lib.metric_config import SeedMetricConfig
from lib.obstacles import (
    _load_collision_mesh, obstacle_centers, occluder_clearance,
    occluder_clearance_3d, occluder_footprint_polys, occluder_footprints_3d,
    occluder_mask_3d, occluder_slice_polys, scene_obstacle_entries,
    surface_distance_to_occluders,
)
from lib.plotting import (
    _draw_eps_sphere, _draw_occluder_solids_3d, _equal_aspect_3d, _line_axis,
    _scene_anchor_markers,
)
from lib.run_io import CLEARANCE_RESULTS_DIR as RESULTS_DIR, Timings
from lib.scene_build import DR_CLEAN, build_cfg
from lib.scene_constants import OCC_HALF_FOOTPRINT, OCCLUDER_COLLISION, PAD_XY
from lib.widest_path import (
    nearest_free_cell, nearest_free_voxel, reconstruct_widest_path,
    reconstruct_widest_path_3d, widest_path_eps, widest_path_eps_3d,
)

from task.occluder_task import make_occluder_task
from metric_diagnostics import (
    phase0_gate_diagnostic, phase1_stack_report, phase2_vertical_report,
    phase3_clearance_report,
)
from metric_viz import (
    LABEL_COLORS, _metric_path3d, feasibility, phase4_visuals, report,
)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
def select_arm(env, args):
    """Resolve which arm the metric runs on and return (arm, planner, grasp_q, grasp_pose, ik),
    with the chosen arm's IK solver already built (so run() doesn't rebuild it).

    args.arm == 'left'/'right' -> use that arm.
    args.arm == 'auto'         -> probe BOTH arms' grasp reachability and pick a reachable one;
                                  nearest arm-base breaks ties (prefer the arm that isn't
                                  over-extending). If neither grasp is reachable, fall back to the
                                  nearest arm and warn (the metric will then read inaccessible)."""
    def planner_for(arm):
        return env.robot.left_planner if arm == "left" else env.robot.right_planner

    if args.arm in ("left", "right"):
        planner = planner_for(args.arm)
        grasp_q, grasp_pose = grasp_orientation(env, args.arm, args.topdown)
        return args.arm, planner, grasp_q, grasp_pose, _build_ik_solver(planner)

    tgt_xy = np.array(env.target_obj.get_pose().p)[:2]
    cands = []
    for arm in ("left", "right"):
        planner = planner_for(arm)
        grasp_q, grasp_pose = grasp_orientation(env, arm, args.topdown)
        base = np.array(planner.robot_origion_pose.p)[:2]
        ref = np.array(grasp_pose[:2]) if grasp_pose is not None else tgt_xy
        dist = float(np.hypot(ref[0] - base[0], ref[1] - base[1]))
        reachable, ik = None, None
        if grasp_pose is not None:
            ik = _build_ik_solver(planner)
            reachable = bool(_solve_grid(env.robot, planner, ik, arm,
                                         np.array([grasp_pose]), chunk=args.chunk)[0])
        cands.append(dict(arm=arm, planner=planner, grasp_q=grasp_q, grasp_pose=grasp_pose,
                          dist=dist, reachable=reachable, ik=ik))
        print(f"[auto-arm] {arm}: grasp reachable={reachable}  base-dist={dist:.3f}m")

    reach = [c for c in cands if c["reachable"]]
    pool = reach if reach else cands
    pool.sort(key=lambda c: c["dist"])
    chosen = pool[0]
    if not reach:
        print(f"[auto-arm] WARNING no arm's grasp is reachable; falling back to nearest "
              f"({chosen['arm']}) -> metric will likely read inaccessible")
    # release IK solvers built for the arm(s) we did not choose
    for c in cands:
        if c is not chosen and c["ik"] is not None:
            del c["ik"]
    torch.cuda.empty_cache()
    ik = chosen["ik"] if chosen["ik"] is not None else _build_ik_solver(chosen["planner"])
    print(f"[auto-arm] chosen: {chosen['arm']}")
    return chosen["arm"], chosen["planner"], chosen["grasp_q"], chosen["grasp_pose"], ik
def phase4_metric(out_dir, args, XX, YY, zs, label, edt, q_gate, grasp_pose, tgt_p, pad_xy, occ_ps,
                  foots, ee_xyz=None):
    """PHASE 4: the 2.5D metric. Seed the grasp voxel and the pad voxel, run the 26-conn widest-path
    (max-min occluder clearance) DSU twice -- UNGATED (reachable + clear only) and GATED on the warm
    branch field (adds the joint-continuity constraint) -- and report eps* for both plus the gated
    climb-over route. eps_gated is the real metric; (eps_ungated vs eps_gated), and especially an
    ungated-merges-but-gated-disconnects case, show how much the branch seams cost the route."""
    free_vol = label == FREE
    carry_z = float(grasp_pose[2]) if grasp_pose is not None else float(np.median(zs))
    grasp_xyz = (np.array(grasp_pose[:3], dtype=float) if grasp_pose is not None
                 else np.array([tgt_p[0], tgt_p[1], carry_z]))
    pad_xyz = np.array([pad_xy[0], pad_xy[1], carry_z])          # pad seed at carry height
    seed_g = nearest_free_voxel(free_vol, XX, YY, zs, grasp_xyz, args.seed_snap)
    seed_p = nearest_free_voxel(free_vol, XX, YY, zs, pad_xyz, args.seed_snap)
    stem = f"metric_seed{args.seed}_{args.arm}"
    if seed_g is None or seed_p is None:
        which = "grasp" if seed_g is None else "pad"
        print(f"[metric] no FREE voxel within {args.seed_snap:.3f}m of the {which} seed -> INACCESSIBLE")
        (out_dir / f"{stem}.json").write_text(json.dumps({"merged_gated": False,
                                                          "reason": f"{which} seed unsnappable"}, indent=2))
        return
    (ga, gd), (pa, pd) = seed_g, seed_p

    def w(v):
        return (float(XX[v[1], v[2]]), float(YY[v[1], v[2]]), float(zs[v[0]]))

    print(f"[metric] grasp seed {tuple(round(c, 3) for c in w(ga))} snap={gd:.3f}m   "
          f"pad seed {tuple(round(c, 3) for c in w(pa))} snap={pd:.3f}m")

    eps_u, bott_u, merged_u = widest_path_eps_3d(label, edt, None, ga, pa, args.gate_tau)
    gated = q_gate is not None
    if gated:
        eps_g, bott_g, merged_g = widest_path_eps_3d(label, edt, q_gate, ga, pa, args.gate_tau)
    else:
        eps_g, bott_g, merged_g = eps_u, bott_u, merged_u
    route = reconstruct_widest_path_3d(free_vol, edt, q_gate, ga, pa, eps_g, args.gate_tau) if merged_g else None
    route_w = [w(v) for v in route] if route else None

    def fmt(eps, merged):
        return "inf (over the top)" if (merged and np.isinf(eps)) else (f"{eps:.3f}m" if merged else "INACCESSIBLE")

    verdict_g, margin_g = feasibility(eps_g, merged_g, args.gripper_r)
    print(f"[metric] eps* UNGATED (reach+clear)  = {fmt(eps_u, merged_u)}")
    print(f"[metric] eps* GATED   (2.5D metric)  = {fmt(eps_g, merged_g)}"
          + (f"   gripper r={args.gripper_r:.3f} -> {verdict_g}"
             + (f" (margin {margin_g:+.3f}m)" if margin_g is not None else "") if gated
             else "   (no warm field -> ungated only; pass --warm-start for the real metric)"))
    if route_w:
        print(f"[metric] gated route: {len(route_w)} voxels, climbs to z={max(p[2] for p in route_w):.3f}")
    if gated and merged_u and not merged_g:
        print("[metric] NOTE: a reachable+clear route EXISTS but the branch gate disconnects it -> no "
              "continuous single-branch climb-over (seams block it); eps_gated reads INACCESSIBLE")

    exact = _metric_path3d(out_dir, args, foots, occ_ps, w(ga), w(pa), (w(bott_g) if bott_g else None),
                           route_w, eps_g, merged_g, tgt_p=tgt_p, ee_xyz=ee_xyz)
    if exact is not None:
        print(f"[metric] bottleneck: EDT eps*={eps_g:.4f}m vs true mesh-surface distance {exact:.4f}m "
              f"(grid bias {eps_g - exact:+.4f}m; the eps* sphere is drawn at the EDT value)")

    summary = {
        "eps_ungated_m": (None if (merged_u and np.isinf(eps_u)) else (round(float(eps_u), 4) if merged_u else 0.0)),
        "eps_gated_m": (None if (merged_g and np.isinf(eps_g)) else (round(float(eps_g), 4) if merged_g else 0.0)),
        "eps_ungated_unbounded": bool(merged_u and np.isinf(eps_u)),
        "eps_gated_unbounded": bool(merged_g and np.isinf(eps_g)),
        "merged_ungated": bool(merged_u), "merged_gated": bool(merged_g),
        "gated": gated, "gate_tau_rad": args.gate_tau, "gripper_r_m": args.gripper_r,
        "feasibility_gated": verdict_g, "margin_m": (None if margin_g is None else round(float(margin_g), 4)),
        "grasp_seed_xyz": [round(c, 4) for c in w(ga)], "pad_seed_xyz": [round(c, 4) for c in w(pa)],
        "bottleneck_gated_xyz": ([round(c, 4) for c in w(bott_g)] if bott_g else None),
        "bottleneck_true_surface_dist_m": (None if exact is None else round(float(exact), 4)),
        "route_len_voxels": (len(route) if route else 0),
        "route_climbs_to_z": (round(max(p[2] for p in route_w), 3) if route_w else None),
        "target_xyz": [round(float(c), 4) for c in tgt_p],
        "gripper_now_xyz": (None if ee_xyz is None else [round(float(c), 4) for c in ee_xyz]),
        "config": {"seed": args.seed, "arm": args.arm, "offset": args.offset,
                   "zmin": args.zmin, "zmax": args.zmax, "zres": args.zres,
                   "occ_shape": args.occ_shape},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    phase4_visuals(out_dir, args, XX, YY, zs, label, edt, foots, w(ga), w(pa), route, route_w,
                   eps_g, merged_g, tgt_p=tgt_p, ee_xyz=ee_xyz)
    print(f"[metric] wrote {stem}.json + 3D path + side/profile/topdown/ceiling figures")


# --------------------------------------------------------------------- step 4b: 3D-legible visuals














# ----------------------------------------------------------------------------- run
def run(args):
    # One folder per run, so re-running a seed never clobbers earlier output.
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] writing to {out_dir}")

    tm = Timings()

    # --- scene (identical setup to reachability_map.run) ---
    with tm.section("scene_setup"):
        env = make_occluder_task()()
        env.spawn_occluder = args.occluder
        env.occluder_offset = args.offset          # ring RADIUS (m), target at centre
        env.num_occluders = args.num_occluders     # bottles equally spaced on the ring (1 = single front)
        env.occluder_angle0 = args.occluder_angle0  # radians; rotates the ring (0 = bottle 0 in front, -y)
        env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, args.seed, DR_CLEAN))
        box_p = np.array(env.occluder.get_pose().p) if getattr(env, "occluder", None) is not None else None
        occ_ps = ([np.array(o.get_pose().p) for o in env.occluders]
                  if getattr(env, "occluders", None) else [])   # ALL ring bottles, for the footprint overlay
        tgt_p = np.array(env.target_obj.get_pose().p)
        pad_xy = np.array(PAD_XY)
    box_txt = f"box=({box_p[0]:.3f},{box_p[1]:.3f})  " if box_p is not None else "box=OFF  "
    print(f"{box_txt}target=({tgt_p[0]:.3f},{tgt_p[1]:.3f})  pad=({pad_xy[0]:.3f},{pad_xy[1]:.3f})")

    # --- z-stack grid (x,y constant across slices; z swept) ---
    with tm.section("grid_build"):
        xs, ys, zs, XX, YY = build_grid(args)
    print(f"[grid] z[{args.zmin},{args.zmax}] zres={args.zres} ({len(zs)} slices)  res={args.res}  "
          f"cells/slice={XX.size} ({len(xs)}x{len(ys)})  total={XX.size * len(zs)} voxels  "
          f"x[{args.xmin},{args.xmax}] y[{args.ymin},{args.ymax}]")

    # --- resolve the grasping arm (explicit or auto) + its IK solver ---
    with tm.section("select_arm"):
        arm, planner, grasp_q, grasp_pose, ik = select_arm(env, args)
    args.arm = arm      # downstream naming / filenames use the resolved arm
    # where the acting gripper currently IS (rest pose), in the same world "gripper pose" convention
    # the grid is swept in -- a scene anchor for the figures, never used by the metric itself
    try:
        ee_pose = env.robot.get_left_ee_pose() if arm == "left" else env.robot.get_right_ee_pose()
        ee_xyz = np.asarray(ee_pose[:3], dtype=float)
        print(f"[{arm}] gripper currently at ({ee_xyz[0]:.3f}, {ee_xyz[1]:.3f}, {ee_xyz[2]:.3f})")
    except Exception as e:
        ee_xyz = None
        print(f"[{arm}] could not read the current gripper pose ({e}); figures will omit that marker")
    if grasp_pose is not None:
        print(f"[{arm}] grasp pose z={grasp_pose[2]:.3f}  (stack z[{args.zmin:.2f},{args.zmax:.2f}]); "
              f"grasp_q={np.array2string(grasp_q, precision=3)}")
    else:
        print(f"[{arm}] using top-down grasp_q={np.array2string(grasp_q, precision=3)}")

    # --- self-check: real grasp pose should be reachable ---
    if grasp_pose is not None:
        with tm.section("grasp_self_check"):
            chk = _solve_grid(env.robot, planner, ik, args.arm, np.array([grasp_pose]), chunk=args.chunk)
        print(f"[{args.arm}] self-check grasp pose reachable = {bool(chk[0])}  (expect True)")

    # --- reach envelope (Tier 1/2): LOAD the precomputed artifact -> prune mask (no FK/IK here) ---
    prune_mask = None
    if args.reach_envelope:
        with tm.section("reach_envelope_load"):
            prune_mask = load_reach_envelope(args.reach_cache_dir, args.arm, xs, ys, zs, XX, YY,
                                             mode=args.reach_mode)
        print(f"[reach-env] masked {int(prune_mask.sum()):,}/{prune_mask.size:,} voxels "
              f"({100 * prune_mask.mean():.1f}%) -> skipped with no IK solve")

    # --- PHASE 1 step 1: label the whole z-stack (+ q per voxel) ---
    with tm.section("label_volume"):
        label, qfield, off_seconds = label_volume(
            env, planner, ik, args.arm, XX, YY, zs, grasp_q, args.chunk, num_seeds=args.ik_seeds,
            free_only=args.free_only, prune_mask=prune_mask)
    counts = {name: int((label == code).sum()) for code, name in LABEL_NAMES.items()}
    print(f"[label] volume FREE={counts['FREE']}  OBSTACLE={counts['OBSTACLE']}  "
          f"BEYOND-REACH={counts['BEYOND-REACH']}  (of {label.size} voxels)")

    # --- warm-start continuity (optional): collect candidate branches, then propagate 2D + 3D ---
    q_warm_vol = q_warm_2d = q_warm_3d = None
    if args.warm_start:
        nz = len(zs); ny, nx = label.shape[1], label.shape[2]
        dof, K = qfield.shape[-1], args.warm_seeds
        free_vol = label == FREE
        print(f"[warm] per-slice multi-branch solve (return_seeds={K}); then 2D per-slice vs "
              f"3D/26-conn continuity propagation")
        with tm.section("warm_multi_solve"):
            cand_q_vol = np.full((nz, ny, nx, K, dof), np.nan, dtype=np.float32)
            cand_ok_vol = np.zeros((nz, ny, nx, K), dtype=bool)
            xr, yr = XX.ravel(), YY.ravel()
            for iz, z in enumerate(zs):
                t_s = time.perf_counter()
                idx = np.flatnonzero(free_vol[iz].ravel())    # Tier 2: solve only the FREE cells
                if idx.size:
                    gp = np.zeros((idx.size, 7))
                    gp[:, 0] = xr[idx]; gp[:, 1] = yr[idx]; gp[:, 2] = z; gp[:, 3:] = grasp_q
                    cand_q, cand_ok = _solve_grid_q_multi(env.robot, planner, ik, args.arm, gp,
                                                          chunk=args.chunk, return_seeds=K, num_seeds=args.ik_seeds)
                    cand_q_vol[iz].reshape(ny * nx, K, dof)[idx] = cand_q     # scatter back into the volume
                    cand_ok_vol[iz].reshape(ny * nx, K)[idx] = cand_ok
                print(f"[warm] slice {iz + 1}/{nz} z={z:.3f}  "
                      f"FREE={idx.size}  {time.perf_counter() - t_s:.2f}s")
        with tm.section("warm_propagate"):
            q_warm_2d = np.full((nz, ny, nx, dof), np.nan, dtype=np.float32)      # per-slice baseline
            for iz in range(nz):
                q_warm_2d[iz] = warm_start_branches(free_vol[iz],
                                                    cand_q_vol[iz].reshape(-1, K, dof),
                                                    cand_ok_vol[iz].reshape(-1, K))
            q_warm_3d = warm_start_branches_3d(free_vol, cand_q_vol, cand_ok_vol)  # enforces vertical
        q_warm_vol = q_warm_3d
        with tm.section("phase2_vertical_report"):
            phase2_vertical_report(out_dir, args, zs, free_vol, q_warm_2d, q_warm_3d)

    # --- step 3: 3D occluder clearance (extruded posed footprints, m) ---
    with tm.section("occluder_edt"):
        foots = occluder_footprints_3d(env, obstacles=args.obstacles)
        if foots:
            print(f"[footprint] obstacle set = {args.obstacles} -> {len(foots)} solid(s)")
            for i, f in enumerate(foots):
                if f["poly"] is not None:
                    bb = tuple(np.round(f["poly"].max(0) - f["poly"].min(0), 3))
                    print(f"[footprint] obs {i}: z-range [{f['zlo']:.3f},{f['zhi']:.3f}]  bbox(x,y)={bb}")
        # occ_ps drives the cylinder fallback and the plot markers; derive it from the
        # footprints so it tracks whatever obstacle set was actually used.
        occ_ps = obstacle_centers(foots) or occ_ps
        edt = occluder_clearance_3d(foots, occ_ps, XX, YY, zs, args.res, args.zres, OCC_HALF_FOOTPRINT,
                                    shape=args.occ_shape)
    print(f"[edt] occluder solid = {args.occ_shape}"
          + ("  (true posed collision mesh, re-cut at every z)" if args.occ_shape == "mesh"
             else "  (widest footprint extruded over the full height -- over-fills the neck)"))
    freev = label == FREE
    if np.isinf(edt).all():
        print("[edt] no occluders -> occluder-clearance unbounded (nothing to squeeze past)")
    elif freev.any():
        fe = edt[freev][np.isfinite(edt[freev])]
        if fe.size:
            print(f"[edt] occluder-clearance over FREE voxels (m): min={fe.min():.3f} "
                  f"median={np.median(fe):.3f} max={fe.max():.3f}")

    with tm.section("phase3_clearance_report"):
        phase3_clearance_report(out_dir, args, xs, ys, zs, label, edt, foots, tgt_p=tgt_p, ee_xyz=ee_xyz)

    # (reach-envelope viz lives in the producer reach_envelope.py, not here -- this is an actual run)

    # --- step 4: gated 26-conn widest-path DSU -> the 2.5D metric (eps*) ---
    with tm.section("metric_dsu"):
        phase4_metric(out_dir, args, XX, YY, zs, label, edt, q_warm_3d, grasp_pose, tgt_p, pad_xy,
                      occ_ps, foots, ee_xyz=ee_xyz)

    with tm.section("phase1_report"):
        phase1_stack_report(out_dir, args, xs, ys, zs, label, qfield, q_warm_vol)

    # --- persist the raw volumes so the run is re-analysable offline (no GPU re-run needed) ---
    with tm.section("save_data"):
        save_kw = {"label": label, "qfield": qfield, "edt": edt, "xs": xs, "ys": ys, "zs": zs}
        if q_warm_3d is not None:
            save_kw.update(q_warm_2d=q_warm_2d, q_warm_3d=q_warm_3d)
        if prune_mask is not None:
            save_kw["reach_envelope"] = prune_mask
        np.savez_compressed(out_dir / "stack_data.npz", **save_kw)
    print(f"[run] saved stack_data.npz ({', '.join(save_kw)}) for offline analysis")

    tm.save(out_dir, off_seconds=off_seconds)
    print(f"[run] steps 1-4 (z-stack, labels, 3D warm propagation, 3D clearance, gated metric) done.")
    print(f"[run] outputs in {out_dir}")

    try:
        env.close_env()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--offset", type=float, default=0.2,
                    help="occluder ring RADIUS (m), target at centre")
    ap.add_argument("--num-occluders", type=int, default=1,
                    help="number of olive-oil bottles equally spaced on the ring (1 = single front)")
    ap.add_argument("--occluder-angle0", type=float, default=0.0,
                    help="radians; rotates the whole ring (0 = bottle 0 directly in front, -y)")
    ap.add_argument("--no-occluder", dest="occluder", action="store_false",
                    help="map the bare (table-only) world instead of spawning the occluder")
    ap.set_defaults(occluder=True)
    ap.add_argument("--arm", choices=["left", "right", "auto"], default="auto",
                    help="grasping arm to compute the metric for; 'auto' probes both arms' grasp "
                         "reachability and picks a reachable one (nearest arm-base breaks ties)")
    ap.add_argument("--zmin", type=float, default=None, help="stack floor (m); ~grasp height (table ~0.74)")
    ap.add_argument("--zmax", type=float, default=None, help="stack ceiling (m); above the occluder top")
    ap.add_argument("--zres", type=float, default=None, help="vertical slice spacing (m); K = (zmax-zmin)/zres")
    ap.add_argument("--topdown", action="store_true", help="use a top-down quat instead of the side grasp")
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xmax", type=float, default=None)
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--ymax", type=float, default=None)
    ap.add_argument("--res", type=float, default=None, help="grid resolution (m)")
    ap.add_argument("--occ-shape", choices=["mesh", "extruded"], default=None,
                    help="geometry the occluder-clearance field (and the figures) use. 'mesh' re-cuts "
                         "the TRUE posed collision mesh at every z, so the bottle tapers and the neck "
                         "is thin -- faithful to what curobo collides against. 'extruded' is the older "
                         "solid: the widest footprint held constant to the cap, which invents up to "
                         "~4cm of phantom obstacle around the neck. CHANGING THIS CHANGES eps*; pass "
                         "'extruded' to reproduce pre-2026-07-24 numbers.")
    ap.add_argument("--obstacles", choices=["all", "occluders"], default=None,
                    help="which actors the clearance field measures against. 'all' = every mesh in "
                         "env.collision_list (the registry curobo's update_world uses) except the "
                         "target and the pad, so procedural table CLUTTER counts as an obstacle "
                         "alongside the occluder ring -- the metric's world then matches the "
                         "planner's. 'occluders' = the curated olive-oil ring only, the "
                         "pre-2026-07-27 behaviour. CHANGING THIS CHANGES eps* AND WHAT IT MEANS: "
                         "under 'all' eps* is clearance to the nearest scene obstacle, not to the "
                         "occluder, so values are not comparable across the two.")
    ap.add_argument("--gate-tau", type=float, default=None,
                    help="joint-space gate (rad): union adjacent FREE voxels only if the max per-joint "
                         "warm-config jump is <= tau. Set from the phase-0/2 histogram gap (~0.2-0.35). "
                         "Only active with --warm-start; without it the DSU runs ungated.")
    ap.add_argument("--seed-snap", type=float, default=None,
                    help="max distance to snap a DSU seed (grasp/pad) to the nearest FREE cell (m)")
    ap.add_argument("--boxed-in-radius", type=float, default=0.05,
                    help="bottleneck within this distance of the target counts as 'boxed-in' (m)")
    ap.add_argument("--gripper-r", type=float, default=0.03,
                    help="gripper half-width (m), compared to eps* at READ time only; never baked "
                         "into the metric (eps* stays embodiment-free)")
    ap.add_argument("--warm-start", action="store_true",
                    help="Phase-0 test: also compute a continuity-propagated branch field (multi-seed "
                         "IK + BFS nearest-branch) and compare its joint-jump distribution to the raw "
                         "best-cost field. Tail collapse => branch-hopping noise; survives => real seams")
    ap.add_argument("--warm-seeds", type=int, default=None,
                    help="return_seeds K for the multi-branch solve (candidate branches offered per cell)")
    ap.add_argument("--ik-seeds", type=int, default=None,
                    help="curobo num_seeds per pose (Tier-1 speedup; default 100 is overkill). Lower = "
                         "faster but may miss a few hard-but-reachable poses; raise (~60) for a final run.")
    ap.add_argument("--free-only", action="store_true", default=None,
                    help="skip the collision-OFF sweep (Tier-4 speedup): ~halves label_volume IK. Loses "
                         "only the OBSTACLE/BEYOND split in the figures; the metric is unaffected. Every "
                         "run reports the projected --free-only time whether or not this is set.")
    ap.add_argument("--reach-envelope", action="store_true",
                    help="Tier-1/2 speedup: LOAD the pose-independent reach envelope precomputed by "
                         "reach_envelope.py and skip IK on every voxel it marks unreachable for ANY "
                         "grasp pose. Errors if the artifact is missing -- run "
                         "`python reach_envelope.py --arms both` once first.")
    ap.add_argument("--reach-mode", choices=["occupancy", "sphere"], default="occupancy",
                    help="which precomputed mask to apply: 'occupancy' (Tier 2, the real reachable-"
                         "workspace shape -- prunes most, but requires the artifact grid to match this "
                         "run's grid) or 'sphere' (Tier 1, the grid-independent max-reach ball).")
    ap.add_argument("--reach-cache-dir", default=str(RESULTS_DIR / "_reach_cache"),
                    help="where reach_envelope.py stored the per-arm envelope artifact (stable across runs)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="IK poses per batch; lower if you hit CUDA OOM (planners already use ~9GB)")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="results location; each run lands in its own <out-dir>/<timestamp>/ subfolder")
    args = ap.parse_args()
    metric_config = SeedMetricConfig.from_args(args)
    for field, value in vars(metric_config).items():
        setattr(args, field, value)
    run(args)


if __name__ == "__main__":
    main()
