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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# repo-root results dir for this metric, anchored to THIS file (same layout as reachability_map);
# separate folder from the frozen 2D tool so the 3D sandbox never mixes into clearance_metric/




# ----------------------------------------------------------------------------- grid




# --------------------------------------------------------- reach envelope: CONSUMER side (Tier 1/2)
# The pose-independent reach envelope is PRECOMPUTED ONCE by reach_envelope.py (the producer) and
# stored as an artifact; this file only LOADS + APPLIES it. See reach_envelope.py for how the sphere
# radius and the occupancy mask are computed and why both are strict. Keeping the two apart is
# deliberate: the envelope depends only on the robot + arm base (never the scene/seed/occluder), so it
# is computed once and reused by every actual run here without a single FK or IK solve.


















# label codes for the three-way raster










# 3D 26-neighbourhood (used by warm_start_branches_3d; forward-referenced, resolved at call time)








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


# ----------------------------------------------------------------------------- report + viz
LABEL_COLORS = {BEYOND: "#9e9e9e", OBSTACLE: "#d84315", FREE: "#2e7d32"}


def feasibility(eps_star, merged, r):
    """Derive an embodiment feasibility verdict at READ time. eps* itself stays robot-free;
    the gripper half-width r is only compared here, never baked into the metric."""
    if not merged:
        return "INACCESSIBLE", None
    if np.isinf(eps_star):
        return "unbounded (no obstacles)", None
    margin = eps_star - r
    return ("fits" if margin >= 0 else "INFEASIBLE"), margin


def _overlays(ax, polys, occ_ps, tgt_p, pad_xy, seed_t_xy, seed_p_xy, bott_xy, path_xy):
    if path_xy is not None and len(path_xy) > 1:      # widest-path route, drawn under the markers
        px, py = zip(*path_xy)
        ax.plot(px, py, "-", color="yellow", lw=2.2, alpha=0.9, label="widest path")
    if polys is not None and any(p is not None for p in polys):
        from matplotlib.patches import Polygon as MplPolygon
        first = True
        for p in polys:                        # true posed mesh footprint per occluder
            if p is None:
                continue
            ax.add_patch(MplPolygon(p, closed=True, fill=False, edgecolor="red", lw=2,
                                    label=("occluder" if first else None)))
            first = False
    else:                                      # fallback: circle of radius OCC_HALF_FOOTPRINT
        for i, op in enumerate(occ_ps):
            ax.add_patch(plt.Circle((op[0], op[1]), OCC_HALF_FOOTPRINT, fill=False,
                                    edgecolor="red", lw=2, label=("occluder" if i == 0 else None)))
    ax.plot(tgt_p[0], tgt_p[1], "b*", ms=16, label="target")
    ax.plot(pad_xy[0], pad_xy[1], "ms", ms=11, label="pad")
    if seed_t_xy is not None:
        ax.plot(*seed_t_xy, "o", mfc="none", mec="cyan", mew=2, ms=13, label="grasp seed")
    if seed_p_xy is not None:
        ax.plot(*seed_p_xy, "o", mfc="none", mec="magenta", mew=2, ms=13, label="pad seed")
    if bott_xy is not None:
        ax.plot(*bott_xy, "kX", ms=8, mew=1.5, label="bottleneck (eps*)")


def report(args, out_dir, XX, YY, label, edt, box_p, occ_ps, polys, tgt_p, pad_xy,
           seed_t_xy, seed_p_xy, eps_star, bott_xy, merged, boxed_dist, counts, path_xy):
    """Write summary (json + txt), cache the raster (npz -- keeps the BEYOND label for a later
    reach-edge channel), and save the two-panel figure (labels + clearance heatmap)."""
    verdict, margin = feasibility(eps_star, merged, args.gripper_r)
    tag = "topdown" if args.topdown else "sidegrasp"
    occ = "occ" if box_p is not None else "noocc"
    stem = f"clearance_seed{args.seed}_off{args.offset}_z{args.z:.2f}_{args.arm}_{tag}_{occ}"

    boxed_in = bool(boxed_dist is not None and boxed_dist <= args.boxed_in_radius)
    summary = {
        "eps_star_m": (None if np.isinf(eps_star) else round(float(eps_star), 4)),
        "eps_star_unbounded": bool(np.isinf(eps_star)),
        "merged": bool(merged),
        "bottleneck_xy": ([round(bott_xy[0], 4), round(bott_xy[1], 4)] if bott_xy else None),
        "boxed_in": boxed_in,
        "boxed_in_dist_m": (None if boxed_dist is None else round(float(boxed_dist), 4)),
        "gripper_r_m": args.gripper_r,
        "feasibility": verdict,
        "margin_m": (None if margin is None else round(float(margin), 4)),
        "label_counts": counts,
        "config": {"seed": args.seed, "offset": args.offset, "z": args.z, "arm": args.arm,
                   "res": args.res, "topdown": args.topdown, "occluder": box_p is not None},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    np.savez_compressed(out_dir / f"{stem}.npz",
                        label=label, edt=edt, XX=XX, YY=YY,
                        box_p=(box_p if box_p is not None else np.array([])),
                        occ_ps=(np.array(occ_ps) if occ_ps else np.array([])),
                        path_xy=(np.array(path_xy) if path_xy else np.array([])),
                        tgt_p=tgt_p, pad_xy=pad_xy, res=args.res)

    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))

    cmap = ListedColormap([LABEL_COLORS[BEYOND], LABEL_COLORS[OBSTACLE], LABEL_COLORS[FREE]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    axL.imshow(label, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal")
    _overlays(axL, polys, occ_ps, tgt_p, pad_xy, seed_t_xy, seed_p_xy, bott_xy, path_xy)
    # eps*-radius circle at the bottleneck: it should just touch the nearest occluder
    # (eps* IS the distance from the bottleneck to that occluder), so a larger eps* = a
    # larger circle that barely kisses a bottle.
    if merged and bott_xy is not None and np.isfinite(eps_star) and eps_star > 0:
        axL.add_patch(plt.Circle(bott_xy, eps_star, fill=False, edgecolor="#00e5ff",
                                  ls="--", lw=1.8, label=f"eps* radius ({eps_star:.3f} m)"))
    from matplotlib.patches import Patch
    proxies = [Patch(color=LABEL_COLORS[c], label=LABEL_NAMES[c]) for c in (FREE, OBSTACLE, BEYOND)]
    h, _ = axL.get_legend_handles_labels()
    axL.legend(handles=proxies + h, loc="upper right", fontsize=8)
    axL.set_title("Three-way labels"); axL.set_xlabel("x (m)"); axL.set_ylabel("y (m)")

    disp = np.where(np.isfinite(edt), edt, np.nan)
    im = axR.imshow(disp, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="clearance to nearest OCCLUDER (m)")
    _overlays(axR, polys, occ_ps, tgt_p, pad_xy, seed_t_xy, seed_p_xy, bott_xy, path_xy)
    axR.legend(loc="upper right", fontsize=8)
    axR.set_title("Occluder clearance (m)"); axR.set_xlabel("x (m)"); axR.set_ylabel("y (m)")

    eps_txt = "inf" if np.isinf(eps_star) else (f"{eps_star:.3f} m" if merged else "0 (inaccessible)")
    fig.suptitle(f"Clearance metric  |  seed {args.seed}, offset {args.offset}, z={args.z:.2f}, "
                 f"arm={args.arm}, occluder {'ON' if box_p is not None else 'OFF'}\n"
                 f"eps* = {eps_txt}   gripper r = {args.gripper_r:.3f} m   ->  {verdict}"
                 + ("" if margin is None else f"  (margin {margin:+.3f} m)"),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"{stem}.png", dpi=130); plt.close(fig)
    print(f"[report] wrote {stem}.png / .json / .npz  ->  feasibility: {verdict}")


# ----------------------------------------------------------------------------- phase 0 diagnostic
GATE_TAU_CANDIDATES = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def _jump_field(free, qfield):
    """For a given per-cell config field, the 8-connected FREE-FREE joint-jump statistics the gate
    would see. Returns (cellmax (ny,nx) NaN off-FREE = max jump to any FREE neighbour, edges 1D =
    each undirected FREE-FREE edge jump once). Cells with non-finite q (e.g. no warm branch) skip."""
    ny, nx = free.shape
    cellmax = np.full(free.shape, np.nan)
    edges = []
    for iy in range(ny):
        for ix in range(nx):
            if not free[iy, ix] or not np.all(np.isfinite(qfield[iy, ix])):
                continue
            qa = qfield[iy, ix]
            best = 0.0
            for dy, dx in _NEIGH8:
                jy, jx = iy + dy, ix + dx
                if (0 <= jy < ny and 0 <= jx < nx and free[jy, jx]
                        and np.all(np.isfinite(qfield[jy, jx]))):
                    d = _wrap_linf(qa, qfield[jy, jx])
                    best = max(best, d)
                    if (jy, jx) > (iy, ix):    # count each undirected edge once
                        edges.append(d)
            cellmax[iy, ix] = best
    return cellmax, np.asarray(edges)


def _print_jump_stats(tag, edges):
    if edges.size == 0:
        print(f"[phase0] {tag}: no FREE-FREE edges"); return
    pct = np.percentile(edges, [50, 90, 95, 99])
    passfrac = "  ".join(f"{t:.2f}:{float((edges <= t).mean()):.3f}" for t in GATE_TAU_CANDIDATES)
    print(f"[phase0] {tag}: edges={edges.size}  jump(rad) median={pct[0]:.3f} p90={pct[1]:.3f} "
          f"p95={pct[2]:.3f} p99={pct[3]:.3f} max={edges.max():.3f}")
    print(f"[phase0] {tag}: pass-frac (jump<=tau)  {passfrac}")


def _draw_pair(fig, axmap, axhist, extent, cellmax, edges, title):
    im = axmap.imshow(cellmax, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=axmap, fraction=0.046, pad=0.04, label="max joint jump to a FREE neighbour (rad)")
    axmap.set_title(f"{title}: per-cell roughness\nbright = branch switch"); axmap.set_xlabel("x (m)")
    axmap.set_ylabel("y (m)")
    if edges.size:
        axhist.hist(edges, bins=60, color="#3949ab"); axhist.set_yscale("log")
        for t in GATE_TAU_CANDIDATES:
            axhist.axvline(t, color="#d84315", ls="--", lw=1)
            axhist.text(t, axhist.get_ylim()[1], f"{t:.2f}", rotation=90, va="top", ha="right",
                        fontsize=7, color="#d84315")
    axhist.set_title(f"{title}: edge jump dist\n(dashed = candidate --gate-tau)")
    axhist.set_xlabel("joint jump (rad)"); axhist.set_ylabel("edge count (log)")


def phase0_gate_diagnostic(out_dir, args, XX, YY, label, qfield, q_warm=None, z_slice=0.0):
    """PHASE 0: is the joint-space edge gate trustworthy on this slice? (single-slice diagnostic)

    For every 8-connected pair of FREE cells, measure the joint jump = max over joints of the
    wrapped |dq| (radians) -- the exact quantity the gate would threshold. A gate is meaningful iff
    that jump distribution is a smooth LOW bulk (adjacent cells share an IK branch) with a clean gap
    to a few HIGH-jump seam edges (real branch switches); the gap is where --gate-tau belongs. A
    broad smear with no gap instead means curobo is branch-hopping on seed noise -> a gate would cut
    good edges.

    RAW field = curobo's single best-cost solution per cell (what --warm-start off measures). If a
    warm field is supplied (--warm-start), it is the continuity-propagated branch assignment; the
    two are drawn/printed side by side. If the warm field's tail COLLAPSES relative to raw, the raw
    smear was branch-hopping noise (fixable, gate becomes usable); if the tail SURVIVES, the seams
    are genuine config-space marginality that a warm-start can't remove. Read-only otherwise."""
    free = label == FREE
    nfree = int(free.sum())
    cellmax_r, edges_r = _jump_field(free, qfield)
    print(f"[phase0] FREE cells={nfree}")
    _print_jump_stats("RAW ", edges_r)

    have_warm = q_warm is not None
    if have_warm:
        cellmax_w, edges_w = _jump_field(free, q_warm)
        missing = int((free & ~np.all(np.isfinite(q_warm), axis=-1)).sum())
        _print_jump_stats("WARM", edges_w)
        print(f"[phase0] WARM: {missing} FREE cells had no converged candidate branch (left blank)")
        if edges_r.size and edges_w.size:
            print(f"[phase0] tail comparison p99  raw={np.percentile(edges_r,99):.3f} -> "
                  f"warm={np.percentile(edges_w,99):.3f}  (collapse => noise, survives => real seams)")
    else:
        print("[phase0] read: smooth bulk + clean gap => gate meaningful; broad smear, no gap => "
              "branch-hopping, re-run with --warm-start to test if it collapses")

    stem = f"gate_diagnostic_seed{args.seed}_z{z_slice:.2f}_{args.arm}" + ("_warm" if have_warm else "")
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    if have_warm:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        _draw_pair(fig, axes[0][0], axes[0][1], extent, cellmax_r, edges_r, "RAW (best-cost)")
        _draw_pair(fig, axes[1][0], axes[1][1], extent, cellmax_w, edges_w, "WARM (continuity)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        _draw_pair(fig, axes[0], axes[1], extent, cellmax_r, edges_r, "RAW (best-cost)")
    fig.suptitle(f"Phase 0 gate diagnostic  |  seed {args.seed}, z={z_slice:.2f}, arm={args.arm}"
                 + ("  |  raw vs warm-start" if have_warm else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / f"{stem}.png", dpi=130); plt.close(fig)
    print(f"[phase0] wrote {stem}.png")


def phase1_stack_report(out_dir, args, xs, ys, zs, label, qfield, q_warm_vol=None):
    """PHASE 1 (stack): per-slice reachability + config-roughness across the z-stack. Prints a
    compact per-height table (FREE count + raw/warm joint-jump median/p99) and writes a montage of
    per-slice roughness maps on a shared 0..1 rad colour scale, so heights are directly comparable.
    No 3D gate yet -- this only verifies the stack builds and shows how FREE-space and IK smoothness
    evolve with height (e.g. does the arm still reach up near the bottle's top for a climb-over?)."""
    nz = len(zs)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    cm_raw, cm_warm = [], []
    header = "[stack]    z    FREE    raw med/p99" + ("     warm med/p99" if q_warm_vol is not None else "")
    print(header)
    for iz, z in enumerate(zs):
        free = label[iz] == FREE
        cmr, er = _jump_field(free, qfield[iz]); cm_raw.append(cmr)
        line = f"[stack] {z:5.2f}  {int(free.sum()):5d}    "
        line += (f"{np.median(er):.3f}/{np.percentile(er, 99):.3f}" if er.size else "  -  ")
        if q_warm_vol is not None:
            cmw, ew = _jump_field(free, q_warm_vol[iz]); cm_warm.append(cmw)
            line += "      " + (f"{np.median(ew):.3f}/{np.percentile(ew, 99):.3f}" if ew.size else "  -  ")
        print(line)

    def _montage(cmlist, tag):
        ncol = min(5, nz)
        nrow = int(np.ceil(nz / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
        im = None
        for k in range(nrow * ncol):
            ax = axes[k // ncol][k % ncol]
            if k >= nz:
                ax.axis("off"); continue
            im = ax.imshow(cmlist[k], origin="lower", extent=extent, cmap="viridis",
                           vmin=0.0, vmax=1.0, aspect="equal")
            ax.set_title(f"z={zs[k]:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        if im is not None:
            fig.colorbar(im, ax=list(axes.ravel()), fraction=0.02, pad=0.02,
                         label="max joint jump to neighbour (rad, capped at 1.0)")
        fig.suptitle(f"Phase 1 stack roughness ({tag})  |  seed {args.seed}, arm {args.arm}", fontsize=12)
        stem = f"stack_roughness_{tag}_seed{args.seed}_{args.arm}"
        fig.savefig(out_dir / f"{stem}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"[stack] wrote {stem}.png")

    _montage(cm_raw, "raw")
    if q_warm_vol is not None:
        _montage(cm_warm, "warm")


def _vertical_edges_by_z(free_vol, q_vol):
    """List (len nz-1) of arrays: FREE-FREE VERTICAL-edge joint jumps for each z->z+1 transition
    (each voxel vs the voxel directly above it). This is the continuity the per-slice 2D propagation
    never enforces and the 3D propagation does -- the edges a climb-over route rides on."""
    nz = free_vol.shape[0]
    out = []
    for iz in range(nz - 1):
        both = free_vol[iz] & free_vol[iz + 1]
        e = []
        for iy, ix in zip(*np.nonzero(both)):
            qa, qb = q_vol[iz, iy, ix], q_vol[iz + 1, iy, ix]
            if np.all(np.isfinite(qa)) and np.all(np.isfinite(qb)):
                e.append(_wrap_linf(qa, qb))
        out.append(np.asarray(e))
    return out


def _edge_stats(e):
    """Compact machine-readable summary of an edge-jump array."""
    if e.size == 0:
        return {"edges": 0}
    return {"edges": int(e.size), "median": round(float(np.median(e)), 4),
            "p90": round(float(np.percentile(e, 90)), 4), "p99": round(float(np.percentile(e, 99)), 4),
            "max": round(float(e.max()), 4),
            "pass_at": {f"{t:.2f}": round(float((e <= t).mean()), 4) for t in GATE_TAU_CANDIDATES}}


def phase2_vertical_report(out_dir, args, zs, free_vol, q_warm_2d, q_warm_3d):
    """PHASE 2 check: does the 3D propagation actually buy vertical continuity? Compares the joint jump
    across vertical (between-slice) FREE-FREE edges for the per-slice-2D-propagated field vs the
    3D-propagated field. The 2D field never saw vertical neighbours, so its columns are branch-
    inconsistent -> a fat vertical-jump tail; the 3D field should collapse that tail (leaving only real
    seams). Prints stats, writes an overlaid histogram AND a phase2_vertical_*.json (overall + per-z
    transition) so the result is machine-readable, not just a picture."""
    by2 = _vertical_edges_by_z(free_vol, q_warm_2d)
    by3 = _vertical_edges_by_z(free_vol, q_warm_3d)
    e2 = np.concatenate(by2) if by2 else np.array([])
    e3 = np.concatenate(by3) if by3 else np.array([])
    for tag, e in (("2D-per-slice", e2), ("3D-propagated", e3)):
        if e.size:
            pct = np.percentile(e, [50, 90, 99])
            print(f"[phase2] vertical jump {tag}: edges={e.size} median={pct[0]:.3f} p90={pct[1]:.3f} "
                  f"p99={pct[2]:.3f} max={e.max():.3f}  pass@0.35={float((e <= 0.35).mean()):.3f}")

    stem = f"phase2_vertical_seed{args.seed}_{args.arm}"
    data = {"config": {"seed": args.seed, "arm": args.arm, "zmin": args.zmin, "zmax": args.zmax,
                       "zres": args.zres, "gate_tau_candidates": list(GATE_TAU_CANDIDATES)},
            "vertical_2d_per_slice": _edge_stats(e2),
            "vertical_3d_propagated": _edge_stats(e3),
            "per_z_transition": [{"z_lo": round(float(zs[i]), 3), "z_hi": round(float(zs[i + 1]), 3),
                                  "stats_2d": _edge_stats(by2[i]), "stats_3d": _edge_stats(by3[i])}
                                 for i in range(len(by2))]}
    (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2))

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0, np.pi, 60)
    if e2.size:
        ax.hist(e2, bins=bins, alpha=0.5, color="#d84315", label="2D per-slice (no vertical continuity)")
    if e3.size:
        ax.hist(e3, bins=bins, alpha=0.5, color="#3949ab", label="3D propagated (vertical continuity)")
    ax.set_yscale("log")
    for t in GATE_TAU_CANDIDATES:
        ax.axvline(t, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("vertical-edge joint jump (rad)"); ax.set_ylabel("edge count (log)")
    ax.set_title(f"Phase 2: vertical continuity 2D vs 3D propagation  |  seed {args.seed}, arm {args.arm}")
    ax.legend()
    fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[phase2] wrote {stem}.png + {stem}.json")


# ----------------------------------------------------------------------------- step 3: 3D occluder EDT
















def phase3_clearance_report(out_dir, args, xs, ys, zs, label, edt, foots, tgt_p=None, ee_xyz=None):
    """PHASE 3 sanity: per-slice montage of the 3D occluder-clearance field (0 inside a footprint,
    growing outward; opening up above the bottle top where there is no occluder), footprint outline
    overlaid where it exists. Prints per-z clearance-over-FREE stats so the height profile is legible."""
    nz = len(zs)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    disp = np.where(np.isfinite(edt), edt, np.nan)
    vmax = float(np.nanpercentile(disp, 99)) if np.isfinite(disp).any() else 1.0
    print("[phase3]    z    FREE   clearance over FREE (min/med/max, m)")
    for iz, z in enumerate(zs):
        free = label[iz] == FREE
        fe = edt[iz][free]
        fe = fe[np.isfinite(fe)]
        s = (f"{fe.min():.3f}/{np.median(fe):.3f}/{fe.max():.3f}" if fe.size else "  -  ")
        print(f"[phase3] {z:5.2f}  {int(free.sum()):5d}   {s}")
    ncol = min(5, nz)
    nrow = int(np.ceil(nz / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    im = None
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= nz:
            ax.axis("off"); continue
        im = ax.imshow(disp[k], origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=vmax,
                       aspect="equal")
        if foots:
            from matplotlib.patches import Polygon as MplPolygon
            for f in foots:
                # outline the occluder as it exists AT THIS HEIGHT: the true cross-section under
                # --occ-shape mesh (so the neck visibly narrows), else the constant footprint
                loops = occluder_slice_polys(f, float(zs[k])) if args.occ_shape == "mesh" else []
                if loops:
                    for l in loops:
                        ax.add_patch(MplPolygon(l, closed=True, fill=False, edgecolor="red", lw=1.2))
                elif f["poly"] is not None and f["zlo"] - 1e-9 <= zs[k] <= f["zhi"] + 1e-9:
                    ax.add_patch(MplPolygon(f["poly"], closed=True, fill=False, edgecolor="red", lw=1.2))
        _scene_anchor_markers(ax, tgt_p, ee_xyz, args.arm)
        ax.set_title(f"z={zs[k]:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=list(axes.ravel()), fraction=0.02, pad=0.02, label="clearance to occluder (m)")
    fig.suptitle(f"Phase 3 occluder clearance  |  seed {args.seed}, arm {args.arm}", fontsize=12)
    stem = f"stack_clearance_seed{args.seed}_{args.arm}"
    fig.savefig(out_dir / f"{stem}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[phase3] wrote {stem}.png")


# ----------------------------------------------------------------------------- step 4: gated 3D DSU














def _metric_path3d(out_dir, args, foots, occ_ps, g_xyz, p_xyz, bott_xyz, route_w, eps_star, merged,
                   tgt_p=None, ee_xyz=None):
    """3D view of the gated climb-over route through the stack: occluder solids, the route, the scene
    anchors (target bottle spawn + the gripper's current pose), and the eps* sphere sitting on the
    bottleneck."""
    fig = plt.figure(figsize=(9.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    _draw_occluder_solids_3d(ax, foots, args.occ_shape)
    if route_w and len(route_w) > 1:
        rx, ry, rz = zip(*route_w)
        ax.plot(rx, ry, rz, "-", color="gold", lw=2.5, label="gated widest path")
    ax.scatter(*g_xyz, c="cyan", marker="o", s=80, label="grasp seed")
    ax.scatter(*p_xyz, c="magenta", marker="s", s=70, label="pad seed")
    if tgt_p is not None:
        ax.scatter(tgt_p[0], tgt_p[1], tgt_p[2], c="blue", marker="*", s=200,
                   label="target bottle (spawn)")
    if ee_xyz is not None:
        ax.scatter(ee_xyz[0], ee_xyz[1], ee_xyz[2], c="darkorange", marker="^", s=110,
                   edgecolors="k", linewidths=0.5, label=f"gripper now ({args.arm})")
    if bott_xyz is not None:
        ax.scatter(*bott_xyz, c="black", marker="X", s=80, label="bottleneck eps*")

    # eps* sphere: radius = the bottleneck's clearance, so it should just touch the nearest occluder
    exact = None
    drew_sphere = merged and bott_xyz is not None and np.isfinite(eps_star) and eps_star > 0
    if drew_sphere:
        _draw_eps_sphere(ax, bott_xyz, float(eps_star))
        exact = surface_distance_to_occluders(foots, bott_xyz)

    corners = [np.asarray(g_xyz), np.asarray(p_xyz), tgt_p, ee_xyz]
    for f in (foots or []):
        if f["poly"] is not None:
            corners += [np.array([f["poly"][:, 0].min(), f["poly"][:, 1].min(), f["zlo"]]),
                        np.array([f["poly"][:, 0].max(), f["poly"][:, 1].max(), f["zhi"]])]
    if route_w:
        corners += [np.asarray(route_w).min(axis=0), np.asarray(route_w).max(axis=0)]
    if drew_sphere:
        corners += [np.asarray(bott_xyz) - eps_star, np.asarray(bott_xyz) + eps_star]
    _equal_aspect_3d(ax, corners)

    eps_txt = "inf" if (merged and np.isinf(eps_star)) else (f"{eps_star:.3f} m" if merged else "INACCESSIBLE")
    sub = ""
    if exact is not None:
        sub = (f"\ntrue mesh-surface distance at the bottleneck = {exact:.3f} m  "
               f"(EDT - true = {float(eps_star) - exact:+.3f} m, grid bias)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"2.5D metric route  |  seed {args.seed}, arm {args.arm}, occluder geom={args.occ_shape}"
                 f"\neps* (gated) = {eps_txt}{sub}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    stem = f"metric_path3d_seed{args.seed}_{args.arm}"
    fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return exact


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




def _viz_side_elevation(out_dir, args, XX, YY, zs, label, foots, g_xyz, p_xyz, route_w, eps_star,
                        merged, tgt_p=None, ee_xyz=None):
    """SIDE ELEVATION: the 3D scene projected onto the vertical plane through grasp->pad. Background =
    labels sampled along that line at every height; the bottle is a red box (footprint projected onto
    the line x its z-range); the route arcs up and over. The most legible 'is it climbing over?' view."""
    xs, ys = XX[0], YY[:, 0]
    p0, u, L = _line_axis(g_xyz[:2], p_xyz[:2])
    ns = max(2, int(round(L / args.res)) + 1)
    svals = np.linspace(0, L, ns)
    img = np.full((len(zs), ns), BEYOND, dtype=np.int8)
    for j, s in enumerate(svals):
        px, py = p0 + s * u
        ix = int(np.argmin(np.abs(xs - px)))
        iy = int(np.argmin(np.abs(ys - py)))
        img[:, j] = label[:, iy, ix]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = ListedColormap([LABEL_COLORS[BEYOND], LABEL_COLORS[OBSTACLE], LABEL_COLORS[FREE]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(img, origin="lower", extent=[0, L, zs.min(), zs.max()], cmap=cmap, norm=norm, aspect="auto")
    if foots:
        for f in foots:
            if f["poly"] is None:
                continue
            # mesh mode: per-height silhouette from the real cross-sections (the bottle necks in);
            # extruded mode: the constant-width box the metric actually used
            sil = []
            if args.occ_shape == "mesh" and f.get("mesh") is not None:
                # sampled on its OWN fine z-grid (not the coarse stack) so the neck reads as a curve
                for z in np.linspace(f["zlo"] + 1e-4, f["zhi"] - 1e-4, 60):
                    ss = [((loop - p0) @ u) for loop in occluder_slice_polys(f, float(z))]
                    if ss:
                        allss = np.concatenate(ss)
                        sil.append((float(allss.min()), float(allss.max()), float(z)))
            if sil:
                lo, hi, zv = zip(*sil)
                ax.plot(lo, zv, "-", color="red", lw=2)
                ax.plot(hi, zv, "-", color="red", lw=2)
                ax.plot([lo[0], hi[0]], [zv[0]] * 2, "-", color="red", lw=2)
                ax.plot([lo[-1], hi[-1]], [zv[-1]] * 2, "-", color="red", lw=2)
            else:
                proj = (f["poly"] - p0) @ u                   # footprint projected onto the line axis
                ax.add_patch(plt.Rectangle((float(proj.min()), f["zlo"]), float(proj.max() - proj.min()),
                                           f["zhi"] - f["zlo"], fill=False, edgecolor="red", lw=2))
    if route_w and len(route_w) > 1:
        rs = [float((np.asarray(r[:2]) - p0) @ u) for r in route_w]
        ax.plot(rs, [r[2] for r in route_w], "-", color="gold", lw=2.5, label="route")
    ax.plot(0, g_xyz[2], "o", color="cyan", ms=11, mec="k", label="grasp")
    ax.plot(L, p_xyz[2], "s", color="magenta", ms=10, mec="k", label="pad")
    # scene anchors, projected onto the same grasp->pad axis
    if tgt_p is not None:
        ax.plot(float((np.asarray(tgt_p[:2]) - p0) @ u), float(tgt_p[2]), "*", color="blue", ms=17,
                mec="k", mew=0.6, label="target bottle (spawn)")
    if ee_xyz is not None:
        ax.plot(float((np.asarray(ee_xyz[:2]) - p0) @ u), float(ee_xyz[2]), "^", color="darkorange",
                ms=12, mec="k", mew=0.6, label=f"gripper now ({args.arm})")
    from matplotlib.patches import Patch
    proxies = [Patch(color=LABEL_COLORS[c], label=LABEL_NAMES[c]) for c in (FREE, OBSTACLE, BEYOND)]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=proxies + h, loc="upper right", fontsize=8)
    eps_txt = "inf" if (merged and np.isinf(eps_star)) else (f"{eps_star:.3f}m" if merged else "INACCESSIBLE")
    ax.set_xlabel("arc distance grasp->pad (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"Side elevation (profile through grasp->pad)  |  seed {args.seed}, arm {args.arm}"
                 f"   eps*={eps_txt}")
    stem = f"metric_side_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def _viz_clearance_profile(out_dir, args, route_w, route_clear, eps_star, merged):
    """CLEARANCE-ALONG-ROUTE: line chart of occluder clearance vs arc length from grasp to pad; the
    minimum is eps*, the gripper half-width is a dashed feasibility line, and route height rides a
    secondary axis. A stats-friendly view that pinpoints the tightest squeeze."""
    if not route_w or len(route_w) < 2:
        return
    P = np.asarray(route_w, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    clr = np.asarray(route_clear, float)
    finite = np.isfinite(clr)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, np.where(finite, clr, np.nan), "-", color="#3949ab", lw=2, label="occluder clearance")
    ax.axhline(args.gripper_r, color="#d84315", ls="--", lw=1.5, label=f"gripper r={args.gripper_r:.3f}m")
    if merged and np.isfinite(eps_star) and finite.any():
        j = int(np.nanargmin(np.where(finite, clr, np.inf)))
        ax.plot(s[j], clr[j], "kX", ms=13, label=f"eps*={eps_star:.3f}m")
    elif merged and np.isinf(eps_star):
        ax.text(0.5, 0.9, "eps* = inf (route clears over the bottle top)", transform=ax.transAxes,
                ha="center", color="#2e7d32", fontsize=11)
    ax.set_xlabel("arc length along route, grasp->pad (m)")
    ax.set_ylabel("clearance to occluder (m)", color="#3949ab")
    ax2 = ax.twinx(); ax2.plot(s, P[:, 2], "-", color="gray", lw=1.2, alpha=0.7)
    ax2.set_ylabel("route height z (m)", color="gray")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Clearance & height along the route  |  seed {args.seed}, arm {args.arm}")
    stem = f"metric_profile_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def _viz_topdown(out_dir, args, XX, YY, label, foots, g_xyz, p_xyz, route_w, tgt_p=None, ee_xyz=None):
    """TOP-DOWN plan: reachable-at-some-height footprint (green) + occluder outline + the route's (x,y)
    coloured by height z. Shows the sideways detour and how high it climbs, in one plan view."""
    any_free = (label == FREE).any(axis=0)
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(np.where(any_free, 1.0, np.nan), origin="lower", extent=extent, cmap="Greens",
              vmin=0, vmax=1.5, aspect="equal")
    if foots:
        from matplotlib.patches import Polygon as MplPolygon
        for f in foots:
            if f["poly"] is not None:
                ax.add_patch(MplPolygon(f["poly"], closed=True, fill=False, edgecolor="red", lw=2))
    if route_w and len(route_w) > 1:
        P = np.asarray(route_w, float)
        sc = ax.scatter(P[:, 0], P[:, 1], c=P[:, 2], cmap="plasma", s=16)
        fig.colorbar(sc, ax=ax, label="route height z (m)")
    ax.plot(g_xyz[0], g_xyz[1], "o", color="cyan", ms=12, mec="k", label="grasp")
    ax.plot(p_xyz[0], p_xyz[1], "s", color="magenta", ms=10, mec="k", label="pad")
    _scene_anchor_markers(ax, tgt_p, ee_xyz, args.arm)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Top-down: route coloured by height  |  seed {args.seed}, arm {args.arm}\n"
                 f"green = reachable at some height")
    stem = f"metric_topdown_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def _viz_ceiling(out_dir, args, XX, YY, zs, label, foots, tgt_p=None, ee_xyz=None):
    """REACHABILITY CEILING: highest FREE z per (x,y) -- the 'lid' of the reachable envelope. Cells
    brighter than the bottle-top height can be cleared over; the red footprint shows where the bottle
    sits. Answers 'where can the arm actually get above the occluder?' at a glance."""
    free = label == FREE
    ceil = np.full(free.shape[1:], np.nan)
    for iz in range(len(zs)):                                 # ascending -> keeps the highest FREE z
        ceil[free[iz]] = zs[iz]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(ceil, origin="lower", extent=[XX.min(), XX.max(), YY.min(), YY.max()],
                   cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, label="highest reachable z (m)")
    zhi = None
    if foots:
        from matplotlib.patches import Polygon as MplPolygon
        for f in foots:
            if f["poly"] is not None:
                ax.add_patch(MplPolygon(f["poly"], closed=True, fill=False, edgecolor="red", lw=2))
        zhis = [f["zhi"] for f in foots if f["poly"] is not None]
        zhi = max(zhis) if zhis else None
    _scene_anchor_markers(ax, tgt_p, ee_xyz, args.arm)
    if tgt_p is not None or ee_xyz is not None:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Reachability ceiling (max FREE z per x,y)  |  seed {args.seed}, arm {args.arm}"
                 + (f"\nbottle top z={zhi:.2f}m -- brighter cells can clear over it" if zhi else ""))
    stem = f"metric_ceiling_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def phase4_visuals(out_dir, args, XX, YY, zs, label, edt, foots, g_xyz, p_xyz, route, route_w,
                   eps_star, merged, tgt_p=None, ee_xyz=None):
    """The four 3D-legible views: side elevation, clearance-along-route, top-down-by-height, ceiling.
    All the spatial ones also carry the scene anchors (target bottle spawn + current gripper pose);
    the clearance-vs-arc-length chart has no xy plane, so it gets neither."""
    route_clear = [float(edt[iz, iy, ix]) for (iz, iy, ix) in route] if route else []
    _viz_side_elevation(out_dir, args, XX, YY, zs, label, foots, g_xyz, p_xyz, route_w, eps_star,
                        merged, tgt_p=tgt_p, ee_xyz=ee_xyz)
    _viz_clearance_profile(out_dir, args, route_w, route_clear, eps_star, merged)
    _viz_topdown(out_dir, args, XX, YY, label, foots, g_xyz, p_xyz, route_w, tgt_p=tgt_p, ee_xyz=ee_xyz)
    _viz_ceiling(out_dir, args, XX, YY, zs, label, foots, tgt_p=tgt_p, ee_xyz=ee_xyz)


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
    ap.add_argument("--zmin", type=float, default=0.78, help="stack floor (m); ~grasp height (table ~0.74)")
    ap.add_argument("--zmax", type=float, default=1.4, help="stack ceiling (m); above the occluder top")
    ap.add_argument("--zres", type=float, default=0.03, help="vertical slice spacing (m); K = (zmax-zmin)/zres")
    ap.add_argument("--topdown", action="store_true", help="use a top-down quat instead of the side grasp")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.35)
    ap.add_argument("--ymax", type=float, default=0.35)
    ap.add_argument("--res", type=float, default=0.01, help="grid resolution (m)")
    ap.add_argument("--occ-shape", choices=["mesh", "extruded"], default="mesh",
                    help="geometry the occluder-clearance field (and the figures) use. 'mesh' re-cuts "
                         "the TRUE posed collision mesh at every z, so the bottle tapers and the neck "
                         "is thin -- faithful to what curobo collides against. 'extruded' is the older "
                         "solid: the widest footprint held constant to the cap, which invents up to "
                         "~4cm of phantom obstacle around the neck. CHANGING THIS CHANGES eps*; pass "
                         "'extruded' to reproduce pre-2026-07-24 numbers.")
    ap.add_argument("--obstacles", choices=["all", "occluders"], default="all",
                    help="which actors the clearance field measures against. 'all' = every mesh in "
                         "env.collision_list (the registry curobo's update_world uses) except the "
                         "target and the pad, so procedural table CLUTTER counts as an obstacle "
                         "alongside the occluder ring -- the metric's world then matches the "
                         "planner's. 'occluders' = the curated olive-oil ring only, the "
                         "pre-2026-07-27 behaviour. CHANGING THIS CHANGES eps* AND WHAT IT MEANS: "
                         "under 'all' eps* is clearance to the nearest scene obstacle, not to the "
                         "occluder, so values are not comparable across the two.")
    ap.add_argument("--gate-tau", type=float, default=0.35,
                    help="joint-space gate (rad): union adjacent FREE voxels only if the max per-joint "
                         "warm-config jump is <= tau. Set from the phase-0/2 histogram gap (~0.2-0.35). "
                         "Only active with --warm-start; without it the DSU runs ungated.")
    ap.add_argument("--seed-snap", type=float, default=0.10,
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
    ap.add_argument("--warm-seeds", type=int, default=8,
                    help="return_seeds K for the multi-branch solve (candidate branches offered per cell)")
    ap.add_argument("--ik-seeds", type=int, default=30,
                    help="curobo num_seeds per pose (Tier-1 speedup; default 100 is overkill). Lower = "
                         "faster but may miss a few hard-but-reachable poses; raise (~60) for a final run.")
    ap.add_argument("--free-only", action="store_true",
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
    ap.add_argument("--chunk", type=int, default=256,
                    help="IK poses per batch; lower if you hit CUDA OOM (planners already use ~9GB)")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="results location; each run lands in its own <out-dir>/<timestamp>/ subfolder")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
