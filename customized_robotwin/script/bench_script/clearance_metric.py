#!/usr/bin/env python3
"""
Scene-difficulty (clearance) metric via curobo batched IK.

Measures "how boxed-in is the target": on a single fixed-z slice we label every
(x, y) gripper cell FREE / OBSTACLE / BEYOND-REACH using two IK sweeps that differ
only in world-collision checking, run a Euclidean distance transform over OBSTACLE
cells to get clearance-in-metres, then read the widest-path bottleneck (eps*)
between the target's grasp cell and the pad. See the design docs for the rationale.

This file reuses the scene / grid / IK machinery from reachability_map.py verbatim
(that stays the visualization tool); here we only add the metric on top.

BUILD STATUS: step 1 of the pipeline -- scene + grid + grasp orientation + IK
solver + self-checks. Later steps (two-pass labelling, EDT, widest-path DSU,
reporting, viz) are stubbed below.

USAGE (from the benchmark folder, env sourced + ROBOTWIN_BENCH_TASK=bench):
    python clearance_metric.py --seed 1 --offset 0.2 --arm right --z 0.90 --res 0.01
"""

import argparse
import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

# reachability_map runs setup_paths() at import and pulls in torch + the env stack,
# so importing it makes all the shared helpers (and curobo's torch) available.
# It also selects matplotlib's Agg backend, so importing pyplot here is headless-safe.
import reachability_map as rm
from reachability_map import (
    make_occluder_task,
    build_cfg,
    DR_CLEAN,
    PAD_XY,
    OCC_HALF_FOOTPRINT,
    _build_ik_solver,
    _solve_grid,
)
from analyze_occluder_visibility import OCCLUDER_COLLISION
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# repo-root results dir for this metric, anchored to THIS file (same layout as reachability_map)
RESULTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "validation" / "results" / "clearance_metric"


# ----------------------------------------------------------------------------- grid
def build_grid(args):
    """Regular (x, y) lattice of gripper positions on the z=args.z slice.
    Returns xs, ys (1D axes) and XX, YY (ny, nx meshgrids), matching reachability_map."""
    xs = np.arange(args.xmin, args.xmax + 1e-9, args.res)
    ys = np.arange(args.ymin, args.ymax + 1e-9, args.res)
    XX, YY = np.meshgrid(xs, ys)                      # (ny, nx)
    return xs, ys, XX, YY


def _build_ik_solver_no_world(planner):
    """Companion to reachability_map._build_ik_solver, but with NO collision world at all,
    so IK success == pure kinematic reachability + self-collision. This is the 'collisions-OFF'
    sweep: a cell that fails here is BEYOND-REACH (kinematic or self-collision), not
    clutter-blocked. The table therefore falls into OBSTACLE, as intended.

    We pass world_model=None (and no world_coll_checker): curobo only builds the primitive
    collision cost when a checker exists (arm_base.py: `... and world_coll_checker is not None`),
    so with both None there is no world-collision cost -- an EMPTY world would instead keep the
    cost active and error with 'Primitive Collision has no obstacles'."""
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    mg = planner.motion_gen
    cfg = IKSolverConfig.load_from_robot_config(
        planner.yml_path,
        None,                                   # world_model=None -> no checker, no world-collision cost
        tensor_args=mg.tensor_args,
        use_cuda_graph=False,
        self_collision_check=True,
        self_collision_opt=False,
    )
    return IKSolver(cfg)


# label codes for the three-way raster
BEYOND, OBSTACLE, FREE = 0, 1, 2
LABEL_NAMES = {BEYOND: "BEYOND-REACH", OBSTACLE: "OBSTACLE", FREE: "FREE"}


def label_grid(env, planner, ik_on, arm_tag, XX, YY, z, grasp_q, chunk):
    """Two IK sweeps on the same grid, differing only in world-collision checking:
      reach_on  = reachable AND collision-free (occluder + table world)      -> FREE
      reach_off = reachable ignoring world collision (empty world)           -> reach envelope
    Labels: FREE = reach_on; OBSTACLE = reach_off & ~reach_on; BEYOND = ~reach_off.
    Returns an int8 label array shaped like XX (0/1/2 = BEYOND/OBSTACLE/FREE)."""
    gp = np.zeros((XX.size, 7))
    gp[:, 0] = XX.ravel(); gp[:, 1] = YY.ravel(); gp[:, 2] = z
    gp[:, 3:] = grasp_q

    # pass ON: reuse the solver already built for the self-checks (shares the loaded world)
    reach_on = _solve_grid(env.robot, planner, ik_on, arm_tag, gp, chunk=chunk).reshape(XX.shape)

    # pass OFF: fresh solver with an empty world -> kinematics + self-collision only
    ik_off = _build_ik_solver_no_world(planner)
    reach_off = _solve_grid(env.robot, planner, ik_off, arm_tag, gp, chunk=chunk).reshape(XX.shape)
    del ik_off
    rm.torch.cuda.empty_cache()

    label = np.where(reach_on, FREE, np.where(reach_off & ~reach_on, OBSTACLE, BEYOND)).astype(np.int8)

    # sanity: reach_on should be a subset of reach_off (collision-free WITH world implies
    # reachable WITHOUT it). A nonzero count here is IK seed noise, not a logic error.
    anomaly = int((reach_on & ~reach_off).sum())
    if anomaly:
        print(f"[label] WARN {anomaly} cells reachable ON but not OFF (IK seed noise); "
              f"kept as FREE")
    return label


def occluder_footprint_polys(env):
    """True 2D footprints of the occluders, for BOTH the distance calc and the drawing.

    Each occluder's COLLISION mesh -- the exact geometry curobo collides against (base<id>.glb,
    used with convex=True) -- is transformed by the occluder's ACTUAL world pose (position +
    orientation, including the random yaw), projected to xy, and convex-hulled. So the footprint
    is orientation-faithful: a non-round bottle comes out as a correctly-rotated polygon, and it
    matches the same mesh that produced the FREE/OBSTACLE labels (no shape mismatch between the
    clearance value and the reachability).

    Returns a list of (K,2) world-xy hull polygons; None if the mesh can't be loaded (the caller
    then falls back to the OCC_HALF_FOOTPRINT circle)."""
    occs = getattr(env, "occluders", None)
    if not occs:
        return []
    try:
        import trimesh
        import transforms3d as t3d
        from scipy.spatial import ConvexHull
        path = os.path.join(os.environ["BENCH_ROOT"], OCCLUDER_COLLISION)
        V = np.asarray(trimesh.load(path, force="mesh").vertices)   # same mesh for every bottle
    except Exception as e:
        print(f"[footprint] could not load occluder collision mesh ({e}); falling back to circle")
        return None
    polys = []
    for occ in occs:
        pose = occ.get_pose()
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=float))   # SAPIEN quat is wxyz
        xy = (V @ R.T + np.asarray(pose.p, dtype=float))[:, :2]         # posed world footprint (xy)
        try:
            polys.append(xy[ConvexHull(xy).vertices])
        except Exception:
            polys.append(None)
    return polys


def occluder_clearance(polys, occ_ps, XX, YY, res, r_foot):
    """Clearance (m) from each grid cell to the nearest OCCLUDER footprint -- not the table /
    furniture / target that also sit in curobo's world.

    Primary path: rasterise the TRUE posed mesh footprints (polys) onto the grid and take the
    Euclidean distance transform -> distance to the real oriented shape, 0 inside a footprint.
    Fallback (polys is None): analytic distance to circles of radius r_foot around occ_ps.

    Measuring to the occluders alone means a table-blocked cell reads FAR here (the table never
    contributes clearance) and we never measure to the reach boundary either. Full-arm/EE-point
    caveat still applies (addendum 2 §5): this is the EE control-point distance to the footprint;
    routability stays body-aware via the IK-FREE node set. No occluders -> +inf."""
    if polys is not None and any(p is not None for p in polys):
        from matplotlib.path import Path as MplPath
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        mask = np.zeros(XX.size, dtype=bool)
        for p in polys:
            if p is not None:
                mask |= MplPath(p).contains_points(pts)
        mask = mask.reshape(XX.shape)
        if not mask.any():
            return np.full(XX.shape, np.inf, dtype=float)
        return distance_transform_edt(~mask, sampling=res)
    if len(occ_ps) == 0:
        return np.full(XX.shape, np.inf, dtype=float)
    d = np.full(XX.shape, np.inf, dtype=float)
    for op in occ_ps:
        d = np.minimum(d, np.hypot(XX - op[0], YY - op[1]) - r_foot)
    return np.clip(d, 0.0, None)


def nearest_free_cell(free, XX, YY, xy, max_dist):
    """Grid cell (iy, ix) of the nearest FREE cell to world point `xy`, or None if the
    closest FREE cell is farther than max_dist (m). Used to plant the DSU seeds: the grasp
    cell (target end) and the pad cell (other end)."""
    if not free.any():
        return None
    d2 = (XX - xy[0]) ** 2 + (YY - xy[1]) ** 2
    d2 = np.where(free, d2, np.inf)
    iy, ix = np.unravel_index(int(np.argmin(d2)), free.shape)
    dist = float(np.sqrt(d2[iy, ix]))
    if dist > max_dist:
        return None
    return (iy, ix), dist


_NEIGH8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def widest_path_eps(label, edt, seed_a, seed_b):
    """Widest-path (max-min) bottleneck between seed_a and seed_b over FREE cells, 8-connected.

    Kruskal: add FREE cells in DESCENDING clearance, unioning each with already-present FREE
    neighbours; the clearance of the cell whose addition first connects the two seeds is eps* --
    the tightest squeeze on the least-bad route (the max-spanning-tree bottleneck). Clearance
    VALUES come from the Euclidean EDT; 8-connectivity only decides whether a diagonal corner
    can be turned, so it never contaminates the number.

    (2.5D hook: the union step is exactly where the joint-space edge gate goes -- only union
    neighbours whose IK configs are close. In 2D, under the no-mazy-middle assumption, it's off.)

    Returns (eps_star_m, bottleneck_iyix, merged_bool)."""
    ny, nx = label.shape
    free = label == FREE
    free_flat = free.ravel()
    edt_flat = edt.ravel()
    parent = np.arange(ny * nx)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sa = seed_a[0] * nx + seed_a[1]
    sb = seed_b[0] * nx + seed_b[1]
    if not (free_flat[sa] and free_flat[sb]):
        return 0.0, None, False          # a seed isn't FREE -> nothing to connect

    idxs = np.flatnonzero(free_flat)
    order = idxs[np.argsort(-edt_flat[idxs], kind="stable")]     # descending clearance
    added = np.zeros(ny * nx, dtype=bool)

    for c in order:
        cy, cx = divmod(int(c), nx)
        added[c] = True
        for dy, dx in _NEIGH8:
            y2, x2 = cy + dy, cx + dx
            if 0 <= y2 < ny and 0 <= x2 < nx:
                nb = y2 * nx + x2
                if added[nb] and free_flat[nb]:
                    union(int(c), int(nb))
        if added[sa] and added[sb] and find(sa) == find(sb):
            return float(edt_flat[c]), (cy, cx), True

    return 0.0, None, False              # seeds never connect -> inaccessible


def reconstruct_widest_path(free, edt, seed_a, seed_b, eps_star):
    """Recover an actual widest-path route to draw. Every cell on the max-min path has clearance
    >= eps* (the bottleneck), and the DSU guarantees seed_a and seed_b are connected within the
    sub-grid {FREE and clearance >= eps*}. So a BFS through exactly those cells returns a valid
    bottleneck-optimal route (the shortest such, since BFS). Returns a list of (iy, ix) or None."""
    if eps_star is None:
        return None
    ny, nx = free.shape
    allowed = free & (edt >= eps_star - 1e-9)     # inf-clearance case: inf >= inf holds
    sa, sb = tuple(seed_a), tuple(seed_b)
    if not (allowed[sa] and allowed[sb]):
        return None
    prev = {sa: None}
    dq = deque([sa])
    while dq:
        cur = dq.popleft()
        if cur == sb:
            break
        cy, cx = cur
        for dy, dx in _NEIGH8:
            y2, x2 = cy + dy, cx + dx
            if 0 <= y2 < ny and 0 <= x2 < nx and allowed[y2, x2] and (y2, x2) not in prev:
                prev[(y2, x2)] = cur
                dq.append((y2, x2))
    if sb not in prev:
        return None
    path, node = [], sb
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


def grasp_orientation(env, arm_tag, topdown):
    """The fixed EE orientation the whole slice is evaluated at: the target's own
    horizontal side-grasp for this arm (or top-down). Returns (grasp_q, grasp_pose)
    where grasp_pose is the full [x,y,z,qw,qx,qy,qz] world grasp (or None)."""
    if topdown:
        return np.array([0, 1, 0, 0], dtype=float), None
    cp_id = env._pick_side_grasp_id(env.target_obj, arm_tag)
    grasp_pose = env._geometric_grasp_pose(env.target_obj, cp_id, pre_dis=0.0) if cp_id is not None else None
    grasp_q = np.array(grasp_pose[-4:]) if grasp_pose is not None else np.array([0, 1, 0, 0], dtype=float)
    return grasp_q, grasp_pose


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
    rm.torch.cuda.empty_cache()
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


# ----------------------------------------------------------------------------- run
def run(args):
    # One folder per run, so re-running a seed never clobbers earlier output.
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] writing to {out_dir}")

    # --- scene (identical setup to reachability_map.run) ---
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

    # --- grid on the z-slice ---
    xs, ys, XX, YY = build_grid(args)
    print(f"[grid] z={args.z:.3f}  res={args.res}  cells={XX.size} ({len(xs)}x{len(ys)})  "
          f"x[{args.xmin},{args.xmax}] y[{args.ymin},{args.ymax}]")

    # --- resolve the grasping arm (explicit or auto) + its IK solver ---
    arm, planner, grasp_q, grasp_pose, ik = select_arm(env, args)
    args.arm = arm      # downstream naming / filenames / report use the resolved arm
    if grasp_pose is not None:
        print(f"[{arm}] grasp pose z={grasp_pose[2]:.3f}  (slice z={args.z:.3f}); "
              f"grasp_q={np.array2string(grasp_q, precision=3)}")
    else:
        print(f"[{arm}] using top-down grasp_q={np.array2string(grasp_q, precision=3)}")

    # --- self-checks: real grasp pose should be reachable; box centre should not ---
    if grasp_pose is not None:
        chk = _solve_grid(env.robot, planner, ik, args.arm, np.array([grasp_pose]), chunk=args.chunk)
        print(f"[{args.arm}] self-check grasp pose reachable = {bool(chk[0])}  (expect True)")
    if box_p is not None:
        inbox = np.array([[box_p[0], box_p[1], args.z, *grasp_q]])
        chk2 = _solve_grid(env.robot, planner, ik, args.arm, inbox, chunk=args.chunk)
        print(f"[{args.arm}] self-check box-centre reachable = {bool(chk2[0])}  (expect False)")

    # --- step 2: two-pass ON/OFF labelling -> FREE / OBSTACLE / BEYOND-REACH ---
    label = label_grid(env, planner, ik, args.arm, XX, YY, args.z, grasp_q, args.chunk)
    n = label.size
    counts = {name: int((label == code).sum()) for code, name in LABEL_NAMES.items()}
    print(f"[label] FREE={counts['FREE']}  OBSTACLE={counts['OBSTACLE']}  "
          f"BEYOND-REACH={counts['BEYOND-REACH']}  (of {n} cells)")

    # --- step 3: clearance to the OCCLUDERS specifically (true posed mesh footprints, m) ---
    polys = occluder_footprint_polys(env)
    if polys:      # self-check: each footprint centroid should sit on its occluder (mesh<->pose OK)
        offs = [float(np.hypot(*(p.mean(0) - op[:2]))) for op, p in zip(occ_ps, polys) if p is not None]
        if offs:
            dims = [tuple(np.round(p.max(0) - p.min(0), 3)) for p in polys if p is not None]
            print(f"[footprint] {len(offs)} posed mesh footprints; max centroid offset from "
                  f"occluder = {max(offs):.3f}m (expect ~0); footprint bbox(x,y) e.g. {dims[0]}")
    edt = occluder_clearance(polys, occ_ps, XX, YY, args.res, OCC_HALF_FOOTPRINT)
    free = label == FREE
    if np.isinf(edt).all():
        print("[edt] no occluders -> occluder-clearance unbounded (nothing to squeeze past)")
    elif free.any():
        fe = edt[free]
        print(f"[edt] occluder-clearance over FREE cells (m): min={fe.min():.3f}  "
              f"median={np.median(fe):.3f}  max={fe.max():.3f}")

    # --- step 4: widest-path bottleneck eps* between the grasp cell and the pad ---
    tgt_xy = np.array(grasp_pose[:2]) if grasp_pose is not None else tgt_p[:2]
    seed_t = nearest_free_cell(free, XX, YY, tgt_xy, max_dist=args.seed_snap)
    seed_p = nearest_free_cell(free, XX, YY, pad_xy, max_dist=args.seed_snap)

    seed_t_xy = seed_p_xy = bott_xy = boxed_dist = path_xy = None
    if seed_t is None or seed_p is None:
        which = "grasp" if seed_t is None else "pad"
        print(f"[dsu] no FREE cell within {args.seed_snap:.3f}m of the {which} seed "
              f"-> target inaccessible (eps*=0)")
        eps_star, merged = 0.0, False
    else:
        (ty, tx), td = seed_t
        (py, px), pd = seed_p
        seed_t_xy = (float(XX[ty, tx]), float(YY[ty, tx]))
        seed_p_xy = (float(XX[py, px]), float(YY[py, px]))
        print(f"[dsu] grasp seed=({seed_t_xy[0]:.3f},{seed_t_xy[1]:.3f}) snap={td:.3f}m   "
              f"pad seed=({seed_p_xy[0]:.3f},{seed_p_xy[1]:.3f}) snap={pd:.3f}m")
        eps_star, bott, merged = widest_path_eps(label, edt, (ty, tx), (py, px))
        if bott is not None:
            by, bx = bott
            bott_xy = (float(XX[by, bx]), float(YY[by, bx]))
            boxed_dist = float(np.hypot(bott_xy[0] - seed_t_xy[0], bott_xy[1] - seed_t_xy[1]))
        if merged:
            route = reconstruct_widest_path(free, edt, (ty, tx), (py, px), eps_star)
            if route is not None:
                path_xy = [(float(XX[iy, ix]), float(YY[iy, ix])) for iy, ix in route]

    if not merged:
        print("[dsu] eps* = 0.000 m  (INACCESSIBLE: grasp and pad not connected through FREE space)")
    elif np.isinf(eps_star):
        print("[dsu] eps* = inf  (no obstacles between grasp and pad)")
    else:
        where = ("near target (boxed-in)" if boxed_dist <= args.boxed_in_radius
                 else "interior (en-route constriction)")
        print(f"[dsu] eps* = {eps_star:.3f} m  bottleneck=({bott_xy[0]:.3f},{bott_xy[1]:.3f})  "
              f"{boxed_dist:.3f}m from target -> {where}")

    # --- step 5: report (eps* vs gripper r, derived) + labels/clearance figure ---
    report(args, out_dir, XX, YY, label, edt, box_p, occ_ps, polys, tgt_p, pad_xy,
           seed_t_xy, seed_p_xy, eps_star, bott_xy, merged, boxed_dist, counts, path_xy)
    print(f"[run] done. outputs in {out_dir}")

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
    ap.add_argument("--z", type=float, default=0.90, help="EE height for the single-slice grid (m)")
    ap.add_argument("--topdown", action="store_true", help="use a top-down quat instead of the side grasp")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.35)
    ap.add_argument("--ymax", type=float, default=0.35)
    ap.add_argument("--res", type=float, default=0.01, help="grid resolution (m)")
    ap.add_argument("--seed-snap", type=float, default=0.10,
                    help="max distance to snap a DSU seed (grasp/pad) to the nearest FREE cell (m)")
    ap.add_argument("--boxed-in-radius", type=float, default=0.05,
                    help="bottleneck within this distance of the target counts as 'boxed-in' (m)")
    ap.add_argument("--gripper-r", type=float, default=0.03,
                    help="gripper half-width (m), compared to eps* at READ time only; never baked "
                         "into the metric (eps* stays embodiment-free)")
    ap.add_argument("--chunk", type=int, default=256,
                    help="IK poses per batch; lower if you hit CUDA OOM (planners already use ~9GB)")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="results location; each run lands in its own <out-dir>/<timestamp>/ subfolder")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
