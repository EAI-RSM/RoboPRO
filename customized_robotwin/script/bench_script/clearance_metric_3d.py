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

USAGE (from the benchmark folder, env sourced + ROBOTWIN_BENCH_TASK=bench):
    python clearance_metric_3d.py --seed 1 --offset 0.2 --arm right --zmin 0.78 --zmax 1.4 --zres 0.03
    #   add --warm-start to also propagate a continuity branch field per slice (slower, one extra IK pass/slice)
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

# repo-root results dir for this metric, anchored to THIS file (same layout as reachability_map);
# separate folder from the frozen 2D tool so the 3D sandbox never mixes into clearance_metric/
RESULTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "validation" / "results" / "clearance_metric_3d"


class Timings:
    """Reusable per-component timer (project convention: time every script by phase and save the
    breakdown with the run). Use `with tm.section("name"):` around each logical phase; call
    `tm.save(out_dir)` at the end to print a summary table and write timings.json into the run folder."""

    def __init__(self):
        self.records = []
        self._wall0 = time.perf_counter()

    @contextmanager
    def section(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.records.append((name, dt))
            print(f"[time] {name}: {dt:.2f}s")

    def save(self, out_dir, off_seconds=0.0):
        total = time.perf_counter() - self._wall0
        projected = total - off_seconds                       # cost if the OFF pass were skipped (--free-only)
        data = {"components": [{"name": n, "seconds": round(s, 3)} for n, s in self.records],
                "total_seconds": round(total, 3),
                "off_pass_seconds": round(off_seconds, 3),
                "projected_free_only_seconds": round(projected, 3)}
        (Path(out_dir) / "timings.json").write_text(json.dumps(data, indent=2))
        width = max((len(n) for n, _ in self.records), default=10)
        print("[time] ---------------- component timing ----------------")
        for n, s in self.records:
            print(f"[time]   {n:<{width}}  {s:8.2f}s  ({100 * s / total if total else 0:4.1f}%)")
        print(f"[time]   {'TOTAL':<{width}}  {total:8.2f}s")
        if off_seconds > 0:
            print(f"[time] projected WITH --free-only: {projected:.2f}s  "
                  f"(OFF pass {off_seconds:.2f}s = {100 * off_seconds / total if total else 0:.1f}% would be skipped)")
        else:
            print("[time] projected WITH --free-only: equals TOTAL above (--free-only already active)")
        print("[time] wrote timings.json")
        return total


# ----------------------------------------------------------------------------- grid
def build_grid(args):
    """Regular (x, y) lattice of gripper positions (constant across z) plus the z-slice axis for the
    stack. Returns xs, ys, zs (1D axes) and XX, YY (ny, nx meshgrids); the stack is swept over zs at
    run time (x,y do not depend on z, so we keep XX/YY 2D and vary z as a scalar per slice)."""
    xs = np.arange(args.xmin, args.xmax + 1e-9, args.res)
    ys = np.arange(args.ymin, args.ymax + 1e-9, args.res)
    zs = np.arange(args.zmin, args.zmax + 1e-9, args.zres)
    XX, YY = np.meshgrid(xs, ys)                      # (ny, nx)
    return xs, ys, zs, XX, YY


def _build_ik_solver_no_world(planner, num_seeds=100):
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
        num_seeds=num_seeds,                    # fewer seeds = faster IK (Tier-1 speedup knob)
        use_cuda_graph=False,
        self_collision_check=True,
        self_collision_opt=False,
    )
    return IKSolver(cfg)


def _solve_grid_q(robot, planner, ik, arm_tag, gp_world, chunk=256, num_seeds=None):
    """Like reachability_map._solve_grid, but ALSO returns the IK joint solution per pose -- the
    fuel for the joint-space gate. curobo already computes it inside solve_batch, so returning it
    costs no extra IK; the 2D tool simply throws it away by reading only result.success.
    num_seeds (None = solver default 100) caps the per-pose seed optimisation (Tier-1 speedup).

    Returns (success (N,) bool, q (N, dof) float32; rows where success is False are NaN)."""
    from curobo.types.math import Pose as CuroboPose
    ta = planner.motion_gen.tensor_args
    N = len(gp_world)
    pos = np.empty((N, 3), dtype=np.float32)
    quat = np.empty((N, 4), dtype=np.float32)
    for i, gp in enumerate(gp_world):
        p, q = rm._world_gripper_to_curobo(robot, planner, arm_tag, gp)
        pos[i], quat[i] = p, q
    succ = np.zeros(N, dtype=bool)
    qout = None
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        goal = CuroboPose(position=ta.to_device(pos[s:e]), quaternion=ta.to_device(quat[s:e]))
        result = ik.solve_batch(goal, num_seeds=num_seeds)
        sc = result.success.detach().cpu().numpy().reshape(e - s, -1)[:, 0].astype(bool)
        # VALIDATE (first GPU run): result.solution is expected (batch, n_seeds, dof); we take seed 0
        # to match the [:, 0] used on success. The one-time shape print below confirms the axis.
        sol = result.solution.detach().cpu().numpy()
        if s == 0:
            print(f"[solve_q] result.solution raw shape = {sol.shape}  (expect (chunk, n_seeds, dof))")
        sol = sol.reshape(e - s, -1, sol.shape[-1])[:, 0, :]        # (batch, dof)
        if qout is None:
            qout = np.full((N, sol.shape[-1]), np.nan, dtype=np.float32)
        succ[s:e] = sc
        qout[s:e] = sol
        del result, goal
        rm.torch.cuda.empty_cache()
    return succ, qout


def _solve_grid_q_multi(robot, planner, ik, arm_tag, gp_world, chunk=256, return_seeds=8, num_seeds=None):
    """Multi-branch IK: like _solve_grid_q but returns the top-`return_seeds` CONVERGED solutions
    per pose. curobo already optimises ~100 seeds internally per pose; asking for K of them back
    (instead of only the best) is nearly free and hands us the distinct IK branches at each cell --
    the menu the warm-start propagation chooses among to enforce branch continuity.

    Returns (cand_q (N, K, dof) float32, NaN where that candidate did not converge; cand_ok (N, K) bool)."""
    from curobo.types.math import Pose as CuroboPose
    ta = planner.motion_gen.tensor_args
    N = len(gp_world)
    pos = np.empty((N, 3), dtype=np.float32)
    quat = np.empty((N, 4), dtype=np.float32)
    for i, gp in enumerate(gp_world):
        p, q = rm._world_gripper_to_curobo(robot, planner, arm_tag, gp)
        pos[i], quat[i] = p, q
    cand_q = None
    cand_ok = np.zeros((N, return_seeds), dtype=bool)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        goal = CuroboPose(position=ta.to_device(pos[s:e]), quaternion=ta.to_device(quat[s:e]))
        result = ik.solve_batch(goal, return_seeds=return_seeds, num_seeds=num_seeds)   # top-K branches per pose
        sol = result.solution.detach().cpu().numpy()                 # (b, K, dof)
        ok = result.success.detach().cpu().numpy().reshape(e - s, -1).astype(bool)  # (b, K)
        if cand_q is None:
            cand_q = np.full((N, return_seeds, sol.shape[-1]), np.nan, dtype=np.float32)
        cand_q[s:e] = sol[:, :return_seeds]
        cand_ok[s:e] = ok[:, :return_seeds]
        del result, goal
        rm.torch.cuda.empty_cache()
    cand_q[~cand_ok] = np.nan
    return cand_q, cand_ok


def _wrap_linf(a, b):
    """Joint distance the gate uses: max over joints of the wrapped |a-b| (radians, wrap to (-pi,pi])."""
    return float(np.abs((a - b + np.pi) % (2.0 * np.pi) - np.pi).max())


def _pick_nearest(candQ_cell, okmask, ref):
    """Among a cell/voxel's CONVERGED candidate branches, the one closest (wrapped-Linf) to ref.
    Vectorised over the candidate axis so the BFS propagations stay fast."""
    valid = candQ_cell[okmask]                                       # (nok, dof)
    d = np.abs((valid - ref + np.pi) % (2.0 * np.pi) - np.pi).max(axis=1)
    return valid[int(np.argmin(d))]


def warm_start_branches(free, cand_q, cand_ok):
    """Continuity propagation over the FREE region: BFS out from a central stable cell; each cell is
    assigned the CANDIDATE branch closest (wrapped-Linf) to the branch already assigned to the
    neighbour it is first reached from. This is the warm-start done offline on the candidate menu.

    Effect: a SPURIOUS branch-hop collapses -- if a nearby branch exists in the cell's candidate set
    it gets chosen, so the jump to the neighbour vanishes. A REAL seam survives -- if the cell simply
    has no candidate near the neighbour, the smallest available jump is still large. So comparing the
    warm field against the raw (best-cost) field separates noise from genuine config-space seams.

    Returns q_warm (ny, nx, dof), NaN on FREE cells with no converged candidate."""
    ny, nx = free.shape
    K, dof = cand_q.shape[1], cand_q.shape[2]
    candQ = cand_q.reshape(ny, nx, K, dof)
    candOK = cand_ok.reshape(ny, nx, K)
    have = free & candOK.any(axis=2)
    q_warm = np.full((ny, nx, dof), np.nan, dtype=np.float32)
    if not have.any():
        return q_warm
    ys, xs = np.nonzero(have)                       # start at the FREE cell nearest the FREE centroid
    start = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    s0 = (int(ys[start]), int(xs[start]))
    q_warm[s0] = candQ[s0][np.flatnonzero(candOK[s0])[0]]      # seed cell: its best-cost branch
    seen = np.zeros_like(have)
    seen[s0] = True
    dq = deque([s0])
    while dq:
        iy, ix = dq.popleft()
        qref = q_warm[iy, ix]
        for dy, dx in _NEIGH8:
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < ny and 0 <= jx < nx and have[jy, jx] and not seen[jy, jx]:
                ks = np.flatnonzero(candOK[jy, jx])
                q_warm[jy, jx] = candQ[jy, jx, ks[int(np.argmin([_wrap_linf(candQ[jy, jx, k], qref)
                                                                 for k in ks]))]]
                seen[jy, jx] = True
                dq.append((jy, jx))
    return q_warm


def warm_start_branches_3d(free_vol, cand_q_vol, cand_ok_vol):
    """3D continuity propagation: like warm_start_branches but BFS over the whole FREE VOLUME,
    26-connected. Because it walks between adjacent z-slices, it enforces VERTICAL branch continuity
    -- the edges a climb-over route actually rides on. The per-slice 2D version never sees vertical
    neighbours, so its columns are branch-inconsistent (a voxel and the one directly above it can be
    on different branches for no reason). Each voxel takes the candidate branch nearest the branch of
    the neighbour it is first reached from. Returns q_warm (nz, ny, nx, dof), NaN where no candidate."""
    nz, ny, nx, _K, dof = cand_q_vol.shape
    have = free_vol & cand_ok_vol.any(axis=-1)
    q_warm = np.full((nz, ny, nx, dof), np.nan, dtype=np.float32)
    if not have.any():
        return q_warm
    zi, yi, xi = np.nonzero(have)                      # start at the FREE voxel nearest the 3D centroid
    start = int(np.argmin((zi - zi.mean()) ** 2 + (yi - yi.mean()) ** 2 + (xi - xi.mean()) ** 2))
    s0 = (int(zi[start]), int(yi[start]), int(xi[start]))
    q_warm[s0] = cand_q_vol[s0][np.flatnonzero(cand_ok_vol[s0])[0]]
    seen = np.zeros_like(have)
    seen[s0] = True
    dq = deque([s0])
    while dq:
        iz, iy, ix = dq.popleft()
        qref = q_warm[iz, iy, ix]
        for dz, dy, dx in _NEIGH26:
            jz, jy, jx = iz + dz, iy + dy, ix + dx
            if (0 <= jz < nz and 0 <= jy < ny and 0 <= jx < nx
                    and have[jz, jy, jx] and not seen[jz, jy, jx]):
                q_warm[jz, jy, jx] = _pick_nearest(cand_q_vol[jz, jy, jx], cand_ok_vol[jz, jy, jx], qref)
                seen[jz, jy, jx] = True
                dq.append((jz, jy, jx))
    return q_warm


# label codes for the three-way raster
BEYOND, OBSTACLE, FREE = 0, 1, 2
LABEL_NAMES = {BEYOND: "BEYOND-REACH", OBSTACLE: "OBSTACLE", FREE: "FREE"}


def label_volume(env, planner, ik_on, arm_tag, XX, YY, zs, grasp_q, chunk, num_seeds=None, free_only=False):
    """Sweep the two-pass ON/OFF labelling over every z in the stack. Per slice, identical to the 2D
    label_grid: reach_on = reachable AND collision-free -> FREE; reach_off = reachable ignoring world
    collision; OBSTACLE = reach_off & ~reach_on; BEYOND = ~reach_off. The ON sweep also keeps the
    joint config q per voxel (the gate's fuel). The empty-world OFF solver is built ONCE and reused
    across slices.

    With free_only=True the collision-OFF sweep is SKIPPED entirely (Tier-4 speedup): non-FREE cells
    are all labelled BEYOND (no OBSTACLE/BEYOND split, which is viz-only), halving the IK per slice.
    The OFF-pass time is measured either way so the run can report the projected --free-only cost.

    Returns (label int8 (nz, ny, nx), qfield float32 (nz, ny, nx, dof), off_seconds float)."""
    ny, nx = XX.shape
    nz = len(zs)
    ik_off = None if free_only else _build_ik_solver_no_world(planner, num_seeds)   # empty world = kinematics + self-collision
    label = np.empty((nz, ny, nx), dtype=np.int8)
    qfield = None
    off_seconds = 0.0
    for iz, z in enumerate(zs):
        t_s = time.perf_counter()
        gp = np.zeros((XX.size, 7))
        gp[:, 0] = XX.ravel(); gp[:, 1] = YY.ravel(); gp[:, 2] = z
        gp[:, 3:] = grasp_q
        reach_on_flat, q_flat = _solve_grid_q(env.robot, planner, ik_on, arm_tag, gp, chunk=chunk, num_seeds=num_seeds)
        reach_on = reach_on_flat.reshape(ny, nx)
        if qfield is None:
            qfield = np.full((nz, ny, nx, q_flat.shape[-1]), np.nan, dtype=np.float32)
        qfield[iz] = q_flat.reshape(ny, nx, q_flat.shape[-1])
        anomaly = 0
        if free_only:
            label[iz] = np.where(reach_on, FREE, BEYOND)      # OBSTACLE/BEYOND split (OFF pass) skipped
        else:
            t_off = time.perf_counter()
            reach_off = _solve_grid(env.robot, planner, ik_off, arm_tag, gp, chunk=chunk).reshape(ny, nx)
            off_seconds += time.perf_counter() - t_off
            label[iz] = np.where(reach_on, FREE, np.where(reach_off & ~reach_on, OBSTACLE, BEYOND))
            anomaly = int((reach_on & ~reach_off).sum())      # IK seed noise, not a logic error
        print(f"[label] slice {iz + 1}/{nz} z={z:.3f}  FREE={int(reach_on.sum())} (of {XX.size})"
              + (f"  WARN {anomaly} ON&~OFF kept FREE" if anomaly else "")
              + f"  {time.perf_counter() - t_s:.2f}s")
    if ik_off is not None:
        del ik_off
        rm.torch.cuda.empty_cache()
    return label, qfield, off_seconds


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
# 3D 26-neighbourhood (used by warm_start_branches_3d; forward-referenced, resolved at call time)
_NEIGH26 = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)]


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
def occluder_footprints_3d(env):
    """Per occluder: the posed convex-hull xy footprint of its COLLISION mesh (base<id>.glb -- the
    exact geometry curobo collides against) plus the mesh's true world z-range [zlo, zhi]. We EXTRUDE
    the footprint over [zlo, zhi] to build the 3D occluder mask, so above zhi (the bottle's top) there
    is no occluder and the gripper is free to pass over -- the whole point of going 3D.

    (Approximation: a constant cross-section over-fills the tapering cap slightly; a true per-height
    mesh cross-section is the refinement if the cap ever matters.) Returns a list of
    dict(poly=(K,2)|None, zlo, zhi); None if the mesh can't load (caller falls back to cylinders)."""
    occs = getattr(env, "occluders", None)
    if not occs:
        return []
    try:
        import trimesh
        import transforms3d as t3d
        from scipy.spatial import ConvexHull
        path = os.path.join(os.environ["BENCH_ROOT"], OCCLUDER_COLLISION)
        V0 = np.asarray(trimesh.load(path, force="mesh").vertices)   # same mesh for every bottle
    except Exception as e:
        print(f"[footprint] could not load occluder collision mesh ({e}); falling back to cylinders")
        return None
    out = []
    for occ in occs:
        pose = occ.get_pose()
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=float))   # SAPIEN quat is wxyz
        Vw = V0 @ R.T + np.asarray(pose.p, dtype=float)                 # (V,3) posed world verts
        try:
            poly = Vw[:, :2][ConvexHull(Vw[:, :2]).vertices]
        except Exception:
            poly = None
        out.append(dict(poly=poly, zlo=float(Vw[:, 2].min()), zhi=float(Vw[:, 2].max())))
    return out


def occluder_clearance_3d(foots, occ_ps, XX, YY, zs, res, zres, r_foot):
    """3D clearance (m) from each voxel to the nearest OCCLUDER -- not the table / furniture / target
    that also sit in curobo's world. Primary path: extrude each posed footprint over its z-range into
    a (nz,ny,nx) occluder mask, then the anisotropic 3D Euclidean distance transform (sampling
    (zres,res,res)) -> 0 inside a bottle, growing outward, and UNBOUNDED above the bottle top (no
    occluder there = free to pass over). Fallback (foots is None): vertical cylinders radius r_foot.
    No occluders -> +inf everywhere. Returns (nz, ny, nx) float."""
    ny, nx = XX.shape
    nz = len(zs)
    if foots is not None and any(f["poly"] is not None for f in foots):
        from matplotlib.path import Path as MplPath
        cols = np.column_stack([XX.ravel(), YY.ravel()])
        inxy = np.zeros((len(foots), ny, nx), dtype=bool)      # per-occluder in-footprint (xy), cached
        for i, f in enumerate(foots):
            if f["poly"] is not None:
                inxy[i] = MplPath(f["poly"]).contains_points(cols).reshape(ny, nx)
        mask = np.zeros((nz, ny, nx), dtype=bool)
        for iz, z in enumerate(zs):
            for i, f in enumerate(foots):
                if f["poly"] is not None and f["zlo"] - 1e-9 <= z <= f["zhi"] + 1e-9:
                    mask[iz] |= inxy[i]
        if not mask.any():
            return np.full((nz, ny, nx), np.inf, dtype=float)
        return distance_transform_edt(~mask, sampling=(zres, res, res))
    if len(occ_ps) == 0:
        return np.full((nz, ny, nx), np.inf, dtype=float)
    d = np.full((nz, ny, nx), np.inf, dtype=float)             # vertical cylinders (xy distance only)
    for op in occ_ps:
        dxy = np.hypot(XX - op[0], YY - op[1]) - r_foot
        d = np.minimum(d, np.broadcast_to(dxy, (nz, ny, nx)))
    return np.clip(d, 0.0, None)


def phase3_clearance_report(out_dir, args, xs, ys, zs, label, edt, foots):
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
                if f["poly"] is not None and f["zlo"] - 1e-9 <= zs[k] <= f["zhi"] + 1e-9:
                    ax.add_patch(MplPolygon(f["poly"], closed=True, fill=False, edgecolor="red", lw=1.2))
        ax.set_title(f"z={zs[k]:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=list(axes.ravel()), fraction=0.02, pad=0.02, label="clearance to occluder (m)")
    fig.suptitle(f"Phase 3 occluder clearance  |  seed {args.seed}, arm {args.arm}", fontsize=12)
    stem = f"stack_clearance_seed{args.seed}_{args.arm}"
    fig.savefig(out_dir / f"{stem}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[phase3] wrote {stem}.png")


# ----------------------------------------------------------------------------- step 4: gated 3D DSU
def nearest_free_voxel(free_vol, XX, YY, zs, xyz, max_dist):
    """Voxel (iz, iy, ix) of the nearest FREE voxel to world point xyz (m), or None if the closest is
    farther than max_dist. Used to plant the DSU seeds (grasp end / pad end) in the volume."""
    if not free_vol.any():
        return None
    dxy2 = (XX - xyz[0]) ** 2 + (YY - xyz[1]) ** 2               # (ny, nx), same for every slice
    best = None
    for iz in range(len(zs)):
        d2 = np.where(free_vol[iz], dxy2 + (zs[iz] - xyz[2]) ** 2, np.inf)
        iy, ix = np.unravel_index(int(np.argmin(d2)), d2.shape)
        if best is None or d2[iy, ix] < best[0]:
            best = (float(d2[iy, ix]), (iz, int(iy), int(ix)))
    dist = float(np.sqrt(best[0]))
    return None if dist > max_dist else (best[1], dist)


def widest_path_eps_3d(label, edt, qvol, seed_a, seed_b, tau):
    """26-connected widest-path (Kruskal max-min occluder clearance) bottleneck between seed_a and
    seed_b over FREE voxels. If qvol is not None the edges are GATED: two FREE neighbours union only if
    their warm IK configs are within tau (branch-continuous) -- the mandatory 2.5D joint gate. Add
    voxels in DESCENDING clearance; the clearance of the voxel whose addition first connects the seeds
    is eps*. Clearance VALUES come from edt; the gate only decides whether an edge exists.
    Returns (eps_star_m, bottleneck (iz,iy,ix), merged_bool)."""
    nz, ny, nx = label.shape
    free = (label == FREE).ravel()
    edt_flat = edt.ravel()
    q_flat = qvol.reshape(-1, qvol.shape[-1]) if qvol is not None else None
    parent = np.arange(nz * ny * nx)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def flat(iz, iy, ix):
        return (iz * ny + iy) * nx + ix

    sa, sb = flat(*seed_a), flat(*seed_b)
    if not (free[sa] and free[sb]):
        return 0.0, None, False
    idxs = np.flatnonzero(free)
    order = idxs[np.argsort(-edt_flat[idxs], kind="stable")]      # descending clearance (inf first)
    added = np.zeros(nz * ny * nx, dtype=bool)
    for c in order:
        cz, rem = divmod(int(c), ny * nx)
        cy, cx = divmod(rem, nx)
        added[c] = True
        for dz, dy, dx in _NEIGH26:
            z2, y2, x2 = cz + dz, cy + dy, cx + dx
            if 0 <= z2 < nz and 0 <= y2 < ny and 0 <= x2 < nx:
                nb = flat(z2, y2, x2)
                if added[nb] and free[nb] and (q_flat is None or _wrap_linf(q_flat[c], q_flat[nb]) <= tau):
                    union(int(c), int(nb))
        if added[sa] and added[sb] and find(sa) == find(sb):
            return float(edt_flat[c]), (cz, cy, cx), True
    return 0.0, None, False


def reconstruct_widest_path_3d(free_vol, edt, qvol, seed_a, seed_b, eps_star, tau):
    """Recover an actual bottleneck-optimal route to draw: a gated BFS through {FREE and clearance >=
    eps*}, 26-connected. Returns a list of (iz,iy,ix) or None."""
    if eps_star is None:
        return None
    nz, ny, nx = free_vol.shape
    allowed = free_vol & (edt >= eps_star - 1e-9)                 # inf >= inf holds
    q_flat = qvol.reshape(-1, qvol.shape[-1]) if qvol is not None else None

    def flat(iz, iy, ix):
        return (iz * ny + iy) * nx + ix

    sa, sb = tuple(seed_a), tuple(seed_b)
    if not (allowed[sa] and allowed[sb]):
        return None
    prev = {sa: None}
    dq = deque([sa])
    while dq:
        cur = dq.popleft()
        if cur == sb:
            break
        cz, cy, cx = cur
        for dz, dy, dx in _NEIGH26:
            z2, y2, x2 = cz + dz, cy + dy, cx + dx
            if (0 <= z2 < nz and 0 <= y2 < ny and 0 <= x2 < nx and allowed[z2, y2, x2]
                    and (z2, y2, x2) not in prev
                    and (q_flat is None or _wrap_linf(q_flat[flat(cz, cy, cx)], q_flat[flat(z2, y2, x2)]) <= tau)):
                prev[(z2, y2, x2)] = cur
                dq.append((z2, y2, x2))
    if sb not in prev:
        return None
    path, node = [], sb
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


def _metric_path3d(out_dir, args, foots, occ_ps, g_xyz, p_xyz, bott_xyz, route_w, eps_star, merged):
    """3D view of the gated climb-over route through the stack; occluders as vertical prisms."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    if foots:
        for f in foots:
            if f["poly"] is None:
                continue
            p, zlo, zhi = f["poly"], f["zlo"], f["zhi"]
            bottom = [(x, y, zlo) for x, y in p]
            top = [(x, y, zhi) for x, y in p]
            faces = [bottom, top] + [[bottom[i], bottom[(i + 1) % len(p)], top[(i + 1) % len(p)], top[i]]
                                     for i in range(len(p))]
            ax.add_collection3d(Poly3DCollection(faces, facecolor="red", alpha=0.12, edgecolor="red", lw=0.5))
    if route_w and len(route_w) > 1:
        rx, ry, rz = zip(*route_w)
        ax.plot(rx, ry, rz, "-", color="gold", lw=2.5, label="gated widest path")
    ax.scatter(*g_xyz, c="cyan", marker="o", s=80, label="grasp seed")
    ax.scatter(*p_xyz, c="magenta", marker="s", s=70, label="pad seed")
    if bott_xyz is not None:
        ax.scatter(*bott_xyz, c="black", marker="X", s=80, label="bottleneck eps*")
    eps_txt = "inf" if (merged and np.isinf(eps_star)) else (f"{eps_star:.3f} m" if merged else "INACCESSIBLE")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"2.5D metric route  |  seed {args.seed}, arm {args.arm}\neps* (gated) = {eps_txt}")
    ax.legend(loc="upper left", fontsize=8)
    stem = f"metric_path3d_seed{args.seed}_{args.arm}"
    fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def phase4_metric(out_dir, args, XX, YY, zs, label, edt, q_gate, grasp_pose, tgt_p, pad_xy, occ_ps, foots):
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
        "route_len_voxels": (len(route) if route else 0),
        "route_climbs_to_z": (round(max(p[2] for p in route_w), 3) if route_w else None),
        "config": {"seed": args.seed, "arm": args.arm, "offset": args.offset,
                   "zmin": args.zmin, "zmax": args.zmax, "zres": args.zres},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    _metric_path3d(out_dir, args, foots, occ_ps, w(ga), w(pa), (w(bott_g) if bott_g else None),
                   route_w, eps_g, merged_g)
    phase4_visuals(out_dir, args, XX, YY, zs, label, edt, foots, w(ga), w(pa), route, route_w,
                   eps_g, merged_g)
    print(f"[metric] wrote {stem}.json + 3D path + side/profile/topdown/ceiling figures")


# --------------------------------------------------------------------- step 4b: 3D-legible visuals
def _line_axis(g_xy, p_xy):
    """Unit direction + length of the grasp->pad line in the xy plane (the profile/side-view axis)."""
    d = np.asarray(p_xy, float) - np.asarray(g_xy, float)
    L = float(np.hypot(*d))
    u = d / L if L > 1e-9 else np.array([1.0, 0.0])
    return np.asarray(g_xy, float), u, L


def _viz_side_elevation(out_dir, args, XX, YY, zs, label, foots, g_xyz, p_xyz, route_w, eps_star, merged):
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
            proj = (f["poly"] - p0) @ u                       # footprint projected onto the line axis
            ax.add_patch(plt.Rectangle((float(proj.min()), f["zlo"]), float(proj.max() - proj.min()),
                                       f["zhi"] - f["zlo"], fill=False, edgecolor="red", lw=2))
    if route_w and len(route_w) > 1:
        rs = [float((np.asarray(r[:2]) - p0) @ u) for r in route_w]
        ax.plot(rs, [r[2] for r in route_w], "-", color="gold", lw=2.5, label="route")
    ax.plot(0, g_xyz[2], "o", color="cyan", ms=11, mec="k", label="grasp")
    ax.plot(L, p_xyz[2], "s", color="magenta", ms=10, mec="k", label="pad")
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


def _viz_topdown(out_dir, args, XX, YY, label, foots, g_xyz, p_xyz, route_w):
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
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Top-down: route coloured by height  |  seed {args.seed}, arm {args.arm}\n"
                 f"green = reachable at some height")
    stem = f"metric_topdown_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def _viz_ceiling(out_dir, args, XX, YY, zs, label, foots):
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
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Reachability ceiling (max FREE z per x,y)  |  seed {args.seed}, arm {args.arm}"
                 + (f"\nbottle top z={zhi:.2f}m -- brighter cells can clear over it" if zhi else ""))
    stem = f"metric_ceiling_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def phase4_visuals(out_dir, args, XX, YY, zs, label, edt, foots, g_xyz, p_xyz, route, route_w, eps_star, merged):
    """The four 3D-legible views: side elevation, clearance-along-route, top-down-by-height, ceiling."""
    route_clear = [float(edt[iz, iy, ix]) for (iz, iy, ix) in route] if route else []
    _viz_side_elevation(out_dir, args, XX, YY, zs, label, foots, g_xyz, p_xyz, route_w, eps_star, merged)
    _viz_clearance_profile(out_dir, args, route_w, route_clear, eps_star, merged)
    _viz_topdown(out_dir, args, XX, YY, label, foots, g_xyz, p_xyz, route_w)
    _viz_ceiling(out_dir, args, XX, YY, zs, label, foots)


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

    # --- PHASE 1 step 1: label the whole z-stack (+ q per voxel) ---
    with tm.section("label_volume"):
        label, qfield, off_seconds = label_volume(env, planner, ik, args.arm, XX, YY, zs, grasp_q,
                                                  args.chunk, num_seeds=args.ik_seeds, free_only=args.free_only)
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
        foots = occluder_footprints_3d(env)
        if foots:
            for i, f in enumerate(foots):
                if f["poly"] is not None:
                    bb = tuple(np.round(f["poly"].max(0) - f["poly"].min(0), 3))
                    print(f"[footprint] occ {i}: z-range [{f['zlo']:.3f},{f['zhi']:.3f}]  bbox(x,y)={bb}")
        edt = occluder_clearance_3d(foots, occ_ps, XX, YY, zs, args.res, args.zres, OCC_HALF_FOOTPRINT)
    freev = label == FREE
    if np.isinf(edt).all():
        print("[edt] no occluders -> occluder-clearance unbounded (nothing to squeeze past)")
    elif freev.any():
        fe = edt[freev][np.isfinite(edt[freev])]
        if fe.size:
            print(f"[edt] occluder-clearance over FREE voxels (m): min={fe.min():.3f} "
                  f"median={np.median(fe):.3f} max={fe.max():.3f}")

    with tm.section("phase3_clearance_report"):
        phase3_clearance_report(out_dir, args, xs, ys, zs, label, edt, foots)

    # --- step 4: gated 26-conn widest-path DSU -> the 2.5D metric (eps*) ---
    with tm.section("metric_dsu"):
        phase4_metric(out_dir, args, XX, YY, zs, label, edt, q_warm_3d, grasp_pose, tgt_p, pad_xy,
                      occ_ps, foots)

    with tm.section("phase1_report"):
        phase1_stack_report(out_dir, args, xs, ys, zs, label, qfield, q_warm_vol)

    # --- persist the raw volumes so the run is re-analysable offline (no GPU re-run needed) ---
    with tm.section("save_data"):
        save_kw = {"label": label, "qfield": qfield, "edt": edt, "xs": xs, "ys": ys, "zs": zs}
        if q_warm_3d is not None:
            save_kw.update(q_warm_2d=q_warm_2d, q_warm_3d=q_warm_3d)
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
    ap.add_argument("--chunk", type=int, default=256,
                    help="IK poses per batch; lower if you hit CUDA OOM (planners already use ~9GB)")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR),
                    help="results location; each run lands in its own <out-dir>/<timestamp>/ subfolder")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
