#!/usr/bin/env python3
"""
Phase 2 of SEED_TRAJECTORY_PLAN.md: build a curobo trajopt seed from the clearance-metric route.

This module turns the 3D clearance metric's *gated widest-path* into a joint-space trajectory that
curobo can be seeded with (idea #5). It reuses clearance_metric_3d.py verbatim as a library -- none
of the metric maths is re-implemented here; we only (2a) drive the pipeline for the grasp-approach
segment and read the per-voxel joint configs off the warm branch field, and (later, 2b/2c) resample
that to curobo's action_horizon and shape it into the seed tensor.

2a ONLY (this step): `compute_route_configs` runs the metric for one arm at one grasp orientation and
returns the ordered per-voxel joint configs along the collision-free climb-over route from the
gripper's CURRENT position to the grasp. The difference from clearance_metric_3d.phase4_metric is the
seed pair: here the route is (current gripper -> grasp) -- the approach the arm must actually plan --
not (grasp -> pad). Endpoint welding + resampling to 28 steps + tensor shaping are 2b/2c.

NOT standalone-runnable: it needs a live scene (env/planner/ik) the caller already has (the expert in
Phase 3, or clearance_metric_3d's own run()). Importing it pulls in clearance_metric_3d -> reachability_map
-> torch + the env stack, exactly as the metric tool does.

COST NOTE (SEED_TRAJECTORY_PLAN.md sec.5): the default grid is the metric's own (res 0.01, zres 0.03
over z in [0.78,1.4]) -> ~1.7e5 voxels of IK PLUS the multi-branch warm solve. That is expensive
per call; a coarser SeedMetricConfig (bigger res/zres) is the first knob if per-candidate cost bites.
"""

from dataclasses import dataclass
import time

import numpy as np

from lib.continuity import warm_start_branches_3d
from lib.ik_grid import _solve_grid_q_multi, build_grid
from lib.labeling import FREE, label_volume
from lib.obstacles import obstacle_centers, occluder_clearance_3d, occluder_footprints_3d
from lib.scene_constants import OCC_HALF_FOOTPRINT
from lib.widest_path import (
    nearest_free_voxel, reconstruct_widest_path_3d, widest_path_eps_3d,
)

# clearance_metric_3d is imported LAZILY inside the scene-coupled functions (2a): it pulls in
# reachability_map -> torch + the env stack. The pure resampler (2b, resample_route_to_seed) must NOT
# require any of that, so it stays CPU-unit-testable via `python seed_from_clearance.py --selftest`.


@dataclass
class SeedMetricConfig:
    """Metric knobs for the seed route. Defaults mirror clearance_metric_3d.py's argparse so the seed
    route matches the tool's eps* geometry. Field names match what build_grid/label_volume read, so an
    instance can be passed straight through as the metric's `args`-like object."""
    # grid bounds (build_grid reads these by name)
    xmin: float = -0.6
    xmax: float = 0.6
    ymin: float = -0.35
    ymax: float = 0.35
    res: float = 0.01
    zmin: float = 0.78           # ~grasp height (table ~0.74)
    # Ceiling of the climb-over grid. 1.23 leaves headroom over the olive-oil top while cutting
    # ~30% of the z-slices vs the old 1.4. It must stay ABOVE the tallest obstacle: a route that
    # needs to pass over something higher than zmax cannot connect, and shows up as a build miss
    # rather than as a long way round.
    zmax: float = 1.23
    zres: float = 0.03
    # metric knobs
    gate_tau: float = 0.35       # rad; joint-continuity edge gate (the mandatory 2.5D gate)
    # Ladder of LOOSER taus tried when gate_tau itself fails to connect the two seeds but the
    # UNGATED path does -- i.e. the route exists geometrically and only the gate cut it. merged
    # is monotone non-decreasing in tau (a larger tau admits strictly more edges), so the scan
    # runs ascending and stops at the first value that connects: the answer is "the smallest tau
    # that would have worked", not a guess. Costs nothing but a few extra widest-path runs over
    # volumes already in memory -- no IK, no rebuild, which is the whole point of keeping label/
    # edt/q_warm_3d on the result. Set () to disable.
    gate_tau_sweep: tuple = (0.5, 0.7, 1.0, 1.5, 2.0)
    seed_snap: float = 0.10      # m; max snap of an endpoint to the nearest FREE voxel
    warm_seeds: int = 8          # multi-branch candidates per voxel (return_seeds)
    ik_seeds: int = 30           # IK seeds per solve
    # IK batch size. This is the knob that bounds PEAK GPU memory during the grid solve --
    # lower it (128 / 64) if a scene still OOMs after the solver-release fix, since it caps
    # the allocation without coarsening the grid or losing any fidelity.
    chunk: int = 256
    occ_shape: str = "mesh"      # true tapering collision mesh (vs "extruded")
    # Which actors the clearance field measures against. "all" = every mesh in
    # env.collision_list except the target and pad, so table CLUTTER is an obstacle too and the
    # seed routes around it -- required for the seed to do anything on a scene with no occluder.
    # "occluders" restores the curated-ring-only behaviour. See clearance_metric_3d.--obstacles.
    obstacles: str = "all"
    free_only: bool = False      # keep the OBSTACLE/BEYOND split off the FREE labels


@dataclass
class RouteResult:
    """What 2a produces. route_qs is the fuel for 2b/2c (resample + weld + shape)."""
    route_qs: np.ndarray | None            # (K, dof) per-voxel joint configs, current-gripper -> grasp; None if no route
    route_voxels: list | None              # [(iz,iy,ix), ...] same order
    route_world: list | None               # [(x,y,z), ...] same order (for debugging/figures)
    merged: bool                           # did the gated widest-path connect the two seeds?
    eps_gated: float                       # bottleneck clearance (m); inf = clears over the top
    reason: str | None                     # why route_qs is None (unsnappable seed / disconnected), else None
    # raw volumes kept so a caller can debug or feed 2b without re-running the metric
    q_warm_3d: np.ndarray | None = None
    label: np.ndarray | None = None
    XX: np.ndarray | None = None
    YY: np.ndarray | None = None
    zs: np.ndarray | None = None
    seconds: float = 0.0
    # diagnostics: the ungated (reach+clear only) result + the raw pieces to re-run the widest-path at
    # other gate_tau without recomputing the metric (geometry-fail vs gate-seam-fail; tau sweeps)
    merged_ungated: bool = False
    eps_ungated: float = 0.0
    # [(tau, merged, eps), ...] from the ascending gate_tau_sweep, present only when the
    # configured gate_tau failed on a route the ungated pass found. The first entry with
    # merged=True is the smallest tau that would have connected this scene.
    tau_sweep: list | None = None
    tau_needed: float | None = None      # that smallest connecting tau, or None if none did
    edt: np.ndarray | None = None
    seed_start: tuple | None = None
    seed_goal: tuple | None = None
    # scene bits for the route visuals (save_route_visuals)
    foots: list | None = None
    occ_ps: list | None = None
    start_xyz: np.ndarray | None = None
    goal_xyz: np.ndarray | None = None
    bott_xyz: tuple | None = None
    tgt_p: np.ndarray | None = None


def _build_warm_field(env, planner, ik, arm, grasp_q, XX, YY, zs, label, cfg):
    """The mandatory joint gate needs a per-voxel branch-consistent config field. This mirrors the
    warm block of clearance_metric_3d.run(): a multi-branch IK solve on the FREE voxels, then 26-conn
    3D continuity propagation (vertical continuity is what a climb-over route rides on). Returns
    q_warm_3d (nz, ny, nx, dof), NaN where no converged candidate."""
    nz = len(zs)
    ny, nx = label.shape[1], label.shape[2]
    dof, K = None, cfg.warm_seeds
    free_vol = label == FREE
    xr, yr = XX.ravel(), YY.ravel()
    cand_q_vol = None
    cand_ok_vol = np.zeros((nz, ny, nx, K), dtype=bool)
    for iz, z in enumerate(zs):
        idx = np.flatnonzero(free_vol[iz].ravel())         # Tier 2: solve only FREE cells
        if not idx.size:
            continue
        gp = np.zeros((idx.size, 7))
        gp[:, 0] = xr[idx]; gp[:, 1] = yr[idx]; gp[:, 2] = z; gp[:, 3:] = grasp_q
        cand_q, cand_ok = _solve_grid_q_multi(env.robot, planner, ik, arm, gp,
                                                 chunk=cfg.chunk, return_seeds=K, num_seeds=cfg.ik_seeds)
        if cand_q_vol is None:
            dof = cand_q.shape[-1]
            cand_q_vol = np.full((nz, ny, nx, K, dof), np.nan, dtype=np.float32)
        cand_q_vol[iz].reshape(ny * nx, K, dof)[idx] = cand_q
        cand_ok_vol[iz].reshape(ny * nx, K)[idx] = cand_ok
    if cand_q_vol is None:                                  # no FREE voxel had a converged branch
        return None
    return warm_start_branches_3d(free_vol, cand_q_vol, cand_ok_vol)


def sweep_gate_tau(widest_path_at, taus, gate_tau):
    """Smallest tau in `taus` that connects the two seeds. Returns ([(tau, merged, eps), ...],
    tau_needed_or_None), the list holding every tau actually tried, in ascending order.

    widest_path_at(tau) -> (eps, bottleneck, merged); the caller closes over the volumes so this
    stays pure and testable without a scene. Only taus strictly LOOSER than the configured
    gate_tau are tried -- it already failed, and the gate is monotone (a larger tau admits a
    superset of edges), so anything tighter cannot connect either. That same monotonicity is why
    the scan stops at the first success: the first hit IS the minimum over the ladder."""
    tried, needed = [], None
    for tau in sorted(float(t) for t in taus if float(t) > float(gate_tau)):
        eps, _bott, merged = widest_path_at(tau)
        tried.append((float(tau), bool(merged), float(eps)))
        if merged:
            needed = float(tau)
            break
    return tried, needed


def compute_route_configs(env, planner, arm, ik, grasp_q, start_xyz, goal_xyz,
                          cfg: SeedMetricConfig = None):
    """2a: run the clearance metric for `arm` at orientation `grasp_q` and return the ordered per-voxel
    joint configs along the gated climb-over route from the gripper's current position (`start_xyz`) to
    the grasp (`goal_xyz`).

    Args:
        env, planner, ik: the live scene + this arm's curobo planner and its (world-collision-ON) IK
            solver -- exactly the objects clearance_metric_3d.run() builds via select_arm.
        arm: "left" | "right".
        grasp_q: (4,) world grasp quaternion the whole grid is evaluated at (the candidate's grasp
            orientation -- the orientation coupling from SEED_TRAJECTORY_PLAN.md sec.5).
        start_xyz, goal_xyz: (3,) world gripper positions for the two widest-path seeds (current
            gripper, grasp). The metric snaps each to the nearest FREE voxel within cfg.seed_snap.
        cfg: SeedMetricConfig (defaults = the metric tool's own knobs).

    Returns:
        RouteResult. route_qs (K, dof) is None when a seed can't be snapped to FREE or the gated
        widest-path can't connect the two (reason is set). The joint order is curobo's active-joint
        order (the IK solver's dof axis), i.e. what 2c must keep to match the seed tensor.
    """
    cfg = cfg or SeedMetricConfig()
    t0 = time.perf_counter()

    xs, ys, zs, XX, YY = build_grid(cfg)

    # label the z-stack (reach-envelope pruning left OFF here: the builder shouldn't depend on a
    # precomputed artifact; the expert can pass a pruned variant later if cost demands it)
    label, qfield, _off = label_volume(
        env, planner, ik, arm, XX, YY, zs, grasp_q, cfg.chunk,
        num_seeds=cfg.ik_seeds, free_only=cfg.free_only, prune_mask=None)

    free_vol = label == FREE
    if not free_vol.any():
        return RouteResult(None, None, None, False, 0.0, "no FREE voxel in the grid",
                           label=label, XX=XX, YY=YY, zs=zs, seconds=time.perf_counter() - t0)

    # warm branch field (mandatory for the gate) -- the per-voxel joint configs the route reads from
    q_warm_3d = _build_warm_field(env, planner, ik, arm, grasp_q, XX, YY, zs, label, cfg)
    if q_warm_3d is None:
        return RouteResult(None, None, None, False, 0.0, "no converged warm branch on any FREE voxel",
                           label=label, XX=XX, YY=YY, zs=zs, seconds=time.perf_counter() - t0)

    # obstacle clearance field (edt) -- the widest-path values come from this. Under
    # cfg.obstacles="all" this covers table clutter as well as the occluder ring, so the route
    # squeezes between whatever is actually on the table (see clearance_metric_3d.--obstacles).
    foots = occluder_footprints_3d(env, obstacles=cfg.obstacles)
    # Centres track the obstacle set actually used (cylinder fallback + plot markers).
    occ_ps = obstacle_centers(foots) or (
        [np.array(o.get_pose().p) for o in env.occluders]
        if getattr(env, "occluders", None) else [])
    edt = occluder_clearance_3d(foots, occ_ps, XX, YY, zs, cfg.res, cfg.zres,
                                   OCC_HALF_FOOTPRINT, shape=cfg.occ_shape)

    # plant the two seeds: current gripper -> grasp (the APPROACH, unlike phase4_metric's grasp->pad)
    seed_s = nearest_free_voxel(free_vol, XX, YY, zs, np.asarray(start_xyz, float), cfg.seed_snap)
    seed_g = nearest_free_voxel(free_vol, XX, YY, zs, np.asarray(goal_xyz, float), cfg.seed_snap)
    if seed_s is None or seed_g is None:
        which = "start(gripper)" if seed_s is None else "goal(grasp)"
        return RouteResult(None, None, None, False, 0.0, f"{which} seed unsnappable within {cfg.seed_snap}m",
                           q_warm_3d=q_warm_3d, label=label, XX=XX, YY=YY, zs=zs,
                           seconds=time.perf_counter() - t0)
    (sa, _sd), (ga, _gd) = seed_s, seed_g

    # ungated (reach+clear only) vs gated (adds the joint-continuity gate). The split turns a NO-ROUTE
    # into an actionable cause: ungated ALSO fails -> geometry/reachability (finer res / wider bounds);
    # ungated connects but gated fails -> a real branch seam the gate cuts (raise gate_tau / finer res,
    # since coarse res inflates adjacent-voxel joint steps and makes the fixed-tau gate over-cut).
    eps_u, _bu, merged_u = widest_path_eps_3d(label, edt, None, sa, ga, cfg.gate_tau)
    eps_g, bott_g, merged_g = widest_path_eps_3d(label, edt, q_warm_3d, sa, ga, cfg.gate_tau)

    def _w(v):                                                    # voxel (iz,iy,ix) -> world (x,y,z)
        return (float(XX[v[1], v[2]]), float(YY[v[1], v[2]]), float(zs[v[0]]))
    try:
        tgt_p = np.asarray(env.target_obj.get_pose().p, dtype=float)
    except Exception:
        tgt_p = None
    # Tau sweep: only when the gate is the thing that cut an otherwise-existing route. If the
    # UNGATED pass also failed, no tau can help (the route is geometrically absent) and running
    # the ladder would just burn time to reprint that. Ascending + early exit, because merged is
    # monotone in tau. Reuses label/edt/q_warm_3d as they are -- no IK, no rebuild.
    tau_sweep, tau_needed = None, None
    if merged_u and not merged_g and cfg.gate_tau_sweep:
        tau_sweep, tau_needed = sweep_gate_tau(
            lambda tau: widest_path_eps_3d(label, edt, q_warm_3d, sa, ga, tau),
            cfg.gate_tau_sweep, cfg.gate_tau)

    diag = dict(merged_ungated=bool(merged_u), eps_ungated=float(eps_u), edt=edt,
                seed_start=sa, seed_goal=ga, q_warm_3d=q_warm_3d, label=label, XX=XX, YY=YY, zs=zs,
                foots=foots, occ_ps=occ_ps, start_xyz=np.asarray(start_xyz, float),
                goal_xyz=np.asarray(goal_xyz, float), tgt_p=tgt_p,
                tau_sweep=tau_sweep, tau_needed=tau_needed,
                bott_xyz=(_w(bott_g) if (merged_g and bott_g) else None))

    if not merged_g:
        reason = ("gripper->grasp has NO reachable+clear route even UNGATED "
                  "(grid too coarse / bounds too tight / not reachable at grasp orientation)"
                  if not merged_u else
                  f"ungated route EXISTS but the joint gate (tau={cfg.gate_tau}) disconnects it "
                  f"(branch seam; coarse res inflates adjacent-voxel joint steps -> gate over-cuts)")
        # Append the sweep's verdict so the actionable number reaches records.jsonl, not just
        # the console: seed_stats.reason is what the summary reports, and "set SEED_GATE_TAU=X"
        # is the whole point of running the sweep.
        if tau_needed is not None:
            reason += (f" -- SWEEP: tau={tau_needed} WOULD connect "
                       f"(eps={dict((t, e) for t, m, e in tau_sweep)[tau_needed]:.3f}m); "
                       f"set SEED_GATE_TAU={tau_needed}")
        elif tau_sweep:
            # No tau connecting is NOT "the gate is too tight" -- at a large enough tau the gate
            # admits every edge the ungated pass does, so if it still fails the blocker cannot be
            # the threshold. It is NaN holes in the warm field: _wrap_linf returns NaN where a
            # voxel has no converged branch, and `NaN <= tau` is False at EVERY tau, so such a
            # voxel is permanently un-gateable while still counting as FREE for the ungated pass.
            # The levers are warm-field convergence (warm_seeds, ik_seeds), not tau and not res.
            reason += (f" -- SWEEP: no tau up to {max(t for t, _, _ in tau_sweep)} connects it, so the "
                       "gate THRESHOLD is not the blocker: the route must cross voxels with no "
                       "converged warm config (NaN, un-gateable at any tau). Levers: warm_seeds / "
                       "ik_seeds, not SEED_GATE_TAU")
        return RouteResult(None, None, None, False, float(eps_g), reason,
                           seconds=time.perf_counter() - t0, **diag)

    route = reconstruct_widest_path_3d(free_vol, edt, q_warm_3d, sa, ga, eps_g, cfg.gate_tau)
    if not route:
        return RouteResult(None, None, None, True, float(eps_g),
                           "seeds merged but route reconstruction returned empty",
                           seconds=time.perf_counter() - t0, **diag)

    # per-voxel joint configs along the route (current-gripper end -> grasp end), in curobo dof order
    route_qs = np.asarray([q_warm_3d[iz, iy, ix] for (iz, iy, ix) in route], dtype=np.float32)
    route_world = [(float(XX[iy, ix]), float(YY[iy, ix]), float(zs[iz])) for (iz, iy, ix) in route]

    # gate guarantees finite configs on traversed voxels, but assert it so a NaN never reaches 2b
    if not np.all(np.isfinite(route_qs)):
        return RouteResult(None, route, route_world, True, float(eps_g),
                           "route contains a voxel with no finite warm config (unexpected under gating)",
                           seconds=time.perf_counter() - t0, **diag)

    return RouteResult(route_qs, route, route_world, True, float(eps_g), None,
                       seconds=time.perf_counter() - t0, **diag)


# --------------------------------------------------------------------- 2b: resample + weld (PURE)
def resample_route_to_seed(route_qs, start_q, goal_q, action_horizon=28, eps=1e-9):
    """2b: turn the route's per-voxel configs into a single trajopt seed ROW of exactly
    `action_horizon` timesteps, welded to the real endpoints.

    Pure / CPU-only (no curobo, no scene) so it is unit-testable. Steps:
      1. waypoints = [start_q] + route_qs + [goal_q]  -- weld the real start and curobo's goal onto the
         route (route_qs[0]/[-1] are grasp-ORIENTATION IK at the endpoints, so they differ from the real
         start/goal configs; keeping them as interior waypoints preserves the climb-over shape while the
         added endpoints pin the trajectory exactly where trajopt needs it).
      2. np.unwrap along time so revolute joints take the SHORTEST angular path (no phantom 2*pi sweep).
         The 2.5D gate already keeps route-internal steps < tau; only the two weld segments can exceed
         pi, and unwrap handles those. unwrap leaves waypoint[0] (=start_q) untouched.
      3. arc-length (joint-space L2) parameterization, then linear-interpolate each joint to
         `action_horizon` samples. tstep 0 == start_q exactly; the last tstep is curobo's goal config
         (up to a co-terminal 2*pi on a wrapped joint -- same pose, kept continuous on purpose).

    Args:
        route_qs: (K, dof) ordered configs current-gripper->grasp (RouteResult.route_qs), or None/empty.
        start_q:  (dof,) the real robot start config (curobo dof order).
        goal_q:   (dof,) curobo's goal config for the grasp (curobo dof order).
        action_horizon: seed length (28 for this robot -- see SEED_TRAJECTORY_PLAN.md 0a).

    Returns:
        (action_horizon, dof) float32, or None if the trajectory is degenerate (zero joint-space
        length, e.g. start==goal and no route) -- caller then supplies no seed (stock behavior).
    """
    if action_horizon < 2:
        raise ValueError("action_horizon must be >= 2")
    start_q = np.asarray(start_q, dtype=np.float64).reshape(-1)
    goal_q = np.asarray(goal_q, dtype=np.float64).reshape(-1)
    dof = start_q.shape[0]
    if goal_q.shape[0] != dof:
        raise ValueError(f"start_q dof {dof} != goal_q dof {goal_q.shape[0]}")

    if route_qs is None or len(route_qs) == 0:
        wp = np.stack([start_q, goal_q], axis=0)
    else:
        route_qs = np.asarray(route_qs, dtype=np.float64).reshape(len(route_qs), -1)
        if route_qs.shape[1] != dof:
            raise ValueError(f"route_qs dof {route_qs.shape[1]} != start/goal dof {dof}")
        wp = np.concatenate([start_q[None, :], route_qs, goal_q[None, :]], axis=0)   # (M, dof)

    wp = np.unwrap(wp, axis=0)                                    # shortest-path continuity per joint
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)            # joint-space segment lengths
    s = np.concatenate([[0.0], np.cumsum(seg)])                  # cumulative arc length
    if s[-1] <= eps:                                             # degenerate: nothing to move
        return None

    ts = np.linspace(0.0, s[-1], action_horizon)
    row = np.empty((action_horizon, dof), dtype=np.float32)
    for j in range(dof):
        row[:, j] = np.interp(ts, s, wp[:, j])
    row[0] = start_q.astype(np.float32)                          # exact start weld (guard vs float drift)
    return row


# --------------------------------------------------------------------- 2c: shape into the seed tensor
def route_qs_to_seed_tensor(route_qs, start_q, goal_q, action_horizon=28):
    """2c: 2b's resampled row -> the CPU torch tensor curobo's injection expects.

    Returns a ``(1, 1, action_horizon, dof)`` float32 CPU tensor (k=1 seed, batch=1) -- the format
    MotionGenPlanConfig.seed_traj accepts (motion_gen normalizes device/dtype and fills the other 15
    slots with linear seeds). CPU on purpose: it is picklable across the plan_path multiprocessing pipe
    (communication_flag), and motion_gen re-moves it to CUDA. The dof axis MUST already be in curobo's
    active-joint order (route_qs comes from the IK solver; start_q/goal_q are the caller's job to order
    identically -- see SEED_TRAJECTORY_PLAN.md 0a: [fl/fr_joint1..6]).

    Returns None when 2b is degenerate (caller then supplies no seed -> stock behavior)."""
    row = resample_route_to_seed(route_qs, start_q, goal_q, action_horizon)   # (H, dof) or None
    if row is None:
        return None
    import torch
    t = torch.from_numpy(np.ascontiguousarray(row, dtype=np.float32))         # (H, dof)
    return t.unsqueeze(0).unsqueeze(0).contiguous()                           # (1, 1, H, dof), CPU float32


def build_seed(env, planner, arm, ik, grasp_q, start_q, goal_q, start_xyz, goal_xyz,
               cfg: SeedMetricConfig = None, action_horizon=28):
    """Phase-3 entry point: run 2a (metric route) + 2b (resample/weld) + 2c (tensor) in one call.

    start_q / goal_q are the REAL start config and curobo's grasp goal config (dof,), in curobo's
    active-joint order. Either may be None: start_q defaults to the route's start-end config and goal_q
    to its grasp-end config (both from the warm field). Weld tstep 0 to the REAL start_q whenever you
    have it (trajopt's start state is fixed there); goal_q as the route's own end is usually fine (a
    seed only initializes; trajopt optimizes toward the goal POSE). start_xyz / goal_xyz are the world
    gripper positions that seed the metric's widest-path (current gripper, grasp). Returns
    (seed_tensor_or_None, RouteResult) so the caller can pass the seed to plan_path(seed_traj=...) and
    log why it was None (RouteResult.reason)."""
    res = compute_route_configs(env, planner, arm, ik, grasp_q, start_xyz, goal_xyz, cfg)
    if res.route_qs is None:
        return None, res
    sq = start_q if start_q is not None else res.route_qs[0]
    gq = goal_q if goal_q is not None else res.route_qs[-1]
    seed = route_qs_to_seed_tensor(res.route_qs, sq, gq, action_horizon)
    return seed, res


def save_route_visuals(res: RouteResult, out_dir, seed_label="0", arm="left", cfg=None, gripper_r=0.03):
    """Render the metric's OWN route views for a seeded rollout into out_dir (reused verbatim, no new
    plotting): the 3D climb-over path (_metric_path3d), top-down route coloured by height (_viz_topdown),
    and the side elevation (_viz_side_elevation) -- so you can SEE the exact path this rollout's seed
    encodes. The two widest-path ends here are (gripper=start, grasp=goal); cm's legend still says
    "grasp/pad seed" (cosmetic). No-op when there's no route; never raises (visuals must not break a
    rollout)."""
    if res is None or res.route_world is None:
        return
    from pathlib import Path as _Path
    from types import SimpleNamespace
    import clearance_metric_3d as cm
    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or SeedMetricConfig()
    args = SimpleNamespace(seed=seed_label, arm=arm, res=cfg.res, zres=cfg.zres,
                           occ_shape=cfg.occ_shape, gripper_r=gripper_r, zmin=cfg.zmin, zmax=cfg.zmax)
    g_xyz, p_xyz = res.start_xyz, res.goal_xyz          # gripper (start) -> grasp (goal)
    try:
        cm._metric_path3d(out_dir, args, res.foots, res.occ_ps, g_xyz, p_xyz, res.bott_xyz,
                          res.route_world, res.eps_gated, res.merged, tgt_p=res.tgt_p, ee_xyz=g_xyz)
        cm._viz_topdown(out_dir, args, res.XX, res.YY, res.label, res.foots, g_xyz, p_xyz,
                        res.route_world, tgt_p=res.tgt_p, ee_xyz=g_xyz)
        cm._viz_side_elevation(out_dir, args, res.XX, res.YY, res.zs, res.label, res.foots,
                               g_xyz, p_xyz, res.route_world, res.eps_gated, res.merged,
                               tgt_p=res.tgt_p, ee_xyz=g_xyz)
        print(f"[seed-viz] wrote 3D/topdown/side route figures -> {out_dir}")
    except Exception as e:  # a plotting failure must never abort the rollout
        print(f"[seed-viz] route visuals failed ({e}); continuing")


def _selftest():
    """CPU self-test for resample_route_to_seed (2b). No curobo/scene needed."""
    rng = np.random.default_rng(0)
    H, dof = 28, 6

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    # 1) straight-line route resamples to a monotone line, endpoints exact/pose-correct
    s0 = np.zeros(dof); g0 = np.full(dof, 0.5)
    route = np.linspace(s0, g0, 10)
    row = resample_route_to_seed(route, s0, g0, H)
    assert row.shape == (H, dof), row.shape
    assert np.allclose(row[0], s0, atol=1e-6), "start not welded"
    assert np.max(np.abs(wrap(row[-1] - g0))) < 1e-5, "goal not welded (mod 2pi)"
    assert np.all(np.diff(row[:, 0]) >= -1e-6), "not monotone on a straight route"
    assert np.isfinite(row).all()

    # 2) welding: route endpoints differ from real start/goal -> output still pinned to start/goal
    s1 = rng.uniform(-1, 1, dof); g1 = rng.uniform(-1, 1, dof)
    route2 = np.linspace(s1 + 0.2, g1 - 0.2, 6)                  # route ends offset from start/goal
    row2 = resample_route_to_seed(route2, s1, g1, H)
    assert np.allclose(row2[0], s1, atol=1e-6), "start weld broken"
    assert np.max(np.abs(wrap(row2[-1] - g1))) < 1e-5, "goal weld broken"

    # 3) wraparound: start/goal near +/-pi should take the SHORT path (no ~2pi sweep)
    s3 = np.full(dof, 3.0); g3 = np.full(dof, -3.0)              # short way is +0.28 rad, long way ~6.0
    row3 = resample_route_to_seed(None, s3, g3, H)
    tv = np.sum(np.abs(np.diff(row3[:, 0])))                     # total variation on joint 0
    assert tv < 0.5, f"took the long way around (tv={tv:.3f}, expected ~0.28)"
    assert np.max(np.abs(wrap(row3[-1] - g3))) < 1e-5

    # 4) degenerate (start==goal, no route) -> None
    assert resample_route_to_seed(None, s0, s0, H) is None, "degenerate should be None"

    # 5) K==1 route still works
    row5 = resample_route_to_seed(np.array([[0.1] * dof]), s0, g0, H)
    assert row5.shape == (H, dof) and np.allclose(row5[0], s0, atol=1e-6)

    # 6) exact horizon length for a few H
    for h in (2, 10, 28, 40):
        assert resample_route_to_seed(route, s0, g0, h).shape == (h, dof)

    print("[selftest] resample_route_to_seed (2b): ALL PASS "
          f"(shape/start-weld/goal-weld/monotone/wrap-shortpath/degenerate/K1/horizons)")

    # 2c: tensor shaping (torch). Skip cleanly if torch isn't importable in this interpreter.
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"[selftest] route_qs_to_seed_tensor (2c): SKIP (no torch: {e})")
        return
    t = route_qs_to_seed_tensor(route, s0, g0, H)
    assert tuple(t.shape) == (1, 1, H, dof), t.shape          # (k=1, batch=1, H, dof) -> injection fmt
    assert str(t.dtype) == "torch.float32" and t.device.type == "cpu" and t.is_contiguous()
    assert np.allclose(t.view(H, dof).numpy(), row, atol=0), "tensor != resampled row"
    assert route_qs_to_seed_tensor(None, s0, s0, H) is None, "degenerate should pass through as None"
    print("[selftest] route_qs_to_seed_tensor (2c): ALL PASS (shape/dtype/cpu/contiguous/None-passthrough)")

    # ---- gate-tau sweep: the scan is what turns "the gate cut it" into an actionable number,
    # so its early exit and its monotonicity assumption both need pinning.
    calls = []

    def fake(connect_at):
        """widest_path stand-in that connects once tau >= connect_at."""
        def f(tau):
            calls.append(tau)
            return (0.05 + 0.01 * tau, None, tau >= connect_at)
        return f

    ladder = (0.5, 0.7, 1.0, 1.5, 2.0)
    calls.clear()
    tried, needed = sweep_gate_tau(fake(1.0), ladder, 0.35)
    assert needed == 1.0, needed
    assert calls == [0.5, 0.7, 1.0], f"must stop at the FIRST connecting tau: {calls}"
    assert [t for t, _, _ in tried] == [0.5, 0.7, 1.0] and [m for _, m, _ in tried] == [False, False, True]

    # nothing connects -> every rung tried, needed is None (the "tau is not the problem" verdict)
    calls.clear()
    tried, needed = sweep_gate_tau(fake(99.0), ladder, 0.35)
    assert needed is None and len(tried) == len(ladder), (needed, tried)

    # only LOOSER taus are worth trying: the configured one already failed and the gate is
    # monotone, so a tighter tau cannot connect. A high gate_tau must shrink the ladder.
    calls.clear()
    tried, needed = sweep_gate_tau(fake(0.5), ladder, 1.0)
    assert calls == [1.5] and needed == 1.5, (calls, needed)   # 0.5/0.7/1.0 filtered out, then early exit
    calls.clear()
    assert sweep_gate_tau(fake(0.5), ladder, 5.0) == ([], None), "no rung above gate_tau -> no work"
    assert calls == [], calls
    # unsorted / duplicate ladders still scan ascending
    calls.clear()
    sweep_gate_tau(fake(99.0), (1.0, 0.5, 0.7), 0.35)
    assert calls == [0.5, 0.7, 1.0], calls
    print("[selftest] sweep_gate_tau: ALL PASS (early-exit/no-connect/monotone-floor/empty/unsorted)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("seed_from_clearance.py: pass --selftest to run the CPU unit test for 2b "
              "(compute_route_configs (2a) needs a live scene and is exercised in Phase 3).")
