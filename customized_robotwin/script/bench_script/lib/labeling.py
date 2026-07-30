"""Reach-envelope loading and three-way reachability labeling."""

import time
from pathlib import Path

import numpy as np

from .ik_grid import _build_ik_solver_no_world, _solve_grid, _solve_grid_q


def load_reach_envelope(cache_dir, arm, xs, ys, zs, XX, YY, mode="occupancy"):
    """Load the precomputed envelope for `arm` and return the prune mask (nz, ny, nx bool, True=skip)
    on THIS run's grid.

    mode='occupancy' (Tier 2, default): use the real reachable-workspace mask -- but it is grid-
    specific, so it is only valid when the artifact grid matches the run grid (else raise with regen
    instructions). mode='sphere' (Tier 1) or an artifact with no occupancy mask: build the max-reach
    ball from the stored shoulder centre + radius, which is grid-INDEPENDENT (works on any grid).

    Raises a clear error pointing at the producer if the artifact is missing (runs never silently
    recompute)."""
    path = Path(cache_dir) / f"reach_envelope_{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"no reach-envelope artifact for arm '{arm}' at {path}. "
            f"Precompute it ONCE with:  python reach_envelope.py --arms both   (see reach_envelope.py)")
    d = np.load(path)
    base_world = np.asarray(d["base_world"], dtype=float)
    reach_radius = float(d["reach_radius"])
    has_occ = "occupancy_prune" in getattr(d, "files", [])

    if mode == "occupancy" and has_occ:
        gmatch = (d["xs"].shape == xs.shape and d["ys"].shape == ys.shape and d["zs"].shape == zs.shape
                  and np.allclose(d["xs"], xs) and np.allclose(d["ys"], ys) and np.allclose(d["zs"], zs))
        if not gmatch:
            raise ValueError(
                f"occupancy artifact grid != run grid for arm '{arm}'. The occupancy mask is grid-"
                f"specific; regenerate it on THIS grid:\n"
                f"  python reach_envelope.py --arms {arm} --xmin {xs[0]:.4g} --xmax {xs[-1]:.4g} "
                f"--res {float(xs[1] - xs[0]):.4g} --ymin {ys[0]:.4g} --ymax {ys[-1]:.4g} "
                f"--zmin {zs[0]:.4g} --zmax {zs[-1]:.4g} --zres {float(zs[1] - zs[0]):.4g}\n"
                f"  (or pass --reach-mode sphere to use the grid-independent max-reach ball instead)")
        prune = np.asarray(d["occupancy_prune"], dtype=bool)
        print(f"[reach-env] loaded OCCUPANCY mask {path.name}: prune {100 * prune.mean():.1f}%  "
              f"(grid matches; {int(d['n_feasible']):,} feasible samples)")
        return prune

    if mode == "occupancy" and not has_occ:
        print(f"[reach-env] artifact {path.name} has no occupancy mask -> falling back to SPHERE "
              f"(regenerate with the current reach_envelope.py for the occupancy mask)")
    prune = geometric_envelope(XX, YY, zs, base_world, reach_radius)      # grid-independent max-reach ball
    print(f"[reach-env] loaded SPHERE {path.name}: prune {100 * prune.mean():.1f}%  "
          f"radius={reach_radius:.3f}m base={np.round(base_world, 3)}")
    return prune


def geometric_envelope(XX, YY, zs, base_world, reach_radius):
    """Stamp the prune mask onto a grid with NO IK and NO FK: a voxel is eliminated iff its distance
    to the arm base exceeds reach_radius (the precomputed pose-independent max reach). This is the
    entire per-run cost of the envelope -- a distance comparison. Returns bool (nz, ny, nx), True =
    eliminated (skip the IK solve, label BEYOND). Shared with the producer so both stamp identically."""
    ny, nx = XX.shape
    env = np.zeros((len(zs), ny, nx), dtype=bool)
    for iz, z in enumerate(zs):
        d = np.sqrt((XX - base_world[0]) ** 2 + (YY - base_world[1]) ** 2 + (z - base_world[2]) ** 2)
        env[iz] = d > reach_radius
    return env


BEYOND, OBSTACLE, FREE = 0, 1, 2


LABEL_NAMES = {BEYOND: "BEYOND-REACH", OBSTACLE: "OBSTACLE", FREE: "FREE"}


def label_volume(env, planner, ik_on, arm_tag, XX, YY, zs, grasp_q, chunk, num_seeds=None,
                 free_only=False, prune_mask=None):
    """Sweep the two-pass ON/OFF labelling over every z in the stack. Per slice, identical to the 2D
    label_grid: reach_on = reachable AND collision-free -> FREE; reach_off = reachable ignoring world
    collision; OBSTACLE = reach_off & ~reach_on; BEYOND = ~reach_off. The ON sweep also keeps the
    joint config q per voxel (the gate's fuel). The empty-world OFF solver is built ONCE and reused
    across slices.

    With free_only=True the collision-OFF sweep is SKIPPED entirely (Tier-4 speedup): non-FREE cells
    are all labelled BEYOND (no OBSTACLE/BEYOND split, which is viz-only), halving the IK per slice.
    The OFF-pass time is measured either way so the run can report the projected --free-only cost.

    REACH ENVELOPE (Tier-1 speedup): if prune_mask (nz, ny, nx bool) is given -- the PRECOMPUTED
    pose-independent envelope, stamped by geometric_envelope from the artifact reach_envelope.py
    produced -- every True voxel is skipped (forced BEYOND, no IK solve); only the un-masked cells
    are solved. This is exact: a FREE cell is reachable at the real grasp pose, hence inside the
    max-reach sphere, so it is never masked. label_volume does NO envelope computation itself; it
    only consumes the mask.

    Returns (label int8 (nz, ny, nx), qfield float32 (nz, ny, nx, dof), off_seconds float)."""
    ny, nx = XX.shape
    nz = len(zs)
    ik_off = None if free_only else _build_ik_solver_no_world(planner, num_seeds)   # empty world = kinematics + self-collision
    label = np.empty((nz, ny, nx), dtype=np.int8)
    qfield = None
    off_seconds = 0.0
    use_env = prune_mask is not None
    xr, yr = XX.ravel(), YY.ravel()
    for iz, z in enumerate(zs):
        t_s = time.perf_counter()
        near = (~prune_mask[iz].ravel()) if use_env else np.ones(XX.size, dtype=bool)
        idx = np.flatnonzero(near)                            # only these cells get an IK solve

        reach_on_flat = np.zeros(XX.size, dtype=bool)
        reach_off_flat = np.zeros(XX.size, dtype=bool)
        gp = None
        if idx.size:
            gp = np.zeros((idx.size, 7))
            gp[:, 0] = xr[idx]; gp[:, 1] = yr[idx]; gp[:, 2] = z; gp[:, 3:] = grasp_q
            ron, qk = _solve_grid_q(env.robot, planner, ik_on, arm_tag, gp, chunk=chunk, num_seeds=num_seeds)
            reach_on_flat[idx] = ron
            if qfield is None:
                qfield = np.full((nz, ny, nx, qk.shape[-1]), np.nan, dtype=np.float32)
            qslab = np.full((XX.size, qk.shape[-1]), np.nan, dtype=np.float32)
            qslab[idx] = qk                                   # scatter solved q back; far cells stay NaN
            qfield[iz] = qslab.reshape(ny, nx, qk.shape[-1])
        reach_on = reach_on_flat.reshape(ny, nx)

        anomaly = 0
        if free_only:
            label[iz] = np.where(reach_on, FREE, BEYOND)      # OBSTACLE/BEYOND split (OFF pass) skipped
        else:
            if idx.size:
                t_off = time.perf_counter()
                reach_off_flat[idx] = _solve_grid(env.robot, planner, ik_off, arm_tag, gp, chunk=chunk)
                off_seconds += time.perf_counter() - t_off
            reach_off = reach_off_flat.reshape(ny, nx)         # far/pruned cells -> False -> BEYOND
            label[iz] = np.where(reach_on, FREE, np.where(reach_off & ~reach_on, OBSTACLE, BEYOND))
            anomaly = int((reach_on & ~reach_off).sum())      # IK seed noise, not a logic error
        pruned = int((~near).sum()) if use_env else 0
        print(f"[label] slice {iz + 1}/{nz} z={z:.3f}  FREE={int(reach_on.sum())} (of {XX.size})"
              + (f"  pruned={pruned} (reach-env, no solve)" if use_env else "")
              + (f"  WARN {anomaly} ON&~OFF kept FREE" if anomaly else "")
              + f"  {time.perf_counter() - t_s:.2f}s")
    if ik_off is not None:
        import torch
        del ik_off
        torch.cuda.empty_cache()
    if qfield is None:                                        # every voxel pruned -> radius too small / grid off
        raise RuntimeError("reach envelope masked the ENTIRE grid; check the artifact radius / base pose / "
                           "grid bounds (no cell fell inside the max-reach sphere)")
    return label, qfield, off_seconds
