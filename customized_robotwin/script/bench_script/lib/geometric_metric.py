"""CPU-only envelope-relaxed geometric eps*.

This module keeps the existing obstacle-clearance widest-path calculation and replaces the
per-scene IK volume plus joint-continuity gate with the validated, precomputed reach envelope.
One label/clearance volume is built per :func:`geometric_eps` call and reused for every requested
leg. The result is a relaxation of the gated metric: ``eps_geom >= eps_gated``. It can make a
scene look easier than the arm-specific gated metric, never harder.

Known and accepted gaps:

* The reach envelope says only that the endlink can occupy a point; full-arm collision against
  this scene's clutter is absent.
* Furniture and walls remain passable because they live in ``cuboid_collision_list`` and are
  deliberately excluded from the obstacle-clearance set (including the table would swamp it).
* Under-table routing is excluded by the configured z volume (normally 0.78--1.23 m).
* The target mesh blocks route nodes but does not enter the EDT: geometric eps* remains clearance
  to obstacles, not clearance to the object being grasped.

There is intentionally no torch, curobo, IK-solver, or seed-builder import here.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .ik_grid import build_grid
from .labeling import BEYOND, FREE, load_reach_envelope
from .metric_config import SeedMetricConfig
from .obstacles import (
    obstacle_centers,
    occluder_clearance_3d,
    occluder_footprints_3d,
    occluder_mask_3d,
)
from .scene_constants import OCC_HALF_FOOTPRINT
from .widest_path import (
    nearest_free_voxel,
    reconstruct_widest_path_3d,
    widest_path_eps_3d,
)


@dataclass
class LegResult:
    """Geometric eps* and route for one requested world-space leg."""

    eps_star: float
    merged: bool
    bottleneck_xyz: tuple | None
    route_world: list | None
    start_xyz: np.ndarray
    goal_xyz: np.ndarray
    n_free: int
    reason: str | None


@dataclass
class _GeometricVolume:
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    XX: np.ndarray
    YY: np.ndarray
    label: np.ndarray
    edt: np.ndarray


def _target_collision_path(env):
    """Resolve the target's actual collision mesh without adding it to the obstacle EDT."""
    target = getattr(env, "target_obj", None)
    if target is None:
        raise ValueError("geometric eps* requires env.target_obj for the mandatory target mask")

    for info in getattr(env, "collision_list", None) or []:
        if info.get("actor") is target and info.get("collision_path") and "link" not in info:
            return str(info["collision_path"])

    explicit = getattr(env, "target_collision_path", None)
    if explicit:
        return str(explicit)

    model = getattr(env, "target_model", None)
    model_id = getattr(env, "target_id", None)
    bench_root = os.environ.get("BENCH_ROOT")
    if bench_root and model is not None and model_id is not None:
        return str(Path(bench_root) / "assets" / "objects" / str(model)
                   / "collision" / f"base{model_id}.glb")

    raise ValueError(
        "cannot resolve target collision mesh: provide a target collision_list entry, "
        "env.target_collision_path, or env.target_model + env.target_id with BENCH_ROOT"
    )


def _target_mask(env, XX, YY, zs, shape):
    """Stamp the posed target collision mesh into route labels, never into clearance values."""
    target = env.target_obj
    proxy = SimpleNamespace(collision_list=[{
        "actor": target,
        "collision_path": _target_collision_path(env),
    }])
    foots = occluder_footprints_3d(proxy, obstacles="all")
    if not foots:
        raise RuntimeError("target collision mesh could not be loaded; refusing a passable target")
    mask = occluder_mask_3d(foots, XX, YY, zs, shape=shape)
    if mask is None:
        raise RuntimeError("target collision mesh produced no voxel mask")
    return mask


def _build_geometric_volume(env, arm, cfg, reach_cache_dir, reach_mode):
    """Build the endpoint-independent envelope/obstacle volume once."""
    xs, ys, zs, XX, YY = build_grid(cfg)
    prune_mask = load_reach_envelope(
        reach_cache_dir, arm, xs, ys, zs, XX, YY, mode=reach_mode)

    foots = occluder_footprints_3d(env, obstacles=cfg.obstacles)
    occ_ps = obstacle_centers(foots) or (
        [np.asarray(o.get_pose().p, dtype=float) for o in getattr(env, "occluders", [])]
    )
    edt = occluder_clearance_3d(
        foots, occ_ps, XX, YY, zs, cfg.res, cfg.zres,
        OCC_HALF_FOOTPRINT, shape=cfg.occ_shape,
    )
    occ_mask = occluder_mask_3d(foots, XX, YY, zs, shape=cfg.occ_shape)
    if occ_mask is None:
        occ_mask = edt <= 0.0
    target_mask = _target_mask(env, XX, YY, zs, cfg.occ_shape)
    label = np.where(prune_mask | occ_mask | target_mask, BEYOND, FREE).astype(np.int8)
    return _GeometricVolume(xs, ys, zs, XX, YY, label, edt)


def _world(volume, voxel):
    iz, iy, ix = voxel
    return (
        float(volume.XX[iy, ix]),
        float(volume.YY[iy, ix]),
        float(volume.zs[iz]),
    )


def geometric_eps(
    env,
    arm,
    legs,
    cfg: SeedMetricConfig = None,
    reach_cache_dir=None,
    reach_mode="occupancy",
):
    """Return one :class:`LegResult` per ``(start_xyz, goal_xyz)`` leg.

    The reach/obstacle/target volume is built exactly once. Only endpoint snapping and the
    widest-path query are repeated per leg.
    """
    cfg = cfg or SeedMetricConfig()
    if reach_cache_dir is None:
        raise ValueError("reach_cache_dir is required; geometric eps* never rebuilds the envelope")
    volume = _build_geometric_volume(env, arm, cfg, reach_cache_dir, reach_mode)
    free = volume.label == FREE
    n_free = int(free.sum())

    results = []
    for start_xyz, goal_xyz in legs:
        start = np.asarray(start_xyz, dtype=float).reshape(3)
        goal = np.asarray(goal_xyz, dtype=float).reshape(3)
        if n_free == 0:
            results.append(LegResult(
                0.0, False, None, None, start, goal, n_free,
                "no FREE voxel in the geometric volume",
            ))
            continue

        seed_start = nearest_free_voxel(
            free, volume.XX, volume.YY, volume.zs, start, cfg.seed_snap)
        seed_goal = nearest_free_voxel(
            free, volume.XX, volume.YY, volume.zs, goal, cfg.seed_snap)
        if seed_start is None or seed_goal is None:
            which = "start" if seed_start is None else "goal"
            results.append(LegResult(
                0.0, False, None, None, start, goal, n_free,
                f"{which} seed unsnappable within {cfg.seed_snap}m",
            ))
            continue

        (seed_a, _), (seed_b, _) = seed_start, seed_goal
        eps_star, bottleneck, merged = widest_path_eps_3d(
            volume.label, volume.edt, None, seed_a, seed_b, cfg.gate_tau)
        if not merged:
            results.append(LegResult(
                float(eps_star), False, None, None, start, goal, n_free,
                "no route through envelope-reachable clear voxels",
            ))
            continue

        route = reconstruct_widest_path_3d(
            free, volume.edt, None, seed_a, seed_b, eps_star, cfg.gate_tau)
        bottleneck_xyz = _world(volume, bottleneck) if bottleneck is not None else None
        if not route:
            results.append(LegResult(
                float(eps_star), True, bottleneck_xyz, None, start, goal, n_free,
                "seeds merged but route reconstruction returned empty",
            ))
            continue
        route_world = [_world(volume, voxel) for voxel in route]
        results.append(LegResult(
            float(eps_star), True, bottleneck_xyz, route_world,
            start, goal, n_free, None,
        ))
    return results
