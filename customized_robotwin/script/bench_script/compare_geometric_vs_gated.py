#!/usr/bin/env python3
"""Compare envelope-only geometric eps* with the gated IK metric on identical scenes.

This is Stage 3 of plans/GEOMETRIC_EPS_VALIDATION_PLAN.md.  Each seed builds one occluder scene, then
measures the same grasp-to-pad leg with:

* the gated graph (scene IK volume plus the joint-continuity gate), and
* the CPU-only geometric graph (reach envelope, no scene IK or joint gate).

Both no-target and target-masked graph pairs use exact common snapped voxels and identical EDTs.
The runner fails immediately unless geometric FREE is a superset and eps_geom >= eps_gated.  It
writes both rank series, native independently-snapped diagnostics, a clearance-preferred geometric
route, Stage-1 false-keep overlap, route diagnostics, overlays, and timings.  A person must still
inspect the route overlays before the Stage-3 gate can pass.
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import spearmanr

from setup_paths import setup_paths

setup_paths()

from lib.geometric_metric import _build_geometric_volume, _label_for
from lib.ik_grid import _build_ik_solver, build_grid, grasp_orientation
from lib.labeling import BEYOND, FREE, LABEL_NAMES, OBSTACLE, load_reach_envelope
from lib.metric_config import SeedMetricConfig
from lib.obstacles import occluder_footprints_3d, occluder_slice_polys
from lib.occluder_ring import draw_ring_config, parse_count_choices, parse_offset_specs
from lib.plotting import _line_axis, _scene_anchor_markers
from lib.run_io import CLEARANCE_RESULTS_DIR, Timings
from lib.scene_build import DR_CLEAN, build_cfg
from lib.scene_constants import PAD_XY
from lib.widest_path import (
    nearest_free_voxel,
    reconstruct_clearance_preferred_path_3d,
    reconstruct_widest_path_3d,
    widest_path_eps_3d,
)
from metric_viz import LABEL_COLORS
from seed_from_clearance import compute_route_configs
from task.occluder_task import make_occluder_task


RESULTS_DIR = CLEARANCE_RESULTS_DIR.parent / "geometric_vs_gated"
FALSE_KEEP_RESULTS_DIR = CLEARANCE_RESULTS_DIR.parent / "reach_envelope_validation"
STAGE1_REACH_CACHE_DIR = CLEARANCE_RESULTS_DIR / "_reach_cache_geometric_stage1"


class AlignmentError(ValueError):
    """The two graphs do not satisfy the relaxation theorem's preconditions."""


class RelaxationInvariantError(AssertionError):
    """An aligned geometric graph produced eps_geom < eps_gated."""


class SceneNotAlignableError(RuntimeError):
    """This scene has no endpoint pair on which the aligned comparison can run."""


def _json_eps(value, merged):
    if not merged or np.isinf(value):
        return None
    return float(value)


def _fmt_eps(value, merged):
    if not merged:
        return "INACCESSIBLE"
    return "inf" if np.isinf(value) else f"{value:.3f} m"


def _rank_value(record, prefix):
    return float(record[f"eps_{prefix}"]) if record[f"merged_{prefix}"] else 0.0


def _json_safe(value):
    """Convert numpy values and non-finite floats to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _summarize_series(rows, gated_prefix, geom_prefix):
    x = np.asarray([_rank_value(r, gated_prefix) for r in rows], dtype=float)
    y = np.asarray([_rank_value(r, geom_prefix) for r in rows], dtype=float)

    rho_all = p_all = None
    if len(rows) >= 2 and np.unique(x).size > 1 and np.unique(y).size > 1:
        result = spearmanr(x, y)
        if np.isfinite(result.statistic):
            rho_all, p_all = float(result.statistic), float(result.pvalue)

    finite_rows = [
        r for r in rows
        if r[f"merged_{gated_prefix}"] and r[f"merged_{geom_prefix}"]
        and np.isfinite(r[f"eps_{gated_prefix}"]) and np.isfinite(r[f"eps_{geom_prefix}"])
    ]
    rho_finite = p_finite = None
    if len(finite_rows) >= 2:
        xf = np.asarray([r[f"eps_{gated_prefix}"] for r in finite_rows], dtype=float)
        yf = np.asarray([r[f"eps_{geom_prefix}"] for r in finite_rows], dtype=float)
        if np.unique(xf).size > 1 and np.unique(yf).size > 1:
            result = spearmanr(xf, yf)
            if np.isfinite(result.statistic):
                rho_finite, p_finite = float(result.statistic), float(result.pvalue)

    disagreements = [
        int(r["seed"]) for r in rows
        if not r[f"merged_{gated_prefix}"] and r[f"merged_{geom_prefix}"]
    ]
    reverse_disagreements = [
        int(r["seed"]) for r in rows
        if r[f"merged_{gated_prefix}"] and not r[f"merged_{geom_prefix}"]
    ]
    sufficient = len(rows) >= 8
    rank_pass = bool(sufficient and rho_all is not None and rho_all >= 0.8)

    return {
        "sufficient_for_rank_gate": sufficient,
        "spearman_all": rho_all,
        "spearman_all_pvalue": p_all,
        "spearman_all_definition": (
            "all completed scenes; INACCESSIBLE is ranked as 0 and unbounded eps* as +inf"
        ),
        "finite_jointly_accessible_scenes": len(finite_rows),
        "spearman_finite_jointly_accessible": rho_finite,
        "spearman_finite_pvalue": p_finite,
        "unbounded_gated_scenes": sum(
            bool(r[f"merged_{gated_prefix}"] and np.isinf(r[f"eps_{gated_prefix}"]))
            for r in rows
        ),
        "unbounded_geom_scenes": sum(
            bool(r[f"merged_{geom_prefix}"] and np.isinf(r[f"eps_{geom_prefix}"]))
            for r in rows
        ),
        "gated_inaccessible_geom_merged_seeds": disagreements,
        "gated_merged_geom_inaccessible_seeds": reverse_disagreements,
        "rank_gate_pass": rank_pass,
    }


def summarize_records(records):
    """Return both aligned Stage-3 rank gates; target-masked is the production series."""
    rows = [r for r in records if r.get("status") == "ok"]
    not_alignable = [r for r in records if r.get("status") == "not_alignable"]
    errors = [r for r in records if r.get("status") == "error"]
    production = _summarize_series(rows, "gated", "geom")
    isolation = _summarize_series(
        rows, "gated_notarget_common", "geom_notarget_common"
    )
    relaxation_violations = [
        int(r["seed"]) for r in rows if r.get("relaxation_violation")
    ]
    geom_route_voxels = sum(int(r["geom_route_voxels"]) for r in rows)
    false_keep_voxels = sum(int(r["false_keep_route_voxels"]) for r in rows)
    rank_pass = bool(
        production["rank_gate_pass"]
        and isolation["rank_gate_pass"]
        and not relaxation_violations
    )
    return {
        "requested_scenes": len(records),
        "completed_scenes": len(rows),
        "failed_scenes": len(records) - len(rows),
        "not_alignable_scenes": len(not_alignable),
        "not_alignable_seeds": [int(r["seed"]) for r in not_alignable],
        "error_scenes": len(errors),
        "error_seeds": [int(r["seed"]) for r in errors],
        "production_target_masked": production,
        "isolation_no_target": isolation,
        # Production aliases keep old report readers working.
        **production,
        "relaxation_violation_seeds": relaxation_violations,
        "geometric_route_voxels": geom_route_voxels,
        "false_keep_route_voxels": false_keep_voxels,
        "false_keep_route_fraction": (
            false_keep_voxels / geom_route_voxels if geom_route_voxels else None
        ),
        "rank_threshold": 0.8,
        "rank_gate_pass": rank_pass,
        "route_overlay_gate": "PENDING_USER_INSPECTION",
        "stage3_verdict": (
            "PENDING_ROUTE_INSPECTION" if rank_pass else "FAIL_OR_INSUFFICIENT_RANK_AGREEMENT"
        ),
    }


def _write_scatter(out_dir, records, summary):
    rows = [r for r in records if r.get("status") == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    series = (
        ("gated", "geom", summary["production_target_masked"], "Target-masked production"),
        (
            "gated_notarget_common",
            "geom_notarget_common",
            summary["isolation_no_target"],
            "No-target isolation",
        ),
    )
    for ax, (gated_prefix, geom_prefix, stats, title) in zip(axes, series):
        finite_xy = [
            (_rank_value(r, gated_prefix), _rank_value(r, geom_prefix), int(r["seed"]))
            for r in rows
            if np.isfinite(_rank_value(r, gated_prefix))
            and np.isfinite(_rank_value(r, geom_prefix))
        ]
        if finite_xy:
            x, y, seeds = map(np.asarray, zip(*finite_xy))
            ax.scatter(x, y, s=65, color="#1565c0")
            for xv, yv, seed in zip(x, y, seeds):
                ax.annotate(
                    str(int(seed)), (xv, yv), xytext=(4, 4),
                    textcoords="offset points", fontsize=9,
                )
            lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
            pad = max(0.005, 0.05 * (hi - lo if hi > lo else 1.0))
            ax.plot(
                [lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", lw=1.3, label="identity",
            )
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.legend(loc="lower right")
        rho = stats["spearman_all"]
        rho_text = "undefined" if rho is None else f"{rho:.3f}"
        ax.text(
            0.03, 0.97,
            f"Spearman = {rho_text}\n"
            f"completed n = {summary['completed_scenes']}\n"
            f"unbounded gated/geom = {stats['unbounded_gated_scenes']}/"
            f"{stats['unbounded_geom_scenes']}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
        ax.set_xlabel("gated eps* (m; inaccessible = 0)")
        ax.set_ylabel("geometric eps* (m; inaccessible = 0)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "eps_geom_vs_gated_scatter.png", dpi=140)
    plt.close(fig)


def write_reports(out_dir, records, config):
    """Write the aggregate JSON and scatter. Records remain the source-of-truth rows."""
    out_dir = Path(out_dir)
    summary = summarize_records(records)
    summary["config"] = config
    summary["scenes"] = [_json_safe(record) for record in records]
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False)
    )
    _write_scatter(out_dir, records, summary)
    return summary


def _load_false_keep_mask(path):
    with np.load(path) as data:
        required = {"false_keep_mask", "xs", "ys", "zs", "arm", "reach_mode"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}")
        return {
            "path": str(path),
            "mask": np.asarray(data["false_keep_mask"], dtype=bool),
            "xs": np.asarray(data["xs"], dtype=float),
            "ys": np.asarray(data["ys"], dtype=float),
            "zs": np.asarray(data["zs"], dtype=float),
            "arm": str(data["arm"].item()),
            "reach_mode": str(data["reach_mode"].item()),
        }


def resolve_false_keep_mask(path, root, arm, reach_mode, cfg):
    """Load an explicit Stage-1 mask or the newest matching validation artifact."""
    if path:
        candidates = [Path(path)]
    else:
        candidates = sorted(Path(root).glob("*/false_keep_mask.npz"), reverse=True)

    chosen = None
    for candidate in candidates:
        data = _load_false_keep_mask(candidate)
        if data["arm"] == arm and data["reach_mode"] == reach_mode:
            chosen = data
            break
    if chosen is None:
        raise FileNotFoundError(
            f"no Stage-1 false_keep_mask.npz found for arm={arm}, reach_mode={reach_mode}"
        )

    expected = {
        "xs": np.arange(cfg.xmin, cfg.xmax + 1e-9, cfg.res),
        "ys": np.arange(cfg.ymin, cfg.ymax + 1e-9, cfg.res),
        "zs": np.arange(cfg.zmin, cfg.zmax + 1e-9, cfg.zres),
    }
    for key, values in expected.items():
        if chosen[key].shape != values.shape or not np.allclose(chosen[key], values):
            raise ValueError(
                f"Stage-1 {key} grid in {chosen['path']} does not match this comparison config"
            )
    if chosen["mask"].shape != (
        len(chosen["zs"]), len(chosen["ys"]), len(chosen["xs"])
    ):
        raise ValueError(f"invalid false-keep mask shape in {chosen['path']}")
    return chosen


def false_keep_overlap(route_world, false_keep):
    """Count geometric-route voxels marked false-keep by the Stage-1 fixed-orientation sweep."""
    if not route_world:
        return 0, 0, None
    hits = 0
    for point in np.asarray(route_world, dtype=float):
        ix = int(np.argmin(np.abs(false_keep["xs"] - point[0])))
        iy = int(np.argmin(np.abs(false_keep["ys"] - point[1])))
        iz = int(np.argmin(np.abs(false_keep["zs"] - point[2])))
        hits += bool(false_keep["mask"][iz, iy, ix])
    total = len(route_world)
    return hits, total, hits / total


def _voxel_world(XX, YY, zs, voxel):
    iz, iy, ix = voxel
    return (float(XX[iy, ix]), float(YY[iy, ix]), float(zs[iz]))


def _snap_common_pair(label, XX, YY, zs, start_xyz, goal_xyz, max_dist):
    free = label == FREE
    start = nearest_free_voxel(free, XX, YY, zs, start_xyz, max_dist)
    goal = nearest_free_voxel(free, XX, YY, zs, goal_xyz, max_dist)
    if start is None or goal is None:
        which = "start" if start is None else "goal"
        raise SceneNotAlignableError(
            f"{which} common endpoint is unsnappable within {max_dist}m"
        )
    return tuple(start[0]), tuple(goal[0])


def _subset_diagnostics(gated_label, geom_label, volume):
    violation = (gated_label == FREE) & (geom_label != FREE)
    attributed = np.zeros_like(violation)
    result = {"total": int(violation.sum())}
    for name in ("prune_mask", "occ_mask", "target_mask"):
        mask = getattr(volume, name, None)
        hits = violation & mask if mask is not None else np.zeros_like(violation)
        attributed |= hits
        result[name] = int(hits.sum())
    result["unattributed"] = int((violation & ~attributed).sum())
    return result


def aligned_widest_pair(
    gated_label, geom_label, gated_edt, geom_edt, qvol, seed_a, seed_b, tau
):
    """Solve one matched-mask graph pair and enforce the relaxation theorem."""
    if gated_label.shape != geom_label.shape or gated_edt.shape != geom_edt.shape:
        raise AlignmentError("aligned comparison grid shapes differ")
    if not np.array_equal(gated_edt, geom_edt):
        finite = np.isfinite(gated_edt) & np.isfinite(geom_edt)
        max_diff = (
            float(np.max(np.abs(gated_edt[finite] - geom_edt[finite])))
            if finite.any() else None
        )
        raise AlignmentError(
            f"aligned comparison EDTs differ (max finite abs diff={max_diff})"
        )

    gated_free = gated_label == FREE
    geom_free = geom_label == FREE
    subset_violations = int((gated_free & ~geom_free).sum())
    if subset_violations:
        raise AlignmentError(
            f"geometric FREE is not a superset of gated FREE ({subset_violations} voxels)"
        )
    seed_a, seed_b = tuple(seed_a), tuple(seed_b)
    if not (gated_free[seed_a] and gated_free[seed_b]
            and geom_free[seed_a] and geom_free[seed_b]):
        raise AlignmentError("common endpoints are not FREE on both aligned graphs")

    eps_g, bott_g, merged_g = widest_path_eps_3d(
        gated_label, gated_edt, qvol, seed_a, seed_b, tau
    )
    eps_m, bott_m, merged_m = widest_path_eps_3d(
        geom_label, geom_edt, None, seed_a, seed_b, tau
    )
    reversal = bool(
        (merged_g and not merged_m)
        or (
            merged_g and merged_m
            and (
                (np.isinf(eps_g) and not np.isinf(eps_m))
                or (
                    np.isfinite(eps_g) and np.isfinite(eps_m)
                    and eps_m < eps_g - 1e-12
                )
            )
        )
    )
    if reversal:
        raise RelaxationInvariantError(
            "aligned relaxation invariant failed: "
            f"gated={_fmt_eps(eps_g, merged_g)}, geom={_fmt_eps(eps_m, merged_m)}"
        )
    return {
        "seed_start": seed_a,
        "seed_goal": seed_b,
        "gated": {"eps": float(eps_g), "merged": bool(merged_g), "bottleneck": bott_g},
        "geom": {"eps": float(eps_m), "merged": bool(merged_m), "bottleneck": bott_m},
    }


def _route_world(route, XX, YY, zs):
    return [_voxel_world(XX, YY, zs, voxel) for voxel in route] if route else None


def route_diagnostics(route_voxels, route_world, edt, zmin):
    if not route_voxels or not route_world:
        return {
            "voxels": 0,
            "z_min": None,
            "z_max": None,
            "zmin_fraction": None,
            "physical_length_m": None,
            "clearance_m": [],
        }
    points = np.asarray(route_world, dtype=float)
    clearance = [float(edt[voxel]) for voxel in route_voxels]
    return {
        "voxels": len(route_voxels),
        "z_min": float(points[:, 2].min()),
        "z_max": float(points[:, 2].max()),
        "zmin_fraction": float(np.isclose(points[:, 2], zmin).mean()),
        "physical_length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
        "clearance_m": clearance,
    }


def _solve_single(label, edt, qvol, seed_a, seed_b, tau):
    eps, bottleneck, merged = widest_path_eps_3d(
        label, edt, qvol, seed_a, seed_b, tau
    )
    return {"eps": float(eps), "merged": bool(merged), "bottleneck": bottleneck}


def _mask_delta(no_target, target):
    if not no_target["merged"] or not target["merged"]:
        return None
    if np.isinf(no_target["eps"]) and np.isinf(target["eps"]):
        return 0.0
    if np.isfinite(no_target["eps"]) and np.isfinite(target["eps"]):
        return float(no_target["eps"] - target["eps"])
    return None


def build_aligned_comparison(gated, volume, start_xyz, goal_xyz, cfg):
    """Build no-target and target-masked pairs on exact common voxel endpoints."""
    for name, gated_grid, geom_grid in (
        ("XX", gated.XX, volume.XX),
        ("YY", gated.YY, volume.YY),
        ("zs", gated.zs, volume.zs),
    ):
        if not np.array_equal(gated_grid, geom_grid):
            raise AlignmentError(f"aligned comparison {name} grids differ")
    missing = [
        name for name in ("edt", "q_warm_3d") if getattr(gated, name) is None
    ]
    if missing:
        if gated.reason:
            raise SceneNotAlignableError(
                f"gated metric ended before aligned comparison ({gated.reason}); "
                f"missing {', '.join(missing)}"
            )
        raise AlignmentError(
            "gated result unexpectedly lacks fields required for aligned comparison: "
            + ", ".join(missing)
        )

    gated_notarget = np.asarray(gated.label, dtype=np.int8)
    geom_notarget = _label_for(volume, mask_target=False)
    if volume.target_mask is None:
        raise AlignmentError("aligned production comparison requires target_mask")
    gated_target = np.where(volume.target_mask, BEYOND, gated_notarget).astype(np.int8)
    geom_target = _label_for(volume, mask_target=True)

    seeds_notarget = _snap_common_pair(
        gated_notarget, gated.XX, gated.YY, gated.zs,
        start_xyz, goal_xyz, cfg.seed_snap,
    )
    seeds_target = _snap_common_pair(
        gated_target, gated.XX, gated.YY, gated.zs,
        start_xyz, goal_xyz, cfg.seed_snap,
    )
    subset_notarget = _subset_diagnostics(gated_notarget, geom_notarget, volume)
    subset_target = _subset_diagnostics(gated_target, geom_target, volume)
    if subset_notarget["total"]:
        raise AlignmentError(
            f"no-target FREE subset precondition failed: {subset_notarget}"
        )
    if subset_target["total"]:
        raise AlignmentError(
            f"target-masked FREE subset precondition failed: {subset_target}"
        )
    notarget = aligned_widest_pair(
        gated_notarget, geom_notarget, gated.edt, volume.edt,
        gated.q_warm_3d, *seeds_notarget, cfg.gate_tau,
    )
    target = aligned_widest_pair(
        gated_target, geom_target, gated.edt, volume.edt,
        gated.q_warm_3d, *seeds_target, cfg.gate_tau,
    )

    sa, sb = seeds_target
    if target["gated"]["merged"]:
        target["gated"]["route_voxels"] = reconstruct_widest_path_3d(
            gated_target == FREE, gated.edt, gated.q_warm_3d,
            sa, sb, target["gated"]["eps"], cfg.gate_tau,
        )
    else:
        target["gated"]["route_voxels"] = None
    target["gated"]["route_world"] = _route_world(
        target["gated"]["route_voxels"], gated.XX, gated.YY, gated.zs
    )

    if target["geom"]["merged"]:
        target["geom"]["route_bfs_voxels"] = reconstruct_widest_path_3d(
            geom_target == FREE, volume.edt, None,
            sa, sb, target["geom"]["eps"], cfg.gate_tau,
        )
        target["geom"]["route_voxels"] = reconstruct_clearance_preferred_path_3d(
            geom_target == FREE, volume.edt, sa, sb, target["geom"]["eps"],
            cfg.res, cfg.zres,
        )
    else:
        target["geom"]["route_bfs_voxels"] = None
        target["geom"]["route_voxels"] = None
    target["geom"]["route_bfs_world"] = _route_world(
        target["geom"]["route_bfs_voxels"], volume.XX, volume.YY, volume.zs
    )
    target["geom"]["route_world"] = _route_world(
        target["geom"]["route_voxels"], volume.XX, volume.YY, volume.zs
    )

    native_seeds = _snap_common_pair(
        geom_target, volume.XX, volume.YY, volume.zs,
        start_xyz, goal_xyz, cfg.seed_snap,
    )
    native_geom = _solve_single(
        geom_target, volume.edt, None, *native_seeds, cfg.gate_tau
    )
    native_geom["seed_start"], native_geom["seed_goal"] = native_seeds

    gated_notarget_at_target_seeds = _solve_single(
        gated_notarget, gated.edt, gated.q_warm_3d, sa, sb, cfg.gate_tau
    )
    geom_notarget_at_target_seeds = _solve_single(
        geom_notarget, volume.edt, None, sa, sb, cfg.gate_tau
    )
    target["gated"]["target_mask_delta"] = _mask_delta(
        gated_notarget_at_target_seeds, target["gated"]
    )
    target["geom"]["target_mask_delta"] = _mask_delta(
        geom_notarget_at_target_seeds, target["geom"]
    )

    target["gated"]["route_diagnostics"] = route_diagnostics(
        target["gated"]["route_voxels"], target["gated"]["route_world"],
        gated.edt, float(gated.zs[0]),
    )
    target["geom"]["route_bfs_diagnostics"] = route_diagnostics(
        target["geom"]["route_bfs_voxels"], target["geom"]["route_bfs_world"],
        volume.edt, float(volume.zs[0]),
    )
    target["geom"]["route_diagnostics"] = route_diagnostics(
        target["geom"]["route_voxels"], target["geom"]["route_world"],
        volume.edt, float(volume.zs[0]),
    )

    return {
        "notarget": notarget,
        "target": target,
        "native_geom": native_geom,
        "subset_notarget": subset_notarget,
        "subset_target": subset_target,
        "subset_native_policy": _subset_diagnostics(
            gated_notarget, geom_target, volume
        ),
        "grid_equal": True,
        "edt_equal": True,
        "edt_max_abs_diff": 0.0,
        "gated_target_label": gated_target,
    }


def _draw_side_obstacles(ax, foots, p0, u, occ_shape):
    for foot in foots or []:
        if foot["poly"] is None:
            continue
        silhouette = []
        if occ_shape == "mesh" and foot.get("mesh") is not None:
            for z in np.linspace(foot["zlo"] + 1e-4, foot["zhi"] - 1e-4, 60):
                projections = [
                    (loop - p0) @ u for loop in occluder_slice_polys(foot, float(z))
                ]
                if projections:
                    values = np.concatenate(projections)
                    silhouette.append((float(values.min()), float(values.max()), float(z)))
        if silhouette:
            lo, hi, zvals = zip(*silhouette)
            ax.plot(lo, zvals, "-", color="red", lw=2)
            ax.plot(hi, zvals, "-", color="red", lw=2)
            ax.plot([lo[0], hi[0]], [zvals[0]] * 2, "-", color="red", lw=2)
            ax.plot([lo[-1], hi[-1]], [zvals[-1]] * 2, "-", color="red", lw=2)
        else:
            projection = (foot["poly"] - p0) @ u
            ax.add_patch(plt.Rectangle(
                (float(projection.min()), foot["zlo"]),
                float(projection.max() - projection.min()),
                foot["zhi"] - foot["zlo"],
                fill=False, edgecolor="red", lw=2,
            ))


def save_route_overlay(out_dir, args, gated, geom, foots, target_xyz):
    """Overlay gated and geometric routes in the existing side/top-down visual vocabulary."""
    start = np.asarray(geom.start_xyz, dtype=float)
    goal = np.asarray(geom.goal_xyz, dtype=float)
    p0, u, length = _line_axis(start[:2], goal[:2])
    xs, ys = gated.XX[0], gated.YY[:, 0]
    ns = max(2, int(round(length / args.res)) + 1)
    svals = np.linspace(0.0, length, ns)
    side_label = np.full((len(gated.zs), ns), BEYOND, dtype=np.int8)
    for j, distance in enumerate(svals):
        px, py = p0 + distance * u
        ix = int(np.argmin(np.abs(xs - px)))
        iy = int(np.argmin(np.abs(ys - py)))
        side_label[:, j] = gated.label[:, iy, ix]

    fig, (side, top) = plt.subplots(1, 2, figsize=(16, 7))
    cmap = ListedColormap([LABEL_COLORS[BEYOND], LABEL_COLORS[OBSTACLE], LABEL_COLORS[FREE]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    side.imshow(
        side_label, origin="lower",
        extent=[0, length, gated.zs.min(), gated.zs.max()],
        cmap=cmap, norm=norm, aspect="auto",
    )
    _draw_side_obstacles(side, foots, p0, u, args.occ_shape)

    route_styles = (
        (gated.route_world, "gated route", "#ffb300", "-", 3.0),
        (geom.route_world, "geometric route", "#039be5", "--", 2.5),
    )
    for route, label, color, style, width in route_styles:
        if route and len(route) > 1:
            projected = [float((np.asarray(point[:2]) - p0) @ u) for point in route]
            side.plot(projected, [point[2] for point in route], style,
                      color=color, lw=width, label=label)
    side.plot(0, start[2], "o", color="cyan", ms=11, mec="k", label="grasp")
    side.plot(length, goal[2], "s", color="magenta", ms=10, mec="k", label="pad")
    target_s = float((np.asarray(target_xyz[:2]) - p0) @ u)
    side.plot(target_s, target_xyz[2], "*", color="blue", ms=17, mec="k", mew=0.6,
              label="target")
    side.set_xlabel("arc distance grasp->pad (m)")
    side.set_ylabel("z (m)")
    side.set_title(
        f"Side elevation\n"
        f"gated={_fmt_eps(gated.eps_gated, gated.merged)}  |  "
        f"geometric={_fmt_eps(geom.eps_star, geom.merged)}"
    )
    from matplotlib.patches import Patch, Polygon as MplPolygon
    proxies = [Patch(color=LABEL_COLORS[c], label=LABEL_NAMES[c])
               for c in (FREE, OBSTACLE, BEYOND)]
    handles, _ = side.get_legend_handles_labels()
    side.legend(handles=proxies + handles, loc="upper right", fontsize=8)

    any_free = (gated.label == FREE).any(axis=0)
    top.imshow(
        np.where(any_free, 1.0, np.nan), origin="lower",
        extent=[gated.XX.min(), gated.XX.max(), gated.YY.min(), gated.YY.max()],
        cmap="Greens", vmin=0, vmax=1.5, aspect="equal",
    )
    for foot in foots or []:
        if foot["poly"] is not None:
            top.add_patch(MplPolygon(
                foot["poly"], closed=True, fill=False, edgecolor="red", lw=2
            ))
    for route, label, color, style, width in route_styles:
        if route and len(route) > 1:
            points = np.asarray(route, dtype=float)
            top.plot(points[:, 0], points[:, 1], style, color=color, lw=width, label=label)
    top.plot(start[0], start[1], "o", color="cyan", ms=11, mec="k", label="grasp")
    top.plot(goal[0], goal[1], "s", color="magenta", ms=10, mec="k", label="pad")
    _scene_anchor_markers(top, target_xyz, None, args.arm)
    top.set_xlabel("x (m)")
    top.set_ylabel("y (m)")
    top.set_title("Top-down route overlay\ngreen = gated FREE at some height")
    top.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Seed {args.seed}, arm {args.arm}: inspect the geometric route for far-field detours")
    fig.tight_layout()
    path = Path(out_dir) / f"route_overlay_seed{args.seed}_{args.arm}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[viz] wrote {path.name}")


def _inflation(gated, geom):
    if np.isinf(geom["eps"]) and np.isinf(gated["eps"]):
        return 0.0, False
    if np.isinf(geom["eps"]):
        return None, True
    if np.isinf(gated["eps"]):
        return None, False
    return float(geom["eps"] - gated["eps"]), False


def _scene_record(
    seed, arm, gated_native, aligned, overlap, start_xyz, goal_xyz, elapsed,
    ring_config=None,
):
    hits, route_voxels, fraction = overlap
    production = aligned["target"]
    isolation = aligned["notarget"]
    gated = production["gated"]
    geom = production["geom"]
    native_geom = aligned["native_geom"]
    inflation, inflation_unbounded = _inflation(gated, geom)
    sa_t, sb_t = production["seed_start"], production["seed_goal"]
    sa_n, sb_n = isolation["seed_start"], isolation["seed_goal"]
    native_reversal = bool(
        gated_native.merged
        and (
            not native_geom["merged"]
            or (
                np.isinf(gated_native.eps_gated)
                and not np.isinf(native_geom["eps"])
            )
            or (
                np.isfinite(gated_native.eps_gated)
                and np.isfinite(native_geom["eps"])
                and native_geom["eps"] < gated_native.eps_gated - 1e-12
            )
        )
    )
    return {
        "status": "ok",
        "seed": int(seed),
        "arm": arm,
        "occluder_ring": ring_config,
        # Primary production series: target-masked, exact common endpoints.
        "merged_gated": gated["merged"],
        "merged_geom": geom["merged"],
        "eps_gated": gated["eps"],
        "eps_geom": geom["eps"],
        "eps_gated_m": _json_eps(gated["eps"], gated["merged"]),
        "eps_geom_m": _json_eps(geom["eps"], geom["merged"]),
        "eps_gated_unbounded": bool(gated["merged"] and np.isinf(gated["eps"])),
        "eps_geom_unbounded": bool(geom["merged"] and np.isinf(geom["eps"])),
        "inflation_m": inflation,
        "inflation_unbounded": inflation_unbounded,
        "relaxation_violation": False,
        # Isolation series: no target on either graph, exact common endpoints.
        "merged_gated_notarget_common": isolation["gated"]["merged"],
        "merged_geom_notarget_common": isolation["geom"]["merged"],
        "eps_gated_notarget_common": isolation["gated"]["eps"],
        "eps_geom_notarget_common": isolation["geom"]["eps"],
        # Native independently snapped values are diagnostic only.
        "merged_gated_native": bool(gated_native.merged),
        "merged_geom_native": native_geom["merged"],
        "eps_gated_native": float(gated_native.eps_gated),
        "eps_geom_native": native_geom["eps"],
        "native_relaxation_reversal": native_reversal,
        "gated_reason": gated_native.reason,
        "geom_reason": None,
        "gated_route_voxels": len(gated["route_world"] or []),
        "geom_route_voxels": route_voxels,
        "false_keep_route_voxels": hits,
        "false_keep_route_fraction": fraction,
        "start_xyz": [float(v) for v in start_xyz],
        "goal_xyz": [float(v) for v in goal_xyz],
        "common_endpoints": {
            "target": {
                "start_voxel": sa_t,
                "goal_voxel": sb_t,
                "start_world": _voxel_world(
                    gated_native.XX, gated_native.YY, gated_native.zs, sa_t
                ),
                "goal_world": _voxel_world(
                    gated_native.XX, gated_native.YY, gated_native.zs, sb_t
                ),
                "start_edt_m": float(gated_native.edt[sa_t]),
                "goal_edt_m": float(gated_native.edt[sb_t]),
            },
            "notarget": {
                "start_voxel": sa_n,
                "goal_voxel": sb_n,
                "start_world": _voxel_world(
                    gated_native.XX, gated_native.YY, gated_native.zs, sa_n
                ),
                "goal_world": _voxel_world(
                    gated_native.XX, gated_native.YY, gated_native.zs, sb_n
                ),
                "start_edt_m": float(gated_native.edt[sa_n]),
                "goal_edt_m": float(gated_native.edt[sb_n]),
            },
        },
        "native_endpoints": {
            "gated_start_voxel": gated_native.seed_start,
            "gated_goal_voxel": gated_native.seed_goal,
            "geom_start_voxel": native_geom["seed_start"],
            "geom_goal_voxel": native_geom["seed_goal"],
        },
        "target_mask_delta_m": {
            "gated": gated["target_mask_delta"],
            "geom": geom["target_mask_delta"],
        },
        "subset_diagnostics": {
            "notarget": aligned["subset_notarget"],
            "target": aligned["subset_target"],
            "native_mismatched_policy": aligned["subset_native_policy"],
        },
        "alignment": {
            "grid_equal": aligned["grid_equal"],
            "edt_equal": aligned["edt_equal"],
            "edt_max_abs_diff": aligned["edt_max_abs_diff"],
        },
        "route_diagnostics": {
            "gated": gated["route_diagnostics"],
            "geometric_bfs": geom["route_bfs_diagnostics"],
            "geometric_clearance_preferred": geom["route_diagnostics"],
        },
        "seconds": float(elapsed),
    }


def _aligned_views(gated_native, aligned):
    production = aligned["target"]
    sa, sb = production["seed_start"], production["seed_goal"]
    start = np.asarray(
        _voxel_world(gated_native.XX, gated_native.YY, gated_native.zs, sa), dtype=float
    )
    goal = np.asarray(
        _voxel_world(gated_native.XX, gated_native.YY, gated_native.zs, sb), dtype=float
    )
    gated = SimpleNamespace(
        start_xyz=start,
        goal_xyz=goal,
        XX=gated_native.XX,
        YY=gated_native.YY,
        zs=gated_native.zs,
        label=aligned["gated_target_label"],
        route_world=production["gated"]["route_world"],
        eps_gated=production["gated"]["eps"],
        merged=production["gated"]["merged"],
    )
    geom = SimpleNamespace(
        start_xyz=start,
        goal_xyz=goal,
        route_world=production["geom"]["route_world"],
        eps_star=production["geom"]["eps"],
        merged=production["geom"]["merged"],
    )
    return gated, geom


def _save_routes(out_dir, seed, arm, gated, geom, aligned):
    np.savez_compressed(
        Path(out_dir) / f"routes_seed{seed}_{arm}.npz",
        gated_route=np.asarray(gated.route_world or [], dtype=float).reshape(-1, 3),
        geometric_route=np.asarray(geom.route_world or [], dtype=float).reshape(-1, 3),
        geometric_bfs_route=np.asarray(
            aligned["target"]["geom"]["route_bfs_world"] or [], dtype=float
        ).reshape(-1, 3),
        start_xyz=np.asarray(geom.start_xyz, dtype=float),
        goal_xyz=np.asarray(geom.goal_xyz, dtype=float),
        common_start_voxel=np.asarray(aligned["target"]["seed_start"], dtype=int),
        common_goal_voxel=np.asarray(aligned["target"]["seed_goal"], dtype=int),
    )


def _ring_config(args, seed):
    offset_text = args.offsets if args.offsets is not None else str(args.offset)
    specs = parse_offset_specs(offset_text)
    if len(specs) != 1:
        raise ValueError(
            "geometric-vs-gated comparison accepts one fixed/range offset spec per run"
        )
    counts = parse_count_choices(args.num_occluders)
    angle0, count, radii = draw_ring_config(
        seed, specs[0], counts, args.random_ring_rotation
    )
    if not args.random_ring_rotation:
        angle0 = float(args.occluder_angle0)
    return {
        "offset_spec": offset_text,
        "offset_nominal_m": float(specs[0][2]),
        "count": int(count),
        "radii_m": [float(radius) for radius in radii],
        "angle0_rad": float(angle0),
        "random_rotation": bool(args.random_ring_rotation),
    }


def _build_scene(args, seed):
    ring = _ring_config(args, seed)
    env = make_occluder_task()()
    env.spawn_occluder = True
    env.occluder_offset = ring["offset_nominal_m"]
    env.num_occluders = ring["count"]
    env.occluder_angle0 = ring["angle0_rad"]
    env.occluder_radii = list(ring["radii_m"])
    env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed, DR_CLEAN))
    ring["spawned_count"] = len(getattr(env, "occluders", []))
    return env, ring


def _read_records(path):
    records = []
    seen = set()
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                seed = int(record["seed"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
            if seed in seen:
                raise ValueError(f"duplicate seed {seed} in {path}")
            seen.add(seed)
            records.append(record)
    return records


def _restore_resume_config(args):
    """Restore the exact scene/metric configuration stored by an interrupted run."""
    if not args.resume_dir:
        return None, None, []
    out_dir = Path(args.resume_dir).resolve()
    config_path = out_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"resume directory lacks config.json: {out_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for name in (
        "seeds", "arm", "base_config", "offset", "num_occluders",
        "occluder_angle0", "reach_cache_dir", "reach_mode",
    ):
        setattr(args, name, config[name])
    args.offsets = config.get("offsets")
    args.random_ring_rotation = bool(config.get("random_ring_rotation", False))
    for name, value in config["metric"].items():
        setattr(args, name, value)
    args.false_keep_mask = config["false_keep_mask"]
    records = _read_records(out_dir / "records.jsonl")
    requested = {int(seed) for seed in args.seeds}
    unexpected = sorted(
        int(record["seed"])
        for record in records
        if int(record["seed"]) not in requested
    )
    if unexpected:
        raise ValueError(f"resume records contain seeds absent from config.json: {unexpected}")
    return out_dir, config, records


def run(args):
    out_dir, stored_config, records = _restore_resume_config(args)
    if out_dir is None:
        out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] writing to {out_dir}")
    else:
        print(f"[run] resuming {out_dir} ({len(records)} recorded scenes)")

    cfg = SeedMetricConfig.from_args(args)
    for field, value in vars(cfg).items():
        setattr(args, field, value)
    false_keep = resolve_false_keep_mask(
        args.false_keep_mask, args.false_keep_results_dir,
        args.arm, args.reach_mode, cfg,
    )
    print(f"[stage1] false-keep mask: {false_keep['path']}")
    xs, ys, zs, XX, YY = build_grid(cfg)
    load_reach_envelope(
        args.reach_cache_dir, args.arm, xs, ys, zs, XX, YY, mode=args.reach_mode
    )
    print(f"[stage1] reach cache: {args.reach_cache_dir}")

    config = {
        "seeds": [int(seed) for seed in args.seeds],
        "arm": args.arm,
        "base_config": args.base_config,
        "offset": args.offset,
        "offsets": args.offsets,
        "num_occluders": args.num_occluders,
        "occluder_angle0": args.occluder_angle0,
        "random_ring_rotation": args.random_ring_rotation,
        "reach_cache_dir": str(args.reach_cache_dir),
        "reach_mode": args.reach_mode,
        "false_keep_mask": false_keep["path"],
        "metric": vars(cfg),
    }
    if stored_config is None:
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    else:
        expected_config = dict(stored_config)
        expected_config.setdefault("offsets", None)
        expected_config.setdefault("random_ring_rotation", False)
        if _json_safe(config) != expected_config:
            raise ValueError("restored resume configuration does not match config.json")
    records_path = out_dir / "records.jsonl"
    timings = Timings()
    recorded_seeds = {int(record["seed"]) for record in records}
    pending_seeds = [int(seed) for seed in args.seeds if int(seed) not in recorded_seeds]
    if stored_config is not None:
        print(
            f"[run] skipping {len(recorded_seeds)} recorded scenes; "
            f"{len(pending_seeds)} remain"
        )

    for index, seed in enumerate(pending_seeds, 1):
        print(f"[scene] {index}/{len(pending_seeds)} remaining, seed={seed}")
        started = time.perf_counter()
        env = planner = ik = gated = volume = aligned = gated_view = geom_view = None
        ring_config = None
        try:
            with timings.section(f"seed_{seed}_scene_setup"):
                env, ring_config = _build_scene(args, seed)
                print(
                    f"[scene] ring count={ring_config['count']} "
                    f"spawned={ring_config['spawned_count']} "
                    f"radii={[round(v, 3) for v in ring_config['radii_m']]} "
                    f"angle0={ring_config['angle0_rad']:.3f} rad"
                )
                if ring_config["spawned_count"] != ring_config["count"]:
                    raise SceneNotAlignableError(
                        "one or more sampled occluders fell outside the table: "
                        f"requested {ring_config['count']}, "
                        f"spawned {ring_config['spawned_count']}"
                    )
                grasp_q, grasp_pose = grasp_orientation(env, args.arm, False)
                if grasp_pose is None:
                    raise RuntimeError("no side-grasp pose; Stage 3 requires fixed grasp->pad endpoints")
                start_xyz = np.asarray(grasp_pose[:3], dtype=float)
                goal_xyz = np.array([PAD_XY[0], PAD_XY[1], start_xyz[2]], dtype=float)
                target_xyz = np.asarray(env.target_obj.get_pose().p, dtype=float)
                planner = (
                    env.robot.left_planner if args.arm == "left" else env.robot.right_planner
                )

            with timings.section(f"seed_{seed}_geometric"):
                volume = _build_geometric_volume(
                    env, args.arm, cfg, args.reach_cache_dir, args.reach_mode
                )

            with timings.section(f"seed_{seed}_gated"):
                ik = _build_ik_solver(planner)
                gated = compute_route_configs(
                    env, planner, args.arm, ik, grasp_q, start_xyz, goal_xyz, cfg
                )

            with timings.section(f"seed_{seed}_aligned_compare"):
                aligned = build_aligned_comparison(
                    gated, volume, start_xyz, goal_xyz, cfg
                )
                gated_view, geom_view = _aligned_views(gated, aligned)

            overlap = false_keep_overlap(geom_view.route_world, false_keep)
            foots = gated.foots or volume.foots or occluder_footprints_3d(
                env, obstacles=cfg.obstacles
            )
            with timings.section(f"seed_{seed}_report"):
                args.seed = int(seed)
                save_route_overlay(
                    out_dir, args, gated_view, geom_view, foots, target_xyz
                )
                _save_routes(
                    out_dir, seed, args.arm, gated_view, geom_view, aligned
                )
            record = _scene_record(
                seed, args.arm, gated, aligned, overlap,
                start_xyz, goal_xyz, time.perf_counter() - started, ring_config,
            )
            print(
                f"[scene] seed={seed} target-common "
                f"gated={_fmt_eps(gated_view.eps_gated, gated_view.merged)}  "
                f"geometric={_fmt_eps(geom_view.eps_star, geom_view.merged)}  "
                f"no-target-common "
                f"gated={_fmt_eps(aligned['notarget']['gated']['eps'], aligned['notarget']['gated']['merged'])}  "
                f"geometric={_fmt_eps(aligned['notarget']['geom']['eps'], aligned['notarget']['geom']['merged'])}  "
                f"false-keep route={overlap[0]}/{overlap[1]}"
            )
        except SceneNotAlignableError as exc:
            record = {
                "status": "not_alignable",
                "seed": int(seed),
                "arm": args.arm,
                "occluder_ring": ring_config,
                "reason": str(exc),
                "seconds": time.perf_counter() - started,
            }
            print(f"[scene] seed={seed} NOT ALIGNABLE: {exc}")
        except (AlignmentError, RelaxationInvariantError) as exc:
            print(f"[scene] seed={seed} FATAL: {type(exc).__name__}: {exc}")
            raise
        except Exception as exc:
            record = {
                "status": "error",
                "seed": int(seed),
                "arm": args.arm,
                "occluder_ring": ring_config,
                "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.perf_counter() - started,
            }
            print(f"[scene] seed={seed} ERROR: {record['error']}")
        finally:
            if env is not None:
                try:
                    env.close_env()
                except Exception:
                    pass
            del ik, gated, volume, aligned, gated_view, geom_view, planner, env
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        records.append(record)
        with records_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")

    with timings.section("aggregate_report"):
        summary = write_reports(out_dir, records, config)
    timings.save(out_dir)
    rho = summary["production_target_masked"]["spearman_all"]
    rho_isolation = summary["isolation_no_target"]["spearman_all"]
    print(
        "[gate] Spearman target-masked = "
        f"{'undefined' if rho is None else f'{rho:.3f}'}, no-target = "
        f"{'undefined' if rho_isolation is None else f'{rho_isolation:.3f}'} "
        f"(threshold {summary['rank_threshold']:.1f} for both)"
    )
    print(f"[gate] numeric verdict: {summary['stage3_verdict']}")
    print("[gate] final verdict remains pending until the route overlays are inspected.")
    print(f"[run] outputs in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--base-config", default="bench_demo_office_clean")
    parser.add_argument(
        "--offset", type=float, default=0.2,
        help="legacy fixed ring radius; ignored when --offsets is supplied",
    )
    parser.add_argument(
        "--offsets",
        help=(
            "one fixed radius or range; 0.1-0.25 samples each occluder radius "
            "independently per seed"
        ),
    )
    parser.add_argument(
        "--num-occluders", default="4",
        help="one count or a comma-separated per-seed menu, e.g. 2,3,4,5",
    )
    parser.add_argument("--occluder-angle0", type=float, default=0.0)
    parser.add_argument(
        "--random-ring-rotation", action="store_true",
        help="draw a deterministic whole-ring rotation for every seed",
    )
    parser.add_argument("--reach-mode", choices=["occupancy", "sphere"], default="occupancy")
    parser.add_argument(
        "--reach-cache-dir", default=str(STAGE1_REACH_CACHE_DIR)
    )
    parser.add_argument("--false-keep-mask")
    parser.add_argument(
        "--false-keep-results-dir", default=str(FALSE_KEEP_RESULTS_DIR),
        help="searched for the newest matching Stage-1 false_keep_mask.npz",
    )
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--resume-dir",
        help=(
            "resume an existing timestamped run directory; its config.json is authoritative "
            "and seeds already present in records.jsonl are skipped"
        ),
    )

    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--res", type=float, default=None)
    parser.add_argument("--zmin", type=float, default=None)
    parser.add_argument("--zmax", type=float, default=None)
    parser.add_argument("--zres", type=float, default=None)
    parser.add_argument("--gate-tau", type=float, default=None)
    parser.add_argument("--seed-snap", type=float, default=None)
    parser.add_argument("--warm-seeds", type=int, default=None)
    parser.add_argument("--ik-seeds", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--occ-shape", choices=["mesh", "extruded"], default=None)
    parser.add_argument("--obstacles", choices=["all", "occluders"], default=None)
    parser.add_argument("--free-only", action="store_true", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
