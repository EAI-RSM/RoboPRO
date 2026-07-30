#!/usr/bin/env python3
"""Compare envelope-only geometric eps* with the gated IK metric on identical scenes.

This is Stage 3 of GEOMETRIC_EPS_VALIDATION_PLAN.md.  Each seed builds one occluder scene, then
measures the same grasp-to-pad leg with:

* the existing gated metric (scene IK volume plus the joint-continuity gate), and
* the CPU-only geometric relaxation (reach envelope, no scene IK or joint gate).

The run writes per-scene values, inflation, route overlays, Stage-1 false-keep overlap, a Spearman
summary, and timings.  The script can evaluate the rank threshold automatically; a person must
still inspect the route overlays for far-field detours before the Stage-3 gate can pass.
"""

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import spearmanr

from setup_paths import setup_paths

setup_paths()

from lib.geometric_metric import geometric_eps
from lib.ik_grid import _build_ik_solver, grasp_orientation
from lib.labeling import BEYOND, FREE, LABEL_NAMES, OBSTACLE
from lib.metric_config import SeedMetricConfig
from lib.obstacles import occluder_footprints_3d, occluder_slice_polys
from lib.plotting import _line_axis, _scene_anchor_markers
from lib.run_io import CLEARANCE_RESULTS_DIR, Timings
from lib.scene_build import DR_CLEAN, build_cfg
from lib.scene_constants import PAD_XY
from metric_viz import LABEL_COLORS
from seed_from_clearance import compute_route_configs
from task.occluder_task import make_occluder_task


RESULTS_DIR = CLEARANCE_RESULTS_DIR.parent / "geometric_vs_gated"
FALSE_KEEP_RESULTS_DIR = CLEARANCE_RESULTS_DIR.parent / "reach_envelope_validation"


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


def summarize_records(records):
    """Return the Stage-3 numeric gate summary for completed scene records."""
    rows = [r for r in records if r.get("status") == "ok"]
    x = np.asarray([_rank_value(r, "gated") for r in rows], dtype=float)
    y = np.asarray([_rank_value(r, "geom") for r in rows], dtype=float)

    rho_all = p_all = None
    if len(rows) >= 2 and np.unique(x).size > 1 and np.unique(y).size > 1:
        result = spearmanr(x, y)
        if np.isfinite(result.statistic):
            rho_all, p_all = float(result.statistic), float(result.pvalue)

    finite_rows = [
        r for r in rows
        if r["merged_gated"] and r["merged_geom"]
        and np.isfinite(r["eps_gated"]) and np.isfinite(r["eps_geom"])
    ]
    rho_finite = p_finite = None
    if len(finite_rows) >= 2:
        xf = np.asarray([r["eps_gated"] for r in finite_rows], dtype=float)
        yf = np.asarray([r["eps_geom"] for r in finite_rows], dtype=float)
        if np.unique(xf).size > 1 and np.unique(yf).size > 1:
            result = spearmanr(xf, yf)
            if np.isfinite(result.statistic):
                rho_finite, p_finite = float(result.statistic), float(result.pvalue)

    disagreements = [
        int(r["seed"]) for r in rows if not r["merged_gated"] and r["merged_geom"]
    ]
    reverse_disagreements = [
        int(r["seed"]) for r in rows if r["merged_gated"] and not r["merged_geom"]
    ]
    relaxation_violations = [
        int(r["seed"]) for r in rows if r.get("relaxation_violation")
    ]
    geom_route_voxels = sum(int(r["geom_route_voxels"]) for r in rows)
    false_keep_voxels = sum(int(r["false_keep_route_voxels"]) for r in rows)
    sufficient = len(rows) >= 8
    rank_pass = bool(sufficient and rho_all is not None and rho_all >= 0.8)

    return {
        "requested_scenes": len(records),
        "completed_scenes": len(rows),
        "failed_scenes": len(records) - len(rows),
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
            bool(r["merged_gated"] and np.isinf(r["eps_gated"])) for r in rows
        ),
        "unbounded_geom_scenes": sum(
            bool(r["merged_geom"] and np.isinf(r["eps_geom"])) for r in rows
        ),
        "gated_inaccessible_geom_merged_seeds": disagreements,
        "gated_merged_geom_inaccessible_seeds": reverse_disagreements,
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
    fig, ax = plt.subplots(figsize=(8, 7))
    finite_xy = [
        (_rank_value(r, "gated"), _rank_value(r, "geom"), int(r["seed"]))
        for r in rows
        if np.isfinite(_rank_value(r, "gated")) and np.isfinite(_rank_value(r, "geom"))
    ]
    if finite_xy:
        x, y, seeds = map(np.asarray, zip(*finite_xy))
        ax.scatter(x, y, s=65, color="#1565c0")
        for xv, yv, seed in zip(x, y, seeds):
            ax.annotate(str(int(seed)), (xv, yv), xytext=(4, 4), textcoords="offset points",
                        fontsize=9)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = max(0.005, 0.05 * (hi - lo if hi > lo else 1.0))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.3,
                label="identity")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    rho = summary["spearman_all"]
    rho_text = "undefined" if rho is None else f"{rho:.3f}"
    ax.text(
        0.03, 0.97,
        f"Spearman = {rho_text}\n"
        f"completed n = {summary['completed_scenes']}\n"
        f"unbounded gated/geom = {summary['unbounded_gated_scenes']}/"
        f"{summary['unbounded_geom_scenes']}",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax.set_xlabel("gated eps* (m; inaccessible = 0)")
    ax.set_ylabel("geometric eps* (m; inaccessible = 0)")
    ax.set_title("Geometric relaxation versus gated metric")
    if finite_xy:
        ax.legend(loc="lower right")
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


def _scene_record(seed, arm, gated, geom, overlap, elapsed):
    hits, route_voxels, fraction = overlap
    relaxation_violation = bool(
        (gated.merged and not geom.merged)
        or (
            gated.merged and geom.merged
            and (
                (np.isinf(gated.eps_gated) and not np.isinf(geom.eps_star))
                or (
                    np.isfinite(gated.eps_gated) and np.isfinite(geom.eps_star)
                    and geom.eps_star < gated.eps_gated - 1e-9
                )
            )
        )
    )
    gated_rank = float(gated.eps_gated) if gated.merged else 0.0
    geom_rank = float(geom.eps_star) if geom.merged else 0.0
    if np.isinf(geom_rank) and np.isinf(gated_rank):
        inflation = 0.0
        inflation_unbounded = False
    elif np.isinf(geom_rank):
        inflation = None
        inflation_unbounded = True
    elif np.isinf(gated_rank):
        inflation = None
        inflation_unbounded = False
    else:
        inflation = float(geom_rank - gated_rank)
        inflation_unbounded = False
    return {
        "status": "ok",
        "seed": int(seed),
        "arm": arm,
        "merged_gated": bool(gated.merged),
        "merged_geom": bool(geom.merged),
        "eps_gated": float(gated.eps_gated),
        "eps_geom": float(geom.eps_star),
        "eps_gated_m": _json_eps(gated.eps_gated, gated.merged),
        "eps_geom_m": _json_eps(geom.eps_star, geom.merged),
        "eps_gated_unbounded": bool(gated.merged and np.isinf(gated.eps_gated)),
        "eps_geom_unbounded": bool(geom.merged and np.isinf(geom.eps_star)),
        "inflation_m": inflation,
        "inflation_unbounded": inflation_unbounded,
        "relaxation_violation": relaxation_violation,
        "gated_reason": gated.reason,
        "geom_reason": geom.reason,
        "gated_route_voxels": len(gated.route_world or []),
        "geom_route_voxels": route_voxels,
        "false_keep_route_voxels": hits,
        "false_keep_route_fraction": fraction,
        "start_xyz": [float(v) for v in geom.start_xyz],
        "goal_xyz": [float(v) for v in geom.goal_xyz],
        "seconds": float(elapsed),
    }


def _save_routes(out_dir, seed, arm, gated, geom):
    np.savez_compressed(
        Path(out_dir) / f"routes_seed{seed}_{arm}.npz",
        gated_route=np.asarray(gated.route_world or [], dtype=float).reshape(-1, 3),
        geometric_route=np.asarray(geom.route_world or [], dtype=float).reshape(-1, 3),
        start_xyz=np.asarray(geom.start_xyz, dtype=float),
        goal_xyz=np.asarray(geom.goal_xyz, dtype=float),
    )


def _build_scene(args, seed):
    env = make_occluder_task()()
    env.spawn_occluder = True
    env.occluder_offset = args.offset
    env.num_occluders = args.num_occluders
    env.occluder_angle0 = args.occluder_angle0
    env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed, DR_CLEAN))
    return env


def run(args):
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] writing to {out_dir}")

    cfg = SeedMetricConfig.from_args(args)
    for field, value in vars(cfg).items():
        setattr(args, field, value)
    false_keep = resolve_false_keep_mask(
        args.false_keep_mask, args.false_keep_results_dir,
        args.arm, args.reach_mode, cfg,
    )
    print(f"[stage1] false-keep mask: {false_keep['path']}")

    config = {
        "seeds": [int(seed) for seed in args.seeds],
        "arm": args.arm,
        "base_config": args.base_config,
        "offset": args.offset,
        "num_occluders": args.num_occluders,
        "occluder_angle0": args.occluder_angle0,
        "reach_cache_dir": str(args.reach_cache_dir),
        "reach_mode": args.reach_mode,
        "false_keep_mask": false_keep["path"],
        "metric": vars(cfg),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    records_path = out_dir / "records.jsonl"
    records = []
    timings = Timings()

    for index, seed in enumerate(args.seeds, 1):
        print(f"[scene] {index}/{len(args.seeds)} seed={seed}")
        started = time.perf_counter()
        env = planner = ik = gated = geom = None
        try:
            with timings.section(f"seed_{seed}_scene_setup"):
                env = _build_scene(args, seed)
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
                geom = geometric_eps(
                    env, args.arm, [(start_xyz, goal_xyz)], cfg,
                    args.reach_cache_dir, args.reach_mode,
                )[0]

            with timings.section(f"seed_{seed}_gated"):
                ik = _build_ik_solver(planner)
                gated = compute_route_configs(
                    env, planner, args.arm, ik, grasp_q, start_xyz, goal_xyz, cfg
                )

            overlap = false_keep_overlap(geom.route_world, false_keep)
            foots = gated.foots or occluder_footprints_3d(env, obstacles=cfg.obstacles)
            with timings.section(f"seed_{seed}_report"):
                args.seed = int(seed)
                save_route_overlay(out_dir, args, gated, geom, foots, target_xyz)
                _save_routes(out_dir, seed, args.arm, gated, geom)
            record = _scene_record(
                seed, args.arm, gated, geom, overlap, time.perf_counter() - started
            )
            print(
                f"[scene] seed={seed} gated={_fmt_eps(gated.eps_gated, gated.merged)}  "
                f"geometric={_fmt_eps(geom.eps_star, geom.merged)}  "
                f"false-keep route={overlap[0]}/{overlap[1]}"
            )
        except Exception as exc:
            record = {
                "status": "error",
                "seed": int(seed),
                "arm": args.arm,
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
            del ik, gated, geom, planner, env
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
    rho = summary["spearman_all"]
    print(f"[gate] Spearman = {'undefined' if rho is None else f'{rho:.3f}'} "
          f"(threshold {summary['rank_threshold']:.1f})")
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
    parser.add_argument("--offset", type=float, default=0.2)
    parser.add_argument("--num-occluders", type=int, default=4)
    parser.add_argument("--occluder-angle0", type=float, default=0.0)
    parser.add_argument("--reach-mode", choices=["occupancy", "sphere"], default="occupancy")
    parser.add_argument(
        "--reach-cache-dir", default=str(CLEARANCE_RESULTS_DIR / "_reach_cache")
    )
    parser.add_argument("--false-keep-mask")
    parser.add_argument(
        "--false-keep-results-dir", default=str(FALSE_KEEP_RESULTS_DIR),
        help="searched for the newest matching Stage-1 false_keep_mask.npz",
    )
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))

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
