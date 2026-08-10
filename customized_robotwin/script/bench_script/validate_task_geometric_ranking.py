#!/usr/bin/env python3
"""Calibrate geometric against gated eps* on canonical put_cup_on_coaster legs."""

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from setup_paths import setup_paths

setup_paths()
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

from compare_geometric_vs_gated import (
    AlignmentError,
    RelaxationInvariantError,
    SceneNotAlignableError,
    build_aligned_comparison,
)
from lib.geometric_metric import _build_geometric_volume
from lib.ik_grid import _build_ik_solver, build_grid
from lib.labeling import load_reach_envelope
from lib.metric_config import SeedMetricConfig
from lib.run_io import CLEARANCE_RESULTS_DIR, Timings, atomic_write_json
from lib.scene_build import build_cfg, dr_measure, get_env_class
from lib.scene_provenance import task_scene_code_version, task_scene_identity
from lib.task_roles import SUPPORTED_TASK, resolve_task_roles
from lib.waypoints import canonical_legs, canonical_waypoints
from seed_from_clearance import compute_route_configs
from task_metric import DEFAULT_CHECKPOINT, REACH_CACHE_DIR, _default_instruction


RESULTS_DIR = CLEARANCE_RESULTS_DIR.parent / "task_geometric_ranking"


def _rank_value(merged, eps):
    return float(eps) if merged else 0.0


def _stored_rank(merged, eps):
    value = _rank_value(merged, eps)
    return (None if np.isposinf(value) else value), bool(np.isposinf(value))


def _record_rank(record, prefix):
    return np.inf if record[f"{prefix}_unbounded"] else float(record[prefix])


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    result = spearmanr(x, y)
    return float(result.statistic) if np.isfinite(result.statistic) else None


def summarize_task_calibration(records):
    rows = [record for record in records if record.get("status") == "ok"]
    summary = {
        "schema": "robopro.task-geometric-ranking-summary.v1",
        "requested_scenes": len(records),
        "aligned_scenes": len(rows),
        "not_alignable_scenes": sum(
            record.get("status") == "not_alignable" for record in records
        ),
        "error_scenes": sum(record.get("status") == "error" for record in records),
        "no_arbitrary_minimum_n_gate": True,
        "route_fidelity_in_scope": False,
        "decision": "USER_REVIEW_REQUIRED",
        "scene_min": {"spearman": None, "n": len(rows)},
        "per_leg": {},
    }
    if rows:
        summary["scene_min"]["spearman"] = _spearman(
            [_record_rank(record, "eps_gated_min_rank") for record in rows],
            [_record_rank(record, "eps_geom_min_rank") for record in rows],
        )
        leg_keys = [(leg["index"], leg["kind"]) for leg in rows[0]["legs"]]
        for index, kind in leg_keys:
            pairs = []
            for record in rows:
                matches = [
                    leg for leg in record["legs"]
                    if leg["index"] == index and leg["kind"] == kind
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"scene {record['scene_id']} lacks exactly one leg {index}:{kind}"
                    )
                pairs.append(matches[0])
            summary["per_leg"][f"{index:02d}_{kind}"] = {
                "n": len(pairs),
                "spearman": _spearman(
                    [_record_rank(pair, "eps_gated_rank") for pair in pairs],
                    [_record_rank(pair, "eps_geom_rank") for pair in pairs],
                ),
                "exact_equal_n": sum(
                    pair["merged_gated"] == pair["merged_geom"]
                    and (
                        (not pair["merged_gated"])
                        or np.isclose(
                            _record_rank(pair, "eps_gated_rank"),
                            _record_rank(pair, "eps_geom_rank"),
                            rtol=0.0, atol=1e-12,
                        )
                    )
                    for pair in pairs
                ),
            }
    return summary


def _plot_scatter(out_dir, records, summary):
    rows = [record for record in records if record.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(8, 7))
    if rows:
        x = np.asarray([_record_rank(record, "eps_gated_min_rank") for record in rows])
        y = np.asarray([_record_rank(record, "eps_geom_min_rank") for record in rows])
        ax.scatter(x, y, s=70, color="#1565c0")
        for xv, yv, record in zip(x, y, rows):
            ax.annotate(
                f"{record['seed']}/d{record['obstacle_density']}",
                (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=9,
            )
        finite = np.r_[x[np.isfinite(x)], y[np.isfinite(y)]]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            pad = max(0.005, (hi - lo) * 0.05)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2)
    rho = summary["scene_min"]["spearman"]
    ax.set(
        xlabel="Minimum gated eps* across canonical legs (m; inaccessible=0)",
        ylabel="Minimum geometric eps across canonical legs (m; inaccessible=0)",
        title=f"Stock-task scalar rank calibration (Spearman={rho})",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "task_eps_geom_vs_gated.png", dpi=160)
    plt.close(fig)


def _json_eps(merged, eps):
    if not merged or np.isinf(eps):
        return None
    return float(eps)


def _validate_orientation_family(waypoints):
    quaternions = [np.asarray(waypoint.quat, dtype=float) for waypoint in waypoints[1:]]
    if not all(np.allclose(quaternions[0], quat, rtol=0.0, atol=1e-10) for quat in quaternions[1:]):
        raise ValueError("canonical task has multiple orientation families; split calibration")
    return quaternions[0]


def _scene_record(
    env, args, seed, density, cfg, code_version, instruction, scene_cfg
):
    roles = resolve_task_roles(env, args.task_name)
    env.target_collision_path = roles.target.collision_path
    waypoints = canonical_waypoints(env, args.task_name)
    legs = canonical_legs(waypoints)
    arm = waypoints[0].arm
    grasp_q = _validate_orientation_family(waypoints)
    planner = env.robot.left_planner if arm == "left" else env.robot.right_planner
    if planner is None:
        raise RuntimeError("stock-task calibration requires the selected arm planner")

    volume = _build_geometric_volume(
        env, arm, cfg, args.reach_cache_dir, args.reach_mode
    )
    ik = _build_ik_solver(planner)
    gated = compute_route_configs(
        env, planner, arm, ik, grasp_q,
        np.asarray(legs[0].start_xyz), np.asarray(legs[0].goal_xyz), cfg,
    )
    if gated.edt is None or gated.q_warm_3d is None:
        raise SceneNotAlignableError(
            f"gated volume unavailable after first canonical leg: {gated.reason}"
        )

    leg_records = []
    for index, leg in enumerate(legs):
        aligned = build_aligned_comparison(
            gated, volume,
            np.asarray(leg.start_xyz), np.asarray(leg.goal_xyz), cfg,
        )
        production = aligned["target"]
        gated_result = production["gated"]
        geom_result = production["geom"]
        gated_rank, gated_rank_unbounded = _stored_rank(
            gated_result["merged"], gated_result["eps"]
        )
        geom_rank, geom_rank_unbounded = _stored_rank(
            geom_result["merged"], geom_result["eps"]
        )
        leg_records.append(
            {
                "index": index,
                "kind": leg.kind,
                "gripper_state": leg.gripper_state,
                "start_xyz": list(leg.start_xyz),
                "goal_xyz": list(leg.goal_xyz),
                "merged_gated": bool(gated_result["merged"]),
                "merged_geom": bool(geom_result["merged"]),
                "eps_gated": _json_eps(gated_result["merged"], gated_result["eps"]),
                "eps_geom": _json_eps(geom_result["merged"], geom_result["eps"]),
                "eps_gated_unbounded": bool(
                    gated_result["merged"] and np.isposinf(gated_result["eps"])
                ),
                "eps_geom_unbounded": bool(
                    geom_result["merged"] and np.isposinf(geom_result["eps"])
                ),
                "eps_gated_rank": gated_rank,
                "eps_gated_rank_unbounded": gated_rank_unbounded,
                "eps_geom_rank": geom_rank,
                "eps_geom_rank_unbounded": geom_rank_unbounded,
                "common_start_voxel": list(production["seed_start"]),
                "common_goal_voxel": list(production["seed_goal"]),
            }
        )

    identity = task_scene_identity(
        env,
        task=args.task_name,
        seed=seed,
        replicate=args.replicate,
        bench_subdir=args.bench_subdir,
        base_config=args.base_config,
        dr_settings=scene_cfg["domain_randomization"],
        checkpoint=args.checkpoint,
        scene_code_version=code_version,
        roles=roles,
        acting_arm=arm,
        instruction=instruction,
    )
    gated_min = min(_record_rank(leg, "eps_gated_rank") for leg in leg_records)
    geom_min = min(_record_rank(leg, "eps_geom_rank") for leg in leg_records)
    return {
        "status": "ok",
        "scene_id": identity["scene_id"],
        "scene_fingerprint": identity["scene_fingerprint"],
        "seed": int(seed),
        "replicate": int(args.replicate),
        "obstacle_density": int(density),
        "clutter_count": identity["clutter_count"],
        "arm": arm,
        "orientation_family_quaternion": [float(value) for value in grasp_q],
        "legs": leg_records,
        "eps_gated_min_rank": None if np.isposinf(gated_min) else gated_min,
        "eps_gated_min_rank_unbounded": bool(np.isposinf(gated_min)),
        "eps_geom_min_rank": None if np.isposinf(geom_min) else geom_min,
        "eps_geom_min_rank_unbounded": bool(np.isposinf(geom_min)),
    }


def run(args):
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"[run] outputs: {out_dir}")
    cfg = SeedMetricConfig.from_env(SeedMetricConfig.from_args(args))
    for name, value in vars(cfg).items():
        setattr(args, name, value)
    densities = [int(value) for value in args.clutter_densities.split(",") if value.strip()]
    if not densities:
        raise ValueError("--clutter-densities parsed to nothing")
    code_version = task_scene_code_version(args.base_config)
    instruction = args.instruction or _default_instruction(args.task_name)
    config = {
        "task": args.task_name,
        "bench_subdir": args.bench_subdir,
        "base_config": args.base_config,
        "seeds": [int(seed) for seed in args.seeds],
        "density_cycle": densities,
        "replicate": args.replicate,
        "checkpoint": args.checkpoint,
        "instruction": instruction,
        "scene_code_version": code_version,
        "reach_cache_dir": str(Path(args.reach_cache_dir).resolve()),
        "reach_mode": args.reach_mode,
        "metric": asdict(cfg),
        "all_canonical_legs_share_one_gated_and_one_geometric_volume_per_scene": True,
        "route_fidelity_in_scope": False,
    }
    atomic_write_json(out_dir / "config.json", config)

    xs, ys, zs, XX, YY = build_grid(cfg)
    for arm in ("left", "right"):
        load_reach_envelope(
            args.reach_cache_dir, arm, xs, ys, zs, XX, YY, mode=args.reach_mode
        )
    records = []
    records_path = out_dir / "records.jsonl"
    timings = Timings()
    env_class = get_env_class(args.task_name, bench_subdir=args.bench_subdir)
    for index, seed in enumerate(args.seeds):
        density = densities[index % len(densities)]
        print(f"[scene] {index + 1}/{len(args.seeds)} seed={seed} d{density}")
        started = time.perf_counter()
        env = None
        try:
            with timings.section(f"seed_{seed}_setup_and_compare"):
                scene_cfg = build_cfg(
                    args.task_name, args.base_config, seed,
                    dr_measure(density), mode="measure",
                )
                env = env_class()
                env.setup_demo(**scene_cfg)
                table_z_bias = float(getattr(env, "table_z_bias", np.nan))
                if not np.isfinite(table_z_bias) or abs(table_z_bias) > 1e-12:
                    raise ValueError("table_z_bias must be zero for stock-task calibration")
                record = _scene_record(
                    env, args, int(seed), density, cfg, code_version, instruction,
                    scene_cfg,
                )
        except SceneNotAlignableError as exc:
            record = {
                "status": "not_alignable", "seed": int(seed),
                "obstacle_density": density, "reason": str(exc),
            }
        except (AlignmentError, RelaxationInvariantError):
            raise
        except Exception as exc:
            record = {
                "status": "error", "seed": int(seed),
                "obstacle_density": density,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            if env is not None:
                try:
                    env.close_env()
                except Exception:
                    pass
            del env
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        record["seconds"] = time.perf_counter() - started
        records.append(record)
        with records_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    with timings.section("report"):
        summary = summarize_task_calibration(records)
        summary["config"] = config
        atomic_write_json(out_dir / "summary.json", summary)
        _plot_scatter(out_dir, records, summary)
    timings.save(out_dir)
    print(
        f"[calibration] aligned={summary['aligned_scenes']}/{summary['requested_scenes']} "
        f"scene-min Spearman={summary['scene_min']['spearman']}"
    )
    print("[calibration] user review is required; no arbitrary n or rho threshold is enforced")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2100, 2106)))
    parser.add_argument("--clutter-densities", default="6,10,15")
    parser.add_argument("--task-name", choices=[SUPPORTED_TASK], default=SUPPORTED_TASK)
    parser.add_argument("--bench-subdir", choices=["study"], default="study")
    parser.add_argument("--base-config", default="bench_demo_study_clean")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--instruction")
    parser.add_argument("--reach-mode", choices=["occupancy", "sphere"], default="occupancy")
    parser.add_argument("--reach-cache-dir", default=str(REACH_CACHE_DIR))
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
    parser.add_argument("--obstacles", choices=["all"], default="all")
    parser.add_argument("--free-only", action="store_true", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
