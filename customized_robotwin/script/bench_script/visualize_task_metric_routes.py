#!/usr/bin/env python3
"""Backfill per-rollout 3D geometric-epsilon route figures.

This command replays initialized Study scenes, verifies the regenerated metric against the
authoritative committed metric record, and then reuses ``metric_viz._metric_path3d``.  It never
loads a policy or changes the rollout/metric source directories.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from setup_paths import setup_paths

setup_paths()
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")
os.environ.setdefault("MPLBACKEND", "Agg")

from lib.geometric_metric import geometric_eps
from lib.metric_buckets import assign_metric_record, load_bucket_spec
from lib.metric_config import SeedMetricConfig
from lib.obstacles import obstacle_centers, occluder_footprints_3d
from lib.run_io import atomic_write_json, sha256_file
from lib.scene_build import build_cfg, get_env_class
from lib.scene_provenance import fingerprint, hash_files, task_scene_code_version
from lib.task_roles import resolve_task_roles
from lib.vla_reporting import read_episode_records
from lib.waypoints import canonical_legs, canonical_waypoints
from metric_viz import _metric_path3d
from task_metric import (
    _metric_record,
    _read_committed_metric_records,
    _read_scene_manifest,
    _validate_expected_identity,
)


SCHEMA = "robopro.task-metric-route-visual.v1"
CONFIG_SCHEMA = "robopro.task-metric-route-visual-config.v1"
REPORT_SCHEMA = "robopro.task-metric-route-visual-report-state.v1"
DEFAULT_ROLLOUT = (
    Path(__file__).resolve().parents[3]
    / "scripts" / "validation" / "results" / "task_metric_vla_full"
    / "association_d6_d10_d15" / "20260731-182037"
)
FLOAT_ATOL = 1e-12




def _recordsha256_file(record):
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_embedded_hash(config, path):
    if not isinstance(config, dict):
        raise ValueError(f"config is not an object: {path}")
    stored = config.get("config_sha256")
    unhashed = {key: value for key, value in config.items() if key != "config_sha256"}
    if not isinstance(stored, str) or stored != fingerprint(unhashed):
        raise ValueError(f"config hash mismatch: {path}")
    return config


def _eps(record_or_leg, value_key, unbounded_key):
    unbounded = record_or_leg.get(unbounded_key)
    value = record_or_leg.get(value_key)
    if unbounded is True:
        if value is not None:
            raise ValueError(f"{value_key} must be null when {unbounded_key}=true")
        return math.inf
    if unbounded is not False:
        raise ValueError(f"{unbounded_key} must be an explicit Boolean")
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"finite {value_key} must contain a finite value")
    return float(value)


def minimum_leg_indices(metric_record):
    """Return every leg that exactly attains the committed ``eps_geom_min``."""
    minimum = _eps(metric_record, "eps_geom_min", "eps_geom_min_unbounded")
    indices = []
    for position, leg in enumerate(metric_record.get("legs") or []):
        index = int(leg.get("index", position))
        value = _eps(leg, "eps_geom", "eps_geom_unbounded")
        if value == minimum:
            indices.append(index)
    if not indices:
        raise ValueError("no canonical leg attains eps_geom_min")
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate minimum-leg index")
    return indices


def _compare_xyz(name, expected, actual, mismatches):
    if expected is None or actual is None:
        if expected is not None or actual is not None:
            mismatches[name] = {"expected": expected, "actual": actual}
        return
    left = np.asarray(expected, dtype=float).reshape(-1)
    right = np.asarray(actual, dtype=float).reshape(-1)
    if left.shape != right.shape or not np.allclose(
        left, right, rtol=0.0, atol=FLOAT_ATOL
    ):
        mismatches[name] = {"expected": left.tolist(), "actual": right.tolist()}


def validate_regenerated_metric(committed, regenerated):
    """Prove a replay produced the same scene and scalar metric before plotting its route."""
    mismatches = {}
    for field in (
        "schema", "status", "scene_id", "scene_fingerprint", "scene_code_version",
        "rollout_episode", "task", "seed", "replicate", "bench_subdir", "base_config",
        "obstacle_density", "clutter_count", "checkpoint", "instruction", "arm", "n_free",
        "gripper_reference_radius_m", "eps_geom_min_unbounded",
    ):
        if committed.get(field) != regenerated.get(field):
            mismatches[field] = {
                "expected": committed.get(field), "actual": regenerated.get(field)
            }

    committed_min = _eps(committed, "eps_geom_min", "eps_geom_min_unbounded")
    regenerated_min = _eps(regenerated, "eps_geom_min", "eps_geom_min_unbounded")
    if not (
        (math.isinf(committed_min) and math.isinf(regenerated_min))
        or math.isclose(committed_min, regenerated_min, rel_tol=0.0, abs_tol=FLOAT_ATOL)
    ):
        mismatches["eps_geom_min"] = {
            "expected": committed_min, "actual": regenerated_min
        }

    expected_legs = committed.get("legs") or []
    actual_legs = regenerated.get("legs") or []
    if len(expected_legs) != len(actual_legs):
        mismatches["leg_count"] = {
            "expected": len(expected_legs), "actual": len(actual_legs)
        }
    else:
        for position, (expected, actual) in enumerate(zip(expected_legs, actual_legs)):
            prefix = f"legs[{position}]"
            for field in (
                "index", "kind", "gripper_state", "merged", "reason",
                "eps_geom_unbounded",
            ):
                if expected.get(field) != actual.get(field):
                    mismatches[f"{prefix}.{field}"] = {
                        "expected": expected.get(field), "actual": actual.get(field)
                    }
            expected_eps = _eps(expected, "eps_geom", "eps_geom_unbounded")
            actual_eps = _eps(actual, "eps_geom", "eps_geom_unbounded")
            if not (
                (math.isinf(expected_eps) and math.isinf(actual_eps))
                or math.isclose(expected_eps, actual_eps, rel_tol=0.0, abs_tol=FLOAT_ATOL)
            ):
                mismatches[f"{prefix}.eps_geom"] = {
                    "expected": expected_eps, "actual": actual_eps
                }
            _compare_xyz(
                f"{prefix}.start_xyz", expected.get("start_xyz"), actual.get("start_xyz"),
                mismatches,
            )
            _compare_xyz(
                f"{prefix}.goal_xyz", expected.get("goal_xyz"), actual.get("goal_xyz"),
                mismatches,
            )
            _compare_xyz(
                f"{prefix}.bottleneck_xyz", expected.get("bottleneck_xyz"),
                actual.get("bottleneck_xyz"), mismatches,
            )

    try:
        expected_minima = minimum_leg_indices(committed)
        actual_minima = minimum_leg_indices(regenerated)
        if expected_minima != actual_minima:
            mismatches["minimum_leg_indices"] = {
                "expected": expected_minima, "actual": actual_minima
            }
    except ValueError as exc:
        mismatches["minimum_leg_indices"] = str(exc)

    if mismatches:
        episode = committed.get("rollout_episode")
        raise ValueError(
            f"regenerated metric mismatch for rollout episode {episode}: {mismatches}"
        )
    return minimum_leg_indices(committed)


def _ensure_symlink(link, target):
    link = Path(link)
    target = Path(target)
    if not target.is_file() or target.stat().st_size <= 0:
        raise FileNotFoundError(f"visual-audit source is missing or empty: {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    expected = os.path.relpath(target.resolve(), link.parent.resolve())
    if os.path.lexists(link):
        if not link.is_symlink() or os.readlink(link) != expected:
            raise ValueError(f"existing visual-audit index entry is inconsistent: {link}")
        return str(link)
    link.symlink_to(expected)
    return str(link)


def _validate_visual_record(
    record, path, out_dir, config_sha256, target_n, expected_source_hashes=None
):
    if record.get("schema") != SCHEMA or record.get("status") != "ok":
        raise ValueError(f"invalid visualization episode record: {path}")
    episode = int(record.get("episode", -1))
    if episode < 0 or episode >= target_n:
        raise ValueError(f"visualization episode is outside target range: {path}")
    if record.get("visualization_config_sha256") != config_sha256:
        raise ValueError(f"visualization episode uses a different config: {path}")
    if (
        expected_source_hashes is not None
        and record.get("source_record_sha256") != expected_source_hashes
    ):
        raise ValueError(f"visualization episode source records changed: {path}")
    figures = record.get("figures")
    if not isinstance(figures, list) or not figures:
        raise ValueError(f"visualization episode has no figures: {path}")
    for figure in figures:
        relative = figure.get("figure_relpath")
        target = Path(out_dir) / relative if isinstance(relative, str) else None
        if target is None or not target.is_file() or target.stat().st_size <= 0:
            raise ValueError(f"visualization episode has a missing figure: {path}")
    return record


def read_visual_records(out_dir, config_sha256, target_n, source_hashes_by_episode=None):
    records = []
    seen = set()
    episodes_dir = Path(out_dir) / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(episodes_dir.glob("episode*.json")):
        suffix = path.stem.removeprefix("episode")
        if not suffix.isdigit():
            raise ValueError(f"malformed visualization episode filename: {path.name}")
        episode = int(suffix)
        if episode in seen:
            raise ValueError(f"duplicate visualization episode: {episode}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if int(record.get("episode", -1)) != episode:
            raise ValueError(f"visualization episode does not match filename: {path}")
        if source_hashes_by_episode is not None and episode not in source_hashes_by_episode:
            raise ValueError(f"visualization episode has no committed source metric: {path}")
        records.append(
            _validate_visual_record(
                record,
                path,
                out_dir,
                config_sha256,
                target_n,
                None if source_hashes_by_episode is None
                else source_hashes_by_episode.get(episode),
            )
        )
        seen.add(episode)
    return sorted(records, key=lambda row: int(row["episode"]))


def regenerate_indexes(out_dir, rollout_dir, records, selected_episodes, source_target_n):
    """Pair every committed figure with its source video in density-first audit folders."""
    out_dir = Path(out_dir)
    rollout_dir = Path(rollout_dir)
    selected = {int(episode) for episode in selected_episodes}
    completed = {int(record["episode"]) for record in records}
    index = []
    for record in sorted(records, key=lambda row: int(row["episode"])):
        episode = int(record["episode"])
        seed = int(record["seed"])
        base = (
            out_dir / "by_density" / f"d{int(record['clutter_density'])}"
            / record["clearance_bucket"] / record["hard_outcome"]
        )
        video_source = rollout_dir / record["video_relpath"]
        video_link = base / f"episode{episode:06d}_seed{seed}_video.mp4"
        _ensure_symlink(video_link, video_source)
        figure_links = []
        for figure in record["figures"]:
            source = out_dir / figure["figure_relpath"]
            link = base / Path(figure["figure_relpath"]).name
            _ensure_symlink(link, source)
            figure_links.append(str(link.relative_to(out_dir)))
        index.append(
            {
                "episode": episode,
                "scene_id": record["scene_id"],
                "seed": seed,
                "clutter_density": int(record["clutter_density"]),
                "clearance_bucket": record["clearance_bucket"],
                "hard_outcome": record["hard_outcome"],
                "video_relpath": record["video_relpath"],
                "video_index_link": str(video_link.relative_to(out_dir)),
                "figure_index_links": figure_links,
            }
        )
    atomic_write_json(out_dir / "route_visual_index.json", index)

    density_counts = Counter(str(row["clutter_density"]) for row in records)
    bucket_counts = Counter(row["clearance_bucket"] for row in records)
    outcome_counts = Counter(row["hard_outcome"] for row in records)
    # Counts the BINDING leg only. Every leg now has a figure, so an unfiltered tally would be a
    # flat per-leg count and would no longer say where eps_geom_min lives.
    leg_counts = Counter(
        figure["kind"] for row in records for figure in row["figures"]
        if figure.get("is_minimum")
    )
    atomic_write_json(
        out_dir / "report_state.json",
        {
            "schema": REPORT_SCHEMA,
            "visualized_episodes": len(records),
            "selected_visualized_episodes": len(selected & completed),
            "target_episodes": len(selected),
            "source_target_episodes": int(source_target_n),
            "selected_episode_ids": sorted(selected),
            "processing_complete": selected <= completed,
            "figure_count": sum(len(row["figures"]) for row in records),
            "video_links": len(index),
            "counts_by_density": dict(sorted(density_counts.items())),
            "counts_by_bucket": dict(sorted(bucket_counts.items())),
            "counts_by_outcome": dict(sorted(outcome_counts.items())),
            "figure_counts_by_minimum_leg": dict(sorted(leg_counts.items())),
        },
    )
    return index


def select_stratified(metric_records, rollout_by_episode, bucket_spec, per_cell):
    """Pick the first N committed scenes per density x frozen-clearance cell."""
    if per_cell <= 0:
        raise ValueError("stratified selection count must be positive")
    names = list(bucket_spec["bucket_order"])
    densities = sorted(
        {int(row["clutter_density"]) for row in rollout_by_episode.values()}
    )
    selected = []
    counts = Counter()
    for metric in sorted(metric_records, key=lambda row: int(row["rollout_episode"])):
        episode = int(metric["rollout_episode"])
        density = int(rollout_by_episode[episode]["clutter_density"])
        bucket = assign_metric_record(metric, bucket_spec)
        cell = (density, bucket)
        if counts[cell] < per_cell:
            selected.append(episode)
            counts[cell] += 1
    missing = [
        (density, bucket, per_cell - counts[(density, bucket)])
        for density in densities for bucket in names
        if counts[(density, bucket)] < per_cell
    ]
    if missing:
        raise ValueError(f"committed metrics do not fill the requested stratified cells: {missing}")
    return sorted(selected)


def select_first(source_target_n, count):
    """Select a bounded prefix of rollout episodes for a small visual audit."""
    if count <= 0:
        raise ValueError("first-episode selection count must be positive")
    if count > source_target_n:
        raise ValueError(
            f"cannot select the first {count} episodes from a {source_target_n}-episode run"
        )
    return list(range(count))


def _visualization_code_version():
    root = Path(__file__).resolve().parent
    return hash_files(
        [Path(__file__), root / "metric_viz.py", root / "lib" / "plotting.py"]
    )


def _visual_config(rollout_dir, rollout_config, metric_config_path, metric_config,
                   bucket_spec_path, target_n):
    config = {
        "schema": CONFIG_SCHEMA,
        "source_rollout": {
            "path": str(Path(rollout_dir).resolve()),
            "config_sha256": rollout_config["config_sha256"],
        },
        "source_metric_config": {
            "path": str(Path(metric_config_path).resolve()),
            "file_sha256": sha256_file(metric_config_path),
            "config_sha256": metric_config["config_sha256"],
            "metric_code_version": metric_config["metric_code_version"],
        },
        "bucket_spec": {
            "path": str(Path(bucket_spec_path).resolve()),
            "sha256": sha256_file(bucket_spec_path),
        },
        "reach_cache_dir": str(Path(metric_config["reach_cache_dir"]).resolve()),
        "reach_mode": metric_config["reach_mode"],
        "source_target_episodes": int(target_n),
        "visualization_code_version": _visualization_code_version(),
        "path_semantics": "geometric_representative_not_executed_vla",
    }
    config["config_sha256"] = fingerprint(config)
    return config


def _load_or_create_config(path, expected):
    path = Path(path)
    if not path.is_file():
        atomic_write_json(path, expected)
        return expected
    stored = _validate_embedded_hash(json.loads(path.read_text(encoding="utf-8")), path)
    if stored != expected:
        raise ValueError(
            "route-visualization configuration differs from the immutable saved config"
        )
    return stored


def _atomic_route_plot(out_dir, plot_args, foots, occ_ps, result, kind, target_p,
                       episode, density, bucket, table_bbox=None, destination_p=None,
                       leg_index=None, leg_count=None, is_minimum=True, minimum_kinds=None):
    figures_dir = Path(out_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # The leg index leads the kind so a plain name sort walks the path in execution order.
    # Sorting on the kind alone gives carry, grasp, place, pre_grasp -- legs 2, 1, 3, 0.
    order = "" if leg_index is None else f"leg{leg_index}_"
    stem = f"episode{episode:06d}_seed{int(plot_args.seed)}_{order}{kind}"
    final = figures_dir / f"{stem}.png"
    # Every leg is drawn, so the caption has to say which one this is and whether it is the one
    # that sets eps_geom_min -- on three figures out of four it is not.
    position = kind if leg_index is None else f"leg {leg_index + 1}/{leg_count}: {kind}"
    if is_minimum:
        role = "BINDS eps_geom_min"
    else:
        which = ", ".join(minimum_kinds or []) or "another leg"
        role = f"not binding; eps_geom_min is at {which}"
    title_context = (
        f"episode {episode}, d{density}, {bucket}  |  {position}  --  {role}"
        "\ngeometric representative path; not executed VLA trajectory"
    )
    frame_xy = ground_z = None
    if table_bbox is not None:
        (xlo, ylo, _), (xhi, yhi, ztop) = table_bbox
        frame_xy, ground_z = ((float(xlo), float(xhi)), (float(ylo), float(yhi))), float(ztop)
    # Only the grasp-side legs end at the target; on the holding legs the cup has left its spawn,
    # so a connector there would assert a relationship that no longer holds.
    tool_link_xyz = result.goal_xyz if kind in {"pre_grasp", "grasp"} else None
    with tempfile.TemporaryDirectory(prefix=f".{stem}.", dir=figures_dir) as temporary:
        temporary = Path(temporary)
        exact = _metric_path3d(
            temporary,
            plot_args,
            foots,
            occ_ps,
            result.start_xyz,
            result.goal_xyz,
            result.bottleneck_xyz,
            result.route_world,
            result.eps_star,
            result.merged,
            tgt_p=target_p,
            ee_xyz=None,
            start_label=f"{kind} leg start",
            goal_label=f"{kind} leg goal",
            route_label="geometric representative path",
            metric_label="eps_geom",
            title_context=title_context,
            stem=stem,
            frame_xy=frame_xy,
            ground_z=ground_z,
            view=(22, -72),
            tool_link_xyz=tool_link_xyz,
            dest_xyz=destination_p,
            target_label="target cup (spawn)",
            geom_label="obstacle",
        )
        staged = temporary / f"{stem}.png"
        if not staged.is_file() or staged.stat().st_size <= 0:
            raise RuntimeError(f"renderer did not produce a valid PNG: {staged}")
        os.replace(staged, final)
    return final, exact


def _hard_outcome(rollout):
    collision = rollout.get("collision_metrics", {}).get("is_collision")
    if not isinstance(collision, bool):
        raise ValueError(f"rollout episode {rollout.get('episode')} has invalid collision status")
    task_success = rollout.get("task_success")
    hard_success = rollout.get("hard_success")
    if not isinstance(task_success, bool) or not isinstance(hard_success, bool):
        raise ValueError(f"rollout episode {rollout.get('episode')} has invalid outcome fields")
    expected = task_success and not collision
    if hard_success != expected:
        raise ValueError(f"rollout episode {rollout.get('episode')} has invalid hard_success")
    return "hard_success" if expected else "hard_fail"


def validate_source_pair(job, rollout, committed):
    """Prove the manifest, rollout/video row, and committed metric describe one episode."""
    comparisons = {
        "episode": (job.get("rollout_episode"), rollout.get("episode"),
                    committed.get("rollout_episode")),
        "seed": (job.get("seed"), rollout.get("seed"), committed.get("seed")),
        "task": (job.get("task"), rollout.get("task"), committed.get("task")),
        "replicate": (job.get("replicate"), rollout.get("replicate"),
                      committed.get("replicate")),
        "bench_subdir": (job.get("bench_subdir"), rollout.get("bench_subdir"),
                         committed.get("bench_subdir")),
        "base_config": (job.get("base_config"), rollout.get("base_config"),
                        committed.get("base_config")),
        "scene_id": (job.get("expected_scene_id"), rollout.get("scene_id"),
                     committed.get("scene_id")),
        "scene_fingerprint": (
            job.get("expected_scene_fingerprint"), rollout.get("scene_fingerprint"),
            committed.get("scene_fingerprint"),
        ),
        "scene_code_version": (
            job.get("expected_scene_code_version"), rollout.get("scene_code_version"),
            committed.get("scene_code_version"),
        ),
        "density": (job.get("obstacle_density"), rollout.get("clutter_density"),
                    committed.get("obstacle_density")),
        "clutter_count": (job.get("expected_clutter_count"),
                          rollout.get("clutter_count"), committed.get("clutter_count")),
        "checkpoint": (job.get("checkpoint"), rollout.get("checkpoint"),
                       committed.get("checkpoint")),
        "instruction": (job.get("instruction"), rollout.get("instruction"),
                        committed.get("instruction")),
        "acting_arm": (job.get("expected_acting_arm"), rollout.get("acting_arm"),
                       committed.get("arm")),
    }
    mismatches = {
        name: values for name, values in comparisons.items()
        if values[0] is None or any(value != values[0] for value in values[1:])
    }
    if mismatches:
        raise ValueError(
            f"source records disagree for rollout episode {job.get('rollout_episode')}: "
            f"{mismatches}"
        )


def _source_hashes(job, rollout, committed):
    return {
        "manifest_row": _recordsha256_file(job),
        "rollout_record": _recordsha256_file(rollout),
        "metric_record": _recordsha256_file(committed),
    }


def _validate_source_video(rollout_dir, rollout):
    relative = rollout.get("video_relpath")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"rollout episode {rollout.get('episode')} has invalid video_relpath")
    rollout_dir = Path(rollout_dir).resolve()
    video = (rollout_dir / relative).resolve()
    if not video.is_relative_to(rollout_dir):
        raise ValueError(f"rollout episode {rollout.get('episode')} video leaves rollout directory")
    if not video.is_file() or video.stat().st_size <= 0:
        raise FileNotFoundError(f"rollout episode {rollout.get('episode')} video is missing: {video}")
    return video


def _render_episode(job, rollout, committed, metric_config, metric_cfg, bucket_spec,
                    out_dir, visualization_config_sha256, source_hashes):
    episode = int(job["rollout_episode"])
    seed = int(job["seed"])
    env = None
    started = time.perf_counter()
    try:
        env_class = get_env_class(job["task"], bench_subdir=job["bench_subdir"])
        code_version = task_scene_code_version(job["base_config"])
        scene_cfg = build_cfg(
            job["task"], job["base_config"], seed, dict(job["dr_settings"]), mode="measure"
        )
        dr = scene_cfg["domain_randomization"]
        if int(dr.get("obstacle_density", 0)) > 0 and dr.get("clean_background_rate") != 0:
            raise ValueError("nonzero-density visualization scene did not force clutter on")
        env = env_class()
        env.setup_demo(**scene_cfg)
        table_z_bias = float(getattr(env, "table_z_bias", np.nan))
        if not np.isfinite(table_z_bias) or abs(table_z_bias) > FLOAT_ATOL:
            raise ValueError(f"table_z_bias must be 0, got {table_z_bias}")
        roles = resolve_task_roles(env, job["task"])
        env.target_collision_path = roles.target.collision_path
        waypoints = canonical_waypoints(env, job["task"])
        leg_specs = canonical_legs(waypoints)
        results = geometric_eps(
            env,
            waypoints[0].arm,
            [(leg.start_xyz, leg.goal_xyz) for leg in leg_specs],
            cfg=metric_cfg,
            reach_cache_dir=metric_config["reach_cache_dir"],
            reach_mode=metric_config["reach_mode"],
            mask_target=True,
        )
        record_args = SimpleNamespace(
            gripper_r=float(metric_config["gripper_reference_radius_m"])
        )
        regenerated = _metric_record(
            env, record_args, job, scene_cfg, roles, waypoints, leg_specs, results,
            code_version, 0.0, [],
        )
        _validate_expected_identity(regenerated, job)
        minimum_indices = validate_regenerated_metric(committed, regenerated)

        bucket = assign_metric_record(committed, bucket_spec)
        density = int(rollout["clutter_density"])
        if density != int(committed["obstacle_density"]):
            raise ValueError(f"episode {episode} rollout/metric density mismatch")
        foots = occluder_footprints_3d(env, obstacles=metric_cfg.obstacles)
        if foots is None:
            raise RuntimeError(f"episode {episode} could not reconstruct obstacle meshes")
        occ_ps = obstacle_centers(foots)
        target_p = np.asarray(roles.target.actor.get_pose().p, dtype=float)
        destination_p = np.asarray(roles.destination.actor.get_pose().p, dtype=float)
        # Function-level: the figure frame is the only consumer, and the module must stay
        # importable by the unit tests without pulling in the scene-generation stack.
        from bench_envs.utils.scene_gen_utils import get_actor_boundingbox
        table_bbox = get_actor_boundingbox(env.table)
        if table_bbox is None or table_bbox[0] is None:
            raise RuntimeError(f"episode {episode} could not measure the table bounds")
        plot_args = SimpleNamespace(
            seed=seed,
            arm=waypoints[0].arm,
            occ_shape=metric_cfg.occ_shape,
            gripper_r=float(metric_config["gripper_reference_radius_m"]),
        )
        # Every canonical leg is drawn, not just the binding one: which leg sets eps_geom_min is
        # only interpretable next to the legs it beat. `is_minimum` keeps that distinction in the
        # record so the binding leg stays identifiable without re-reading the metric.
        minimum_set = set(minimum_indices)
        minimum_kinds = [leg_specs[index].kind for index in minimum_indices]
        figures = []
        for index, (spec, result) in enumerate(zip(leg_specs, results)):
            is_minimum = index in minimum_set
            figure_path, exact = _atomic_route_plot(
                out_dir, plot_args, foots, occ_ps, result, spec.kind, target_p,
                episode, density, bucket, table_bbox=table_bbox,
                destination_p=destination_p, leg_index=index, leg_count=len(leg_specs),
                is_minimum=is_minimum, minimum_kinds=minimum_kinds,
            )
            figures.append(
                {
                    "leg_index": int(index),
                    "kind": spec.kind,
                    "is_minimum": is_minimum,
                    "figure_relpath": str(figure_path.relative_to(out_dir)),
                    "route_points": len(result.route_world or []),
                    "exact_mesh_surface_distance_m": (
                        None if exact is None else float(exact)
                    ),
                }
            )

        epsilon = _eps(committed, "eps_geom_min", "eps_geom_min_unbounded")
        radius = float(metric_config["gripper_reference_radius_m"])
        return {
            "schema": SCHEMA,
            "status": "ok",
            "visualization_config_sha256": visualization_config_sha256,
            "source_record_sha256": source_hashes,
            "episode": episode,
            "scene_id": committed["scene_id"],
            "scene_fingerprint": committed["scene_fingerprint"],
            "seed": seed,
            "clutter_density": density,
            "clutter_count": int(committed["clutter_count"]),
            "acting_arm": committed["arm"],
            "eps_geom_min": None if math.isinf(epsilon) else epsilon,
            "eps_geom_min_unbounded": math.isinf(epsilon),
            "rho_geom_min": None if math.isinf(epsilon) else epsilon / radius,
            "rho_geom_min_unbounded": math.isinf(epsilon),
            "clearance_bucket": bucket,
            "minimum_leg_indices": minimum_indices,
            "figures": figures,
            "hard_outcome": _hard_outcome(rollout),
            "video_relpath": rollout["video_relpath"],
            "regeneration_validated": True,
            "wall_seconds": float(time.perf_counter() - started),
        }
    finally:
        if env is not None:
            try:
                env.close_env()
            except Exception:
                pass
        del env
        gc.collect()


def run(args):
    rollout_dir = Path(args.rollout_run).resolve()
    if not rollout_dir.is_dir():
        raise FileNotFoundError(f"rollout run does not exist: {rollout_dir}")
    metric_dir = rollout_dir / "metric_postprocess"
    metric_config_path = metric_dir / "config.json"
    manifest_path = rollout_dir / "metric_scene_manifest.jsonl"
    rollout_config_path = rollout_dir / "config.json"
    for path in (metric_config_path, manifest_path, rollout_config_path):
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact is missing: {path}")

    rollout_config = _validate_embedded_hash(
        json.loads(rollout_config_path.read_text(encoding="utf-8")), rollout_config_path
    )
    metric_config = _validate_embedded_hash(
        json.loads(metric_config_path.read_text(encoding="utf-8")), metric_config_path
    )
    if metric_config.get("source_rollout", {}).get("config_sha256") != rollout_config["config_sha256"]:
        raise ValueError("metric config is not bound to the supplied rollout")
    if metric_config.get("scene_manifest", {}).get("sha256") != sha256_file(manifest_path):
        raise ValueError("metric config scene-manifest hash mismatch")

    _, jobs = _read_scene_manifest(manifest_path)
    target_n = int(metric_config.get("target_metrics", -1))
    if target_n <= 0 or target_n != len(jobs):
        raise ValueError("metric target count does not match the scene manifest")
    rollouts = read_episode_records(rollout_dir)
    if len(rollouts) != target_n:
        raise ValueError("rollout episode count does not match the metric target")
    rollout_by_episode = {int(row["episode"]): row for row in rollouts}
    metric_records = _read_committed_metric_records(
        metric_dir, jobs, metric_config["config_sha256"]
    )
    metric_by_episode = {
        int(row["rollout_episode"]): row for row in metric_records
    }

    bucket_spec_path = Path(metric_config["bucket_spec"]["path"]).resolve()
    if sha256_file(bucket_spec_path) != metric_config["bucket_spec"]["sha256"]:
        raise ValueError("frozen bucket-spec hash differs from the metric config")
    bucket_spec = load_bucket_spec(bucket_spec_path)
    metric_cfg = SeedMetricConfig(**metric_config["metric"])
    reach_cache_dir = Path(metric_config["reach_cache_dir"])
    if not reach_cache_dir.is_dir():
        raise FileNotFoundError(f"reach cache is missing: {reach_cache_dir}")

    selected = None
    if args.episode:
        selected = sorted(set(args.episode))
    elif args.stratified_per_cell is not None:
        selected = select_stratified(
            metric_records, rollout_by_episode, bucket_spec, args.stratified_per_cell
        )
    if selected is None:
        selected = select_first(target_n, args.first)
    for episode in selected:
        if episode < 0 or episode >= target_n:
            raise ValueError(f"selected episode is outside [0,{target_n}): {episode}")
        if episode not in metric_by_episode:
            raise ValueError(f"selected episode has no committed metric yet: {episode}")
        validate_source_pair(
            jobs[episode], rollout_by_episode[episode], metric_by_episode[episode]
        )
        _validate_source_video(rollout_dir, rollout_by_episode[episode])

    source_hashes_by_episode = {
        episode: _source_hashes(
            jobs[episode], rollout_by_episode[episode], metric_by_episode[episode]
        )
        for episode in metric_by_episode
    }

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir else rollout_dir / "metric_route_visuals"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    expected_config = _visual_config(
        rollout_dir, rollout_config, metric_config_path, metric_config,
        bucket_spec_path, target_n,
    )
    visual_config = _load_or_create_config(out_dir / "config.json", expected_config)
    records = read_visual_records(
        out_dir, visual_config["config_sha256"], target_n,
        source_hashes_by_episode,
    )
    completed = {int(row["episode"]) for row in records}
    selected_complete = len(completed & set(selected))
    print(
        f"[route-viz] selected={len(selected)} complete={selected_complete}/{len(selected)} "
        f"out={out_dir}"
    )

    started = time.perf_counter()
    rendered_this_run = 0
    try:
        for position, episode in enumerate(selected, 1):
            if episode in completed:
                print(f"[route-viz] {position}/{len(selected)} episode={episode} already complete")
                continue
            scene_started = time.perf_counter()
            record = _render_episode(
                jobs[episode], rollout_by_episode[episode], metric_by_episode[episode],
                metric_config, metric_cfg, bucket_spec, out_dir,
                visual_config["config_sha256"], source_hashes_by_episode[episode],
            )
            episode_path = out_dir / "episodes" / f"episode{episode:06d}.json"
            if episode_path.exists():
                raise FileExistsError(f"refusing to overwrite visualization record: {episode_path}")
            atomic_write_json(episode_path, record)
            records.append(record)
            records.sort(key=lambda row: int(row["episode"]))
            completed.add(episode)
            rendered_this_run += 1
            elapsed = time.perf_counter() - started
            mean = elapsed / rendered_this_run
            remaining = len([value for value in selected[position:] if value not in completed])
            print(
                f"[route-viz] {position}/{len(selected)} episode={episode} "
                f"figures={len(record['figures'])} scene={time.perf_counter()-scene_started:.1f}s "
                f"mean={mean:.1f}s eta={remaining*mean/60.0:.1f}min"
            )
            if rendered_this_run % args.report_every == 0:
                regenerate_indexes(out_dir, rollout_dir, records, selected, target_n)
    finally:
        records = read_visual_records(
            out_dir, visual_config["config_sha256"], target_n,
            source_hashes_by_episode,
        )
        regenerate_indexes(out_dir, rollout_dir, records, selected, target_n)
    selected_complete = len({int(row["episode"]) for row in records} & set(selected))
    print(
        f"[route-viz] done: {selected_complete}/{len(selected)} selected episodes visualized "
        f"({len(records)} total records in index)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay committed task-metric scenes and pair the existing 3D eps/path renderer "
            "with their rollout videos."
        )
    )
    parser.add_argument("--rollout-run", default=str(DEFAULT_ROLLOUT))
    parser.add_argument("--out-dir", help="default: <rollout-run>/metric_route_visuals")
    parser.add_argument(
        "--episode", type=int, action="append",
        help="render one episode; repeat for multiple episodes (allows an incomplete metric run)",
    )
    parser.add_argument(
        "--stratified-per-cell", type=int,
        help="render the first N committed episodes in every density x clearance-bucket cell",
    )
    parser.add_argument(
        "--first", type=int, default=50,
        help="render the first N rollout episodes when no other selector is given (default: 50)",
    )
    parser.add_argument("--report-every", type=int, default=10)
    args = parser.parse_args()
    if args.episode and args.stratified_per_cell is not None:
        parser.error("--episode and --stratified-per-cell are mutually exclusive")
    if args.stratified_per_cell is not None and args.stratified_per_cell <= 0:
        parser.error("--stratified-per-cell must be positive")
    if args.first <= 0:
        parser.error("--first must be positive")
    if args.report_every <= 0:
        parser.error("--report-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
