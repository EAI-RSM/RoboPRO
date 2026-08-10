"""Crash-tolerant summaries, tables, and figures for VLA rollout records."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plotting import save_figure_atomic

from lib.run_io import atomic_write_json, atomic_write_text


SUMMARY_SCHEMA = "robopro.vla-rollout-summary.v2"


def _validate_records(records, source):
    seen_episodes = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{source}: record {index} is not an object")
        episode = record.get("episode")
        if not isinstance(episode, int) or isinstance(episode, bool):
            raise ValueError(f"{source}: record {index} has invalid episode")
        if episode in seen_episodes:
            raise ValueError(f"{source}: duplicate episode {episode}")
        seen_episodes.add(episode)
        for field in ("task_success", "hard_success"):
            if not isinstance(record.get(field), bool):
                raise ValueError(f"{source}: episode {episode} has invalid {field}")
    records.sort(key=lambda row: row["episode"])
    expected = list(range(len(records)))
    actual = [record["episode"] for record in records]
    if actual != expected:
        raise ValueError(f"{source}: episodes must be contiguous from zero, got {actual[:10]}")
    return records


def read_episode_records(run_dir):
    """Read atomic per-episode files, falling back to legacy JSONL runs."""
    run_dir = Path(run_dir)
    episode_paths = sorted((run_dir / "episodes").glob("episode*.json"))
    if episode_paths:
        records = []
        for path in episode_paths:
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid atomic episode record {path}: {exc}") from exc
        return _validate_records(records, run_dir / "episodes")

    records_path = run_dir / "records.jsonl"
    if not records_path.is_file():
        return []
    records = []
    with records_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL record at {records_path}:{line_number}: {exc}"
                ) from exc
    return _validate_records(records, records_path)


def sync_records_jsonl(run_dir, records):
    """Rebuild the convenience JSONL view from authoritative episode files."""
    payload = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    atomic_write_text(Path(run_dir) / "records.jsonl", payload)


def metric_scene_manifest(records):
    """Strip committed rollout records to outcome-blind scene regeneration inputs."""
    required = (
        "episode", "seed", "replicate", "task", "bench_subdir", "base_config",
        "dr_settings", "clutter_density", "clutter_count", "scene_id",
        "scene_fingerprint", "scene_code_version", "checkpoint", "instruction",
        "acting_arm",
    )
    manifest = []
    seen_scene_ids = set()
    for record in records:
        missing = [name for name in required if record.get(name) is None]
        if missing:
            raise ValueError(
                f"episode {record.get('episode')} cannot produce a metric scene manifest; "
                f"missing {missing}"
            )
        scene_id = str(record["scene_id"])
        if scene_id in seen_scene_ids:
            raise ValueError(f"duplicate scene_id in metric scene manifest: {scene_id}")
        seen_scene_ids.add(scene_id)
        manifest.append(
            {
                "schema": "robopro.metric-scene-manifest.v1",
                "rollout_episode": int(record["episode"]),
                "seed": int(record["seed"]),
                "replicate": int(record["replicate"]),
                "task": str(record["task"]),
                "bench_subdir": str(record["bench_subdir"]),
                "base_config": str(record["base_config"]),
                "dr_settings": record["dr_settings"],
                "obstacle_density": int(record["clutter_density"]),
                "expected_clutter_count": int(record["clutter_count"]),
                "expected_scene_id": scene_id,
                "expected_scene_fingerprint": str(record["scene_fingerprint"]),
                "expected_scene_code_version": str(record["scene_code_version"]),
                "checkpoint": str(record["checkpoint"]),
                "instruction": str(record["instruction"]),
                "expected_acting_arm": str(record["acting_arm"]),
            }
        )
    return manifest


def write_metric_scene_manifest(path, records):
    rows = metric_scene_manifest(records)
    payload = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    atomic_write_text(path, payload)
    return rows


def wilson_interval(successes, n, z=1.959963984540054):
    if n == 0:
        return None, None
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    # Roundoff at the binomial endpoints can otherwise put the lower bound a
    # few ulps above p=0 (or the upper bound below p=1).  Besides being outside
    # the parameter space, that yields a tiny negative Matplotlib yerr.
    lower = max(0.0, min(p, center - half_width))
    upper = min(1.0, max(p, center + half_width))
    return lower, upper


def _density_order(records, config):
    configured = config.get("density_cycle", []) if isinstance(config, dict) else []
    observed = {int(record["clutter_density"]) for record in records}
    order = [int(value) for value in configured if int(value) in observed]
    order.extend(sorted(observed - set(order)))
    return order


def summarize(records, config=None):
    config = config or {}
    n = len(records)
    task_successes = sum(record["task_success"] for record in records)
    hard_successes = sum(record["hard_success"] for record in records)
    hard_low, hard_high = wilson_interval(hard_successes, n)
    by_density = {}
    for density in _density_order(records, config):
        rows = [row for row in records if int(row["clutter_density"]) == density]
        successes = sum(row["hard_success"] for row in rows)
        low, high = wilson_interval(successes, len(rows))
        by_density[str(density)] = {
            "n": len(rows),
            "n_hard_success": successes,
            "hard_success_rate": successes / len(rows),
            "hard_success_wilson_95": [low, high],
        }
    collision_episodes = sum(
        bool(record.get("collision_metrics", {}).get("is_collision"))
        for record in records
    )
    target = config.get("target_rollouts")
    return {
        "schema": SUMMARY_SCHEMA,
        "n_episodes": n,
        "target_rollouts": target,
        "collection_complete": isinstance(target, int) and n == target,
        "n_task_success": task_successes,
        "task_success_rate": task_successes / n if n else None,
        "n_hard_success": hard_successes,
        "hard_success_rate": hard_successes / n if n else None,
        "hard_success_wilson_95": [hard_low, hard_high],
        "hard_success_non_degenerate": len(
            {record["hard_success"] for record in records}
        ) > 1,
        "n_collision_episodes": collision_episodes,
        "n_policy_errors": sum(record.get("policy_error") is not None for record in records),
        "n_missing_videos": sum(record.get("video_relpath") is None for record in records),
        "records_by_density": dict(
            Counter(str(record["clutter_density"]) for record in records)
        ),
        "by_density": by_density,
    }


CSV_FIELDS = (
    "episode", "record_id", "seed", "replicate", "task", "bench_subdir",
    "base_config", "clutter_density", "clutter_count", "scene_id",
    "scene_fingerprint", "acting_arm", "task_success", "hard_success",
    "is_collision", "robot_to_furniture", "robot_to_static_object",
    "target_to_static_object", "total_collision_count", "steps_taken",
    "step_lim", "wall_seconds", "scene_setup_seconds", "policy_seconds",
    "policy_error", "failure_reason", "video_camera", "video_relpath",
    "instruction", "checkpoint", "scene_code_version", "run_config_sha256",
    "rollout_code_version", "final_xy_l2_error_m",
)


def _csv_row(record):
    collision = record.get("collision_metrics") or {}
    timing = record.get("timing_seconds") or {}
    final_state = record.get("final_state") or {}
    return {
        "episode": record.get("episode"),
        "record_id": record.get("record_id"),
        "seed": record.get("seed"),
        "replicate": record.get("replicate"),
        "task": record.get("task"),
        "bench_subdir": record.get("bench_subdir"),
        "base_config": record.get("base_config"),
        "clutter_density": record.get("clutter_density"),
        "clutter_count": record.get("clutter_count"),
        "scene_id": record.get("scene_id"),
        "scene_fingerprint": record.get("scene_fingerprint"),
        "acting_arm": record.get("acting_arm"),
        "task_success": record.get("task_success"),
        "hard_success": record.get("hard_success"),
        "is_collision": collision.get("is_collision"),
        "robot_to_furniture": collision.get("robot_to_furniture"),
        "robot_to_static_object": collision.get("robot_to_static_object"),
        "target_to_static_object": collision.get("target_to_static_object"),
        "total_collision_count": collision.get("total_collision_count"),
        "steps_taken": record.get("steps_taken"),
        "step_lim": record.get("step_lim"),
        "wall_seconds": record.get("wall_seconds"),
        "scene_setup_seconds": timing.get("scene_setup"),
        "policy_seconds": timing.get("policy"),
        "policy_error": record.get("policy_error"),
        "failure_reason": record.get("failure_reason"),
        "video_camera": record.get("video_camera"),
        "video_relpath": record.get("video_relpath"),
        "instruction": record.get("instruction"),
        "checkpoint": record.get("checkpoint"),
        "scene_code_version": record.get("scene_code_version"),
        "run_config_sha256": record.get("run_config_sha256"),
        "rollout_code_version": record.get("rollout_code_version"),
        "final_xy_l2_error_m": final_state.get("xy_l2_error_m"),
    }


def write_records_csv(path, records):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(_csv_row(record) for record in records)
    atomic_write_text(path, output.getvalue())


def plot_running_hsr(records, path):
    outcomes = np.asarray([record["hard_success"] for record in records], dtype=int)
    indices = np.arange(1, len(records) + 1)
    cumulative = np.cumsum(outcomes)
    rates = cumulative / indices
    bounds = np.asarray(
        [wilson_interval(int(successes), int(n)) for successes, n in zip(cumulative, indices)]
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(indices, rates, color="#1f77b4", linewidth=2.2, label="Cumulative HSR")
    ax.fill_between(
        indices, bounds[:, 0], bounds[:, 1], color="#1f77b4", alpha=0.18,
        label="95% Wilson interval",
    )
    ax.scatter(indices, outcomes, color=np.where(outcomes, "#2ca02c", "#d62728"),
               s=14, alpha=0.35, label="Episode outcome")
    ax.set(xlabel="Completed rollout", ylabel="Hard success rate", ylim=(-0.03, 1.03))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.set_title(f"Running hard success rate ({int(cumulative[-1])}/{len(records)})")
    save_figure_atomic(fig, path, bbox_inches="tight")


def plot_outcome_diagnostics(records, config, path):
    densities = _density_order(records, config)
    groups = [
        [record for record in records if int(record["clutter_density"]) == density]
        for density in densities
    ]
    successes = np.asarray([sum(record["hard_success"] for record in rows) for rows in groups])
    totals = np.asarray([len(rows) for rows in groups])
    rates = successes / totals
    intervals = np.asarray(
        [wilson_interval(int(success), int(total)) for success, total in zip(successes, totals)]
    )
    x = np.arange(len(densities))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(x, rates, color="#4c78a8")
    axes[0].errorbar(
        x,
        rates,
        yerr=np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates)),
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    axes[0].set_xticks(x, [f"d{density}\nn={total}" for density, total in zip(densities, totals)])
    axes[0].set(ylabel="Hard success rate", ylim=(0, 1.04), title="HSR by density")
    axes[0].grid(axis="y", alpha=0.25)

    step_groups = [
        [float(record["steps_taken"]) for record in rows]
        for rows in groups
    ]
    axes[1].boxplot(step_groups, tick_labels=[f"d{density}" for density in densities],
                    showfliers=True)
    axes[1].set(xlabel="Configured density", ylabel="Executed policy steps",
                title="Rollout length by density")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(f"Outcome diagnostics after {len(records)} completed rollouts")
    fig.tight_layout()
    save_figure_atomic(fig, path, bbox_inches="tight")


def write_rollout_reports(run_dir):
    """Regenerate all derived artifacts from committed raw episode records."""
    run_dir = Path(run_dir)
    records = read_episode_records(run_dir)
    config_path = run_dir / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    sync_records_jsonl(run_dir, records)
    manifest_path = run_dir / "metric_scene_manifest.jsonl"
    if not records or all(record.get("schema") == "robopro.vla-rollout.v3" for record in records):
        manifest = write_metric_scene_manifest(manifest_path, records)
    else:
        # Historical v1/v2 pilots predate exact scene identity. Keep their outcome
        # reports regenerable, but never manufacture a join manifest for them.
        manifest = []
        manifest_path.unlink(missing_ok=True)
    summary = summarize(records, config)
    atomic_write_json(run_dir / "summary.json", summary)
    write_records_csv(run_dir / "records.csv", records)
    if records:
        plot_running_hsr(records, run_dir / "running_hsr.png")
        plot_outcome_diagnostics(records, config, run_dir / "outcome_diagnostics.png")
    atomic_write_json(
        run_dir / "report_state.json",
        {
            "schema": "robopro.vla-rollout-report-state.v1",
            "records_in_report": len(records),
            "source": "atomic episode records",
            "derived_files": [
                "records.jsonl", "records.csv", "metric_scene_manifest.jsonl",
                "summary.json",
                "running_hsr.png", "outcome_diagnostics.png",
            ] if manifest or not records else [
                "records.jsonl", "records.csv", "summary.json",
                "running_hsr.png", "outcome_diagnostics.png",
            ],
            "metric_scene_manifest_records": len(manifest),
        },
    )
    return summary
