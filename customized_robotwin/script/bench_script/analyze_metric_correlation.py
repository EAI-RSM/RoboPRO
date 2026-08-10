#!/usr/bin/env python3
"""Strictly join task metrics to VLA outcomes and report their association."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from setup_paths import setup_paths

setup_paths()

from analyze_metric_distribution import _metric_value, read_metric_records
from lib.metric_buckets import assign_metric_record, load_bucket_spec
from lib.run_io import CLEARANCE_RESULTS_DIR, Timings, atomic_write_json, atomic_write_text
from lib.vla_reporting import read_episode_records, wilson_interval


RESULTS_DIR = CLEARANCE_RESULTS_DIR.parent / "task_metric_correlation"
DEFAULT_BUCKET_SPEC = Path(__file__).with_name("bucket_spec.json")
JOIN_SCHEMA = "robopro.task-metric-outcome-join.v1"
DENSITY_REPORT_ORDER = (6, 10, 15)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_all_metric_records(paths):
    sources = []
    records = []
    seen_ids = set()
    seen_fingerprints = set()
    for value in paths:
        source, rows = read_metric_records(value)
        sources.append(source)
        for row in rows:
            scene_id = row["scene_id"]
            fingerprint = row.get("scene_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError(f"metric scene {scene_id} lacks scene_fingerprint")
            if scene_id in seen_ids:
                raise ValueError(f"duplicate metric scene_id across inputs: {scene_id}")
            if fingerprint in seen_fingerprints:
                raise ValueError(
                    f"duplicate metric scene_fingerprint across inputs: {fingerprint}"
                )
            seen_ids.add(scene_id)
            seen_fingerprints.add(fingerprint)
            records.append(row)
    return sources, records


def _validated_hard_success(record):
    task_success = record.get("task_success")
    collision = (record.get("collision_metrics") or {}).get("is_collision")
    hard_success = record.get("hard_success")
    if not isinstance(task_success, bool) or not isinstance(collision, bool):
        raise ValueError(
            f"rollout episode {record.get('episode')} lacks Boolean task/collision outcome"
        )
    expected = bool(task_success and not collision)
    if not isinstance(hard_success, bool) or hard_success != expected:
        raise ValueError(
            f"rollout episode {record.get('episode')} has inconsistent hard_success"
        )
    return expected


def join_records(rollouts, metrics, bucket_spec, *, require_complete=True):
    """Join exact scenes; intermediate metrics may be a validated rollout subset."""
    rollout_by_id = {}
    rollout_fingerprints = set()
    for row in rollouts:
        scene_id = row.get("scene_id")
        fingerprint = row.get("scene_fingerprint")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"rollout episode {row.get('episode')} lacks scene_id")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"rollout episode {row.get('episode')} lacks scene_fingerprint")
        if scene_id in rollout_by_id:
            raise ValueError(f"duplicate rollout scene_id: {scene_id}")
        if fingerprint in rollout_fingerprints:
            raise ValueError(f"duplicate rollout scene_fingerprint: {fingerprint}")
        _validated_hard_success(row)
        rollout_by_id[scene_id] = row
        rollout_fingerprints.add(fingerprint)

    metric_by_id = {}
    metric_fingerprints = set()
    for row in metrics:
        scene_id = row.get("scene_id")
        fingerprint = row.get("scene_fingerprint")
        if scene_id in metric_by_id:
            raise ValueError(f"duplicate metric scene_id: {scene_id}")
        if fingerprint in metric_fingerprints:
            raise ValueError(f"duplicate metric scene_fingerprint: {fingerprint}")
        metric_by_id[scene_id] = row
        metric_fingerprints.add(fingerprint)
    missing_metrics = sorted(set(rollout_by_id) - set(metric_by_id))
    extra_metrics = sorted(set(metric_by_id) - set(rollout_by_id))
    if extra_metrics or (require_complete and missing_metrics):
        raise ValueError(
            "metric/outcome join is not one-to-one: "
            f"missing_metrics={missing_metrics[:10]}, extra_metrics={extra_metrics[:10]}"
        )

    joined = []
    selected = (
        rollout_by_id.items()
        if require_complete
        else ((scene_id, rollout_by_id[scene_id]) for scene_id in metric_by_id)
    )
    for scene_id, rollout in sorted(
        selected, key=lambda item: int(item[1]["episode"])
    ):
        metric = metric_by_id[scene_id]
        if metric["scene_fingerprint"] != rollout["scene_fingerprint"]:
            raise ValueError(f"scene fingerprint mismatch for {scene_id}")
        comparisons = {
            "task": (metric.get("task"), rollout.get("task")),
            "seed": (metric.get("seed"), rollout.get("seed")),
            "replicate": (metric.get("replicate"), rollout.get("replicate")),
            "bench_subdir": (metric.get("bench_subdir"), rollout.get("bench_subdir")),
            "base_config": (metric.get("base_config"), rollout.get("base_config")),
            "density": (metric.get("obstacle_density"), rollout.get("clutter_density")),
            "clutter_count": (metric.get("clutter_count"), rollout.get("clutter_count")),
            "checkpoint": (metric.get("checkpoint"), rollout.get("checkpoint")),
            "scene_code_version": (
                metric.get("scene_code_version"), rollout.get("scene_code_version")
            ),
            "instruction": (metric.get("instruction"), rollout.get("instruction")),
            "acting_arm": (metric.get("arm"), rollout.get("acting_arm")),
        }
        mismatches = {
            key: {"metric": left, "rollout": right}
            for key, (left, right) in comparisons.items()
            if left != right
        }
        metric_episode = metric.get("rollout_episode")
        if metric_episode is not None and int(metric_episode) != int(rollout["episode"]):
            mismatches["rollout_episode"] = {
                "metric": metric_episode,
                "rollout": rollout["episode"],
            }
        if mismatches:
            raise ValueError(f"joined provenance mismatch for {scene_id}: {mismatches}")

        radius = float(metric.get("gripper_reference_radius_m", -1.0))
        expected_radius = float(bucket_spec["bucket_variable"]["reference_radius_m"])
        if not np.isclose(radius, expected_radius, rtol=0.0, atol=1e-12):
            raise ValueError(f"gripper reference radius mismatch for {scene_id}")
        eps = float(_metric_value(metric, "eps_geom_min", "eps_geom_min_unbounded"))
        bucket = assign_metric_record(metric, bucket_spec)
        hard_success = _validated_hard_success(rollout)
        joined.append(
            {
                "schema": JOIN_SCHEMA,
                "episode": int(rollout["episode"]),
                "scene_id": scene_id,
                "scene_fingerprint": rollout["scene_fingerprint"],
                "seed": int(rollout["seed"]),
                "replicate": int(rollout["replicate"]),
                "task": rollout["task"],
                "clutter_density": int(rollout["clutter_density"]),
                "clutter_count": int(rollout["clutter_count"]),
                "acting_arm": rollout["acting_arm"],
                "eps_geom_min": None if np.isposinf(eps) else eps,
                "eps_geom_min_unbounded": bool(np.isposinf(eps)),
                "rho_geom_min": None if np.isposinf(eps) else eps / expected_radius,
                "rho_geom_min_unbounded": bool(np.isposinf(eps)),
                "clearance_bucket": bucket,
                "task_success": bool(rollout["task_success"]),
                "is_collision": bool(rollout["collision_metrics"]["is_collision"]),
                "hard_success": hard_success,
                "steps_taken": int(rollout["steps_taken"]),
                "video_relpath": rollout.get("video_relpath"),
                "legs": metric["legs"],
                "metric_record": metric,
                "rollout_record": rollout,
            }
        )
    return joined


def _eps(row):
    return np.inf if row["eps_geom_min_unbounded"] else float(row["eps_geom_min"])


def spearman_binary(rows):
    if len(rows) < 2:
        return None
    x = np.asarray([_eps(row) for row in rows], dtype=float)
    y = np.asarray([row["hard_success"] for row in rows], dtype=int)
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    result = spearmanr(x, y)
    return float(result.statistic) if np.isfinite(result.statistic) else None


def bootstrap_spearman(rows, *, resamples, seed):
    point = spearman_binary(rows)
    if int(resamples) == 0:
        return {
            "spearman_rho": point,
            "bootstrap_95": [None, None],
            "bootstrap_resamples": 0,
            "bootstrap_seed": int(seed),
            "valid_bootstrap_resamples": 0,
            "valid_bootstrap_fraction": None,
            "undefined_resamples_are_not_zero": True,
            "deferred_until_complete": True,
        }
    rng = np.random.default_rng(seed)
    valid = []
    n = len(rows)
    for _ in range(int(resamples)):
        sample = [rows[index] for index in rng.integers(0, n, size=n)]
        value = spearman_binary(sample)
        if value is not None:
            valid.append(value)
    interval = (
        [float(v) for v in np.quantile(valid, [0.025, 0.975])]
        if valid else [None, None]
    )
    return {
        "spearman_rho": point,
        "bootstrap_95": interval,
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
        "valid_bootstrap_resamples": len(valid),
        "valid_bootstrap_fraction": len(valid) / int(resamples),
        "undefined_resamples_are_not_zero": True,
    }


def _hsr_summary(rows):
    n = len(rows)
    successes = sum(row["hard_success"] for row in rows)
    low, high = wilson_interval(successes, n)
    return {
        "n": n,
        "n_hard_success": successes,
        "hard_success_rate": successes / n if n else None,
        "hard_success_wilson_95": [low, high],
    }


def summarize(
    joined,
    bucket_spec,
    *,
    resamples,
    seed,
    require_complete=True,
    target_n=None,
    densities=None,
):
    bucket_order = bucket_spec["bucket_order"]
    observed = set(
        int(value) for value in (
            densities
            if densities is not None
            else {row["clutter_density"] for row in joined}
        )
    )
    density_order = [value for value in DENSITY_REPORT_ORDER if value in observed]
    density_order.extend(sorted(observed - set(density_order)))
    pooled = {
        **_hsr_summary(joined),
        "association": bootstrap_spearman(joined, resamples=resamples, seed=seed),
        "by_clearance_bucket": {
            name: _hsr_summary(
                [row for row in joined if row["clearance_bucket"] == name]
            )
            for name in bucket_order
        },
        "bucket_counts": dict(Counter(row["clearance_bucket"] for row in joined)),
        "positive_infinity_n": sum(row["eps_geom_min_unbounded"] for row in joined),
        "interpretation": (
            "Secondary only: pooling mixes clutter density with within-density "
            "scene variation."
        ),
    }
    summary = {
        "schema": "robopro.task-metric-correlation-summary.v2",
        "n_joined": len(joined),
        "strict_one_to_one_join": bool(require_complete),
        "provisional": not require_complete,
        "target_n": int(target_n) if target_n is not None else len(joined),
        "processing_complete": bool(
            require_complete
            and (target_n is None or len(joined) == int(target_n))
        ),
        "exact_fingerprint_match": True,
        "primary_predictor": "eps_geom_min",
        "primary_outcome": "hard_success = task_success and not is_collision",
        "analysis_hierarchy": {
            "primary": "separate association analyses by clutter density",
            "primary_density_order": density_order,
            "secondary": "pooled association across clutter densities",
        },
        "by_density": {},
        "secondary_pooled": pooled,
        "positive_infinity_treatment": "shared top rank and high_clearance bucket",
        "sample_is_expert_independent": True,
        "visibility_measured_or_adjusted": False,
        "pooled_density_interpretation": (
            "The pooled association mixes density and placement; use per-density results "
            "for density-specific interpretation."
        ),
        "approximation_bias_direction": "unknown",
    }
    for density in density_order:
        rows = [row for row in joined if row["clutter_density"] == density]
        summary["by_density"][str(density)] = {
            **_hsr_summary(rows),
            "clutter_density": density,
            "association": bootstrap_spearman(
                rows, resamples=resamples, seed=seed + int(density)
            ),
            "by_clearance_bucket": {
                name: _hsr_summary(
                    [row for row in rows if row["clearance_bucket"] == name]
                )
                for name in bucket_order
            },
            "bucket_counts": dict(Counter(row["clearance_bucket"] for row in rows)),
            "positive_infinity_n": sum(row["eps_geom_min_unbounded"] for row in rows),
        }
    return summary


def _save_figure(fig, path):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    fig.savefig(temporary, dpi=160, bbox_inches="tight")
    os.replace(temporary, path)
    plt.close(fig)


def plot_bucket_hsr(summary, bucket_spec, path, *, title):
    names = bucket_spec["bucket_order"]
    labels = [name.replace("_clearance", "").replace("_", "\n") for name in names]
    rows = [summary["by_clearance_bucket"][name] for name in names]
    rates = np.asarray([row["hard_success_rate"] or 0.0 for row in rows])
    intervals = np.asarray(
        [row["hard_success_wilson_95"] if row["n"] else [0.0, 0.0] for row in rows]
    )
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, rates, color="#4c78a8")
    ax.errorbar(
        x, rates,
        yerr=np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates)),
        fmt="none", ecolor="black", capsize=5,
    )
    ax.set_xticks(x, [f"{label}\nn={row['n']}" for label, row in zip(labels, rows)])
    ax.set(ylabel="Hard success rate", ylim=(0, 1.04), title=title)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def plot_metric_by_outcome(joined, path, *, title):
    finite = [_eps(row) for row in joined if not row["eps_geom_min_unbounded"]]
    top = (max(finite) if finite else 0.0) + 0.01
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for outcome, color, label in ((False, "#d62728", "Hard fail"), (True, "#2ca02c", "Hard success")):
        rows = [row for row in joined if row["hard_success"] is outcome]
        x = [top if row["eps_geom_min_unbounded"] else _eps(row) for row in rows]
        y = np.full(len(rows), int(outcome), dtype=float) + rng.uniform(-0.08, 0.08, len(rows))
        ax.scatter(x, y, s=24, alpha=0.55, color=color, label=f"{label} (n={len(rows)})")
    ax.set_yticks([0, 1], ["Hard fail", "Hard success"])
    ax.set(xlabel="eps_geom_min (m; +inf shown at right)", ylim=(-0.25, 1.25),
           title=title)
    if any(row["eps_geom_min_unbounded"] for row in joined):
        ax.axvline(top, color="0.4", linestyle="--", linewidth=1)
        ax.text(top, 1.18, "+inf", ha="center")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")
    _save_figure(fig, path)


def write_joined_records(out_dir, joined):
    jsonl = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in joined)
    atomic_write_text(Path(out_dir) / "joined_records.jsonl", jsonl)
    fields = (
        "episode", "scene_id", "scene_fingerprint", "seed", "replicate", "task",
        "clutter_density", "clutter_count", "acting_arm", "eps_geom_min",
        "eps_geom_min_unbounded", "rho_geom_min", "rho_geom_min_unbounded",
        "clearance_bucket", "task_success", "is_collision", "hard_success",
        "steps_taken", "video_relpath",
    )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field) for field in fields} for row in joined)
    atomic_write_text(Path(out_dir) / "joined_records.csv", output.getvalue())


def index_videos(out_dir, rollout_dir, joined):
    index = []
    for row in joined:
        relative = row.get("video_relpath")
        source = Path(rollout_dir) / relative if relative else None
        outcome = "hard_success" if row["hard_success"] else "hard_fail"
        link = (
            Path(out_dir) / "videos_by_clearance" / row["clearance_bucket"] / outcome
            / f"episode{row['episode']:06d}_seed{row['seed']}.mp4"
        )
        linked = bool(source is not None and source.is_file() and source.stat().st_size > 0)
        if linked:
            link.parent.mkdir(parents=True, exist_ok=True)
            expected = os.path.relpath(source.resolve(), link.parent.resolve())
            if os.path.lexists(link):
                if not link.is_symlink() or os.readlink(link) != expected:
                    raise ValueError(f"existing video index entry is inconsistent: {link}")
            else:
                link.symlink_to(expected)
        index.append(
            {
                "episode": row["episode"],
                "scene_id": row["scene_id"],
                "clutter_density": row["clutter_density"],
                "clearance_bucket": row["clearance_bucket"],
                "hard_outcome": outcome,
                "source_video": relative,
                "index_link": str(link.relative_to(out_dir)) if linked else None,
                "video_available": linked,
            }
        )
    atomic_write_json(Path(out_dir) / "video_index.json", index)
    return index


def write_correlation_reports(
    out_dir,
    rollout_dir,
    metric_records,
    bucket_spec_path,
    *,
    require_complete,
    bootstrap_resamples,
    bootstrap_seed,
):
    """Regenerate one fixed analysis directory from committed source records."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = Path(rollout_dir).resolve()
    bucket_spec_path = Path(bucket_spec_path).resolve()
    rollouts = read_episode_records(rollout_dir)
    if not rollouts:
        raise ValueError(f"no rollout records in {rollout_dir}")
    metric_sources, metrics = read_all_metric_records(metric_records)
    bucket_spec = load_bucket_spec(bucket_spec_path)
    joined = join_records(
        rollouts, metrics, bucket_spec, require_complete=require_complete
    )
    summary = summarize(
        joined,
        bucket_spec,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        require_complete=require_complete,
        target_n=len(rollouts),
        densities={row["clutter_density"] for row in rollouts},
    )
    write_joined_records(out_dir, joined)
    primary_reports = []
    for density in summary["analysis_hierarchy"]["primary_density_order"]:
        key = str(density)
        rows = [row for row in joined if row["clutter_density"] == density]
        density_dir = out_dir / "by_density" / f"d{density}"
        density_dir.mkdir(parents=True, exist_ok=True)
        write_joined_records(density_dir, rows)
        atomic_write_json(
            density_dir / "summary.json",
            {
                "schema": "robopro.task-metric-density-summary.v1",
                "analysis_role": "primary",
                **summary["by_density"][key],
            },
        )
        plot_bucket_hsr(
            summary["by_density"][key],
            bucket_spec,
            density_dir / "hsr_by_clearance_bucket.png",
            title=f"d{density}: hard success rate by frozen clearance bucket",
        )
        plot_metric_by_outcome(
            rows,
            density_dir / "eps_geom_by_outcome.png",
            title=f"d{density}: episode outcomes across geometric clearance",
        )
        primary_reports.append(
            {
                "clutter_density": density,
                "directory": str(density_dir.relative_to(out_dir)),
                "n": len(rows),
            }
        )
    plot_bucket_hsr(
        summary["secondary_pooled"],
        bucket_spec,
        out_dir / "hsr_by_clearance_bucket.png",
        title="Secondary pooled: HSR by clearance bucket across all densities",
    )
    plot_metric_by_outcome(
        joined,
        out_dir / "eps_geom_by_outcome.png",
        title="Secondary pooled: episode outcomes across all densities",
    )
    atomic_write_json(
        out_dir / "report_index.json",
        {
            "schema": "robopro.task-metric-report-index.v1",
            "primary_reports": primary_reports,
            "primary_density_order": [
                row["clutter_density"] for row in primary_reports
            ],
            "secondary_pooled": {
                "summary": "association_summary.json:secondary_pooled",
                "hsr_plot": "hsr_by_clearance_bucket.png",
                "outcome_plot": "eps_geom_by_outcome.png",
            },
        },
    )
    video_index = index_videos(out_dir, rollout_dir, joined)
    summary["source"] = {
        "rollout_run": str(rollout_dir),
        "metric_records": [
            {"path": str(path), "sha256": _sha256(path)} for path in metric_sources
        ],
        "bucket_spec": {
            "path": str(bucket_spec_path),
            "sha256": _sha256(bucket_spec_path),
        },
    }
    summary["video_index"] = {
        "available": sum(row["video_available"] for row in video_index),
        "missing": sum(not row["video_available"] for row in video_index),
        "organization": "clearance_bucket / hard_outcome",
        "non_destructive_symlinks": True,
    }
    atomic_write_json(out_dir / "association_summary.json", summary)
    atomic_write_json(
        out_dir / "report_state.json",
        {
            "schema": "robopro.task-metric-postprocess-report-state.v1",
            "metrics_in_report": len(joined),
            "target_metrics": len(rollouts),
            "processing_complete": summary["processing_complete"],
            "provisional": summary["provisional"],
            "bootstrap_deferred_until_complete": bootstrap_resamples == 0,
            "primary_density_order": summary["analysis_hierarchy"][
                "primary_density_order"
            ],
            "pooled_analysis_is_secondary": True,
        },
    )
    if require_complete and summary["video_index"]["missing"]:
        raise ValueError(
            "final clearance video index is incomplete: "
            f"{summary['video_index']['missing']} source video(s) missing"
        )
    return summary


def run(args):
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    timings = Timings()
    with timings.section("complete_join_statistics_and_artifacts"):
        summary = write_correlation_reports(
            out_dir,
            args.rollout_run,
            args.metric_records,
            args.bucket_spec,
            require_complete=True,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    timings.save(out_dir)
    density_rho = {
        f"d{density}": summary["by_density"][str(density)]["association"][
            "spearman_rho"
        ]
        for density in summary["analysis_hierarchy"]["primary_density_order"]
    }
    print(
        f"[analysis] joined {summary['n_joined']} episodes; "
        f"primary_rho={density_rho}; "
        f"secondary_pooled_rho={summary['secondary_pooled']['association']['spearman_rho']} "
        f"-> {out_dir}"
    )
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-run", required=True)
    parser.add_argument("--metric-records", nargs="+", required=True)
    parser.add_argument("--bucket-spec", default=str(DEFAULT_BUCKET_SPEC))
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")
    run(args)


if __name__ == "__main__":
    main()
