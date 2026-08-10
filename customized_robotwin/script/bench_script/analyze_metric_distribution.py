#!/usr/bin/env python3
"""Visualize metric-only eps_geom distributions before any bucket choice.

The input must be Stage C task_metric records. Outcome, rollout, collision, success, and HSR
fields are rejected. This script never creates or recommends analysis buckets.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib.run_io import (
    CLEARANCE_RESULTS_DIR,
    Timings,
    atomic_write_json,
)


RESULTS_DIR = CLEARANCE_RESULTS_DIR.parent / "task_metric_distribution"
FORBIDDEN_OUTCOME_KEYS = {
    "success",
    "task_success",
    "hard_success",
    "hsr",
    "collision_metrics",
    "is_collision",
    "outcome",
    "rollout",
}
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden_path(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in FORBIDDEN_OUTCOME_KEYS:
                return path
            found = _forbidden_path(item, path)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _forbidden_path(item, f"{prefix}[{index}]")
            if found:
                return found
    return None


def read_metric_records(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"metric records not found: {path}")
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            forbidden = _forbidden_path(row)
            if forbidden:
                raise ValueError(
                    f"outcome-bearing field {forbidden!r} found at {path}:{line_number}; "
                    "Stage C2 accepts metric-only records"
                )
            if row.get("schema") != "robopro.task-metric.v1" or row.get("status") != "ok":
                raise ValueError(f"not a complete Stage C metric record at {path}:{line_number}")
            scene_id = row.get("scene_id")
            if not scene_id or scene_id in seen:
                raise ValueError(f"missing or duplicate scene_id at {path}:{line_number}")
            seen.add(scene_id)
            _metric_value(row, "eps_geom_min", "eps_geom_min_unbounded")
            if not isinstance(row.get("legs"), list) or not row["legs"]:
                raise ValueError(f"missing canonical leg vector at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no metric records in {path}")
    return path, rows


def _metric_value(row, value_key, inf_key):
    value = row.get(value_key)
    is_inf = row.get(inf_key)
    if is_inf is True:
        if value is not None:
            raise ValueError(f"{value_key} must be null when {inf_key}=true")
        return np.inf
    if is_inf is not False:
        raise ValueError(f"{inf_key} must be an explicit Boolean")
    if value is None:
        raise ValueError(f"finite {value_key} is null")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"finite {value_key} is nonfinite")
    return value


def _series_summary(values):
    values = np.asarray(values, dtype=float)
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError("metric series contains NaN or -inf")
    finite = values[np.isfinite(values)]
    counts = Counter(float(value) if np.isfinite(value) else "+inf" for value in values)
    tie_groups = [count for count in counts.values() if count > 1]
    summary = {
        "n": int(values.size),
        "finite_n": int(finite.size),
        "positive_infinity_n": int(np.isposinf(values).sum()),
        "unique_count_including_positive_infinity": len(counts),
        "finite_unique_count": int(np.unique(finite).size),
        "tie_group_count": len(tie_groups),
        "tied_observation_count": int(sum(tie_groups)),
        "finite_min": float(finite.min()) if finite.size else None,
        "finite_max": float(finite.max()) if finite.size else None,
        "descriptive_finite_quantiles": {},
    }
    if finite.size:
        summary["descriptive_finite_quantiles"] = {
            f"q{int(round(q * 100)):02d}": float(np.quantile(finite, q))
            for q in QUANTILES
        }
    return summary


def summarize_records(records):
    primary = [
        _metric_value(row, "eps_geom_min", "eps_geom_min_unbounded")
        for row in records
    ]
    labels = [(leg["index"], leg["kind"]) for leg in records[0]["legs"]]
    if len(labels) != len(set(labels)):
        raise ValueError("canonical leg labels are not unique")
    per_leg = {}
    for expected_index, expected_kind in labels:
        values = []
        for row in records:
            matches = [
                leg for leg in row["legs"]
                if leg.get("index") == expected_index and leg.get("kind") == expected_kind
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"scene {row['scene_id']} does not have exactly one "
                    f"leg {expected_index}:{expected_kind}"
                )
            values.append(_metric_value(matches[0], "eps_geom", "eps_geom_unbounded"))
        per_leg[f"{expected_index:02d}_{expected_kind}"] = _series_summary(values)

    summary = _series_summary(primary)
    summary.update(
        {
            "metric": "eps_geom_min",
            "quantiles_are_descriptive_only": True,
            "per_leg": per_leg,
            "realized_clutter_count_frequencies": {
                str(key): int(value)
                for key, value in sorted(Counter(int(r["clutter_count"]) for r in records).items())
            },
        }
    )
    return summary


def _primary_values(records):
    return np.asarray(
        [_metric_value(r, "eps_geom_min", "eps_geom_min_unbounded") for r in records],
        dtype=float,
    )


def _save_figure(fig, out_path, *, bbox_inches=None):
    """Atomically replace a derived plot so interruption cannot expose a partial PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(
        f".{out_path.stem}.{os.getpid()}.tmp{out_path.suffix}"
    )
    fig.savefig(temporary, dpi=160, bbox_inches=bbox_inches)
    os.replace(temporary, out_path)
    plt.close(fig)


def plot_primary_distribution(records, out_path, *, scope_label=None):
    values = _primary_values(records)
    finite = np.sort(values[np.isfinite(values)])
    inf_n = int(np.isposinf(values).sum())
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [2, 2, 0.8]})
    if finite.size:
        axes[0].hist(finite, bins="auto", color="#2878b5", edgecolor="white")
        axes[0].plot(finite, np.zeros_like(finite), "|", color="#14213d", markersize=12)
        y = np.arange(1, finite.size + 1) / finite.size
        axes[1].step(finite, y, where="post", color="#d1495b", linewidth=2.5)
        axes[1].plot(finite, np.zeros_like(finite), "|", color="#14213d", markersize=12)
    else:
        axes[0].text(0.5, 0.5, "No finite values", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No finite values", ha="center", va="center", transform=axes[1].transAxes)
    axes[0].set(title="Finite-value histogram (display only)", xlabel="eps_geom_min (m)", ylabel="Scenes")
    axes[1].set(title="Finite-value ECDF and rug", xlabel="eps_geom_min (m)", ylabel="ECDF", ylim=(-0.04, 1.04))
    axes[2].bar(["+inf"], [inf_n], color="#f4a261")
    axes[2].set(title="Top-censored", ylabel="Scenes", ylim=(0, max(1, inf_n) * 1.2))
    axes[2].text(0, inf_n, str(inf_n), ha="center", va="bottom", fontsize=13)
    title = "Metric-only eps_geom_min distribution — no outcomes loaded"
    if scope_label:
        title = f"{scope_label}: {title}"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    _save_figure(fig, out_path)


def plot_by_leg(records, out_path, *, scope_label=None):
    labels = [(leg["index"], leg["kind"]) for leg in records[0]["legs"]]
    series = []
    for index, kind in labels:
        values = []
        for row in records:
            leg = next(leg for leg in row["legs"] if leg["index"] == index and leg["kind"] == kind)
            values.append(_metric_value(leg, "eps_geom", "eps_geom_unbounded"))
        series.append((f"{index}: {kind}", np.asarray(values, dtype=float)))
    finite_all = np.concatenate([values[np.isfinite(values)] for _, values in series])
    fig, axes = plt.subplots(len(series), 1, figsize=(14, 3.2 * len(series)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (label, values) in zip(axes, series):
        finite = values[np.isfinite(values)]
        if finite.size:
            ax.hist(finite, bins="auto", color="#5aa469", alpha=0.8, edgecolor="white")
            ax.plot(finite, np.zeros_like(finite), "|", color="#14213d", markersize=12)
        else:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel(label)
        ax.text(0.99, 0.88, f"+inf: {int(np.isposinf(values).sum())}", ha="right", va="top", transform=ax.transAxes, fontsize=12)
    if finite_all.size:
        pad = max(0.005, 0.04 * max(float(np.ptp(finite_all)), 0.01))
        axes[-1].set_xlim(float(finite_all.min()) - pad, float(finite_all.max()) + pad)
    axes[-1].set_xlabel("eps_geom by leg (m); +inf counts shown separately")
    title = "Aligned canonical-leg distributions"
    if scope_label:
        title = f"{scope_label}: {title}"
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    _save_figure(fig, out_path)


def plot_by_scene(records, out_path, *, scope_label=None):
    values = _primary_values(records)
    order = sorted(range(len(records)), key=lambda i: (np.isposinf(values[i]), values[i], records[i]["scene_id"]))
    sorted_values = values[order]
    finite = sorted_values[np.isfinite(sorted_values)]
    if finite.size:
        span = max(float(np.ptp(finite)), 0.01)
        inf_level = float(finite.max()) + 0.12 * span
    else:
        inf_level = 1.0
    y = np.where(np.isposinf(sorted_values), inf_level, sorted_values)
    # The pilot plot labeled every point and widened by 0.55 inch per scene.  At
    # the declared n=3000 that would request a 1650-inch image.  Keep the full
    # sorted point series visible, but reserve per-point labels for small runs;
    # exact scene identities remain in records.jsonl/joined_records.csv.
    width = min(28, max(14, 0.035 * len(records)))
    fig, ax = plt.subplots(figsize=(width, 8))
    colors = ["#f4a261" if np.isposinf(v) else "#3a86ff" for v in sorted_values]
    ax.scatter(np.arange(len(records)), y, c=colors, s=60, zorder=3)
    if len(records) <= 100:
        for xpos, row_index in enumerate(order):
            row = records[row_index]
            text = (
                f"seed={row['seed']} | {row['scene_id']} | "
                f"d={row['obstacle_density']} | clutter={row['clutter_count']}"
            )
            ax.annotate(
                text, (xpos, y[xpos]), xytext=(0, 7),
                textcoords="offset points", rotation=90,
                ha="center", va="bottom", fontsize=8,
            )
    if np.isposinf(sorted_values).any():
        ax.axhline(inf_level, color="#f4a261", linestyle="--", alpha=0.65)
        ticks = list(ax.get_yticks()) + [inf_level]
        labels = [f"{tick:.3g}" for tick in ax.get_yticks()] + ["+inf"]
        ax.set_yticks(ticks, labels)
    title = "Sorted metric-only scenes"
    if scope_label:
        title = f"{scope_label}: {title}"
    if len(records) > 100:
        title += " (exact scene labels in records.jsonl)"
    ax.set(title=title, xlabel="Scenes sorted by eps_geom_min", ylabel="eps_geom_min (m)")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, out_path, bbox_inches="tight")


def write_distribution_reports(out_dir, records, provenance):
    """Regenerate metric-only summaries and plots in one fixed directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_records(records)
    observed = {int(row["obstacle_density"]) for row in records}
    density_order = [value for value in (6, 10, 15) if value in observed]
    density_order.extend(sorted(observed - set(density_order)))
    multiple_densities = len(density_order) > 1
    if multiple_densities:
        summary["analysis_hierarchy"] = {
            "primary": "separate metric distributions by clutter density",
            "primary_density_order": density_order,
            "secondary": "pooled metric distribution across clutter densities",
        }
        summary["by_density"] = {}
        for density in density_order:
            rows = [
                row for row in records
                if int(row["obstacle_density"]) == density
            ]
            density_dir = out_dir / "by_density" / f"d{density}"
            density_dir.mkdir(parents=True, exist_ok=True)
            density_summary = summarize_records(rows)
            density_summary.update(
                {"analysis_role": "primary", "clutter_density": density}
            )
            summary["by_density"][str(density)] = density_summary
            density_provenance = {
                **provenance,
                "record_count": len(rows),
                "clutter_density_filter": density,
            }
            density_summary["source_records"] = density_provenance
            atomic_write_json(
                density_dir / "distribution_summary.json", density_summary
            )
            atomic_write_json(
                density_dir / "source_records.json", density_provenance
            )
            plot_primary_distribution(
                rows,
                density_dir / "eps_geom_min_distribution.png",
                scope_label=f"d{density}",
            )
            plot_by_leg(
                rows,
                density_dir / "eps_geom_by_leg.png",
                scope_label=f"d{density}",
            )
            plot_by_scene(
                rows,
                density_dir / "eps_geom_min_by_scene.png",
                scope_label=f"d{density}",
            )
    summary["source_records"] = provenance
    atomic_write_json(out_dir / "distribution_summary.json", summary)
    atomic_write_json(out_dir / "source_records.json", provenance)
    pooled_label = "Secondary pooled" if multiple_densities else None
    plot_primary_distribution(
        records, out_dir / "eps_geom_min_distribution.png", scope_label=pooled_label
    )
    plot_by_leg(records, out_dir / "eps_geom_by_leg.png", scope_label=pooled_label)
    plot_by_scene(
        records, out_dir / "eps_geom_min_by_scene.png", scope_label=pooled_label
    )
    return summary


def run(args):
    timings = Timings()
    with timings.section("read_and_validate_metric_records"):
        records_path, records = read_metric_records(args.records)
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    source_config = records_path.parent / "config.json"
    provenance = {
        "records_path": str(records_path),
        "records_sha256": _sha256(records_path),
        "record_count": len(records),
        "source_config_path": str(source_config.resolve()) if source_config.is_file() else None,
        "source_config_sha256": _sha256(source_config) if source_config.is_file() else None,
        "outcome_data_loaded": False,
    }
    with timings.section("summarize"):
        write_distribution_reports(out_dir, records, provenance)
    timings.save(out_dir)
    print(f"[run] metric-only distribution artifacts: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", help="Stage C records.jsonl file or its run directory")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
