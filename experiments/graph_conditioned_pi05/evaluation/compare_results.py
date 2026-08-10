#!/usr/bin/env python3
"""Compare paired visual-only and retrieved-graph evaluation results."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any


MAIN_METRICS = (
    ("success_rate", "success", "rate"),
    ("hard_success_rate", "hard_success", "rate"),
    ("collision_rate", "collision", "rate"),
    ("mean_collision_count", "collision_count", "mean"),
)


def resolve_episode_logs(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path]
    direct_log = path / "_episodes.jsonl"
    if direct_log.is_file():
        return [direct_log]
    logs = sorted(path.rglob("_episodes.jsonl")) if path.is_dir() else []
    if not logs:
        raise FileNotFoundError(f"no _episodes.jsonl files found under: {path}")
    return logs


def load_records(path: Path) -> tuple[Path, dict[int, dict[str, Any]]]:
    source_path = path.expanduser().resolve()
    records: dict[int, dict[str, Any]] = {}
    for log_path in resolve_episode_logs(path):
        lines = log_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "seed" not in record:
                raise ValueError(f"{log_path}:{line_number}: missing seed")
            seed = int(record["seed"])
            if seed in records:
                raise ValueError(f"{log_path}:{line_number}: duplicate seed {seed}")
            records[seed] = record
    if not records:
        raise ValueError(f"episode logs are empty under: {source_path}")
    return source_path, records


def metric(records: list[dict[str, Any]], field: str, operation: str) -> float | None:
    values = [record.get(field) for record in records if record.get(field) is not None]
    if not values:
        return None
    if operation == "rate":
        return mean(bool(value) for value in values)
    return mean(float(value) for value in values)


def collision_names(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(set(map(str, record.get("collision_names", []))))
    return counts


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episodes": len(records)}
    for name, field, operation in MAIN_METRICS:
        summary[name] = metric(records, field, operation)
    summary["total_collision_count"] = sum(
        int(record.get("collision_count") or 0) for record in records
    )
    summary["collision_name_episode_counts"] = dict(
        sorted(collision_names(records).items())
    )
    return summary


def fmt(value: float | int | None, rate: bool = False) -> str:
    if value is None:
        return "n/a"
    if rate:
        return f"{100 * float(value):.1f}%"
    return f"{float(value):.2f}"


def print_report(
    visual_path: Path,
    graph_path: Path,
    seeds: list[int],
    visual: list[dict[str, Any]],
    graph: list[dict[str, Any]],
) -> None:
    visual_summary, graph_summary = summarize(visual), summarize(graph)
    print(f"visual_only:            {visual_path}")
    print(f"visual_retrieved_graph: {graph_path}")
    print(f"paired seeds ({len(seeds)}): {', '.join(map(str, seeds))}")
    print()
    print(f"{'metric':<24} {'visual_only':>14} {'retrieved_graph':>16} {'delta':>12}")
    print("-" * 70)
    for name, _, operation in MAIN_METRICS:
        left, right = visual_summary[name], graph_summary[name]
        delta = None if left is None or right is None else right - left
        is_rate = operation == "rate"
        print(
            f"{name:<24} {fmt(left, is_rate):>14} "
            f"{fmt(right, is_rate):>16} {fmt(delta, is_rate):>12}"
        )
    left_total = visual_summary["total_collision_count"]
    right_total = graph_summary["total_collision_count"]
    print(
        f"{'total_collision_count':<24} {left_total:>14} "
        f"{right_total:>16} {right_total - left_total:>12}"
    )
    print()
    print("Per-seed outcomes:")
    print(f"{'seed':>6} {'condition':<24} {'success':>8} {'hard':>8} {'collision':>10} {'count':>7}")
    for seed, left, right in zip(seeds, visual, graph):
        for condition, record in (("visual_only", left), ("retrieved_graph", right)):
            print(
                f"{seed:>6} {condition:<24} {str(record.get('success')):>8} "
                f"{str(record.get('hard_success')):>8} "
                f"{str(record.get('collision')):>10} "
                f"{str(record.get('collision_count')):>7}"
            )
    print()
    print("Collision-name episode counts (visual -> retrieved):")
    names = sorted(
        set(visual_summary["collision_name_episode_counts"])
        | set(graph_summary["collision_name_episode_counts"])
    )
    for name in names:
        left = visual_summary["collision_name_episode_counts"].get(name, 0)
        right = graph_summary["collision_name_episode_counts"].get(name, 0)
        print(f"  {name}: {left} -> {right}")
    if len(seeds) < 10:
        print()
        print("CAUTION: fewer than 10 paired seeds; treat differences as descriptive only.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare main metrics on seeds shared by two evaluation runs."
    )
    parser.add_argument("visual_only", type=Path, help="visual-only result directory or JSONL")
    parser.add_argument("retrieved_graph", type=Path, help="retrieved-graph result directory or JSONL")
    parser.add_argument("--json-output", type=Path, help="optional machine-readable report path")
    args = parser.parse_args()

    visual_path, visual_by_seed = load_records(args.visual_only)
    graph_path, graph_by_seed = load_records(args.retrieved_graph)
    shared_seeds = sorted(set(visual_by_seed) & set(graph_by_seed))
    if not shared_seeds:
        raise SystemExit("No shared seeds; paired comparison is not possible")
    visual = [visual_by_seed[seed] for seed in shared_seeds]
    graph = [graph_by_seed[seed] for seed in shared_seeds]
    print_report(visual_path, graph_path, shared_seeds, visual, graph)

    if args.json_output:
        report = {
            "paired_seeds": shared_seeds,
            "visual_only": {"path": str(visual_path), **summarize(visual)},
            "visual_retrieved_graph": {"path": str(graph_path), **summarize(graph)},
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report: {args.json_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
