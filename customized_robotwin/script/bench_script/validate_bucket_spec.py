#!/usr/bin/env python3
"""Validate the frozen metric bucket specification against its outcome-blind pilot."""

import argparse
import hashlib
import json
from pathlib import Path

from lib.metric_buckets import count_metric_record_buckets, load_bucket_spec


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_SPEC = SCRIPT_DIR / "bucket_spec.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if row.get("schema") != "robopro.task-metric.v1" or row.get("status") != "ok":
                raise ValueError(f"not a complete task-metric record at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no task-metric records in {path}")
    return rows


def validate_frozen_source(spec, records_path):
    source = spec["source_distribution"]
    records_path = Path(records_path).resolve()
    if _sha256(records_path) != source["records_sha256"]:
        raise ValueError("records SHA-256 does not match frozen source_distribution")
    metric_run = REPO_ROOT / source["metric_run"]
    distribution_run = REPO_ROOT / source["distribution_run"]
    if _sha256(metric_run / "config.json") != source["config_sha256"]:
        raise ValueError("metric config SHA-256 does not match frozen source_distribution")
    if _sha256(distribution_run / "distribution_summary.json") != source["distribution_summary_sha256"]:
        raise ValueError("distribution summary SHA-256 does not match frozen source_distribution")

    records = _read_records(records_path)
    if len(records) != source["record_count"]:
        raise ValueError("record count does not match frozen source_distribution")
    seeds = [row.get("seed") for row in records]
    expected_seeds = list(range(source["seed_start"], source["seed_end"] + 1))
    if sorted(seeds) != expected_seeds:
        raise ValueError("pilot seeds do not exactly match the frozen seed range")
    scene_ids = [row.get("scene_id") for row in records]
    if not all(scene_ids) or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("pilot scene IDs must be present and unique")

    counts = count_metric_record_buckets(records, spec)
    if counts != source["bucket_counts"]:
        raise ValueError(f"pilot bucket counts changed: expected {source['bucket_counts']}, got {counts}")
    return {
        "status": "pass",
        "record_count": len(records),
        "seed_start": min(seeds),
        "seed_end": max(seeds),
        "bucket_counts": counts,
        "records_sha256": source["records_sha256"],
        "outcome_data_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--records", default=None,
                        help="override the frozen source records.jsonl path")
    args = parser.parse_args()
    spec = load_bucket_spec(args.spec)
    records = args.records
    if records is None:
        records = REPO_ROOT / spec["source_distribution"]["metric_run"] / "records.jsonl"
    report = validate_frozen_source(spec, records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
