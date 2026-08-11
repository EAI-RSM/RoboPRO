#!/usr/bin/env python3
"""Focused CPU checks for the frozen task-metric bucket specification."""

import copy
import math
from pathlib import Path

from lib.metric_buckets import (
    assign_eps_geom_min,
    assign_metric_record,
    count_metric_record_buckets,
    load_bucket_spec,
    validate_bucket_spec,
)


SPEC_PATH = Path(__file__).resolve().parent.parent / "bucket_spec.json"


def _raises_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _record(value):
    return {
        "eps_geom_min": None if math.isinf(value) else value,
        "eps_geom_min_unbounded": math.isinf(value),
    }


def test_boundaries_and_infinity():
    spec = load_bucket_spec(SPEC_PATH)
    cases = [
        (0.0, "very_low_clearance"),
        (0.074999999, "very_low_clearance"),
        (0.075, "low_clearance"),
        (0.119999999, "low_clearance"),
        (0.12, "medium_clearance"),
        (0.149999999, "medium_clearance"),
        (0.15, "high_clearance"),
        (math.inf, "high_clearance"),
    ]
    for value, expected in cases:
        assert assign_eps_geom_min(value, spec) == expected, (value, expected)
        assert assign_metric_record(_record(value), spec) == expected, (value, expected)
    print("  [1] exact boundaries assign upward; +inf enters high clearance        PASS")


def test_ties_and_pilot_counts():
    spec = load_bucket_spec(SPEC_PATH)
    records = (
        [_record(0.05)] * 17
        + [_record(0.09)] * 35
        + [_record(0.12)] * 30
        + [_record(0.17)] * 17
        + [_record(math.inf)]
    )
    assert count_metric_record_buckets(records, spec) == {
        "very_low_clearance": 17,
        "low_clearance": 35,
        "medium_clearance": 30,
        "high_clearance": 18,
    }
    assert {assign_metric_record(record, spec) for record in [_record(0.12)] * 8} == {
        "medium_clearance"
    }
    print("  [2] ties stay together and frozen pilot counts are reproducible       PASS")


def test_malformed_specs_and_records():
    original = load_bucket_spec(SPEC_PATH)
    bad = copy.deepcopy(original)
    bad["rho_boundaries"] = [2.5, 2.5, 5.0]
    _raises_value_error(lambda: validate_bucket_spec(bad))
    bad = copy.deepcopy(original)
    bad["eps_geom_min_boundaries_m"][1] = 0.121
    _raises_value_error(lambda: validate_bucket_spec(bad))
    bad = copy.deepcopy(original)
    bad["positive_infinity_bucket"] = "medium_clearance"
    _raises_value_error(lambda: validate_bucket_spec(bad))
    bad = copy.deepcopy(original)
    bad["source_distribution"]["bucket_counts"]["high_clearance"] = 17
    _raises_value_error(lambda: validate_bucket_spec(bad))
    _raises_value_error(lambda: assign_metric_record(
        {"eps_geom_min": None, "eps_geom_min_unbounded": False}, original
    ))
    _raises_value_error(lambda: assign_metric_record(
        {"eps_geom_min": 0.1, "eps_geom_min_unbounded": True}, original
    ))
    print("  [3] malformed boundaries, provenance counts, and records are rejected PASS")


def main():
    print("task metric bucket specification -- CPU checks")
    test_boundaries_and_infinity()
    test_ties_and_pilot_counts()
    test_malformed_specs_and_records()
    print("ALL PASS")


if __name__ == "__main__":
    main()
