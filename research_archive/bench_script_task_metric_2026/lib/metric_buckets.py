"""Pure helpers for the frozen task-metric clearance buckets."""

import json
import math
from bisect import bisect_right
from collections import Counter
from pathlib import Path


SCHEMA = "robopro.metric-bucket-spec.v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_numbers(values, name):
    _require(isinstance(values, list) and values, f"{name} must be a non-empty list")
    converted = []
    for value in values:
        _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                 f"{name} values must be numbers")
        value = float(value)
        _require(math.isfinite(value), f"{name} values must be finite")
        converted.append(value)
    _require(all(a < b for a, b in zip(converted, converted[1:])),
             f"{name} must be strictly increasing")
    return converted


def validate_bucket_spec(spec):
    """Validate and return a frozen v1 bucket specification."""
    _require(isinstance(spec, dict), "bucket spec must be a JSON object")
    _require(spec.get("schema") == SCHEMA, f"bucket spec schema must be {SCHEMA!r}")
    _require(spec.get("status") == "frozen", "bucket spec status must be 'frozen'")
    _require(spec.get("task") == "put_cup_on_coaster", "unexpected bucket-spec task")

    variable = spec.get("bucket_variable")
    _require(isinstance(variable, dict), "bucket_variable must be an object")
    _require(variable.get("name") == "rho_geom_min", "unexpected bucket variable")
    _require(variable.get("source_metric") == "eps_geom_min", "unexpected source metric")
    _require(variable.get("operation") == "divide_by_reference_radius",
             "unexpected bucket-variable operation")
    radius = variable.get("reference_radius_m")
    _require(isinstance(radius, (int, float)) and not isinstance(radius, bool),
             "reference_radius_m must be numeric")
    radius = float(radius)
    _require(math.isfinite(radius) and radius > 0, "reference_radius_m must be positive and finite")

    names = spec.get("bucket_order")
    _require(isinstance(names, list) and len(names) >= 2, "bucket_order must contain at least two names")
    _require(all(isinstance(name, str) and name for name in names), "bucket names must be non-empty strings")
    _require(len(names) == len(set(names)), "bucket names must be unique")
    rho_boundaries = _finite_numbers(spec.get("rho_boundaries"), "rho_boundaries")
    eps_boundaries = _finite_numbers(
        spec.get("eps_geom_min_boundaries_m"), "eps_geom_min_boundaries_m"
    )
    _require(len(rho_boundaries) == len(names) - 1, "rho boundary count must be bucket count minus one")
    _require(len(eps_boundaries) == len(names) - 1, "eps boundary count must be bucket count minus one")
    for rho, eps in zip(rho_boundaries, eps_boundaries):
        _require(math.isclose(eps, rho * radius, rel_tol=0.0, abs_tol=1e-12),
                 "rho and eps boundaries disagree with reference_radius_m")

    _require(spec.get("interval_closure") == "lower_closed_upper_open",
             "interval_closure must be lower_closed_upper_open")
    _require(spec.get("boundary_assignment") == "higher_bucket",
             "boundary_assignment must be higher_bucket")
    _require(spec.get("tie_policy") == "identical_values_never_split",
             "unexpected tie policy")
    _require(spec.get("positive_infinity_bucket") == names[-1],
             "positive infinity must enter the last bucket")
    _require(spec.get("labels_encode") == "clearance_not_outcome_difficulty",
             "bucket labels must encode clearance, not outcome difficulty")
    _require(spec.get("primary_association_predictor") == "eps_geom_min",
             "continuous primary predictor must remain eps_geom_min")

    source = spec.get("source_distribution")
    _require(isinstance(source, dict), "source_distribution must be an object")
    _require(source.get("outcome_data_loaded") is False,
             "bucket source must explicitly state outcome_data_loaded=false")
    record_count = source.get("record_count")
    _require(isinstance(record_count, int) and not isinstance(record_count, bool) and record_count > 0,
             "source record_count must be a positive integer")
    expected_counts = source.get("bucket_counts")
    _require(isinstance(expected_counts, dict) and set(expected_counts) == set(names),
             "source bucket_counts must contain exactly the bucket names")
    _require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                 for value in expected_counts.values()),
             "source bucket counts must be non-negative integers")
    _require(sum(expected_counts.values()) == record_count,
             "source bucket counts must sum to source record_count")
    for key in ("records_sha256", "config_sha256", "distribution_summary_sha256"):
        digest = source.get(key)
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(char in "0123456789abcdef" for char in digest),
                 f"{key} must be a lowercase SHA-256 digest")
    return spec


def load_bucket_spec(path):
    path = Path(path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid bucket spec JSON at {path}: {exc}") from exc
    return validate_bucket_spec(spec)


def _metric_eps(record):
    value = record.get("eps_geom_min")
    is_inf = record.get("eps_geom_min_unbounded")
    if is_inf is True:
        _require(value is None, "unbounded eps_geom_min must be null")
        return math.inf
    _require(is_inf is False, "eps_geom_min_unbounded must be an explicit Boolean")
    _require(value is not None, "finite eps_geom_min must not be null")
    value = float(value)
    _require(math.isfinite(value) and value >= 0, "eps_geom_min must be finite and non-negative")
    return value


def _assignment(eps_geom_min, spec):
    radius = float(spec["bucket_variable"]["reference_radius_m"])
    eps_geom_min = float(eps_geom_min)
    _require(not math.isnan(eps_geom_min) and eps_geom_min >= 0,
             "eps_geom_min must be non-negative and not NaN")
    rho = eps_geom_min / radius
    rho_index = bisect_right(spec["rho_boundaries"], rho)
    eps_index = bisect_right(spec["eps_geom_min_boundaries_m"], eps_geom_min)
    _require(rho_index == eps_index, "rho and metre boundary assignments disagree")
    return spec["bucket_order"][rho_index]


def assign_eps_geom_min(eps_geom_min, spec):
    """Assign one finite or +inf eps_geom_min value using the frozen closure rules."""
    validate_bucket_spec(spec)
    return _assignment(eps_geom_min, spec)


def assign_metric_record(record, spec):
    """Assign one task-metric record, preserving explicit JSON representation of +inf."""
    validate_bucket_spec(spec)
    return _assignment(_metric_eps(record), spec)


def count_metric_record_buckets(records, spec):
    """Return ordered counts after proving every supplied record has one assignment."""
    validate_bucket_spec(spec)
    counts = Counter(_assignment(_metric_eps(record), spec) for record in records)
    return {name: int(counts[name]) for name in spec["bucket_order"]}
