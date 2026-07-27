"""Canonical benchmark relation schema and geometry-only relation kernels."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np


class RelationName(str, Enum):
    ON = "on"
    IN = "in"
    SUPPORTS = "supports"
    CONTAINS = "contains"
    HELD_BY = "held_by"
    NEAR = "near"
    BLOCKS = "blocks"
    OCCLUDES = "occludes"
    REACHABLE_BY = "reachable_by"
    CONTACT_RISK_WITH = "contact_risk_with"
    COLLIDES_WITH = "collides_with"
    VISIBLE_TO = "visible_to"
    PART_OF = "part_of"


@dataclass(frozen=True)
class RelationSpec:
    name: RelationName
    implemented: bool
    domain: str
    codomain: str
    symmetric: bool = False
    validity_field: str | None = None


RELATION_SPECS = (
    RelationSpec(RelationName.ON, True, "object", "object"),
    RelationSpec(RelationName.IN, True, "object", "object", validity_field="containment_valid"),
    RelationSpec(RelationName.SUPPORTS, True, "object", "object"),
    RelationSpec(RelationName.CONTAINS, True, "object", "object", validity_field="contains_valid"),
    RelationSpec(
        RelationName.HELD_BY, True, "object", "effector", validity_field="held_by_valid",
    ),
    RelationSpec(RelationName.NEAR, True, "object", "object", symmetric=True),
    RelationSpec(RelationName.BLOCKS, False, "object", "object"),
    RelationSpec(RelationName.OCCLUDES, False, "object", "object"),
    RelationSpec(
        RelationName.REACHABLE_BY, True, "object", "effector",
        validity_field="reachable_by_valid",
    ),
    RelationSpec(RelationName.CONTACT_RISK_WITH, False, "object", "object", symmetric=True),
    RelationSpec(RelationName.COLLIDES_WITH, True, "object", "object", symmetric=True),
    RelationSpec(
        RelationName.VISIBLE_TO, True, "object", "camera",
        validity_field="visible_to_valid",
    ),
    RelationSpec(RelationName.PART_OF, True, "object", "object"),
)

RELATION_SPEC_BY_NAME = {spec.name: spec for spec in RELATION_SPECS}
CANONICAL_RELATION_NAMES = tuple(spec.name.value for spec in RELATION_SPECS)
IMPLEMENTED_RELATION_NAMES = tuple(
    spec.name.value for spec in RELATION_SPECS if spec.implemented
)
IMPLEMENTED_BINARY_RELATION_NAMES = tuple(
    spec.name.value
    for spec in RELATION_SPECS
    if spec.implemented and spec.domain == "object" and spec.codomain == "object"
)
IMPLEMENTED_BIPARTITE_RELATION_NAMES = tuple(
    spec.name.value
    for spec in RELATION_SPECS
    if spec.implemented and spec.codomain in {"effector", "camera"}
)

NEAR_HORIZONTAL_THRESHOLD_M = 0.10
NEAR_VERTICAL_MARGIN_M = 0.08
MIN_GEOMETRY_EXTENT_M = 1e-6
ON_SUPPORTS_MAX_VERTICAL_PENETRATION_M = 0.03
ON_SUPPORTS_MAX_VERTICAL_SEPARATION_M = 0.06
ON_SUPPORTS_MIN_XY_OVERLAP_RATIO = 0.20
ON_SUPPORTS_MIN_XY_AREA_M2 = 1e-8
IN_CONTAINS_CENTER_TOLERANCE_M = 1e-4
HELD_BY_MAX_OBJECT_TCP_DISTANCE_M = 0.16
CONTAINER_LABEL_TOKENS = (
    "basket", "bin", "box", "cabinet", "drawer", "fridge",
    "microwave", "sink", "bowl", "cup", "fileholder",
    "file_holder", "dishrack", "trash", "container",
)


def is_container_entry(entry: Mapping) -> bool:
    """Return whether catalog metadata identifies a plausible container."""
    label = " ".join(
        str(entry.get(field) or "").lower()
        for field in ("name", "semantic_label", "asset_ref")
    )
    return any(token in label for token in CONTAINER_LABEL_TOKENS)


def compute_in_contains_relations(
    object_catalog,
    aabb_by_id: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    center_tolerance_m: float = IN_CONTAINS_CENTER_TOLERANCE_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute center-based ``in``/``contains`` relations and validity masks."""
    if not np.isfinite(center_tolerance_m) or center_tolerance_m < 0:
        raise ValueError("Containment center tolerance must be finite and non-negative")

    count = len(object_catalog)
    inside = np.zeros((count, count), dtype=np.bool_)
    valid = np.zeros_like(inside)

    for container_idx, container_entry in enumerate(object_catalog):
        if not is_container_entry(container_entry):
            continue
        container_aabb = aabb_by_id.get(int(container_entry["object_id"]))
        if container_aabb is None:
            continue
        container_min, container_max = container_aabb
        container_min = np.asarray(container_min, dtype=np.float64)
        container_max = np.asarray(container_max, dtype=np.float64)

        for object_idx, object_entry in enumerate(object_catalog):
            if object_idx == container_idx:
                continue
            object_aabb = aabb_by_id.get(int(object_entry["object_id"]))
            if object_aabb is None:
                continue
            object_min, object_max = object_aabb
            center = 0.5 * (
                np.asarray(object_min, dtype=np.float64)
                + np.asarray(object_max, dtype=np.float64)
            )
            valid[object_idx, container_idx] = True
            inside[object_idx, container_idx] = bool(
                np.all(center >= container_min - center_tolerance_m)
                and np.all(center <= container_max + center_tolerance_m)
            )

    return inside, inside.T.copy(), valid, valid.T.copy()


def compute_held_by_relations(
    object_ids: np.ndarray,
    center_by_id: Mapping[int, np.ndarray],
    effector_positions: np.ndarray,
    effector_closed: np.ndarray,
    object_effector_contact: np.ndarray,
    *,
    max_object_tcp_distance_m: float = HELD_BY_MAX_OBJECT_TCP_DISTANCE_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute contact-gated object-to-effector grasp state.

    An entry is valid when both the object center and effector position are
    finite and available. A valid edge is true only when the gripper is closed,
    contact evidence is present, and center-to-TCP distance is within the
    inclusive configured threshold.
    """
    object_ids = np.asarray(object_ids)
    if object_ids.ndim != 1:
        raise ValueError(f"object_ids must be one-dimensional; got {object_ids.shape}")
    count = object_ids.shape[0]
    positions = np.asarray(effector_positions, dtype=np.float64)
    closed = np.asarray(effector_closed, dtype=np.bool_)
    contact = np.asarray(object_effector_contact, dtype=np.bool_)
    if positions.shape != (2, 3):
        raise ValueError(
            f"effector_positions has shape {positions.shape}; expected (2, 3)"
        )
    if closed.shape != (2,):
        raise ValueError(f"effector_closed has shape {closed.shape}; expected (2,)")
    if contact.shape != (count, 2):
        raise ValueError(
            f"object_effector_contact has shape {contact.shape}; expected {(count, 2)}"
        )
    if (
        not np.isfinite(max_object_tcp_distance_m)
        or max_object_tcp_distance_m < 0
    ):
        raise ValueError(
            "Maximum object-to-TCP distance must be finite and non-negative"
        )

    held_by = np.zeros((count, 2), dtype=np.bool_)
    valid = np.zeros_like(held_by)
    for object_idx, object_id in enumerate(object_ids.tolist()):
        center = center_by_id.get(int(object_id))
        if center is None:
            continue
        center = np.asarray(center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            continue
        for effector_idx in range(2):
            tcp = positions[effector_idx]
            if not np.all(np.isfinite(tcp)):
                continue
            valid[object_idx, effector_idx] = True
            held_by[object_idx, effector_idx] = bool(
                closed[effector_idx]
                and contact[object_idx, effector_idx]
                and np.linalg.norm(center - tcp) <= max_object_tcp_distance_m
            )

    grasped_by_code = np.full((count,), -1, dtype=np.int8)
    grasped_by_code[np.logical_and(held_by[:, 0], ~held_by[:, 1])] = 0
    grasped_by_code[np.logical_and(~held_by[:, 0], held_by[:, 1])] = 1
    grasped_by_code[np.logical_and(held_by[:, 0], held_by[:, 1])] = 2
    return held_by, valid, grasped_by_code


def compute_xy_overlap_ratio(
    upper_aabb,
    lower_aabb,
    *,
    min_xy_area_m2: float = ON_SUPPORTS_MIN_XY_AREA_M2,
) -> float:
    """Return XY intersection area divided by the upper AABB's XY area."""
    if min_xy_area_m2 <= 0 or not np.isfinite(min_xy_area_m2):
        raise ValueError("Minimum XY area must be finite and positive")
    upper_min, upper_max = upper_aabb
    lower_min, lower_max = lower_aabb
    overlap_x = max(
        0.0,
        min(float(upper_max[0]), float(lower_max[0]))
        - max(float(upper_min[0]), float(lower_min[0])),
    )
    overlap_y = max(
        0.0,
        min(float(upper_max[1]), float(lower_max[1]))
        - max(float(upper_min[1]), float(lower_min[1])),
    )
    upper_area = max(
        (float(upper_max[0]) - float(upper_min[0]))
        * (float(upper_max[1]) - float(upper_min[1])),
        min_xy_area_m2,
    )
    return overlap_x * overlap_y / upper_area


def compute_on_supports_relations(
    object_ids: np.ndarray,
    aabb_by_id: Mapping[int, tuple[np.ndarray, np.ndarray]],
    raw_contact: np.ndarray,
    *,
    max_vertical_penetration_m: float = ON_SUPPORTS_MAX_VERTICAL_PENETRATION_M,
    max_vertical_separation_m: float = ON_SUPPORTS_MAX_VERTICAL_SEPARATION_M,
    min_xy_overlap_ratio: float = ON_SUPPORTS_MIN_XY_OVERLAP_RATIO,
    min_xy_area_m2: float = ON_SUPPORTS_MIN_XY_AREA_M2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute contact-gated ``on`` and its exact inverse ``supports``."""
    values = (
        max_vertical_penetration_m,
        max_vertical_separation_m,
        min_xy_area_m2,
    )
    if not all(np.isfinite(value) and value >= 0 for value in values[:2]):
        raise ValueError("On/supports vertical tolerances must be finite and non-negative")
    if not np.isfinite(min_xy_overlap_ratio) or not 0 <= min_xy_overlap_ratio <= 1:
        raise ValueError("On/supports minimum XY overlap ratio must be in [0, 1]")
    if not np.isfinite(min_xy_area_m2) or min_xy_area_m2 <= 0:
        raise ValueError("On/supports minimum XY area must be finite and positive")

    object_ids = np.asarray(object_ids, dtype=np.int64).reshape(-1)
    contact = np.asarray(raw_contact, dtype=np.bool_)
    expected_shape = (len(object_ids), len(object_ids))
    if contact.shape != expected_shape:
        raise ValueError(
            f"raw_contact has shape {contact.shape}; expected {expected_shape}"
        )
    on = np.zeros(expected_shape, dtype=np.bool_)

    def supported_by(upper_aabb, lower_aabb):
        upper_min, upper_max = upper_aabb
        lower_min, lower_max = lower_aabb
        vertical_gap = float(upper_min[2] - lower_max[2])
        if (
            vertical_gap < -max_vertical_penetration_m
            or vertical_gap > max_vertical_separation_m
        ):
            return False
        upper_center_z = 0.5 * float(upper_min[2] + upper_max[2])
        lower_center_z = 0.5 * float(lower_min[2] + lower_max[2])
        if upper_center_z <= lower_center_z:
            return False
        return compute_xy_overlap_ratio(
            upper_aabb,
            lower_aabb,
            min_xy_area_m2=min_xy_area_m2,
        ) >= min_xy_overlap_ratio

    for i, object_id_i in enumerate(object_ids):
        aabb_i = aabb_by_id.get(int(object_id_i))
        if aabb_i is None:
            continue
        for j in range(i + 1, len(object_ids)):
            if not contact[i, j]:
                continue
            aabb_j = aabb_by_id.get(int(object_ids[j]))
            if aabb_j is None:
                continue
            on[i, j] = supported_by(aabb_i, aabb_j)
            on[j, i] = supported_by(aabb_j, aabb_i)

    return on, on.T.copy()


def compute_xy_aabb_gap(aabb_a, aabb_b) -> float:
    """Return the shortest Euclidean gap between two AABB projections in XY."""
    min_a, max_a = aabb_a
    min_b, max_b = aabb_b
    gap_x = max(
        0.0,
        max(float(min_a[0]) - float(max_b[0]), float(min_b[0]) - float(max_a[0])),
    )
    gap_y = max(
        0.0,
        max(float(min_a[1]) - float(max_b[1]), float(min_b[1]) - float(max_a[1])),
    )
    return float(np.hypot(gap_x, gap_y))


def compute_near_relations(
    object_ids: np.ndarray,
    aabb_by_id: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    horizontal_threshold_m: float = NEAR_HORIZONTAL_THRESHOLD_M,
    vertical_margin_m: float = NEAR_VERTICAL_MARGIN_M,
    min_geometry_extent_m: float = MIN_GEOMETRY_EXTENT_M,
) -> np.ndarray:
    """Compute the symmetric object-object ``near`` adjacency matrix."""
    if horizontal_threshold_m < 0 or vertical_margin_m < 0:
        raise ValueError("Near-relation distance thresholds must be non-negative")
    if min_geometry_extent_m <= 0:
        raise ValueError("Minimum geometry extent must be positive")

    object_ids = np.asarray(object_ids, dtype=np.int64).reshape(-1)
    near = np.zeros((len(object_ids), len(object_ids)), dtype=np.bool_)

    for i, object_id_i in enumerate(object_ids):
        aabb_i = aabb_by_id.get(int(object_id_i))
        if aabb_i is None:
            continue
        min_i, max_i = aabb_i
        center_i = 0.5 * (min_i + max_i)
        extent_i = np.maximum(max_i - min_i, min_geometry_extent_m)

        for j in range(i + 1, len(object_ids)):
            aabb_j = aabb_by_id.get(int(object_ids[j]))
            if aabb_j is None:
                continue
            min_j, max_j = aabb_j
            center_j = 0.5 * (min_j + max_j)
            extent_j = np.maximum(max_j - min_j, min_geometry_extent_m)

            horizontal_gap = compute_xy_aabb_gap(aabb_i, aabb_j)
            vertical_center_gap = abs(float(center_i[2] - center_j[2]))
            vertical_tolerance = (
                max(float(extent_i[2]), float(extent_j[2])) + vertical_margin_m
            )
            if (
                horizontal_gap <= horizontal_threshold_m
                and vertical_center_gap <= vertical_tolerance
            ):
                near[i, j] = True
                near[j, i] = True

    return near


def serialize_and_validate_relations(
    relations: Mapping[RelationName, np.ndarray],
    *,
    object_count: int,
) -> dict[str, np.ndarray]:
    """Validate implemented relations and convert enum keys to schema strings."""
    expected = {spec.name for spec in RELATION_SPECS if spec.implemented}
    actual = set(relations)
    missing = sorted(name.value for name in expected - actual)
    unexpected = sorted(str(name) for name in actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Invalid benchmark relation snapshot: missing={missing}, unexpected={unexpected}"
        )
    for name, value in relations.items():
        spec = RELATION_SPEC_BY_NAME[name]
        matrix = np.asarray(value)
        if matrix.ndim != 2 or matrix.shape[0] != object_count:
            raise RuntimeError(
                f"Relation {name.value!r} has shape {matrix.shape}; "
                f"expected a rank-2 matrix with first dimension {object_count}"
            )
        if spec.codomain == "object" and matrix.shape[1] != object_count:
            raise RuntimeError(
                f"Relation {name.value!r} has shape {matrix.shape}; "
                f"expected ({object_count}, {object_count})"
            )
        if spec.symmetric and not np.array_equal(matrix, matrix.T):
            raise RuntimeError(f"Symmetric relation {name.value!r} is not symmetric")
    return {name.value: value for name, value in relations.items()}
