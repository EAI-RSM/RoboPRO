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
    STATIC_CONTACT_WITH = "static_contact_with"
    INTENTIONAL_CONTACT_WITH = "intentional_contact_with"
    ROBOT_COLLISION_WITH = "robot_collision_with"
    UNEXPECTED_COLLISION_WITH = "unexpected_collision_with"
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
    RelationSpec(
        RelationName.BLOCKS, True, "object", "object", validity_field="blocks_valid",
    ),
    RelationSpec(
        RelationName.OCCLUDES, True, "object", "object_camera",
        validity_field="occludes_valid",
    ),
    RelationSpec(
        RelationName.REACHABLE_BY, True, "object", "effector",
        validity_field="reachable_by_valid",
    ),
    RelationSpec(RelationName.CONTACT_RISK_WITH, False, "object", "object", symmetric=True),
    RelationSpec(
        RelationName.STATIC_CONTACT_WITH, True, "object", "object", symmetric=True,
        validity_field="contact_semantics_valid",
    ),
    RelationSpec(
        RelationName.INTENTIONAL_CONTACT_WITH, True, "object", "object", symmetric=True,
        validity_field="contact_semantics_valid",
    ),
    RelationSpec(
        RelationName.ROBOT_COLLISION_WITH, True, "object", "object", symmetric=True,
        validity_field="contact_semantics_valid",
    ),
    RelationSpec(
        RelationName.UNEXPECTED_COLLISION_WITH, True, "object", "object", symmetric=True,
        validity_field="contact_semantics_valid",
    ),
    RelationSpec(
        RelationName.VISIBLE_TO, True, "object", "camera",
        validity_field="visible_to_valid",
    ),
    RelationSpec(
        RelationName.PART_OF, True, "object", "object",
        validity_field="part_of_valid",
    ),
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
IMPLEMENTED_CAMERA_CONDITIONED_RELATION_NAMES = tuple(
    spec.name.value
    for spec in RELATION_SPECS
    if spec.implemented and spec.codomain == "object_camera"
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
VISIBLE_TO_MIN_VISIBLE_PIXEL_COUNT = 1
OCCLUDES_MIN_OVERLAP_PIXEL_COUNT = 1
OCCLUDES_MIN_DEPTH_MARGIN_M = 1e-3
OCCLUDES_MIN_OVERLAP_FRACTION = 0.01
BLOCKS_CORRIDOR_CLEARANCE_M = 0.04
BLOCKS_ENDPOINT_MARGIN_M = 0.02
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


def compute_visible_to_relations(
    object_ids: np.ndarray,
    segmentation_ids_by_object_id: Mapping[int, set[int]],
    segmentation_by_camera: Mapping[str, np.ndarray | None],
    *,
    min_visible_pixel_count: int = VISIBLE_TO_MIN_VISIBLE_PIXEL_COUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute object-to-camera visibility from actor-segmentation images."""
    object_ids = np.asarray(object_ids)
    if object_ids.ndim != 1:
        raise ValueError(f"object_ids must be one-dimensional; got {object_ids.shape}")
    if (
        isinstance(min_visible_pixel_count, (bool, np.bool_))
        or not isinstance(min_visible_pixel_count, (int, np.integer))
        or min_visible_pixel_count < 1
    ):
        raise ValueError("Minimum visible pixel count must be an integer >= 1")

    camera_names = sorted(segmentation_by_camera)
    shape = (len(object_ids), len(camera_names))
    visible_to = np.zeros(shape, dtype=np.bool_)
    valid = np.zeros(shape, dtype=np.bool_)
    pixel_count = np.zeros(shape, dtype=np.int64)
    for camera_idx, camera_name in enumerate(camera_names):
        segmentation = segmentation_by_camera[camera_name]
        if segmentation is None:
            continue
        segmentation = np.asarray(segmentation)
        if segmentation.ndim < 2:
            raise ValueError(
                f"Segmentation for camera {camera_name!r} must have at least two dimensions"
            )
        for object_idx, object_id in enumerate(object_ids.tolist()):
            target_ids = segmentation_ids_by_object_id.get(int(object_id))
            if not target_ids:
                continue
            count = int(np.count_nonzero(np.isin(segmentation, tuple(target_ids))))
            pixel_count[object_idx, camera_idx] = count
            visible_to[object_idx, camera_idx] = count >= min_visible_pixel_count
            valid[object_idx, camera_idx] = True
    return visible_to, valid, pixel_count, camera_names



def compute_occludes_relations(
    object_ids: np.ndarray,
    aabb_by_id: Mapping[int, tuple[np.ndarray, np.ndarray]],
    segmentation_ids_by_object_id: Mapping[int, set[int]],
    segmentation_by_camera: Mapping[str, np.ndarray | None],
    depth_m_by_camera: Mapping[str, np.ndarray | None],
    camera_config_by_camera: Mapping[str, Mapping[str, np.ndarray]],
    *,
    target_eligible: np.ndarray | None = None,
    source_eligible: np.ndarray | None = None,
    min_overlap_pixel_count: int = OCCLUDES_MIN_OVERLAP_PIXEL_COUNT,
    min_overlap_fraction: float = OCCLUDES_MIN_OVERLAP_FRACTION,
    min_depth_margin_m: float = OCCLUDES_MIN_DEPTH_MARGIN_M,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, list[str],
]:
    """Compute camera-conditioned occlusion with auditable visual evidence.

    The target's privileged 3-D AABB supplies an amodal convex projected
    silhouette. Source actor-segmentation pixels inside that silhouette provide
    overlap evidence, and their median observed depth must be closer than the
    target AABB's nearest projected depth. Depth inputs are meters.
    """
    object_ids = np.asarray(object_ids)
    if object_ids.ndim != 1:
        raise ValueError(f"object_ids must be one-dimensional; got {object_ids.shape}")
    count = len(object_ids)
    if target_eligible is None:
        target_eligible = np.ones(count, dtype=np.bool_)
    if source_eligible is None:
        source_eligible = np.ones(count, dtype=np.bool_)
    target_eligible = np.asarray(target_eligible, dtype=np.bool_)
    source_eligible = np.asarray(source_eligible, dtype=np.bool_)
    if target_eligible.shape != (count,) or source_eligible.shape != (count,):
        raise ValueError("Occlusion eligibility masks must match object_ids")
    if (
        isinstance(min_overlap_pixel_count, (bool, np.bool_))
        or not isinstance(min_overlap_pixel_count, (int, np.integer))
        or min_overlap_pixel_count < 1
    ):
        raise ValueError("Minimum occlusion overlap pixel count must be an integer >= 1")
    if not np.isfinite(min_overlap_fraction) or not 0 <= min_overlap_fraction <= 1:
        raise ValueError("Minimum occlusion overlap fraction must be finite and in [0,1]")
    if not np.isfinite(min_depth_margin_m) or min_depth_margin_m < 0:
        raise ValueError("Minimum occlusion depth margin must be finite and non-negative")

    camera_names = sorted(segmentation_by_camera)
    shape = (count, count, len(camera_names))
    occludes = np.zeros(shape, dtype=np.bool_)
    valid = np.zeros(shape, dtype=np.bool_)
    overlap_count = np.zeros(shape, dtype=np.int64)
    overlap_fraction = np.zeros(shape, dtype=np.float32)
    source_depth_m = np.full(shape, np.nan, dtype=np.float32)
    target_front_depth_m = np.full(
        (count, len(camera_names)), np.nan, dtype=np.float32
    )
    target_projected_pixel_count = np.zeros(
        (count, len(camera_names)), dtype=np.int64
    )

    def corners(aabb):
        lower, upper = (np.asarray(value, dtype=np.float64) for value in aabb)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("AABB bounds must each have shape (3,)")
        return np.asarray([
            [x, y, z, 1.0]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ], dtype=np.float64)

    def convex_hull(points):
        points = sorted(set(map(tuple, np.asarray(points, dtype=np.float64))))
        if len(points) <= 1:
            return np.asarray(points, dtype=np.float64)

        def cross(origin, a, b):
            return (
                (a[0] - origin[0]) * (b[1] - origin[1])
                - (a[1] - origin[1]) * (b[0] - origin[0])
            )

        lower = []
        for point in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)

    def polygon_mask(hull, x0, x1, y0, y1):
        yy, xx = np.mgrid[y0:y1, x0:x1]
        px = xx.astype(np.float64) + 0.5
        py = yy.astype(np.float64) + 0.5
        mask = np.ones(px.shape, dtype=np.bool_)
        for idx in range(len(hull)):
            start, finish = hull[idx], hull[(idx + 1) % len(hull)]
            cross_value = (
                (finish[0] - start[0]) * (py - start[1])
                - (finish[1] - start[1]) * (px - start[0])
            )
            mask &= cross_value >= -1e-9
        return mask

    for camera_idx, camera_name in enumerate(camera_names):
        segmentation = segmentation_by_camera[camera_name]
        depth_m = depth_m_by_camera.get(camera_name)
        camera_config = camera_config_by_camera.get(camera_name)
        if segmentation is None or depth_m is None or camera_config is None:
            continue
        segmentation = np.asarray(segmentation)
        depth_m = np.asarray(depth_m, dtype=np.float64)
        if segmentation.ndim != 2:
            raise ValueError(f"Segmentation for camera {camera_name!r} must be rank 2")
        if depth_m.shape != segmentation.shape:
            raise ValueError(
                f"Depth for camera {camera_name!r} has shape {depth_m.shape}; "
                f"expected {segmentation.shape}"
            )
        intrinsic = np.asarray(camera_config.get("intrinsic_cv"), dtype=np.float64)
        extrinsic = np.asarray(camera_config.get("extrinsic_cv"), dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(
                f"Camera {camera_name!r} requires intrinsic_cv with shape (3,3); "
                f"got {intrinsic.shape}"
            )
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack((
                extrinsic,
                np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            ))
        elif extrinsic.shape != (4, 4):
            raise ValueError(
                f"Camera {camera_name!r} requires extrinsic_cv with shape "
                f"(3,4) or (4,4); got {extrinsic.shape}"
            )
        height, width = segmentation.shape

        for target_idx, target_id in enumerate(object_ids.tolist()):
            if not target_eligible[target_idx]:
                continue
            target_aabb = aabb_by_id.get(int(target_id))
            if target_aabb is None:
                continue
            camera_corners = (extrinsic @ corners(target_aabb).T).T
            in_front = camera_corners[:, 2] > 1e-9
            if not np.any(in_front):
                continue
            target_front = float(np.min(camera_corners[in_front, 2]))
            projected = (intrinsic @ camera_corners[in_front, :3].T).T
            projected = projected[:, :2] / projected[:, 2:3]
            hull = convex_hull(projected)
            if len(hull) < 3:
                continue
            x0 = max(0, int(np.floor(np.min(hull[:, 0]))))
            x1 = min(width, int(np.ceil(np.max(hull[:, 0]))) + 1)
            y0 = max(0, int(np.floor(np.min(hull[:, 1]))))
            y1 = min(height, int(np.ceil(np.max(hull[:, 1]))) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            projected_mask = polygon_mask(hull, x0, x1, y0, y1)
            projected_pixels = int(np.count_nonzero(projected_mask))
            if projected_pixels == 0:
                continue
            target_front_depth_m[target_idx, camera_idx] = target_front
            target_projected_pixel_count[target_idx, camera_idx] = projected_pixels
            target_segmentation = segmentation[y0:y1, x0:x1]
            target_depth = depth_m[y0:y1, x0:x1]

            for source_idx, source_id in enumerate(object_ids.tolist()):
                if source_idx == target_idx or not source_eligible[source_idx]:
                    continue
                source_ids = segmentation_ids_by_object_id.get(int(source_id))
                if not source_ids:
                    continue
                source_mask = np.logical_and(
                    projected_mask,
                    np.isin(target_segmentation, tuple(source_ids)),
                )
                pixels = int(np.count_nonzero(source_mask))
                overlap_count[source_idx, target_idx, camera_idx] = pixels
                fraction = pixels / projected_pixels
                overlap_fraction[source_idx, target_idx, camera_idx] = fraction
                if pixels == 0:
                    valid[source_idx, target_idx, camera_idx] = True
                    continue
                usable_depth = target_depth[
                    np.logical_and(source_mask, np.isfinite(target_depth))
                    & (target_depth > 0)
                ]
                if usable_depth.size == 0:
                    continue
                median_depth = float(np.median(usable_depth))
                source_depth_m[source_idx, target_idx, camera_idx] = median_depth
                valid[source_idx, target_idx, camera_idx] = True
                occludes[source_idx, target_idx, camera_idx] = bool(
                    pixels >= min_overlap_pixel_count
                    and fraction >= min_overlap_fraction
                    and median_depth + min_depth_margin_m < target_front
                )
    return (
        occludes,
        valid,
        overlap_count,
        overlap_fraction,
        source_depth_m,
        target_front_depth_m,
        target_projected_pixel_count,
        camera_names,
    )


def compute_blocks_relations(
    object_ids: np.ndarray,
    aabb_by_id: Mapping[int, tuple[np.ndarray, np.ndarray]],
    effector_positions: np.ndarray,
    *,
    source_eligible: np.ndarray | None = None,
    target_eligible: np.ndarray | None = None,
    corridor_clearance_m: float = BLOCKS_CORRIDOR_CLEARANCE_M,
    endpoint_margin_m: float = BLOCKS_ENDPOINT_MARGIN_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find objects intersecting a nominal straight approach corridor.

    ``blocks_by_effector[source, target, effector]`` is true when the source
    AABB, inflated by ``corridor_clearance_m``, intersects the segment from the
    end effector to the target center before the target endpoint. The canonical
    ``blocks`` edge is the union across effectors. It describes obstruction of
    a nominal direct approach, not proof that no alternate trajectory exists.

    Missing geometry or effector pose is represented through validity masks,
    rather than silently becoming a negative edge.
    """
    object_ids = np.asarray(object_ids)
    effector_positions = np.asarray(effector_positions, dtype=np.float64)
    if object_ids.ndim != 1:
        raise ValueError(f"object_ids must be one-dimensional; got {object_ids.shape}")
    if effector_positions.ndim != 2 or effector_positions.shape[1] != 3:
        raise ValueError(
            "effector_positions must have shape (E,3); "
            f"got {effector_positions.shape}"
        )
    if not np.isfinite(corridor_clearance_m) or corridor_clearance_m < 0:
        raise ValueError("corridor_clearance_m must be finite and non-negative")
    if not np.isfinite(endpoint_margin_m) or endpoint_margin_m < 0:
        raise ValueError("endpoint_margin_m must be finite and non-negative")

    count = len(object_ids)
    effector_count = len(effector_positions)
    if source_eligible is None:
        source_eligible = np.ones(count, dtype=np.bool_)
    if target_eligible is None:
        target_eligible = np.ones(count, dtype=np.bool_)
    source_eligible = np.asarray(source_eligible, dtype=np.bool_)
    target_eligible = np.asarray(target_eligible, dtype=np.bool_)
    if source_eligible.shape != (count,) or target_eligible.shape != (count,):
        raise ValueError("blocks eligibility masks must match object_ids")

    by_effector = np.zeros((count, count, effector_count), dtype=np.bool_)
    valid_by_effector = np.zeros_like(by_effector)

    def segment_intersects_aabb(start, finish, lower, upper):
        direction = finish - start
        t_min, t_max = 0.0, 1.0
        for axis in range(3):
            if abs(direction[axis]) <= 1e-12:
                if start[axis] < lower[axis] or start[axis] > upper[axis]:
                    return False
                continue
            inv = 1.0 / direction[axis]
            enter = (lower[axis] - start[axis]) * inv
            leave = (upper[axis] - start[axis]) * inv
            if enter > leave:
                enter, leave = leave, enter
            t_min = max(t_min, enter)
            t_max = min(t_max, leave)
            if t_min > t_max:
                return False
        return True

    for target_idx, target_id in enumerate(object_ids.tolist()):
        if not target_eligible[target_idx]:
            continue
        target_aabb = aabb_by_id.get(int(target_id))
        if target_aabb is None:
            continue
        target_lower, target_upper = (
            np.asarray(bound, dtype=np.float64) for bound in target_aabb
        )
        if (
            target_lower.shape != (3,) or target_upper.shape != (3,)
            or not np.all(np.isfinite(target_lower))
            or not np.all(np.isfinite(target_upper))
        ):
            continue
        target_center = (target_lower + target_upper) / 2.0

        for effector_idx, start in enumerate(effector_positions):
            if not np.all(np.isfinite(start)):
                continue
            direction = target_center - start
            distance = float(np.linalg.norm(direction))
            if distance <= endpoint_margin_m:
                continue
            finish = target_center - direction * (endpoint_margin_m / distance)

            for source_idx, source_id in enumerate(object_ids.tolist()):
                if source_idx == target_idx or not source_eligible[source_idx]:
                    continue
                source_aabb = aabb_by_id.get(int(source_id))
                if source_aabb is None:
                    continue
                lower, upper = (
                    np.asarray(bound, dtype=np.float64) for bound in source_aabb
                )
                if (
                    lower.shape != (3,) or upper.shape != (3,)
                    or not np.all(np.isfinite(lower))
                    or not np.all(np.isfinite(upper))
                ):
                    continue
                valid_by_effector[source_idx, target_idx, effector_idx] = True
                by_effector[source_idx, target_idx, effector_idx] = (
                    segment_intersects_aabb(
                        start,
                        finish,
                        lower - corridor_clearance_m,
                        upper + corridor_clearance_m,
                    )
                )

    blocks = np.any(by_effector, axis=2)
    valid = np.any(valid_by_effector, axis=2)
    np.fill_diagonal(blocks, False)
    np.fill_diagonal(valid, False)
    return blocks, valid, by_effector, valid_by_effector
def classify_collision_semantics(
    non_support_contact: np.ndarray,
    is_robot: np.ndarray,
    is_furniture: np.ndarray,
    intentional_contact: np.ndarray,
    baseline_static_contact: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition non-support contacts into mutually exclusive semantic edges."""
    non_support_contact = np.asarray(non_support_contact, dtype=np.bool_)
    if non_support_contact.ndim != 2 or non_support_contact.shape[0] != non_support_contact.shape[1]:
        raise ValueError("non_support_contact must be a square matrix")
    count = non_support_contact.shape[0]
    is_robot = np.asarray(is_robot, dtype=np.bool_)
    is_furniture = np.asarray(is_furniture, dtype=np.bool_)
    intentional_contact = np.asarray(intentional_contact, dtype=np.bool_)
    if is_robot.shape != (count,) or is_furniture.shape != (count,):
        raise ValueError("role masks must match the collision matrix")
    if intentional_contact.shape != non_support_contact.shape:
        raise ValueError("intentional_contact must match the collision matrix")
    if baseline_static_contact is None:
        baseline_static_contact = np.zeros_like(non_support_contact)
    baseline_static_contact = np.asarray(baseline_static_contact, dtype=np.bool_)
    if baseline_static_contact.shape != non_support_contact.shape:
        raise ValueError("baseline_static_contact must match the collision matrix")
    for name, matrix in (
        ("non_support_contact", non_support_contact),
        ("intentional_contact", intentional_contact),
        ("baseline_static_contact", baseline_static_contact),
    ):
        if not np.array_equal(matrix, matrix.T):
            raise ValueError(f"{name} must be symmetric")
        if np.any(np.diag(matrix)):
            raise ValueError(f"{name} must have a false diagonal")
    if np.any(np.logical_and(intentional_contact, ~non_support_contact)):
        raise ValueError("intentional_contact must be backed by non_support_contact")

    intentional = np.logical_and(non_support_contact, intentional_contact)
    remaining = np.logical_and(non_support_contact, ~intentional)
    robot_pair = np.logical_or.outer(is_robot, is_robot)
    static_pair = np.logical_or(
        np.logical_and(baseline_static_contact, ~robot_pair),
        np.logical_and.outer(is_furniture, is_furniture),
    )
    static = np.logical_and(remaining, static_pair)
    remaining = np.logical_and(remaining, ~static)
    robot_collision = np.logical_and(remaining, robot_pair)
    unexpected = np.logical_and(remaining, ~robot_collision)
    valid = np.ones_like(non_support_contact)
    return static, intentional, robot_collision, unexpected, valid


def compute_part_of_relations(
    object_ids: np.ndarray,
    parent_by_child_id: Mapping[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a direct, closed-world structural membership relation."""
    object_ids = np.asarray(object_ids)
    if object_ids.ndim != 1:
        raise ValueError(f"object_ids must be one-dimensional; got {object_ids.shape}")
    ids = [int(value) for value in object_ids.tolist()]
    if len(ids) != len(set(ids)):
        raise ValueError("object_ids must be unique")
    index_by_id = {object_id: idx for idx, object_id in enumerate(ids)}
    part_of = np.zeros((len(ids), len(ids)), dtype=np.bool_)
    valid = np.ones_like(part_of)
    for raw_child_id, raw_parent_id in parent_by_child_id.items():
        child_id, parent_id = int(raw_child_id), int(raw_parent_id)
        if child_id not in index_by_id:
            raise ValueError(f"part_of child id {child_id} is absent from object_ids")
        if parent_id not in index_by_id:
            raise ValueError(f"part_of parent id {parent_id} is absent from object_ids")
        if child_id == parent_id:
            raise ValueError(f"part_of self-membership is invalid for object id {child_id}")
        part_of[index_by_id[child_id], index_by_id[parent_id]] = True
    return part_of, valid


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
        expected_rank = 3 if spec.codomain == "object_camera" else 2
        if matrix.ndim != expected_rank or matrix.shape[0] != object_count:
            raise RuntimeError(
                f"Relation {name.value!r} has shape {matrix.shape}; "
                f"expected rank {expected_rank} with first dimension {object_count}"
            )
        if spec.codomain in {"object", "object_camera"} and matrix.shape[1] != object_count:
            raise RuntimeError(
                f"Relation {name.value!r} has shape {matrix.shape}; "
                f"expected the first two dimensions to be ({object_count}, {object_count})"
            )
        if spec.symmetric and not np.array_equal(matrix, matrix.T):
            raise RuntimeError(f"Symmetric relation {name.value!r} is not symmetric")
    return {name.value: value for name, value in relations.items()}
