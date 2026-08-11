"""Yaw-aware 2D footprint geometry for object placement."""

import json
import os
from pathlib import Path

import numpy as np
import transforms3d as t3d


_MODEL_DATA_CACHE = {}


def yaw_from_quat(q):
    """Return world-frame yaw for an upright object's pose quaternion."""
    rotation = t3d.quaternions.quat2mat(q)
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def footprint_half_extents(extents, scale):
    """Return footprint half-extents after local mesh Y is stood up as world Z."""
    return (
        float(extents[0] * scale[0] / 2.0),
        float(extents[2] * scale[2] / 2.0),
    )


def model_footprint_half_extents(modelname, model_id, scale_override=None):
    """Load a model's scaled footprint half-extents without creating an actor."""
    key = (modelname, model_id)
    model_data = _MODEL_DATA_CACHE.get(key)
    if model_data is None:
        path = (
            Path(os.environ["BENCH_ROOT"])
            / "assets"
            / "objects"
            / modelname
            / f"model_data{model_id}.json"
        )
        with open(path, encoding="utf-8") as handle:
            model_data = json.load(handle)
        _MODEL_DATA_CACHE[key] = model_data
    scale = scale_override
    if scale is None:
        scale = model_data.get("scale", [1.0, 1.0, 1.0])
    if isinstance(scale, (int, float)):
        scale = [scale, scale, scale]
    return footprint_half_extents(model_data["extents"], scale)


def rect_corners(center_xy, yaw, half_x, half_y):
    """World-space corners of a yaw-rotated rectangle footprint."""
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local = np.array(
        [[half_x, half_y], [half_x, -half_y], [-half_x, -half_y], [-half_x, half_y]]
    )
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray(center_xy, dtype=float)


def _closest_segment_distance(p1, q1, p2, q2):
    """Distance between closest points on two 2D line segments."""
    d1, d2, relative = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ relative
    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ relative
        if e <= eps:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            denominator = a * e - b * b
            s = np.clip((b * f - c * e) / denominator, 0.0, 1.0) if denominator > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


def _rects_overlap(poly_a, poly_b):
    """Separating-axis test for two convex quadrilaterals."""
    for polygon in (poly_a, poly_b):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                continue
            axis /= norm
            projection_a, projection_b = poly_a @ axis, poly_b @ axis
            if (
                projection_a.max() < projection_b.min() - 1e-9
                or projection_b.max() < projection_a.min() - 1e-9
            ):
                return False
    return True


def min_rect_distance(center_a, yaw_a, half_a, center_b, yaw_b, half_b):
    """Exact minimum distance between two yaw-rotated rectangle footprints."""
    poly_a = rect_corners(center_a, yaw_a, *half_a)
    poly_b = rect_corners(center_b, yaw_b, *half_b)
    if _rects_overlap(poly_a, poly_b):
        return 0.0
    return min(
        _closest_segment_distance(
            poly_a[i], poly_a[(i + 1) % 4], poly_b[j], poly_b[(j + 1) % 4]
        )
        for i in range(4)
        for j in range(4)
    )


def solve_center_offset_for_gap(
    target_center,
    target_yaw,
    target_half,
    occluder_yaw,
    occluder_half,
    desired_gap,
    direction=(0.0, -1.0),
):
    """Solve the center offset along ``direction`` that yields ``desired_gap``."""
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    direction /= norm
    diagonal_a = float(np.hypot(*target_half))
    diagonal_b = float(np.hypot(*occluder_half))
    low, high = 0.0, float(desired_gap) + diagonal_a + diagonal_b + 0.05
    for _ in range(60):
        midpoint = (low + high) / 2.0
        occluder_center = np.asarray(target_center, dtype=float) + midpoint * direction
        gap = min_rect_distance(
            target_center,
            target_yaw,
            target_half,
            occluder_center,
            occluder_yaw,
            occluder_half,
        )
        if gap < desired_gap:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0
