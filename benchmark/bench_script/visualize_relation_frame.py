#!/usr/bin/env python3
"""
Render a single-frame relation graph overlay from a RoboPRO benchmark HDF5 export.

The output is a top-down world-space schematic intended for relation debugging:
objects become nodes and supported canonical relations become color-coded edges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def _decode_string_array(dataset) -> list[str]:
    values = dataset[()]
    if isinstance(values, np.ndarray):
        return [
            item.decode("utf-8", errors="replace") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in values.tolist()
        ]
    return [str(values)]


def _load_object_info(root: h5py.File):
    support = root["benchmark_support"]
    catalog = support["object_catalog"]
    state = support["object_state"]

    object_ids = state["object_ids"][()]
    pose_world = state["pose_world"][()]
    is_present = state["is_present"][()]
    names = _decode_string_array(catalog["names"])
    roles = _decode_string_array(catalog["roles"])

    id_to_meta = {}
    for idx, object_id in enumerate(catalog["object_ids"][()].tolist()):
        id_to_meta[int(object_id)] = {
            "name": names[idx] if idx < len(names) else str(object_id),
            "role": roles[idx] if idx < len(roles) else "other",
        }
    return object_ids, pose_world, is_present, id_to_meta


def _load_relation_info(root: h5py.File):
    state = root["benchmark_support"]["relation_state"]
    relations = {
        "on": state["on"][()] if "on" in state else None,
        "supports": state["supports"][()] if "supports" in state else None,
        "near": state["near"][()] if "near" in state else None,
        "collides_with": state["collides_with"][()] if "collides_with" in state else None,
        "held_by": state["held_by"][()] if "held_by" in state else None,
    }
    effector_names = _decode_string_array(state["held_by_effector_names"]) if "held_by_effector_names" in state else []
    return relations, effector_names


def _load_gripper_positions(root: h5py.File):
    if "endpose" not in root:
        return {}
    endpose = root["endpose"]
    positions = {}
    if "left_endpose" in endpose:
        positions["left_end_effector"] = endpose["left_endpose"][()][:, :3]
    if "right_endpose" in endpose:
        positions["right_end_effector"] = endpose["right_endpose"][()][:, :3]
    return positions


def _compute_bounds(points_xy: np.ndarray):
    if points_xy.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    min_xy = points_xy.min(axis=0)
    max_xy = points_xy.max(axis=0)
    span = np.maximum(max_xy - min_xy, np.array([0.4, 0.4], dtype=np.float32))
    pad = 0.15 * span
    return (float(min_xy[0] - pad[0]), float(max_xy[0] + pad[0])), (float(min_xy[1] - pad[1]), float(max_xy[1] + pad[1]))


def _world_to_canvas(points_xy: np.ndarray, xlim, ylim, width: int, height: int, margin: int):
    x0, x1 = xlim
    y0, y1 = ylim
    usable_w = max(width - 2 * margin, 1)
    usable_h = max(height - 2 * margin, 1)
    xs = margin + (points_xy[:, 0] - x0) / max(x1 - x0, 1e-6) * usable_w
    ys = height - margin - (points_xy[:, 1] - y0) / max(y1 - y0, 1e-6) * usable_h
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _role_color(role: str):
    if role == "target":
        return (40, 80, 230)
    if role == "distractor":
        return (0, 170, 255)
    if role == "furniture":
        return (100, 100, 100)
    return (80, 180, 80)


def _draw_edge(frame, p0, p1, color, label: str, thickness: int = 2):
    cv2.line(frame, tuple(p0), tuple(p1), color, thickness, cv2.LINE_AA)
    mid = ((int(p0[0]) + int(p1[0])) // 2, (int(p0[1]) + int(p1[1])) // 2)
    cv2.putText(frame, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def render_relation_frame(hdf5_path: Path, frame_idx: int, output_path: Path, width: int, height: int):
    with h5py.File(hdf5_path, "r") as root:
        object_ids, pose_world, is_present, id_to_meta = _load_object_info(root)
        relations, effector_names = _load_relation_info(root)
        gripper_positions = _load_gripper_positions(root)

        num_frames = pose_world.shape[0]
        if frame_idx < 0 or frame_idx >= num_frames:
            raise SystemExit(f"Frame index {frame_idx} out of range (num_frames={num_frames})")

        mask = is_present[frame_idx].astype(bool)
        object_points = pose_world[frame_idx, :, :2]
        extra_points = []
        for effector_name, positions in gripper_positions.items():
            if frame_idx < len(positions):
                extra_points.append(positions[frame_idx, :2])
        points_for_bounds = object_points[mask]
        if extra_points:
            points_for_bounds = np.concatenate([points_for_bounds, np.array(extra_points, dtype=np.float32)], axis=0)
        xlim, ylim = _compute_bounds(points_for_bounds)

        margin = 40
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.rectangle(frame, (margin, margin), (width - margin, height - margin), (220, 220, 220), 1)

        object_canvas = _world_to_canvas(object_points, xlim, ylim, width, height, margin)
        effector_canvas = {}
        for effector_name, positions in gripper_positions.items():
            if frame_idx < len(positions):
                effector_canvas[effector_name] = _world_to_canvas(
                    positions[frame_idx:frame_idx + 1, :2],
                    xlim,
                    ylim,
                    width,
                    height,
                    margin,
                )[0]

        relation_order = (
            ("near", (120, 120, 120)),
            ("collides_with", (20, 20, 220)),
            ("on", (20, 140, 20)),
            ("supports", (150, 90, 20)),
        )
        for relation_name, color in relation_order:
            relation = relations.get(relation_name)
            if relation is None:
                continue
            matrix = relation[frame_idx]
            for i in range(matrix.shape[0]):
                if not mask[i]:
                    continue
                for j in range(matrix.shape[1]):
                    if not matrix[i, j] or not mask[j]:
                        continue
                    if relation_name in {"near", "collides_with"} and j <= i:
                        continue
                    _draw_edge(frame, object_canvas[i], object_canvas[j], color, relation_name)

        held_by = relations.get("held_by")
        if held_by is not None:
            held_frame = held_by[frame_idx]
            for i in range(held_frame.shape[0]):
                if not mask[i]:
                    continue
                for eff_idx, effector_name in enumerate(effector_names):
                    if eff_idx >= held_frame.shape[1] or not held_frame[i, eff_idx]:
                        continue
                    if effector_name not in effector_canvas:
                        continue
                    _draw_edge(frame, object_canvas[i], effector_canvas[effector_name], (170, 0, 170), "held_by")

        for effector_name, point in effector_canvas.items():
            cv2.circle(frame, tuple(point), 7, (170, 0, 170), -1, cv2.LINE_AA)
            cv2.putText(frame, effector_name, (int(point[0] + 8), int(point[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 0, 170), 1, cv2.LINE_AA)

        for idx, point in enumerate(object_canvas):
            if not mask[idx]:
                continue
            object_id = int(object_ids[idx])
            meta = id_to_meta.get(object_id, {"name": str(object_id), "role": "other"})
            color = _role_color(meta["role"])
            cv2.circle(frame, tuple(point), 7, color, -1, cv2.LINE_AA)
            cv2.putText(frame, meta["name"], (int(point[0] + 8), int(point[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        legend = [
            ("near", (120, 120, 120)),
            ("collides_with", (20, 20, 220)),
            ("on", (20, 140, 20)),
            ("supports", (150, 90, 20)),
            ("held_by", (170, 0, 170)),
        ]
        cv2.putText(frame, f"frame {frame_idx}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        legend_y = height - 20
        for label, color in legend:
            cv2.putText(frame, label, (20, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            legend_y -= 20

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise SystemExit(f"Failed to write relation frame to {output_path}")
        print(f"Saved relation frame visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render a single-frame relation graph overlay from a RoboPRO benchmark HDF5 export")
    parser.add_argument("--file", required=True, help="Path to the HDF5 episode file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--width", type=int, default=1400, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Image height in pixels")
    args = parser.parse_args()

    hdf5_path = Path(args.file).expanduser().resolve()
    if not hdf5_path.exists():
        raise SystemExit(f"HDF5 file not found: {hdf5_path}")

    output_path = Path(args.output).expanduser().resolve()
    render_relation_frame(hdf5_path, args.frame, output_path, args.width, args.height)


if __name__ == "__main__":
    main()
