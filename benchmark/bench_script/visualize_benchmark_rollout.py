#!/usr/bin/env python3
"""
Render a schematic verification video from a RoboPRO benchmark HDF5 export.

The video is a top-down world-space debug view intended to verify that exported
object states, gripper trajectories, and robot link polylines are present and
temporally aligned.
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

    object_ids = catalog["object_ids"][()]
    names = _decode_string_array(catalog["names"])
    roles = _decode_string_array(catalog["roles"])

    object_state_ids = state["object_ids"][()]
    pose_world = state["pose_world"][()]
    is_present = state["is_present"][()]

    id_to_meta = {}
    for idx, object_id in enumerate(object_ids.tolist()):
        id_to_meta[int(object_id)] = {
            "name": names[idx] if idx < len(names) else str(object_id),
            "role": roles[idx] if idx < len(roles) else "other",
        }

    return object_state_ids, pose_world, is_present, id_to_meta


def _load_link_info(root: h5py.File):
    support = root["benchmark_support"]
    if "link_state" not in support:
        return None, None, None, None, None

    link_state = support["link_state"]
    positions_world = link_state["positions_world"][()]
    side_code = link_state["side_code"][()]
    link_names = _decode_string_array(link_state["link_names"])
    chain_index = link_state["chain_index"][()] if "chain_index" in link_state else None
    parent_index = link_state["parent_index"][()] if "parent_index" in link_state else None
    return positions_world, side_code, link_names, chain_index, parent_index


def _load_gripper_poses(root: h5py.File):
    if "endpose" not in root:
        return None, None
    endpose = root["endpose"]
    left = endpose["left_endpose"][()] if "left_endpose" in endpose else None
    right = endpose["right_endpose"][()] if "right_endpose" in endpose else None
    return left, right


def _compute_bounds(object_pose_world, object_present, link_positions_world, left_gripper, right_gripper):
    xy_points = []

    if object_pose_world is not None and object_present is not None:
        mask = object_present.astype(bool)
        if mask.any():
            xy_points.append(object_pose_world[:, :, :2][mask])

    if link_positions_world is not None and link_positions_world.size > 0:
        xy_points.append(link_positions_world[:, :, :2].reshape(-1, 2))

    if left_gripper is not None:
        xy_points.append(left_gripper[:, :2])
    if right_gripper is not None:
        xy_points.append(right_gripper[:, :2])

    if not xy_points:
        return (-1.0, 1.0), (-1.0, 1.0)

    points = np.concatenate(xy_points, axis=0)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)

    span = np.maximum(max_xy - min_xy, np.array([0.4, 0.4], dtype=np.float32))
    pad = 0.15 * span
    xlim = (float(min_xy[0] - pad[0]), float(max_xy[0] + pad[0]))
    ylim = (float(min_xy[1] - pad[1]), float(max_xy[1] + pad[1]))
    return xlim, ylim


def _world_to_canvas(points_xy: np.ndarray, xlim, ylim, width: int, height: int, margin: int):
    x0, x1 = xlim
    y0, y1 = ylim
    usable_w = max(width - 2 * margin, 1)
    usable_h = max(height - 2 * margin, 1)
    xs = margin + (points_xy[:, 0] - x0) / max(x1 - x0, 1e-6) * usable_w
    ys = height - margin - (points_xy[:, 1] - y0) / max(y1 - y0, 1e-6) * usable_h
    return np.stack([xs, ys], axis=1).astype(np.int32)


def _draw_axes(frame, xlim, ylim, width: int, height: int, margin: int):
    axis_color = (220, 220, 220)
    cv2.rectangle(frame, (margin, margin), (width - margin, height - margin), axis_color, 1)
    cv2.putText(frame, f"x:[{xlim[0]:.2f}, {xlim[1]:.2f}]", (margin, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"y:[{ylim[0]:.2f}, {ylim[1]:.2f}]", (margin, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1, cv2.LINE_AA)


def _role_color(role: str):
    if role == "target":
        return (40, 80, 230)
    if role == "distractor":
        return (0, 170, 255)
    if role == "furniture":
        return (100, 100, 100)
    return (80, 180, 80)


def _draw_link_chain(frame, frame_positions, side_mask, chain_index, xlim, ylim, width, height, margin, color):
    if not np.any(side_mask):
        return

    side_points = frame_positions[side_mask, :2]
    canvas_points = _world_to_canvas(side_points, xlim, ylim, width, height, margin)

    if chain_index is not None:
        side_chain_index = chain_index[side_mask]
        order = np.argsort(side_chain_index, kind="stable")
        canvas_points = canvas_points[order]

    if len(canvas_points) >= 2:
        cv2.polylines(frame, [canvas_points.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    for point in canvas_points:
        cv2.circle(frame, tuple(point), 3, color, -1, cv2.LINE_AA)


def render_video(hdf5_path: Path, output_path: Path, width: int, height: int, fps: int, trail: int):
    with h5py.File(hdf5_path, "r") as root:
        object_ids, object_pose_world, object_present, object_meta = _load_object_info(root)
        link_positions_world, link_side_code, _, link_chain_index, _ = _load_link_info(root)
        left_gripper, right_gripper = _load_gripper_poses(root)

        num_frames = object_pose_world.shape[0]
        xlim, ylim = _compute_bounds(object_pose_world, object_present, link_positions_world, left_gripper, right_gripper)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise SystemExit(f"Failed to open video writer for {output_path}")

        margin = 40

        for frame_idx in range(num_frames):
            frame = np.full((height, width, 3), 255, dtype=np.uint8)
            _draw_axes(frame, xlim, ylim, width, height, margin)

            if link_positions_world is not None and link_side_code is not None:
                for side_value, color in ((0, (220, 80, 80)), (1, (80, 160, 220))):
                    mask = link_side_code == side_value
                    _draw_link_chain(
                        frame,
                        link_positions_world[frame_idx],
                        mask,
                        link_chain_index,
                        xlim,
                        ylim,
                        width,
                        height,
                        margin,
                        color,
                    )

            if left_gripper is not None:
                start = max(0, frame_idx - trail + 1)
                pts = _world_to_canvas(left_gripper[start:frame_idx + 1, :2], xlim, ylim, width, height, margin)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, (0, 0, 180), 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(pts[-1]), 5, (0, 0, 220), -1, cv2.LINE_AA)

            if right_gripper is not None:
                start = max(0, frame_idx - trail + 1)
                pts = _world_to_canvas(right_gripper[start:frame_idx + 1, :2], xlim, ylim, width, height, margin)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, (180, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(pts[-1]), 5, (220, 0, 0), -1, cv2.LINE_AA)

            object_points = object_pose_world[frame_idx, :, :2]
            object_canvas = _world_to_canvas(object_points, xlim, ylim, width, height, margin)
            for idx, point in enumerate(object_canvas):
                if not bool(object_present[frame_idx, idx]):
                    continue
                object_id = int(object_ids[idx])
                meta = object_meta.get(object_id, {"name": str(object_id), "role": "other"})
                color = _role_color(meta["role"])
                cv2.circle(frame, tuple(point), 6, color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    meta["name"],
                    (int(point[0] + 6), int(point[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(frame, f"frame {frame_idx + 1}/{num_frames}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.putText(frame, "red trail: left gripper", (20, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
            cv2.putText(frame, "blue trail: right gripper", (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "arm polylines: robot link_state", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)

            writer.write(frame)

        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Render a schematic verification video from a RoboPRO benchmark HDF5 export")
    parser.add_argument("--file", required=True, help="Path to the HDF5 episode file")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--width", type=int, default=1280, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Video height in pixels")
    parser.add_argument("--fps", type=int, default=20, help="Output frames per second")
    parser.add_argument("--trail", type=int, default=25, help="Number of frames to keep in gripper trail history")
    args = parser.parse_args()

    hdf5_path = Path(args.file).expanduser().resolve()
    if not hdf5_path.exists():
        raise SystemExit(f"HDF5 file not found: {hdf5_path}")

    output_path = Path(args.output).expanduser().resolve()
    render_video(hdf5_path, output_path, args.width, args.height, args.fps, args.trail)
    print(f"Saved rollout visualization to {output_path}")


if __name__ == "__main__":
    main()
