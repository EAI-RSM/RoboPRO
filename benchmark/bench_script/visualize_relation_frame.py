#!/usr/bin/env python3
"""
Render a readable scene/action graph snapshot from a RoboPRO benchmark HDF5 export.

The output is a top-down world-space schematic intended for relation debugging:
objects become nodes and supported canonical relations become color-coded edges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


LEGEND_FONT_SCALE = 0.50
LEGEND_HEADER_SCALE = 0.62
NODE_LABEL_FONT_SCALE = 0.60
SMALL_NODE_LABEL_FONT_SCALE = 0.55
ACTION_NODE_LABEL_FONT_SCALE = 0.62
KNOWN_EDGE_NAMES = {
    "near", "blocks", "static_contact_with", "intentional_contact_with", "robot_collision_with", "unexpected_collision_with", "on", "supports", "part_of", "in", "contains",
    "held_by", "reachable_by", "visible_to", "occludes", "agent", "target", "destination",
}


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
        "blocks": state["blocks"][()] if "blocks" in state else None,
        "static_contact_with": state["static_contact_with"][()] if "static_contact_with" in state else None,
        "intentional_contact_with": state["intentional_contact_with"][()] if "intentional_contact_with" in state else None,
        "robot_collision_with": state["robot_collision_with"][()] if "robot_collision_with" in state else None,
        "unexpected_collision_with": state["unexpected_collision_with"][()] if "unexpected_collision_with" in state else None,
        "held_by": state["held_by"][()] if "held_by" in state else None,
        "part_of": state["part_of"][()] if "part_of" in state else None,
        "in": state["in"][()] if "in" in state else None,
        "contains": state["contains"][()] if "contains" in state else None,
        "reachable_by": state["reachable_by"][()] if "reachable_by" in state else None,
        "visible_to": state["visible_to"][()] if "visible_to" in state else None,
        "occludes": state["occludes"][()] if "occludes" in state else None,
    }
    effector_names = _decode_string_array(state["held_by_effector_names"]) if "held_by_effector_names" in state else []
    reachable_names = _decode_string_array(state["reachable_by_effector_names"]) if "reachable_by_effector_names" in state else []
    camera_names = _decode_string_array(state["visible_to_camera_names"]) if "visible_to_camera_names" in state else []
    return relations, effector_names, reachable_names, camera_names


def _load_active_action(root: h5py.File, frame_idx: int):
    support = root["benchmark_support"]
    if "action_nodes" not in support:
        return None, []
    actions = support["action_nodes"]
    if "active" not in actions or frame_idx >= actions["active"].shape[0]:
        return None, []
    indices = np.flatnonzero(actions["active"][frame_idx])
    if not len(indices):
        return None, []
    idx = int(indices[-1])
    action_id = int(actions["action_ids"][idx])
    action = {
        "id": action_id,
        "type": _decode_string_array(actions["action_types"])[idx],
        "phase": _decode_string_array(actions["execution_phases"])[idx],
        "start": int(actions["start_frame"][idx]),
        "end": int(actions["end_frame"][idx]),
    }
    edges = []
    if "action_entity_edges" in support:
        group = support["action_entity_edges"]
        roles = _decode_string_array(group["roles"])
        for edge_idx, edge_action_id in enumerate(group["action_id"][()]):
            if int(edge_action_id) == action_id:
                edges.append((int(group["object_id"][edge_idx]), roles[edge_idx]))
    return action, edges


def _load_gripper_positions(root: h5py.File):
    if "endpose" not in root:
        return {}
    endpose = root["endpose"]
    positions = {}
    if "left_endpose" in endpose:
        positions["left_ee"] = endpose["left_endpose"][()][:, :3]
    if "right_endpose" in endpose:
        positions["right_ee"] = endpose["right_endpose"][()][:, :3]
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


def _draw_edge(frame, p0, p1, color, label: str, thickness: int = 2, directed: bool = True,
               dashed: bool = False, curved: bool = True):
    p0f, p1f = np.asarray(p0, float), np.asarray(p1, float)
    delta = p1f - p0f
    length = np.linalg.norm(delta)
    if curved and length > 1:
        normal = np.array([-delta[1], delta[0]]) / length
        control = (p0f + p1f) / 2 + normal * min(55.0, max(18.0, length * 0.16))
        t = np.linspace(0.0, 1.0, 41)[:, None]
        points = (1-t) ** 2 * p0f + 2 * (1-t) * t * control + t ** 2 * p1f
    else:
        points = np.linspace(p0f, p1f, 41)
    points_i = points.astype(np.int32)
    if dashed:
        for idx in range(0, len(points_i)-1, 4):
            cv2.polylines(frame, [points_i[idx:min(idx+3, len(points_i))]], False,
                          color, thickness, cv2.LINE_AA)
    else:
        cv2.polylines(frame, [points_i], False, color, thickness, cv2.LINE_AA)
    if directed:
        # Keep the arrow at the relation midpoint: source -> relation -> destination.
        arrow_start, arrow_end = points_i[16], points_i[24]
        cv2.arrowedLine(frame, tuple(arrow_start), tuple(arrow_end),
                        color, thickness + 1, cv2.LINE_AA, tipLength=0.35)
    mid = points_i[len(points_i)//2]
    cv2.putText(frame, label, (int(mid[0] + 4), int(mid[1] - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _parse_edge_list(value: str, option_name: str) -> set[str]:
    """Accept visible_to,near as well as [visible_to, near]."""
    normalized = value.strip().strip("[]")
    excluded = {item.strip().strip("'\"") for item in normalized.split(",") if item.strip()}
    unknown = excluded - KNOWN_EDGE_NAMES
    if unknown:
        choices = ", ".join(sorted(KNOWN_EDGE_NAMES))
        raise SystemExit(f"Unknown edge(s) in {option_name}: {', '.join(sorted(unknown))}. Valid names: {choices}")
    return excluded


def _draw_node(frame, point, color, shape="circle", size=7):
    x, y = map(int, point)
    if shape == "square":
        cv2.rectangle(frame, (x-size, y-size), (x+size, y+size), color, -1, cv2.LINE_AA)
    elif shape == "diamond":
        cv2.fillConvexPoly(frame, np.array([(x, y-size-2), (x+size+2, y), (x, y+size+2), (x-size-2, y)]), color, cv2.LINE_AA)
    elif shape == "triangle":
        cv2.fillConvexPoly(frame, np.array([(x, y-size-2), (x+size+2, y+size), (x-size-2, y+size)]), color, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), size, color, -1, cv2.LINE_AA)


def _abstract_graph_layout(mask, object_ids, relations, frame_idx, effector_names,
                           reachable_names, camera_names, action, action_edges,
                           excluded_edges, graph_width, height, margin):
    """Deterministic force-directed layout for relation-centric inspection."""
    nodes = [("object", int(i)) for i in np.flatnonzero(mask)]
    layout_effectors = set()
    if "held_by" not in excluded_edges:
        layout_effectors.update(effector_names)
    if "reachable_by" not in excluded_edges:
        layout_effectors.update(reachable_names)
    nodes += [("effector", name) for name in sorted(layout_effectors)]
    if "visible_to" not in excluded_edges:
        nodes += [("camera", name) for name in camera_names]
    if action:
        nodes.append(("action", int(action["id"])))
    node_index = {node: idx for idx, node in enumerate(nodes)}
    edges = set()

    for relation_name in ("near", "blocks", "static_contact_with", "intentional_contact_with", "robot_collision_with", "unexpected_collision_with", "on", "supports", "part_of", "in", "contains"):
        relation = relations.get(relation_name)
        if relation is None or relation_name in excluded_edges:
            continue
        for source, destination in np.argwhere(relation[frame_idx]):
            a, b = ("object", int(source)), ("object", int(destination))
            if a in node_index and b in node_index and a != b:
                edges.add(tuple(sorted((node_index[a], node_index[b]))))
    for relation_name, names in (("held_by", effector_names), ("reachable_by", reachable_names),
                                 ("visible_to", camera_names)):
        relation = relations.get(relation_name)
        if relation is None or relation_name in excluded_edges:
            continue
        target_type = "camera" if relation_name == "visible_to" else "effector"
        for source, target_idx in np.argwhere(relation[frame_idx]):
            if target_idx >= len(names):
                continue
            a, b = ("object", int(source)), (target_type, names[target_idx])
            if a in node_index and b in node_index:
                edges.add(tuple(sorted((node_index[a], node_index[b]))))
    if action:
        action_node = ("action", int(action["id"]))
        id_to_object_idx = {int(value): idx for idx, value in enumerate(object_ids)}
        for object_id, role in action_edges:
            object_idx = id_to_object_idx.get(int(object_id))
            object_node = ("object", object_idx) if object_idx is not None else None
            if role not in excluded_edges and object_node in node_index:
                edges.add(tuple(sorted((node_index[action_node], node_index[object_node]))))

    count = len(nodes)
    if not count:
        return {}, {}, {}, None
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2*np.pi, count, endpoint=False)
    positions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    positions += rng.normal(0, 0.025, positions.shape)
    ideal = 1.6 / np.sqrt(max(count, 1))
    edge_list = list(edges)
    for iteration in range(180):
        delta = positions[:, None, :] - positions[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=2), 1e-3)
        np.fill_diagonal(distance, np.inf)
        displacement = np.sum(delta / distance[:, :, None] * (ideal * ideal / distance)[:, :, None], axis=1)
        for source, destination in edge_list:
            vector = positions[source] - positions[destination]
            edge_distance = max(np.linalg.norm(vector), 1e-3)
            attraction = vector / edge_distance * (edge_distance * edge_distance / ideal)
            displacement[source] -= attraction
            displacement[destination] += attraction
        displacement -= positions * 0.08
        temperature = 0.12 * (1.0 - iteration / 180.0) + 0.005
        norms = np.maximum(np.linalg.norm(displacement, axis=1), 1e-6)
        positions += displacement / norms[:, None] * np.minimum(norms, temperature)[:, None]

    low, high = positions.min(axis=0), positions.max(axis=0)
    span = np.maximum(high-low, 1e-6)
    inset = 75
    canvas = np.empty_like(positions)
    canvas[:, 0] = margin + inset + (positions[:, 0]-low[0]) / span[0] * max(graph_width-2*(margin+inset), 1)
    canvas[:, 1] = margin + inset + (positions[:, 1]-low[1]) / span[1] * max(height-2*(margin+inset), 1)
    canvas = canvas.astype(np.int32)
    objects = {node[1]: canvas[idx] for idx, node in enumerate(nodes) if node[0] == "object"}
    effectors = {node[1]: canvas[idx] for idx, node in enumerate(nodes) if node[0] == "effector"}
    cameras = {node[1]: canvas[idx] for idx, node in enumerate(nodes) if node[0] == "camera"}
    action_point = next((canvas[idx] for idx, node in enumerate(nodes) if node[0] == "action"), None)
    return objects, effectors, cameras, action_point


def render_relation_frame(hdf5_path: Path, frame_idx: int, output_path: Path, width: int, height: int,
                          show_edge_labels: bool = True, excluded_edges: set[str] | None = None,
                          abstract_layout: bool = False, occlusion_camera: str | None = None):
    excluded_edges = excluded_edges or set()
    with h5py.File(hdf5_path, "r") as root:
        object_ids, pose_world, is_present, id_to_meta = _load_object_info(root)
        relations, effector_names, reachable_names, camera_names = _load_relation_info(root)
        gripper_positions = _load_gripper_positions(root)
        action, action_edges = _load_active_action(root, frame_idx)
        if occlusion_camera is None:
            occlusion_camera_idx = 0 if camera_names else None
        elif occlusion_camera not in camera_names:
            raise SystemExit(
                f"Unknown occlusion camera {occlusion_camera!r}; available={camera_names}"
            )
        else:
            occlusion_camera_idx = camera_names.index(occlusion_camera)
        selected_occlusion_camera = (
            camera_names[occlusion_camera_idx]
            if occlusion_camera_idx is not None else None
        )

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

        legend_width = 330
        margin = 40
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        graph_width = width - legend_width
        cv2.rectangle(frame, (margin, margin), (graph_width - margin, height - margin), (220, 220, 220), 1)

        object_canvas = _world_to_canvas(object_points, xlim, ylim, graph_width, height, margin)
        effector_canvas = {}
        for effector_name, positions in gripper_positions.items():
            if frame_idx < len(positions):
                effector_canvas[effector_name] = _world_to_canvas(
                    positions[frame_idx:frame_idx + 1, :2],
                    xlim,
                    ylim,
                    graph_width,
                    height,
                    margin,
                )[0]
        camera_canvas = {
            camera_name: np.array([margin + 25 + camera_idx * 36, margin + 25])
            for camera_idx, camera_name in enumerate(camera_names)
        }
        action_point = np.array([graph_width - margin - 70, margin + 55])
        if abstract_layout:
            abstract_objects, effector_canvas, camera_canvas, abstract_action = _abstract_graph_layout(
                mask, object_ids, relations, frame_idx, effector_names, reachable_names, camera_names,
                action, action_edges, excluded_edges, graph_width, height, margin,
            )
            for object_idx, point in abstract_objects.items():
                object_canvas[object_idx] = point
            if abstract_action is not None:
                action_point = abstract_action

        relation_order = (
            ("near", (120, 120, 120)),
            ("blocks", (180, 80, 20)),
            ("static_contact_with", (110, 110, 110)),
            ("intentional_contact_with", (20, 160, 20)),
            ("robot_collision_with", (220, 80, 20)),
            ("unexpected_collision_with", (220, 20, 20)),
            ("on", (20, 140, 20)),
            ("supports", (150, 90, 20)),
            ("part_of", (150, 0, 150)),
            ("in", (20, 140, 20)),
            ("contains", (20, 160, 90)),
            ("occludes", (40, 80, 230)),
        )
        for relation_name, color in relation_order:
            if relation_name in excluded_edges:
                continue
            relation = relations.get(relation_name)
            if relation is None:
                continue
            if relation_name == "occludes":
                if occlusion_camera_idx is None:
                    continue
                matrix = relation[frame_idx, :, :, occlusion_camera_idx]
            else:
                matrix = relation[frame_idx]
            for i in range(matrix.shape[0]):
                if not mask[i]:
                    continue
                for j in range(matrix.shape[1]):
                    if not matrix[i, j] or not mask[j]:
                        continue
                    if relation_name in {"near", "static_contact_with", "intentional_contact_with", "robot_collision_with", "unexpected_collision_with"} and j <= i:
                        continue
                    edge_label = relation_name
                    if relation_name == "occludes" and selected_occlusion_camera:
                        edge_label = f"occludes[{selected_occlusion_camera}]"
                    _draw_edge(frame, object_canvas[i], object_canvas[j], color, edge_label if show_edge_labels else "",
                               directed=relation_name not in {"near", "static_contact_with", "intentional_contact_with", "robot_collision_with", "unexpected_collision_with"},
                               dashed=relation_name in {"part_of"})

        held_by = relations.get("held_by")
        if held_by is not None and "held_by" not in excluded_edges:
            held_frame = held_by[frame_idx]
            for i in range(held_frame.shape[0]):
                if not mask[i]:
                    continue
                for eff_idx, effector_name in enumerate(effector_names):
                    if eff_idx >= held_frame.shape[1] or not held_frame[i, eff_idx]:
                        continue
                    if effector_name not in effector_canvas:
                        continue
                    _draw_edge(frame, object_canvas[i], effector_canvas[effector_name], (170, 0, 170), "held_by" if show_edge_labels else "")

        reachable = relations.get("reachable_by")
        if reachable is not None and "reachable_by" not in excluded_edges:
            for i, eff_idx in np.argwhere(reachable[frame_idx]):
                if mask[i] and eff_idx < len(reachable_names) and reachable_names[eff_idx] in effector_canvas:
                    _draw_edge(frame, object_canvas[i], effector_canvas[reachable_names[eff_idx]],
                               (0, 150, 210), "reachable_by" if show_edge_labels else "", dashed=True)

        visible = relations.get("visible_to")
        if visible is not None and "visible_to" not in excluded_edges:
            for i, camera_idx in np.argwhere(visible[frame_idx]):
                if mask[i] and camera_idx < len(camera_names):
                    _draw_edge(frame, object_canvas[i], camera_canvas[camera_names[camera_idx]],
                               (220, 120, 20), "visible_to" if show_edge_labels else "", dashed=True)

        action_edge_styles = {
            "agent": ((170, 0, 170), False),
            "target": ((40, 80, 230), False),
            "destination": ((20, 160, 90), True),
        }
        id_to_point = {int(object_ids[i]): object_canvas[i] for i in range(len(object_ids)) if mask[i]}
        if action:
            for object_id, role in action_edges:
                if object_id in id_to_point and role not in excluded_edges:
                    color, dashed = action_edge_styles[role]
                    _draw_edge(frame, action_point, id_to_point[object_id], color, role if show_edge_labels else "", dashed=dashed)

        for effector_name, point in effector_canvas.items():
            _draw_node(frame, point, (170, 0, 170), "diamond")
            cv2.putText(frame, effector_name, (int(point[0] + 8), int(point[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, NODE_LABEL_FONT_SCALE, (170, 0, 170), 1, cv2.LINE_AA)

        for idx, point in enumerate(object_canvas):
            if not mask[idx]:
                continue
            object_id = int(object_ids[idx])
            meta = id_to_meta.get(object_id, {"name": str(object_id), "role": "other"})
            color = _role_color(meta["role"])
            _draw_node(frame, point, color, "square" if meta["role"] == "furniture" else "circle")
            cv2.putText(frame, meta["name"], (int(point[0] + 8), int(point[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, NODE_LABEL_FONT_SCALE, color, 1, cv2.LINE_AA)

        for camera_name, point in camera_canvas.items():
            _draw_node(frame, point, (220, 120, 20), "triangle")
            cv2.putText(frame, camera_name, (int(point[0] + 9), int(point[1] + 4)), cv2.FONT_HERSHEY_SIMPLEX, SMALL_NODE_LABEL_FONT_SCALE, (220, 120, 20), 1, cv2.LINE_AA)
        if action:
            _draw_node(frame, action_point, (0, 150, 210), "diamond", 11)
            cv2.putText(frame, f"A{action['id']} {action['type']}", (int(action_point[0]-45), int(action_point[1]-18)), cv2.FONT_HERSHEY_SIMPLEX, ACTION_NODE_LABEL_FONT_SCALE, (0, 120, 180), 1, cv2.LINE_AA)

        legend = [
            ("near", (120, 120, 120)),
            ("blocks", (180, 80, 20)),
            ("static_contact_with", (110, 110, 110)),
            ("intentional_contact_with", (20, 160, 20)),
            ("robot_collision_with", (220, 80, 20)),
            ("unexpected_collision_with", (220, 20, 20)),
            ("on", (20, 140, 20)),
            ("supports", (150, 90, 20)),
            ("part_of", (150, 0, 150)),
            ("held_by", (170, 0, 170)),
            ("in", (20, 140, 20)),
            ("contains", (20, 160, 90)),
            ("reachable_by", (0, 150, 210)),
            ("visible_to", (220, 120, 20)),
            ("occludes", (40, 80, 230)),
        ]
        legend_relation = {
            "near": "near", "blocks": "blocks", "static_contact_with": "static contact", "intentional_contact_with": "intentional contact", "robot_collision_with": "robot collision", "unexpected_collision_with": "unexpected collision", "on": "on", "supports": "supports",
            "part_of": "part_of", "held_by": "held_by", "reachable_by": "reachable_by",
            "visible_to": "visible_to", "occludes": "occludes",
            "in": "in", "contains": "contains",
        }
        legend = [(label, color) for label, color in legend
                  if label not in legend_relation or legend_relation[label] not in excluded_edges]
        layout_name = "abstract graph layout" if abstract_layout else "physical-pose layout"
        cv2.putText(frame, f"frame {frame_idx} | {layout_name}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        legend_x, legend_y = graph_width + 20, 28
        if action:
            cv2.putText(frame, "ACTIVE ACTION", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, LEGEND_HEADER_SCALE, (40,40,40), 2, cv2.LINE_AA)
            legend_y += 23
            cv2.putText(frame, f"A{action['id']} {action['type']} | {action['phase']}", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, LEGEND_FONT_SCALE, (0,120,180), 1, cv2.LINE_AA)
            legend_y += 19
            cv2.putText(frame, f"frames {action['start']}..{action['end']}", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, LEGEND_FONT_SCALE, (70,70,70), 1, cv2.LINE_AA)
            legend_y += 30
        cv2.putText(frame, "NODE TYPES", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, LEGEND_HEADER_SCALE, (40,40,40), 2, cv2.LINE_AA)
        legend_y += 25
        for label, color, shape in (("target", (40,80,230), "circle"), ("distractor", (0,170,255), "circle"),
                                    ("furniture", (100,100,100), "square"), ("end effector", (170,0,170), "diamond"),
                                    ("camera", (220,120,20), "triangle"), ("action", (0,150,210), "diamond")):
            _draw_node(frame, (legend_x+9, legend_y-5), color, shape)
            cv2.putText(frame, label, (legend_x+28, legend_y), cv2.FONT_HERSHEY_SIMPLEX, LEGEND_FONT_SCALE, (55,55,55), 1, cv2.LINE_AA)
            legend_y += 22
        legend_y += 10
        cv2.putText(frame, "EDGE TYPES", (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, LEGEND_HEADER_SCALE, (40,40,40), 2, cv2.LINE_AA)
        legend_y += 22
        for label, color in legend:
            dashed = label in {"part_of", "reachable_by", "visible_to", "in / contains*"}
            _draw_edge(frame, np.array([legend_x, legend_y-5]), np.array([legend_x+42, legend_y-5]),
                       color, "", dashed=dashed, curved=False)
            cv2.putText(frame, label, (legend_x+52, legend_y), cv2.FONT_HERSHEY_SIMPLEX, LEGEND_FONT_SCALE, (55,55,55), 1, cv2.LINE_AA)
            legend_y += 21
        cv2.putText(frame, "inverse relations curve to opposite sides", (legend_x, min(legend_y+12, height-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90,90,90), 1, cv2.LINE_AA)

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
    parser.add_argument("--show-edge-labels", type=int, choices=(0, 1), default=1,
                        help="Draw labels on scene edges (1=yes, 0=no); the edge legend is retained")
    parser.add_argument("--excluded-edges", default="",
                        help="Comma-separated or bracketed edge names to omit, e.g. '[visible_to,near]'")
    parser.add_argument("--included-edges", default="",
                        help="Comma-separated or bracketed edge allowlist, e.g. '[in,held_by]'")
    parser.add_argument("--abstract-layout", type=int, choices=(0, 1), default=0,
                        help="Use deterministic relation-based node positions instead of physical poses")
    parser.add_argument(
        "--occlusion-camera", default="",
        help="Camera condition for occludes edges; defaults to the first exported camera",
    )
    args = parser.parse_args()

    hdf5_path = Path(args.file).expanduser().resolve()
    if not hdf5_path.exists():
        raise SystemExit(f"HDF5 file not found: {hdf5_path}")

    excluded_edges = _parse_edge_list(args.excluded_edges, "--excluded-edges")
    included_edges = _parse_edge_list(args.included_edges, "--included-edges")
    if excluded_edges and included_edges:
        raise SystemExit("--included-edges and --excluded-edges cannot be used together")
    if included_edges:
        excluded_edges = KNOWN_EDGE_NAMES - included_edges

    output_path = Path(args.output).expanduser().resolve()
    render_relation_frame(hdf5_path, args.frame, output_path, args.width, args.height,
                          show_edge_labels=bool(args.show_edge_labels),
                          excluded_edges=excluded_edges,
                          abstract_layout=bool(args.abstract_layout),
                          occlusion_camera=args.occlusion_camera or None)


if __name__ == "__main__":
    main()
