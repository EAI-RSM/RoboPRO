#!/usr/bin/env python3
"""
Inspect and lightly visualize a RoboPRO benchmark HDF5 export.

Examples:
    python inspect_benchmark_hdf5.py --file /path/to/episode0.hdf5
    python inspect_benchmark_hdf5.py --file /path/to/episode0.hdf5 --show-tree
    python inspect_benchmark_hdf5.py --file /path/to/episode0.hdf5 --camera demo_camera --frame 0 --save-preview /tmp/frame0.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_string_array(dataset):
    values = dataset[()]
    if isinstance(values, np.ndarray):
        return [
            item.decode("utf-8", errors="replace") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in values.tolist()
        ]
    return [str(values)]


def _decode_frame(encoded_blob) -> np.ndarray:
    if isinstance(encoded_blob, (bytes, bytearray)):
        raw = np.frombuffer(encoded_blob.rstrip(b"\0"), dtype=np.uint8)
    else:
        raw = np.frombuffer(bytes(encoded_blob).rstrip(b"\0"), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode frame bytes from HDF5 dataset")
    return frame


def _print_tree(node, prefix=""):
    for key, item in node.items():
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            _print_tree(item, prefix + "  ")
        else:
            print(f"{prefix}{key}  shape={item.shape} dtype={item.dtype}")


def _print_attrs(root: h5py.File):
    if not root.attrs:
        print("No top-level attrs found.")
        return
    print("Top-level attrs:")
    for key in sorted(root.attrs.keys()):
        print(f"  {key}: {_decode_attr(root.attrs[key])}")


def _summarize_benchmark_support(root: h5py.File):
    if "benchmark_support" not in root:
        print("No benchmark_support group found.")
        return

    support = root["benchmark_support"]
    print("\nbenchmark_support:")

    if "scenario_metadata" in support:
        print("  scenario_metadata:")
        scenario = support["scenario_metadata"]
        for key in sorted(scenario.keys()):
            value = scenario[key][()]
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            elif isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O", "U"}:
                value = _decode_string_array(scenario[key])
            elif isinstance(value, np.generic):
                value = value.item()
            print(f"    {key}: {value}")

    if "object_catalog" in support:
        catalog = support["object_catalog"]
        ids = catalog["object_ids"][()].tolist() if "object_ids" in catalog else []
        names = _decode_string_array(catalog["names"]) if "names" in catalog else []
        roles = _decode_string_array(catalog["roles"]) if "roles" in catalog else []
        print(f"  object_catalog: {len(ids)} objects")
        for idx, object_id in enumerate(ids[:20]):
            name = names[idx] if idx < len(names) else ""
            role = roles[idx] if idx < len(roles) else ""
            print(f"    [{idx}] id={object_id} role={role} name={name}")
        if len(ids) > 20:
            print(f"    ... ({len(ids) - 20} more objects)")

    if "object_state" in support:
        state = support["object_state"]
        print("  object_state:")

        object_ids = state["object_ids"][()].tolist() if "object_ids" in state else []
        pose_world = state["pose_world"][()] if "pose_world" in state else None
        is_present = state["is_present"][()] if "is_present" in state else None
        is_target = state["is_target"][()] if "is_target" in state else None
        is_furniture = state["is_furniture"][()] if "is_furniture" in state else None

        if pose_world is not None:
            print(f"    pose_world shape: {pose_world.shape}")
        if is_present is not None:
            print(f"    is_present shape: {is_present.shape}")
        if is_target is not None:
            print(f"    is_target shape: {is_target.shape}")
        if is_furniture is not None:
            print(f"    is_furniture shape: {is_furniture.shape}")

        if object_ids:
            print(f"    object_ids: {len(object_ids)}")

        checks = {}
        if pose_world is not None:
            checks["pose_world dims are (T,N,7)"] = (
                pose_world.ndim == 3 and pose_world.shape[2] == 7
            )
        if is_present is not None:
            checks["is_present dims are (T,N)"] = is_present.ndim == 2
        if pose_world is not None and is_present is not None and object_ids:
            checks["same N across catalog/state"] = (
                pose_world.shape[1] == is_present.shape[1] == len(object_ids)
            )
        if pose_world is not None and pose_world.shape[0] > 1:
            moved = np.abs(np.diff(pose_world[:, :, :3], axis=0)).sum(axis=2)
            checks["objects with any movement"] = int((moved > 1e-6).any(axis=0).sum())

        for key, value in checks.items():
            print(f"    {key}: {value}")

        if pose_world is not None and object_ids:
            preview_count = min(len(object_ids), 5)
            print("    first-frame preview:")
            for idx in range(preview_count):
                pose = pose_world[0, idx].tolist()
                target_flag = bool(is_target[idx]) if is_target is not None and len(is_target) > idx else False
                furniture_flag = bool(is_furniture[idx]) if is_furniture is not None and len(is_furniture) > idx else False
                print(
                    f"      [{idx}] id={object_ids[idx]} "
                    f"target={target_flag} furniture={furniture_flag} pose={pose}"
                )

    if "link_state" in support:
        state = support["link_state"]
        print("  link_state:")

        positions_world = state["positions_world"][()] if "positions_world" in state else None
        side_code = state["side_code"][()] if "side_code" in state else None
        link_names = _decode_string_array(state["link_names"]) if "link_names" in state else []
        chain_index = state["chain_index"][()] if "chain_index" in state else None
        parent_index = state["parent_index"][()] if "parent_index" in state else None

        if positions_world is not None:
            print(f"    positions_world shape: {positions_world.shape}")
        if side_code is not None:
            print(f"    side_code shape: {side_code.shape}")
        if link_names:
            print(f"    link_names: {len(link_names)}")
        if chain_index is not None:
            print(f"    chain_index shape: {chain_index.shape}")
        if parent_index is not None:
            print(f"    parent_index shape: {parent_index.shape}")

        checks = {}
        if positions_world is not None:
            checks["positions_world dims are (T,L,3)"] = (
                positions_world.ndim == 3 and positions_world.shape[2] == 3
            )
        if positions_world is not None and side_code is not None:
            checks["same L across positions/side_code"] = positions_world.shape[1] == len(side_code)
        if positions_world is not None and chain_index is not None:
            checks["same L across positions/chain_index"] = positions_world.shape[1] == len(chain_index)
        if positions_world is not None and parent_index is not None:
            checks["same L across positions/parent_index"] = positions_world.shape[1] == len(parent_index)
        for key, value in checks.items():
            print(f"    {key}: {value}")

        if link_names and side_code is not None:
            preview_count = min(len(link_names), 10)
            print("    link preview:")
            for idx in range(preview_count):
                side = "left" if int(side_code[idx]) == 0 else "right"
                chain_pos = int(chain_index[idx]) if chain_index is not None else idx
                parent = int(parent_index[idx]) if parent_index is not None else -1
                print(f"      [{idx}] side={side} chain_index={chain_pos} parent_index={parent} name={link_names[idx]}")

    if "relation_state" in support:
        state = support["relation_state"]
        print("  relation_state:")

        object_ids = state["object_ids"][()].tolist() if "object_ids" in state else []
        raw_contact = state["raw_contact"][()] if "raw_contact" in state else None
        near = state["near"][()] if "near" in state else None
        grasped_by_code = state["grasped_by_code"][()] if "grasped_by_code" in state else None
        on_matrix = state["on"][()] if "on" in state else None
        in_matrix = state["in"][()] if "in" in state else None
        supports_matrix = state["supports"][()] if "supports" in state else None
        contains_matrix = state["contains"][()] if "contains" in state else None
        collides_with = state["collides_with"][()] if "collides_with" in state else None
        held_by = state["held_by"][()] if "held_by" in state else None
        reachable_by = state["reachable_by"][()] if "reachable_by" in state else None
        visible_to = state["visible_to"][()] if "visible_to" in state else None
        part_of = state["part_of"][()] if "part_of" in state else None
        held_by_effector_names = _decode_string_array(state["held_by_effector_names"]) if "held_by_effector_names" in state else []
        canonical_relation_names = _decode_string_array(state["canonical_relation_names"]) if "canonical_relation_names" in state else []
        implemented_relation_names = _decode_string_array(state["implemented_relation_names"]) if "implemented_relation_names" in state else []
        implemented_binary_relation_names = _decode_string_array(state["implemented_binary_relation_names"]) if "implemented_binary_relation_names" in state else []
        implemented_bipartite_relation_names = _decode_string_array(state["implemented_bipartite_relation_names"]) if "implemented_bipartite_relation_names" in state else []
        auxiliary_relation_state_names = _decode_string_array(state["auxiliary_relation_state_names"]) if "auxiliary_relation_state_names" in state else []

        if raw_contact is not None:
            print(f"    raw_contact shape: {raw_contact.shape}")
        if near is not None:
            print(f"    near shape: {near.shape}")
        if grasped_by_code is not None:
            print(f"    grasped_by_code shape: {grasped_by_code.shape}")
        if on_matrix is not None:
            print(f"    on shape: {on_matrix.shape}")
        if in_matrix is not None:
            print(f"    in shape: {in_matrix.shape}")
        if supports_matrix is not None:
            print(f"    supports shape: {supports_matrix.shape}")
        if contains_matrix is not None:
            print(f"    contains shape: {contains_matrix.shape}")
        if collides_with is not None:
            print(f"    collides_with shape: {collides_with.shape}")
        if held_by is not None:
            print(f"    held_by shape: {held_by.shape}")
        if reachable_by is not None:
            print(f"    reachable_by shape: {reachable_by.shape}")
        if visible_to is not None:
            print(f"    visible_to shape: {visible_to.shape}")
        if part_of is not None:
            print(f"    part_of shape: {part_of.shape}")
        if held_by_effector_names:
            print(f"    held_by_effector_names: {held_by_effector_names}")
        if canonical_relation_names:
            print(f"    canonical_relation_names: {canonical_relation_names}")
        if implemented_relation_names:
            print(f"    implemented_relation_names: {implemented_relation_names}")
        if implemented_binary_relation_names:
            print(f"    implemented_binary_relation_names: {implemented_binary_relation_names}")
        if implemented_bipartite_relation_names:
            print(f"    implemented_bipartite_relation_names: {implemented_bipartite_relation_names}")
        if auxiliary_relation_state_names:
            print(f"    auxiliary_relation_state_names: {auxiliary_relation_state_names}")

        checks = {}
        if raw_contact is not None:
            checks["raw_contact dims are (T,N,N)"] = raw_contact.ndim == 3
        if near is not None:
            checks["near dims are (T,N,N)"] = near.ndim == 3
        if grasped_by_code is not None:
            checks["grasped_by_code dims are (T,N)"] = grasped_by_code.ndim == 2
        if on_matrix is not None:
            checks["on dims are (T,N,N)"] = on_matrix.ndim == 3
        if in_matrix is not None:
            checks["in dims are (T,N,N)"] = in_matrix.ndim == 3
        if supports_matrix is not None:
            checks["supports dims are (T,N,N)"] = supports_matrix.ndim == 3
        if collides_with is not None:
            checks["collides_with dims are (T,N,N)"] = collides_with.ndim == 3
        if contains_matrix is not None:
            checks["contains is inverse of in"] = (
                in_matrix is not None and np.array_equal(contains_matrix, in_matrix.transpose(0, 2, 1))
            )
        if held_by is not None:
            checks["held_by dims are (T,N,E)"] = held_by.ndim == 3
        if reachable_by is not None:
            checks["reachable_by dims are (T,N,E)"] = reachable_by.ndim == 3
        if visible_to is not None:
            checks["visible_to dims are (T,N,C)"] = visible_to.ndim == 3
        if part_of is not None:
            checks["part_of dims are (T,N,N)"] = part_of.ndim == 3
        if object_ids and raw_contact is not None:
            checks["same N across object_ids/raw_contact"] = raw_contact.shape[1] == len(object_ids)
        if object_ids and grasped_by_code is not None:
            checks["same N across object_ids/grasped_by_code"] = grasped_by_code.shape[1] == len(object_ids)
        if object_ids and held_by is not None:
            checks["same N across object_ids/held_by"] = held_by.shape[1] == len(object_ids)
        if object_ids and part_of is not None:
            checks["same N across object_ids/part_of"] = part_of.shape[1] == len(object_ids)
        if held_by is not None and held_by_effector_names:
            checks["same E across held_by/effector_names"] = held_by.shape[2] == len(held_by_effector_names)
        if on_matrix is not None and supports_matrix is not None:
            checks["supports is transpose of on (frame0)"] = np.array_equal(supports_matrix[0], on_matrix[0].T)
        for key, value in checks.items():
            print(f"    {key}: {value}")

    if "collision_metric_contact_events" in support:
        events = support["collision_metric_contact_events"]
        print("  collision_metric_contact_events:")
        count = len(events["t_step"]) if "t_step" in events else 0
        print(f"    count: {count}")
        if "event_semantics" in events:
            print(f"    event_semantics: {_decode_string_array(events['event_semantics'])}")
        if "event_type" in events and count:
            event_types = sorted(set(_decode_string_array(events["event_type"])))
            print(f"    event_types: {event_types}")

        if contact is not None and contact.shape[0] > 0:
            frame0_edges = int(np.count_nonzero(np.triu(contact[0], k=1)))
            print(f"    frame0 contact edges: {frame0_edges}")
        if grasped_by_code is not None and grasped_by_code.shape[0] > 0:
            frame0_grasped = int(np.count_nonzero(grasped_by_code[0] >= 0))
            print(f"    frame0 grasped objects: {frame0_grasped}")
        if on_matrix is not None and on_matrix.shape[0] > 0:
            print(f"    frame0 on edges: {int(np.count_nonzero(on_matrix[0]))}")
        if collides_with is not None and collides_with.shape[0] > 0:
            print(f"    frame0 collides_with edges: {int(np.count_nonzero(np.triu(collides_with[0], k=1)))}")
        if held_by is not None and held_by.shape[0] > 0:
            print(f"    frame0 held_by edges: {int(np.count_nonzero(held_by[0]))}")
        if part_of is not None and part_of.shape[0] > 0:
            print(f"    frame0 part_of edges: {int(np.count_nonzero(part_of[0]))}")


def _list_cameras(root: h5py.File) -> list[str]:
    if "observation" not in root:
        return []
    return [key for key in root["observation"].keys() if isinstance(root["observation"][key], h5py.Group) and "rgb" in root["observation"][key]]


def _save_preview(root: h5py.File, camera: str, frame_idx: int, save_path: Path | None):
    cameras = _list_cameras(root)
    if not cameras:
        print("\nNo RGB camera groups found under /observation.")
        return

    selected_camera = camera or cameras[0]
    if selected_camera not in cameras:
        raise SystemExit(f"Camera '{selected_camera}' not found. Available cameras: {', '.join(cameras)}")

    rgb_ds = root["observation"][selected_camera]["rgb"]
    num_frames = len(rgb_ds)
    if frame_idx < 0 or frame_idx >= num_frames:
        raise SystemExit(f"Frame index {frame_idx} out of range for {selected_camera} (num_frames={num_frames})")

    frame = _decode_frame(rgb_ds[frame_idx])
    print(f"\nPreview camera={selected_camera} frame={frame_idx} shape={frame.shape}")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), frame):
            raise SystemExit(f"Failed to write preview image to {save_path}")
        print(f"Saved preview image to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect a RoboPRO benchmark HDF5 export")
    parser.add_argument("--file", required=True, help="Path to the HDF5 episode file")
    parser.add_argument("--show-tree", action="store_true", help="Print the full HDF5 tree")
    parser.add_argument("--camera", default=None, help="Camera name for preview extraction")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for preview extraction")
    parser.add_argument("--save-preview", default=None, help="If set, decode and save a preview image to this path")
    parser.add_argument("--dump-json", action="store_true", help="Dump a compact JSON summary to stdout")
    args = parser.parse_args()

    hdf5_path = Path(args.file).expanduser().resolve()
    if not hdf5_path.exists():
        raise SystemExit(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as root:
        print(f"File: {hdf5_path}")
        _print_attrs(root)
        cameras = _list_cameras(root)
        if cameras:
            print(f"\nAvailable RGB cameras: {', '.join(cameras)}")
        _summarize_benchmark_support(root)

        if args.show_tree:
            print("\nHDF5 tree:")
            _print_tree(root, prefix="  ")

        if args.save_preview:
            _save_preview(root, args.camera, args.frame, Path(args.save_preview))
        elif args.camera is not None:
            _save_preview(root, args.camera, args.frame, None)

        if args.dump_json:
            summary = {
                "file": str(hdf5_path),
                "attrs": {key: _decode_attr(root.attrs[key]) for key in root.attrs.keys()},
                "cameras": cameras,
            }
            if "benchmark_support" in root:
                support = root["benchmark_support"]
                if "scenario_metadata" in support:
                    scenario = {}
                    for key in support["scenario_metadata"].keys():
                        value = support["scenario_metadata"][key][()]
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        elif isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O", "U"}:
                            value = _decode_string_array(support["scenario_metadata"][key])
                        elif isinstance(value, np.generic):
                            value = value.item()
                        scenario[key] = value
                    summary["scenario_metadata"] = scenario
                if "object_catalog" in support:
                    summary["object_count"] = len(support["object_catalog"]["object_ids"])
                if "object_state" in support:
                    state = support["object_state"]
                    object_state_summary = {}
                    if "object_ids" in state:
                        object_state_summary["object_count"] = len(state["object_ids"])
                    if "pose_world" in state:
                        pose_world = state["pose_world"]
                        object_state_summary["pose_world_shape"] = list(pose_world.shape)
                        if pose_world.ndim == 3 and pose_world.shape[0] > 1:
                            moved = np.abs(np.diff(pose_world[(), :, :3], axis=0)).sum(axis=2)
                            object_state_summary["objects_with_any_movement"] = int((moved > 1e-6).any(axis=0).sum())
                    if "is_present" in state:
                        object_state_summary["is_present_shape"] = list(state["is_present"].shape)
                    summary["object_state"] = object_state_summary
                if "link_state" in support:
                    state = support["link_state"]
                    link_state_summary = {}
                    if "positions_world" in state:
                        link_state_summary["positions_world_shape"] = list(state["positions_world"].shape)
                    if "side_code" in state:
                        link_state_summary["side_code_shape"] = list(state["side_code"].shape)
                    if "chain_index" in state:
                        link_state_summary["chain_index_shape"] = list(state["chain_index"].shape)
                    if "parent_index" in state:
                        link_state_summary["parent_index_shape"] = list(state["parent_index"].shape)
                    if "link_names" in state:
                        link_state_summary["link_count"] = len(state["link_names"])
                    summary["link_state"] = link_state_summary
                if "relation_state" in support:
                    state = support["relation_state"]
                    relation_state_summary = {}
                    if "object_ids" in state:
                        relation_state_summary["object_count"] = len(state["object_ids"])
                    if "raw_contact" in state:
                        raw_contact = state["raw_contact"][()]
                        relation_state_summary["raw_contact_shape"] = list(raw_contact.shape)
                        if raw_contact.ndim == 3 and raw_contact.shape[0] > 0:
                            relation_state_summary["frame0_raw_contact_edges"] = int(np.count_nonzero(np.triu(raw_contact[0], k=1)))
                    if "near" in state:
                        relation_state_summary["near_shape"] = list(state["near"].shape)
                    if "grasped_by_code" in state:
                        grasped_by_code = state["grasped_by_code"][()]
                        relation_state_summary["grasped_by_code_shape"] = list(grasped_by_code.shape)
                        if grasped_by_code.ndim == 2 and grasped_by_code.shape[0] > 0:
                            relation_state_summary["frame0_grasped_objects"] = int(np.count_nonzero(grasped_by_code[0] >= 0))
                    if "on" in state:
                        on_matrix = state["on"][()]
                        relation_state_summary["on_shape"] = list(on_matrix.shape)
                        if on_matrix.ndim == 3 and on_matrix.shape[0] > 0:
                            relation_state_summary["frame0_on_edges"] = int(np.count_nonzero(on_matrix[0]))
                    if "in" in state:
                        in_matrix = state["in"][()]
                        relation_state_summary["in_shape"] = list(in_matrix.shape)
                        if in_matrix.ndim == 3 and in_matrix.shape[0] > 0:
                            relation_state_summary["frame0_in_edges"] = int(np.count_nonzero(in_matrix[0]))
                    if "supports" in state:
                        relation_state_summary["supports_shape"] = list(state["supports"].shape)
                    if "contains" in state:
                        relation_state_summary["contains_shape"] = list(state["contains"].shape)
                    if "collides_with" in state:
                        collides_with = state["collides_with"][()]
                        relation_state_summary["collides_with_shape"] = list(collides_with.shape)
                        if collides_with.ndim == 3 and collides_with.shape[0] > 0:
                            relation_state_summary["frame0_collides_with_edges"] = int(np.count_nonzero(np.triu(collides_with[0], k=1)))
                    if "held_by" in state:
                        held_by = state["held_by"][()]
                        relation_state_summary["held_by_shape"] = list(held_by.shape)
                        if held_by.ndim == 3 and held_by.shape[0] > 0:
                            relation_state_summary["frame0_held_by_edges"] = int(np.count_nonzero(held_by[0]))
                    if "reachable_by" in state:
                        reachable_by = state["reachable_by"][()]
                        relation_state_summary["reachable_by_shape"] = list(reachable_by.shape)
                        if reachable_by.ndim == 3 and reachable_by.shape[0] > 0:
                            relation_state_summary["frame0_reachable_by_edges"] = int(np.count_nonzero(reachable_by[0]))
                    if "visible_to" in state:
                        visible_to = state["visible_to"][()]
                        relation_state_summary["visible_to_shape"] = list(visible_to.shape)
                        if visible_to.ndim == 3 and visible_to.shape[0] > 0:
                            relation_state_summary["frame0_visible_to_edges"] = int(np.count_nonzero(visible_to[0]))
                    if "part_of" in state:
                        part_of = state["part_of"][()]
                        relation_state_summary["part_of_shape"] = list(part_of.shape)
                        if part_of.ndim == 3 and part_of.shape[0] > 0:
                            relation_state_summary["frame0_part_of_edges"] = int(np.count_nonzero(part_of[0]))
                    if "held_by_effector_names" in state:
                        relation_state_summary["held_by_effector_names"] = _decode_string_array(state["held_by_effector_names"])
                    if "canonical_relation_names" in state:
                        relation_state_summary["canonical_relation_names"] = _decode_string_array(state["canonical_relation_names"])
                    if "implemented_relation_names" in state:
                        relation_state_summary["implemented_relation_names"] = _decode_string_array(state["implemented_relation_names"])
                    if "implemented_binary_relation_names" in state:
                        relation_state_summary["implemented_binary_relation_names"] = _decode_string_array(state["implemented_binary_relation_names"])
                    if "implemented_bipartite_relation_names" in state:
                        relation_state_summary["implemented_bipartite_relation_names"] = _decode_string_array(state["implemented_bipartite_relation_names"])
                    if "auxiliary_relation_state_names" in state:
                        relation_state_summary["auxiliary_relation_state_names"] = _decode_string_array(state["auxiliary_relation_state_names"])
                    summary["relation_state"] = relation_state_summary
                if "collision_metric_contact_events" in support:
                    events = support["collision_metric_contact_events"]
                    events_summary = {}
                    if "t_step" in events:
                        events_summary["count"] = len(events["t_step"])
                    if "event_semantics" in events:
                        events_summary["event_semantics"] = _decode_string_array(events["event_semantics"])
                    if "event_type" in events:
                        events_summary["event_types"] = sorted(set(_decode_string_array(events["event_type"])))
                    summary["collision_metric_contact_events"] = events_summary
            print("\nJSON summary:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
