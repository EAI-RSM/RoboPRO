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


def _record_checks(checks: dict, failures: list[str], indent: str = "    ") -> None:
    """Print checks and retain every failed boolean invariant."""
    for key, value in checks.items():
        print(f"{indent}{key}: {value}")
        if isinstance(value, (bool, np.bool_)) and not bool(value):
            failures.append(key)


def _print_attrs(root: h5py.File):
    if not root.attrs:
        print("No top-level attrs found.")
        return
    print("Top-level attrs:")
    for key in sorted(root.attrs.keys()):
        print(f"  {key}: {_decode_attr(root.attrs[key])}")


def _summarize_benchmark_support(root: h5py.File) -> None:
    if "benchmark_support" not in root:
        raise ValueError("Required benchmark_support group is missing")

    support = root["benchmark_support"]
    schema_version = str(_decode_attr(root.attrs.get("schema_version", "0.0.0")))
    try:
        schema_tuple = tuple(int(part) for part in schema_version.split(".")[:3])
    except ValueError:
        schema_tuple = (0, 0, 0)
    requires_policy_contract = schema_tuple >= (1, 6, 0)
    requires_collision_semantics = schema_tuple >= (1, 7, 0)
    requires_occludes = schema_tuple >= (1, 8, 0)
    requires_refined_occludes = schema_tuple >= (1, 8, 1)
    requires_blocks = schema_tuple >= (1, 9, 0)
    failures: list[str] = []
    print("\nbenchmark_support:")
    raw_contact = None
    grasped_by_code = None
    on_matrix = None
    static_contact_with = None
    intentional_contact_with = None
    robot_collision_with = None
    unexpected_collision_with = None
    contact_semantics_valid = None
    held_by = None
    held_by_valid = None
    part_of = None
    part_of_valid = None
    occludes = None
    occludes_valid = None
    occlusion_overlap_pixel_count = None
    occlusion_overlap_fraction = None
    occlusion_source_depth_m = None
    occlusion_target_front_depth_m = None
    occlusion_target_projected_pixel_count = None
    occludes_threshold = 0
    occludes_fraction_threshold = 0.0

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

        _record_checks(checks, failures)

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
        _record_checks(checks, failures)

        if link_names and side_code is not None:
            preview_count = min(len(link_names), 10)
            print("    link preview:")
            for idx in range(preview_count):
                side = "left" if int(side_code[idx]) == 0 else "right"
                chain_pos = int(chain_index[idx]) if chain_index is not None else idx
                parent = int(parent_index[idx]) if parent_index is not None else -1
                print(f"      [{idx}] side={side} chain_index={chain_pos} parent_index={parent} name={link_names[idx]}")

    visible_threshold = 1
    if "relation_parameters" in support:
        parameters = support["relation_parameters"]
        print("  relation_parameters:")
        if "near" in parameters:
            near_parameters = parameters["near"]
            expected = (
                "horizontal_threshold_m",
                "vertical_margin_m",
                "min_geometry_extent_m",
            )
            values = {
                name: float(near_parameters[name][()])
                for name in expected
                if name in near_parameters
            }
            print(f"    near: {values}")
            checks = {
                "near parameter fields complete": set(values) == set(expected),
                "near parameters are finite": all(np.isfinite(value) for value in values.values()),
                "near thresholds are non-negative": all(
                    values.get(name, -1.0) >= 0
                    for name in ("horizontal_threshold_m", "vertical_margin_m")
                ),
                "near minimum extent is positive": values.get("min_geometry_extent_m", 0.0) > 0,
            }
            _record_checks(checks, failures)
        if "on_supports" in parameters:
            support_parameters = parameters["on_supports"]
            expected = (
                "max_vertical_penetration_m",
                "max_vertical_separation_m",
                "min_xy_overlap_ratio",
                "min_xy_area_m2",
            )
            values = {
                name: float(support_parameters[name][()])
                for name in expected
                if name in support_parameters
            }
            print(f"    on_supports: {values}")
            checks = {
                "on/supports parameter fields complete": set(values) == set(expected),
                "on/supports parameters are finite": all(
                    np.isfinite(value) for value in values.values()
                ),
                "on/supports vertical tolerances are non-negative": all(
                    values.get(name, -1.0) >= 0
                    for name in (
                        "max_vertical_penetration_m",
                        "max_vertical_separation_m",
                    )
                ),
                "on/supports overlap ratio is in [0,1]": (
                    0 <= values.get("min_xy_overlap_ratio", -1.0) <= 1
                ),
                "on/supports minimum area is positive": (
                    values.get("min_xy_area_m2", 0.0) > 0
                ),
            }
            _record_checks(checks, failures)
        if "in_contains" in parameters:
            containment_parameters = parameters["in_contains"]
            tolerance = (
                float(containment_parameters["center_tolerance_m"][()])
                if "center_tolerance_m" in containment_parameters else None
            )
            tokens = (
                _decode_string_array(containment_parameters["container_label_tokens"])
                if "container_label_tokens" in containment_parameters else []
            )
            print(
                "    in_contains: "
                f"center_tolerance_m={tolerance}, container_label_tokens={tokens}"
            )
            checks = {
                "in/contains center tolerance exists": tolerance is not None,
                "in/contains center tolerance is finite and non-negative": (
                    tolerance is not None and np.isfinite(tolerance) and tolerance >= 0
                ),
                "in/contains container token vocabulary is non-empty": bool(tokens),
                "in/contains container tokens are unique": len(tokens) == len(set(tokens)),
            }
            _record_checks(checks, failures)
        if "held_by" in parameters:
            held_parameters = parameters["held_by"]
            distance = (
                float(held_parameters["max_object_tcp_distance_m"][()])
                if "max_object_tcp_distance_m" in held_parameters else None
            )
            print(f"    held_by: max_object_tcp_distance_m={distance}")
            checks = {
                "held_by maximum object-TCP distance exists": distance is not None,
                "held_by maximum object-TCP distance is finite and non-negative": (
                    distance is not None and np.isfinite(distance) and distance >= 0
                ),
                "held_by is marked contact-gated": bool(
                    held_parameters.attrs.get("contact_gated", False)
                ),
                "held_by is marked closed-gripper-gated": bool(
                    held_parameters.attrs.get("requires_closed_gripper", False)
                ),
            }
            _record_checks(checks, failures)

        if "contact_semantics" in parameters:
            collision_parameters = parameters["contact_semantics"]
            checks = {
                "contact semantics evidence is raw simulator contact": (
                    str(_decode_attr(collision_parameters.attrs.get("evidence", "")))
                    == "raw_simulator_contact"
                ),
                "contact semantics exclude support contact": (
                    str(_decode_attr(
                        collision_parameters.attrs.get("support_contact_policy", "")
                    )) == "excluded"
                ),
                "collision subtype precedence is declared": (
                    not requires_collision_semantics
                    or str(_decode_attr(
                        collision_parameters.attrs.get("classification_precedence", "")
                    )) == "intentional,static,robot_collision,unexpected"
                ),
                "robot contact mapping is declared": (
                    not requires_collision_semantics
                    or bool(_decode_attr(
                        collision_parameters.attrs.get("robot_contact_mapping", "")
                    ))
                ),
                "baseline static-contact rule is declared": (
                    not requires_collision_semantics
                    or str(_decode_attr(
                        collision_parameters.attrs.get("static_contact_rule", "")
                    )) == "frame0_non_robot_non_support_contact_or_furniture_pair"
                ),
                "baseline static-contact frame is zero": (
                    not requires_collision_semantics
                    or int(collision_parameters.attrs.get("baseline_frame_index", -1)) == 0
                ),
            }
            _record_checks(checks, failures)
        if "blocks" in parameters:
            blocks_parameters = parameters["blocks"]
            clearance = float(blocks_parameters["corridor_clearance_m"][()]) if "corridor_clearance_m" in blocks_parameters else -1.0
            endpoint_margin = float(blocks_parameters["endpoint_margin_m"][()]) if "endpoint_margin_m" in blocks_parameters else -1.0
            print(f"    blocks: corridor_clearance_m={clearance}, endpoint_margin_m={endpoint_margin}")
            _record_checks({
                "blocks corridor clearance is finite and non-negative": np.isfinite(clearance) and clearance >= 0,
                "blocks endpoint margin is finite and non-negative": np.isfinite(endpoint_margin) and endpoint_margin >= 0,
                "blocks provenance is AABB plus effector pose": str(_decode_attr(blocks_parameters.attrs.get("provenance", ""))) == "world_aabb_and_effector_pose",
                "blocks sim-to-real contract is declared": bool(_decode_attr(blocks_parameters.attrs.get("sim_to_real_contract", ""))),
                "blocks limitation is declared": "does not prove" in str(_decode_attr(blocks_parameters.attrs.get("definition", ""))),
            }, failures)
        elif requires_blocks:
            _record_checks({"schema-1.9 blocks parameters are present": False}, failures)

        if "occludes" in parameters:
            occlusion_parameters = parameters["occludes"]
            occludes_threshold = (
                int(occlusion_parameters["min_overlap_pixel_count"][()])
                if "min_overlap_pixel_count" in occlusion_parameters else 0
            )
            occludes_fraction_threshold = (
                float(occlusion_parameters["min_overlap_fraction"][()])
                if "min_overlap_fraction" in occlusion_parameters else -1.0
            )
            depth_margin = (
                float(occlusion_parameters["min_depth_margin_m"][()])
                if "min_depth_margin_m" in occlusion_parameters else None
            )
            print(
                "    occludes: "
                f"min_overlap_pixel_count={occludes_threshold}, "
                f"min_overlap_fraction={occludes_fraction_threshold}, "
                f"min_depth_margin_m={depth_margin}"
            )
            checks = {
                "occludes overlap pixel threshold is positive": (
                    occludes_threshold >= 1
                ),
                "occludes overlap fraction is in [0,1]": (
                    not requires_refined_occludes
                    or (
                        np.isfinite(occludes_fraction_threshold)
                        and 0 <= occludes_fraction_threshold <= 1
                    )
                ),
                "occludes depth margin is finite and non-negative": (
                    depth_margin is not None
                    and np.isfinite(depth_margin)
                    and depth_margin >= 0
                ),
                "occludes provenance is projected AABB plus segmentation": (
                    str(_decode_attr(
                        occlusion_parameters.attrs.get("provenance", "")
                    )) == "privileged_projected_aabb_and_actor_segmentation"
                ),
                "occludes sim-to-real contract is declared": (
                    not requires_refined_occludes
                    or bool(_decode_attr(
                        occlusion_parameters.attrs.get("sim_to_real_contract", "")
                    ))
                ),
                "occludes approximation is declared": bool(
                    _decode_attr(
                        occlusion_parameters.attrs.get("approximation", "")
                    )
                ),
            }
            _record_checks(checks, failures)
        elif requires_occludes:
            _record_checks(
                {"schema-1.8 occludes parameters are present": False},
                failures,
            )
        if "part_of" in parameters:
            part_parameters = parameters["part_of"]
            checks = {
                "part_of provenance is privileged catalog structure": (
                    str(_decode_attr(part_parameters.attrs.get("provenance", "")))
                    == "privileged_catalog_structure"
                ),
                "part_of exports direct membership only": (
                    str(_decode_attr(part_parameters.attrs.get("closure", "")))
                    == "direct_membership_only"
                ),
            }
            _record_checks(checks, failures)
        if "visible_to" in parameters:
            visibility_parameters = parameters["visible_to"]
            visible_threshold = (
                int(visibility_parameters["min_visible_pixel_count"][()])
                if "min_visible_pixel_count" in visibility_parameters else 0
            )
            print(f"    visible_to: min_visible_pixel_count={visible_threshold}")
            checks = {
                "visible_to minimum pixel count is positive": visible_threshold >= 1,
                "visible_to evidence is actor segmentation": (
                    str(_decode_attr(visibility_parameters.attrs.get("evidence", "")))
                    == "actor_segmentation_pixels"
                ),
            }
            _record_checks(checks, failures)

    if "relation_state" in support:
        state = support["relation_state"]
        print("  relation_state:")

        object_ids = state["object_ids"][()].tolist() if "object_ids" in state else []
        raw_contact = state["raw_contact"][()] if "raw_contact" in state else None
        near = state["near"][()] if "near" in state else None
        blocks = state["blocks"][()] if "blocks" in state else None
        blocks_valid = state["blocks_valid"][()] if "blocks_valid" in state else None
        blocks_by_effector = state["blocks_by_effector"][()] if "blocks_by_effector" in state else None
        blocks_by_effector_valid = state["blocks_by_effector_valid"][()] if "blocks_by_effector_valid" in state else None
        grasped_by_code = state["grasped_by_code"][()] if "grasped_by_code" in state else None
        on_matrix = state["on"][()] if "on" in state else None
        in_matrix = state["in"][()] if "in" in state else None
        supports_matrix = state["supports"][()] if "supports" in state else None
        contains_matrix = state["contains"][()] if "contains" in state else None
        containment_valid = state["containment_valid"][()] if "containment_valid" in state else None
        contains_valid = state["contains_valid"][()] if "contains_valid" in state else None
        static_contact_with = state["static_contact_with"][()] if "static_contact_with" in state else None
        intentional_contact_with = state["intentional_contact_with"][()] if "intentional_contact_with" in state else None
        robot_collision_with = state["robot_collision_with"][()] if "robot_collision_with" in state else None
        unexpected_collision_with = state["unexpected_collision_with"][()] if "unexpected_collision_with" in state else None
        contact_semantics_valid = state["contact_semantics_valid"][()] if "contact_semantics_valid" in state else None
        held_by = state["held_by"][()] if "held_by" in state else None
        held_by_valid = state["held_by_valid"][()] if "held_by_valid" in state else None
        reachable_by = state["reachable_by"][()] if "reachable_by" in state else None
        reachable_by_valid = state["reachable_by_valid"][()] if "reachable_by_valid" in state else None
        reachable_by_evaluated = state["reachable_by_evaluated"][()] if "reachable_by_evaluated" in state else None
        visible_to = state["visible_to"][()] if "visible_to" in state else None
        visible_to_valid = state["visible_to_valid"][()] if "visible_to_valid" in state else None
        visible_pixel_count = state["visible_pixel_count"][()] if "visible_pixel_count" in state else None
        occludes = state["occludes"][()] if "occludes" in state else None
        occludes_valid = state["occludes_valid"][()] if "occludes_valid" in state else None
        occlusion_overlap_pixel_count = (
            state["occlusion_overlap_pixel_count"][()]
            if "occlusion_overlap_pixel_count" in state else None
        )
        occlusion_overlap_fraction = (
            state["occlusion_overlap_fraction"][()]
            if "occlusion_overlap_fraction" in state else None
        )
        occlusion_source_depth_m = (
            state["occlusion_source_depth_m"][()]
            if "occlusion_source_depth_m" in state else None
        )
        occlusion_target_front_depth_m = (
            state["occlusion_target_front_depth_m"][()]
            if "occlusion_target_front_depth_m" in state else None
        )
        occlusion_target_projected_pixel_count = (
            state["occlusion_target_projected_pixel_count"][()]
            if "occlusion_target_projected_pixel_count" in state else None
        )
        part_of = state["part_of"][()] if "part_of" in state else None
        part_of_valid = state["part_of_valid"][()] if "part_of_valid" in state else None
        held_by_effector_names = _decode_string_array(state["held_by_effector_names"]) if "held_by_effector_names" in state else []
        reachable_by_effector_names = _decode_string_array(state["reachable_by_effector_names"]) if "reachable_by_effector_names" in state else []
        blocks_effector_names = _decode_string_array(state["blocks_effector_names"]) if "blocks_effector_names" in state else []
        visible_to_camera_names = _decode_string_array(state["visible_to_camera_names"]) if "visible_to_camera_names" in state else []
        canonical_relation_names = _decode_string_array(state["canonical_relation_names"]) if "canonical_relation_names" in state else []
        implemented_relation_names = _decode_string_array(state["implemented_relation_names"]) if "implemented_relation_names" in state else []
        implemented_binary_relation_names = _decode_string_array(state["implemented_binary_relation_names"]) if "implemented_binary_relation_names" in state else []
        implemented_bipartite_relation_names = _decode_string_array(state["implemented_bipartite_relation_names"]) if "implemented_bipartite_relation_names" in state else []
        implemented_camera_conditioned_relation_names = _decode_string_array(
            state["implemented_camera_conditioned_relation_names"]
        ) if "implemented_camera_conditioned_relation_names" in state else []
        auxiliary_relation_state_names = _decode_string_array(state["auxiliary_relation_state_names"]) if "auxiliary_relation_state_names" in state else []

        if raw_contact is not None:
            print(f"    raw_contact shape: {raw_contact.shape}")
        if near is not None:
            print(f"    near shape: {near.shape}")
        if blocks is not None:
            print(f"    blocks shape: {blocks.shape}")
        if blocks_by_effector is not None:
            print(f"    blocks_by_effector shape: {blocks_by_effector.shape}")
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
        if held_by is not None:
            print(f"    held_by shape: {held_by.shape}")
        if reachable_by is not None:
            print(f"    reachable_by shape: {reachable_by.shape}")
        if reachable_by_evaluated is not None:
            print(f"    reachable_by fresh-evaluation frames: {int(np.count_nonzero(reachable_by_evaluated))}/{len(reachable_by_evaluated)}")
        if visible_to is not None:
            print(f"    visible_to shape: {visible_to.shape}")
        if occludes is not None:
            print(f"    occludes shape: {occludes.shape}")
        if part_of is not None:
            print(f"    part_of shape: {part_of.shape}")
        if held_by_effector_names:
            print(f"    held_by_effector_names: {held_by_effector_names}")
        if reachable_by_effector_names:
            print(f"    reachable_by_effector_names: {reachable_by_effector_names}")
        if blocks_effector_names:
            print(f"    blocks_effector_names: {blocks_effector_names}")
        if visible_to_camera_names:
            print(f"    visible_to_camera_names: {visible_to_camera_names}")
        if canonical_relation_names:
            print(f"    canonical_relation_names: {canonical_relation_names}")
        if implemented_relation_names:
            print(f"    implemented_relation_names: {implemented_relation_names}")
        if implemented_binary_relation_names:
            print(f"    implemented_binary_relation_names: {implemented_binary_relation_names}")
        if implemented_bipartite_relation_names:
            print(f"    implemented_bipartite_relation_names: {implemented_bipartite_relation_names}")
        if implemented_camera_conditioned_relation_names:
            print(
                "    implemented_camera_conditioned_relation_names: "
                f"{implemented_camera_conditioned_relation_names}"
            )
        if auxiliary_relation_state_names:
            print(f"    auxiliary_relation_state_names: {auxiliary_relation_state_names}")

        checks = {}
        if raw_contact is not None:
            checks["raw_contact dims are (T,N,N)"] = raw_contact.ndim == 3
            checks["raw_contact is symmetric"] = np.array_equal(
                raw_contact, raw_contact.transpose(0, 2, 1)
            )
            checks["raw_contact has a false diagonal"] = (
                not np.any(np.diagonal(raw_contact, axis1=1, axis2=2))
            )
        if near is not None:
            checks["near dims are (T,N,N)"] = near.ndim == 3
        if requires_blocks:
            checks["schema-1.9 blocks relation is present"] = blocks is not None
            checks["schema-1.9 blocks validity is present"] = blocks_valid is not None
            checks["schema-1.9 per-effector blocks evidence is present"] = blocks_by_effector is not None
        if blocks is not None:
            checks["blocks dims are (T,N,N)"] = blocks.ndim == 3
            checks["blocks has a false object diagonal"] = not np.any(np.diagonal(blocks, axis1=1, axis2=2))
        if blocks_valid is not None:
            checks["blocks_valid matches blocks shape"] = blocks is not None and blocks_valid.shape == blocks.shape
            checks["true blocks edges are valid"] = blocks is not None and np.all(np.logical_or(~blocks, blocks_valid))
        if blocks_by_effector is not None:
            checks["blocks_by_effector dims are (T,N,N,E)"] = blocks_by_effector.ndim == 4
            checks["blocks evidence E matches effector names"] = blocks_by_effector.ndim == 4 and blocks_by_effector.shape[3] == len(blocks_effector_names)
            checks["blocks is union of per-effector evidence"] = blocks is not None and np.array_equal(blocks, np.any(blocks_by_effector, axis=3))
        if blocks_by_effector_valid is not None:
            checks["blocks_by_effector_valid matches evidence shape"] = blocks_by_effector is not None and blocks_by_effector_valid.shape == blocks_by_effector.shape
        if grasped_by_code is not None:
            checks["grasped_by_code dims are (T,N)"] = grasped_by_code.ndim == 2
        if on_matrix is not None:
            checks["on dims are (T,N,N)"] = on_matrix.ndim == 3
        if in_matrix is not None:
            checks["in dims are (T,N,N)"] = in_matrix.ndim == 3
        if supports_matrix is not None:
            checks["supports dims are (T,N,N)"] = supports_matrix.ndim == 3
        collision_subtypes = (
            static_contact_with,
            intentional_contact_with,
            robot_collision_with,
            unexpected_collision_with,
        )
        subtype_names = (
            "static_contact_with",
            "intentional_contact_with",
            "robot_collision_with",
            "unexpected_collision_with",
        )
        if requires_collision_semantics:
            checks["schema-1.7 collision semantic subtypes are present"] = all(
                matrix is not None for matrix in collision_subtypes
            )
            checks["schema-1.7 contact semantic validity is present"] = (
                contact_semantics_valid is not None
            )
        if all(matrix is not None for matrix in collision_subtypes):
            for name, matrix in zip(subtype_names, collision_subtypes):
                checks[f"{name} matches raw_contact shape"] = (
                    raw_contact is not None and matrix.shape == raw_contact.shape
                )
                checks[f"{name} is symmetric"] = np.array_equal(
                    matrix, matrix.transpose(0, 2, 1)
                )
            subtype_count = sum(
                matrix.astype(np.uint8) for matrix in collision_subtypes
            )
            checks["collision semantic subtypes are mutually exclusive"] = bool(
                np.all(subtype_count <= 1)
            )
            expected_non_support_contact = np.logical_and(
                raw_contact,
                np.logical_not(np.logical_or(on_matrix, supports_matrix)),
            )
            checks["collision semantic subtypes partition non-support contact"] = (
                raw_contact is not None and on_matrix is not None
                and supports_matrix is not None
                and np.array_equal(subtype_count > 0, expected_non_support_contact)
            )
        if contact_semantics_valid is not None:
            checks["contact_semantics_valid matches raw_contact shape"] = (
                raw_contact is not None
                and contact_semantics_valid.shape == raw_contact.shape
            )
            checks["contact semantic validity is closed-world"] = bool(
                np.all(contact_semantics_valid)
            )
        if contains_matrix is not None:
            checks["contains is inverse of in"] = (
                in_matrix is not None and np.array_equal(contains_matrix, in_matrix.transpose(0, 2, 1))
            )
        if containment_valid is not None:
            checks["containment_valid matches in shape"] = (
                in_matrix is not None and containment_valid.shape == in_matrix.shape
            )
            checks["true in edges are valid"] = (
                in_matrix is not None and np.all(np.logical_or(~in_matrix, containment_valid))
            )
        if contains_valid is not None:
            checks["contains_valid is inverse of containment_valid"] = (
                containment_valid is not None
                and np.array_equal(contains_valid, containment_valid.transpose(0, 2, 1))
            )
        if held_by is not None:
            checks["held_by dims are (T,N,E)"] = held_by.ndim == 3
        if held_by_valid is not None:
            checks["held_by_valid matches held_by shape"] = (
                held_by is not None and held_by_valid.shape == held_by.shape
            )
            checks["true held_by edges are valid"] = (
                held_by is not None and np.all(np.logical_or(~held_by, held_by_valid))
            )
        if held_by is not None and grasped_by_code is not None:
            expected_codes = np.full(held_by.shape[:2], -1, dtype=np.int8)
            expected_codes[np.logical_and(held_by[:, :, 0], ~held_by[:, :, 1])] = 0
            expected_codes[np.logical_and(~held_by[:, :, 0], held_by[:, :, 1])] = 1
            expected_codes[np.logical_and(held_by[:, :, 0], held_by[:, :, 1])] = 2
            checks["grasped_by_code matches held_by"] = np.array_equal(
                grasped_by_code, expected_codes
            )
        if reachable_by is not None:
            checks["reachable_by dims are (T,N,E)"] = reachable_by.ndim == 3
        if reachable_by_valid is not None:
            checks["reachable_by_valid matches reachable_by shape"] = (
                reachable_by is not None and reachable_by_valid.shape == reachable_by.shape
            )
            checks["true reachable_by edges are valid"] = (
                reachable_by is not None and np.all(np.logical_or(~reachable_by, reachable_by_valid))
            )
        if visible_to is not None:
            checks["visible_to dims are (T,N,C)"] = visible_to.ndim == 3
        if visible_to_valid is not None:
            checks["visible_to_valid matches visible_to shape"] = (
                visible_to is not None and visible_to_valid.shape == visible_to.shape
            )
            checks["true visible_to edges are valid"] = (
                visible_to is not None and np.all(np.logical_or(~visible_to, visible_to_valid))
            )
        if visible_pixel_count is not None:
            checks["visible_pixel_count matches visible_to shape"] = (
                visible_to is not None and visible_pixel_count.shape == visible_to.shape
            )
            checks["visible_to matches configured pixel threshold"] = (
                visible_to is not None
                and np.array_equal(visible_to, visible_pixel_count >= visible_threshold)
            )
        if requires_refined_occludes:
            checks["schema-1.8.1 overlap fraction is present"] = (
                occlusion_overlap_fraction is not None
            )
            checks["schema-1.8.1 source depth is present"] = (
                occlusion_source_depth_m is not None
            )
            checks["schema-1.8.1 target-front depth is present"] = (
                occlusion_target_front_depth_m is not None
            )
            checks["schema-1.8.1 projected target area is present"] = (
                occlusion_target_projected_pixel_count is not None
            )
        if requires_occludes:
            checks["schema-1.8 occludes relation is present"] = occludes is not None
            checks["schema-1.8 occludes validity is present"] = (
                occludes_valid is not None
            )
            checks["schema-1.8 occludes evidence count is present"] = (
                occlusion_overlap_pixel_count is not None
            )
        if occludes is not None:
            checks["occludes dims are (T,N,N,C)"] = occludes.ndim == 4
            checks["occludes has a false object diagonal"] = (
                occludes.ndim == 4
                and not np.any(np.diagonal(occludes, axis1=1, axis2=2))
            )
        if occludes_valid is not None:
            checks["occludes_valid matches occludes shape"] = (
                occludes is not None and occludes_valid.shape == occludes.shape
            )
            checks["true occludes edges are valid"] = (
                occludes is not None
                and np.all(np.logical_or(~occludes, occludes_valid))
            )
        if occlusion_overlap_pixel_count is not None:
            checks["occlusion overlap count matches occludes shape"] = (
                occludes is not None
                and occlusion_overlap_pixel_count.shape == occludes.shape
            )
            checks["occludes edges meet overlap threshold"] = (
                occludes is not None
                and np.all(
                    np.logical_or(
                        ~occludes,
                        occlusion_overlap_pixel_count >= occludes_threshold,
                    )
                )
            )
        if occlusion_overlap_fraction is not None:
            checks["occlusion overlap fraction matches occludes shape"] = (
                occludes is not None
                and occlusion_overlap_fraction.shape == occludes.shape
            )
            checks["occlusion overlap fractions are in [0,1]"] = bool(
                np.all(np.logical_and(
                    occlusion_overlap_fraction >= 0,
                    occlusion_overlap_fraction <= 1,
                ))
            )
            checks["occludes edges meet overlap fraction threshold"] = (
                occludes is not None
                and np.all(np.logical_or(
                    ~occludes,
                    occlusion_overlap_fraction >= occludes_fraction_threshold,
                ))
            )
        if occlusion_source_depth_m is not None:
            checks["occlusion source depth matches occludes shape"] = (
                occludes is not None
                and occlusion_source_depth_m.shape == occludes.shape
            )
            checks["true occludes edges have finite positive source depth"] = (
                occludes is not None
                and np.all(np.logical_or(
                    ~occludes,
                    np.isfinite(occlusion_source_depth_m)
                    & (occlusion_source_depth_m > 0),
                ))
            )
        if occlusion_target_front_depth_m is not None:
            checks["target-front depth dims are (T,N,C)"] = (
                occlusion_target_front_depth_m.ndim == 3
            )
        if occlusion_target_projected_pixel_count is not None:
            checks["target projected pixel count matches target depth shape"] = (
                occlusion_target_front_depth_m is not None
                and occlusion_target_projected_pixel_count.shape
                == occlusion_target_front_depth_m.shape
            )
        if part_of is not None:
            checks["part_of dims are (T,N,N)"] = part_of.ndim == 3
            checks["part_of is constant across frames"] = (
                part_of.shape[0] > 0 and np.all(part_of == part_of[0])
            )
            checks["part_of has no self edges"] = (
                not np.any(np.diagonal(part_of, axis1=1, axis2=2))
            )
            checks["part_of children have at most one direct parent"] = (
                np.all(part_of.sum(axis=2) <= 1)
            )
        if part_of_valid is not None:
            checks["part_of_valid matches part_of shape"] = (
                part_of is not None and part_of_valid.shape == part_of.shape
            )
            checks["true part_of edges are valid"] = (
                part_of is not None and np.all(np.logical_or(~part_of, part_of_valid))
            )
            checks["part_of validity is closed-world"] = bool(np.all(part_of_valid))
        if object_ids and raw_contact is not None:
            checks["same N across object_ids/raw_contact"] = raw_contact.shape[1] == len(object_ids)
        if object_ids and grasped_by_code is not None:
            checks["same N across object_ids/grasped_by_code"] = grasped_by_code.shape[1] == len(object_ids)
        if object_ids and held_by is not None:
            checks["same N across object_ids/held_by"] = held_by.shape[1] == len(object_ids)
        if object_ids and part_of is not None:
            checks["same N across object_ids/part_of"] = part_of.shape[1] == len(object_ids)
        if object_ids and occludes is not None:
            checks["same N across object_ids/occludes"] = (
                occludes.shape[1:3] == (len(object_ids), len(object_ids))
            )
        if held_by is not None and held_by_effector_names:
            checks["same E across held_by/effector_names"] = held_by.shape[2] == len(held_by_effector_names)
        if reachable_by is not None and reachable_by_effector_names:
            checks["same E across reachable_by/effector_names"] = (
                reachable_by.shape[2] == len(reachable_by_effector_names)
            )
        if visible_to is not None and visible_to_camera_names:
            checks["same C across visible_to/camera_names"] = (
                visible_to.shape[2] == len(visible_to_camera_names)
            )
        if occludes is not None and visible_to_camera_names:
            checks["same C across occludes/camera_names"] = (
                occludes.shape[3] == len(visible_to_camera_names)
            )
        catalog_ids = support["object_catalog"]["object_ids"][()].tolist() if "object_catalog" in support else []
        if catalog_ids and object_ids:
            checks["catalog object_ids match relation object_ids"] = catalog_ids == object_ids
        if on_matrix is not None and supports_matrix is not None:
            checks["supports is transpose of on (frame0)"] = np.array_equal(supports_matrix[0], on_matrix[0].T)
        _record_checks(checks, failures)

    if "action_nodes" in support:
        actions = support["action_nodes"]
        print("  action_nodes:")
        action_ids = actions["action_ids"][()] if "action_ids" in actions else np.zeros(0, dtype=np.int64)
        action_types = _decode_string_array(actions["action_types"]) if "action_types" in actions else []
        phases = _decode_string_array(actions["execution_phases"]) if "execution_phases" in actions else []
        arms = _decode_string_array(actions["arms"]) if "arms" in actions else []
        statuses = _decode_string_array(actions["statuses"]) if "statuses" in actions else []
        provenance = _decode_string_array(actions["provenance"]) if "provenance" in actions else []
        recorded_frame_count = (
            actions["recorded_frame_count"][()]
            if "recorded_frame_count" in actions else np.zeros(0, dtype=np.int64)
        )
        starts = actions["start_frame"][()] if "start_frame" in actions else np.zeros(0, dtype=np.int64)
        ends = actions["end_frame"][()] if "end_frame" in actions else np.zeros(0, dtype=np.int64)
        active = actions["active"][()] if "active" in actions else None
        canonical = _decode_string_array(actions["canonical_action_names"]) if "canonical_action_names" in actions else []
        phase_names = _decode_string_array(actions["execution_phase_names"]) if "execution_phase_names" in actions else []
        count = len(action_ids)
        print(f"    count: {count}")
        for idx in range(min(count, 20)):
            print(
                f"    [{int(action_ids[idx])}] {action_types[idx]} phase={phases[idx]} "
                f"arm={arms[idx]} frames={int(starts[idx])}..{int(ends[idx])} status={statuses[idx]}"
            )
        checks = {}
        parallel_lengths = [
            len(values) for values in (
                action_types, phases, arms, statuses, provenance,
                starts, ends, recorded_frame_count,
            )
        ]
        checks["all action fields share A"] = all(length == count for length in parallel_lengths)
        checks["action_ids are contiguous"] = np.array_equal(action_ids, np.arange(count))
        checks["action types are canonical"] = set(action_types).issubset(set(canonical))
        checks["execution phases are canonical"] = set(phases).issubset(set(phase_names))
        checks["statuses are terminal"] = set(statuses).issubset({"succeeded", "failed"})
        checks["action provenance is canonical"] = set(provenance).issubset({
            "expert_planner_attempt", "expert_executed_action", "task_success_check",
        })
        planner_attempt = np.asarray(
            [value == "expert_planner_attempt" for value in provenance], dtype=np.bool_
        )
        checks["planner attempts own no recorded frames"] = bool(
            len(recorded_frame_count) == count
            and np.all(recorded_frame_count[planner_attempt] == 0)
        )
        frame_count = (
            support["object_state"]["pose_world"].shape[0]
            if "object_state" in support and "pose_world" in support["object_state"] else 0
        )
        checks["action intervals are ordered"] = bool(np.all(starts <= ends))
        checks["action intervals are in frame range"] = bool(
            count == 0 or (frame_count > 0 and np.all(starts >= 0) and np.all(ends < frame_count))
        )
        if active is not None:
            checks["active mask shape is (T,A)"] = active.shape == (frame_count, count)
            expected_active = np.zeros((frame_count, count), dtype=np.bool_)
            for idx, (start, end) in enumerate(zip(starts, ends)):
                if len(recorded_frame_count) != count or recorded_frame_count[idx] <= 0:
                    continue
                if frame_count:
                    expected_active[int(start):int(end) + 1, idx] = True
            checks["active mask matches intervals"] = np.array_equal(active, expected_active)
        for dataset_name in (
            "parameters_json", "preconditions_json", "postconditions_json",
            "observed_effects_json",
        ):
            if dataset_name in actions:
                try:
                    decoded = _decode_string_array(actions[dataset_name])
                    checks[f"{dataset_name} is valid JSON"] = (
                        len(decoded) == count and all(json.loads(value) is not None for value in decoded)
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    checks[f"{dataset_name} is valid JSON"] = False

        catalog_ids = set(
            support["object_catalog"]["object_ids"][()].tolist()
            if "object_catalog" in support else []
        )
        for field in ("target_object_id", "destination_object_id", "effector_object_id"):
            valid_name = f"{field}_valid"
            if field in actions and valid_name in actions:
                values = actions[field][()]
                valid = actions[valid_name][()].astype(bool)
                checks[f"{field} validity shape matches"] = len(values) == len(valid) == count
                checks[f"valid {field} references catalog"] = all(
                    int(value) in catalog_ids for value in values[valid]
                )
        if "target_object_id_valid" in actions:
            target_valid = actions["target_object_id_valid"][()].astype(bool)
            target_required = np.array([
                action_type in {"approach", "grasp", "lift", "place", "release", "approach_handle", "grasp_handle", "open_articulation", "close_articulation"}
                for action_type in action_types
            ], dtype=bool)
            checks["grounded manipulation actions have targets"] = bool(
                len(target_valid) == count and np.all(target_valid[target_required])
            )
        if "destination_object_id_valid" in actions:
            destination_valid = actions["destination_object_id_valid"][()].astype(bool)
            destination_required = np.array([
                action_type == "place" for action_type in action_types
            ], dtype=bool)
            checks["place actions have destinations"] = bool(
                len(destination_valid) == count and np.all(destination_valid[destination_required])
            )
        if "effector_object_id_valid" in actions:
            effector_valid = actions["effector_object_id_valid"][()].astype(bool)
            effector_required = np.array([
                action_type != "verify_success" for action_type in action_types
            ], dtype=bool)
            checks["executed actions have effectors"] = bool(
                len(effector_valid) == count and np.all(effector_valid[effector_required])
            )

        if requires_policy_contract:
            checks["policy_action_contract group exists"] = "policy_action_contract" in support
            checks["tool_calls_json exists"] = "tool_calls_json" in actions
        contract_provider = None
        if "policy_action_contract" in support:
            contract = support["policy_action_contract"]
            required_contract_fields = {
                "version", "provider_name", "provider_kind", "action_representation",
                "provider_config_json", "provider_registry_json", "tool_schema_json",
            }
            checks["policy contract fields complete"] = required_contract_fields.issubset(contract.keys())
            if "provider_name" in contract:
                contract_provider = _decode_attr(contract["provider_name"][()])
                checks["provider metadata is consistent"] = (
                    contract_provider == _decode_attr(root.attrs.get("action_provider", ""))
                )
            for field in ("provider_config_json", "provider_registry_json", "tool_schema_json"):
                if field in contract:
                    try:
                        json.loads(_decode_attr(contract[field][()]))
                        checks[f"{field} is valid JSON"] = True
                    except (TypeError, ValueError, json.JSONDecodeError):
                        checks[f"{field} is valid JSON"] = False
        if "tool_calls_json" in actions:
            try:
                tool_calls = [json.loads(value) for value in _decode_string_array(actions["tool_calls_json"])]
                checks["tool calls share A"] = len(tool_calls) == count
                checks["tool calls use ACT decision"] = all(call.get("decision") == "ACT" for call in tool_calls)
                checks["tool calls match provider"] = (
                    contract_provider is not None
                    and all(call.get("provider") == contract_provider for call in tool_calls)
                )
                checks["tool call names are canonical"] = all(
                    call.get("tool", {}).get("name") == "execute_high_level_action"
                    for call in tool_calls
                )
                checks["tool calls align with action nodes"] = bool(
                    len(tool_calls) == count and all(
                        call.get("tool", {}).get("arguments", {}).get("action_type") == action_types[idx]
                        and call.get("tool", {}).get("arguments", {}).get("arm") == arms[idx]
                        for idx, call in enumerate(tool_calls)
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                checks["tool_calls_json is valid JSON"] = False

        if "action_entity_edges" in support:
            edges = support["action_entity_edges"]
            edge_actions = edges["action_id"][()] if "action_id" in edges else np.zeros(0, dtype=np.int64)
            edge_objects = edges["object_id"][()] if "object_id" in edges else np.zeros(0, dtype=np.int64)
            edge_roles = _decode_string_array(edges["roles"]) if "roles" in edges else []
            checks["action edge fields share M"] = len(edge_actions) == len(edge_objects) == len(edge_roles)
            checks["action edges reference action nodes"] = all(int(value) in set(action_ids.tolist()) for value in edge_actions)
            checks["action edges reference catalog objects"] = all(int(value) in catalog_ids for value in edge_objects)
            checks["action edge roles are canonical"] = set(edge_roles).issubset({"agent", "target", "destination"})
            print(f"    action_entity_edges: {len(edge_actions)}")
        _record_checks(checks, failures)

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

        if raw_contact is not None and raw_contact.shape[0] > 0:
            frame0_edges = int(np.count_nonzero(np.triu(raw_contact[0], k=1)))
            print(f"    frame0 contact edges: {frame0_edges}")
        if grasped_by_code is not None and grasped_by_code.shape[0] > 0:
            frame0_grasped = int(np.count_nonzero(grasped_by_code[0] >= 0))
            print(f"    frame0 grasped objects: {frame0_grasped}")
        if on_matrix is not None and on_matrix.shape[0] > 0:
            print(f"    frame0 on edges: {int(np.count_nonzero(on_matrix[0]))}")
        if held_by is not None and held_by.shape[0] > 0:
            print(f"    frame0 held_by edges: {int(np.count_nonzero(held_by[0]))}")
        if part_of is not None and part_of.shape[0] > 0:
            print(f"    frame0 part_of edges: {int(np.count_nonzero(part_of[0]))}")

    if failures:
        raise ValueError("Benchmark invariant failure(s): " + ", ".join(failures))


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
