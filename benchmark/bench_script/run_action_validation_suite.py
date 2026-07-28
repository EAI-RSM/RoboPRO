#!/usr/bin/env python3
"""Collect and validate the schema-1.5 cross-task action suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = REPO_ROOT / "customized_robotwin"
DEFAULT_MATRIX = REPO_ROOT / "benchmark/bench_task_config/action_validation_suite.yml"


def _decode(values):
    return [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in values.tolist()]


def _load_matrix(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle) or {}
    tasks = matrix.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"No tasks defined in {path}")
    task_ids = [task.get('id') for task in tasks]
    if any(not value for value in task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Task ids must be non-empty and unique in {path}")
    for task in tasks:
        if not task.get("task_name"):
            raise ValueError(f"Task {task.get('id')!r} has no task_name")
        if int(task.get("episodes", 1)) <= 0:
            raise ValueError(f"Task {task['id']!r} episodes must be positive")
        acceptance = task.get("acceptance", {}) or {}
        allowed_failed_types = acceptance.get("allowed_failed_action_types", [])
        if not isinstance(allowed_failed_types, list) or any(
            not isinstance(value, str) or not value for value in allowed_failed_types
        ):
            raise ValueError(
                f"allowed_failed_action_types must be a list of names for {task['id']}"
            )
        if int(acceptance.get("max_failed_executed_actions", 0)) < 0:
            raise ValueError(f"max_failed_executed_actions must be non-negative for {task['id']}")
        effect = acceptance.get("require_articulation_joint_effect")
        if effect:
            direction = effect.get("direction")
            if direction not in {None, "increase", "decrease"}:
                raise ValueError(f"Unsupported joint-effect direction {direction!r} for {task['id']}")
            if float(effect.get("min_abs_delta", 0.0)) < 0:
                raise ValueError(f"min_abs_delta must be non-negative for {task['id']}")
        if task.get("expected_validation_failure") and not task.get("expected_failure_patterns"):
            raise ValueError(
                f"Task {task['id']!r} must declare expected_failure_patterns; blanket XFAIL is forbidden"
            )
    return matrix


def _selected_tasks(matrix: dict, requested: str | None) -> list[dict]:
    tasks = matrix["tasks"]
    if not requested:
        return tasks
    wanted = {value.strip() for value in requested.split(",") if value.strip()}
    selected = [task for task in tasks if task['id'] in wanted or task["task_name"] in wanted]
    matched = {value for task in selected for value in (task['id'], task["task_name"])}
    missing = wanted - matched
    if missing:
        raise ValueError(f"Unknown suite task selector(s): {', '.join(sorted(missing))}")
    return selected


def _run_dir(output_root: Path, task: dict, task_config: str) -> Path:
    return output_root / task["task_name"] / task_config


def collect(matrix: dict, tasks: list[dict], args) -> list[dict]:
    task_config = matrix.get("task_config", "relation_validation_d14")
    failures = []
    for task in tasks:
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (task.get("environment") or {}).items()})
        env.update({
            "ROBOTWIN_BENCH_TASK": "bench",
            "ROBOPRO_REACHABLE_BY_FRAME_STRIDE": str(args.reachability_interval),
            "ROBOPRO_NEAR_HORIZONTAL_THRESHOLD_M": str(args.near_horizontal_threshold),
            "ROBOPRO_NEAR_VERTICAL_MARGIN_M": str(args.near_vertical_margin),
            "ROBOPRO_NEAR_MIN_GEOMETRY_EXTENT_M": str(args.near_min_geometry_extent),
            "ROBOPRO_ON_SUPPORTS_MAX_VERTICAL_PENETRATION_M": str(
                args.on_supports_max_vertical_penetration
            ),
            "ROBOPRO_ON_SUPPORTS_MAX_VERTICAL_SEPARATION_M": str(
                args.on_supports_max_vertical_separation
            ),
            "ROBOPRO_ON_SUPPORTS_MIN_XY_OVERLAP_RATIO": str(
                args.on_supports_min_xy_overlap_ratio
            ),
            "ROBOPRO_ON_SUPPORTS_MIN_XY_AREA_M2": str(args.on_supports_min_xy_area),
            "ROBOPRO_IN_CONTAINS_CENTER_TOLERANCE_M": str(
                args.in_contains_center_tolerance
            ),
            "ROBOPRO_RELATION_OBSTACLE_DENSITY": str(args.obstacle_density),
            "ROBOPRO_RELATION_EPISODE_NUM": str(task.get("episodes", args.episodes)),
            "ROBOPRO_HELD_BY_MAX_OBJECT_TCP_DISTANCE_M": str(
                args.held_by_max_object_tcp_distance
            ),
            "ROBOPRO_VISIBLE_TO_MIN_VISIBLE_PIXEL_COUNT": str(
                args.visible_to_min_visible_pixel_count
            ),
            "ROBOPRO_OCCLUDES_MIN_OVERLAP_PIXEL_COUNT": str(
                args.occludes_min_overlap_pixel_count
            ),
            "ROBOPRO_OCCLUDES_MIN_DEPTH_MARGIN_M": str(
                args.occludes_min_depth_margin
            ),
            "ROBOPRO_OCCLUDES_MIN_OVERLAP_FRACTION": str(
                args.occludes_min_overlap_fraction
            ),
            "ROBOPRO_OCCLUDES_MOVABLE_TARGETS_ONLY": (
                "1" if bool(args.occludes_movable_targets_only) else "0"
            ),
            "ROBOPRO_BLOCKS_CORRIDOR_CLEARANCE_M": str(args.blocks_corridor_clearance),
            "ROBOPRO_BLOCKS_ENDPOINT_MARGIN_M": str(args.blocks_endpoint_margin),
            "ROBOPRO_BLOCKS_MOVABLE_SOURCES_ONLY": (
                "1" if bool(args.blocks_movable_sources_only) else "0"
            ),
            "ROBOPRO_BLOCKS_MOVABLE_TARGETS_ONLY": (
                "1" if bool(args.blocks_movable_targets_only) else "0"
            ),
            "ROBOPRO_RELATION_SAVE_PATH": str(args.output_root),
            "COLLECT_START_SEED": str(args.start_seed),
        })
        command = ["bash", "collect_data.sh", task["task_name"], task_config, args.gpu]
        print(f"[collect:{task['id']}] {' '.join(command)}")
        print(f"[collect:{task['id']}] output={_run_dir(args.output_root, task, task_config)}")
        if args.dry_run:
            visible_env = {
                key: env[key] for key in sorted(env)
                if key.startswith("ROBOPRO_") or key == "COLLECT_START_SEED"
            }
            print(f"[collect:{task['id']}] env={json.dumps(visible_env, sort_keys=True)}")
            continue
        try:
            subprocess.run(
                command, cwd=ROBOTWIN_ROOT, env=env, check=True,
                timeout=args.task_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            failures.append({
                "id": task['id'], "task_name": task["task_name"],
                "kind": "timeout", "timeout_seconds": args.task_timeout,
                "command": command, "message": str(exc),
            })
        except subprocess.CalledProcessError as exc:
            failures.append({
                "id": task['id'], "task_name": task["task_name"],
                "kind": "nonzero_exit", "returncode": exc.returncode,
                "command": command, "message": str(exc),
            })
    return failures


def _write_collection_failure_report(args, expected_schema: str, failures: list[dict]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "action_validation_report.json"
    report = {
        "schema_version": expected_schema,
        "output_root": str(args.output_root),
        "status": "COLLECTION_FAILED",
        "collection_failures": failures,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[collect] FAILED ({len(failures)} task(s)); report={report_path}")


def _final_relation_holds(support: h5py.Group, rule: dict) -> bool:
    relation_name = rule["relation"]
    if "relation_state" not in support:
        raise ValueError("Required relation_state group is missing")
    relation_state = support["relation_state"]
    if relation_name not in relation_state:
        raise ValueError(f"Required relation {relation_name!r} is missing")
    if "object_catalog" not in support:
        raise ValueError("Required object_catalog group is missing")
    catalog = support["object_catalog"]
    names = _decode(catalog["names"][()])
    catalog_ids = [int(value) for value in catalog["object_ids"][()]]
    if len(names) != len(catalog_ids):
        raise ValueError("object_catalog names and object_ids lengths differ")
    ids_by_name = {}
    for name, object_id in zip(names, catalog_ids):
        ids_by_name.setdefault(name, []).append(object_id)
    resolved_ids = {}
    for role in ("source_name", "destination_name"):
        requested_name = rule[role]
        matches = ids_by_name.get(requested_name, [])
        if not matches:
            raise ValueError(f"Acceptance object name {requested_name!r} is absent from catalog")
        if len(matches) != 1:
            raise ValueError(
                f"Acceptance object name {requested_name!r} is ambiguous across object ids {matches}"
            )
        resolved_ids[role] = matches[0]

    relation_ids = [int(value) for value in relation_state["object_ids"][()]]
    duplicate_ids = sorted({value for value in relation_ids if relation_ids.count(value) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate relation-state object id(s): {duplicate_ids}")
    relation_index = {object_id: index for index, object_id in enumerate(relation_ids)}
    missing_ids = sorted(set(catalog_ids) - set(relation_index))
    if missing_ids:
        raise ValueError(f"Catalog object id(s) absent from relation_state: {missing_ids}")

    relation = relation_state[relation_name]
    expected_shape = (len(relation_ids), len(relation_ids))
    if relation.ndim != 3 or tuple(relation.shape[1:]) != expected_shape or relation.shape[0] == 0:
        raise ValueError(
            f"Relation {relation_name!r} has shape {relation.shape}; expected "
            f"(T, {expected_shape[0]}, {expected_shape[1]}) with T > 0"
        )
    source = relation_index[resolved_ids["source_name"]]
    destination = relation_index[resolved_ids["destination_name"]]
    return bool(relation[-1, source, destination])


def _validate_action_sequence(action_types, arms, statuses, starts, ends, targets, target_valid):
    failures = []
    if np.any(np.diff(starts) < 0):
        failures.append("action start frames are not monotonic")
    verify_indices = [idx for idx, kind in enumerate(action_types) if kind == "verify_success"]
    if verify_indices != [len(action_types) - 1]:
        failures.append("verify_success must occur exactly once as the terminal action")

    held = {"left": None, "right": None}
    for idx, (kind, arm, status) in enumerate(zip(action_types, arms, statuses)):
        if arm not in held or status != "succeeded":
            continue
        target = int(targets[idx]) if target_valid[idx] else None
        if kind == "grasp":
            if target is None:
                failures.append(f"action {idx} grasp has no target")
            elif held[arm] is not None:
                failures.append(f"action {idx} grasps while {arm} already holds {held[arm]}")
            else:
                held[arm] = target
        elif kind in {"lift", "place", "release"}:
            if held[arm] is None:
                failures.append(f"action {idx} {kind} occurs before a successful grasp on {arm}")
            elif target != held[arm]:
                failures.append(
                    f"action {idx} {kind} target {target} breaks held-object continuity {held[arm]}"
                )
            if kind == "release" and held[arm] is not None and target == held[arm]:
                held[arm] = None
    return failures


def _inspect_episode(path: Path, expected_schema: str, acceptance: dict) -> tuple[dict, list[str]]:
    failures = []
    strict = subprocess.run(
        [sys.executable, str(REPO_ROOT / "benchmark/bench_script/inspect_benchmark_hdf5.py"),
         "--file", str(path)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if strict.returncode:
        detail = strict.stderr.strip().splitlines()[-1] if strict.stderr.strip() else "unknown failure"
        failures.append(f"strict inspector failed: {detail}")
    with h5py.File(path, "r") as root:
        schema = str(root.attrs.get("schema_version", ""))
        success = bool(root.attrs.get("success", False))
        support = root.get("benchmark_support")
        if support is None or "action_nodes" not in support:
            return {"file": str(path), "schema": schema, "success": success}, ["missing action_nodes"]
        actions = support["action_nodes"]
        all_action_types = _decode(actions["action_types"][()])
        all_provenance = _decode(actions["provenance"][()])
        if len(all_action_types) != len(all_provenance):
            raise ValueError("action_types and provenance lengths differ")
        executed_mask = np.asarray(
            [value != "expert_planner_attempt" for value in all_provenance], dtype=np.bool_
        )
        planner_attempt_count = int(np.count_nonzero(~executed_mask))
        if "recorded_frame_count" not in actions:
            raise ValueError("action_nodes/recorded_frame_count is required")
        recorded_frame_count = actions["recorded_frame_count"][()]
        if np.any(recorded_frame_count[~executed_mask] != 0):
            raise ValueError("planner attempts must not own recorded execution frames")

        action_types = [value for value, keep in zip(all_action_types, executed_mask) if keep]
        arms = [value for value, keep in zip(_decode(actions["arms"][()]), executed_mask) if keep]
        statuses = [value for value, keep in zip(_decode(actions["statuses"][()]), executed_mask) if keep]
        starts = actions["start_frame"][()][executed_mask]
        ends = actions["end_frame"][()][executed_mask]
        active = actions["active"][()].astype(bool)[:, executed_mask]
        targets = actions["target_object_id"][()][executed_mask]
        target_valid = actions["target_object_id_valid"][()].astype(bool)[executed_mask]
        executed_arms = {arm for arm, kind in zip(arms, action_types)
                         if arm in {"left", "right"} and kind != "verify_success"}

        if schema != expected_schema:
            failures.append(f"schema {schema!r} != {expected_schema!r}")
        if not success:
            failures.append("episode success is false")
        missing_actions = set(acceptance.get("required_action_types", [])) - set(action_types)
        if missing_actions:
            failures.append("missing action types: " + ", ".join(sorted(missing_actions)))
        missing_arms = set(acceptance.get("required_arms", [])) - executed_arms
        if missing_arms:
            failures.append("missing arms: " + ", ".join(sorted(missing_arms)))
        same_episode = set(acceptance.get("require_same_episode_arms", []))
        if same_episode and not same_episode.issubset(executed_arms):
            failures.append("same episode does not contain both required arms")
        failed_action_types = [
            kind for kind, status in zip(action_types, statuses) if status == "failed"
        ]
        allowed_failed_types = set(acceptance.get("allowed_failed_action_types", []))
        unexpected_failed_types = [
            kind for kind in failed_action_types if kind not in allowed_failed_types
        ]
        if unexpected_failed_types:
            failures.append(
                "unexpected failed executed action type(s): "
                + ", ".join(unexpected_failed_types)
            )
        max_failed = int(acceptance.get("max_failed_executed_actions", 0))
        if len(failed_action_types) > max_failed:
            failures.append(
                f"found {len(failed_action_types)} failed executed action(s), maximum is {max_failed}"
            )
        if np.any(starts > ends):
            failures.append("unordered action interval")
        failures.extend(_validate_action_sequence(
            action_types, arms, statuses, starts, ends, targets, target_valid
        ))

        simultaneous = False
        for frame in range(active.shape[0]):
            frame_arms = {arms[idx] for idx in np.flatnonzero(active[frame])
                          if arms[idx] in {"left", "right"}}
            if len(frame_arms) > 1:
                simultaneous = True
                break
        expected_simultaneous = acceptance.get("simultaneous_dual_arm_expected")
        if expected_simultaneous is not None and simultaneous != bool(expected_simultaneous):
            failures.append(
                f"simultaneous dual-arm={simultaneous}, expected={bool(expected_simultaneous)}"
            )

        if acceptance.get("require_articulated_destination"):
            catalog_ids = support["object_catalog"]["object_ids"][()].tolist()
            articulated = support["object_catalog"]["is_articulated"][()].astype(bool)
            articulated_ids = {int(object_id) for object_id, flag in zip(catalog_ids, articulated) if flag}
            destination = actions["destination_object_id"][()][executed_mask]
            destination_valid = actions["destination_object_id_valid"][()].astype(bool)[executed_mask]
            if not any(int(value) in articulated_ids for value in destination[destination_valid]):
                failures.append("no action references an articulated destination")

        articulated_target_actions = set(acceptance.get("require_articulated_target_actions", []))
        if articulated_target_actions:
            catalog_ids = support["object_catalog"]["object_ids"][()].tolist()
            articulated = support["object_catalog"]["is_articulated"][()].astype(bool)
            articulated_ids = {int(object_id) for object_id, flag in zip(catalog_ids, articulated) if flag}
            targets = actions["target_object_id"][()][executed_mask]
            target_valid = actions["target_object_id_valid"][()].astype(bool)[executed_mask]
            for required_type in articulated_target_actions:
                indices = [idx for idx, kind in enumerate(action_types) if kind == required_type]
                if not indices or not all(
                    target_valid[idx] and int(targets[idx]) in articulated_ids for idx in indices
                ):
                    failures.append(f"{required_type} actions do not all target an articulation")

        interaction_part_actions = set(acceptance.get("require_interaction_part_actions", []))
        parameters = [
            json.loads(value) for value, keep
            in zip(_decode(actions["parameters_json"][()]), executed_mask) if keep
        ]
        for required_type in interaction_part_actions:
            indices = [idx for idx, kind in enumerate(action_types) if kind == required_type]
            if not indices or not all(parameters[idx].get("interaction_part") for idx in indices):
                failures.append(f"{required_type} actions lack interaction_part")

        joint_effect_rule = acceptance.get("require_articulation_joint_effect")
        if joint_effect_rule:
            effects = [
                json.loads(value) for value, keep
                in zip(_decode(actions["observed_effects_json"][()]), executed_mask) if keep
            ]
            required_type = joint_effect_rule["action_type"]
            direction = joint_effect_rule.get("direction")
            min_abs_delta = float(joint_effect_rule.get("min_abs_delta", 0.0))
            matching_deltas = [
                float(effect["delta"])
                for idx, kind in enumerate(action_types) if kind == required_type
                for effect in effects[idx]
                if effect.get("attribute") == "joint_position" and "delta" in effect
            ]
            direction_ok = lambda delta: (
                direction == "decrease" and delta < 0
                or direction == "increase" and delta > 0
                or direction not in {"decrease", "increase"}
            )
            if not any(abs(delta) >= min_abs_delta and direction_ok(delta) for delta in matching_deltas):
                failures.append(
                    f"no {required_type} joint-position effect satisfies "
                    f"direction={direction}, min_abs_delta={min_abs_delta}"
                )

        final_relation = acceptance.get("final_relation")
        if final_relation and not _final_relation_holds(support, final_relation):
            failures.append(
                f"final relation {final_relation['relation']}({final_relation['source_name']}, "
                f"{final_relation['destination_name']}) is false"
            )

        result = {
            "file": str(path),
            "schema": schema,
            "success": success,
            "action_count": len(action_types),
            "planner_attempt_count": planner_attempt_count,
            "action_types": action_types,
            "arms": sorted(executed_arms),
            "failed_action_count": statuses.count("failed"),
            "simultaneous_dual_arm": simultaneous,
        }
    instruction = path.parents[1] / "instructions" / f"{path.stem}.json"
    if not instruction.is_file():
        failures.append(f"missing instruction sidecar {instruction}")
    return result, failures


def check(matrix: dict, tasks: list[dict], args) -> int:
    task_config = matrix.get("task_config", "relation_validation_d14")
    expected_schema = str(matrix.get("schema_version", "1.5.0"))
    report = {"schema_version": expected_schema, "output_root": str(args.output_root), "tasks": []}
    total_failures = 0
    for task in tasks:
        run_dir = _run_dir(args.output_root, task, task_config)
        files = sorted((run_dir / "data").glob("episode*.hdf5"))
        task_report = {
            "id": task['id'], "task_name": task["task_name"],
            "run_dir": str(run_dir), "expected_gaps": task.get("expected_gaps", []),
            "episodes": [], "failures": [],
        }
        expected_count = int(task.get("episodes", args.episodes))
        missing_outputs = len(files) != expected_count
        if missing_outputs:
            task_report["failures"].append(f"found {len(files)} episode(s), expected {expected_count}")
        for path in files:
            episode, failures = _inspect_episode(path, expected_schema, task.get("acceptance", {}))
            episode["failures"] = failures
            task_report["episodes"].append(episode)
            task_report["failures"].extend(f"{path.name}: {failure}" for failure in failures)
        expected_failure = bool(task.get("expected_validation_failure", False))
        expected_patterns = task.get("expected_failure_patterns", [])
        failures_match_expected = bool(task_report["failures"]) and all(
            any(pattern in failure for pattern in expected_patterns)
            for failure in task_report["failures"]
        )
        if expected_failure and not task_report["failures"]:
            task_report["failures"].append("unexpected pass for expected-failure task")
            total_failures += 1
            state = "XPASS"
        elif task_report["failures"] and expected_failure and failures_match_expected and not missing_outputs:
            state = "XFAIL"
        else:
            total_failures += len(task_report["failures"])
            state = "PASS" if not task_report["failures"] else "FAIL"
        task_report["status"] = state
        print(f"[check:{task['id']}] {state} ({len(files)} episode(s))")
        for failure in task_report["failures"]:
            print(f"  - {failure}")
        report["tasks"].append(task_report)

    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "action_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[check] report={report_path}")
    return 1 if total_failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "check", "all"))
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-root", type=Path, default=Path("./data/action_validation_suite_v1"))
    parser.add_argument("--tasks", default=None, help="Comma-separated task ids or task names")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--reachability-interval", type=int, default=10)
    parser.add_argument("--near-horizontal-threshold", type=float, default=0.10)
    parser.add_argument("--near-vertical-margin", type=float, default=0.08)
    parser.add_argument("--near-min-geometry-extent", type=float, default=1e-6)
    parser.add_argument("--on-supports-max-vertical-penetration", type=float, default=0.03)
    parser.add_argument("--on-supports-max-vertical-separation", type=float, default=0.06)
    parser.add_argument("--on-supports-min-xy-overlap-ratio", type=float, default=0.20)
    parser.add_argument("--held-by-max-object-tcp-distance", type=float, default=0.16)
    parser.add_argument("--visible-to-min-visible-pixel-count", type=int, default=1)
    parser.add_argument("--occludes-min-overlap-pixel-count", type=int, default=1)
    parser.add_argument("--occludes-min-depth-margin", type=float, default=1e-3)
    parser.add_argument("--occludes-min-overlap-fraction", type=float, default=0.01)
    parser.add_argument(
        "--occludes-movable-targets-only", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--blocks-corridor-clearance", type=float, default=0.04)
    parser.add_argument("--blocks-endpoint-margin", type=float, default=0.02)
    parser.add_argument(
        "--blocks-movable-sources-only", type=int, choices=(0, 1), default=1
    )
    parser.add_argument(
        "--blocks-movable-targets-only", type=int, choices=(0, 1), default=1
    )
    parser.add_argument("--on-supports-min-xy-area", type=float, default=1e-8)
    parser.add_argument("--in-contains-center-tolerance", type=float, default=1e-4)
    parser.add_argument("--obstacle-density", type=int, default=10)
    parser.add_argument("--task-timeout", type=float, default=1800.0,
                        help="Maximum collection seconds per task")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.matrix = args.matrix.resolve()
    args.output_root = args.output_root.resolve()
    matrix = _load_matrix(args.matrix)
    tasks = _selected_tasks(matrix, args.tasks)
    collection_failures = []
    if args.mode in {"collect", "all"}:
        collection_failures = collect(matrix, tasks, args)
    if collection_failures:
        _write_collection_failure_report(
            args, str(matrix.get("schema_version", "1.5.0")), collection_failures
        )
        return 1
    if args.mode in {"check", "all"} and not args.dry_run:
        return check(matrix, tasks, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
