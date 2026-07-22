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
    if not matrix.get("tasks"):
        raise ValueError(f"No tasks defined in {path}")
    return matrix


def _selected_tasks(matrix: dict, requested: str | None) -> list[dict]:
    tasks = matrix["tasks"]
    if not requested:
        return tasks
    wanted = {value.strip() for value in requested.split(",") if value.strip()}
    selected = [task for task in tasks if task["id"] in wanted or task["task_name"] in wanted]
    matched = {value for task in selected for value in (task["id"], task["task_name"])}
    missing = wanted - matched
    if missing:
        raise ValueError(f"Unknown suite task selector(s): {', '.join(sorted(missing))}")
    return selected


def _run_dir(output_root: Path, task: dict, task_config: str) -> Path:
    return output_root / task["task_name"] / task_config


def collect(matrix: dict, tasks: list[dict], args) -> None:
    task_config = matrix.get("task_config", "relation_validation_d14")
    for task in tasks:
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in (task.get("environment") or {}).items()})
        env.update({
            "ROBOTWIN_BENCH_TASK": "bench",
            "ROBOPRO_REACHABLE_BY_FRAME_STRIDE": str(args.reachability_interval),
            "ROBOPRO_RELATION_OBSTACLE_DENSITY": str(args.obstacle_density),
            "ROBOPRO_RELATION_EPISODE_NUM": str(task.get("episodes", args.episodes)),
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
        subprocess.run(command, cwd=ROBOTWIN_ROOT, env=env, check=True)


def _final_relation_holds(support: h5py.Group, rule: dict) -> bool:
    relation_name = rule["relation"]
    if "relation_state" not in support or relation_name not in support["relation_state"]:
        return False
    catalog = support["object_catalog"]
    names = _decode(catalog["names"][()])
    by_name = {name: idx for idx, name in enumerate(names)}
    source = by_name.get(rule["source_name"])
    destination = by_name.get(rule["destination_name"])
    if source is None or destination is None:
        return False
    relation = support["relation_state"][relation_name]
    return bool(relation[-1, source, destination])


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
        action_types = _decode(actions["action_types"][()])
        arms = _decode(actions["arms"][()])
        statuses = _decode(actions["statuses"][()])
        starts = actions["start_frame"][()]
        ends = actions["end_frame"][()]
        active = actions["active"][()].astype(bool)
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
        if not acceptance.get("allow_failed_actions", False) and "failed" in statuses:
            failures.append("unexpected failed action node")
        if np.any(starts > ends):
            failures.append("unordered action interval")

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
            destination = actions["destination_object_id"][()]
            destination_valid = actions["destination_object_id_valid"][()].astype(bool)
            if not any(int(value) in articulated_ids for value in destination[destination_valid]):
                failures.append("no action references an articulated destination")

        articulated_target_actions = set(acceptance.get("require_articulated_target_actions", []))
        if articulated_target_actions:
            catalog_ids = support["object_catalog"]["object_ids"][()].tolist()
            articulated = support["object_catalog"]["is_articulated"][()].astype(bool)
            articulated_ids = {int(object_id) for object_id, flag in zip(catalog_ids, articulated) if flag}
            targets = actions["target_object_id"][()]
            target_valid = actions["target_object_id_valid"][()].astype(bool)
            for required_type in articulated_target_actions:
                indices = [idx for idx, kind in enumerate(action_types) if kind == required_type]
                if not indices or not all(
                    target_valid[idx] and int(targets[idx]) in articulated_ids for idx in indices
                ):
                    failures.append(f"{required_type} actions do not all target an articulation")

        interaction_part_actions = set(acceptance.get("require_interaction_part_actions", []))
        parameters = [json.loads(value) for value in _decode(actions["parameters_json"][()])]
        for required_type in interaction_part_actions:
            indices = [idx for idx, kind in enumerate(action_types) if kind == required_type]
            if not indices or not all(parameters[idx].get("interaction_part") for idx in indices):
                failures.append(f"{required_type} actions lack interaction_part")

        joint_effect_rule = acceptance.get("require_articulation_joint_effect")
        if joint_effect_rule:
            effects = [json.loads(value) for value in _decode(actions["observed_effects_json"][()])]
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
            "id": task["id"], "task_name": task["task_name"],
            "run_dir": str(run_dir), "expected_gaps": task.get("expected_gaps", []),
            "episodes": [], "failures": [],
        }
        expected_count = int(task.get("episodes", args.episodes))
        missing_outputs = len(files) < expected_count
        if missing_outputs:
            task_report["failures"].append(f"found {len(files)} episode(s), expected {expected_count}")
        for path in files:
            episode, failures = _inspect_episode(path, expected_schema, task.get("acceptance", {}))
            episode["failures"] = failures
            task_report["episodes"].append(episode)
            task_report["failures"].extend(f"{path.name}: {failure}" for failure in failures)
        expected_failure = bool(task.get("expected_validation_failure", False))
        if task_report["failures"] and expected_failure and not missing_outputs:
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
    parser.add_argument("--obstacle-density", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.matrix = args.matrix.resolve()
    args.output_root = args.output_root.resolve()
    matrix = _load_matrix(args.matrix)
    tasks = _selected_tasks(matrix, args.tasks)
    if args.mode in {"collect", "all"}:
        collect(matrix, tasks, args)
    if args.mode in {"check", "all"} and not args.dry_run:
        return check(matrix, tasks, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
