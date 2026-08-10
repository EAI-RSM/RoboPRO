#!/usr/bin/env python3
"""Every method lib/ calls on the task through the env object must still exist.

lib/ never imports the task (the dependency rule); it reaches back through a duck-typed
``env`` handle -- ``env._pick_side_grasp_id(...)``, ``env._geometric_grasp_pose(...)``.
Those call sites are INVISIBLE to a by-name reference scan of task/, which is exactly how
a301e2e (2026-07-29, "remove dead task methods") deleted ``_pick_side_grasp_id`` as unused
while lib/ik_grid.py was still calling it.

The failure was silent for a full day of GPU runs: _get_approach_seed catches every
exception so seeding can never break the expert, so the only symptom was an approach-leg
firing rate of 0 and an `exception:AttributeError` buried in seed_stats.reason. Nothing
crashed, and the A/B kept producing numbers that measured nothing.

This is a pure CPU/AST check -- no scene, no curobo, no GPU.

Run from bench_script/:  python -m checks.test_lib_env_api
"""

import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LIB = HERE / "lib"
TASK = HERE / "task"
# Methods the RoboTwin base task provides (envs/_base_task.py and the per-task envs), not
# bench code. Out of scope here: this test guards OUR refactors, and the base task is
# upstream. Listed explicitly so a new name can never be waved through by accident.
BASE_TASK_PROVIDED = {
    "check_success", "close_env", "merge_pkl_to_hdf5_video", "play_once",
    "remove_data_cache", "setup_demo", "_take_picture",
}


def env_calls_in_lib():
    """[(method, file, lineno), ...] for every ``env.<name>(...)`` call under lib/."""
    out = []
    for py in sorted(LIB.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == "env"):
                out.append((f.attr, py.relative_to(HERE), node.lineno))
    return out


def methods_defined_in_task():
    """Every ``def`` name across task/ -- the mixins are all composed onto the task class,
    so a name defined anywhere in the package is reachable as an attribute."""
    names = set()
    for py in sorted(TASK.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
    return names


def main():
    calls = env_calls_in_lib()
    defined = methods_defined_in_task()
    assert calls, "found no env.<method>() calls under lib/ -- the scan is broken"

    missing, checked = [], []
    for name, where, line in calls:
        if name in BASE_TASK_PROVIDED:
            continue
        (checked if name in defined else missing).append((name, where, line))

    print(f"[lib->task] {len(calls)} env.<method>() call sites under lib/; "
          f"{len(BASE_TASK_PROVIDED)} base-task names skipped")
    for name, where, line in sorted(set(checked)):
        print(f"  OK      {name:26} <- {where}:{line}")
    for name, where, line in sorted(set(missing)):
        print(f"  MISSING {name:26} <- {where}:{line}")

    if missing:
        print(f"\nFAIL: {len(set(missing))} method(s) called from lib/ do not exist in task/.")
        print("Do not 'fix' this by deleting the call site -- lib/ is the consumer. Restore")
        print("the method, or move it into lib/ if it belongs there.")
        return 1
    print("\nALL PASS: every lib/ -> task call resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
