#!/usr/bin/env python3
"""CPU-only import gate for every ``benchmark/bench_envs`` Python module."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

from setup_paths import setup_paths


BENCH_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_SCRIPT_ROOT.parents[2]
BENCH_ENVS = REPO_ROOT / "benchmark" / "bench_envs"
CHILD_ENV = "ROBOPRO_BENCH_IMPORT_CHILD"


def _module_names() -> list[str]:
    names = []
    for path in sorted(BENCH_ENVS.rglob("*.py")):
        relative = path.relative_to(BENCH_ENVS)
        parts = relative.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(("bench_envs", *parts)))
    return names


def _child_import_all() -> int:
    setup_paths()
    os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/robopro-matplotlib")
    robot_module = types.ModuleType("envs.robot")
    robot_module.Robot = object
    sys.modules["envs.robot"] = robot_module

    failures: list[tuple[str, BaseException]] = []
    names = _module_names()
    for name in names:
        try:
            importlib.import_module(name)
        except BaseException as exc:  # report every broken module in one run
            failures.append((name, exc))
    print(f"{len(names) - len(failures)} imported, {len(failures)} failed")
    for name, exc in failures:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    return int(bool(failures))


def _run_isolated() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env[CHILD_ENV] = "1"
    return subprocess.run(
        [sys.executable, "-m", "checks.test_bench_envs_import"],
        cwd=BENCH_SCRIPT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_bench_envs_import() -> None:
    result = _run_isolated()
    assert result.returncode == 0, result.stdout
    assert "95 imported, 0 failed" in result.stdout


if __name__ == "__main__":
    if os.environ.get(CHILD_ENV) == "1":
        raise SystemExit(_child_import_all())
    outcome = _run_isolated()
    print(outcome.stdout, end="")
    raise SystemExit(outcome.returncode)
