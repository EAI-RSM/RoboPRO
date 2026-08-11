# S2 — pytest harness

Technical plan for section **S2** of [REHAUL_PLAN.md](REHAUL_PLAN.md).
Working branch: **`peng-training-branch`** (deliberately the old branch — see below). ~0.5 day.

**Status: completed 2026-08-11.** Executed by Codex — see
`agent-memory/feedback_role_boundary.md`: Claude plans, Codex implements, and approving a plan is
not approving execution.

---

## Context

There is currently no way to ask this repo *"did I break anything?"* The only verification that
exists is `--help` on each script (which proves imports resolve and nothing more) plus 12 separate
manual `python -m checks.<module>` invocations that nobody runs as a set.

Those checks are not filler — 2,618 lines of real verification. `test_ring_config` asserts the
occluder formation is byte-identical per (seed, offset-spec), which is what guarantees the scene you
*measured* is the scene you *rolled out*. `test_lib_env_api` exists because commit `a301e2e` deleted
a method `lib/` was calling, nothing caught it, and a full day of GPU runs measured nothing.

**S2 turns those 12 into one command.** Sections S3–S12 restructure 17,000 lines — moving files,
extracting shared drivers, rewiring imports — and without this they would be done blind.

**Why on `peng-training-branch` and not the new one:** a suite is only useful if you know what
passing looked like *before*. Recording the baseline here means anything that flips pass→fail after
the S3 port is something the port broke. Build it after the move and there is nothing to compare
against — you would just be documenting whatever state you landed in.

## What the investigation changed

**1. Most checks are already pytest-shaped.** 10 of 12 modules define module-level `def test_*()`
functions that pytest collects with no modification; each also keeps a `main()` that runs them in
sequence, so `python -m checks.<module>` keeps working. `smoke_test_seed_2a.py` and the load-bearing
CPU AST guard `test_lib_env_api.py` were both `main()`-only and need thin wrappers. **This section
is configuration plus two wrappers, not conversion.**

**2. The "duplicated fixtures" do not exist.** REHAUL_PLAN's Cluster F claims fake-env stubs
duplicated across four checks and `_record()` defined three times. They share *names* and nothing
else:

| Helper | Sites | Signatures |
|---|---|---|
| `_record` | 3 | `(episode, density, config_hash, *, hard_success)` · `(scene, seed, values, clutter, outcome=False)` · `(value)` |
| `_rollout` | 2 | `(episode, *, eps_id, success, collision)` · `(episode=0, density=6)` |
| `_metric` | 2 | `(rollout, eps)` · `(values, *, episode=0)` |
| `_fake_env` | 1 | `test_vla_office_smoke.py` has `_task_api_env(root)` — a different function |

These are correctly-scoped local helpers, not duplication. Under the agreed "merge only
byte-identical" rule, **nothing is merged.** Cluster F in REHAUL_PLAN is wrong and gets corrected.

## Decisions taken

- pytest declared as a **dev-dependency group** in `pyproject.toml`, to be sent upstream as a small PR
  — the fork policy for shared files. `uv pip install` was rejected: `make sync` runs `uv sync`,
  which prunes anything not in the lock, so pytest would vanish unpredictably.
- **`conftest.py` bootstraps the benchmark environment**; use the repository's `.venv` interpreter
  (or activate it) because `set_env.sh` does not change which Python the shell runs.

---

## Step 1 — Declare pytest

`pyproject.toml` has no dev group and no `[tool.pytest]` section. Add only:

```toml
[dependency-groups]
dev = ["pytest>=8"]
```

Then `uv sync` and **inspect the lockfile delta before accepting it**:

```bash
git diff -- uv.lock     # expect only pytest + pluggy/iniconfig/pygments/tomli blocks
```

If `uv.lock` churns beyond those few packages, stop and report — it means the lock was stale and the
sync is doing more than intended.

⚠️ **`pyproject.toml` and `uv.lock` are both SHARED with `origin/dev` and byte-identical today.**
These are the only shared-file edits in S2. Send **both files atomically** upstream (per
`repo_env_and_git.md`, PRs target `8-research-branch`, not `dev`); a `pyproject.toml`-only PR makes
`uv sync --locked` fail. Everything else below is exclusive.

> **→ COMMIT 1 — `Declare pytest as a dev dependency`**
> `pyproject.toml` + `uv.lock` only. Keeping the shared-file change alone in one commit is what makes
> it cherry-pickable for the upstream PR. Nothing exclusive goes in here.

## Step 2 — `bench_script/pytest.ini` (new, exclusive)

```ini
[pytest]
testpaths = checks
python_files = test_*.py smoke_test_*.py
pythonpath = ../../envs/curobo/src
markers =
    gpu: requires CUDA and a real scene build; skipped unless --gpu
```

Both settings are load-bearing:

- **`testpaths = checks`** confines collection. Without it, pytest's default patterns
  (`test_*.py` *and* `*_test.py`) would sweep up ~20 vendored NVIDIA curobo tests under
  `customized_robotwin/envs/curobo/tests/` (all `*_test.py`), plus
  `benchmark/bench_script/test_collision_metrics.py` (442 L — a manual CLI that builds a scene and
  writes MP4s), `customized_robotwin/code_gen/test_gen_code.py`, and
  `customized_robotwin/script/test_render.py`. None of those are tests.
- **`python_files`** adds `smoke_test_*.py`, which matches *neither* default pattern — without it
  `smoke_test_seed_2a.py` is silently never collected.
- **`pythonpath`** keeps collection import-safe immediately after `uv sync`, which prunes the local
  editable CuRobo install before `bootstrap_uv.sh` reinstalls it.

## Step 3 — `bench_script/conftest.py` (new, exclusive)

```python
import os
import pytest
from setup_paths import setup_paths

setup_paths()                                          # derives ROBOTWIN_ROOT / BENCH_ROOT, extends sys.path
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")  # NOT set by setup_paths


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False,
                     help="also run tests marked gpu (needs CUDA)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        return
    skip = pytest.mark.skip(reason="needs --gpu (CUDA + scene build)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
```

Notes on why this works:

- `bench_script/` has no `__init__.py`, so pytest prepends it to `sys.path` and
  `from setup_paths import setup_paths` resolves. `setup_paths()` fills `ROBOTWIN_ROOT` and
  `BENCH_ROOT` from the file's own location when unset, and only fills gaps — explicit exports win.
- `checks/` **does** have `__init__.py`, so pytest imports modules as `checks.test_foo`, exactly the
  path `python -m checks.test_foo` uses. Imports of `lib.*` and `task.*` resolve identically.
- `ROBOTWIN_BENCH_TASK=bench` is set here because `setup_paths` does not set it, and without it the
  system silently loads upstream RoboTwin configs instead of benchmark ones.
- **Skip, not deselect.** Skipped GPU tests stay visible in the output, so the baseline records what
  did not run rather than hiding it.

> **→ COMMIT 2 — `Add pytest config and conftest bootstrap`**
> `bench_script/pytest.ini` + `bench_script/conftest.py`, both new and exclusive. Commit here
> **before** Step 5, which is the unpredictable part — if the GPU triage goes badly, the working
> harness survives.

## Step 4 — Make both `main()`-only checks collectable

Add thin wrappers without changing assertions. The smoke `main()` accepts optional argv so pytest's
own CLI flags are not passed into its `argparse` parser:

```python
@pytest.mark.gpu
def test_seed_2a_smoke():
    main([])
```

Also add `test_lib_env_api(): assert main() == 0`. Both `__main__` blocks stay, so the legacy module
commands are unchanged.

## Step 5 — Classify GPU tests empirically

**Do not guess from imports.** Run the suite and mark based on what actually fails:

```bash
cd customized_robotwin/script/bench_script && ../../../.venv/bin/python -m pytest -q
```

Add `@pytest.mark.gpu` to any test that errors on CUDA/curobo/scene construction, then re-run until
the CPU set is green. Likely candidates from their contents — `test_vla_office_smoke`'s
`test_live_task_roles_and_waypoints`, `test_planner_free_qpos_moves_arms`,
`test_planner_qpos_keeps_topp_path`, and anything in `test_validate_reach_envelope` needing a
`_reach_cache` on disk — but **verify, do not assume**. Mark at test-function granularity, not whole
files, so partially-CPU modules still contribute. Empirically, every existing function was CPU-safe;
only `test_seed_2a_smoke` needed the GPU marker.

> **→ COMMIT 3 — `Collect main-only checks and mark GPU smoke`**
> The two wrappers and the `@pytest.mark.gpu` addition from Step 4. Commit once
> the CPU set is green — that is a real milestone worth a boundary, and it separates "the harness
> works" from "we learned which tests need hardware."

## Step 6 — Record the baseline

Write the pass/skip/fail set into `agent-memory/status_current.md` — the count of passing CPU tests,
which are marked `gpu`, and anything failing for a pre-existing reason. **This record is the
deliverable**; S3 is judged against it.

Also correct REHAUL_PLAN's Cluster F row to say the check helpers are name collisions rather than
duplication, and mark S2 done.

> **→ COMMIT 4 — `Record S2 baseline and mark section done`**
> `agent-memory/status_current.md` + both plan files. Then push to
> `backup/peng-training-branch`, per S1's fork policy — the branch has no tracking ref, so pushes
> are explicit.

---

## Out of scope

- **No fixture merging** — the investigation found no true duplicates.
- **No new tests, no changed assertions, no additional coverage.** S2 makes existing checks runnable;
  it does not make them verify more. A baseline that shifts while being built is worse than none.
- Not fixing tests that already fail — record them as pre-existing and move on.
- No `make test` target: the Makefile is shared, and `conftest.py` makes it unnecessary.
- Nothing on `peng-dev-new`; that branch does not exist until S3.

## Done when

`../../../.venv/bin/python -m pytest -q` from `bench_script/` collects all 12 runnable check modules,
the CPU set is green, the GPU set reports as skipped (not errored, not silently absent),
`python -m checks.<module>` still
works unchanged for at least `test_ring_config` and `test_obstacle_set`, and the baseline is written
into `status_current.md`.

## Test plan

**1. Regression gate.** S2 *creates* the baseline rather than being measured against one, so there
is no prior count to meet. Its output — the pass/skip/fail set — becomes the gate every later
section is judged by.

**2. Commands.**

```bash
cd customized_robotwin/script/bench_script
../../../.venv/bin/python -m pytest -q                  # 49 passed, 1 skipped
../../../.venv/bin/python -m pytest --collect-only -q   # 50 items across 12 modules
../../../.venv/bin/python -m pytest --collect-only -q | grep smoke_test_seed_2a
../../../.venv/bin/python -m checks.test_ring_config    # legacy path unaffected
../../../.venv/bin/python -m checks.test_obstacle_set
```

**3. Equivalence.** S2 must not change what any check verifies. Two guards: no assertion is edited
(the check-file changes are one marker, two thin wrappers, and optional argv plumbing), and
`python -m checks.<module>` must produce the same result as before for at least `test_ring_config`
and `test_obstacle_set`.

**4. What S2 CANNOT verify — state this in the baseline record.**

- **Whether the GPU-marked checks pass.** They are skipped, not run. "Green" after S2 means only
  that the CPU subset passes. Anyone reading the baseline must be able to see how much was skipped.
- **Whether the checks are correct.** S2 proves they *run*, not that they assert the right things.
  Several of the 12 modules had never been recorded as CPU-passing, so this run establishes their
  first unified baseline.
- **Anything about `lib/`, `task/` or the entry points not already covered.** The 12 checks are the
  coverage; S2 adds none. Real blind spots remain — `scripts/`, the Makefile, the install path, the
  eval drivers, `collect_data.py`, and the seed-band contract have no automated verification at all.

This matters for S3: **"pytest is green" will not mean "the port is correct."** It means the CPU
subset of an incomplete suite still passes. Record the skip count next to the pass count so that
distinction is impossible to miss.

## Risks

| Risk | Mitigation |
|---|---|
| Collection escapes `checks/` and pulls in vendored curobo tests | `testpaths = checks`; verify with `--collect-only` |
| `smoke_test_seed_2a.py` silently never collected | `python_files` includes `smoke_test_*.py`; it must appear in `--collect-only` |
| `uv sync` churns the shared lockfile | Inspect the actual `uv.lock` diff; accepted delta is five new blocks: pytest, iniconfig, pluggy, pygments, tomli |
| `uv sync` prunes the editable CuRobo install before bootstrap restores it | Add vendored `envs/curobo/src` to pytest's `pythonpath` so CPU collection remains valid |
| A GPU test errors instead of skipping | Skip via marker in `conftest`, and confirm the count of skips in the baseline |
| Marking whole files GPU hides CPU-testable functions | Mark at function granularity |
| `pytest` run from repo root finds no config and collects everything | Documented as "run from `bench_script/`"; `--collect-only` catches it immediately |

## Rollback

Delete `bench_script/pytest.ini` and `bench_script/conftest.py`, revert both thin wrappers and the
`@pytest.mark.gpu` line, and revert `pyproject.toml` plus `uv.lock`. No check assertion is modified,
so nothing else is affected.
