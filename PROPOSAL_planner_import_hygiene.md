# Planning-stack imports are eager, and CuRobo's optionality does not hold

## Summary

Three separate things on the simulator side, in increasing order of effort:

1. `envs/_base_task.py` imports `toppra` to silence a logger and for nothing else.
2. `envs/robot/planner.py` imports `mplib` at module level, so it is required even by runs
   that never plan.
3. CuRobo is *presented* as optional — its import sits in a `try`/`except` that prints advice
   and continues — but `envs/robot/robot.py` then imports `CuroboPlanner` at module level, so
   a missing CuRobo raises `ImportError` a moment later instead of degrading.

The third is the one that misleads: the code says CuRobo is optional and behaves as if it is
mandatory.

## What is *not* proposed

CuRobo is **not** removable from evaluation. With `enable_collision_metrics: true` — the
setting the eval task configs ship, e.g. `benchmark/bench_task_config/bench_demo_office_clean.yml:46` —
`_bench_base_task.update_world()` (line 2006) pushes the scene into CuRobo's collision world
model, and that is where the `collision`, `collision_count` and `hard_success` fields in
`_episodes.jsonl` come from. Policy rollout does not *plan* — `_base_task.take_action()` with
`action_type='qpos'` writes joint targets directly — but it does depend on CuRobo for
collision accounting.

So this is about import hygiene and honest optionality, not about dropping a dependency.

## 1. `toppra` in `_base_task.py`

`envs/_base_task.py:9` imports it:

```python
import toppra as ta
```

Its only use in the file is line 223:

```python
ta.setup_logging("CRITICAL")  # hide logging
```

That is a logging-suppression call for a library the file does not otherwise use. `toppra`
does real work in `envs/robot/planner.py` and `envs/robot/robot.py`, which import it
themselves.

**Proposed change**: remove the import and line 223 from `envs/_base_task.py`. If the log
suppression is still wanted, move it next to a real `toppra` user — `envs/robot/planner.py`
already imports the library at line 8.

## 2. `mplib` at module level in `planner.py`

`envs/robot/planner.py:1-9`:

```python
import mplib.planner
import mplib
...
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
```

These serve `MplibPlanner`, defined near the bottom of the file. Runs that use `CuroboPlanner`
never touch it, but the module cannot be imported without `mplib` present.

**Proposed change**: move the three `mplib` imports inside `MplibPlanner.__init__` (or behind
a module-level `try`/`except` that mirrors the CuRobo one and raises only on use). The file
already establishes this pattern — `envs/robot/robot.py:805` does a local
`from .planner import CuroboPlanner` inside a function.

## 3. CuRobo's optionality is defeated one file away

`envs/robot/planner.py:17` opens a `try` covering the CuRobo imports and the whole
`CuroboPlanner` class, ending at line 806:

```python
except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! ...')
    traceback.print_exc()
```

The intent is clear: carry on without CuRobo. But when that branch is taken, `CuroboPlanner`
is never defined, and `envs/robot/robot.py:15` runs unconditionally:

```python
from .planner import CuroboPlanner
```

which raises `ImportError: cannot import name 'CuroboPlanner'`. The user sees the friendly
message *and* a crash, one traceback after the other.

`Robot.__init__` then constructs two planners eagerly (`envs/robot/robot.py:271` and `276`),
whether or not the run will plan.

**Proposed change**, smallest version that makes the stated contract true:

1. In `planner.py`, set a module-level flag in both branches, e.g. `CUROBO_AVAILABLE = True`
   in the `try` and `False` in the `except`, and define `CuroboPlanner = None` in the
   `except` so the name always exists.
2. In `robot.py`, import the flag alongside the class and guard construction at lines 271-276
   on it, leaving `self.left_planner = self.right_planner = None`.
3. Raise a clear error at the point of use — `plan_path`, `plan_grippers`, `update_world` —
   rather than at import: *"this run needs CuRobo (collision metrics or motion planning are
   enabled); install it from https://github.com/NVlabs/curobo"*.

A larger version would defer construction until first use, which also saves the CuRobo warmup
on runs that never plan. That is worth doing separately, since it changes startup timing and
wants its own before/after numbers.

## Effect

With 1-3 applied, a run that neither plans nor measures collisions imports the environments
without `toppra`, `mplib` or CuRobo, and a run that does need them fails with one clear
message instead of a printed apology followed by an `ImportError`.

## Verifying it

- **Unchanged behaviour** on a real evaluation: run one task with
  `enable_collision_metrics: true` before and after, and compare `_episodes.jsonl`
  seed-for-seed — the `success`, `collision` and `hard_success` fields must match.
- **The optionality claim**: in an environment without CuRobo installed, importing a task
  module should succeed, and starting a run with collision metrics enabled should fail with
  the new explicit message rather than an `ImportError` from `robot.py:15`.
- **`mplib` laziness**: in an environment without `mplib`, `import envs.robot.planner` should
  succeed and only constructing `MplibPlanner` should fail.
