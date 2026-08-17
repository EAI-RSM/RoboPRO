# The model server pulls in SAPIEN for an unused constant

## Summary

`script/policy_model_server.py` imports one constant it never uses. Because of how the `envs`
package initialises, that single import loads SAPIEN into the process that holds the policy —
a process that never opens a scene, never renders, and never touches physics.

Removing one line lets the model side run without the simulator installed at all.

## Why this matters

The dual-process design is what makes RoboPRO easy to integrate with: the simulator and the
policy talk over a socket, so they can live in different environments. That is a real
benefit, because policy stacks and simulator stacks disagree about dependencies — a current
`openpi` checkout wants `torch>=2.7`, while the benchmark pins `torch==2.4.1` for `pytorch3d`
and CuRobo.

Today that benefit is only partial. A policy author who wants to serve a model must still
install `sapien` (and, on Python 3.11, pin `setuptools<81`, because `sapien 3.0.0b1` imports
the removed `pkg_resources` at module level). Neither is used by the serving process.

## The chain

All paths relative to `customized_robotwin/`.

| # | file:line | code |
|---|---|---|
| 1 | `script/policy_model_server.py:20` | `from envs._GLOBAL_CONFIGS import CONFIGS_PATH` |
| 2 | `envs/__init__.py:1` | `from .utils import *` — runs because importing a submodule initialises the package |
| 3 | `envs/utils/__init__.py:8` | `from .transforms import *` |
| 4 | `envs/utils/transforms.py:4` | `import sapien.core as sapien` |

The destination module, `envs/_GLOBAL_CONFIGS.py`, is pure data — path strings and quaternion
tables — with no side effects that the server depends on. SAPIEN arrives entirely through
step 2.

## What is actually unused

**`CONFIGS_PATH`** appears exactly once in the server, on the import line:

```
$ grep -n "CONFIGS_PATH" script/policy_model_server.py
20:from envs._GLOBAL_CONFIGS import CONFIGS_PATH
```

**`class_decorator`** (`script/policy_model_server.py:191`) instantiates a task environment
and is never called:

```
$ grep -n "class_decorator" script/policy_model_server.py
191:def class_decorator(task_name):
```

It looks like a copy of the same helper in `script/eval_policy_client.py`, where it is used.
On the server it is the only other reference to `envs`, and it would need the whole simulator
stack if it ever ran.

## Proposed change

1. Delete `script/policy_model_server.py:20`.
2. Delete `class_decorator` from `script/policy_model_server.py` (lines 191-196), which
   removes the last `envs` reference from the file.

Nothing else in the file changes.

## Verifying it

In an environment with the policy stack but **without** `sapien` installed:

```bash
cd customized_robotwin
python -c "import ast,sys; ast.parse(open('script/policy_model_server.py').read())"   # syntax
python script/policy_model_server.py --port 9999 --config <policy>.yml --overrides \
    --task_name close_drawer --task_config bench_demo_office_clean --policy_name <policy>
```

Before the change this fails with `ModuleNotFoundError: No module named 'sapien'`. After it,
the server starts and serves actions normally. A full episode against the unchanged client
should produce identical results, since neither removed symbol was reachable.

## Scope

This does not change the client, the environments, or any behaviour on the simulator side. It
only removes two dead references from the serving process.
