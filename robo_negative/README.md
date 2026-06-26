# robo_negative

Targeted negative-data generation for RoboPRO — a twin-pair **desync** library. It produces
*labelled failure* demonstrations by causally steering the scripted expert into specific,
named failure modes, and logs enough state to label (and replay) every episode offline.

See the end-to-end walk-through in [`negative_data_demo.ipynb`](../negative_data_demo.ipynb)
and the project [`README.md`](../README.md#targeted-negative-data-failure-generation).

## Idea

For each (task, scene, seed) it records a **baseline** — a clean scripted-expert success — and
one or more **perturbed twins** that fail in a *known* way. The twin is not random noise: the
planner's collision-world model is deliberately desynced from the true scene, so the failure
is *causal* and labelled by construction.

Perturbation types (`ptype`):

| ptype | scene | what it does |
|---|---|---|
| `shift_object` | clean | move the **target** after its grasp is planned → empty/awkward grasp |
| `shift_target` | clean | move the **destination** → placement misses |
| `shift_obstacle` | cluttered | shift a corridor **obstacle** into the path → collision |
| `hide_obstacle` | cluttered | hide an obstacle from the planner → collision with an unseen body |

## How the desync works

`TargetedRuntime` attaches to a bench env as `env.targeted`. The only touch-point in the base
simulator is one hook in
[`benchmark/bench_envs/_bench_base_task.py`](../benchmark/bench_envs/_bench_base_task.py):

- `update_world(...)` consults `env.targeted.is_excluded(actor)` and
  `env.targeted.override_pose(actor, pose)` — so excluded actors stay out of the CuRobo world
  and shifted actors keep reporting their **pre-shift** pose to the planner, persistently,
  across every re-plan.
- the grasp planner fires `env.targeted.notify("after_grasp_plan", env)` once the grasp poses
  are committed — the moment the runtime applies a shift so the expert grasps stale geometry.

With no `env.targeted` attached, the base behaviour is unchanged (the hooks are no-ops).

## What it captures

Aligned 1:1 with the saved video frames, into the standard bench `episode0.hdf5`:

- `targeted_state/` — per-frame `actor_pos [T,A,3]`, `actor_quat [T,A,4]`, `actor_names`,
  `frame_idx`, and an `entities_json` role map. The minimal **replayable state trace**.
- `contact_log` — sparse contact events.
- `targeted_labels/` — the label record (see below).

## Offline labels (pure projectors)

Every semantic is a pure, versioned, unit-tested function over the logged signals — never
recomputed by re-simulating:

- `derive_outcome` → one of `clean_success | success_with_collision | empty_grasp |
  placement_failure | planning_failure | collision_only | ...`
- `annotate` → WHAT flag-set (`grasp_failure`, `placement_failure`, `collision`,
  `planning_failure`) + WHY (per ptype) + quality
- `compute_progress`, `compute_safety` → progress / safety curves from the pose trace
- `sample_shift_params`, `construct_world_shift` → the stratified, isolated-RNG perturbation
  sampler (magnitude bins; depth / lateral / mixed perceptual classes)

The package is intentionally a single file
([`src/robo_negative/__init__.py`](src/robo_negative/__init__.py)) holding the sampler, the
`TargetedRuntime`, the loggers, the annotators, the task **registry** (`task_entry`,
`is_supported`, `SUPPORTED_TASKS`), and the unit tests.

## Usage

Run the orchestrator from the repo root (it self-configures `BENCH_ROOT` / `ROBOTWIN_BENCH_TASK`
and runs each episode in an isolated subprocess so CuRobo/SAPIEN state never leaks across a
twin):

```bash
# defaults: put_mouse_on_pad, ptypes {shift_object,shift_target,shift_obstacle}, seeds 0:40,
# 3 baselines + up to 30 twins per group -> ./targetted_dataset/
python collect_targeted_data.py --gpu 0

# one episode directly (normally a subprocess of the collector)
python run_targeted_episode.py \
    --task-name put_mouse_on_pad --task-config bench_demo_office_clean \
    --seed 0 --role baseline --out-dir /tmp/ep

# annotated HTML gallery of a collected set (offline; no sim)
python visualize_negative_data.py --root targetted_dataset --out gallery.html
```

Run the unit tests with any test runner pointed at the module, e.g. `pytest
src/robo_negative/__init__.py`.

## Relationship to the replayable-state demo

The `targeted_state` trace this library logs is exactly what
[`replayable_state_demo/`](../replayable_state_demo/README.md) consumes to reconstruct an
episode's full 3D scene offline — the same "capture sufficient state, project semantics later"
principle, generalised in [`REPLAYABLE_STATE_PROPOSAL.md`](../REPLAYABLE_STATE_PROPOSAL.md).
