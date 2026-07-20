# lookahead-search

**Decouple SEARCH from EXECUTION.**

A *search* uses the simulator's snapshot / rollback ("branch & rollback") to explore
a task with a candidate policy (or, later, a motion planner), scores branches with a
fitness function, and emits a **motion plan** whose payload is the winning
trajectory's **raw action sequence**. A separate *replay policy* (`policy/replay`) —
compatible with the existing eval and data-collection pipelines — just replays those
raw actions. Because raw-action replay involves **no policy re-inference**, it is
deterministic: it reproduces the searched trajectory exactly and lets you enrich the
observation / annotation stream at replay time (extra cameras, depth/seg/pointcloud,
masks, event labels, language relabeling, …) without changing the outcome.

```
   candidate policy / planner            replay policy (policy/replay)
            │                                     │
   ┌────────▼─────────┐   plan.json     ┌─────────▼──────────┐
   │  run_search.py   │ ───────────────▶│  deterministic      │
   │  branch&rollback │  raw actions +  │  raw-action playback │──▶ eval_policy.py
   │  + fitness score │  fingerprints + │  + PinPolicy checks  │──▶ collect_rollout_client.py
   └──────────────────┘  pinned spec    └──────────────────────┘
```

## Package layout

| module | role |
| --- | --- |
| `rollback.py` | `snapshot` / `restore` / `apply_chunk` / `settle` / `state_fingerprint` + the `TaskAdapter` Protocol. PhysX `pack()`/`unpack()` **plus** controller drive targets, and the critical `task.eval_success = False` reset on restore. |
| `candidates.py` | `CandidateSource` ABC; `PolicyCandidates` (reuse a deploy-policy model; K chunks by latent mode `z` or by K noises). `CuroboCandidates` seam. |
| `fitness.py` | `Fitness` ABC; `OracleFitness` = `({HARD:0, SOFT:1}.get(tier,2), dist_to_goal)`, minimised. Collision/distance terms are optional so target-less tasks fall back to success-only. |
| `search.py` | `SearchStrategy` ABC; `BeamSearch` (default), `MonteCarloSearch`, `FullTreeSearch`. Returns a `SearchResult` carrying the winning path's **raw actions** (not modes). |
| `plan.py` | `TaskSpec` / `Plan` dataclasses + JSON `save` / `load`. Self-contained, portable, inspectable; actions inline as nested lists. |
| `pins.py` | `Pin` / `PinPolicy` / `PinViolation` — the configurable replay-time verification knob. |
| `run_search.py` | CLI: build task → capture spec + provenance → search → **verify via from-start replay** → write plan. |
| `tests/test_smoke.py` | Sim-free smoke tests against a mock task (no jax / no sapien). |

The pure modules (`rollback`, `fitness`, `search`, `plan`, `pins`) have **no jax
dependency**; only `candidates` / `run_search` touch a policy model.

## Pinned vs free

The plan pins everything that determines the outcome. Everything that is outcome
**invariant** stays a replay-time knob. The **t0 fingerprint check is the arbiter**
for the gray zone: a nominally "free" knob that perturbs the seeded scene is caught.

| PINNED (in the plan; changing → re-search) | FREE (replay-time knob; outcome invariant) |
| --- | --- |
| `task_name`, `task_config`, `seed` | cameras / sensors (count, pose, resolution, FOV) |
| physics-changing randomization (`cluttered_table`, `random_table_height`, object instance ids) | modalities (rgb / depth / seg / pointcloud / tactile) |
| embodiment | annotations / labels (masks, events, success flags, language relabeling) |
| control / action semantics (action dim, chunk, control freq) | visual-only randomization (background, lighting) *— subject to the RNG caveat* |
| physics params | recording config (save_freq, codec, fps) |
| the raw actions | consumer (eval vs collect) |
| harness provenance (commit, config hash) | |

**RNG caveat / gray-zone rule.** Visual-only randomization is only free if it does
not perturb the *seeded* scene. Rule: **apply free knobs AFTER the seeded scene is
built, from an independent RNG.** If a "free" knob draws from the same seeded RNG
that places objects, it shifts the physics scene — and the `fingerprint_t0` pin will
catch it.

## PinPolicy — the verification knob

Each named pin has a `level` (`enforce` | `warn` | `off`, default `enforce`) and, for
numeric pins, an `atol` (default `1e-5`).

| pin | compares | default |
| --- | --- | --- |
| `provenance` | harness commit + `bench_config_hash` | enforce |
| `task_config` | `task_name` + `task_config` | enforce |
| `embodiment` | robot embodiment | enforce |
| `control` | action_dim / chunk / control_freq | enforce |
| `fingerprint_t0` | **the seeded-scene arbiter** (state at t0) | enforce (atol 1e-5) |
| `fingerprint_terminal` | state at the scored terminal | enforce |
| `success` | replay success vs plan success | enforce |

Ship the default (all enforce), then **relax any single pin** in the replay
`deploy_policy.yml` `pins:` block. A value is either a bare level string or a
`{level, atol}` map:

```yaml
pins:
  provenance: off            # stop pinning the commit hash while iterating
  fingerprint_t0:
    level: enforce
    atol: 1.0e-5
  fingerprint_terminal: warn
```

In code:

```python
from lookahead.pins import PinPolicy
policy = PinPolicy.from_dict({"provenance": "off"})
policy.check("fingerprint_t0", expected, actual)   # enforce -> raises PinViolation on mismatch
```

## Plan schema (`lookahead_plan_v1`)

```jsonc
{
  "format": "lookahead_plan_v1",
  "task_spec": {                       // PINNED
    "task_name": "put_milktea_on_shelf",
    "task_config": "bench_demo_office_d8",
    "seed": 40000,
    "embodiment": {"embodiment": ["aloha-agilex"], "embodiment_name": "aloha-agilex"},
    "domain_randomization": { /* the FULL dict from the task config */ },
    "control": {"action_dim": 32, "chunk": 50, "control_freq": null},
    "provenance": {"robotwin_commit": "…", "bench_config_hash": "…", "created_at": "…"}
  },
  "actions": [[...], [...], …],         // raw action rows (the motion plan payload)
  "fingerprints": {"t0": [...], "terminal": [...]},
  "meta": {"strategy": "beam", "fitness": "oracle", "candidate_policy": "pi05",
           "score": [0, 0.021], "outcome": {…}, "success": true, "verified": true}
}
```

## search → plan → replay

### 1. Search a plan

```bash
cd customized_robotwin
python -m lookahead.run_search \
    --task put_milktea_on_shelf --task_config bench_demo_office_d8 --seed 40000 \
    --candidate_policy pi05 --policy_config policy/pi05/deploy_policy.yml \
    --strategy beam --width 3 --k 6 --depth 4 --fitness oracle \
    --out /work/mohammed/datasets/lookahead_plans
# -> /work/mohammed/datasets/lookahead_plans/put_milktea_on_shelf/bench_demo_office_d8/seed40000.json
```

`run_search` captures the full `TaskSpec` (incl. `domain_randomization`) + t0
fingerprint + provenance, runs the search, then **verifies** by a from-start
raw-action replay in a fresh scene at the same seed (apply the plan's actions → assert
the terminal fingerprint / success reproduce the searched outcome) before writing the
plan (`meta.verified`).

### 2. Replay a plan

Point the replay policy at the plan store and run it through **either** consumer,
unmodified. Configure the plan store + pins in `policy/replay/deploy_policy.yml`
(`plan_dir` = the `run_search` output dir, or a single `plan_path`).

```bash
# eval
python script/eval_policy.py --config policy/replay/deploy_policy.yml \
    --overrides --task_name put_milktea_on_shelf --task_config bench_demo_office_d8 --seed 40000

# collect (env-pipeline recording; enrich observations/annotations at replay time)
#   COLLECT_FIXED_SEED=1 replays a fixed seed's plan (skips the expert seed search)
```

Because RoboTwin does not store the episode seed on the task, the replay model
selects the plan for a **configured seed** (`seed` in yml) or a single `plan_path`;
if a future env exposes `task.seed`, `ReplayModel.select_for_task` honours it.

## Extension points

- **Candidate sources** — subclass `CandidateSource`; `CuroboCandidates` is a seam
  for a planner-based source (return K planner trajectories as `[chunk, action_dim]`
  arrays). No change to the search core.
- **Fitness** — subclass `Fitness` (`make_context` / `outcome` / `score`, lower =
  better) and register it in `FITNESS_REGISTRY`.
- **Strategies** — subclass `SearchStrategy`; `MonteCarloSearch` / `FullTreeSearch`
  ship as simple alternatives to `BeamSearch`.

## Tests

```bash
cd customized_robotwin
python -m lookahead.tests.test_smoke     # sim-free; mock task, no jax/sapien/GPU
```
