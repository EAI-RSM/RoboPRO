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
| `candidates.py` | `CandidateSource` ABC; policy-agnostic `PolicyCandidates(model, encode_obs, k, propose_fn=None)` — default proposer is `noise_propose` (K noise re-samples); a policy overrides via its own `propose_candidates` hook. `mode_propose` is a WTA helper; `CuroboCandidates` seam. |
| `fitness.py` | `Fitness` ABC; `OracleFitness` = `({HARD:0, SOFT:1}.get(tier,2), dist_to_goal)`, minimised. Collision/distance terms are optional so target-less tasks fall back to success-only. |
| `search.py` | `SearchStrategy` ABC; `BeamSearch` (default), `MonteCarloSearch`, `FullTreeSearch`. `search(..., m=1) -> List[SearchResult]` returns the top-M distinct plans (best first), each carrying the winning path's **raw actions**. |
| `plan.py` | `TaskSpec` / `Plan` dataclasses + JSON `save` / `load`. Self-contained, portable, inspectable; actions inline as nested lists. |
| `pins.py` | `Pin` / `PinPolicy` / `PinViolation` — the configurable replay-time verification knob. |
| `run_search.py` | CLI: build task → capture spec + provenance → search → **verify via from-start replay** → write plan (`--dump-trace` also emits a `SearchTrace`). |
| `trace.py` | `SearchTrace` (`lookahead_trace_v1`) + `save_trace`/`load_trace` — a **superset of `Plan`**: the committed spine plus every decision's K held-to-terminal candidate branches (raw actions + outcome tier). The policy-free artifact viz + Q-data consume; `committed_plan()` yields a replayable `Plan`. |
| `viz.py` | Ghost-futures branching video. `capture_trace` (policy stage → `SearchTrace`), `render_trace` (deterministic replay of saved actions → frame bundle, **no policy**), `compose_scene_video` (composited MP4). |
| `run_viz.py` | CLI: search+capture+render+composite, or `--trace <path>` to render a pre-done `SearchTrace` with no policy / no search. |
| `tests/test_smoke.py` | Sim-free smoke tests against a mock task (no jax / no sapien). |

The pure modules (`rollback`, `fitness`, `search`, `plan`, `pins`) have **no jax
dependency**; only `candidates` / `run_search` touch a policy model.

## Replay hash

The **replay hash** is everything the plan captures that *determines the
trajectory* — what must match for a replay to reproduce the exact task spec:

| replay-hash property | in the plan |
| --- | --- |
| task / config / seed | `task_spec.{task_name, task_config, seed}` |
| physics-changing randomization (`cluttered_table`, `random_table_height`, object instance ids) | `task_spec.domain_randomization` |
| embodiment | `task_spec.embodiment` |
| control & action semantics (action dim, chunk, control freq) | `task_spec.control` |
| physics params | `task_spec.control` (`physics_timestep`) |
| the raw actions | `actions` |
| harness provenance (commit, config hash) | `task_spec.provenance` |

The **t0 fingerprint is the arbiter**: it is a deterministic digest of the built
scene, so if any replay-hash property differs the fingerprint differs. On replay the
`pins` verify these (`enforce | warn | off`, per-pin `atol`), and any can be relaxed
in one line (e.g. `pins: {provenance: off}`). Apply any replay-control knob (below)
*after* the seeded scene is built and from an independent RNG, so it never perturbs
the replay hash.

## Replay control

**Replay control** is the additive knobs available at replay time that don't change
the outcome — the point of decoupling search from execution: search once, then
replay to collect richer data or eval under new sensors while the physics stays fixed.

| replay-control knob | examples |
| --- | --- |
| cameras / sensors | count, pose, resolution, FOV |
| modalities | rgb / depth / seg / point-cloud / tactile |
| annotations & labels | masks, events, success flags, language relabeling |
| recording config | save_freq, codec, fps |
| consumer | eval vs collect |

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
  "task_spec": {                       // the replay hash
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
    --strategy beam --width 3 --k 6 --depth 4 --fitness oracle --top-m 1 \
    --out /work/mohammed/datasets/lookahead_plans
# -> /work/mohammed/datasets/lookahead_plans/put_milktea_on_shelf/bench_demo_office_d8/seed40000.json
```

`--top-m M` (default 1) emits the top-M distinct plans: rank-0 →
`seed<N>.json`, rank-r → `seed<N>_rank<r>.json`; each plan carries its `rank` and
`score` in `meta` and is **verified independently**.

`run_search` captures the full `TaskSpec` (incl. `domain_randomization`) + t0
fingerprint + provenance, runs the search, then **verifies each ranked plan** by a
from-start raw-action replay in a fresh scene at the same seed (apply the plan's
actions → assert the terminal fingerprint / success reproduce the searched outcome)
before writing the
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

### 3. Visualize the search (ghost futures)

A `SearchTrace` (`run_search --dump-trace`, or produced by `run_viz`) saves the
committed spine **plus** every decision's K candidate futures (held to terminal —
raw actions + outcome tier), so the visualization consumes a **pre-done search
output**, exactly like replay consumes a plan: no re-search, no policy inference.

![ghost futures](https://github.com/EAI-RSM/RoboPRO/raw/lookahead-search/customized_robotwin/lookahead/docs/ghost_futures.gif)

```bash
cd customized_robotwin
# search + capture + render + composite (beam width=1, K=6, depth = task step-limit / chunk)
python -m lookahead.run_viz \
    --task put_stapler_in_drawer --task_config bench_demo_office_clean --seed 40000 \
    --candidate_policy pi05 --policy_config policy/pi05/deploy_policy.yml \
    --width 1 --k 6 --out /work/mohammed/datasets/branching_video/lookahead
# -> traces/<scene>.json (+.npz) · bundles/<scene>/frames.npz · futures_<scene>.mp4

# render a PRE-DONE trace with NO policy / NO search:
python -m lookahead.run_viz --trace <trace>.json --out <dir>
```

Playback follows the committed trajectory; at each decision point the video
freezes and the K candidate futures are revealed one at a time as translucent
ghosts, tinted by their held-to-terminal outcome (green = success, red =
collision, gray = fail), the committed future revealed last with a "selected"
tag. Ghost tiers use the **same fitness as the eval pipeline**
(`get_collision_metrics`), so a red ghost is a real (impulse-based) collision.

## Candidate generation (policy-agnostic)

The search engine and `run_search` know nothing about *how* candidates are produced —
that lives entirely in the policy layer behind one interface:
`propose_fn(model, obs, k) -> list[chunks]`.

- **Default:** `noise_propose` — K flow-matching noise re-samples. Works for any
  single-mode flow/diffusion policy, so `run_search` needs no special flag.
- **Policy override (the hook):** a candidate policy opts into a custom mechanism by
  exposing an optional module-level `propose_candidates(model, obs, k)` — mirroring
  the `get_model` / `eval` / `reset_model` convention. `run_search` resolves it with
  `getattr(<policy_module>, "propose_candidates", None)`; if absent it falls back to
  the default noise proposer. There is no "mode" flag on the CLI.
- **WTA example:** a policy with latent action modes implements the hook with the
  `mode_propose` helper (K latent modes `z=0..k-1` under one fixed noise):

  ```python
  # in the WTA policy's deploy_policy.py (alongside get_model / eval / reset_model)
  from lookahead.candidates import mode_propose, noise_propose

  def propose_candidates(model, obs, k):
      # use latent MODES when the checkpoint has a multimodal action head, else noise
      if getattr(model, "_enable_mm", False) and int(getattr(model, "_num_modes", 1)) > 1:
          return mode_propose(model, obs, k)
      return noise_propose(model, obs, k)
  ```

- **Escape hatch:** `--candidate-opt key=val` (repeatable) passes opaque kwargs to the
  proposer (e.g. `--candidate-opt seed=1234`); no key is interpreted by `run_search`.

## Extension points

- **Candidate sources** — subclass `CandidateSource`; `CuroboCandidates` is a seam
  for a planner-based source (return K planner trajectories as `[chunk, action_dim]`
  arrays). No change to the search core. Or, for a policy-driven source, expose
  `propose_candidates` (above) instead of a whole subclass.
- **Fitness** — subclass `Fitness` (`make_context` / `outcome` / `score`, lower =
  better) and register it in `FITNESS_REGISTRY`.
- **Strategies** — subclass `SearchStrategy`; `MonteCarloSearch` / `FullTreeSearch`
  ship as simple alternatives to `BeamSearch`.

## Tests

```bash
cd customized_robotwin
python -m lookahead.tests.test_smoke     # sim-free; mock task, no jax/sapien/GPU
```
