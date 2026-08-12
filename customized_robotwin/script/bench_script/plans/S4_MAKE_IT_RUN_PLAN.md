# S4 — Make it run

Technical plan for section **S4** of [REHAUL_PLAN.md](REHAUL_PLAN.md).
Working branch: **`peng-dev-new`**. ~1 day plus a user-run GPU session.
**SHARED runtime behaviour — the failures here are silent, not loud.**

**Status: completed 2026-08-12.** Codex executed Steps 1–3 and prepared the validation harness; the
user ran all GPU work in Step 4, as required by `agent-memory/feedback_role_boundary.md`.

## Execution record

- CPU gate: **50 passed, 1 skipped**. The new patch guard also passes standalone.
- Ring runtime fix: explicit `update_world(exclude_obstacles=False)`, with the exact delivered
  CuRobo collision-entry count captured only after `robot.update_world` succeeds. Both the
  setup-default and explicit-full counts are frozen per rollout and written to `records.jsonl`.
- Built geometry: seed 0, requested gap 0.100 m, measured collision-hull gap 0.109905 m,
  error 0.009905 m against a 0.010 m gate, with 0.125012 m z overlap. Raw actor segmentation
  resolved target id 75 and 207 visible pixels. Artifacts:
  `scripts/validation/results/s4_make_it_run/geometry/20260812-121536/`.
- Ring smoke: seed 0 at clutter density 8 succeeded with records, MP4, HDF5 and an empty
  `rollout_seed_stats` list (correct for direct mode); zero physics collisions; CuRobo world
  **11 full > 2 default**. Artifacts:
  `scripts/validation/results/occluder_visibility/s4_ring/20260812-135100/`.
- Port comparison: matched standard/direct seeds 0–4 produced identical per-seed outcomes on
  `peng-dev-new` and `peng-training-branch`: successes 0,1,3; seed 2 failed at `grasp` with six
  unreachable rotations; seed 4 failed at `placement:pre_place_descent` with
  `MotionGenStatus.FINETUNE_TRAJOPT_FAIL`. Both branches were **3/5**. This is a gross-regression
  smoke, not a powered equality result.
- No runtime reconciliation commit was needed: every standard-cell failure reproduced identically
  on the pre-rehaul branch.

Two planning assumptions were corrected during execution. Visibility masks use
`get_segmentation_raw(level="actor")`; RGB is only the saved overlay. Also, generic `seed_traj`
text exists in pristine CuRobo, so the guard checks four patch-specific sentinels instead.

---

## Context

S3 proved the ported tree **imports** and the CPU suite passes at 49/1. That gate is real but narrow:
it exercises metric math and ring determinism, never scene construction, planning, or the expert.
S4 is where the port meets six weeks of dev's *runtime* behaviour.

**The headline defect, confirmed.** Dev's `_office_base_task._init_task_env_` resolves clutter
visibility as:

```python
exclude_obs = self.planner_exclude_obstacles
if exclude_obs is None:
    exclude_obs = self.enable_collision_metrics   # legacy coupling
```

`benchmark/bench_task_config/bench_demo_office_clean.yml:46` sets `enable_collision_metrics: true`,
so on the ring config the base task deliberately hides clutter from curobo and prints
*"curobo planner skips clutter obstacles"*.

Our ring task then calls bare `self.update_world()` at `task/occluder_task.py:198` to put clutter
back, under a nine-line comment stating that without it *"the expert would plan straight THROUGH
clutter and knock it over (even on 'successful' rollouts)"*. Under our old signature
(`exclude_obstacles: bool = False`) that was true. Under dev's (`= None`) the bare call resolves
straight back to `enable_collision_metrics` — **true** — and excludes clutter again.

**The line now does exactly what its own comment says it exists to prevent, and nothing errors.**

Two findings that *reduce* scope:

- Dev's base task **guards** its own bare call behind `if not exclude_obs` (lines 270–278), so it is
  self-consistent. No dev-side fix needed; do not touch `_office_base_task.py`.
- Visibility measurement delegates to dev's raw actor-segmentation implementation; RGB is only
  used for the saved overlay. This was confirmed in the built scene rather than refactored.

**The structural risk.** `customized_robotwin/envs/curobo` is **gitignored**
(`customized_robotwin/.gitignore:7`) — untracked, not a submodule. The `seed_traj` patch survived S3
only because git never touched those files. S2 found `uv sync` prunes the editable curobo install
before `bootstrap_uv.sh` reinstalls it, so a routine `make sync` reverts the patch silently. Today
the only guard is a grep inside `run_approach_mode_ab.sh`.

## Decisions taken

- GPU run covers **expert rollouts plus a comparison against pre-rehaul**.
- The curobo patch gets a **CPU check module**, so loss is caught by every `pytest` run.
- The edge-to-edge geometry is **verified physically in a built scene**, not just at unit level.

---

## Step 1 — Make the clutter inclusion explicit, and prove it fired

`task/occluder_task.py:198` — an exclusive file, so this is a one-line change with no shared-file cost:

```python
self.update_world(exclude_obstacles=False)
```

Rewrite the comment above it: the point is no longer "bare means include", it is "dev's default
resolves to the config's blindness, so the ring path must override explicitly."

**Then prove the treatment was delivered.** `feedback_scientific_rigor.md` records the most expensive
lesson of 2026-07-30 — isolation is not enough, you must measure that the treatment actually fired.
An explicit argument that silently fails is the same failure in a new costume. Record, in the
per-rollout log, the **count of collision entries curobo received** (the `collision_dict` passed at
`_bench_base_task.py:2089`), so a rollout with clutter invisible to the planner is visible in the
record rather than inferred from behaviour.

There is also a free runtime tell: the base task prints
*"curobo planner skips clutter obstacles"* in orange. After the fix, the ring path must show clutter
back in curobo's world despite that message.

> **→ COMMIT 1 — `Restore explicit clutter inclusion on the ring path`**
> The semantic fix plus its diagnostic. Isolated: this is the only behaviour change in S4 that is
> not a response to something the GPU run surfaces.

## Step 2 — Guard the curobo patch in the test suite

New `checks/test_curobo_patch.py` (CPU, no CUDA): assert four RoboPRO-specific patch sentinels appear
in `customized_robotwin/envs/curobo/src/curobo/wrap/reacher/motion_gen.py`. A generic `seed_traj`
match is insufficient because pristine CuRobo already uses that local name. The launcher calls this
same guard. Its failure message names the repair command:

```
cd <repo-root> && git -C customized_robotwin/envs/curobo apply ../../script/bench_script/curobo_seed_traj.patch
```

**This changes the baseline from 49 passed to 50 passed.** Record that deliberately.

> **→ COMMIT 2 — `Guard the vendored curobo seed_traj patch`**

## Step 3 — Audit the rest of the runtime surface (CPU, before burning GPU time)

Read-only sweep; fix only what is provably broken:

- **Other `update_world` call sites.** Only `occluder_task.py:198` exists in our code today —
  re-verify after Step 1 that no other ring-path call is bare.
- **Collision-metric streaming.** Dev added `start_metric_streams`, `_init_proximity_tracking`,
  `_mark_intended_contact`. Our `analyze_occluder_visibility.py:309` and `vla_rollout.py:599` already
  branch on `enable_collision_metrics`. Confirm the ring path neither double-starts nor skips them.
- **Segmentation.** Confirm dev's raw actor-ID path produces a nonempty target mask in the live
  scene; RGB only validates the saved overlay.

> **→ COMMIT 3 — `Reconcile runtime surface`** — only if this step changes anything. Skip otherwise.

## Step 4 — Hand over the GPU session

**The agent prepares commands and stops. The user runs them.** Per `feedback_role_boundary.md`,
Claude does not run GPU work. Three runs, in this order:

**4a. Geometry verification (~5 min).** Build one scene at a known offset and measure the actual
closest-surface gap between the target and occluder collision meshes. S3 verified only the unit-level
arithmetic (a requested 0.10 m gap → 0.1714 m centre distance); this confirms SAPIEN agrees.
Expected: measured surface gap ≈ requested, within voxel tolerance.

**4b. The unconfounded port comparison — use the 0-occluder `standard` config.**

> **This design choice matters.** A naive "same seeds on both branches" comparison is **confounded**:
> S3 changed occluder spacing, so the same nominal offset now builds a physically different scene.
> Comparing success rates across branches would measure the geometry change, not the port. The
> `standard` config has **zero occluders**, which makes edge-to-edge versus centre-to-centre moot,
> so it isolates everything else the port touched. `domain_expert_baseline.md` records a historical
> reference of **22.2% over n=54** for that cell.

Run matched seeds on `peng-dev-new`, then the same seeds on `peng-training-branch`. Compare success
rate, and — more informative at small n — the **failure-reason distribution** (`grasp`,
`placement:pre_place_descent`, `place_actor`).

**4c. The occluder path, qualitative.** A few ring rollouts on `peng-dev-new` confirming
`records.jsonl` and `rollout_seed_stats` populate, the `[SEED FIRING]` block reports, and the full
collision-world entry count exceeds the setup-default count at clutter density > 0. A merely
nonzero total is insufficient because the table and curated occluder exist in both worlds. Do
**not** compare these numerically to pre-rehaul — different scenes.

## Step 5 — Reconcile what the runs surface, then record

Fix only what the runs prove broken. Anything that also fails on `peng-training-branch` is
pre-existing: record it, do not repair it inside S4.

Write into `agent-memory/status_current.md`: the new 50/1 baseline, the geometry measurement, the
standard-cell comparison with its n, and any behaviour that changed. Mark S4 done in `REHAUL_PLAN.md`.

> **→ COMMIT 4 — `Record S4 completion`** then push to `origin/peng-dev-new`.

---

## Out of scope

- The 84-file `bench_envs` cleanup (S5) and all consolidation (S6–S12).
- Repairing pre-existing failures that reproduce on `peng-training-branch`.
- Re-running the `phase4_approach_mode` `off` cell — deferred by decision, Part 6b.
- Any change to `_office_base_task.py` — dev's resolution is self-consistent and it is a shared file.
- Re-tuning the expert. If success rates differ, **report it**; diagnosing why is the user's call
  (`feedback_role_boundary.md`).

## Done when

`pytest` reads **50 passed / 1 skipped**; one ring rollout completes with `records.jsonl`,
`rollout_seed_stats` and a full CuRobo world count greater than the default at clutter density > 0;
the measured surface gap matches the requested edge-to-edge offset; and the 0-occluder comparison
against `peng-training-branch` shows no unexplained divergence.

## Test plan

**1. Regression gate.** Baseline moves from 49 to **50 passed, 1 skipped** (51 items) once Step 2
lands. State the change explicitly in the record so S5 measures against the right number.

**2. Commands.**

```bash
cd customized_robotwin/script/bench_script
../../../.venv/bin/python -m pytest -q            # 50 passed, 1 skipped
../../../.venv/bin/python -m checks.test_curobo_patch
# GPU commands prepared in Step 4 — run by the user, from customized_robotwin/
# with `source set_env.sh` and `export ROBOTWIN_BENCH_TASK=bench`
```

**3. Equivalence — two independent checks.**

- **Treatment fired:** the recorded collision-entry count with `exclude_obstacles=False` must exceed
  the count under the base task's default. A rollout where those are equal means the fix did not take.
- **Port faithful:** on the 0-occluder `standard` config with matched seeds, `peng-dev-new` and
  `peng-training-branch` must agree on success rate and failure-reason families. This is the only
  comparison in S4 that the geometry change does not confound.

**4. What S4 CANNOT verify.**

- **Small-sample power.** A handful of rollouts cannot distinguish, say, 22% from 15%. The comparison
  detects gross breakage — a leg that never completes, a failure mode that appears from nowhere — not
  a modest regression. Do not report "success rate unchanged" as if it were measured; report the n.
- **That edge-to-edge is scientifically right**, only that it is implemented as specified and that
  SAPIEN agrees with the arithmetic. Whether the new spacing is the right experimental variable is
  the user's question.
- **The occluder path against pre-rehaul.** Different scenes; no valid numerical comparison exists.
- **GPU paths outside the smoke set** — `checks.smoke_test_seed_2a` and the real
  `diag_kitchen_curobo` diagnostic stay unrun unless explicitly included.

## Risks

| Risk | Mitigation |
|---|---|
| The `update_world` fix looks applied but does not take effect | Clutter-count diagnostic recorded per rollout — the explicit "treatment fired" check |
| `uv sync` / `make sync` silently reverts the curobo patch mid-session | Step 2's check module fails on every `pytest` run, not just in the A/B script |
| Port comparison confounded by the geometry change | Comparison runs on the 0-occluder `standard` config, where spacing is irrelevant |
| A pre-existing failure gets "fixed" inside S4, muddying the diff | Reproduce on `peng-training-branch` first; if it fails there too, record and leave |
| Segmentation turns out to matter after all | Named as first suspect if visibility masks come back empty in 4c |
| GPU session serialises on the user | Steps 1–3 are CPU and commit before the handover, so the session is a clean single hand-off |

## Rollback

Every S4 change is on `peng-dev-new` in separate commits. Revert COMMIT 1 to restore the bare call
(restoring the defect), COMMIT 2 to drop the guard. `peng-training-branch` is untouched and remains
tagged `pre-rehaul-2026-08-11`; the five `backup/*` refs on origin are unaffected.
