# RoboPRO rehaul: scope and feasibility

**A scoping document, not an execution plan.** All numbers are measured. Read Part 4 and the
Feasibility section first — everything else is small.

---

## Context

`peng-training-branch` has drifted from the team's `origin/dev`, and the worry is that *future*
reconciliation becomes impossible — not that today's merge is impossible. The proposal is a fresh
`peng-dev-new` off `origin/dev` with the scripts redeveloped there.

Decisions taken: goals are **cheap future merges**, **clean architecture**, **dead weight removal**.
Data continuity is **partial** — results stay readable, not re-runnable. Fork policy is **track dev
closely, merge often**. Budget is **1–2 weeks** before the next experimental result.

### The measurement that reframes the problem

Fork point `aabeff4` (2026-06-25). Against `origin/dev` (`64840ce`, 2026-08-06):

| | |
|---|---|
| Commits ahead / behind | 82 / 60 |
| My files changed since fork | 208 (+26,969 / −1,116) |
| dev's files changed since fork | 460 (+13,996 / −1,433) |
| Files changed on **both** sides | 37 |
| Real conflicts (`git merge-tree`) | **9** |

Splitting my 208 changed files by whether `origin/dev` also has the file:

| Class | Files | Lines | Merge cost |
|---|---|---|---|
| **Shared** (dev has it too) | 106 | +2,135 / −1,116 | 100% of the tax |
| **Exclusive** (mine only) | 102 | +24,834 | zero |

**And the shared surface is mostly already upstream.** `8-research-branch` merged into dev as PR #57
and the visibility work re-landed as PR #56, so most of my `customized_robotwin/envs/` edits are
byte-identical on dev already — and several of my copies are now *behind* it.

**Genuine carry-over across the entire shared surface is ~150 lines.** Reconciliation is a much
smaller problem than it looks. What is large is the 17,051-LOC research layer in `bench_script/` —
which costs *nothing* to merge, because nobody else touches those files.

### What that implies

The disease is not "the branches diverged." It is three separate things:

1. **Staleness.** Six weeks of dev fixes unabsorbed, some of which change what the data *means*:
   occluder spacing became edge-to-edge (`cf966f7`), and `update_world()`'s default flipped. Fixed
   by merging, not by rewriting.
2. **No safety net.** There was no test runner in the repo. Twelve hand-rolled `checks/` modules
   (2,618 LOC) ran as `python -m checks.<module>`; the real verification bar was `--help`.
3. **Coupling by habit.** Research code edits shared files, and reaches into `envs/` through five
   doors where one was intended.

> **One concern, stated once.** `agent-memory/feedback_minimal_changes.md` records over-engineering
> as the costliest failure mode here, and the 2026-07-29 refactor succeeded *because* it was
> "structural only — no behaviour or algorithm changes," keeping collected A/B data comparable. Part
> 4 below therefore **ports and consolidates** rather than retyping from scratch. That hits all three
> goals at a fraction of the risk. Classified per `feedback_role_boundary.md`: Parts 1–3 are
> **validity** (the branch is running on superseded scene geometry). Parts 4–5 are **quality**.
> Nothing here blocks a run except Part 3.

---

## Work breakdown — the executable sections

**How to use this.** Each section below is self-contained: "write a technical plan for S6" or
"execute S8" should be enough to start cold, without reading the rest of this document. Parts 0–7
further down are the *rationale*; the sections here are the *work*. Numbering is stable — do not
renumber.

**Every section's plan must state a stopping condition.** `agent-memory/feedback_minimal_changes.md`
records over-engineering as the costliest failure mode on this repo, and that without an explicit
"done when" there is always one more defect to fix. Sections marked NO-BEHAVIOUR-CHANGE must not
alter algorithms, thresholds or retry logic — the 2026-07-29 refactor succeeded precisely because it
was structural only, which kept already-collected A/B data comparable.

**COMMIT DISCIPLINE — every section's plan must name its commit points.** A section never ends
uncommitted, and long sections commit at internal boundaries too. Rules:

- **One commit minimum per section**, at its end, once the done-when gate passes.
- **Isolate shared-file changes into their own commit**, separate from exclusive-file work. That is
  what makes a hunk cherry-pickable for the upstream PR the fork policy requires.
- **Commit before any step whose outcome is unpredictable** — iterative fix-and-rerun loops, GPU
  runs, anything that might need to be abandoned. If it goes wrong, the prior steps survive.
- Each section's plan states its commits explicitly as `→ COMMIT` markers between steps. Do not
  leave the count to the implementing agent's judgement.

The point is granular rollback and a bisectable history across a 17,000-line restructure: when
something breaks three sections later, the commit boundaries are what let you find where.

**EVERY DETAILED PLAN MUST CARRY A TEST PLAN.** The `Verify with` blocks in the sections below are
*seeds*, not complete test plans — only S2, S3 and S6 have one at all. A per-section
`S<N>_<TOPIC>_PLAN.md` is not finished until it contains all four of these:

1. **The universal regression gate.** From S2 onward, every section must end with `pytest` at or
   above the baseline S2 established: **50 items across 12 modules — 49 passed, 1 skipped**
   (`test_seed_2a_smoke`, the only GPU-marked test). A section that lowers the pass count has broken
   something, whatever else it achieved. State the expected count, not just "tests pass."
2. **Section-specific commands** — copy-pasteable, with the expected output. Not "confirm imports
   work" but the actual invocation and what it should print.
3. **An equivalence check for anything marked NO-BEHAVIOUR-CHANGE.** This is the one that matters
   most and the one most likely to be skipped. Proving code *runs* after a restructure proves
   nothing; you must prove the output is *unchanged*. Precedents already in this document: S8's
   offline CPU rebuild reproducing recorded `eps_geom` at max abs diff 0.000e+00; S9's figures and
   summary numbers unchanged on a recorded run; S10's success rate unchanged on a matched seed set.
   A structural section without an equivalence check is unverified, no matter how green the suite is.
4. **What the section CANNOT verify** — stated plainly. GPU-only paths, anything needing a rollout,
   anything with no CPU-reachable assertion. `agent-memory/feedback_scientific_rigor.md` records the
   2026-07-30 lesson that isolation is not enough — you must confirm the treatment actually fired.
   The same applies here: a test plan that hides its blind spots invites exactly that failure.

### How testing operates (from S2 onward)

**Running it.** From `bench_script/`:

```bash
../../../.venv/bin/python -m pytest -q          # expect: 49 passed, 1 skipped
../../../.venv/bin/python -m pytest -q --gpu    # adds the CUDA smoke test
```

⚠️ **Use the `.venv` interpreter explicitly.** `source set_env.sh` sets `BENCH_ROOT`/`ROBOTWIN_ROOT`
but does **not** change which Python the shell runs — a bare `python -m pytest` may hit a different
interpreter with no pytest. `conftest.py` handles the rest of the bootstrap, so no exports are
needed.

**Triage rule when something fails mid-section.** Compare against the baseline before debugging.
49/1 is the reference; if the failing test also fails on `pre-rehaul-2026-08-11`, it is pre-existing
and belongs in the section's "cannot verify" list, not in its fix list. Do not repair pre-existing
failures inside a restructuring section — that mixes two changes in one diff and destroys the
comparison.

**The equivalence toolkit.** Requirement 3 above demands an equivalence check for structural work.
Three patterns already exist in this repo; prefer them over inventing one:

| Pattern | Use for | Cost |
|---|---|---|
| **Offline CPU rebuild** — `records.jsonl`'s `scene_fingerprint_source.scene.metric_obstacles` carries every obstacle's mesh path, pose and scale, enough to rebuild the exact `geometric_eps` input with no SAPIEN and no GPU. Verified to reproduce all 100 recorded values at **max abs diff 0.000e+00**. See `agent-memory/tool_task_metric_validity.md`. | S7, S8 — anything touching the metric pipeline | ~40 s/scene, single core |
| **`checks/test_ring_config`** — asserts the occluder formation is byte-identical per (seed, offset-spec). Already in the suite. | S6, and any change near `occluder_ring.py` or scene construction | seconds |
| **Recorded-run replay** — regenerate figures and summary numbers from an existing run directory under `scripts/validation/results/` and diff them. | S9 (visualisation, record loading) | minutes, CPU |

If a section can build none of these, say so in its "cannot verify" section rather than substituting
"the tests still pass" — the suite covers the metric math and ring determinism, not the code most
sections are moving.

**Known coverage, stated honestly.** The 12 modules cover metric math, obstacle-set and ring
determinism, and a handful of VLA reporting paths. They cover **nothing** in `scripts/`, the
Makefile, the install path, the eval drivers, `collect_data.py`, or the seed-band contract. The
suite going green is necessary, never sufficient.

**One fragility to expect again.** `uv sync` prunes the editable CuRobo install before
`bootstrap_uv.sh` reinstalls it, which breaks collection. `pytest.ini` works around it with
`pythonpath = ../../envs/curobo/src`. Any section that re-syncs dependencies may hit this.

**Which branch am I on?** This changes at S3 — check before starting any section.

| Sections | Working branch |
|---|---|
| S1, S2 | **`peng-training-branch`** (the current branch) |
| S3 | starts on `peng-training-branch`, **creates and switches to `peng-dev-new`** |
| S4 – S12 | **`peng-dev-new`** |

S2 deliberately runs on the *old* branch: its whole value is a passing baseline to diff the port
against, and running it after the move measures the new tree against nothing. Its output carries
over for free — `bench_script/checks/` and the new `conftest.py` are exclusive files, and
`origin/dev` has no `checks/` directory at all, so S3's bulk copy picks them up.

`peng-training-branch` is not deleted. It stays as the reference tree, tagged `pre-rehaul-2026-08-11`
at `e3a09ce`, until `peng-dev-new` is verified. Unlike the current branch, **give `peng-dev-new` a
remote tracking branch immediately** — the fork policy is track-dev-weekly, and 64 unpushed commits
is how the current situation arose.

**Environment for every verification command** (from repo root):
```bash
cd customized_robotwin && source set_env.sh && export ROBOTWIN_BENCH_TASK=bench
```

| S | Title | Risk surface | Owner | Days | Depends on |
|---|---|---|---|---|---|
| S1 | Safety net | — | user | mins | — |
| S2 | pytest harness | mostly exclusive; dependency files shared | agent | 1 | — |
| S3 | Branch + port | **SHARED** | agent | 1 | S1, S2 |
| S4 | Make it run | **SHARED** | agent + user GPU | 1 | S3 |
| S5 | 84-file cleanup script | **SHARED** | agent | 0.5 | S3 |
| S6 | `envs/` doors 5 → 1 | exclusive | agent | 1 | S4 |
| S7 | One config surface | exclusive | agent | 1 | S4 |
| S8 | Cluster A — shared gated driver | exclusive | agent | 1 | S7 |
| S9 | Clusters C + D — viz and records | exclusive | agent | 1 | S7 |
| S10 | `task/` instrumentation | exclusive | agent + user GPU | 1 | S6 |
| S11 | `task/` prune | exclusive | agent | 0.5 | S10 |
| S12 | Supporting layer | mixed | agent | 1 | S4 |

S5 is independent of S6–S12 and can run in parallel. S10 needs a GPU run from the user in the
middle, so schedule it early rather than discovering it late.

---

### S1 — Safety net ✅ DONE 2026-08-11
**Owner: user. Depends on: nothing.**

`git rev-list --all --not --remotes` went **71 → 0**, verified against a fresh bare clone rather than
local tracking state. Five archive refs on `origin`:

| Remote ref | Tip | Holds |
|---|---|---|
| `backup/peng-training-branch` | `8dc44cb` | the main line |
| `backup/occluder-testbench` | `91b2d26` | Hamid's expert vendored as a 5th `--plan-algo` |
| `backup/pre-merge-35-occluder` | `35801cb` | reachability-path revert |
| `backup/reachability-motion-validated-placement` | `585d307` | the vanilla `--plan-algo` baseline |
| `backup/visibility-constraint-experiments` | `41ee00f` | free-target reachability experiment |

Plus tag `pre-rehaul-2026-08-11` on `e3a09ce`. No PR opened — the `backup/` prefix marks these as
archives.

**Scope correction found during execution.** The original scope ("push `peng-training-branch`") would
have saved only 64 of 71 commits. Four other local branches held unique work, including `585d307` —
the vanilla baseline `agent-memory/domain_expert_baseline.md` calls *"recoverable from that commit"*,
which until this push existed on one disk.

**Out of scope.** Touching `scripts/validation/results/` — 13 GB, untracked, unregenerable.

---

### S2 — pytest harness ✅ DONE 2026-08-11
**Mostly exclusive files. Depends on: nothing — run this on `peng-training-branch` BEFORE the move.**

Doing this first is deliberate: the harness itself is exclusive and gives a **passing baseline on
the old branch** to diff the port against. Only the atomic `pyproject.toml`/`uv.lock` dependency
change is shared. Every later section now has a regression gate.

📄 **Detailed technical plan and execution record:
[S2_PYTEST_HARNESS_PLAN.md](S2_PYTEST_HARNESS_PLAN.md).**

**Working branch: `peng-training-branch`.** Everything here is exclusive except the two dependency
files below.

**Two findings from execution that shrink this section:** 10 of 12 check modules already define
module-level `test_*()` functions, so pytest collects them unmodified; `smoke_test_seed_2a.py` and
`test_lib_env_api.py` only needed thin wrappers. And the "duplicated fixtures" below are **name
collisions, not duplication**
(`_record` ×3, `_rollout` ×2, `_metric` ×2 all have different signatures; `_fake_env` exists once),
so nothing is merged. See §4b Cluster F, which is corrected accordingly.

**In scope.**
- Add `pytest` to `pyproject.toml`, update `uv.lock`, and add a `conftest.py`.
  ⚠️ **`pyproject.toml` and `uv.lock` are SHARED.** Send them upstream atomically: a pyproject-only
  change breaks `uv sync --locked`. Everything else in this section is exclusive — `origin/dev`
  has no `checks/` directory.
- Wire the 12 runnable modules under `bench_script/checks/` into pytest, keeping
  `python -m checks.<module>`
  working so existing habits and docs don't break.
- Keep the similarly named local helpers separate; they have different signatures and purposes.
- Mark GPU-only tests `@pytest.mark.gpu`, skipped by default: only `smoke_test_seed_2a` needed it.

**Out of scope.** Changing what any check asserts. Adding new tests.

**Result:** 50 items collected across 12 runnable modules; **49 passed, 1 GPU smoke skipped**. The
legacy `test_ring_config`, `test_obstacle_set`, and `test_lib_env_api` module commands also pass.

**Verify with**
```bash
cd customized_robotwin/script/bench_script && ../../../.venv/bin/python -m pytest -q
../../../.venv/bin/python -m checks.test_ring_config
../../../.venv/bin/python -m checks.test_obstacle_set
```

---

### S3 — Branch and port
**DONE 2026-08-11. SHARED files — highest care in the whole plan. Depends on: S1, S2.**

📄 **Detailed technical plan: [S3_BRANCH_AND_PORT_PLAN.md](S3_BRANCH_AND_PORT_PLAN.md)** — approved,
executed. The summary below is orientation only.

⚠️ **This summary was incomplete and the detailed plan corrects it.** Two gaps found during
planning:

- **Five files inside `bench_script/` are SHARED with dev and cannot be bulk-copied** — each needs
  an explicit decision: `analyze_occluder_visibility.py` (ours 447 L vs dev's 2,736 L),
  `reachability_map.py` (351 vs 416), `visualize_task_scene.py`, `diag_kitchen_curobo.py`,
  `setup_paths.py`. The first two are **add/add** conflicts where neither side is takeable whole.
- **The edge-to-edge occluder fix (`cf966f7`) lives inside dev's `analyze_occluder_visibility.py`
  and exists on our branch in no form at all.** Taking "ours" silently discards it. The detailed
  plan adds a dedicated step to lift the ~127 lines of footprint geometry into `lib/`, isolated in
  its own commit because it is the only part of S3 that changes scene semantics.

The raw source-only count becomes **107** when this S3 plan is committed, not 102 (the earlier figure
counted only files *changed since the fork*). S3 deliberately excludes `swept_volume_3d.py`, leaving
**106 retained files** to port, including `agent-memory/`, `plans/`, and S2's `conftest.py` /
`pytest.ini`.

**In scope.**
- Branch `peng-dev-new` off **`origin/dev` @ `64840ce`** — not local `dev`, which is two months
  stale at `4b57b67`. Cherry-pick `18abd2e` so the pytest gate can run.
- Bulk-copy the 106 exclusive files from `peng-training-branch` (no conflicts are possible; they
  exist on no other branch), then resolve the five shared ones above by decision.
- Apply the shared carry-over listed in Part 2: `Robot.build_planner`, `seed_traj` +
  `attempts`/`trajopt_attempts`/`seeded`, `pi05_robopro_top_cam_jax` TrainConfig, `eval.sh` arg
  passing, `put_mouse_on_pad: 600`, the `include_collision` typo shim, two `.gitignore` lines.
- Re-apply the **five `_bench_base_task.py` hooks** onto dev's version, plus
  `benchmark/bench_envs/eval_video.py`. See Part 2 — `setup_demo` is load-bearing for S5.

**Out of scope — do NOT carry these; taking ours regresses dev.**
`envs/_base_task.py`, `envs/camera/camera.py` (ours is behind — dev moved segmentation to raw
`uint16`), `envs/utils/pkl2hdf5.py`, `bench_envs/_bench_base_task.py` wholesale, the four per-env
base tasks, `deploy_policy.py`, `Makefile`, `pyproject.toml`/`uv.lock`, `task_objects.yml`.
Do not set `action_noise_var: 0.001` — dev set it to 0 deliberately in `1f4a5f3`.

**Done when** `--help` passes on every top-level command and S2's pytest reports the same pass rate
as the baseline.

**Completion:** `peng-dev-new` was created from `origin/dev@64840ce`; all 106 retained exclusive
files were ported byte-identically before shared carry-over, and `swept_volume_3d.py` remained
excluded. The five omitted shared `bench_script/` files were resolved explicitly. Final CPU result:
**49 passed, 1 skipped**; all top-level help commands and the `lib/task` layering gate pass. The
edge-to-edge geometry is isolated in its own commit and its ring baseline was reviewed. Runtime/GPU
reconciliation remains S4.

**Verify with**
```bash
for f in customized_robotwin/script/bench_script/*.py; do python "$f" --help >/dev/null || echo "FAIL $f"; done
cd customized_robotwin/script/bench_script && python -m pytest -q
```

---

### S4 — Make it run
**SHARED files. Depends on: S3. Needs one GPU run from the user.**

📄 **Detailed technical plan: [S4_MAKE_IT_RUN_PLAN.md](S4_MAKE_IT_RUN_PLAN.md)** — **completed
2026-08-12**. The summary below is orientation; the detailed plan holds the execution record.

**Three things planning settled that this summary did not know:**

- The `update_world` fix is **one argument in one exclusive file** (`task/occluder_task.py:198` →
  `exclude_obstacles=False`). Dev's `_office_base_task` guards its own bare call behind
  `if not exclude_obs`, so it is self-consistent — **do not touch it.**
- **Segmentation needed runtime confirmation.** The target mask actually comes from dev's raw actor
  segmentation path; RGB is only used to draw the overlay. The built-scene check resolved target id
  75 and measured 207 target pixels, so the ported raw-ID path works.
- **The pre-rehaul comparison must run on the 0-occluder `standard` config.** Comparing the ring
  path across branches is confounded — S3's edge-to-edge change means the same nominal offset builds
  a physically different scene, so a success-rate diff would measure the geometry, not the port.

Also: `customized_robotwin/envs/curobo` is **gitignored**, so the `seed_traj` patch is untracked and
`uv sync` silently reverts it. S4 adds a CPU check module, moving the suite baseline to
**50 passed / 1 skipped**.

S3 makes it import; this makes it work against six weeks of dev changes.

**In scope.**
- **`update_world()` default flip — the top semantic risk.** Ours declares
  `exclude_obstacles: bool = False`; dev's declares `= None`, resolving via
  `planner_exclude_obstacles` then `enable_collision_metrics`, which `bench_demo_office_clean.yml`
  sets **true**. A bare call therefore flips from *include all clutter* to *exclude all clutter* —
  silently, no error, rollouts just resume knocking clutter over. Make every ring-path call pass
  `exclude_obstacles=False` explicitly.
- Reconcile against dev's raw-`uint16` segmentation and its collision-metric streaming
  (`start_metric_streams`, `_init_proximity_tracking`, `_mark_intended_contact`).
- Confirm `seed_traj` still reaches the vendored curobo patch (`curobo_seed_traj.patch`, applied by
  hand; `run_approach_mode_ab.sh` checks for it).

**Constraint.** The agent prepares and hands over the command; **the user runs the GPU rollout.**

**Done when** one `analyze_occluder_visibility.py` run completes end to end with `records.jsonl` and
`rollout_seed_stats` populated, and `test_ring_config` passes. Note its formation baselines **will**
change once edge-to-edge spacing lands — review that diff, do not suppress it.

**Completed evidence.** CPU baseline: **50 passed, 1 skipped**. The live geometry check measured a
requested 0.100 m collision-mesh gap as 0.109905 m (within the 0.010 m gate) with a nonempty raw
target mask. The cluttered ring rollout succeeded, wrote records/video/HDF5, recorded zero physics
collisions, and proved clutter delivery with CuRobo world entries **11 full > 2 default**. On matched
zero-occluder standard seeds 0–4, `peng-dev-new` and `peng-training-branch` were identical: **3/5**
success, seed 2 failed at `grasp`, and seed 4 at `placement:pre_place_descent`. This rules out gross
port divergence for the smoke set; n=5 does not measure modest success-rate changes.

---

### S5 — 84-file cleanup script
**SHARED files, but mechanical. Depends on: S3. Independent of S6–S12.**

Reproduce our systematic cleanup as a **scripted pass on top of dev's versions** — do not port the
files. Net +265/−1,000 across 88 files, six mechanical kinds, no per-file design content:
delete the boilerplate `setup_demo` override (80 files); prune unused imports (`glob` ×59,
`math` ×58, `deepcopy` ×57, `sapien` ×49); delete commented-out `self.info["info"]` blocks
(18 files); `include_collison` → `include_collision`; delete four dead per-task helpers; normalise
trailing whitespace.

**Out of scope.** `study/put_cup_on_coaster.py` — both sides fixed the same missing-`abs` bug and
dev's is equivalent. Skip it.

**Critical.** dev spent six weeks fixing `check_success` false positives in 22 of these files
(`d7427da`, `0dce73c`, `6697bb4`, `82a776d`), including one where every seed IK-failed. Those live at
lines ~95–200; our edits live at lines 1–35. **Do not touch `check_success` or `play_once` bodies.**

**Done when** 80 files have lost their `setup_demo` override, dev's `check_success` bodies are
byte-identical to `origin/dev`, and `--help` still passes.

---

### S6 — `envs/` doors, five down to one
**Exclusive files. Depends on: S4.** This is the lever that actually delivers cheap future merges.

`lib/scene_build.py` was meant to be the only door into `envs/`. There are five:
`lib/scene_build.py` (`CONFIGS_PATH`, `Base_Task` — the intended one); `task/occluder_task.py`
(`ArmTag`, `create_actor`, `create_box`, `rand_pose`); `task/pose_geometry.py` and
`task/placement_mixin.py` (`envs._GLOBAL_CONFIGS.GRASP_DIRECTION_DIC`); `vla_rollout.py`
(`envs.utils.create_actor.UnStableError`); `visualize_task_scene.py` and `diag_kitchen_curobo.py`
(`CONFIGS_PATH`).

**Also in scope.** Make the duck-typed env calls an explicit protocol — `lib/ik_grid.py` reaches
`env.target_obj`, `env._pick_side_grasp_id`, `env._geometric_grasp_pose`; `lib/visibility.py` reaches
`env.play_once`, `env._take_picture`. `checks/test_lib_env_api.py` exists solely as an AST guard
because `a301e2e` deleted a method `lib/` was calling and **a full day of GPU runs measured
nothing**. Keep the guard regardless.

**NO-BEHAVIOUR-CHANGE.** Import routing and an explicit protocol only.

**Done when** nothing outside `lib/scene_build.py` imports from `envs.`, and `test_lib_env_api`
passes.

**Verify with**
```bash
cd customized_robotwin/script/bench_script
grep -rn "from envs\.\|import envs" --include=*.py . | grep -v 'lib/scene_build.py'   # must be empty
python -m checks.test_lib_env_api
```

---

### S7 — One config surface
**Exclusive files. Depends on: S4. Blocks S8, S9, and any future entry-point work.**

`lib/metric_config.py::SeedMetricConfig` was designed as the single source (`from_args` plus a
`SEED_<FIELD>` env overlay) and got bypassed. Today there are three surfaces: `task_metric.py`
(32 args), `clearance_metric_3d.py` (31) and `reach_envelope.py` (27) each re-declare overlapping
grid flags, and `lib/planning_tuning.py` is a separate env-var surface.

**In scope.**
- Route the overlapping grid flags through `SeedMetricConfig`.
- Kill `from lib.planning_tuning import *` and `from lib.scene_constants import *` — star-imported
  by all 7 `task/` modules plus `analyze_occluder_visibility.py`, which makes "who uses which
  constant" statically unanswerable.
- Fix `lib/vla_reporting.py`'s absolute `from lib.run_io import ...`; every other intra-`lib` import
  is relative, and this one pins `lib/` to being rooted at `bench_script/`.

**NO-BEHAVIOUR-CHANGE.** Defaults must not move. `SeedMetricConfig` defaults are x[−0.6,0.6]
y[−0.35,0.35] res 0.01, z[0.78,1.23] zres 0.03, gate_tau 0.35, ik_seeds 30, chunk 256,
occ_shape "mesh", obstacles "all".

**Done when** no `import *` remains in `task/`, and the three scripts' arg counts drop.

---

### S8 — Cluster A, the shared gated driver
**Exclusive files. Depends on: S7.**

There are **two** eps\* quantities, not three: gated and geometric. `lib/geometric_metric.py` calls
only `build_grid` + `widest_path_eps_3d` — 2 of the 5 steps, no IK solver — so it is a shorter
pipeline, not a duplicate. **Leave it alone.**

The real duplication is that `clearance_metric_3d.py` and `lib/seed_from_clearance.py` independently
drive the same full 5-step gated sequence: `build_grid` (:189 / :153), `label_volume` (:230 / :157),
`warm_start_branches_3d` (:267 / :107), `widest_path_eps_3d` (:92,:95 / :197,:198,:213),
`reconstruct_widest_path_3d` (:98 / :250). Extract one shared driver; keep both entry points. They
are not identical — the seed path adds a tau bisection the metric path lacks.

**Also fix.** `lib/seed_from_clearance.py`'s docstring claims it "reuses clearance_metric_3d.py
verbatim as a library — none of the metric maths is re-implemented here." That is false and hides
this duplication; the 2026-07-29 layering rule forced it onto the `lib/` primitives.

**Out of scope.** `reach_envelope.py` / `validate_reach_envelope.py` — the producer/consumer/
validator split is deliberate. `agent-memory/tool_reach_envelope.md` says outright: do NOT re-tangle
compute/viz back into the runs file.

**NO-BEHAVIOUR-CHANGE.**

**Done when** the offline CPU rebuild reproduces recorded `eps_geom` values at max abs diff
0.000e+00. That trick is documented in `agent-memory/tool_task_metric_validity.md`: `records.jsonl`'s
`scene_fingerprint_source.scene.metric_obstacles` carries enough to rebuild the exact input with no
SAPIEN and no GPU, ~40 s/scene.

---

### S9 — Clusters C and D, visualization and record loading
**Exclusive files. Depends on: S7.**

- **Visualization (~1,470 LOC).** Two ceiling/heatmap renderers and overlapping 3D drawing across
  `lib/plotting.py`, `lib/metric_viz.py`, `lib/metric_diagnostics.py`, `lib/reachability_view.py`,
  `visualize_task_metric_routes.py` — the last is 861 LOC whose actual drawing is one call into
  `metric_viz._metric_path3d`.
- **Record loading (~1,554 LOC).** `analyze_metric_correlation.py` and
  `analyze_metric_distribution.py` both ingest the same JSONL and both bucket/histogram it.
  `lib/vla_reporting.summarize()` collides by name with `analyze_metric_correlation.summarize()`;
  `_eps(...)` is defined twice independently.

**Out of scope — do not merge these two scripts.** The distribution/correlation split is a
deliberate methodological firewall: `analyze_metric_distribution.py` *rejects* input containing
outcome/success/HSR fields, so bucket boundaries get fixed while blind to outcomes. Share the
loader, keep the separation.

**NO-BEHAVIOUR-CHANGE.**

**Done when** figures and summary numbers are unchanged on a recorded run.

---

### S10 — `task/` instrumentation
**Exclusive files. Depends on: S6. Needs a GPU run from the user — schedule early.**

2,654 LOC of which ~2,000 is retry/fallback plumbing. **Carry it over unchanged, add counters, then
measure.** The expert's success rate is an empirical property of exactly this plumbing.

**In scope — make explicit without changing behaviour.**
- `self._last_fail_reason` — a string-typed error channel written at 18 sites across three modules.
- `self._grasp_baseline_transform` / `self._slow_attached_replay` — meaning differs before vs after
  grasp; genuine temporal coupling.
- `self.rollout_plan_effort` / `self.rollout_seed_stats` — initialised in `play_once`, yet
  `planning_mixin._record_plan_effort` and `seeding_mixin._note_seed_stat` both defensively re-create
  them with `hasattr` guards; those guards are evidence init order isn't guaranteed.
- `self._approach_seed_cache` / `self._carry_seed_cache` — never cleared by `play_once`, so lifetime
  is per-instance rather than per-episode.
- `APPROACH_MODE` / `PLACEMENT_MODE` read from `os.environ` inside `seeding_mixin`, making behaviour
  mode process-global rather than a constructor argument.
- Add a counter to `seeding_mixin._get_approach_seed`'s catch-all `except` — **do not remove it
  yet.** It exists so seeding can never break the expert; it is also how a silent failure went
  unnoticed for a day of GPU runs.

**NO-BEHAVIOUR-CHANGE.** Counters and explicit state only.

**Done when** counters appear in the records and success rate is unchanged on a matched seed set.
`rollout_seed_stats` and the summariser's `[SEED FIRING]` block must keep working —
`feedback_scientific_rigor.md` records this as the most expensive lesson of 2026-07-30: isolation is
not enough, you must measure the treatment was actually delivered.

---

### S11 — `task/` prune
**Exclusive files. Depends on: S10 and the user's GPU run.**

Delete only retry branches the counters show never fired. **Evidence-gated: no branch is removed on
judgement.** If a branch fires rarely, report it and ask rather than cutting.

**Done when** every removed branch has a counter reading of zero over the measured run, and success
rate is unchanged on a re-run.

---

### S12 — Supporting layer
**Mixed. Depends on: S4.**

**In scope.**
- Generate the Makefile `help` target — 55 hand-maintained printf lines that duplicate every one of
  the ~50 `?=` variables.
- Deduplicate the four near-identical 12-flag eval invocations.
- Split `summarize_approach_mode_ab.py` (1,239 LOC against the project's own 1,000-line ceiling).
- Fix `analyze_occluder_rollout_failures.py`'s default `--results-dir`, which points at
  `phase2_occluder_rollout` — a directory that does not exist. Live one is
  `occluder_visibility/{rollout,no_rollout}`. **That file is shared with dev — send it as a PR.**

**Out of scope — nothing under `scripts/` gets deleted.** `scripts/upload/`, `scripts/slurm/` and the
five June OOD scripts all exist on `origin/dev` unmodified by us; deleting them would be a
shared-file edit, which is the exact merge tax this rehaul exists to remove.

**Keep verbatim.** `scripts/install/` (221 LOC) — without `patch_aloha_curobo.py`, CuRobo's
`attach_external_objects_to_robot` raises `KeyError: 'attached_object'`. And the ~50 Makefile
variables; they *are* the experiment config file.

**Done when** `make help` is generated rather than hand-written and no Makefile target references a
missing path.

---

### Not sectioned, deliberately

- **§4e entry-point collapse (14 → ~4 subcommands)** — awaiting an explicit yes. It renames every
  command and breaks Makefile targets, `run_approach_mode_ab.sh:220`, and command strings recorded
  throughout `agent-memory/` and `plans/`.
- **The fingertip-clearance validity study** — CPU-only, touches no shared file, orthogonal to the
  rehaul, and can run in parallel at any time. Kept out because it is a *research* unit; folding it
  into an engineering section invites an agent to "fix" the metric during a refactor.
- **Part 6b, the `phase4_approach_mode` `off` cell** — deferred by user decision; the user runs it.

---

## Part 0 — Safety net

**64 of this branch's 82 commits exist only on this machine.** `git branch -r --contains HEAD` is
empty.

- Push `peng-training-branch` to origin as a backup ref.
- Tag the current tip `e3a09ce`.
- `scripts/validation/results/` is **13 GB, untracked, unregenerable** (rollouts are
  non-deterministic by design). No branch operation touches it; keep it that way.

Effort: minutes. No alternative.

---

## Part 1 — Establish `peng-dev-new`

Branch off **`origin/dev` @ `64840ce`**, not local `dev` — which is two months stale at `4b57b67`,
and that staleness is exactly what made an earlier pass conclude there was nothing to merge.

Effort: minutes.

---

## Part 2 — Carry the thin shared surface (~150 lines)

| Item | File | Lines |
|---|---|---|
| `Robot.build_planner` opt-out | `customized_robotwin/envs/robot/robot.py` | ~50 |
| `seed_traj` + `attempts`/`trajopt_attempts`/`seeded` keys | `envs/robot/planner.py`, `robot.py` | ~30 |
| `pi05_robopro_top_cam_jax` TrainConfig | `policy/pi05/src/openpi/training/config.py` | ~15 |
| `eval.sh` checkpoint arg passing | `policy/pi05/eval.sh` | ~10 |
| `put_mouse_on_pad: 600` | `task_config/_eval_step_limit.yml` | 1 |
| `include_collision` typo shim | `bench_envs/study/_study_base_task.py` | 1 |
| `.gitignore` additions | root | 2 |
| **The five `_bench_base_task.py` hooks** (below) | `bench_envs/_bench_base_task.py` | ~80 |
| `bench_envs/eval_video.py` | new file, dev has none | ~40 |

**The five hooks are not optional.** "Do not carry the file" means re-apply these onto dev's
version, not skip them:

1. `setup_demo(self, is_test=False, **kwargs)` base hook — **Part 3 depends on this.** The 84-file
   cleanup deletes each task's `setup_demo` override *because the base class provides it*. Carry the
   84-file cleanup without this hook and all 80 files break.
2. `_write_eval_video_frame()` + `select_eval_video_camera` (+ the new `bench_envs/eval_video.py`).
   Dev landed its own `EVAL_VIDEO_CAMERAS` in `358acdd` — **verify whether dev's covers your case
   before carrying yours**; this is the one genuinely open question in Part 2.
3. `allow_duplicate_clutter` retry-budget change in the clutter loop.
4. The `if not self.robot.build_planner:` bypass of the TOPP block. **Pairs with the
   `Robot.build_planner` row above** — carrying one without the other leaves the flag non-functional.
5. The opt-in `step_hook` per-physics-step observer.

**Do not carry:** `envs/_base_task.py`, `envs/camera/camera.py`, `envs/utils/pkl2hdf5.py`,
`bench_envs/_bench_base_task.py`, the four per-env base tasks, `deploy_policy.py`, `Makefile`,
`pyproject.toml`/`uv.lock` (the uv migration is shared history — `origin/add-uv-support` is already
an ancestor of both sides), `task_objects.yml` (byte-identical to dev). **My `camera.py` is a
regression** — dev moved segmentation to raw `uint16` ids; my copy still has the `ImageColor`
palette.

Two traps to respect rather than revert:
- `deploy_policy.yml` — dev set `action_noise_var: 0` deliberately (`1f4a5f3`). My `0.001` would
  silently undo it. Carry only the `collect_mode: branching` keys.
- `Makefile` — dev renamed `OCC_OFFSETS` → `OCC_DISTANCE_CM` as part of the edge-to-edge fix.

Per the track-dev-closely policy, rows 1–2 go to dev as **two small PRs**, not onto the research
branch.

Effort: ~half a day, mostly verification.

---

## Part 3 — Re-derive the 84-file cleanup as a script

My edit to the 84 `benchmark/bench_envs/**` task files is **one systematic cleanup**, net
+265/−1,000, in six mechanical kinds: delete the boilerplate `setup_demo` override (80 files), prune
unused imports (`glob` ×59, `math` ×58, `deepcopy` ×57, `sapien` ×49), delete commented-out
`self.info["info"]` blocks (18 files), fix `include_collison` → `include_collision`, delete four dead
per-task helpers, normalise whitespace. No per-file design content.

Run it as a **scripted pass on top of dev's versions**. Do not port the files: dev spent six weeks
fixing `check_success` false positives in 22 of these same files (`d7427da`, `0dce73c`, `6697bb4`,
`82a776d`) — including one task where every seed IK-failed. Those fixes sit in
`check_success`/`play_once` (~lines 95–200); my edits sit in the header (lines 1–35). Disjoint, and
git auto-merges 21 of 22.

Skip `study/put_cup_on_coaster.py` — both sides fixed the same missing-`abs` bug; dev's is
equivalent.

Effort: ~half a day, and re-runnable if it goes wrong.

---

## Part 4 — The research layer (the bulk of the work)

`customized_robotwin/script/bench_script/` — 61 files, **17,051 LOC**:

| Area | Files | LOC |
|---|---|---|
| Root entry points | 14 | 6,432 |
| `lib/` | 25 | 5,347 |
| `task/` | 8 | 2,654 |
| `checks/` | 12 | 2,618 |

Per your decisions the task-metric line stays **live** and the reach/clearance tools come over. So
this part is mostly consolidation.

**Not ported (decided 2026-08-11):**

- **`swept_volume_3d.py` (412 LOC)** — the one genuine staleness outlier: last run 2026-07-15,
  27 days, while every other tool clusters at 11–15. A one-off issue-#35 visualisation with no
  design rationale for hand-running it (unlike the reach-envelope trio). Its 285 MB of output under
  `results/swept_volume_3d/` stays on disk; recover the script from git history if needed.
- **`compare_geometric_vs_gated.py`** — was never in scope. It has been cold source under
  `research_archive/bench_script_task_metric_2026/` since `3dc788a`, and Part 6 leaves that
  directory untouched. Listed here only so it does not get revived by accident. Its 2.8 MB of
  results stay.

**Stage 3 status, for the record.** The final comparison run (`geometric_vs_gated/20260731-110834`)
returned **Spearman 1.0, p = 0.0** between geometric and gated eps\*, past the 0.8 threshold — but
`rank_gate_pass: false`, because only 6 of 10 scenes completed (4 were not alignable) and the runner
judged n = 6 insufficient. Treat "geometric ≈ gated" as concluded-but-under-powered. It is a
separate question from whether eps\* predicts grasp difficulty, which
`agent-memory/tool_task_metric_validity.md` measured at rho = 0.078 on 2026-08-07.

Consequence for Cluster A below: with geometric ≈ gated accepted, geometric can serve measurement
while gated continues to serve seeding. `clearance_metric_3d.py` stays either way, because seeds
must keep coming from the proper gated graph via `compute_route_configs`.

**Stale docstring to fix while porting.** `lib/seed_from_clearance.py`'s header claims *"It reuses
clearance_metric_3d.py verbatim as a library — none of the metric maths is re-implemented here."*
That is no longer true: its import block references `.continuity`, `.ik_grid`, `.labeling` and
`.widest_path` directly, and no `clearance_metric_3d` at all. The 2026-07-29 layering rule
("library code must never import from a CLI script") forced the rewiring, and the docstring was
never updated. It now actively misdescribes the file.

**The good news:** the 2026-07-29 refactor's layering already holds. `lib/ ← task/ ← root scripts`,
no import cycles, and the rule "library code must never import from a CLI script" verifies clean.
Keep that rule; it is the most valuable thing in the tree.

### 4a. Cut the coupling to `envs/` — five doors down to one

This is the lever that actually delivers *cheap future merges*, and it is the cheapest item here.
`lib/scene_build.py` was meant to be the single sanctioned door into `envs/`. There are five:

- `lib/scene_build.py` — `CONFIGS_PATH`, `Base_Task` *(the intended one)*
- `task/occluder_task.py` — `ArmTag`, `create_actor`, `create_box`, `rand_pose`
- `task/pose_geometry.py`, `task/placement_mixin.py` — `envs._GLOBAL_CONFIGS.GRASP_DIRECTION_DIC`
- `vla_rollout.py` — `envs.utils.create_actor.UnStableError`
- `visualize_task_scene.py`, `diag_kitchen_curobo.py` — `CONFIGS_PATH`

Route all five through `scene_build.py`. Then every future dev change to `envs/` has exactly one
file to check.

**Related and higher-risk:** `lib/` calls duck-typed methods on the env handle — `env.target_obj`,
`env._pick_side_grasp_id`, `env._geometric_grasp_pose` (`lib/ik_grid.py`), `env.play_once`,
`env._take_picture` (`lib/visibility.py`). This is an invisible `lib → task` dependency.
`checks/test_lib_env_api.py` exists solely as an AST guard because commit `a301e2e` deleted a method
`lib/` was calling and **a full day of GPU runs measured nothing**. Make these an explicit protocol;
keep the guard either way.

Also: `make_occluder_task()` builds the class at call time from the stock RoboTwin
`put_mouse_on_pad` office task. That is a live dependency on a dev-owned file that no file-level
diff reveals. (Checked: dev's success-criteria sweep `d7427da` did *not* touch `put_mouse_on_pad`,
so ring-task success labels are unaffected — but this needs re-checking on every merge.)

### 4b. Collapse the duplication clusters

Six clusters, measured:

| Cluster | LOC | What |
|---|---|---|
| **A — two drivers of the gated pipeline** | ~900 | **Corrected 2026-08-11 — an earlier draft said "three eps\* pipelines"; that was wrong.** There are **two eps\* quantities**: gated (IK + continuity gate, GPU) and geometric (`lib/geometric_metric.py`, CPU envelope-relaxed, which calls only `build_grid` + `widest_path_eps_3d` — 2 of the 5 steps, no IK solver, so it is a shorter pipeline, not a copy). The real duplication is that **`clearance_metric_3d.py` and `lib/seed_from_clearance.py` independently drive the same full 5-step gated sequence** — `build_grid` (:189 / :153), `label_volume` (:230 / :157), `warm_start_branches_3d` (:267 / :107), `widest_path_eps_3d` (:92,:95 / :197,:198,:213), `reconstruct_widest_path_3d` (:98 / :250). Not identical: the seed path adds a tau bisection the metric path lacks. Extract the shared driver; keep both entry points. |
| **B — reach tooling** | — | **Dropped as a consolidation target.** `reach_envelope.py` / `clearance_metric_3d.py` / `validate_reach_envelope.py` is a producer / consumer / validator split the user explicitly insisted on — *"precompute once, never again; clearance_metric are ACTUAL RUNS"* — and `tool_reach_envelope.md` says outright: do NOT re-tangle compute/viz back into the runs file. Leave the split alone. At most, share the argparse preamble. |
| **C — visualization** | ~1,470 | Two ceiling/heatmap renderers and overlapping 3D drawing across `lib/plotting.py`, `lib/metric_viz.py`, `lib/metric_diagnostics.py`, `lib/reachability_view.py`, `visualize_task_metric_routes.py` (861 LOC whose actual drawing is one call into `metric_viz._metric_path3d`). Shrunk from 1,881 — `swept_volume_3d.py` is no longer ported, which removes one of the three volume renderers outright. |
| **D — record loading / stats** | 1,554 | `analyze_metric_correlation.py` and `analyze_metric_distribution.py` both ingest the same JSONL, both bucket and histogram it. `lib/vla_reporting.summarize()` collides by name with `analyze_metric_correlation.summarize()`; `_eps(...)` is defined twice independently. |
| **E — three scene drivers** | — | `analyze_occluder_visibility.py`, `vla_rollout.py`, `visualize_task_scene.py` each do config load → `build_cfg`/`DR_CLEAN` → env construction → seed loop → video/record writing, with three CLI vocabularies (17 / 22 / 11 args). |
| **F — check fixtures** | **0** | **Corrected 2026-08-11: there is no duplication here.** The earlier claim (fake-env stubs across four checks, `_record()` three times) counted *names*, not functions. `_record` appears 3× with signatures `(episode, density, config_hash, *, hard_success)` / `(scene, seed, values, clutter, outcome=False)` / `(value)`; `_rollout` 2× and `_metric` 2×, likewise unrelated; `_fake_env` exists once (`test_vla_office_smoke.py` has a different function, `_task_api_env`). These are correctly-scoped local helpers. **Merge nothing.** |

### 4c. Make config one surface again

`lib/metric_config.py::SeedMetricConfig` was supposed to be the single source (`from_args` +
`SEED_<FIELD>` env overlay). It is now three: `task_metric.py` (32 args), `clearance_metric_3d.py`
(31) and `reach_envelope.py` (27) each re-declare overlapping grid flags, and
`lib/planning_tuning.py` is a separate env-var surface. Kill the star-imports — `from
lib.planning_tuning import *` and `from lib.scene_constants import *` are star-imported by all 7
`task/` modules plus `analyze_occluder_visibility.py`, which makes "who uses which constant"
statically unanswerable.

Fix `lib/vla_reporting.py:20`'s absolute `from lib.run_io import ...` (every other intra-`lib`
import is relative), which is what currently pins `lib/` to being rooted at `bench_script/`.

### 4d. `task/` — port, instrument, then prune

2,654 LOC of which ~2,000 is retry/fallback plumbing. Per your decision: **carry it over unchanged,
add counters, run once, delete what never fires.**

What to make explicit while porting, without changing behaviour:
- `self._last_fail_reason` — a string-typed error channel written at 18 sites across three modules.
- `self._grasp_baseline_transform` / `self._slow_attached_replay` — meaning differs before vs. after
  grasp. Genuine temporal coupling.
- `self.rollout_plan_effort` / `self.rollout_seed_stats` — initialised in `play_once`, but
  `planning_mixin._record_plan_effort` and `seeding_mixin._note_seed_stat` both defensively
  re-create them with `hasattr` guards. Those guards are evidence the init order isn't guaranteed.
- `self._approach_seed_cache` / `self._carry_seed_cache` — never cleared by `play_once`; lifetime is
  per-instance, not per-episode.
- `APPROACH_MODE` / `PLACEMENT_MODE` read from `os.environ` inside `seeding_mixin`, so behaviour mode
  is process-global rather than a constructor argument.

**Do not touch** `seeding_mixin._get_approach_seed`'s catch-all `except` yet — it exists so seeding
"can never break the expert." It is also exactly how the silent failure above went unnoticed. Add a
counter first, decide after.

The instrumented run is a GPU run, so it is yours to execute and it sits on the critical path.

Effort: 5–8 days, dominated by 4b.

---

## Part 5 — Supporting layer

Better shape than expected. `scripts/` is 16 files / 3,158 LOC; the root `Makefile` is 336 LOC /
23 targets with **zero dangling file paths** — the recent prune commits updated it in lockstep.

**Keep verbatim:**
- `scripts/install/` (221 LOC, 3 files). The repo is unusable without it — without
  `patch_aloha_curobo.py`, CuRobo's `attach_external_objects_to_robot` raises
  `KeyError: 'attached_object'`.
- The `RUN_IN_CUSTOMIZED` contract: `cd customized_robotwin && source set_env.sh && export
  ROBOTWIN_BENCH_TASK=bench`. Without that export the system silently uses upstream RoboTwin configs
  instead of benchmark configs.
- The ~50 `?=` Makefile variables — they *are* the experiment config file; there is no other one.
- `bench_script/curobo_seed_traj.patch` (157 lines), applied by hand to `envs/curobo` and kept in
  sync with `lib/seed_from_clearance.py`.

**Drop: nothing. This was an error in the first draft — corrected 2026-08-11.**

The three "drop" candidates (`scripts/upload/` 152 LOC, the five June-2026 OOD/compositional
scripts 856 LOC, `scripts/slurm/` 134 LOC) **all exist on `origin/dev`, unmodified by this branch
since the fork.** They are not mine to delete. Removing them would be a deletion diff against dev —
a shared-file edit, i.e. exactly the permanent merge tax this rehaul exists to eliminate, and it
would also remove the team's only OOD-asset validation and their cluster eval path.

They cost nothing to leave in place. Leave them.

The only supporting-layer change worth making:
- Fix `analyze_occluder_rollout_failures.py`'s default `--results-dir`, which points at
  `phase2_occluder_rollout` — a directory that does not exist. The live one is
  `occluder_visibility/{rollout,no_rollout}`. (This file *is* shared with dev, so send it as a PR.)

**Adopt `pytest`.** This is the highest-leverage item in the entire rehaul: wire the 12 existing
`checks/` modules into a real runner. Everything in Part 4 is a restructure with no safety net until
this exists, so **do it first, not last.** Roughly a day.

Also rebuild (behaviour preserved, no capability lost): the 55-line hand-maintained `help` printf
block (duplicates every variable), the four near-identical 12-flag eval invocations, and
`summarize_approach_mode_ab.py` at 1,239 LOC against the project's own stated 1,000-line ceiling.

**Note on the "dead weight removal" goal — honest sizing.** With the `scripts/` drop list retracted
and the task-metric line kept live, the only outright removal is `swept_volume_3d.py` (412 LOC).
The rest of the goal is served by *consolidation*.

An earlier draft of this plan cited "~5,900 LOC collapsed." That was the **size of the duplication
clusters, not the recoverable overlap**, and it overstated the result. Realistic per-cluster saving:
A 250–400 (one shared gated driver across two callers), C 300–400, D 200–400, E 200–300, B zero
(dropped as a target). Call it **950–1,500 lines net**, against a plan that also *adds* pytest
fixtures, an explicit env protocol, explicit state in place of implicit, and retry counters.

Expect roughly **10% smaller, not 30%**. And note the standing risk: refactoring without tests tends
to accrete defensive code, so if Part 5's runner slips, Part 4 makes the tree bigger rather than
smaller.

**If leanness matters more than that, the lever is entry-point count, not duplication.** Fourteen
root commands with three different CLI vocabularies is what makes this tree feel heavy; collapsing
them behind a few entry points with subcommands would do more for legibility than every
consolidation cluster combined, and removes zero functionality. Not currently in scope — flagged as
the obvious Part 4e if you want it.

Effort: ~1.5 days, mostly the test runner.

---

## Part 6 — Data and reproducibility continuity

Partial by decision: existing results stay readable, new runs use new semantics, the two are never
pooled. That requires:

- **Do not move or rename `scripts/validation/results/<topic>/<timestamp>/`** — a literal path baked
  into `summarize_approach_mode_ab.py` and `analyze_occluder_rollout_failures.py`.
- **Do not change the `records.jsonl` schema** without a version field.
- **Preserve `rollout_seed_stats`** (per-episode `built`/`reason`/`leg`) and the summariser's
  `[SEED FIRING]` block. `feedback_scientific_rigor.md` records this as the most expensive lesson of
  2026-07-30: isolation is not enough — measure that the treatment was actually *delivered*.
- Protect `phase4_approach_mode/20260730-160723/curated/direct/` (the 8/15 milk-box run behind the
  project's strongest claim, Fisher p = 0.0008) and
  `task_metric_vla_full/association_d6_d10_d15/20260731-182037/` (3,000 episodes, 620 non-resumable
  metric records, 200 route figures).
- Keep the three seed bands intact: train `0,1,2,…`, eval `40000+`, upstream `100000*(1+s)`.
- **Free win:** dev's PR #68 wires `benchmark/eval_seeds/` into `eval_policy.py` via
  `resolve_eval_seeds()`, and `expert_check = expert_check and (seed_list is None)` correctly skips
  the live expert gate for pre-verified seeds. The frozen 3,158-seed set that previously fed nothing
  is live on the new baseline — `domain_seed_conventions.md`'s "nothing reads `eval_seeds/`" is
  superseded.

Effort: ~half a day of guardrails.

---

## Part 6b — OPTIONAL, DEFERRED: close out the `phase4_approach_mode` `off` cell

**Deferred by decision, 2026-08-11 — not a prerequisite for any other part.** Recorded here so the
trade-off is not rediscovered later.

The ablation is parked mid-flight. Record counts across all 9 runs: `direct` plentiful (several
cells of 15), `seed` 13 episodes, **`off` 5 episodes** — and the single most recent thing that ran
in this repo (`20260731-151143`, 11 days ago) is an `off`-only run that produced 3 records and
stopped. `domain_expert_baseline.md` states the strategic reframe *"depends on the `off` cell
existing"*, and estimates a full cell at ~70 s/rollout, roughly 18–30 minutes of GPU.

**The pooling consequence, which is the only time-sensitive part.** Every existing `direct` and
`seed` record was collected under **centre-to-centre** occluder spacing. Dev's `cf966f7`
reinterprets `occluder_offset` as a true edge-to-edge gap, so after Part 1 the same nominal offset
describes physically different scenes and the new records **cannot be pooled with what is on disk**.

- Run `off` *before* the merge → ~20–30 min, and the ablation completes under the semantics every
  existing record shares.
- Run it *after* → the 5 existing `off` records are stranded, and finishing the comparison means
  re-running `direct` and `seed` too.

Deferring is a legitimate call — it only costs something if the ablation is later resumed. If it is,
resume it *before* merging, not after.

---

## Part 7 — Keep it from recurring

- Merge `origin/dev` weekly. The whole problem was a six-week gap *plus* stale local refs.
- **Shared-file edits go to dev as PRs; the research branch holds exclusive files only.** This is the
  structural fix, and it is what makes the exclusive layer's 24,834 lines cost zero.
- Rewrite `agent-memory/status_current.md` — it names branch `codex/bench-script-refactor` and
  describes unstaged changes that no longer exist.

---

## Feasibility

**Verdict: feasible in 1–2 weeks, with one structural caveat.**

Rough sizing: Part 0–1 minutes, Part 2 half a day, Part 3 half a day, Part 4 5–8 days, Part 5 2 days,
Part 6 half a day. **Total ≈ 9–12 working days**, which fits the budget only if nothing goes badly
wrong and if the GPU checkpoints don't stall.

**The caveat worth deciding on up front:** the *cheap future merges* goal is delivered almost
entirely by Parts 1–3 plus §4a — roughly **2–3 days**. The remaining 6–9 days buys legibility and
dead-weight removal, which are real goals you chose, but they do not further reduce merge cost. If
the schedule tightens, that is the clean place to stop.

### Genuine risks, in order

1. **No safety net for a 17,000-LOC restructure.** The verification bar today is `--help` plus 13
   manually-run checks. This is the top risk and Part 5's pytest adoption is the mitigation — which
   is why it must come first rather than last.
2. **The duck-typed `env.<method>()` calls from `lib/`.** Invisible to a by-name dead-code scan.
   This already cost a full day of GPU runs measuring nothing. Any rename during consolidation can
   reproduce it exactly.
3. **Semantic landmines already identified.** `update_world()`'s default flips meaning under dev's
   version (`= None` → `planner_exclude_obstacles` → `enable_collision_metrics`, which
   `bench_demo_office_clean.yml` sets true), so a bare call goes from *include all clutter* to
   *exclude all clutter* — silently, with no error, just rollouts knocking clutter over again. Ring
   code must pass `exclude_obstacles=False` explicitly. And occluder spacing is now edge-to-edge, so
   `--offsets 0.1-0.25` describes physically different scenes than the ones every existing expert
   episode was collected under.
4. **GPU work serialises on you.** Per `feedback_role_boundary.md` I write scripts and you run them;
   each cycle is ~15 min plus a context switch. Part 4d's instrument-then-prune step needs a real
   run mid-stream, so it should be scheduled early rather than discovered late.
5. **The curobo patch.** `curobo_seed_traj.patch` is applied by hand and must stay in sync with
   `lib/seed_from_clearance.py`. It is the most fragile external dependency in the tree.
6. **`make_occluder_task()` resolves its base class at call time** from a dev-owned task file, so a
   dev change can alter the research task with no diff on any file I own.

### What is *not* a risk, despite appearances

- The merge itself: 9 conflicts, 4 of them "take one side."
- The 84-file bench_envs cleanup: mechanical, scriptable, disjoint from dev's edits.
- The uv/packaging migration: already shared history on both sides.
- The Makefile: zero dangling targets; better maintained than the rest of the tree.

---

## Verification

No step is complete until these pass. Run from `customized_robotwin/` with `source set_env.sh` and
`export ROBOTWIN_BENCH_TASK=bench`.

- `--help` on every top-level `bench_script/` command — exercises the full import chain without a GPU.
- `python -m checks.<module>` for the CPU-safe checks; once Part 5 lands, `pytest` for all of them.
- `checks/test_ring_config.py` after any `occluder_ring.py` change — it asserts the formation is
  byte-identical per (seed, offset-spec), which is what guarantees the measured scene equals the
  rolled-out scene. **Its baselines will change** when the edge-to-edge geometry lands; that diff is
  the thing to review, not to suppress.
- `checks/test_lib_env_api.py` — the AST guard on duck-typed env calls.
- `grep -rn "^from \(analyze_\|clearance_\|reachability_\|visualize_\|seed_from\)" lib/` → must be
  empty. The layering rule.
- Count the doors into `envs/`: after §4a, only `lib/scene_build.py` may import from `envs.`.
- `checks/smoke_test_seed_2a.py` and `diag_kitchen_curobo.py` are **GPU-only** — yours to run, and
  not to be claimed as passing otherwise.
- End-to-end: one `analyze_occluder_visibility.py` run and one `vla_rollout.py` run, confirming
  `records.jsonl` and `rollout_seed_stats` still populate and the `[SEED FIRING]` block still reports.
