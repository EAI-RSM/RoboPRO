# S3 — Branch and port

Technical plan for section **S3** of [REHAUL_PLAN.md](REHAUL_PLAN.md).
Starts on `peng-training-branch`, creates and switches to `peng-dev-new`.
~1 day. **SHARED files — highest care in the whole rehaul.**

**Status: complete (2026-08-11).** Implemented on `peng-dev-new` from `origin/dev@64840ce`.
The Step 4 and final gates both report **49 passed, 1 skipped**; every top-level `--help`, the
legacy ring entry point, and the layering check pass.

Execution notes: the retained exclusive count was the corrected 106; `swept_volume_3d.py` was not
ported. Dev already contained the Makefile's `OCC_DISTANCE_CM` rename, so no Makefile diff was
needed; the refactored analyzer gained a centimetre-to-metre compatibility path for that existing
entry point. The edge-to-edge baseline was deliberately reviewed in `test_ring_config`.

---

## Context

S3 is **the move**: lift the research layer off `peng-training-branch` and stand it up on top of six
weeks of the team's work on `origin/dev`.

The bar is deliberately low. S3 succeeds when the tree **imports** and the CPU test suite passes at
S2's baseline. Making it genuinely *run* — reconciling against dev's changed `envs/` behaviour, the
`update_world()` default flip, a real rollout — is **S4**. Do not pull S4's work forward; a section
that tries to do both produces a diff nobody can review.

Why it is safe to attempt: of our 208 files changed since the fork, 106 exist only on our branch and
cannot conflict, and the genuine shared carry-over is ~150 lines. Most of what looks like divergence
is already upstream — `8-research-branch` merged as PR #57, the visibility work re-landed as PR #56.

## What the investigation found

**1. The port is not "bulk-copy 106 files."** Five files inside `customized_robotwin/script/bench_script/`
are **shared** with dev and each needs an explicit decision — REHAUL_PLAN's S3 text omits them:

| File | Ours | Dev's | Note |
|---|---|---|---|
| `analyze_occluder_visibility.py` | 447 | **2,736** | add/add. Dev's is the pre-refactor monolith and holds the edge-to-edge fix |
| `reachability_map.py` | 351 | 416 | add/add |
| `visualize_task_scene.py` | — | — | shared, both sides edited |
| `diag_kitchen_curobo.py` | — | — | shared |
| `setup_paths.py` | — | — | shared and currently byte-identical; S2's `conftest.py` imports it |

**2. The edge-to-edge fix exists on our branch in no form at all.** `cf966f7` added +220/−22 *inside*
dev's `analyze_occluder_visibility.py`. Our 447-line file is small because the 2026-07-29 refactor
shed 3,081 lines into `task/` and `lib/`. So taking "ours" silently discards a scene-geometry
correction the team has already declared correct; taking "theirs" discards the refactor and orphans
`task/` and `lib/`. **Neither side is takeable whole.**

**3. `origin/dev` has no pytest**, so the 49-test gate cannot run on a fresh branch without it.

## Decisions taken

- **Port the edge-to-edge geometry inside S3**, as its own isolated step and commit, so
  `peng-dev-new` is never in a state where it builds scenes on semantics dev has declared wrong.
- **Cherry-pick `18abd2e`** (`s2-pytest-dev-dependency`, `pyproject.toml` + `uv.lock`, already on
  origin) rather than waiting on a PR. It becomes a no-op when the change lands upstream.
  Both files must travel together — `bootstrap_uv.sh` uses `uv sync --locked`, so a pyproject-only
  state fails.

---

## Step 1 — Create the branch and restore the test runner

Before switching, commit this approved plan plus its linked memory/rehaul updates on
`peng-training-branch`. That planning checkpoint is not an S3 implementation commit, but it must be
in the source tree so the exclusive-file port can carry it.

```bash
git fetch origin
git switch -c peng-dev-new origin/dev          # 64840ce, NOT local dev (2 months stale at 4b57b67)
git cherry-pick 18abd2e                        # pytest dev-group + lock, atomically
git push -u origin peng-dev-new                # tracking ref from day one, per the fork policy
```

The tracking ref is deliberate: 64 unpushed commits is how the original mess arose.

> **→ COMMIT 1** — the cherry-pick is itself the commit. Nothing else in it.

## Step 2 — Bulk-copy the 106 retained exclusive files

These exist on no other branch, so no conflict is possible. Once this plan is committed the raw
source-only set is 107 files; filter out the deliberately retired `swept_volume_3d.py` before the
copy, leaving 106 retained files:

```bash
comm -23 <(git ls-tree -r --name-only peng-training-branch | sort) \
         <(git ls-tree -r --name-only origin/dev | sort) \
  | grep -vx 'customized_robotwin/script/bench_script/swept_volume_3d.py' \
  > /tmp/exclusive.txt
wc -l /tmp/exclusive.txt          # expect 106
xargs -a /tmp/exclusive.txt git checkout peng-training-branch --
```

Composition: 72 under `customized_robotwin/` (the `bench_script` tree — root scripts, `lib/`,
`task/`, `checks/`, `plans/`, plus S2's `conftest.py` and `pytest.ini`), 25 `agent-memory/`, 4
`research_archive/`, 2 `scripts/validation/`, 1 `benchmark/`, plus `CLAUDE.md` and `AGENTS.md`.

**`swept_volume_3d.py` is NOT ported** — the generated path list excludes it before any copy.

> **→ COMMIT 2 — `Port exclusive research layer onto dev`**
> Exclusive files only. This is the largest commit; keeping it pure is what makes the following
> steps reviewable.

**Equivalence gate for this step** (see Test plan): the exclusive tree must be byte-identical to the
old branch.

## Step 3 — Resolve the five shared `bench_script/` files

Not a bulk copy. Per file:

- **`analyze_occluder_visibility.py` → take OURS** (447 L). Dev's monolith would orphan `task/` and
  `lib/`. Its edge-to-edge fix is recovered in Step 5, not here.
- **`reachability_map.py` → take OURS** (351 L). Follows the same resolution.
- **`visualize_task_scene.py`, `diag_kitchen_curobo.py` → inspect and union.** Run
  `git diff origin/dev peng-training-branch -- <path>` and take dev's changes where they are
  additive. These are small; do not take ours wholesale.
- **`setup_paths.py` → keep dev's byte-identical copy.** S2's `conftest.py` imports
  `setup_paths()`; verify it still derives `ROBOTWIN_ROOT`/`BENCH_ROOT` from file location when
  unset, but do not manufacture a diff where none exists.

> **→ COMMIT 3 — `Resolve shared bench_script files`**

## Step 4 — Re-apply the shared carry-over (~150 lines)

Derive each hunk with `git diff origin/dev peng-training-branch -- <path>` and apply onto **dev's**
version. Do not copy whole files.

| Item | File |
|---|---|
| `Robot.build_planner` opt-out (~50 L, kwarg defaulting True, threaded through `reset`, `_init_planner`, `update_world*`, `*_plan_grippers`, `_linear_gripper_plan` fallback) | `customized_robotwin/envs/robot/robot.py` |
| `seed_traj` plumbing + `attempts` / `trajopt_attempts` / `seeded` result keys (~30 L) | `envs/robot/planner.py`, `robot.py` |
| `pi05_robopro_top_cam_jax` TrainConfig | `policy/pi05/src/openpi/training/config.py` |
| `eval.sh` checkpoint arg passing | `policy/pi05/eval.sh`, `eval_office_put_mouse.sh` |
| `put_mouse_on_pad: 600` | `task_config/_eval_step_limit.yml` |
| `include_collision` typo shim | `bench_envs/study/_study_base_task.py` |
| Two `.gitignore` lines | root |

**Keep dev's `_position_only_pose_cost_metric()` refactor** — we inline it in two places; dev's
factored version wins.

**Plus the five `_bench_base_task.py` hooks**, applied onto dev's version (ours is +123/−604, mostly
a *reversion* of dev's collision-metrics subsystem — do not carry the file):

1. `setup_demo(self, is_test=False, **kwargs)` base hook — **load-bearing**: S5's 84-file cleanup
   deletes each task's override *because the base provides it*.
2. `_write_eval_video_frame()` + the `select_eval_video_camera` import, and
   `benchmark/bench_envs/eval_video.py`. **Check first whether dev's `EVAL_VIDEO_CAMERAS`
   (`358acdd`, countertop-first) already covers this** — if so, prefer dev's and drop ours.
3. `allow_duplicate_clutter` retry-budget change in the clutter loop.
4. The `if not self.robot.build_planner:` TOPP bypass — **pairs with the `build_planner` row above**;
   one without the other leaves the flag inert.
5. The opt-in `step_hook` per-physics-step observer.

**Do NOT carry** (our copy regresses dev): `envs/_base_task.py`, `envs/camera/camera.py` (ours is
*behind* — dev moved segmentation to raw `uint16`), `envs/utils/pkl2hdf5.py`, `_bench_base_task.py`
wholesale, the four per-env base tasks, `policy/pi05/deploy_policy.py`, `Makefile`, `task_objects.yml`.
**Do not set `action_noise_var: 0.001`** — dev set it to 0 deliberately in `1f4a5f3`.

> **→ COMMIT 4 — `Re-apply research hooks onto dev base classes`**
> **Run the regression gate here, before Step 5.** This is the last point at which the suite should
> read exactly 49 passed / 1 skipped.

## Step 5 — Port the edge-to-edge occluder geometry

Lift the footprint geometry out of dev's `analyze_occluder_visibility.py` (`cf966f7`, ~8 pure
functions, ~127 lines, depending only on numpy, json, `BENCH_ROOT` and a local model-data cache) into
`lib/`, then rewire `lib/occluder_ring.py` so `occluder_offset` means the closest-surface gap between
target and occluder, yaw-aware, rather than centre-to-centre.

Also take dev's `Makefile` flag rename (`OCC_OFFSETS` → `OCC_DISTANCE_CM`) so the CLI matches the
semantics — but only that hunk; the Makefile is otherwise dev's.

**`checks/test_ring_config` formation baselines WILL change.** That is correct and expected — it
asserts byte-identical formations per (seed, offset-spec), and the spec now means something
different. **Review that diff; do not suppress it.** Re-baseline deliberately and record the change.

> **→ COMMIT 5 — `Port edge-to-edge occluder geometry`**
> Isolated on purpose: this is the only commit in S3 that changes scene semantics, and it must be
> reviewable on its own. Every other S3 commit is behaviour-preserving.

## Step 6 — Verify and record

Run the full test plan below. Write into `agent-memory/status_current.md`: the branch is now
`peng-dev-new`, the post-port pass/skip counts, the `test_ring_config` re-baseline, and which S3
decisions were taken. Mark S3 done in `REHAUL_PLAN.md` and note that its S3 text omitted the five
shared `bench_script/` files.

> **→ COMMIT 6 — `Record S3 completion`** then push to `origin/peng-dev-new`.

---

## Out of scope

- **Everything in S4.** The `update_world()` default flip, dev's raw-`uint16` segmentation, the
  collision-metric streaming reconciliation, and any GPU rollout belong there. S3 stops at "imports
  and CPU tests pass."
- The 84-file `bench_envs` cleanup — that is S5, and it depends on the `setup_demo` hook landing here.
- Any consolidation, deduplication or restructuring (S6–S12).
- Fixing pre-existing test failures. Record them; do not repair them inside the port.
- `swept_volume_3d.py` — not ported, by decision.

## Done when

`peng-dev-new` exists off `origin/dev` with a tracking ref; `--help` passes on every top-level
`bench_script/` command; the CPU suite reads **49 passed / 1 skipped at the end of Step 4**; the
exclusive tree is byte-identical to `peng-training-branch`; and after Step 5 the only test delta is
`test_ring_config`'s deliberately re-baselined formations.

## Test plan

**1. Regression gate — checked twice, and the two readings differ.**

- **After Step 4:** `../../../.venv/bin/python -m pytest -q` from `bench_script/` must read exactly
  **49 passed, 1 skipped** (50 items, 12 modules). Any deviation means the port broke something —
  stop and diagnose before Step 5.
- **After Step 5:** `test_ring_config` failures are *expected*; every other test must still pass.
  Re-baseline only `test_ring_config`, and only after reviewing the formation diff.

**2. Commands.**

```bash
cd customized_robotwin/script/bench_script
../../../.venv/bin/python -m pytest -q                 # 49 passed, 1 skipped
for f in *.py; do ../../../.venv/bin/python "$f" --help >/dev/null || echo "FAIL $f"; done
../../../.venv/bin/python -m checks.test_ring_config   # legacy path still works
git -C ../../.. log --oneline origin/dev..peng-dev-new # expect the 6 S3 commits
```

**3. Equivalence — the port must be faithful.** After Step 2, the exclusive layer must be
byte-identical to its source:

```bash
git diff peng-training-branch peng-dev-new -- $(cat /tmp/exclusive.txt | tr '\n' ' ')   # MUST be empty
```

An empty diff proves the 106 files transferred without silent modification. Re-run after Step 4;
only files touched by the carry-over should appear.

**4. What S3 CANNOT verify.**

- **That anything actually runs.** The CPU suite covers metric math, obstacle-set and ring
  determinism — not scene construction, planning, or rollout. A green S3 does not mean the expert
  works; that is S4's question and it needs a GPU.
- **That the edge-to-edge port is numerically right.** Step 5 changes geometry, and confirming the
  new spacing is physically correct requires building a scene. S3 can only show the code runs and
  the formations changed deliberately.
- **Anything dev changed in `envs/` that only bites at runtime** — the `update_world()` default flip
  in particular fails silently with no error, just clutter being knocked over. S4 owns it.

## Risks

| Risk | Mitigation |
|---|---|
| Taking dev's `analyze_occluder_visibility.py` orphans `task/` + `lib/` | Explicit "take ours" in Step 3; the geometry is recovered separately in Step 5 |
| Taking ours silently drops the edge-to-edge fix | Step 5 is mandatory and separately committed |
| Bulk copy silently modifies exclusive files | Byte-identical diff gate after Step 2 |
| `setup_paths.py` resolution breaks S2's `conftest.py` | Whichever version survives must still derive roots from file location; the test gate catches it immediately |
| `build_planner` carried without its TOPP bypass | Called out as a pair in Step 4 |
| `test_ring_config` failures after Step 5 get "fixed" rather than reviewed | Stated as expected and required to be reviewed; isolated in COMMIT 5 |
| pytest missing on the new branch | Cherry-pick in Step 1, both dependency files together |

## Rollback

`peng-dev-new` is a new branch; `peng-training-branch` is untouched throughout and remains tagged
`pre-rehaul-2026-08-11`. To abandon: `git switch peng-training-branch && git branch -D peng-dev-new`
(and `git push origin --delete peng-dev-new` if already pushed). No work is lost.
