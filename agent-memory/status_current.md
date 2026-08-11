---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Last rewritten 2026-08-11.

## Branch and worktree

Branch: `peng-training-branch` (the previous entry naming `codex/bench-script-refactor` was stale;
that branch is 1 commit behind this one and the same line of work).

- **S1 IS DONE (2026-08-11). All local-only work is backed up off-machine.** Before it,
  `git rev-list --all --not --remotes` was **71**; it is now **0**, verified against a fresh bare
  clone rather than local tracking state. Five archive refs on `origin`:
  `backup/peng-training-branch` (`0d7df5c`), `backup/occluder-testbench` (`91b2d26`),
  `backup/pre-merge-35-occluder` (`35801cb`),
  `backup/reachability-motion-validated-placement` (`585d307`),
  `backup/visibility-constraint-experiments` (`41ee00f`), plus tag `pre-rehaul-2026-08-11` on
  `e3a09ce`. The `backup/` prefix marks these as archives, not review requests — no PR was opened.
- Scope note: pushing `peng-training-branch` alone would have saved only 64 of the 71 commits. Four
  other branches held unique work, including `585d307`, the vanilla `--plan-algo` baseline that
  [[domain_expert_baseline]] calls "recoverable from that commit."
- S2 is committed in four reviewable boundaries: dependency declaration/lock, harness bootstrap,
  main-only wrappers plus GPU marker, and this baseline/plan record. Unrelated edits to
  `MEMORY.md`, `feedback_role_boundary.md`, and `repo_env_and_git.md` remain unstaged and are not
  part of S2.
- Against `origin/dev` (`64840ce`): **82 ahead, 60 behind**, merge base `aabeff4` (2026-06-25).
  See [[repo_env_and_git]] for the full divergence numbers and the stale-local-`dev` trap.

## The rehaul plan (2026-08-11)

`plans/REHAUL_PLAN.md` scopes a fresh `peng-dev-new` off **`origin/dev`**, with the research layer
ported and consolidated rather than rewritten. Parts 0–3 (safety net, branch, ~150 lines of shared
carry-over, the scripted 84-file cleanup) are ~2 days; Part 4 (research layer) is the bulk.

Decisions taken this session, so they are not relitigated:

- Goals: cheap future merges, clean architecture, dead-weight removal. Budget 1–2 weeks. Data
  continuity **partial** — results stay readable, not re-runnable. Fork policy: track dev weekly.
- **Task-metric line stays LIVE** — the correlation question is the research programme. Reach/
  clearance tools port. `task/` is ported, instrumented, then pruned by measurement.
- **`swept_volume_3d.py` is NOT ported** (412 L; last run 2026-07-15, the one real staleness
  outlier). Its 285 MB of results stay on disk.
- **Nothing under `scripts/` is dropped.** An earlier draft proposed deleting `scripts/upload/`,
  `scripts/slurm/` and the five June OOD scripts (~1,142 L); all three **exist on `origin/dev`
  unmodified by us**, so deleting them would be a shared-file edit — the exact merge tax the rehaul
  exists to remove.
- `phase4_approach_mode` `off` cell: **deferred, optional** (Part 6b). See [[domain_expert_baseline]]
  for the pooling deadline that makes deferral non-free.
- Entry-point collapse (14 → ~4 subcommands) is **flagged but NOT in scope** — awaiting an explicit
  yes, because it renames every command and breaks Makefile targets and `run_approach_mode_ab.sh`.

## Preserved research state

- Source rollout campaign complete at 3000 episodes with all source videos.
- Metric post-processing partial at **620/3000**, under
  `results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`.
  Not resumable (stored code hash matches no commit), and moot — any fingertip-clearance remedy
  recomputes all of them.
- Route audit complete at 50 episodes / 200 figures. No result directory deleted or rewritten.
- `results/` totals ~13 GB, untracked and unregenerable. Rollouts are non-deterministic by design.

## Verification state

**S2 CPU baseline, 2026-08-11:** from `customized_robotwin/script/bench_script/`,
`../../../.venv/bin/python -m pytest -q` collects **50 items across 12 runnable modules** and reports
**49 passed, 1 skipped**, with one existing Sapien `pkg_resources` deprecation warning. The only
GPU-marked item is `checks/smoke_test_seed_2a.py::test_seed_2a_smoke`; it was collected and skipped,
not executed. No existing test function needed a GPU marker.

Legacy entry points still pass for `checks.test_ring_config`, `checks.test_obstacle_set`, and the
newly collected CPU AST guard `checks.test_lib_env_api`. The earlier prune gates also remain: all
active/archive Python parses; CPU-safe top-level commands pass `--help`; the lib/task→top-level
dependency count is zero. `checks.smoke_test_seed_2a` and `diag_kitchen_curobo.py` remain **unrun
(CUDA-only)** — do not claim those gates passed.

`uv.lock` added exactly five package blocks (`pytest`, `iniconfig`, `pluggy`, `pygments`, `tomli`).
`uv sync --locked` exposed an existing packaging gap: it prunes the editable local CuRobo install
before `bootstrap_uv.sh` reinstalls it. `pytest.ini` therefore adds vendored `envs/curobo/src` to
pytest's import path; this keeps CPU collection valid without pretending the GPU install was tested.

## Next action

S1 and S2 are done. Before S3 creates `peng-dev-new`, cherry-pick the atomic pytest dependency
commit (`pyproject.toml` + `uv.lock`) onto an upstream PR branch targeting `8-research-branch`; S3
must inherit both files together because `bootstrap_uv.sh` uses `uv sync --locked`. Then execute S3
from `peng-training-branch` and compare its port against the 49-pass/1-skip baseline above.

`plans/REHAUL_PLAN.md` now opens with a **Work breakdown** of twelve executable sections, S1–S12,
each self-contained enough to prompt an agent with "write a technical plan for S6" or "execute S8"
without reading the rest of the document. **Numbering is stable; do not renumber.** Each section
carries in-scope / out-of-scope / a done-when gate / verification commands, and the ones that must
not change behaviour are marked NO-BEHAVIOUR-CHANGE.

Order: ~~S1~~ **done**, ~~S2~~ **done**, then S3/S4 (branch, port, make it run). S5 is
parallelisable. S10 needs a user GPU run mid-section — schedule it early.

The completed S2 technical plan and corrections are in
`customized_robotwin/script/bench_script/plans/S2_PYTEST_HARNESS_PLAN.md`. The investigation found
10 of 12 modules already pytest-shaped; `smoke_test_seed_2a.py` and `test_lib_env_api.py` needed thin
wrappers. The similarly named local helpers are name collisions, not duplicate fixtures, so none
were merged.

One open question: whether to add §4e, the entry-point collapse (14 → ~4 subcommands). It is
deliberately unsectioned because it renames every command and breaks Makefile targets,
`run_approach_mode_ab.sh:220`, and command strings recorded across `agent-memory/` and `plans/`.
The `checks/` fixture consolidation is resolved — it is folded into S2.
