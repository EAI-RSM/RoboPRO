---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Everything else in `agent-memory/`
holds durable reasoning and gotchas. Last rewritten 2026-08-10.

## Branch and worktree

Branch: `codex/bench-script-refactor`.

The cleanup plan in
`customized_robotwin/script/bench_script/plans/CLEANUP_PLAN.md` has been executed through Phase 4.
The numbered cleanup commits before the final Phase-4 commit are:

- `05532eb` — Phase 3 repository hygiene.
- `aa6ba96` through `34165ef` — Phase 1 benchmark-environment cleanup.
- `0fb4253` through `14bc0cf` — Phase 2 benchmark-script cleanup.
- `2380734` — Phase 2.5 imported-module moves.
- `c7c09ed` — Phase 2.6 plan-document moves.

Two concurrent changes are not part of cleanup and were deliberately left unstaged: the modified
`agent-memory/MEMORY.md` and untracked `agent-memory/domain_seed_conventions.md`.

## Cleanup result

- Phase 1 removed 371 dead imports, consolidated leaf `setup_demo` dispatch into the base class,
  removed the approved inert helpers and stale office metadata comments, and corrected collision-key
  writers while preserving typo compatibility. There are 23 `setup_demo` definitions: 22 genuine
  leaf variants plus `Bench_base_task.setup_demo`.
- Phase 2 removed 143 dead imports and consolidated hashing, atomic figure writes, reachability
  helpers, and embodiment-config loading without changing algorithms.
- Imported-only modules now live under `bench_script/lib/`; internal execution/design documents
  live under `bench_script/plans/`. Both `lib/` and `task/` have zero imports from top-level CLI
  scripts.
- Phase 4 moved `metric_viz.py` and `seed_from_clearance.py` under `lib/`, repaired every importer,
  changed seed selftest/help usage to `python -m lib.seed_from_clearance`, and updated the
  visualization source-hash path. The new source hash intentionally differs from the completed v4
  audit's recorded hash; that audit remains an immutable artifact of the old source version.
- The cleanup plan's stale live-metadata count was corrected from 17 to the verified 41 producers.
- Section 5 was not executed. None of its orphan-file candidates were deleted.
- Final Python line count is 36,838 versus the 37,906 baseline: 1,068 lines removed.
- `graphify update .` refreshed the project graph after the source moves.

## Preserved research data

The 3000-rollout source campaign is complete: 1000 episodes each at d6/d10/d15, with 554 hard
successes and all source videos present.

Integrated metric post-processing remains intentionally partial at **620/3000** under
`scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`.
All 620 episode JSON records remain on disk and `processing_complete` remains false. The user waived
in-place resumption; a future post-process may restart from episode 1.

The independent `metric_route_visuals_v4` audit remains complete at 50/50 episodes and 200 figures,
with `processing_complete: true`.

## Verification state

Passed after Phase 4:

- compilation of `benchmark/bench_envs` and all of `bench_script`;
- zero dead imports in both cleanup roots;
- 12 CPU regression scripts, including the office smoke test;
- 16 CPU-safe CLI import/help checks;
- moved-module selftests and the expected visualization source-hash change;
- all Phase 2.5/2.6/4 path, document-reference, and dependency gates;
- preservation receipts for 620 partial metric records and the 50-episode/200-figure route audit;
- `git diff --check` and the 36,838-line final receipt.

GPU gap: CUDA is unavailable in this environment. The real-scene before/after gate, the 80-task
import/construct gate, and `diag_kitchen_curobo.py --help` remain unrun because they initialize
cuRobo/CUDA. Static dispatch checks and the CPU suite passed, but they do not replace that GPU gate.

## Next action

No cleanup phase remains authorized. If a CUDA environment becomes available, run the documented
Phase-1 GPU gates as an optional verification follow-up. Otherwise, future metric post-processing
may start a new run from episode 1 while preserving the existing 620-record partial run.
