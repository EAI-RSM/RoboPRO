---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Last rewritten 2026-08-10.

## Branch and worktree

Branch: `codex/bench-script-refactor`.

- The corrected second-pass plan is commit `20b7f80` at
  `customized_robotwin/script/bench_script/plans/PRUNE_PLAN.md`.
- Prune §1 is commit `3dc788a`: four confirmed orphan entry points were removed.
- Prune §2 is implemented and verified but not yet committed: the invalidated task-metric slice is
  cold source under `research_archive/bench_script_task_metric_2026/`, its six tests and three
  execution plans are deleted, and the optional VLA post-process hook is detached.
- Concurrent changes to `agent-memory/MEMORY.md` and the untracked
  `agent-memory/domain_seed_conventions.md` are unrelated and must remain unstaged.

## Preserved research state

- The source rollout campaign remains complete at 3000 episodes with all source videos.
- Metric post-processing remains partial at **620/3000**. All 620 episode JSONs remain under
  `scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`.
- The independent route audit remains complete at 50 episodes and 200 figures.
- No result directory has been deleted or rewritten, and no metric process is active.

## Verification state

Prune §2 gates passed with the RoboTwin interpreter: all 58 active/archive Python files parsed,
`test_vla_office_smoke.py` passed including focused live task-role/waypoint coverage,
`test_vla_reporting.py` passed, the lib/task-to-top-level dependency count is zero, the archive has
exactly ten source/spec files plus README, `git diff --check` passed, and the preservation receipt
still reports 620 metric records and 200 route figures. Active `bench_script` currently measures
13,032 Python lines before relocating its seven surviving checks.

The CUDA-only scene/import gates from the first cleanup and `smoke_test_seed_2a.py` remain
unavailable in this environment.

## Next action

Commit prune §2 independently without staging concurrent memory changes, then execute §3: move the
seven surviving checks under `bench_script/checks/`, repair file-relative roots, run all six
CPU-safe module checks, record final survivor and line-count receipts, update graphify, and commit.
