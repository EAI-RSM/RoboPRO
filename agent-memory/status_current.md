---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Last rewritten 2026-08-10.

## Branch and worktree

Branch: `codex/bench-script-refactor`.

- The first cleanup pass is complete through `d7d001a`.
- The corrected second-pass pruning plan is `20b7f80` and lives at
  `customized_robotwin/script/bench_script/plans/PRUNE_PLAN.md`.
- Prune §1 is complete in the current staged phase: the four confirmed orphan entry points were
  removed and the environment note was repaired.
- Concurrent changes to `agent-memory/MEMORY.md` and the untracked
  `agent-memory/domain_seed_conventions.md` are unrelated and must remain unstaged.

## Preserved research state

- The source rollout campaign remains complete at 3000 episodes with all source videos.
- Metric post-processing remains partial at **620/3000**. All 620 episode JSONs remain under
  `scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`.
- The independent route audit remains complete at 50 episodes and 200 figures.
- No result directory has been deleted or rewritten.

## Verification state

Pre-prune receipts passed: 20,152 Python lines, 34 top-level Python files, 620 partial metric
records, 200 completed route figures, and no active metric process.

The CUDA-only scene/import gates from the first cleanup remain unavailable in this environment.

## Next action

Verify and commit prune §1 independently, then execute §2: cold-archive the retired metric
implementation outside `bench_script`, delete its obsolete tests/plans, detach the optional VLA
post-process hook, and preserve focused coverage for live task-role/waypoint APIs.
