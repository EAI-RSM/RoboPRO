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
- Prune §2 is commit `7f759f4`: the invalidated task-metric implementation is cold source under
  `research_archive/bench_script_task_metric_2026/`; its six obsolete tests and three plans are
  deleted, and the optional VLA metric post-process hook is detached.
- Prune §3 is committed: seven surviving checks now live under
  `customized_robotwin/script/bench_script/checks/` and run as modules.
- Concurrent changes to `agent-memory/MEMORY.md` and the untracked
  `agent-memory/domain_seed_conventions.md` are unrelated and must remain unstaged.

## Preserved research state

- The source rollout campaign remains complete at 3000 episodes with all source videos.
- Metric post-processing remains partial at **620/3000**. All 620 episode JSONs remain under
  `scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`.
- The independent route audit remains complete at 50 episodes and 200 figures.
- No result directory was deleted or rewritten, and no metric process is active.

## Verification state

Final CPU gates passed with the RoboTwin interpreter:

- all 59 active/archive Python files parse;
- all eight CPU-safe top-level commands pass `--help`;
- all six CPU-safe checks pass as `python -m checks.<module>`;
- the lib/task-to-top-level dependency count and active retired-module reference count are zero;
- the top level contains exactly ten intended Python commands/bootstrap files and no checks;
- the cold archive contains exactly ten source/spec files plus README and no `__init__.py`;
- preservation receipts still report 620 metric records and 200 route figures;
- `git diff --check` passes and `graphify update .` completed.

Measured active-tree size is **13,033 Python lines** including checks and **11,715** excluding
checks, down from the 20,152-line baseline. The cold archive holds 4,832 tracked lines and is not a
repository-wide deletion.

`checks.smoke_test_seed_2a` and `diag_kitchen_curobo.py` remain unrun because they require CUDA;
do not claim those GPU gates passed.

## Next action

The approved pruning plan is complete. No further deletion or result processing is pending;
preserve the two unrelated worktree changes noted above. The only unverified cleanup gates are the
two explicitly CUDA-only commands.
