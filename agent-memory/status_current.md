---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Last rewritten 2026-08-12.

## Branch and worktrees

Branch: `peng-dev-new`, tracking `origin/peng-dev-new`. S1 through S5 are complete. S5 is split into
six reviewable local commits: tool/import gate `7b86360`, T1 `1cc381f`, T4 `aa58e03`, T2 `12b55fc`,
T3 `ba83c6a`, and T5 `d7a8609`. The source line remains backed up at
`origin/backup/peng-training-branch`; tag `pre-rehaul-2026-08-11` remains on `e3a09ce`.

Two temporary worktrees remain and are disposable after review:

- S4 comparison: `/tmp/robopro-s4.yUGVZp/peng-training`.
- S5 upstream: `/tmp/robopro-s5.AkfeOd/cleanup`, branch `cleanup/bench-envs-mechanical`, clean and
  pushed. PR [#72](https://github.com/EAI-RSM/RoboPRO/pull/72) is open against `dev`; do not merge
  that branch into `peng-dev-new`.

## S5 completion evidence

- Current upstream baseline is `origin/dev@600089d`; its two post-plan commits did not touch
  `benchmark/bench_envs`.
- Shared-tree diff from the pre-transform local ref: **80 files, +34/−880**. It removes 58 exact
  boilerplate overrides, 13 dead defs, unused imports in 77 files, and 18 commented info blocks,
  then atomically renames the collision option in one reader and three writers.
- `python -m tools.bench_envs_cleanup --check`: `no changes`.
- Manifest-subtraction equivalence: 92 modules. Every one of 160 leaf `check_success`/`play_once`
  ASTs matches current dev.
- Permanent import gate: **95 imported, 0 failed** locally. Upstream imports **94/0** because dev
  does not contain the research-only `eval_video.py`.
- Full CPU baseline: **51 passed, 1 skipped** with one existing Sapien deprecation warning.
- All 20 top-level bench-script files accept `--help`; the old typo is absent from Python code.
- Upstream commit order is load-bearing hook `266091e`, then T1–T5: `e728bd5`, `ac1a118`,
  `5075952`, `3f3a988`, `c8bf785`.

No GPU or rollout was run for S5. Its proof is structural (AST, import, regression suite), not a
runtime SAPIEN equivalence result.

## Preserved research state

The source rollout campaign remains complete at 3000 episodes with all videos. Metric
post-processing remains partial at 620/3000 under
`results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/` and is not
resumable. Route audit remains 50 episodes / 200 figures. No result directory was deleted or
rewritten; untracked results total about 13 GB.

## Next action

Review the user-supplied `S6_S8_ROUGH_PLAN.md`, then execute the next approved detailed rehaul
subplan. No S6 or later implementation has started.
