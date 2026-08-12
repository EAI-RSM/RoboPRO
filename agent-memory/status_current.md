---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Last rewritten 2026-08-12.

## Branch and worktree

Branch: `peng-dev-new`, tracking `origin/peng-dev-new`. S1 through S4 are complete. S4's runtime
implementation is split into four reviewable commits before the final completion record:

- `14be2e3` — expose the exact collision-world entry count after `robot.update_world` succeeds;
- `9b60bbd` — explicitly include clutter on the ring path and record default/full counts per rollout;
- `ff82142` — guard the gitignored vendored CuRobo patch with patch-specific sentinels, including in
  `run_approach_mode_ab.sh`;
- `acd4976` — add the built-scene geometry validator and record the staged GPU handoff.

The source line remains backed up at `origin/backup/peng-training-branch`; tag
`pre-rehaul-2026-08-11` remains on `e3a09ce`. The temporary comparison worktree remains at
`/tmp/robopro-s4.yUGVZp/peng-training`; it is disposable after review.

## S4 completion evidence

CPU gates from `customized_robotwin/script/bench_script/`:

- `../../../.venv/bin/python -m pytest -q`: **50 passed, 1 skipped** (51 items; only
  `checks/smoke_test_seed_2a.py::test_seed_2a_smoke` is skipped; one existing Sapien warning).
- `../../../.venv/bin/python -m checks.test_curobo_patch`: pass.
- `checks/test_vla_office_smoke.py`: **8 passed**, including excluded=2 versus full=3 world-entry
  instrumentation on a fake scene.

User-run GPU/runtime gates:

- Built geometry at `s4_make_it_run/geometry/20260812-121536/`: requested gap 0.100 m, measured
  collision-hull gap 0.109905 m, error 0.009905 m within the 0.010 m gate, with 0.125012 m z overlap.
  Raw actor segmentation resolved target id 75 and 207 target pixels. The initialized-scene PNG
  visibly overlays the intended bottle.
- Ring smoke at `occluder_visibility/s4_ring/20260812-135100/`: seed 0, one 0.10 m-gap occluder,
  clutter density 8, direct/direct. The rollout succeeded in 114.0 s, wrote records/video/HDF5,
  retained the expected empty direct-mode `rollout_seed_stats`, and recorded zero physics
  collisions. Treatment delivery is explicit: CuRobo world **11 full > 2 default**.
- Unconfounded standard/direct comparison on matched seeds 0–4: `peng-dev-new` and
  `peng-training-branch` were identical per seed. Seeds 0,1,3 succeeded; seed 2 failed at `grasp`
  after the same six unreachable rotations; seed 4 failed at `placement:pre_place_descent` with the
  same `MotionGenStatus.FINETUNE_TRAJOPT_FAIL`. Both branches were **3/5**. This rules out gross port
  divergence for the smoke set; n=5 cannot detect modest success-rate changes.

No runtime reconciliation commit was needed. The audit also established that visibility masks use
raw actor segmentation (RGB is only the overlay), no second bare ring `update_world` exists, and
aggregate collision metrics initialize without a stream double-start. Optional per-contact JSONL
streams have no callers, but existing records consume aggregate `get_collision_metrics()`.

## Preserved research state

The source rollout campaign remains complete at 3000 episodes with all videos. Metric
post-processing remains partial at 620/3000 under
`results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/` and is not
resumable. Route audit remains 50 episodes / 200 figures. No result directory was deleted or
rewritten; untracked results total about 13 GB.

## Next action

Execute S5's approved 84-file mechanical cleanup plan. S5 is independent of S6–S12; no later
rehaul section has started.
