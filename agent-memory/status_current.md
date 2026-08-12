---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Last rewritten 2026-08-12.

## Branch and worktree

Branch: `peng-dev-new`, tracking `origin/peng-dev-new`. S1, S2, and S3 are complete. S4's CPU
implementation is complete and committed locally; the branch is four commits ahead of its remote:

- `14be2e3` — expose the exact collision-world entry count after `robot.update_world` succeeds;
- `9b60bbd` — explicitly include clutter on the ring path and record default/full counts per rollout;
- `ff82142` — guard the gitignored vendored CuRobo patch with patch-specific sentinels, including in
  `run_approach_mode_ab.sh`;
- current S4 handoff commit — add the self-contained built-scene geometry validation and record the
  pending GPU gate.

The user's approved planning edits remain deliberately uncommitted and were not swept into the S4
implementation commits: modified `plans/REHAUL_PLAN.md` and untracked
`plans/S4_MAKE_IT_RUN_PLAN.md`. The source line remains backed up at
`origin/backup/peng-training-branch`; tag `pre-rehaul-2026-08-11` remains on `e3a09ce`.

## S4 CPU result

- `OccluderTask.play_once` now calls `update_world(exclude_obstacles=False)`. It snapshots both the
  setup-default world count and the explicit-full count, prints them, and
  `analyze_occluder_visibility.py` writes both into each rollout record. The fields are cleared
  before each reused-env rollout so an early failure cannot inherit the prior episode.
- The exact diagnostic is `sum(len(entries) for entries in collision_dict.values())`; it is written
  only after the robot accepts the update. Counting `collision_list` would be wrong because one
  directory entry can expand to multiple CuRobo meshes.
- `checks.test_curobo_patch` checks four RoboPRO-specific sentinels. A generic `seed_traj` grep is a
  false guard because pristine CuRobo already has unrelated locals by that name. The working repair
  command is printed with an explicit repository-root `cd`.
- The Step 3 audit found no other bare ring-path `update_world`, no collision-stream double start,
  and no missing aggregate-metric initialization. `start_metric_streams` has no callers, but those
  JSONL streams are optional and existing records consume aggregate `get_collision_metrics()`.
- The S4 plan's RGB premise was wrong: visibility masks use
  `Base_Task.measure_target_visibility -> get_segmentation_raw(level="actor")`; RGB is only the
  overlay. The dedicated raw path preserves actor IDs as `int32`, so no CPU-side compatibility
  defect was found. The GPU scene check must still confirm a real nonempty target segmentation.

Final CPU gates from `customized_robotwin/script/bench_script/`:

- `../../../.venv/bin/python -m pytest -q`: **50 passed, 1 skipped** (51 items; only the GPU smoke
  is skipped; one existing Sapien warning).
- `../../../.venv/bin/python -m checks.test_curobo_patch`: pass.
- `checks/test_vla_office_smoke.py`: **8 passed**, including excluded=2 versus full=3 world-entry
  instrumentation on a fake scene.
- `validate_s4_scene_geometry.py --help` and `py_compile`: pass. The user-run live scene measured a
  requested 0.100 m gap as 0.109905 m (error 0.009905 m, within the 0.010 m gate), with 0.125012 m
  target/occluder z overlap and a nonempty 207-pixel raw target-segmentation mask.
  The first user attempt failed before scene construction because the standalone validator omitted
  `benchmark/` from `sys.path` and therefore could not resolve `bench_envs.office.put_mouse_on_pad`;
  the handoff script now installs that package root and checks resolution without initializing CUDA.

## Preserved research state

The source rollout campaign remains complete at 3000 episodes with all videos. Metric
post-processing remains partial at 620/3000 under
`results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/` and is not
resumable. Route audit remains 50 episodes / 200 figures. No result directory was deleted or
rewritten; untracked results total about 13 GB.

## Unverified and next action

S4 is **not complete**. Two of three GPU gates pass:

1. built-scene geometry and raw segmentation pass at
   `s4_make_it_run/geometry/20260812-121536/`;
2. `peng-dev-new`'s five matched standard seeds completed 3/5, with failures at `grasp` and
   `placement:pre_place_descent`; every rollout recorded full CuRobo world 10 > default 1;
3. the cluttered ring smoke completed successfully at
   `occluder_visibility/s4_ring/20260812-135100/`, wrote records/video/HDF5, recorded empty
   `rollout_seed_stats` correctly for direct mode, had zero physics collisions, and proved treatment
   delivery with full CuRobo world 11 > default 2.

The `peng-training-branch` comparison cell did **not** run: its temporary worktree lacked the
gitignored `benchmark/assets`, so import failed before scene construction. The worktree survives at
`/tmp/robopro-s4.yUGVZp/peng-training`; its shared CuRobo and benchmark-assets links are now ready.
Only that five-seed standard/direct cell must be rerun. Then compare its n/success/failure families
against `peng-dev-new`, mark S4 complete, commit, refresh graphify, and push. S5 remains
parallelisable but has not started.
