---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Last rewritten 2026-08-11.

## Branch and worktree

Branch: `peng-dev-new`, created from freshly fetched `origin/dev@64840ce` and tracking
`origin/peng-dev-new`. The source line remains backed up at `origin/backup/peng-training-branch`;
tag `pre-rehaul-2026-08-11` remains on `e3a09ce`.

S1, S2, and S3 are complete. S3's review boundaries are:

- `5b806cc` — pytest dependency and lockfile, cherry-picked atomically;
- `73f167d` — 106 retained exclusive files, byte-identical to the source at the copy gate;
- `35ef06a` — five shared `bench_script/` files resolved explicitly;
- `f5271af` — research hooks reapplied onto dev-owned base classes;
- `cf4883b` — yaw-aware edge-to-edge occluder geometry and reviewed ring baseline;
- final S3 record/import-gate fixes, followed by push to the tracking branch.

`swept_volume_3d.py` was deliberately not ported. Generated `graphify-out/` trees are ignored and
were refreshed after the code changes.

## S3 decisions and verification

- Dev's `_position_only_pose_cost_metric`, collision metrics, countertop-first eval camera, raw
  `uint16` segmentation, and existing duplicate-clutter behavior were preserved.
- The research `build_planner`/TOPP bypass, `seed_traj` and effort reporting,
  `pi05_robopro_top_cam_jax`, eval checkpoint selection, step limit, collision-key typo shim,
  `setup_demo`, and opt-in `step_hook` were carried narrowly.
- Dev already had the `OCC_DISTANCE_CM` Makefile rename. The refactored analyzer now accepts that
  centimetre range and converts it to its metre-based gap spec; no unrelated Makefile hunk moved.
- `test_ring_config` deliberately changed from center-radius baselines to true footprint gaps. In
  the reviewed fixture, a requested 0.10 m gap now uses a 0.1713740509 m center distance and the
  measured rotated-rectangle distance remains 0.10 m.
- The final import sweep exposed two small port gaps: `subgoal_reachability_map.py` referenced a
  removed video helper, and `diag_kitchen_curobo.py` imported its CUDA environment before argparse.
  Both were repaired narrowly; no GPU diagnostic or rollout was executed.

Final CPU gates from `customized_robotwin/script/bench_script/`:

- `../../../.venv/bin/python -m pytest -q`: **49 passed, 1 skipped** (50 items; only
  `checks/smoke_test_seed_2a.py::test_seed_2a_smoke` is skipped; one existing Sapien warning).
- Every top-level `*.py --help`: pass, including the CUDA-only diagnostic without initializing CUDA.
- `python -m checks.test_ring_config`: pass through the legacy entry point.
- `lib/task -> top-level script dependencies`: 0.

## Preserved research state

The source rollout campaign remains complete at 3000 episodes with all videos. Metric
post-processing remains partial at 620/3000 under
`results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/` and is not
resumable. Route audit remains 50 episodes / 200 figures. No result directory was deleted or
rewritten; untracked results total about 13 GB.

## Unverified and next action

S3 proves imports and CPU behavior only. It does **not** prove scene construction, the physical
edge-to-edge spacing, CuRobo planning, or an expert/policy rollout. `checks.smoke_test_seed_2a` and
the actual `diag_kitchen_curobo` diagnostic remain unrun because they require CUDA.

Next: execute S4. Its first runtime risk is dev's `update_world(exclude_obstacles=None)` resolution;
the ring task must explicitly retain the intended obstacle inclusion. S4 needs the user-run GPU
smoke/rollout and owns any runtime reconciliation. S5 remains parallelisable.
