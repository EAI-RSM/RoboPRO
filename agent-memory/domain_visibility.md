---
name: domain_visibility
description: "Controlled-visibility axis: 5-bucket taxonomy, why the denominator's capture timing matters, and the one safe render speedup"
metadata:
  type: project
---

**User scope directive (2026-07-31): visibility is a separate artifact, not part of
`TASK_METRIC_CORRELATION_PLAN.md`.** Do not measure it, add it to task-metric records, or use it as
an association covariate unless the user explicitly brings visibility back into scope. Preserve
the existing visibility tooling and findings; this directive defers their use rather than deleting
them.

The "controlled target visibility" benchmark axis: at episode start, place the target into a
visibility bucket as seen from the **countertop camera** (not head), measured via
`actor_segmentation`. Parent design = issue #27; sub-issues #28 (Phase 0 primitive), #30 (Phase 1
distribution), #35 (Phase 2 occluder).

**Bucket taxonomy** — `DEFAULT_VISIBILITY_BUCKETS` in `customized_robotwin/envs/_base_task.py`:
`not_visible` (0 px) / `heavily_occluded` (0–0.20) / `mostly_occluded` (0.20–0.5) /
`partially_occluded` (0.5–0.9) / `fully_visible` (≥0.9), where
`visible_fraction = visible_px / full_px(clean, same seed)`.

**Design decision (supersedes the original #27 plan):** a height binary search on a scalable
occluder was the original core of Phase 2 — it is **deferred**. Primary path is a fixed
deterministic occluder + post-hoc bucket selection, which works because the occluder *populates
all buckets* (natural clutter leaves the occluded buckets empty). A fully general "hit any ratio
for any seed under any constraint" guarantee is not feasible and is explicitly not promised.
Empirically one fixed occluder + a 20% no-occluder probability gave a near-uniform 5-bucket spread
(not_visible 24 / heavily 32 / mostly 16 / partially 26 / fully 2 %, KL₅ ≈ 0.11).

**Clutter cannot occlude the target on the countertop view** (confirmed twice). The target's
prohibited area, the gripper operating-area reservation, and furniture footprints all sit on or
around the camera→target sightline, and clutter scatters uniformly, so it almost never lands on
the steep −y sightline tall enough. Occlusion must be engineered with a dedicated occluder.
Related: the clutter "radius" metric `(ext_x+ext_z)/4` badly underestimates a flat carton's
occluding silhouette.

**Denominator timing (still to wire, Phase 3).** In the overhead countertop view the gripper can
already overlap the target at t=0. `full_target_pixel_count` is therefore sensitive to *when* it is
captured — a contaminated denominator silently inflates `visible_fraction`. Capture
`capture_target_pixel_count` as the FIRST scene measurement, before any obstructor/occluder
(obstructors-first / occluder-last ordering), and verify the arms are clear of the sightline at
capture time.

**The only safe speedup for the no-rollout sweeps** is a `measurement_only` cfg flag (default
False, set True in `build_cfg`) that renders ONLY the measured camera instead of all ~5 static +
2 wrist cameras per `measure_target_visibility`. Implementation: `Office_base_task._init_task_env_`
sets `self.measurement_only` (office tasks go `Office_base_task → Bench_base_task → Base_Task`;
editing `envs/_base_task.py` alone is a no-op for office); `measure_target_visibility` calls
`update_picture(camera_names=[camera_name])`; `Camera.update_picture`/`get_rgba`/`get_rgb` take an
optional `camera_names` and must skip the WRIST cameras too — reading a never-rendered wrist cam
during `save_overlay` is what caused repeated `IndexError: _Map_base::at`.

**Dead ends — do not retry on this SAPIEN build:**
- Rasterizer shader (`set_camera_shader_dir("default")`) → that pack lacks the `Segmentation`
  render target → `IndexError: _Map_base::at`. Keep `"rt"`.
- Lowering ray-tracing spp 32→1 → unconfirmed/suspect, abandoned.
- Skipping `set_planner` when `need_plan=False` → `communication_flag` is set only there and is
  read by `Robot.reset` → AttributeError. The planner is built once per env anyway.

**Debugging unlock:** the sweep swallows exceptions into a one-line message. Add
`traceback.print_exc()` to the `except` blocks to see the real failing line.

Scene-side facts live in [[domain_scene]]; per-scene difficulty is [[tool_clearance_metric]].
