# Task-metric per-rollout 3D route visualization plan

## 0. Purpose and boundary

Give every already collected `put_cup_on_coaster` rollout a directly paired diagnostic figure
showing:

- the true posed collision meshes used by the geometric metric;
- the representative 3D route for the leg that determined `eps_geom_min`;
- the bottleneck and an equal-aspect wireframe sphere of radius `eps_geom_min`; and
- enough episode, density, bucket, and leg provenance to audit the video grouping.

This is a **validity audit and presentation change**, not a new metric. It must not change
`eps_geom`, `eps_geom_min`, the frozen bucket boundaries, rollout outcomes, or the completed source
records.

The plotted path is the envelope-relaxed geometric metric's representative max-min path. It is
not the executed VLA trajectory, a CuRobo seed, or proof that the robot can execute that route.
Existing rollout artifacts do not contain a time series of end-effector poses, so the actual VLA
trajectory cannot be reconstructed here.

## 1. Reuse what already exists

Do not build another renderer.

Reuse:

- `lib.geometric_metric.geometric_eps` for the already implemented per-leg `route_world`,
  `bottleneck_xyz`, and scalar `eps_star`;
- `metric_viz._metric_path3d` for the 3D axes, path, endpoints, bottleneck, title, and output;
- `lib.plotting._draw_occluder_solids_3d` for the true posed collision meshes;
- `lib.plotting._draw_eps_sphere` for the epsilon sphere;
- `lib.plotting._equal_aspect_3d` so the sphere is not visually distorted;
- `lib.obstacles.surface_distance_to_occluders` for the exact mesh-surface-distance annotation;
- `lib.metric_buckets.assign_metric_record` and the frozen `bucket_spec.json`; and
- the existing scene-manifest and scene-identity validators from `task_metric.py`.

One small renderer change is allowed: give `_metric_path3d` optional path/title/stem parameters so
the new caller can say `geometric representative path` and `eps_geom` instead of the existing
gated-metric wording. Defaults must preserve every existing caller and existing filename.

## 2. Do not touch the active metric run

The real metric post-process is still advancing. `task_metric.py`, `lib/geometric_metric.py`,
`lib/metric_buckets.py`, `lib/task_roles.py`, and `lib/waypoints.py` are bound into its immutable
metric-code hash. Editing them before the 3000/3000 metric run completes would make a later resume
reject its saved config.

Therefore:

1. Let the existing `<rollout>/metric_postprocess` finish unchanged.
2. Require `metric_postprocess/report_state.json` to say `processing_complete=true` and
   `metrics_in_report=target_metrics=3000` before the full visualization pass.
3. Implement visualization as a separate command and a separate output tree. Never delete,
   overwrite, or reinterpret the authoritative rollout or metric episode records.

`lib/metric_viz.py` is not in the current metric-code hash, but avoid changing it while the job is
running anyway; land the complete visualization change after the run so the working state stays
easy to reason about.

## 3. New thin command

Add:

```text
visualize_task_metric_routes.py --rollout-run <completed-rollout-directory>
```

The command is a crash-safe, idempotent backfill. It does not load pi05, run policy inference, or
regenerate rollout videos.

For each manifest episode, in episode order:

1. Read the rollout record, committed metric record, and outcome-blind scene-manifest row.
2. Rebuild the initialized Study scene from the manifest's exact task, seed, config, density, and
   domain-randomization values, using the same scene setup path as `task_metric.py`.
3. Re-run the existing task-role and canonical-waypoint adapters.
4. Recompute all four legs in one `geometric_eps` call with the exact metric config, reach-cache
   path, reach mode, target policy, and grid settings stored in
   `metric_postprocess/config.json`.
5. Recompute the posed obstacle meshes with `occluder_footprints_3d(..., obstacles="all")`.
6. Refuse to plot unless regenerated identity and metric values match the committed record.
7. Find every leg whose `eps_star` equals the committed `eps_geom_min`.
8. Call the existing `_metric_path3d` once for each minimum leg.
9. Commit one small visualization record only after every required PNG for the episode exists.
10. Close the scene, release memory, and continue.

Use deterministic scene replay rather than inventing a serialized-scene proxy. The replay path is
already validated by metric post-processing and keeps this change focused on hooking up the plot.
If runtime later proves unacceptable, a snapshot-only optimization can be planned separately; it
is not part of this work.

## 4. Hard regeneration checks

Before a figure is accepted, require:

- exact `scene_id` and `scene_fingerprint` equality;
- exact episode, seed, density, realized clutter count, and acting-arm equality;
- the same number and ordering of canonical legs;
- per-leg equality for `kind`, `merged`, and unbounded status, plus equality of the shared
  record-level `n_free`;
- finite `eps_star` agreement within `1e-12` metres;
- bottleneck coordinates equal on the metric grid within `1e-12` metres; and
- the recomputed set of minimum-leg indices equal to the set derived from the committed record.

Any mismatch is a validity failure. Stop loudly with the episode number; do not quietly draw a
figure from a different scene or a different metric implementation.

The figure must show both the voxel-centre EDT epsilon and the existing exact mesh-surface distance
at the bottleneck. The sphere may differ from exact contact by the known grid discretization; that
difference should remain visible rather than being cosmetically corrected.

## 5. Minimum-leg and tie policy

`eps_geom_min` is the minimum over the four canonical legs. The plot must therefore show the leg or
legs that actually assigned the episode's bucket.

- One unique minimum: write one PNG.
- Exact tie: write one PNG per tied minimum leg by invoking the same existing renderer repeatedly.
- `+inf`: draw the route and label epsilon as unbounded; do not draw an infinite sphere.
- Inaccessible/no-route: retain the endpoints and status label; do not invent a path or sphere.

Writing one file per tied leg avoids new multi-route plotting code and keeps the meaning of every
sphere unambiguous.

Suggested names:

```text
metric_route_visuals/
  config.json
  episodes/
    episode000000.json
    episode000001.json
  figures/
    episode000000_seed3000_grasp.png
    episode000000_seed3000_carry.png   # only when tied
  route_visual_index.json
  report_state.json
```

Each episode record should contain the source scene ID/fingerprint, episode, seed, density, outcome,
`eps_geom_min`, `rho_geom_min`, frozen bucket, minimum-leg list, PNG paths, video path, exact
mesh-distance values, and the regeneration checks above. Do not duplicate the large metric or
rollout records inside it.

## 6. Pair figures with videos by density

The existing top-level `videos_by_clearance` tree pools d6, d10, and d15, which makes visual
inspection harder even when numerical assignment is correct. Add a non-destructive audit index:

```text
metric_route_visuals/by_density/
  d6/<clearance_bucket>/<hard_outcome>/
  d10/<clearance_bucket>/<hard_outcome>/
  d15/<clearance_bucket>/<hard_outcome>/
```

For each episode, place relative symlinks to:

- the finalized rollout video; and
- every minimum-leg 3D PNG.

Use filenames containing both episode and seed. Never move, rename, or overwrite source videos.
An existing path is accepted only when it is the expected symlink target; otherwise fail rather
than replacing an unrelated file.

`route_visual_index.json` is authoritative for the pairing. Do not add fields to the existing
`video_index.json`, because the regular correlation report regenerates that file.

## 7. Crash safety and resume

The full pass may take hours, so it must resume without repeating finished work.

- Bind `config.json` to the source rollout config hash, completed metric config hash, bucket-spec
  hash, reach-cache path, visualization code hash, and target episode count.
- Write one atomic, fsynced `episodes/episodeNNNNNN.json` after the PNGs are atomically renamed into
  place.
- Treat those episode JSON files as authoritative and regenerate the combined index/report state
  from them.
- Skip an episode only after validating its visualization record, source hashes, and every PNG.
- Refresh the density/bucket/outcome symlink index periodically and at clean exit.
- Refuse resume under a changed immutable visualization config.

Do not couple visualization success to the authoritative metric records. An interrupted or failed
visualization pass must leave the completed scientific inputs untouched.

## 8. Focused verification

### CPU checks

Add `test_task_metric_route_visualization.py` covering:

1. unique minimum selects one leg and one plot call;
2. exact ties select all tied legs without splitting or silently choosing one;
3. `+inf` and inaccessible legs do not request an invalid sphere/path;
4. committed-versus-regenerated epsilon, bottleneck, identity, and leg mismatch rejection;
5. frozen bucket assignment is reused rather than reimplemented;
6. density/bucket/outcome video and PNG links point to the expected sources;
7. repeated indexing is idempotent;
8. a conflicting existing path is rejected;
9. interrupted output without an episode commit is recomputed safely; and
10. completed episodes are skipped only when their full artifact set validates.

Keep scene construction and `geometric_eps` mocked in focused tests. Existing geometric-metric,
bucket, correlation, obstacle, task-metric, and CLI-help regressions must remain unchanged.

### One real plumbing gate

The user runs one already committed episode through the new command with an episode selector. Pass
only if:

- regenerated identity and all four metric legs match the committed record;
- the expected minimum-leg figure(s) exist;
- the image visibly contains the true obstacle meshes, 3D route, bottleneck, and epsilon sphere;
- the title says geometric representative path, not gated path or executed VLA path;
- epsilon, density, bucket, episode, seed, arm, and leg are legible; and
- the paired video symlink resolves to the correct source video.

### Stratified visual gate

Before launching all 3000 episodes, the user renders two episodes from each frozen bucket within
each density (up to 24 episodes, retaining all ties). Inspect these for mesh placement, route leg,
sphere scale, view readability, and correct video pairing. This is a renderer/plumbing gate only;
do not revise the frozen buckets from outcome-bearing videos.

### Full completion gate

The full pass is complete only when:

- 3000 visualization episode records exist and validate;
- every rollout has at least one PNG;
- every tied minimum has its own PNG;
- all 3000 video links resolve;
- all PNG links resolve;
- there are zero identity, metric-regeneration, bucket, or source-video mismatches; and
- `report_state.json` says `processing_complete=true` with counts by density, bucket, outcome, and
  minimum leg.

## 9. Expected cost

Current metric records average about 10.5 seconds per scene including scene setup. Treat nine hours
as the conservative serial upper bound for 3000 episodes, plus plotting and filesystem overhead.
No policy model or VLA rollout is loaded. One PNG per episode is expected to use roughly 0.6 GB in
total; tied minima add some overhead.

The command should print completed/target count, elapsed time, mean seconds per episode, and a
simple ETA so a long backfill is observable.

## 10. Deliverables

1. Backward-compatible semantic/output parameters on `metric_viz._metric_path3d`.
2. `visualize_task_metric_routes.py` implementing replay, checks, plotting, resume, and indexing.
3. `test_task_metric_route_visualization.py` with the focused cases above.
4. A one-episode real plumbing artifact inspected by the user.
5. A stratified density-by-bucket visual gate inspected by the user.
6. The completed `metric_route_visuals/` tree for all 3000 existing rollouts.

## 11. Explicit non-goals

- No new renderer or plotting framework.
- No changes to geometric epsilon, route reconstruction, reach envelopes, canonical waypoints, or
  obstacle selection.
- No changes to `bucket_spec.json` or post-hoc threshold tuning from videos.
- No VLA/model rerun and no expert/CuRobo run.
- No claim that the representative route is the executed or arm-feasible trajectory.
- No deletion or regeneration of the authoritative rollout or metric post-process directories.
- No serialized-scene proxy or performance refactor before the straightforward replay path is
  measured.
