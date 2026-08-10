---
name: tool_geometric_metric
description: "CPU-only envelope-relaxed geometric eps*: what it computes, why the target is label-only, and why Stage 3 rank validation is mandatory"
metadata:
  type: project
---

**Retired 2026-08-10.** The source described here is preserved for inspection at
`research_archive/bench_script_task_metric_2026/lib/geometric_metric.py`; it is intentionally not
an importable or supported live module. `geometric_eps` was the additive CPU-only relaxation of
[[tool_clearance_metric]]. It builds one endpoint-independent volume per `(scene, arm, cfg)` call
and reuses it across an ordered list of world-space legs. Reachability comes only from the
validated precomputed [[tool_reach_envelope]]; it imports no torch/curobo and constructs no IK
solver. The existing `build_grid`, obstacle mesh/mask/EDT, and ungated `widest_path_eps_3d`
implementations are reused.

The target's posed collision mesh is stamped into the route label but deliberately excluded from
the EDT. Without that mask, removing the IK sweep makes the grasped object itself passable and a
route can cut through it. Adding it to the EDT would change the scientific variable from
"clearance to obstacles" to "clearance to obstacles plus target." The target path is resolved from
an existing collision-list entry when available, then `target_collision_path`, then the occluder
task's `target_model`/`target_id`.

`LegResult` contains exactly `eps_star`, `merged`, `bottleneck_xyz`, `route_world`, `start_xyz`,
`goal_xyz`, `n_free`, and `reason`. The geometric volume is built once even for multiple legs;
only endpoint snapping, widest-path, and reconstruction repeat.

This is a relaxation, not the gated metric at lower precision. It omits full-arm scene collision
and the joint-continuity gate; the z bounds exclude under-table paths. The one-sided theorem
`eps_geom >= eps_gated` applies only to graph
pairs with identical grid/EDT, exact common snapped voxels, matched target policy, and
`gated_FREE ⊆ geom_FREE`. Native calls snap independently and are not an invariant test. Stage 1
proved the envelope is safe but loose (~38–42% false-keeps among kept cells). Stage 3's aligned
scalar/rank comparison is mandatory before this quantity can replace gated eps* in a rank-based
study.

After eps* is fixed, geometric routes use a clearance-preferred Dijkstra reconstruction inside
the unchanged `FREE & edt>=eps*` component. Physical anisotropic edge length is divided by local
clearance. This prevents the old BFS tie-break from selecting a short `zmin`-hugging route while
leaving eps* unchanged; it is reporting-only and never enters `compute_route_configs` or a seed.

The deleted `test_geometric_metric.py` formerly verified CPU-only imports, label-only target
masking, and shared-volume `LegResult` behavior. It remains recoverable from Git history at
`3dc788a`.

The archived `research_archive/bench_script_task_metric_2026/compare_geometric_vs_gated.py` was the
Stage 3 runner. It rebuilds the same live scene once per
seed, then constructs two aligned pairs from the shared raw volumes: no-target isolation and
target-masked production. Each pair snaps once against gated FREE and passes that exact voxel pair
to both solvers. Grid/EDT equality, FREE-set inclusion, and the one-sided eps invariant are hard
preconditions; any failure stops before rank reporting. Independently snapped native values remain
diagnostics only. The runner writes `records.jsonl`, `summary.json`, a two-panel scatter, raw old
BFS/preferred routes, route height/length/clearance profiles, side/top-down overlays, Stage-1
false-keep overlap, and `timings.json`. Spearman ranks INACCESSIBLE as 0 and unbounded eps* as
+inf and is gated separately for both aligned series.

**"Furniture is excluded" is FALSE for the study scene (measured 2026-08-07).** The
`scene_obstacle_entries` docstring says the table and walls live in `cuboid_collision_list` and so
never reach the obstacle set. That holds for the OFFICE base task only. `Study_base_task.add_collision`
appends every entry of `scene_obj_info` to plain `collision_list`, and for `put_cup_on_coaster` that
is **`014_bookcase` and `042_wooden_box`** — two large static furniture pieces that `obstacles="all"`
therefore puts in the EDT. The *table itself* is still excluded (it is a `create_table`, not an
actor), so the EDT is not swamped. Impact is small but real: over 50 audited episodes the furniture
binds eps* in **3/50 (6%)**, and the density trend survives — mean eps* **d6 0.138 / d10 0.112 /
d15 0.089**. Binding obstacle over those 50: `001_bottle` 19, `090_trophy` 11, `012_plant-pot` 7,
`086_woodenblock` 4, `014_bookcase` 3, `045_sand-clock` 3, `108_block` 2, `080_pillbottle` 1. Do not
repeat the "furniture is excluded" claim for a study/kitchen scene without re-checking `add_collision`.

**eps* is endpoint-pinned, and that is a construct-validity problem — see
[[tool_task_metric_validity]]** for the authoritative measurement (800/800 legs, plus the 12 cm
wrist offset and grasp-candidate arbitrariness). Independently confirmed on the 50 audited
`put_cup_on_coaster` rollout episodes: the bottleneck sits at z = 0.93 in 49/50 with its xy equal to
the snapped leg endpoint. One consequence specific to the per-leg view: legs 1 and 2 (`grasp`,
`carry`) share the grasp waypoint and therefore tie whenever it binds — but not always, e.g. seed
3007 has grasp 0.1600 vs carry 0.1342 because carry's bottleneck moved off the shared endpoint.

All **2036 legs across the 509 committed metric records are merged, finite, and have a bottleneck** —
zero degenerate legs, so `merged=False` / `+inf` figure paths are untested against real data.

**Study table geometry equals the old ring bench**: SAPIEN bbox `[-0.6,-0.35,0] → [0.6,0.35,0.74]`,
`table_z_bias = 0.0`, `table_height = 0.74`. The default metric grid (`x ±0.6, y ±0.35,
z 0.78–1.23`) therefore matches the study tabletop exactly — no re-tuning was needed when the study
task replaced the office scene, and "the table must be different" is a dead hypothesis.

**Study scope decision (user, 2026-07-31): Stage 3 is an eps* approximation test, not a route-
fidelity test.** The paper's downstream claim is whether the scalar metric correlates with VLA
outcomes. Geometric representative routes are diagnostics for understanding eps*, but they are not
fed to CuRobo and need not match the gated representative path. Any route used as a CuRobo seed
continues to come from the proper gated graph via `compute_route_configs`. This distinction matters
because max-min paths are non-unique when endpoint clearance fixes eps*: two routes can differ
substantially while attaining the same correct scalar bottleneck. Do not turn route disagreement
alone into a failure of the scalar approximation claim; equally, do not claim that a geometric
route is executable or faithful to the arm.
