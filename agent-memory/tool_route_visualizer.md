---
name: tool_route_visualizer
description: "Per-rollout 3D eps* route figures: the 'everything looks shifted' false alarm and how it was refuted, the tool-offset gap, the config-hash immutability trap, and render cost"
metadata:
  type: project
---

**Retired 2026-08-10.** The cold source at
`research_archive/bench_script_task_metric_2026/visualize_task_metric_routes.py` replayed each
committed `put_cup_on_coaster` scene (no policy, no rollout), re-derived the metric, checked it
against the committed record, and rendered `metric_viz._metric_path3d`. The completed 50-episode,
200-figure audit and its metric records remain preserved; the command is no longer maintained.

## The figures are NOT geometrically shifted — do not re-investigate

Reported 2026-08-07 as "everything looks shifted, maybe the table dimensions/height differ from the
old task". **Both halves are wrong, and this was verified twice; do not spend time on it again.**

1. Every posed obstacle mesh the metric reconstructs was compared against **SAPIEN's own physx
   collision-shape vertices** (`get_actor_boundingbox` style: `entity_pose @ shape.local_pose`,
   vertices × `shape.scale`). Over all 8 obstacles of seed 3000 the world AABBs agree to
   **≤ 0.2 mm**. So `occluder_footprints_3d`'s `(V*scale) @ R.T + p` is faithful — there is no
   hidden mesh-local transform in `create_actor` (it builds collisions with `scale=` and no local
   pose), and glTF/SAPIEN axis conventions match.
2. The rendered layout was compared against **frame 0 of the episode's own rollout video** —
   wooden box front-left, bookcase back-right, trophy far right, bottle back-left. Identical.

The illusion came entirely from `lib/plotting.py::_equal_aspect_3d`, which achieves 1:1:1 by padding
**every** axis out to the longest span. The tabletop is ~1.12 m wide but only ~0.31 m tall, so z was
inflated to 1.12 m (ticks 0.4→1.4) and the whole scene collapsed into the middle third of an empty
cube with no tick near it and no ground plane under it.

Fixed by `_true_aspect_3d(ax, lo, hi)` (explicit box + `set_box_aspect(hi-lo)` — still exactly
1:1:1, so the eps* sphere stays a sphere and "just kisses the nearest obstacle" remains checkable)
plus `_draw_ground_plane`. Both are **opt-in kwargs** on `_metric_path3d` defaulting to the old
behaviour, so the other two callers — `clearance_metric_3d.py` and `lib/seed_from_clearance.py` — are
untouched. `_equal_aspect_3d` itself was left alone.

## The route floats ~18 cm above the target — the drawing is right, the metric is not

`lib/waypoints.py::canonical_waypoints` emits **end-effector-LINK** poses carrying a fixed 0.12 m
tool offset (`pre_grasp` −0.22 m, `grasp` −0.12 m along the approach axis). For seed 3000 the cup's
actor origin is z = 0.740 with its top at 0.801, while the `grasp` waypoint is z = 0.920, and the
fingertip contact is at z ≈ 0.80 — the cup's rim. So the figure is drawing the right point and the
blue star only *looks* like a miss; the dotted "tool offset (EE link -> target)" connector on the
grasp-side legs exists to say so (`carry`/`place` get none — the cup has left its spawn).

**Do not conclude from that connector that the offset is harmless.** [[tool_task_metric_validity]]
measured it as the metric's biggest single defect: the wrist waypoint floats above 6–8 cm clutter,
reads higher than the rim in 92/100 scenes (median +0.023 m), and drives
`spearman(fingertip clearance, eps_geom_min) = 0.078`. The figures are a faithful picture of a
quantity that is itself mis-specified — which is exactly why drawing them was worth doing.

## Config immutability: every code edit forces a fresh `--out-dir`

`_visualization_code_version()` hashed `visualize_task_metric_routes.py` + `lib/metric_viz.py` +
`lib/plotting.py` into `config_sha256`. `_load_or_create_config` compares that against the saved
`config.json` and **raises** `route-visualization configuration differs from the immutable saved
config`. Partially-rendered episodes under the old hash are **rejected, not skipped**
(`read_visual_records` validates `visualization_config_sha256` per episode), so there is no partial
resume across an edit — it is always a full re-render into a new directory. This fired three times
in one session. Finish the code changes *first*, then launch the long run.

## Cost (measured 2026-08-07)

Per figure, dominated by rasterizing translucent `Poly3DCollection` obstacle meshes — **3.6 s at 57k
obstacle triangles (8 meshes), 5.6 s at 90k (12 meshes)**. `surface_distance_to_occluders` is
negligible (0.03–0.4 s). PNGs ~220 KiB. Scene rebuild is ~13–20 s and is per-episode.
With all four legs drawn: **~33 s/episode → 50 episodes ≈ 28 min**. Extrapolating to the full 3000
committed metrics: ~12,000 figures, ~2.6 GB, ~29 h single-process.

## All four legs are drawn (changed 2026-08-07, user request)

Originally only the leg(s) attaining `eps_geom_min` were plotted (65 figures over 50 episodes),
which made the figures uninterpretable — you could not see what the binding leg beat. Now every
canonical leg gets a figure, with `"is_minimum": bool` in the episode record and the caption saying
either `BINDS eps_geom_min` or `not binding; eps_geom_min is at <kinds>`.
`report_state.json::figure_counts_by_minimum_leg` filters on `is_minimum` so it still reports where
eps* lives instead of a flat per-leg tally.

Filenames are `episode{N:06d}_seed{S}_leg{i}_{kind}.png`. **The `leg{i}` component is load-bearing**:
sorting on `kind` alone gives `carry, grasp, place, pre_grasp` — legs 2, 1, 3, 0 — so browsing the
folder walked the path out of order. With the index the name sort is execution order, and in the
`by_density/` audit folders the four figures still group ahead of that episode's `_video.mp4`.

Leg order for this task is `0 pre_grasp → 1 grasp → 2 carry → 3 place`; a leg's `kind` always names
its GOAL waypoint. Which leg binds depends on where the cup and coaster happened to spawn — a
"place-only" scene just means the destination was in the crowded region and the pick was not.
See [[tool_geometric_metric]] for why the bottleneck is the endpoint.

The deleted route-visualization plan specified minimum-legs-only and was out of sync with the
archived code; it remains recoverable from Git history at `3dc788a`.
