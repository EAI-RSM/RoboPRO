---
name: domain_bench_script_layout
description: "Post-refactor layout of customized_robotwin/script/bench_script — the lib/ and task/ packages, the dependency rule, and where things moved"
metadata:
  type: project
---

Codex refactored `customized_robotwin/script/bench_script/` on 2026-07-29 (branch
`codex/bench-script-refactor`, commits `ea31499`..`abb917a`, following `REFACTOR_PLAN.md` which is
still in that directory). **Structural only — no behaviour or algorithm changes**, deliberately, so
already-collected A/B data stays comparable. Net ≈ −6800/+5700 lines.

**THE RULE, and the most important outcome: library code must never import from a CLI script.**
Before the refactor the opposite was true (`reachability_map.py` imported
`analyze_occluder_visibility.py`), which is what forced 6 function-local imports and 12
`# noqa: E402` markers. Now `lib/` imports nothing from any top-level script; CLI scripts import
from `lib/`. Verify with
`grep -rn "^from \(analyze_\|clearance_\|reachability_\|visualize_\|seed_from\)" lib/` → must be
empty. Keep this rule over any file-size preference.

**`lib/` is the reusable layer.** Core modules include `metric_config.py` (`SeedMetricConfig` — the
single config source, see below), `ik_grid.py`, `labeling.py`, `continuity.py` (warm-start branch
propagation), `widest_path.py` (the DSU/eps* core), `obstacles.py` (occluder footprints, EDT,
mesh slicing), `occluder_ring.py` (formation), `scene_build.py`, `scene_constants.py`,
`visibility.py`, `planning_tuning.py`, `plotting.py`, and `run_io.py`. Later task-metric work added
further reusable modules; do not rely on a hard-coded file count.

**`task/` (7 modules — `analyze_occluder_visibility.py` shed 3081 lines into it):**
`occluder_task.py` (the class), plus `grasp_mixin.py`, `placement_mixin.py` (647 L, the largest
file in the tree and an accepted exception), `planning_mixin.py`, `seeding_mixin.py`,
`state_checks_mixin.py`, `pose_geometry.py`. **`APPROACH_MODE`/`PLACEMENT_MODE` logic now lives in
these mixins**, not in `analyze_occluder_visibility.py`.

**Also split out of `clearance_metric_3d.py` (−1730 L):** `lib/metric_diagnostics.py` and
`metric_viz.py`. The 2026-08-10 cleanup also moved `reachability_view.py` and
`carry_object_spheres.py` under `lib/`; their runnable forms are
`python -m lib.reachability_view` and `python -m lib.carry_object_spheres`.

**Deleted outright — do not go looking for them:** `clearance_metric.py` (the 2D tool; 10 of its 14
shared functions were byte-identical to the 3D file and nothing imported it), plus
`subgoal_reachability_map.py`, `pickup_reachability_map.py`, `gripper_path_3d.py` (its `VIEWS`,
`_box_wireframe`, `_write_video` moved verbatim into `lib/plotting.py`). The
`pickup-reachability` Makefile target went with them.

**Config is now ONE surface, not three.** `lib/metric_config.py::SeedMetricConfig` is a dataclass
holding the whole metric grid + knobs; the CLI scripts' argparse defaults are `None` and get
overlaid onto it (`from_args`), and **every field is env-overridable as `SEED_<FIELD>`**
(`from_env`) — so `SEED_GATE_TAU`, `SEED_CHUNK`, `SEED_RES`, `SEED_ZMAX`… are instances of one
scheme, not one-off variables. Defaults: x[−0.6,0.6] y[−0.35,0.35] res 0.01, z[0.78,**1.23**]
zres 0.03, gate_tau 0.35, ik_seeds 30, chunk 256, occ_shape "mesh", obstacles "all".

**Verification bar for changes here** (no test runner is configured): `--help` on each top-level
script exercises the full import chain without a GPU; `test_obstacle_set.py` and
`test_ring_config.py` are CPU-only and fast (`test_ring_config` is the important one after any
`occluder_ring.py` change — it asserts the formation is byte-identical per (seed, offset-spec),
which is what guarantees the measured scene equals the rolled-out scene); `smoke_test_seed_2a.py`
needs a GPU and is the only check that the seed pipeline still produces a correctly shaped tensor.
Nothing may exceed 1000 lines.

**Note on citing code in these notes:** prefer symbol names over `file:line`. The refactor moved
almost every line number in the older memories, and grep-able names survive.
