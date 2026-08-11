---
name: domain_bench_script_layout
description: "Post-refactor layout of customized_robotwin/script/bench_script — the lib/ and task/ packages, the dependency rule, and where things moved"
metadata:
  type: project
---

Codex refactored `customized_robotwin/script/bench_script/` on 2026-07-29 (branch
`codex/bench-script-refactor`, commits `ea31499`..`abb917a`, following `customized_robotwin/script/bench_script/plans/REFACTOR_PLAN.md`, which is
kept as the historical execution record). **Structural only — no behaviour or algorithm changes**, deliberately, so
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
`visibility.py`, `planning_tuning.py`, `plotting.py`, and `run_io.py`. The invalidated task-metric
implementation was removed from this live layer on 2026-08-10 and preserved as cold source under
`research_archive/bench_script_task_metric_2026/`; do not treat that archive as an importable
package.

**`task/` (7 modules — `analyze_occluder_visibility.py` shed 3081 lines into it):**
`occluder_task.py` (the class), plus `grasp_mixin.py`, `placement_mixin.py` (647 L, the largest
file in the tree and an accepted exception), `planning_mixin.py`, `seeding_mixin.py`,
`state_checks_mixin.py`, `pose_geometry.py`. **`APPROACH_MODE`/`PLACEMENT_MODE` logic now lives in
these mixins**, not in `analyze_occluder_visibility.py`.

**Also split out of `clearance_metric_3d.py` (−1730 L):** `lib/metric_diagnostics.py` and
`lib/metric_viz.py`. The 2026-08-10 cleanup also moved `reachability_view.py`,
`carry_object_spheres.py`, and `seed_from_clearance.py` under `lib/`; the runnable forms are
`python -m lib.reachability_view`, `python -m lib.carry_object_spheres`, and
`python -m lib.seed_from_clearance`.

**Internal execution/design documents live under `bench_script/plans/`.** Keep top-level
`bench_script/` for runnable entry points and the intentional `setup_paths.py` bootstrap. Surviving
checks live under `bench_script/checks/`; retired execution plans are recoverable from Git history
rather than retained beside active plans. Do not move plans into the published `docs/` site.

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
script exercises the full import chain without a GPU. From `bench_script/`, run CPU checks as
`python -m checks.<module>`; `checks.test_obstacle_set` and `checks.test_ring_config` are fast
(`test_ring_config` is the important one after any `occluder_ring.py` change — it asserts the
formation is byte-identical per (seed, offset-spec), which is what guarantees the measured scene
equals the rolled-out scene). `checks.smoke_test_seed_2a` needs a GPU and is the only check that the
seed pipeline still produces a correctly shaped tensor. Nothing may exceed 1000 lines.

**There are TWO eps\* quantities, not three (corrected 2026-08-11 — a structural sweep miscounted
and the user caught it).** The two quantities are **gated** (IK + joint-continuity gate, GPU) and
**geometric** (`lib/geometric_metric.py`, CPU envelope-relaxed). `geometric_metric` calls only
`build_grid` + `widest_path_eps_3d` — 2 of the 5 steps, no IK solver — so it is a *shorter*
pipeline, not a third copy.

The real duplication is that **`clearance_metric_3d.py` and `lib/seed_from_clearance.py`
independently drive the same full 5-step gated sequence**: `build_grid` → `label_volume` →
`warm_start_branches_3d` → `widest_path_eps_3d` → `reconstruct_widest_path_3d`. Same quantity, two
implementations, different purposes (measurement vs seeding). Not identical — the seed path adds a
tau bisection the metric path lacks.

**`lib/seed_from_clearance.py`'s docstring is STALE and hides this.** It claims *"It reuses
clearance_metric_3d.py verbatim as a library — none of the metric maths is re-implemented here."*
False since this refactor: enforcing "library code must never import from a CLI script" forced it
off `clearance_metric_3d` and onto the `lib/` primitives directly (`.continuity`, `.ik_grid`,
`.labeling`, `.widest_path`), and the header was never updated. **Worth a general check when
touching `lib/`: does each module's docstring still name the modules it actually imports?** This one
survived both a refactor and a prune pass.

**Note on citing code in these notes:** prefer symbol names over `file:line`. The refactor moved
almost every line number in the older memories, and grep-able names survive.
