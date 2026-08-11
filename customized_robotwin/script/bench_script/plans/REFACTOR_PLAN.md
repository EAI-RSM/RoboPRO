# bench_script refactor plan

Executable plan for Codex. Written 2026-07-29 against commit `83e392f` (branch `peng-research-branch`).

The goal is **structural only**: no behaviour changes, no algorithm changes, no new features. Every
stage below is either a deletion of code nothing calls, or a move of code from one file to another.
If a stage cannot be done without changing behaviour, stop and report rather than improvising.

A separate section at the end lists **suspected bugs**. Those are *not* part of the refactor. Do them
in their own commits, after the structural work, and treat every one as unconfirmed — they were found
by reading, not by running. Several may turn out to be intentional.

---

## 0. Ground rules

**File size.** Target 150–500 lines per file. 1000 lines is a hard ceiling — if a file would exceed
it, split it by *purpose*, not by line count. A file should have one job you can name in a sentence.
Files under ~80 lines are usually a sign the split went too far; fold them back into a sibling.

**Dependency direction.** After this refactor the rule is: **library code must never import from a CLI
script.** Today the opposite is true (`reachability_map.py` imports `analyze_occluder_visibility.py`),
which is the root cause of the function-local-import hacks and the twelve `# noqa: E402` markers. The
new `lib/` package imports nothing from any top-level script. CLI scripts import from `lib/`. This is
the single most important outcome of the plan; if you have to choose between a nice file size and
keeping this rule, keep the rule.

**Mechanical moves only.** When moving a function, move it *verbatim* — same body, same defaults,
same docstring. Fix only the import lines. Do not rename, do not "clean up while you're in there",
do not reorder arguments. Any behaviour change makes the diff unreviewable and the A/B data
incomparable to what has already been collected.

**One stage per commit.** Stages are ordered so each one leaves the tree working. Run the checks in
§7 after every stage. Do not batch stages together.

**Environment.** These scripts need the repo's env exports and are run from `benchmark/`. Import-only
smoke checks (`python -c "import X"`) need `setup_paths` to have run, so use the existing entry points
rather than bare imports where possible.

---

## 1. Stage 1 — delete four files

Delete outright:

| File | Lines | Why it is safe |
|---|---|---|
| `clearance_metric.py` | 625 | Superseded by `clearance_metric_3d.py`. 10 of its 14 shared functions are **byte-identical** to the 3D file (`report`, `widest_path_eps`, `select_arm`, `occluder_footprint_polys`, `reconstruct_widest_path`, `occluder_clearance`, `_overlays`, `feasibility`, `grasp_orientation`, `nearest_free_cell`). Only `label_grid` is unique, and nothing calls it. Zero importers; the only textual reference in the repo is `CLEARANCE_2D_TO_3D.md` describing it as the thing that was replaced. |
| `subgoal_reachability_map.py` | 356 | Zero references anywhere in the repo — no importer, no Makefile target, no doc, no shell script. |
| `pickup_reachability_map.py` | 160 | One reference: a Makefile target (see fallout below). Superseded by the clearance route. |
| `gripper_path_3d.py` | 249 | Draws the occluder as a milk box (wrong asset — see Bug 2). One importer: `swept_volume_3d.py` (see fallout below). |

### 1a. Fallout that must be handled in the same commit

**`swept_volume_3d.py:80`** does `from gripper_path_3d import VIEWS, _box_wireframe, _write_video`.
Those three live in the file being deleted:

- `VIEWS` — `gripper_path_3d.py:54`, a 1-line list of four `(name, elev, azim)` camera angles
- `_box_wireframe` — `gripper_path_3d.py:57`, ~20 lines
- `_write_video` — `gripper_path_3d.py:204`, ~34 lines

Move all three verbatim into the new `lib/plotting.py` (created in Stage 2). If you are doing Stage 1
before Stage 2, park them in `lib/plotting.py` anyway and create the package early — do not copy them
into `swept_volume_3d.py`, which would just relocate the duplication problem.

Do **not** move `_plot_path` (`gripper_path_3d.py:77`) — it is specific to the deleted script.

**`Makefile:340`** invokes `script/bench_script/pickup_reachability_map.py`. Delete that target and
its help-text line. Check the surrounding target block for now-unused variables (`OCC_*` grid vars
that only that target read) and remove those too, but leave any variable another target still uses.

**`benchmark/bench_envs/_bench_base_task.py:1508`** has a comment naming `gripper_path_3d.py` as the
example consumer of `env.step_hook`. The hook itself must stay — `swept_volume_3d.py` still uses it
(`:314`, `:323`). Update the comment to name `swept_volume_3d.py` instead. Do not touch the hook code.

**`PICKUP_ONLY` becomes dead.** `analyze_occluder_visibility.py:593` defines `PICKUP_ONLY = False` and
`:817` branches on it inside `play_once`. The only place that ever set it `True` was
`pickup_reachability_map.py:105`. Once that file is gone the branch is unreachable. Remove the class
attribute and the branch. This is the one deletion in Stage 1 that touches live code — do it as a
separate commit from the file deletions so it can be reverted independently.

### 1b. What does *not* break

`reachability_map._build_ik_solver` and `_solve_grid` lose two of their four consumers here, but
`clearance_metric_3d.py` still imports both (`:59`, `:60`) and calls them at `:429`, `:650`, `:662`,
`:663`, `:1901`. `reachability_map.py` stays. Do not delete it.

---

## 2. Stage 2 — create `lib/`, the reusable core

Create `bench_script/lib/` with an `__init__.py`. `bench_script/` is already on `sys.path` (via
`setup_paths`), so `from lib.obstacles import occluder_footprints_3d` resolves without any path
changes.

**Leave `setup_paths.py` where it is**, at the top level. Eighteen files across the repo do
`from setup_paths import setup_paths`; moving it means touching all of them for no gain. It is the one
intentional exception to the package layout.

### 2a. Modules to create, and what goes in each

Sizes are estimates from the current line counts of the functions being moved.

| New module | ~Lines | Contents (move verbatim) | Moved from |
|---|---|---|---|
| `lib/scene_constants.py` | ~130 | `TARGET_MODEL`, `TARGET_ID`, `TARGET_XLIM/YLIM`, `TABLE_XLIM/YLIM`, `PAD_XY`, `OCCLUDER_MODEL`, `OCCLUDER_ID`, `OCCLUDER_COLLISION`, `OCCLUDER_QPOS`, `OCC_HALF_FOOTPRINT`, `OCC_PAD_CLEARANCE`, `_PAD_HALF`, `_OCC_HALF`, `OCC_PAD_MIN_DIST` | `analyze_occluder_visibility.py:124–148`, `:247`, `:564–567` |
| `lib/occluder_ring.py` | ~120 | `occluder_ring_xy`, `parse_offset_specs`, `parse_count_choices`, `draw_ring_config` | `analyze_occluder_visibility.py:151–246` |
| `lib/planning_tuning.py` | ~220 | The ~41 tuning constants and their `os.environ` reads: all `GRASP_*`, `OBJECT_*`, `CONTACT_*`, `DESCENT_*`, `LANDING_*`, `SIDE_WAYPOINT_*`, `PLACEMENT_*`, `WAYPOINT_*`, `LOCAL_WAYPOINT_ATTEMPTS`, `STAGE_ORDER`, `REACH_X_LIMIT`, `ATTACHED_TRAJECTORY_SLOWDOWN` | `analyze_occluder_visibility.py:248–562` |
| `lib/scene_build.py` | ~180 | `build_cfg`, `DR_CLEAN`, `dr_measure`, `get_env_class`, `get_embodiment_config` | `analyze_natural_visibility.py`, `analyze_occluder_visibility.py:570`, `visualize_task_scene.py` |
| `lib/visibility.py` | ~250 | `save_overlay`, `analyze`, `analyze_rollout`, `run_rollout`, `_resolve_target`, `TARGET_ATTRS`, `_bucket_of` | `analyze_natural_visibility.py` |
| `lib/ik_grid.py` | ~200 | `_build_ik_solver`, `_solve_grid` (from `reachability_map`); `build_grid`, `_build_ik_solver_no_world`, `_solve_grid_q`, `_solve_grid_q_multi` (from `clearance_metric_3d`) | `reachability_map.py`, `clearance_metric_3d.py:112–284` |
| `lib/continuity.py` | ~90 | `_wrap_linf`, `_pick_nearest`, `warm_start_branches`, `warm_start_branches_3d` | `clearance_metric_3d.py:285–374` |
| `lib/labeling.py` | ~140 | `label_volume`, `load_reach_envelope`, `geometric_envelope` | `clearance_metric_3d.py:148–214`, `:375–447` |
| `lib/obstacles.py` | ~350 | `_load_collision_mesh`, `scene_obstacle_entries`, `occluder_footprints_3d`, `occluder_footprint_polys`, `obstacle_centers`, `occluder_slice_polys`, `occluder_mask_3d`, `occluder_clearance`, `occluder_clearance_3d`, `surface_distance_to_occluders` | `clearance_metric_3d.py:448–514`, `:1018–1293`, `:1443–1461` |
| `lib/widest_path.py` | ~200 | `nearest_free_cell`, `nearest_free_voxel`, `widest_path_eps`, `reconstruct_widest_path`, `widest_path_eps_3d`, `reconstruct_widest_path_3d` | `clearance_metric_3d.py:515–622`, `:1341–1442` |
| `lib/plotting.py` | ~200 | `VIEWS`, `_box_wireframe`, `_write_video` (rescued from `gripper_path_3d`); `_equal_aspect_3d`, `_draw_occluder_solids_3d`, `_draw_eps_sphere`, `_scene_anchor_markers`, `_line_axis` | `gripper_path_3d.py`, `clearance_metric_3d.py:1462–1515`, `:1655–1675` |
| `lib/run_io.py` | ~120 | `Timings`, `effective_out_dir`, `_Tee`, `_prune_empty_topdirs` | `clearance_metric_3d.py:71–111`, `analyze_occluder_visibility.py:71–121` |

`lib/scene_build.py` and `lib/visibility.py` are the ones to be most careful with: `build_cfg` and
`DR_CLEAN` currently live in `analyze_natural_visibility.py` and are imported by five modules, and
`get_env_class` lives in `visualize_task_scene.py` and is imported by two. Moving them is what
finally lets `analyze_natural_visibility.py` and `visualize_task_scene.py` become plain CLI scripts
instead of accidental libraries.

### 2b. Update every importer

After the moves, rewrite the import blocks in: `analyze_occluder_visibility.py`,
`analyze_natural_visibility.py`, `clearance_metric_3d.py`, `reachability_map.py`,
`reachability_view.py`, `visualize_task_scene.py`, `swept_volume_3d.py`, `seed_from_clearance.py`,
`carry_object_spheres.py`, `reach_envelope.py`, `validate_reach_envelope.py`,
`validate_visibility_measurement.py`, `smoke_test_seed_2a.py`, `test_obstacle_set.py`,
`test_ring_config.py`.

Two things to delete as you go, because they exist only to work around the inverted dependency:

- The **function-local imports** at `analyze_occluder_visibility.py:2554`, `:2728`, `:2812` and
  `seed_from_clearance.py:122`, `:188`, `:418`. Once `lib/` has no back-edge to the CLI scripts these
  can be plain module-level imports. Verify no cycle remains before promoting each one.
- The `# noqa: E402` markers that become unnecessary. Some will still be needed (`setup_paths()` must
  still run before importing anything that pulls in torch/sapien); keep those and delete the rest.

---

## 3. Stage 3 — split `analyze_occluder_visibility.py` (3476 lines)

`make_occluder_task()` at `:579–3130` is a **2551-line function** wrapping one class with 48 methods.
The nesting exists for exactly one reason: `Base = get_env_class("put_mouse_on_pad",
bench_subdir="office")` at `:580` is a runtime lookup, so the base class is not known at import time.
The function takes no arguments and the class body closes over nothing but `Base` — verified by
reading the whole function body. That makes the split below safe.

### 3a. The technique

Define each method group as a **plain mixin class at module level** (inheriting `object`, not `Base`),
then assemble in a thin factory:

```python
# task/occluder_task.py
def make_occluder_task():
    Base = get_env_class("put_mouse_on_pad", bench_subdir="office")

    class OccluderTask(SeedingMixin, PlacementMixin, PlanningMixin,
                       GraspMixin, StateChecksMixin, PoseGeometryMixin, Base):
        ...class attributes...
        ...load_actors, play_once...

    return OccluderTask
```

Two things that will bite if you miss them:

1. **Method groups are disjoint.** All 48 methods were checked: no name appears in two groups, so
   there is no MRO ambiguity and mixin order does not change behaviour. Keep `Base` last. If you find
   yourself needing to duplicate a method into two mixins, the grouping is wrong — stop and report.
2. **Default arguments are evaluated at class-definition time.** Eight methods use module constants as
   defaults, e.g. `def _around_box_waypoint(self, arm_tag, ref_pose, gap=SIDE_WAYPOINT_GAPS[1], ...)`.
   Every mixin module must import the constants it uses from `lib/planning_tuning.py` at module level,
   or those defaults will raise `NameError` at import.

### 3b. The groups

Create `bench_script/task/` with an `__init__.py`. Line totals are measured from the current file.

| New module | Lines | Methods |
|---|---|---|
| `task/occluder_task.py` | ~500 | Class attributes, `load_actors` (65L), `play_once` (383L), the factory |
| `task/planning_mixin.py` | ~390 | `_plan_pose_trajectory_sequence`, `_plan_pose_with_shrinking_waypoint`, `_plan_pose_with_local_waypoint_retry`, `_plan_and_replay_pose`, `_replay_planned_move`, `_replay_planned_sequence`, `_plan_pose_sequence`, `_local_waypoint_candidates`, `_time_stretch_trajectory`, `_record_plan_effort`, `_arm_plan_func`, `_arm_active_joint_indices`, `_roll_qpos_forward` |
| `task/placement_mixin.py` | ~610 | `_plan_pose_with_descent_slices` (189L), `_local_landing_search_and_place` (226L), `_select_attached_placement_plan` (135L), `_placement_execution_steps`, `_backward_subgoal_poses`, `_descent_tstep_fraction_for_attempt`, `_post_grasp_escape_poses` |
| `task/grasp_mixin.py` | ~410 | `_plan_grasp_side`, `_plan_candidate`, `_select_pick_place_candidate`, `_candidate_specs`, `_rank_side_grasp_ids`, `_geometric_grasp_pose`, `grasp_actor_from_table`, `_grasp_via_cached_trajectories` |
| `task/seeding_mixin.py` | ~370 | `_get_approach_seed`, `_get_carry_seed`, `_note_seed_stat`, `_approach_mode`, `_placement_mode`, `_carry_seed_on` |
| `task/state_checks_mixin.py` | ~180 | `_trajectory_path_metrics`, `_object_retained`, `_object_near_placement_target`, `_object_near_support_height`, `_placement_xy_error`, `_gripper_relative_object_transform` |
| `task/pose_geometry.py` | ~65 | `_blend_pose`, `_verified_intermediate`, `_box_side_x`, `_around_box_waypoint` |

`analyze_occluder_visibility.py` then keeps only `run()` (`:3130`, 268L) and `main()` (`:3398`, 78L)
plus its argparse — roughly **350 lines**, a normal CLI script.

### 3c. Two methods to delete first

Before splitting, remove these — neither has a single `self.` reference anywhere in the file:

- `_execute_actions_via_plan_and_replay` — `:2079`, 45 lines
- `_pick_side_grasp_id` — `:1376`, 3 lines

Do this as its own commit so it is separable from the move. If either turns out to be called through
`getattr` or from the base class, revert just that commit.

### 3d. Note on `placement_mixin.py`

At ~610 lines it is over the 500 target, though under the 1000 ceiling. It is acceptable as-is for
this pass. If you want it inside the band, the natural seam is `_plan_pose_with_descent_slices` +
`_descent_tstep_fraction_for_attempt` (~200L, "get the gripper down safely") into
`task/descent.py`, leaving `task/placement_mixin.py` at ~410L. Do this only after the main split is
committed and verified — not in the same commit.

---

## 4. Stage 4 — split `clearance_metric_3d.py` (2100 lines)

Most of this file is already destined for `lib/` in Stage 2. What remains after those moves is the
diagnostics, the visualisation, and the orchestration. Split it as:

| Module | ~Lines | Contents |
|---|---|---|
| `metric_diagnostics.py` | ~250 | `_jump_field`, `_print_jump_stats`, `_draw_pair`, `phase0_gate_diagnostic`, `phase1_stack_report`, `_vertical_edges_by_z`, `_edge_stats`, `phase2_vertical_report`, `phase3_clearance_report` |
| `metric_viz.py` | ~390 | `feasibility`, `_overlays`, `report`, `_metric_path3d`, `_viz_side_elevation`, `_viz_clearance_profile`, `_viz_topdown`, `_viz_ceiling`, `phase4_visuals` |
| `clearance_metric_3d.py` | ~380 | `phase4_metric`, `select_arm`, `grasp_orientation`, `run`, `main` + argparse |

These three are *tools*, not library code — they stay at the top level of `bench_script/` rather than
going into `lib/`. `lib/` is for things multiple tools import; `report` and the `_viz_*` functions are
called only from this pipeline.

`seed_from_clearance.py` currently imports `clearance_metric_3d` as a module and reaches for
`build_grid`, `label_volume`, `occluder_footprints_3d`, `occluder_clearance_3d`, `widest_path_eps_3d`
and friends. After Stage 2 those all live in `lib/`, so its three function-local
`import clearance_metric_3d as cm` statements become module-level `from lib import ...`. Check
`seed_from_clearance.py:214`, `:220`, and the `save_route_visuals` path, which also reaches into the
`_viz_*` functions — that one legitimately needs `metric_viz`.

---

## 5. Stage 5 — one source of truth for metric config

The metric's knobs are currently defined independently in three places:

1. `clearance_metric_3d.py` argparse — 31 flags
2. `SeedMetricConfig` dataclass — `seed_from_clearance.py:37`
3. Nineteen `SEED_*` environment variables read in `analyze_occluder_visibility.py`
   (`SEED_RES`, `SEED_ZMAX`, `SEED_ZRES`, `SEED_GATE_TAU`, `SEED_OBSTACLES`, `SEED_CHUNK`,
   `SEED_VISUALS`, `SEED_VISUALS_DIR`, `SEED_MEM_LOG`, …)

Make `SeedMetricConfig` the single definition (it already documents itself as mirroring the argparse
defaults) and have both the argparse layer and the env-var layer *populate* it rather than re-declare
defaults:

- Move `SeedMetricConfig` to `lib/metric_config.py`.
- Add `SeedMetricConfig.from_args(args)` and `SeedMetricConfig.from_env(base=None)` — both thin, both
  returning a config object. Neither should contain a literal default value; the dataclass field
  declarations remain the only place a default appears.
- `clearance_metric_3d.py`'s argparse keeps its flags but sets `default=None`, and `from_args` leaves
  the dataclass default in place for anything unset. This is the mechanical way to guarantee the two
  surfaces can never drift again.

**Before doing any of this, resolve Bug 1 below** — the two surfaces have already drifted and you need
to know which value is correct before you collapse them into one.

---

## 6. Suspected bugs — unconfirmed, handle separately

These came from reading, not running. Treat each as a question, not a defect report. Verify before
changing anything, and if a fix would alter numbers that have already been collected, flag it rather
than silently changing behaviour.

### Bug 1 — `zmax` default drift (medium confidence, needs a human decision)

`clearance_metric_3d.py:2033` has `--zmax` default `1.4`. `seed_from_clearance.py:52` has
`SeedMetricConfig.zmax = 1.23`. The dataclass docstring claims "Defaults mirror clearance_metric_3d.py's
argparse so the seed route matches the tool's eps* geometry" — which is no longer true. Commit
`1bf2dc5` ("Lower the seed grid ceiling to 1.23 m") changed one and not the other.

Either the standalone tool and the in-rollout seed builder are measuring different volumes, or the
1.23 ceiling is deliberately seed-only and the docstring is stale. **Do not guess.** Ask, and record
the answer in the docstring either way. This blocks Stage 5.

### Bug 2 — `OCC_HEIGHT` is the wrong object (high confidence, low impact)

`reachability_view.py:32` sets `OCC_HEIGHT = 0.2542`. That is the milk-box asset's long-axis extent
(`benchmark/assets/objects/038_milk-box/model_data2.json` → `0.25418`). The occluder is now
`029_olive-oil` id 3, whose extent is **`0.30542`**
(`benchmark/assets/objects/029_olive-oil/model_data3.json`).

`swept_volume_3d.py:206` uses it to draw the occluder wireframe, so that figure shows the obstacle
~5 cm (17%) too short, and `reachability_view.py:120` and `:165` print "milk box" in the figure text.
This is display-only — no metric reads `OCC_HEIGHT` — so it does not invalidate any measurement.

Suggested fix: move the constant to `lib/scene_constants.py` as `OCC_HEIGHT`, set it from the
olive-oil extent, and update the three label strings. Worth confirming the asset scale is 1.0 before
trusting the raw extent (`analyze_occluder_visibility.py:6` says scale 1.0).

For contrast, `OCC_HALF_FOOTPRINT = 0.04` **is** correct — olive-oil id 3 is 0.0795 × 0.0796, so half
is 0.0398. Leave it alone.

### Bug 3 — cache key collapses when the scene signature throws (medium confidence)

`analyze_occluder_visibility.py:2537–2544` (`_get_approach_seed`):

```python
try:
    _sig = [tuple(np.round(np.asarray(self.target_obj.get_pose().p, float), 3))]
    for _o in (getattr(self, "occluders", None) or []):
        _sig.append(...)
    scene_sig = tuple(_sig)
except Exception:
    scene_sig = None
key = (tag, scene_sig)
```

If the pose read throws, `scene_sig` becomes `None` and the key degenerates to `(tag, None)` — which
is shared by every scene. The comment two lines above says the scene signature was added specifically
to fix a "stale-seed correctness bug" where "the env-level cache reused the FIRST scene's seed for
every later episode". The `except` path silently reinstates exactly that bug.

`_get_carry_seed` has the same pattern at `:2714–2719`.

This only fires if the pose read throws, which may never happen in practice — that is why confidence
is medium, not high. Suggested fix: on the exception path, skip the cache entirely (build fresh,
don't store) rather than caching under a null key. Cheap, and it fails safe.

### Bug 4 — dead code (high confidence, already covered)

`_execute_actions_via_plan_and_replay` (45L) and `_pick_side_grasp_id` (3L) have zero references.
Handled in Stage 3c. Listed here only so the count is complete.

### Bug 5 — broad exception swallowing (low confidence, informational)

`analyze_occluder_visibility.py` has 20 `except Exception` handlers, seven of which are
`except Exception: pass`. Most are defensible cleanup (`del ik` / `empty_cache()` in a `finally`) or
deliberate "never let seeding break the expert" guards, which is a reasonable policy for a research
harness. Not proposing a change. Noting it because if a future seed bug appears to vanish without
explanation, these are where it went — `:99`, `:106`, `:2344`, `:2649`, `:2659`, `:2814`, `:2819`.

### Non-bugs, checked and cleared

- `_prune_empty_topdirs` (`:110`) uses `rmdir()`, which only removes empty directories and leaves
  anything unexpected in place. Safe.
- No mutable default arguments anywhere in `bench_script/`.
- After Stage 1, `reachability_map._build_ik_solver` / `_solve_grid` still have a live consumer
  (`clearance_metric_3d.py`). Not orphaned.

---

## 7. Verification after every stage

There is no test runner configured for `bench_script/`, so verification is manual. Minimum bar:

1. **Import check.** Every remaining top-level script must still import. From `benchmark/` with the
   env sourced, run each script with `--help` — that exercises the full import chain and the argparse
   block without touching the GPU.
2. **Unit tests.** From `bench_script/`, `python -m checks.test_obstacle_set` and
   `python -m checks.test_ring_config`. Both are CPU-only and fast. `test_ring_config` is the
   important one after any `occluder_ring.py` move: it asserts the formation is byte-identical per
   `(seed, offset-spec)`, which is what guarantees the measured scene equals the rolled-out scene.
3. **Seed smoke test.** `python -m checks.smoke_test_seed_2a` from `bench_script/` after Stages 2–4. It needs a GPU
   and a real scene, but it is the only check that the seed pipeline still produces a correctly
   shaped tensor.
4. **No cycles.** After Stage 2, confirm nothing under `lib/` imports a top-level script:
   `grep -rn "^from \(analyze_\|clearance_\|reachability_\|visualize_\|seed_from\)" lib/` must return
   nothing.
5. **Size check.** `wc -l bench_script/*.py bench_script/lib/*.py bench_script/task/*.py` — nothing
   over 1000, and note anything over 500 in the commit message with a reason.
6. **One A/B cell.** After Stage 3 (the riskiest stage), run a short
   `scripts/validation/run_approach_mode_ab.sh` with a small seed count in both `direct` and `seed`
   mode and confirm the per-episode records still populate and the seed firing rate is unchanged from
   the last run on `83e392f`. A pure refactor must not move that number.

---

## 8. Expected end state

| | Before | After |
|---|---|---|
| Files in `bench_script/` | 22 | ~18 top-level + 12 in `lib/` + 7 in `task/` |
| Largest file | 3476 L | ~610 L (`task/placement_mixin.py`), or ~500 with §3d |
| Files over 1000 L | 2 | 0 |
| Lines deleted outright | — | ~1438 (4 files + 48 L of dead methods) |
| Function-local imports to break cycles | 6 | 0 |
| `# noqa: E402` markers | 12 | only the genuine `setup_paths()` ones |
| Config surfaces for the metric | 3 | 1 |
