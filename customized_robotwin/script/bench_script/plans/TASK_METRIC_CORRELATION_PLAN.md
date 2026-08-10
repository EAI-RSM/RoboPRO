# Plan 2 of 2 — port the metric to stock RoboPRO tasks and correlate it with pi05 outcomes

Executable plan for Codex. Written 2026-07-30 against branch `codex/bench-script-refactor`,
HEAD `21a64d7`.

**Prerequisite decision (user-waived 2026-07-31):** accept the current six aligned validation
scenes as sufficient evidence for using scalar `eps_geom` in this study. The earlier `n >= 8`
threshold was arbitrary and is not a remaining gate. Route fidelity is **not** established and is
diagnostic only; no claim or implementation in this plan depends on the representative geometric
path matching CuRobo's path.

This waiver does not establish transfer to stock-task geometry. Before the main rollout campaign,
run small aligned geometric-vs-gated calibration checks for the selected stock task's relevant
leg/orientation families. If rank fidelity is poor there, use the orientation-grouped gated
fallback (`label_volume`, `qvol=None`) and rescope the primary study to one task.

Stage A below is independent of that gate and can start immediately.

---

## Goal

Compute the clearance metric **per leg** across a nominal waypoint chain for
`put_cup_on_coaster`, and estimate its association with RoboPRO HSR for pi05. This is an
**association study**, not a held-out prediction study, and its inference scope is this one named
task.

**Locked user choices (2026-07-31):** one task (`put_cup_on_coaster`); episode-level hard success
rate (HSR) as the primary outcome; `eps_geom_min` as the primary predictor; the grasped object's
geometry has no effect on metric construction or normalization; no visibility measurement or
covariate; and the user supplies final sample size `N`. The analysis reports 95% CIs for that
supplied sample rather than selecting `N` from a power or precision target.

For this study the user defines, per episode,
`hard_success = task_success and not collision_metrics.is_collision`; HSR is the arithmetic mean of
that binary indicator over the declared sample. This operational definition governs the analysis
even though `docs/manifest.json` uses the expansion “holistic success rate with collision penalty.”

**Why not the occluder scene.** pi05 half-reaches for the bottle and collapses under occluders
there: it is out of distribution, so its failures carry no information about scene difficulty. A
correlation needs the policy on tasks it was trained for.

**Why per leg.** Task motion is ultimately expressed as pose-to-pose moves, so leg-level clearance
preserves where in a nominal task path the geometric constraint occurs. The metric kernel is
task-agnostic, but task roles and canonical waypoint construction are not; those require explicit
adapters for each selected task.

**Read first:** `agent-memory/tool_vla_pi05_port.md`, `agent-memory/tool_clearance_metric.md`,
`agent-memory/domain_bench_script_layout.md`, `agent-memory/status_current.md`,
`GEOMETRIC_EPS_VALIDATION_PLAN.md`.

---

## 0. Ground rules

**Do not move any already-collected number.** Additive only. The gated metric path and the Phase 4
A/B machinery must be untouched.

**Do not put the metric in the rollout loop.** Metric and rollouts are computed independently and
joined offline by an immutable `scene_id` plus exact scene-fingerprint equality (§C). Coupling them
makes both harder to re-run and risks the metric build perturbing the RNG stream the scene is drawn
from.

**Do not chase VLA success rate.** Report it; never tune the scene against it. Any scene selection
criterion must be stated and must be orthogonal to the metric.

**Standing user directives.**
- **Do not "fix" the expert grasp failures.** Stage B exists precisely so the expert is never needed.
- **Do not touch placement** (2026-07-28): `place_actor` / landing-search / object-ejection failures
  are reported and left alone.
- **Do not measure or adjust for visibility in this study.** Visibility tooling is a separate,
  deferred artifact and returns only if the user explicitly brings it back into scope.
- **Do not model the grasped object in the clearance metric.** Holding state remains descriptive
  leg metadata only; it does not change occupancy, clearance, dilation, or normalization.

**Dependency rule holds.** `lib/` imports nothing from a CLI script.

**Script conventions apply unasked** (`agent-memory/feedback_script_conventions.md`): timestamped
run folder, `timings.json`, results under `scripts/validation/results/<topic>/`, legible figures.

**Environment.** Run from `customized_robotwin/` with `source set_env.sh` and
`export ROBOTWIN_BENCH_TASK=bench`. Single GPU (RTX 4080, 16 GB); `GPU_SPEC` must be `0:0`.

**GPU steps are user-owned.** Codex writes code, CPU tests, and the exact command line; the user
runs anything needing SAPIEN/curobo/pi05.

**One stage per commit.**

---

## A. Stage A (GPU, user-owned, AFTER C2) — is pi05 viable as an outcome measure at all?

Conceptually independent of the metric work and of plan 1, but deliberately postponed until the
metric-only Stage C2 distribution review freezes the bucket specification. It is a hard prerequisite
for stage D only; stages B, C, and C2 stand on their own as a metric contribution regardless.

The checkpoint trains on `repo_id="roboreal_lerobot"`
(`policy/pi05/src/openpi/training/config.py`, `pi05_robopro_top_cam_jax`) — RoboPRO **real** data.
So SAPIEN rendering is a distribution shift *even on the correct task*. Running on trained tasks
removes one of two shifts, not both. Two things settle whether a correlation is findable:

1. **Training-task membership confirmed (2026-07-31).** The user reports that the RoboPRO team
   confirmed `put_cup_on_coaster` was included in the `roboreal_lerobot` data used to train
   `mzxuan/robopro_jax_30000`. This clears the membership gate. Keep the provenance distinction
   explicit: the repository/model-card evidence is consistent with the answer (the checkpoint card
   says 80 tasks and this benchmark's 80-task list contains `put_cup_on_coaster`), but it does not
   itself provide the checkpoint's explicit task inventory; the direct confirmation came from the
   team via the user.
2. Run a 10-episode clean-scene `put_cup_on_coaster` pilot (`obstacle_density: 0`) with seeds
   reserved for piloting and never reused in the final analysis. Do not expand to another task in
   this plan.

The current `vla_rollout.py` is hardcoded to Office `put_mouse_on_pad`, its occluder variant, and
Office density 0. Add only the narrow generic plumbing needed here: `--task-name`,
`--bench-subdir`, `--base-config`, instruction-bank lookup, and density/config provenance stamping.
Each rollout record also writes
`hard_success = success and not collision_metrics["is_collision"]`, while retaining both source
fields. Do not introduce a new rollout framework.

**Engineering complete (2026-07-31), valid GPU pilot pending.** `vla_rollout.py` now has an additive
`--scene task` path while preserving the existing Office/occluder modes. Stock-task runs require a
pinned `--max-steps`, use the first declared instruction-bank phrase unless `--instruction` is
explicitly pinned, stamp task/subdir/config/DR/checkpoint provenance, and write `task_success`, raw
collision metrics, and derived `hard_success`. Missing or non-Boolean `is_collision` is fatal. The
focused stock/Office/outcome test and surrounding CPU regressions pass. The next validation is the
reserved 10-episode clean-scene GPU pilot below.

**Invalid pilot attempt (2026-07-31):**
`scripts/validation/results/task_metric_vla_pilot/clean_viability/20260731-170002` must not be used
as HSR evidence. Several episodes were labelled successful after one action and produced nearly
empty videos because `put_cup_on_coaster.check_success()` compared a signed XY difference with the
2 cm tolerance. The predicate is corrected to use absolute XY error while preserving the original
per-axis tolerance and open-gripper requirements; this is a task-predicate fix, not a minimum-step
heuristic. The Study task base now also honors the rollout driver's explicit
`countertop_camera` video selection instead of silently falling back to `demo_camera`. Rerun and
visually inspect all ten reserved clean-scene episodes before accepting the Stage A gate. The
focused signed-offset/open-gripper regression, surrounding task-metric/bucket/lib/obstacle/ring
CPU checks, compilation, and diff check pass after these corrections.

Rerun from `customized_robotwin/` with the same reserved pilot seeds; the launcher creates a fresh
timestamped directory, so the invalid attempt remains intact and auditable:

```bash
GPU_SPEC=0:0 bash policy/pi05/vla_occluder_rollout.sh \
  --scene task \
  --task-name put_cup_on_coaster \
  --bench-subdir study \
  --base-config bench_demo_study_clean \
  --seed-start 2000 \
  --num-seeds 10 \
  --clutter-densities 0 \
  --max-steps 600 \
  --replicate 0 \
  --out-dir ../scripts/validation/results/task_metric_vla_pilot \
  --run-type clean_viability
```

**Pass:** every episode has a valid Boolean task-success value and non-null
`collision_metrics.is_collision`, so `hard_success` is computable, and the pilot outcome is
non-degenerate enough to estimate an association. Missing collision metrics are fatal, never
silently treated as collision-free. This pilot gate is diagnostic; it does not choose the final
user-supplied `N`.

**On fail, STOP stage D and report.** Do not tune the scene or substitute a different outcome to
rescue the study. A constant HSR contribution is not a null association; it is no estimable
association.

---

## A2. Stage A2 (CPU, do this FIRST of the metric stages) — scene-generator audit

**Everything downstream assumes the selected task's scene generator is represented faithfully.
The audit below separates confirmed behavior from role, geometry, and provenance gaps that can
produce silently wrong numbers rather than errors.** Resolve the applicable gaps before stage C,
and repeat the selected-task audit whenever scope expands.

### Confirmed working — no change needed

- `lib/scene_build.py::get_env_class` already declares
  `BENCH_SUBDIRS = ["office", "study", "kitchenl", "kitchens"]`, so `bench_envs.study.<task>` loads
  with no loader change.
- `benchmark/bench_task_config/instruction_bank.json` (81 keys) covers **all 20 Study tasks and all
  5 named Office tasks** — zero missing. Stage D's prompts are available.
- `bench_demo_study_clean.yml` and `bench_demo_study_d6..d15.yml` all exist — exactly the density
  ladder stage D needs. Use these as `base_config`; do not write new ones.
- Study clutter registers through the same `clutter_surface_split` → `collision_list` with
  `is_obstacle: True` and a real collision-mesh path, so `scene_obstacle_entries(env, "all")` sees
  it the same way it sees Office clutter.

### A2.1 — task roles are NOT represented by one universal attribute

`scene_obstacle_entries` currently skips actors by `("target_obj", "des_obj")`, but task semantics
are more varied. In particular, `move_book_onto_table` and `move_seal_onto_table` use a table
region/pose as the destination; they are not missing destination actors. Nor does
`_get_target_object_names()` universally encode target, destination, and obstacle roles.

Add a small explicit **task adapter** for every selected task. It must identify the movable target,
the destination representation (actor, procedural region, or pose), and task-owned actors that
remain obstacles. Match actor-backed roles by actor identity/name, with loud ambiguity and absence
errors. Target meshes and every included obstacle mesh are mandatory. A procedural or deliberately
skipped destination does not need a collision mesh merely to satisfy a universal preflight rule.

Add focused tests for the selected adapters, including both an actor-backed destination and a
table-region destination. Do not claim coverage of all 25 tasks until adapters for all 25 exist.

### A2.2 — step-limit provenance, no YAML edit

The study reads `benchmark/bench_task_config/_bench_eval_step_limit.yml`, and all Study task entries
already exist. Do not modify a global step-limit file for this experiment. Supply a pinned
`--max-steps` to every pilot and final rollout and stamp it into each rollout record so episode
budgets are comparable and auditable.

### A2.3 — `random_table_height` is live in the Study base

`_study_base_task.py:135` computes
`self.table_z_bias = np.random.uniform(low=-self.random_table_height, high=0) + table_height_bias`,
whereas `_office_base_task.py:190` hardcodes `self.table_z_bias = 0`. Across
`benchmark/bench_task_config/` 376 configs set `random_table_height: 0` and **one sets `0.03`**.

The metric grid is fixed at z[0.78, 1.23] against a 0.74 table. A moving table under a fixed grid
makes `eps_geom` incomparable across seeds — a per-seed change in what volume is measured, masquerading
as scene difficulty. **Assert `table_z_bias == 0` at build time in `task_metric.py` and refuse to
record the scene otherwise.** Do not silently tolerate it, and do not make the grid follow the table
(that is a bigger change than this study needs).

### A2.4 — Study has a second registration path

`_study_base_task.py:210 add_collision` populates `collision_list` from `scene_obj_info`, separate
from the clutter path, and only 2 of the 20 Study tasks reference it. So obstacle-set composition
varies per task in a way Office does not.

Fix: `task_metric.py` records `n_obstacles` and the obstacle name list per scene, and stage A2
prints a one-line-per-task table at one fixed seed for every **selected** task. Expand the table as
tasks are added. **Any selected task whose obstacle set is empty at a nonzero realized clutter
count is broken for this study — report and drop it.** This table is a deliverable; it is the
evidence that the metric is seeing each scene at all.

### A2.5 — clutter mechanisms: what exists (verified 2026-07-30)

**Random clutter: yes, all four domains.** `clutter_surface_split` is called from
`study:351`, `office:625`, `kitchenl:1261`, `kitchens:914`; office additionally clutters two shelf
levels via `clutter_surface`. Every clutter actor is registered into `collision_list` with
`is_obstacle: True` and a real mesh path, so `scene_obstacle_entries(env, "all")` sees it.

Knobs, all from `random_setting`: `cluttered_table` (default **False**), `obstacle_density`
(default 3), `clean_background_rate` (default **1** — this is a *skip* probability, so the default
suppresses clutter entirely), `obstacle_distribution`, `obstacle_allow_duplicates`.
**Verify the density-ladder configs override `clean_background_rate`**, or every scene comes out
clean and `eps_geom` has no variance.

Pools come from `task_objects.yml` via `get_obstacle_objects_subset`, split short/tall, with
objects already on the scene filtered out: **study 6 short / 4 tall, office 5 short / 8 tall**.

**Custom clutter: exists, office-only, and currently dormant.** `_office_base_task` has a
`handcrafted_clutter` hook (`name -> {ids, weight, scale_mult}`) plus `tall_obstacles_only`.
`_build_handcrafted_clutter` computes scale/radius/z_max directly from each model's
`model_data{id}.json` and `task_objects.yml` scales, so **it works for objects outside the obstacle
pool** (docstring names `038_milk-box`). Three caveats:

- **Nothing sets either flag** — both are read via `getattr`, no config key and no caller anywhere.
  They are attribute-injection hooks, the same pattern as the occluder task setting
  `spawn_occluder` / `num_occluders` before `setup_demo`.
- **Study has no equivalent** — `_study_base_task.py:348` goes straight from
  `get_obstacle_objects_subset` to `clutter_surface_split`. Porting it is small but real work.
- **It controls the POOL, not the POSES.** Placement stays random. Anything needing *controlled
  geometry* (clutter at specified positions, like the occluder ring) is the occluder-task pattern —
  spawn actors directly and append to `collision_list` — not this hook.

The §D primary design (fix clutter count, vary placement) needs only stock random clutter. Do not
port `handcrafted_clutter` for this plan.

### A2.6 — asset preflight: resolve meshes by selected task role

The metric consumes **collision meshes, not point clouds** — `_load_collision_mesh` returns
`(vertices, faces)` and `occluder_slice_polys` calls `mesh.section(...)`, which needs faces.

Audited every model id in both obstacle pools against the two paths `clutter_surface_split` uses
(`objects/{name}/collision/base{id}.glb`, and the objaverse `coacd_collision.obj` fallback):
**79 meshes found, 0 missing.** All resolved as glb; the objaverse fallback was never needed.
Target/destination meshes for the provisional tasks exist: `021_cup` (13 ids), `019_coaster` (1),
`100_seal` (6), `047_mouse` (3), `048_stapler` (7).

Sweeping all 79 object dirs, **11 lack `collision/base*.glb`**. Ten are furniture/articulated
(`015_laptop`, `036_cabinet`, `044_microwave`, `121_wall-shelf`, `122_cabinet_nkrgez`,
`122_file-holder`, `124_fridge_hivvdf`, `125_cabinet_tynnnw`, `135_dish-rack`, `cube`) — none are in
the clutter pools, and the articulated ones are already out of scope.

The named pen tasks use **`058_markpen`**, not `010_pen`, and
`058_markpen/collision/base0..5.glb` exist. Do not exclude `put_pen_in_box`, `put_pen_in_pencup`, or
`move_pen_to_box` on the earlier mistaken asset claim.

(`cube` was a false alarm: `put_phone_next_to_cube` uses `073_rubikscube`, which is in the office
short pool with a mesh, plus a `create_box` destination.)

**Deliverable: a preflight check** in `task_metric.py` that uses the selected task adapter and
resolves the movable target plus every actor included as an obstacle. Actor-backed destinations
that the metric includes must also resolve; procedural or intentionally masked destinations are
recorded by role and do not require a fictitious mesh. Refuse the task on any required miss.

---

## B. Stage B (CPU) — planning-free waypoint harvest

**Do not run the expert to get the waypoints.** Two reasons, and the second is the important one:

- `play_once` needs curobo and has a history of dying in the feasibility gate
  (`put_mouse_on_pad.grasp_actor_from_table` → `IndexError`).
- Seeds where the expert fails are disproportionately the **tight** ones, so harvesting via the
  expert truncates the predictor exactly where the signal is. That is selection bias, not just
  inconvenience.

Do not assume the task pose helpers are pure geometry. `Base_Task.get_grasp_pose()` calls the
planner-backed `choose_best_pose()`, while `get_place_pose()` and `move_by_displacement()` read live
object/EE state. A universal call-through harvester would therefore either invoke planning or
quietly construct the wrong chain.

Add `lib/waypoints.py` with an explicit adapter for each selected task:

```
canonical_waypoints(env, task_name) -> list[Waypoint]  # xyz, quat, kind, arm, gripper_state
```

- Per selected task, encode a **nominal canonical task path** from initialized actor poses and task
  constants. It is not the exact expert-realized path and must not be described as one.
- Start from the arm's initialized EE pose. Resolve target and destination through the same task
  adapter used by scene preflight.
- `kind` ∈ {`pre_grasp`, `grasp`, `lift`, `carry`, `pre_place`, `place`} — stage D needs this to
  correlate per leg rather than only on the aggregate.
- `gripper_state` ∈ {`empty`, `holding`} per leg, so stage C knows which legs carry an object.
- Track the nominal gripper pose forward explicitly; do not call helpers whose result depends on a
  previous live motion having occurred.

**Validation.** CPU tests verify deterministic adapter output, task-role resolution, leg labels,
and state transitions. A small expert-parity comparison may log real `Action.target_pose` values
for the selected task, but it is a **user-owned GPU/SAPIEN diagnostic**, not a CPU equality test or
a promise of exact equality. Record the deviations and decide whether the nominal path remains a
scientifically defensible proxy before the final campaign. Do not commit temporary instrumentation.

### Why exactly one task — the methodology

Tasks answer **two different questions**, and conflating them is what makes the set too big or too
small.

- **The metric core is task-independent.** `geometric_eps` takes `(start_xyz, goal_xyz)` pairs and a
  scene; it reads `env.collision_list` and never learns what the task is. Adding tasks buys nothing
  for pipeline validation *per se* — what varies is the **scene plumbing** (§A2 found four silent
  differences between Study and Office).
- **Within-task precision comes from seeds.** A seed costs one rollout plus one CPU metric. Another
  task would require another A2 audit, role adapter, canonical-waypoint adapter, and calibration,
  while changing the inference population.
- **The scope is intentionally narrow.** The result may say only that `eps_geom` is associated with
  HSR on `put_cup_on_coaster`. It must not be generalized to RoboPRO tasks as a class.

**Task locked by the user:** `put_cup_on_coaster`. Its checkpoint training-set membership was
confirmed by the RoboPRO team via the user on 2026-07-31. No second task or cross-domain validation
is part of this plan.

**Selection criteria, in priority order:**

1. **In the VLA's training distribution.** Dominant, and now resolved: the RoboPRO team confirmed
   the selected task's membership via the user on 2026-07-31.
2. **Study plumbing** — audit `_study_base_task`, live `table_z_bias`,
   `place_actor`/`get_position_limits`, and `scene_objs` for this task.
3. **`eps_geom` variance.** `get_position_limits(boundary_thr=0.15, robot_reach_thr=0.6,
   arm_x_pose=0.15)` on a 1.2 x 0.7 m table gives a placement region of about **0.60 x 0.40 m**, and
   target *and* destination are both drawn from it with only a `col_thr` (0.10–0.15 m) minimum
   separation. Transit length therefore varies from ~0.1 m to ~0.7 m across seeds — good spread. The
   `add_prohibit_area` paddings (0.05 target, 0.1 destination) block clutter at the endpoints but
   not mid-corridor once the two are far apart, so clutter can actually intervene.
   *(Derived from office table dims + `scene_constants.TABLE_XLIM/YLIM`, not measured on a live
   Study scene — confirm in A2.)*
4. **Grasp feasibility.** The occluder lesson: infeasible grasps kill episodes upstream of anything
   the metric touches and add pure noise to both arms. Favour chunky objects; avoid thin and
   large-flat.
5. **No payload model.** Per the user, the grasped cup does not alter the metric on holding legs.
6. **Existing mileage.**

### Selected task

| Task | Domain | Why |
|---|---|---|
| `put_cup_on_coaster` | Study | User-approved recommendation: atomic, chunky target, actor-backed destination, and exercises the Study placement path. |

Office and `structure_mask` work are outside this one-task plan.

**Tempting shortcut, rejected:** setting `SPAWN_BACK_FURNITURE = False` for office would make it
match the scene every `eps_geom` number to date was measured in (the occluder task sets exactly that at
`task/occluder_task.py:35`). But it changes what the VLA sees, which cuts directly against the
in-distribution requirement that is the entire reason this plan exists. Not free — do not do it
silently. If it is done, it is a disclosed scene-selection decision.

**Deliberately NOT selected:** `move_book_onto_table` and `move_seal_onto_table` are the two atomic
Study tasks missing `des_obj` (the §A2.1 cases). Tempting as a live test of that fix, but their
destination is *the table itself*, so the place endpoint is a large weakly-defined region rather than
a pose — bad for a per-leg metric, and it would confound the correlation with "where did we decide
the destination is." Cover A2.1 with the unit test over all in-scope tasks instead.

### Future task scope — explicitly outside this plan

**Do not implement adapters for these now.** If the user later expands scope, the 15 atomic Study
tasks are the next eligible pool. The
three named pen tasks are eligible because their actual `058_markpen` collision meshes exist. All 20 Study tasks are
free of drawers, shelves, fileholders, cabinets and fridges (verified by grep), but 5 are
**compositional** per `TASKS.md` — `empty_box`, `move_cup_put_pen_in_cup`, `move_cups_into_box`,
`move_seal_cup_next_to_box`, `move_seal_onto_book`. Those run multiple pick-place cycles and
reassign `target_obj` mid-episode, so the waypoint chain is not a single grasp→place and the
"which object is the target" question changes partway through. Leave them out until tier 1 works.

The 5 non-articulated Office tasks are —
`put_mouse_on_pad`, `put_book_on_book`, `put_phone_next_to_cube`, `put_phone_on_holder`,
`put_stapler_next_to_mouse`. `put_mouse_on_pad` is the one with existing rollout mileage, so it is
the natural first cross-domain check.

The 5 compositional Study tasks are also deferred. Handling them means a
per-cycle waypoint chain with a changing target, and a decision about whether `eps_geom` aggregates
within or across cycles. Do not attempt it in this plan.

**Out entirely:** everything articulated. Articulated entries carry a `"link"` key and are dropped
by `scene_obstacle_entries` (the rigid pose-the-whole-mesh transform would place them wrong), so
the metric reads those scenes as wide open and the numbers would be **silently** wrong. Do not
attempt to fix that here.

**Note the interaction with A2.1:** of the 5 tasks missing `des_obj`, three are compositional and
already out of tier 1, but **`move_book_onto_table` and `move_seal_onto_table` are atomic and in
tier 1**. They are the concrete cases the A2.1 fix has to handle.

---

## C. Stage C (CPU) — `script/bench_script/task_metric.py`

The driver. Per scene instance: build the scene, produce canonical waypoints, run the metric over
consecutive waypoint pairs, and write one record.

- Build with `build_cfg(task_name, base_config, seed, dr_overrides, mode="measure")` — the existing
  no-planning measurement mode. **Nothing new is needed in `scene_build.py`.**
- Resolve the stock target through the selected adapter. Support the repository's actual patterns
  (`target_name`/`target_id`, `mouse_id`, or an adapter-provided equivalent); never infer a target
  by taking the first collision actor.
- Legs are consecutive canonical waypoint pairs. Build **one** geometric volume per scene and use
  it for every leg. Apply the same calibrated target-mask policy to empty and holding legs.
  `gripper_state` is recorded only as descriptive leg metadata.
- Define `eps_geom` as the raw bottleneck obstacle clearance. The grasped cup does not change the
  obstacle field, dilate clearance, create an effective-clearance value, or change normalization.
  If `rho_geom` is retained as a secondary output, use the same fixed gripper reference radius for
  every leg and record that constant.
- Current `geometric_eps()` accepts XYZ endpoints and ignores waypoint quaternions. Keep that scope
  explicit. Before the main study, run small aligned stock-domain checks for the selected task's
  relevant leg/orientation families; do not generalize the occluder-scene rank result to unseen
  orientations without evidence.
- Office `structure_mask` work is deferred because the selected task is in Study.
- **Record every leg AND the primary `eps_geom_min` aggregate. Do not discard the leg vector.** The
  `pre_grasp`-has-no-headroom / `carry_transit`-has-all-of-it finding is direct evidence that a min
  can be dominated by a leg nothing ever fails on. The user has nevertheless selected
  `eps_geom_min` as primary; stage C keeps the vector for transparent diagnostics.
- Do not invoke `lib/visibility.py`, construct a clean-denominator scene, render a visibility
  artifact, or add a visibility field to the study record.

Assign a unique `scene_id` to every intended metric/rollout pair. Its canonical fingerprint must
include task, seed, bench subdir, base config, all DR settings, configured `obstacle_density`, actual
clutter count, clutter identities and poses, target/destination identities and poses, instruction,
checkpoint, code/config version, and replicate. Stamp the same canonical fingerprint in metric and
rollout records. The offline join requires exactly one record on each side and byte-for-byte equal
fingerprints; duplicates, missing matches, or mismatches are fatal. `(task, seed)` is not a key
across densities or reruns.

`records.jsonl`, one line per `scene_id`: `scene_id`, `scene_fingerprint`, `task`, `seed`,
`replicate`, `bench_subdir`, `base_config`, `dr_settings`, `obstacle_density`, `clutter_count`,
clutter/role pose provenance, instruction/checkpoint/version stamps, `arm`, per-leg
`[{kind, gripper_state, eps_geom, rho_geom, merged, reason}]`, `eps_geom_min`, `n_free`, and
`wall_seconds`. Plus `timings.json`.
Generated records remain in a timestamped gitignored results directory; commit code and a compact
provenance/result summary, not `records.jsonl`.

**Run stage C before any rollouts.** Stage C2 makes the spread review a formal user gate rather than
an informal glance at the records.

---

## C2. Stage C2 (CPU, USER DECISION GATE) — visualize the metric distribution first

Add `script/bench_script/analyze_metric_distribution.py`. It reads **metric records only**; it must
not accept, discover, or join rollout/HSR records. Run it over the intended scene-generation
protocol before implementing or freezing any bucket-based correlation summary.

**Frozen metric-only pilot:** generate 100 d10 scenes with reserved seeds 1000--1099. This pilot
pool exists only to expose the `eps_geom_min` distribution for the bucket decision; it is separate
from the user-supplied final rollout `N`, must not be reused for rollout outcomes, and is excluded
from HSR or association inference.

**Frozen bucket decision (user-approved 2026-07-31):** define the bucket variable as
`rho_geom_min = eps_geom_min / 0.03 m`, using the same fixed gripper reference radius already
stamped in the metric records. This rescaling does not replace `eps_geom_min` as the continuous
primary predictor. It defines four outcome-blind clearance bands:

| Clearance label | `rho_geom_min` interval | Equivalent `eps_geom_min` interval | d10 pilot count |
|---|---:|---:|---:|
| `very_low_clearance` | `< 2.5` | `< 0.075 m` | 17 |
| `low_clearance` | `[2.5, 4.0)` | `[0.075, 0.120) m` | 35 |
| `medium_clearance` | `[4.0, 5.0)` | `[0.120, 0.150) m` | 30 |
| `high_clearance` | `>= 5.0` | `>= 0.150 m` | 18 |

Intervals are lower-closed and upper-open except for the unbounded final interval. A value exactly
at 2.5, 4.0, or 5.0 enters the higher-clearance bucket; identical metric values are never split;
and `+inf` enters `high_clearance` as a right-censored/top-tied observation. Store both rho and
equivalent metre boundaries plus the fixed 0.03 m denominator in `bucket_spec.json`. Formal output
uses clearance labels, not `easy`/`hard`: outcome-difficulty language is permitted only as a later
interpretation if the HSR results support it. These bands are for HSR summaries; primary Spearman
inference remains on continuous `eps_geom_min`.

Required outputs in a timestamped results directory:

- `eps_geom_min_distribution.png`: finite-value histogram plus an ECDF/rug panel; show the
  `+inf` count in a separate labeled top-censored category rather than forcing infinity onto a
  numeric axis.
- `eps_geom_by_leg.png`: one aligned distribution panel per canonical leg, preserving ties and
  separately reporting `+inf` counts.
- `eps_geom_min_by_scene.png`: sorted scene points labeled by seed/`scene_id`, with configured
  density, realized clutter count, and nonfinite values visible.
- `distribution_summary.json`: total n, finite n, `+inf` n, unique/tied counts, finite min/max,
  standard quantiles as **descriptive reference values only**, per-leg summaries, and realized
  clutter-count frequencies.
- `timings.json` and the exact source-record/provenance path.

The script must not generate default buckets, outcome columns, HSR estimates, p-values, or a
recommended cutoff. After reviewing these artifacts, the user chooses the number of buckets,
boundary rule/values, and tie/`+inf` assignment. Record that frozen choice in a small committed
`bucket_spec.json` (schema/version, source distribution run, boundaries, closure convention,
tie policy, `+inf` policy) **before any rollout outcomes are loaded into the correlation analysis**.

**The user decision portion of this gate is cleared.** Before loading any outcome data, encode the
frozen choice above in `bucket_spec.json` and test complete, exactly-once assignment. Stage D and
bucket-specific analysis remain unimplemented until that artifact exists.

---

## D. Stage D — the correlation run

### Design — this is what decides whether the result means anything

**Interpretation boundary.** This study does not measure visibility. The result is an association
between `eps_geom` and HSR under the selected clutter-generation protocol, not a causal or
visibility-adjusted effect of geometric clearance. Visual changes that co-vary with obstacle
placement remain part of the association and must be disclosed.

**Primary design — hold the realized clutter count fixed, vary placement.** Pick one configured
`obstacle_density` after the pilot, then retain or generate scenes with one declared realized
`clutter_count`. Configured density alone does not guarantee the realized count. `eps_geom` varies
because placement varies while visual object count is controlled; record both configured density
and realized count.

**Secondary, disclosed as confounded.** Sweep density d6–d15. Report separately and say plainly that
it mixes the two mechanisms.

**Run size:** the user supplies final `N`; the implementation does not choose it. Record that `N`
before the final campaign and do not extend the run based on the observed result. Pilot seeds are
excluded from final inference. Rollouts use the narrow generic `vla_rollout.py` plumbing from stage
A; metric records come from stage C and join by `scene_id` plus exact fingerprint equality.

**Long-run collection engineering (implemented 2026-07-31; GPU validation still belongs to the
clean pilot gate):** the user's declared scale is 1000 rollouts at each of d6, d10, and d15
(3000 total). `--rollouts-per-density 1000 --clutter-densities 6,10,15` commits episodes in the
temporal order d6 → d10 → d15 → repeat, retrying an unstable scene at the same scheduled density so
each density still ends with exactly 1000 usable records. This records final `N=3000` before the
campaign and prevents `--num-seeds 1000` from ambiguously producing the wrong total.

Each completed rollout is first written as an individually atomic, fsynced
`episodes/episodeNNNNNN.json`, then appended and fsynced to `records.jsonl`. The per-episode files
are authoritative: `--resume-dir <timestamped-run>` validates the immutable hashed `config.json`,
reconstructs `records.jsonl` if power failed between those two writes, rejects density-order or
config mismatches, skips every committed episode, and continues at the next episode/seed. A partial
active video or log may be overwritten on retry; no committed episode record or finalized video is
overwritten. Records retain the nested collision data plus a flattened `records.csv`, exact scene
ID/fingerprint source, scene/rollout code hashes, instruction/checkpoint/config provenance,
task/hard outcomes, acting arm, realized clutter count, step counts, wall/phase timings, policy
errors, and video path/camera.

`analyze_vla_rollouts.py` is the outcome-only descriptive report. The collector calls it after
every 10 committed episodes and at clean exit or handled teardown. It atomically regenerates
`summary.json` (overall and per-density HSR with Wilson 95% intervals), `records.csv`,
`running_hsr.png`, `outcome_diagnostics.png`, and `report_state.json`. A plotting failure is loud
but does not sacrifice subsequently collectible raw episodes. All derived files can be regenerated
from the atomic records at any time.

The same report pass writes `metric_scene_manifest.jsonl`, an outcome-free projection containing
only exact committed scene-regeneration inputs and expected identity. `task_metric.py
--scene-manifest ...` accepts only that closed schema and refuses any scene-ID, fingerprint,
code-version, arm, or realized-clutter mismatch.

**Integrated post-processing correction (2026-08-07).** Deferring metric reports to an unrelated
manual command did not satisfy the required long-run workflow. The metric still must not run in
the policy action loop, but future task rollouts use one sequential command: policy collection
tears down pi05/SAPIEN first, then `--postprocess-metrics` automatically starts or resumes
`task_metric.py --rollout-run <run>`. Re-entering a fully collected rollout skips policy startup
and proceeds directly to metric post-processing.

Metric post-processing is now independently crash-safe. `<run>/metric_postprocess/episodes/`
contains one atomic, fsynced metric record per rollout episode; those files are authoritative.
The immutable hashed metric config binds the rollout config, outcome-free manifest hash, frozen
bucket-spec hash, metric settings, reach cache, and metric code version. Resume repairs
`records.jsonl` from atomic episode files, skips committed metrics, and refuses configuration or
scene-identity mismatches. Existing rollout videos suppress redundant initialized-scene PNGs.

After every 10 committed metrics and at handled interruption, the post-processor regenerates in
that same fixed directory: all three metric-distribution figures, joined JSONL/CSV, provisional
HSR-by-clearance and metric-by-outcome figures, `video_index.json`, and idempotent non-destructive
`videos_by_clearance/<clearance_bucket>/<hard_outcome>/` symlinks. Intermediate joins require the
metric set to be an exact fingerprint-valid subset of rollout records and are explicitly marked
`provisional`; the expensive bootstrap is deferred. Completion switches to the original strict
one-to-one join and runs the frozen 10,000-resample final inference. Finalized source videos remain
in `hard_success/video/` and `hard_fail/video/`; bucket views never move or relabel them.

### Analysis — `script/bench_script/analyze_metric_correlation.py`

**Implemented.** It enforces the one-to-one join below, writes complete joined JSONL/CSV data,
per-density and secondary pooled Spearman bootstrap summaries, Wilson HSR summaries, figures, and
non-destructive clearance-bucket × hard-outcome video indexes.

**Reporting hierarchy frozen by the user (2026-08-07):** d6, d10, and d15 are separate primary
analyses and are always written in that order under `by_density/d6`, `by_density/d10`, and
`by_density/d15`. Each directory contains its own metric-distribution summary and three metric
plots, joined JSONL/CSV, association summary, HSR-by-clearance plot, and metric-by-outcome plot.
The mixed-density analysis is retained only as an explicitly labeled secondary pooled result; it
must not be called the primary association.

- The primary episode outcome is
  `hard_success = task_success and not collision_metrics.is_collision`; HSR is its sample mean.
  The primary predictor is `eps_geom_min`, the minimum `eps_geom` across every canonical leg in the
  scene. The population is `put_cup_on_coaster` scenes under the declared clutter protocol.
  Per-leg effects, density sweeps, and alternate aggregators are exploratory and need labeled
  multiplicity handling.
- Load the user-approved `bucket_spec.json` produced after Stage C2. Refuse to infer, optimize, or
  silently replace bucket boundaries from outcome data. Derive
  `rho_geom_min = eps_geom_min / 0.03 m` and validate complete, exactly-once assignment of every
  value, including boundary ties and `+inf`, before computing HSR by bucket. Cross-check assignment
  against the equivalent metre boundaries stored in the spec.
- Keep `rho_geom = 1` as a diagnostic only; the frozen bucket boundaries are 2.5, 4.0, and 5.0.
- Report HSR separately for d6, d10, and d15 with two-sided 95% Wilson intervals. Report pooled HSR
  secondarily and label it as mixed-density.
- Within each density, the primary association statistic is Spearman correlation between
  `eps_geom_min` and the binary `hard_success` indicator. Treat `+inf` values as a shared top rank.
  Report a two-sided 95%
  episode-bootstrap CI, resampling complete joined episode records so metric, success, and collision
  fields stay paired. If a bootstrap resample has a constant outcome or predictor, record it as
  undefined and report the valid-resample fraction; never coerce it to zero. The same statistic
  over all densities is a secondary pooled analysis because it mixes density and scene placement.
- Show HSR by the frozen `rho_geom_min` clearance buckets for interpretation, with Wilson 95%
  intervals. All per-leg and alternate-predictor associations are exploratory.
- Treat `+inf` as a right-censored/top-tied clearance observation, not as missing. Preserve
  high-clearance scenes in counts and outcome summaries; use a top-tied rank or a predeclared
  censored/sensitivity treatment rather than dropping the nonfinite subset.
- `eps_geom >= eps_gated` is a one-sided approximation error, but that fact alone does **not** fix
  the direction of association bias. Do not claim it necessarily attenuates correlation toward
  zero; report the validation evidence and state that the bias direction is unknown.
- **Report the expert-independence of the sample.** Because stage B never runs the expert, no seed
  is dropped for expert failure and the predictor is not truncated. Say so — it is a design strength
  worth one sentence.

---

## E. Verification

No test runner is configured. The bar, in order:

1. `python script/bench_script/task_metric.py --help`,
   `python script/bench_script/analyze_metric_distribution.py --help`, and eventually
   `python script/bench_script/analyze_metric_correlation.py --help` — full import chain, no GPU.
2. `python script/bench_script/test_lib_env_api.py` — the AST check that every `env.<method>()` call
   in `lib/` resolves in `task/`. New `lib/` code must not break it. This check exists because a
   by-name refactor scan deleted a duck-typed method and voided a week of runs.
3. `test_ring_config.py`, `test_obstacle_set.py` — must pass unchanged.
4. `vla_rollout.py --help`, `clearance_metric_3d.py --help`, `python -m lib.seed_from_clearance --help` — proves
   nothing additive broke the existing callers.
5. New focused tests: selected task-role adapter (§A2.1), `table_z_bias == 0` (§A2.3), canonical
   waypoint determinism/state transitions (§B), identical empty/holding metric treatment,
   distribution summaries/plots with finite, tied, and all-`inf` inputs; proof that the Stage C2
   script rejects outcome-bearing records; bucket-spec boundary/tie/`+inf` assignment; confirmed
   HSR calculation and 95% CI; scene-fingerprint stability; and strict one-to-one
   offline join rejection for missing, duplicate, or mismatched records.
6. **One task, 2 seeds, end to end on CPU** before any rollout campaign: `records.jsonl` has two
   complete lines with a non-empty per-leg vector.

---

## F. Deliverables

| # | Artifact | Stage |
|---|---|---|
| 1 | Training-list verification + 10-episode clean-scene HSR pilot for `put_cup_on_coaster` | A |
| 2 | Explicit target/destination/obstacle adapter + focused tests for each selected task | A2 |
| 3 | Pinned `--max-steps` accepted and stamped by generic rollout plumbing | A/A2 |
| 4 | `table_z_bias == 0` build-time assert (A2.3) | A2 |
| 5 | **Per-task obstacle-set table** at one fixed seed: `n_obstacles` + names for every selected task | A2 |
| 5b | Confirmation that the density-ladder configs override `clean_background_rate` (A2.5) | A2 |
| 5c | Role-aware asset preflight refusing any missing required target/obstacle mesh (A2.6) | A2 |
| 5d | Office `structure_mask` explicitly deferred; no Office adapter in this plan | A2 |
| 6 | `lib/waypoints.py` + canonical adapter tests; optional GPU expert-parity diagnostic | B |
| 7 | `task_metric.py` + gitignored timestamped records + committed provenance/spread summary | C |
| 7b | Canonical `scene_id`/fingerprint schema shared by metric and generic VLA rollout | C |
| 7c | Metric-only distribution script + three figures + `distribution_summary.json` | C2 |
| 7d | User-approved, committed `bucket_spec.json`; implementation pauses until approval | C2 |
| 8 | `analyze_metric_correlation.py` + the joined analysis + figures | D |
| 9 | A stated association verdict with n, CI, and approximation limits — including if null | D |
| 10 | `agent-memory/` updated: `tool_vla_pi05_port.md`, a new note on the task-metric port, `status_current.md` rewritten | — |

---

## G. What NOT to do

- **Do not skip stage A2, and do not assume one domain's scene generator matches another's.**
  Study differs from Office in several task-role and scene-plumbing details (§A2). The same audit is required
  before KitchenL/KitchenS, and before any task outside the tiers in §B.
- **Do not port articulated tasks** (§B). The metric would read them as open space.
- **Do not start with the compositional Study tasks** (§B tier 3).
- **Do not run the expert to harvest waypoints** (§B). Selection bias, not just fragility.
- **Do not put the metric in the rollout loop** (§0).
- **Do not chase or tune against HSR** (§0).
- **Do not add the occluder ring to these scenes.** It reintroduces the exact out-of-distribution
  scene this plan exists to escape.
- **Do not invent default buckets.** Visualize the metric-only distribution, then use the user's
  frozen `bucket_spec.json` (§C2/D).
- **Do not report only the min across legs.** Keep the vector; the aggregator is an analysis choice
  with evidence against the obvious default.
- **Do not rename eps\*.** Call the geometric quantity `eps_geom`, never plain `eps*` — different
  construct, and the existing notes define `eps*` as the arm-followable one.
- **Do not extend the run to rescue a weak result.** Use the user-supplied final `N`, report the 95%
  CI, and do not add seeds until something crosses 0.05.

---

## H. Known risks

| Risk | Signal | Response |
|---|---|---|
| pi05 HSR is degenerate on the selected task | Pilot has no usable HSR variation | Report that the association is not estimable; do not substitute an outcome or task |
| **Task roles misclassified** | Target/destination/obstacle semantics inferred from one attribute | Explicit selected-task adapter; include actor-backed and procedural destination tests |
| **Table height varies under a fixed grid** | A2.3; one config sets `random_table_height: 0.03` | Build-time assert, refuse the scene. Never silently tolerate |
| Study scene generator differs from Office in a way not yet found | A2.4 table shows an unexpected or empty obstacle set for some task | Drop that task and report. The audit exists because Office was generalised from once already |
| **All scenes come out clean** | `clean_background_rate` defaults to 1 and is a SKIP probability | Verify the density-ladder configs override it (A2.5). Cheapest possible cause of no `eps_geom` variance |
| **Missing required collision mesh → optimistic `eps_geom`** | Any target/included-obstacle path fails preflight | Refuse the selected task up front; do not require meshes for procedural/skipped destinations |
| **Office back furniture never exercised** | `SPAWN_BACK_FURNITURE=True` is the office default; the occluder task sets it False, so no `eps_geom` number to date has seen a shelf or cabinet | Validate on Study first; treat `put_mouse_on_pad` as the `structure_mask` case (plan 1 §2b) |
| Wall-shelf inflates CPU cost | 34 MB glb sectioned at all 16 z slices | Expected once office is in scope; move it to the label-only `structure_mask` (plan 1 §2b) and it leaves the EDT path |
| **No `eps_geom` variance at fixed clutter density** | Stage C records cluster at `inf` | Treat `inf` as top-tied/right-censored; use pilot evidence to select density, never drop high-clearance scenes merely to manufacture spread |
| Clutter never lands near the transit corridor | `eps_geom` unassociated with density in stage C | `clutter_surface_split` respects `prohibited_area` around target and destination, so clutter is pushed away **by construction**. Measure the spread in stage C first |
| Unmeasured visual pathway | Obstacle placement changes both `eps_geom` and rendered input | Disclose that the association is not visibility-adjusted; do not add visibility measurement without user direction |
| HSR has weak information | Pilot HSR contributions are near-constant | Report the wide 95% CI at the user-supplied `N`; do not change the outcome post hoc |
| Stock-task rank calibration fails | Geometric-vs-gated ordering poor for relevant leg/orientation family | Orientation-grouped gated fallback and one-task primary scope |
| Per-task rollout cost blows the budget | Rollouts dominate the schedule | Metric is CPU and runs alongside; cut task count, never seed count — seeds are what power the correlation |
| VRAM / co-residency instability | OOM, CUDA illegal-memory-access, error spin | Known history (`tool_vla_pi05_port.md`). Metric is CPU here precisely to avoid it; never run curobo beside the pi05 server |

---

## I. Sequencing

The RoboPRO team has confirmed `put_cup_on_coaster` was in the checkpoint's training data. Its
role/waypoint adapters, A2 audit, metric records, Stage C2 review, frozen `bucket_spec.json`, and
narrow generic rollout/outcome plumbing are complete. The first clean pilot attempt
(`20260731-170002`) is invalid because the task's signed XY success predicate created one-step
false successes; none of its HSR values may be analyzed. After the predicate and Study video-camera
propagation fixes, the corrected pilot `20260731-171503` passed: 3/10 episode-level hard successes,
successes took 129--130 steps, failures took the full 600 steps, and all ten playable videos used
`countertop_camera`. No one-step false success remained. The outcome join and Stage D analysis are
implemented. Run the small stock-domain
geometric-vs-gated calibration before the final campaign. The user supplies final `N`; no task
expansion is part of this plan.

**Cheapest decisive checks, in order:** stage A2 (does the metric see these scenes correctly?),
stages C/C2 (what is the predictor distribution, and how should the user bucket it?), then stage A
(is the outcome measure real?). The first two gates precede every rollout; all three can kill stage
D before a final campaign is spent.

**Why A2 exists at all, and the lesson to carry:** this plan was originally written after reading
one Office task in detail and generalising. Several assumptions did not survive contact with the
Study generators, and the real failures are **silent** — a polluted EDT, a drifting table, and a
per-task obstacle set. None would have raised an exception; all would have
produced plausible numbers. Before adding any domain beyond Study and Office (KitchenL, KitchenS),
run the equivalent audit rather than assuming the pattern holds.

---

## J. Scientist-owned decisions

The protocol is locked: association rather than prediction; episode-level
`hard_success = task_success and not collision`; HSR as the mean of that indicator;
`eps_geom_min` as the primary predictor; one `put_cup_on_coaster` task; no grasped-object
adjustment; no visibility measurement; user-supplied `N`; and reported two-sided 95% CIs.

**Stage C2 bucket decision is frozen:** four `rho_geom_min = eps_geom_min / 0.03 m` clearance
buckets with boundaries 2.5, 4.0, and 5.0; lower-closed/upper-open intervals; boundary values enter
the higher bucket; identical values are never split; and `+inf` enters `high_clearance`. The
equivalent `eps_geom_min` boundaries are 0.075, 0.120, and 0.150 m. The distribution-only pilot
used 100 d10 scenes with reserved seeds 1000--1099 and produced counts 17/35/30/18; it does not set
or constrain final rollout `N`. No outcome data was used for this choice.
