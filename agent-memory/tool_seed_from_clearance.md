---
name: tool_seed_from_clearance
description: "Seeding curobo trajopt with the clearance route (#5): APPROACH_MODE/PLACEMENT_MODE interface, seed format, A/B design, and why the triage gate stays parked"
metadata:
  type: project
---

Seed curobo's trajopt with the [[tool_clearance_metric]] widest-path route so the expert plans the
approach around the occluder instead of relying on hand-tuned waypoints. Implemented end-to-end,
flag-gated. **`customized_robotwin/script/bench_script/plans/SEED_TRAJECTORY_PLAN.md` is the source of
truth — read it first.** Current run status is in [[status_current]].

**Interface — two env vars**, both mirroring each other, both stamped per-record:
- `APPROACH_MODE` (`_approach_mode()`): `off` = stock around-box waypoint, byte-identical to
  before; `direct` = waypoints OFF, plan pre_grasp straight from rest, NO seed = the
  **generalization baseline**; `seed` = direct WITH the clearance-route seed = the method.
  Legacy `SEED_FROM_CLEARANCE=1` ⇒ `seed`. `direct` vs `seed` differ by ONLY the seed (clean
  attribution) and NEITHER falls back to the waypoint — a miss fails the candidate.
- `PLACEMENT_MODE` (`scripted|direct`): `direct` makes `_backward_subgoal_poses` return `[]`
  (killed at the single source), skips `_select_attached_placement_plan`, and goes lift → one
  `_verified_intermediate` → `place_actor`. Code default is `scripted` so other tools are
  unchanged; `run_approach_mode_ab.sh` defaults to `direct`.
- Other knobs: `SEED_MEM_LOG` (default 1, prints cuda allocated/reserved after every build) and
  `CARRY_SEED=0|1` (diagnosis override; defaults to following `APPROACH_MODE=="seed"` so the A/B
  stays 2 cells). Everything else is a `SeedMetricConfig` field reachable as `SEED_<FIELD>` —
  `SEED_GATE_TAU` (0.35, feeds BOTH seed builders), `SEED_CHUNK` (256, the peak-memory knob;
  lowering it costs no fidelity, unlike coarsening the grid), `SEED_RES`, `SEED_ZMAX`, …
  See [[domain_bench_script_layout]].

**Post-refactor location:** the mode logic now lives in `task/seeding_mixin.py`,
`task/planning_mixin.py` and `task/placement_mixin.py`, NOT in `analyze_occluder_visibility.py`,
which is now a thin CLI. Behaviour is unchanged by design — the refactor was structural so that
already-collected A/B data stays comparable.

**CRITICAL gotcha:** the seeding lives in `play_once → run_rollout`, gated by `if rollout:` — it
runs ONLY with `--rollout`. The no-`--rollout` visibility pass never reaches `_plan_grasp_side` /
`_get_approach_seed`. (A different early planning call fired the debug dumps, which misled Claude
once.)

**Files.** New: `seed_from_clearance.py` (`build_seed` / `compute_route_configs` /
`resample_route_to_seed` / `route_qs_to_seed_tensor` / `save_route_visuals` / `sweep_gate_tau`),
`smoke_test_seed_2a.py`, `carry_object_spheres.py`. Edited: VENDORED curobo
`envs/curobo/.../motion_gen.py` — the edit is applied in-tree AND mirrored as
`bench_script/curobo_seed_traj.patch`, so it survives a curobo re-vendor; keep the two in sync.
It adds `MotionGenPlanConfig.seed_traj` + injection in
`_plan_from_solve_state`, mirroring the graph-plan seed via `TrajOptSolver.get_seed_set`'s
`seed_success` branch), `envs/robot/planner.py` (plan_path seed_traj), `envs/robot/robot.py`
(left/right_plan_path + comm-pipe handler, with `.detach().cpu()` — CUDA tensors aren't picklable),
`analyze_occluder_visibility.py`.

**Seed format (verified by round-trip):** `(16,1,28,6)` float32 CUDA = (num_trajopt_seeds*noisy,
batch, action_horizon=28, dof=6). No gripper joint; joint order `[fl/fr_joint1..6]`, arm-specific.
Each stock row is a start→goal linear interp; inject k rows and curobo fills the other 16−k with
linear seeds, so robustness is native and **a bad seed cannot cause a false reject**. curobo's own
default seed is joint-space linear interp + graph-planner fallback after 3 failed attempts; their
ablation puts the graph-planner seed at 85%→96% success — that is the lever this taps.

**Design decisions.** The route is computed ONCE per (scene, arm) and cached; the seed welds
tstep-0 to the REAL rest config (exact) and the goal to the route's grasp end. Route is seeded at
(current gripper → grasp), NOT grasp→pad. Cache key is (arm, SCENE-pose-signature) — an arm-only
key reused the first scene's seed for every episode (fixed). `save_route_visuals` REUSES the
metric's own plots into `<rollout_out_dir>/seed_route_visuals/episode<N>_<arm>/`. IK is
NONDETERMINISTIC, so `build_seed` is opportunistic: returns None sometimes → stock fallback,
never a failure.

**Carry leg (Phase C).** `_get_carry_seed(arm_tag, goal_xyz)` mirrors `_get_approach_seed`: copies
the attached spheres onto a fresh collision-ON IKSolver, labels the grid at the LIVE post-lift
orientation, builds the route, detaches + releases in a `finally`. Injected at the carry transit via
`_plan_and_replay_pose(..., seed_traj=, stage_label=)`. **The carry grid is labelled
GRASP-ALIGNED** (post-lift orientation) — one build, no reorientation move, rotation into the
handoff absorbed by `place_actor`. Rationale: `choose_best_pose`'s rotation search
(`create_target_pose_list`, ee-local-Y, ROTATE_NUM=10 over rotate_lim=[0,1] rad) tilts the held
object up to **51.6°**, so no nominal model can be fixed before the grasp is planned — the
object-in-ee transform is otherwise identical across all 8 contact points (they are yaw variants
about the bottle's symmetry axis). The only remaining approximation is the single-orientation
slice, and EDT inflation by `carry_sphere_extent()` affects eps* *meaningfulness* only, not
feasibility (attached IK handles that). The seed goal must match the pose actually being planned
to. `_note_seed_stat` carries `leg="approach"|"carry"`; the summarizer splits firing by leg.

**`sweep_gate_tau()`** re-runs the widest path at a looser ladder (0.5/0.7/1.0/1.5/2.0) over
volumes ALREADY in memory — no IK, no rebuild — ascending with early exit (merged is monotone in
tau, so the first hit is the minimum). Fires only when ungated merged but gated did not. The
verdict is appended to `RouteResult.reason` so it reaches `records.jsonl` as
`... -- SWEEP: tau=X WOULD connect (eps=...); set SEED_GATE_TAU=X`, or "no tau connects it, lower
SEED_RES instead". The pure helper is selftested.

**A/B design (2026-07-27, the user's calls).** `off` is NOT in the default cell set — the
around-box waypoint is a hardcoded one-occluder-in-front heuristic, so on general scenes it is a
*different task*, not a control. `direct → seed` is the whole experiment. Two scene types with
EQUAL seed counts: `curated` (randomized formation: rotation ~U[0,2π), 2–4 occluders, per-occluder
radii ~U[0.1,0.25]) and `standard` (no occluder, clutter density 8). eps* joins across modes BY
SEED — exact, because eps* depends on scene+arm+grasp orientation only and the scene is drawn from
the seed, not the mode. Harness: `scripts/validation/run_approach_mode_ab.sh` +
`summarize_approach_mode_ab.py`.

**#1 triage-gate (skip curobo when the metric says infeasible) stays PARKED.** The metric is
structurally TIGHTER than the rollout (sufficient-not-necessary joint gate; single
orientation × grasp × arm; reduced IK seeds; grid resolution), so it can false-reject real
successes. Seeding is robust to that because curobo stays the arbiter. Superset-fix recipe if ever
revived: ungated eps* + union over orientations/grasps/arms + full seeds + empirical calibration.
See customized_robotwin/script/bench_script/plans/SEED_TRAJECTORY_PLAN.md §7.
