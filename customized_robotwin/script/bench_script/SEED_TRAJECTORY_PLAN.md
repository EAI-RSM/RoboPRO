# Seeding curobo with the clearance-metric route

This note is the implementation plan for **feeding the clearance metric's widest-path route into
curobo as a trajectory-optimization seed** ("idea #5"), so the planner routes *around/over* the
occluder on the first attempt instead of burning its default linear seeds and falling back to its
generic geometric planner.

It records the feasibility verdict, the exact injection point in the vendored curobo, the change
surface, and a phased task list. Tasks are checkboxes; they will be completed one at a time.

---

## 1. The idea in one paragraph

curobo's trajectory optimizer is a **local** optimizer: it refines whatever initial trajectory
("seed") it is handed. With no seed supplied, curobo seeds trajopt with a **linear interpolation
in joint space** from the start config to the IK-solved goal config — which, on an occluder scene,
drives the gripper *straight through the bottle*. It then wastes ~3 attempts on such seeds before
enabling its own generic geometric planner (PRM) to produce a collision-free seed. Our clearance
metric already computes a **clearance-optimal, reachability-honest route** around/over the occluder,
with a joint configuration (`qfield`) at every point. Handing that route to curobo as one of its
seeds replaces the generic geometric seed with a better one — the same mechanism curobo's own
ablation credits for 85% → 96% success.

Why this beats hand-tuned waypoints (the current expert): a waypoint is a **hard constraint**
(the path *must* pass through it); a seed is a **soft initialization** (curobo may deviate). A
wrong seed degrades to today's behavior; a wrong hard waypoint can make a scene infeasible. See
the seed-vs-waypoint discussion — routing waypoints get **demoted to a fallback**, task subgoals
(pre-grasp, lift, flat-carry, place) stay.

---

## 2. Feasibility verdict: FEASIBLE, moderate scope (not a port)

The seed machinery **already exists end-to-end inside the vendored curobo**; it is simply not
exposed at the public entry point.

- **`TrajOptSolver.get_seed_set`** (`envs/curobo/src/curobo/wrap/reacher/trajopt.py:1588`) already
  supports a *partial* external seed: when `num_seeds > seed_traj.shape[0]`, it **concatenates our
  seeds and fills the remaining slots with its default linear seeds**. So we pass our route as
  **one seed out of `num_trajopt_seeds` (16)** and curobo fills the other 15 as usual. Best-of-batch
  scoring then picks the winner — our seed can *only help*, natively, with no fallback logic.
- **`_plan_from_solve_state`** (`envs/curobo/src/curobo/wrap/reacher/motion_gen.py:3366-3382`)
  already builds a `trajopt_seed_traj` from the **graph planner's** trajectory and passes it through
  that same `get_seed_set` path. Our route does exactly what curobo's own geometric seed does.

**The only gap:** `plan_single` / `MotionGenPlanConfig` / `plan_path` do not expose
`trajopt_seed_traj` — it is a local variable only ever populated by the internal graph search.
Because **curobo is vendored in-repo**, we can add the parameter. That is why this is moderate and
not a port: we thread an argument through machinery that already consumes it.

> **Vendored-curobo edits are NOT tracked by this repo** — `envs/curobo` is gitignored (and is its own
> checkout of upstream NVlabs, detached HEAD). The Phase-1 `motion_gen.py` change is therefore exported
> as `curobo_seed_traj.patch` next to this file; re-apply after any curobo reinstall with:
> `git -C envs/curobo apply script/bench_script/curobo_seed_traj.patch`.
> Without it, seeding silently no-ops back to the stock path (no error).

**Seed format required:** `(n_seeds, batch, action_horizon, dof)` tensor (or `JointState`).
`action_horizon` ≈ the trajopt horizon (~32); `dof` = active joints; joint order = the solver's.

---

## 3. Change surface

| Layer | File | Change |
|------|------|--------|
| Vendored curobo | `envs/curobo/.../motion_gen.py` | add optional `seed_traj` to `MotionGenPlanConfig` (or `plan_single` kwarg); route it into `trajopt_seed_traj` before the `get_seed_set` call, mirroring the graph-plan branch |
| Planner wrapper | `envs/robot/planner.py` (`plan_path`, ~L197; `plan_single` call L302) | accept + forward an optional seed |
| Expert | `script/bench_script/analyze_occluder_visibility.py` | compute route per grasp candidate, build seed, try seeded plan as **primary** with the existing waypoint sweep as **fallback** |
| **New** builder | (new helper) | convert widest-path `qfield` → curobo seed tensor: joint order, interpolate to `action_horizon`, weld endpoints to real `start_state` + grasp goal config |

The route→seed builder is the actual work of #5; everything else is plumbing.

---

## 4. Phased task list

### Phase 0 — de-risk the format (cheap, standalone, do first)
- [x] **0a.** Instrument a normal plan to dump the shape / joint-order / dtype of curobo's
  `trajopt_seed_traj`. **DONE — result below.**
- [x] **0b.** Round-trip test: feed curobo's own seed back through the injection path and confirm an
  identical batch. **DONE — `pass=True, max_abs_diff=0.000e+00`** (bit-exact; format/mask/positioning
  all correct). Self-check block lives in `motion_gen.py` behind `ROBOPRO_SEED_ROUNDTRIP` (remove with
  the 0a dump after Phase 2).

> **Discovered seed format (0a):** `trajopt_seed_traj` is a **`(16, 1, 28, 6)`** `float32` `cuda`
> contiguous tensor = `(num_trajopt_seeds × noisy_trajopt_seeds, batch, action_horizon, dof)`.
> - `num_trajopt_seeds = 16`, `noisy_trajopt_seeds = 1` → total batch 16; inject *k* rows, curobo
>   fills 16−*k* with linear seeds.
> - `action_horizon = 28` timesteps; `dof = 6` (**no gripper joint**).
> - joint order = `[fl_joint1..6]` (left arm) / `[fr_joint1..6]` (right) — **arm-specific**, our
>   `qfield` must match this order.
> - Each row is a **start→goal linear interpolation**: tstep 0 = start config, tstep 27 = goal
>   config. Our route seed must weld tstep 0 to the real `start_state` and tstep 27 to the goal
>   config, resampled to 28 steps.
> - Instrumentation lives in `motion_gen.py` behind `ROBOPRO_SEED_DUMP` (remove after Phase 1).

### Phase 1 — plumbing (small)
- [x] **1a.** Add `seed_traj` to `MotionGenPlanConfig` + wire into `_plan_from_solve_state`.
  **DONE** — field + `clone()` + injection block (mirrors the graph-seed format; our `k` rows become
  the first `k` of 16, `get_seed_set` fills the rest with linear seeds). `plan_single` path only.
- [x] **1b.** Forward the seed through `planner.py` `plan_path` → `plan_single`. **DONE** —
  `plan_path(..., seed_traj=None)` sets `plan_config.seed_traj` before the `plan_single` call.
- [x] **1c.** Guard: seed absent ⇒ byte-identical to today. **Satisfied by construction** — injection
  only fires when `seed_traj is not None`; default None ⇒ stock path.

> **For Phase 3 (expert wiring), the remaining links to `plan_path`:** the expert calls curobo via
> `robot.left_plan_path` / `right_plan_path` (`envs/robot/robot.py:509/557`) → `planner.plan_path`.
> Add `seed_traj=None` to those wrappers and forward it. **Gotcha:** when
> `communication_flag` is true (arms load different ymls, `robot.py:265`) the request crosses a
> multiprocessing pipe — send the seed as a **CPU tensor** (`.detach().cpu()`); motion_gen's
> injection re-moves it to CUDA. Also forward it in the subprocess handler (`robot.py:~795`,
> `seed_traj=msg.get("seed_traj")`). Only `plan_single` is wired — `plan_batch` is not (not needed
> for grasp-approach seeding).

### Phase 2 — route→seed builder (the meat)
- [x] **2a.** Run the clearance metric for the candidate's arm **at that candidate's grasp
  orientation**; get the gated widest-path + per-voxel configs. **DONE** — new module
  `seed_from_clearance.py`: `compute_route_configs(env, planner, arm, ik, grasp_q, start_xyz,
  goal_xyz, cfg)` reuses `clearance_metric_3d` as a library, seeds the widest-path at
  **(current gripper → grasp)** (the approach, *not* grasp→pad), builds the mandatory warm branch
  field, and returns `RouteResult.route_qs` `(K, dof)` in curobo dof order (None + `reason` when a
  seed can't snap or the gated path can't connect). *Syntax-checked; GPU-validated when wired in
  Phase 3.* Cost caveat (sec.5) noted in the module: default grid ≈1.7e5 voxels of IK + warm solve.
- [x] **2b.** Resample the route's per-voxel configs to `action_horizon` (28); weld tstep 0 to the
  real `start_q`, last tstep to curobo's goal (mod 2π). **DONE + VERIFIED** — pure
  `resample_route_to_seed()` (np.unwrap → shortest-path, arc-length resample, degenerate→None); CPU
  self-test passes (`python seed_from_clearance.py --selftest`).
- [x] **2c.** Shape to `(k, 1, action_horizon, dof)` CPU tensor in curobo dof order. **DONE + VERIFIED**
  — `route_qs_to_seed_tensor()` → `(1,1,28,6)` float32 CPU contiguous (matches 0a); top-level
  `build_seed(...)` ties 2a+2b+2c into one Phase-3 call returning `(seed_tensor|None, RouteResult)`.
  Self-test covers shape/dtype/cpu/contiguous/None-passthrough.

> **2a GPU-validated (smoke_test_seed_2a.py, seed 1, left arm, coarse res 0.03/zres 0.06):** PASS —
> 16-voxel route climbing z 0.78→1.02 m, eps=0.170 m, correct `(1,1,28,6)` tensor, `action_horizon=28`
> confirmed. Findings: (a) the joint **gate is not binding** on this scene (ungated==gated at every tau
> 0.35–2.0), so no gate-policy decision needed now; (b) **IK is nondeterministic** → `build_seed`
> occasionally returns `None` for identical inputs (an earlier run NO-ROUTE'd), absorbed by the stock
> fallback → seeding is *opportunistic*. **Firing rate across seeds is a Phase-4 measurement.** Metric
> cost ≈38 s at coarse res (the per-candidate cost caveat is real — a Phase-3/tuning concern).

### Phase 3 — wire into the expert (primary + fallback)
Decision: **compute the route ONCE per (scene, arm), cache it, reuse across candidates** (route
depends on scene+arm+grasp orientation, not on gap/z_lift/…); **flag-gated, OFF by default** (env var)
so the baseline is untouched and A/B is clean. Split into Chunk A (plumbing) + Chunk B (behavior).

- [x] **3-plumbing (Chunk A).** Thread `seed_traj` end-to-end, no-op when None (zero behavior change):
  `_plan_pose_with_local_waypoint_retry(seed_traj=)` → `robot.left/right_plan_path(seed_traj=)`
  (`.detach().cpu()` for the `communication_flag` pipe) → in-process **and** subprocess-handler
  `planner.plan_path(seed_traj=)`. Seed warm-starts the DIRECT attempt only; shrink-retry fallback
  stays unseeded. Syntax-checked. **DONE.**
- [x] **3a (Chunk B).** `_plan_grasp_side`: when flag on + cached seed, a **seeded direct plan to
  `pre_grasp`** (bypassing the around-box waypoint) runs first; on any miss `seeded_direct`
  stays False and the stock waypoint path runs. **DONE.**
  > **CORRECTION (2026-07-27).** The original claim here — "no 'waypoint' trajectory captured →
  > play_once's None-guarded replay skips it" — was **WRONG**, and it was a live bug. `play_once`
  > branches on the waypoint **POSE**, and when the pose is present with a `None` trajectory it falls
  > through to `_plan_and_replay_pose`, which live-plans *and executes* the around-box waypoint. So
  > `direct`/`seed` still ran the heuristic, then **snapped back to rest** when the from-rest
  > `pre_grasp` trajectory was replayed from the waypoint's qpos (`_replay_planned_move` plays joint
  > positions back directly and never re-plans). Visible as: gripper moves forward, jumps back to
  > start, then moves normally. **Fixed** by suppressing the waypoint at the source —
  > `_candidate_specs` emits `grasp_waypoint=None` outside `off` — which also covers the
  > deepest-progress *fallback* candidate, that never reaches `_plan_candidate`'s tail. `play_once`
  > additionally warns and skips if a pose ever leaks through again. Any `direct`/`seed` rollouts
  > from before this fix are NOT valid A/B data: their "waypoints off" cells had waypoints on.
- [x] **3b (Chunk B).** Stock waypoint path + `_local_waypoint_retry` preserved verbatim under
  `if not seeded_direct` (byte-identical when flag off); replay/capture untouched; grasp+lift tail shared.
  **DONE.**
- [x] **3-cache (Chunk B).** `_get_approach_seed(arm)`: `build_seed(...)` once per (scene,arm), cached on
  `self._approach_seed_cache`; `start_q` = arm rest config in curobo dof order (exact tstep-0 weld), goal
  defaults to the route grasp-end; passes the planner's real `action_horizon`. Any failure (flag off / no
  route / exception) → None → stock. `build_seed` now defaults None start/goal to the route ends. **DONE.**
- Verification: all 5 edited files parse; 2b/2c self-test still green; `SEED_FROM_CLEARANCE` unset ⇒
  seeded block skipped, stock path verbatim (zero behavior change).
- [ ] Deferred **3c**: collapse the `gap × z_lift × orient × y_offset` loop onto the fallback — only
  after Phase-4 shows seeding reliably fires (invasive; not worth it until measured).

### Phase 3+ — waypoints-off baseline (3-mode selector)
Motivation: the current waypoint generation is overfit to "one occluder in front"; to claim
generalization we need an honest floor AND to deconfound the seed's effect (seeded-vs-stock changes
two variables: waypoint→direct AND no-seed→seed).
- [x] **`APPROACH_MODE` env var** (`_approach_mode()`), `off` | `direct` | `seed`:
  - `off` (default) = stock around-box waypoint — byte-identical to today.
  - `direct` = waypoints OFF, plan pre_grasp straight from rest, **no seed** (generalization baseline).
  - `seed` = waypoints OFF, direct pre_grasp **with** the clearance seed (the method).
  `direct` vs `seed` differ by **only the seed** (clean attribution); **neither falls back to the
  around-box waypoint** (a miss fails the candidate — no heuristic contamination of the floor).
  Legacy `SEED_FROM_CLEARANCE=1` ⇒ `seed`. Placement/carry subgoals unchanged in all modes.
  A/B/C = same rollout command with `APPROACH_MODE=off|direct|seed`, compare `rollout_success`.

### Route visuals for seeded rollouts
- [x] `save_route_visuals(res, out_dir, ...)` in `seed_from_clearance.py` reuses the metric's OWN plots
  verbatim (`_metric_path3d` 3D climb-over route + `_viz_topdown` + `_viz_side_elevation`) — no new viz.
  `_get_approach_seed` calls it after a successful build (seed mode only). Folder layout: INSIDE the
  rollout's own output folder — `<out_dir>/seed_route_visuals/episode<N>_<arm>/` (env sets
  `_rollout_out_dir`/`_rollout_ep` per episode; standalone `<results>/seed_visuals/` fallback if unset).
  **Seed cache keyed by (arm, SCENE signature)** — target+occluder world poses — so each scene recomputes
  its own seed (fixed a stale-seed bug: arm-only key reused the first scene's seed for every episode,
  which also capped visuals at one set). Controls: `SEED_VISUALS=0` disables; `SEED_VISUALS_DIR` overrides
  the fallback base. The gold route line = the
  gated widest-path the seed encodes (the 28-step joint seed is its resampled/welded form). cm's legend
  says "grasp/pad seed" but the two ends here are gripper(start)→grasp(goal) (cosmetic). Never raises.

### Cleanup (do at Phase 4)
- [ ] Strip the two temporary debug blocks from `motion_gen.py` (`ROBOPRO_SEED_DUMP` 0a dump +
  `ROBOPRO_SEED_ROUNDTRIP` 0b self-check). Env-guarded/zero-impact meanwhile.

### Phase 4 — validate (research, user-driven)
- [x] **4-harness.** Measurement + driver + summary built (engineering half):
  - **Instrumentation** (per-episode, into `records.jsonl`): `approach_mode`;
    `rollout_seconds` (wall-clock, successes *and* failures); `rollout_plan_effort` —
    curobo's `attempts`/`trajopt_attempts` per plan, recorded on the DIRECT attempt only so
    the count stays comparable across modes (the shrink-retry fallback is excluded);
    `rollout_seed_stats` — per-build seed outcome (`built`, `reason`, `seconds`,
    `route_voxels`, `eps_gated`; cache hits tagged `reason="cached"` so build cost isn't
    double-counted). `planner.plan_path` now returns `attempts`/`trajopt_attempts`/`seeded`
    on both the success and failure branches.
  - **Driver** `scripts/validation/run_approach_mode_ab.sh` — runs **`direct` + `seed`** over
    the same seed set with frozen curobo knobs, one process + one log per cell, then
    summarizes. Refuses to start if the vendored curobo lost the `seed_traj` patch (that
    would make the `seed` cell silently measure `direct`).
  - **`off` is NOT in the default cell set** (2026-07-27, user's call): the around-box
    waypoint is hardcoded for one-occluder-in-front, so on the general scenes this benchmark
    targets it is a *different task*, not a control. `direct→seed` is the whole experiment.
    Opt in with `MODES="off direct seed"` for a reference number.
  - **Summary** `scripts/validation/summarize_approach_mode_ab.py` — per-cell Wilson CI,
    usable-samples/hour, failure-stage breakdown, **seed firing rate**, paired McNemar for
    `direct→seed` (and `off→direct` only if that cell was run), and a 3-panel figure.
    `--selftest` verifies loader/stats/figure with and without the opt-in `off` cell.
  - **Firing rate is the gate:** a miss falls back to an unseeded plan silently, so a null
    `direct→seed` is unreadable unless the seed actually fired. The summary prints it first
    and warns when it is 0 or under 50%.
- [ ] **4-run.** Run the A/B and interpret it (user): `NUM_SEEDS=50 bash
  scripts/validation/run_approach_mode_ab.sh`. Expect the `seed` cell to be the slowest —
  it pays a clearance-metric build per (scene, arm); that cost is reported, not hidden.

---

## 5. Risks / open unknowns

- **Per-candidate metric cost.** The route is not free (IK-batch per candidate). May need caching
  across candidates / `clearance_z`, or a lightweight single-slice route. This is the main thing that
  could make #5 net-negative on wall-clock even if it lifts success — measure early.
- **Endpoint alignment.** If the route's goal config disagrees with curobo's IK goal config (a
  different IK branch), the seed pulls the wrong way. Weld to curobo's own goal config, not the
  metric's.
- **Placement / carry deferred.** The metric does not model the held bottle (held-object hook
  unbuilt), so Phase 2 targets the **empty-gripper grasp approach only**; the placement subgoals stay
  as-is for now.

---

## 6. Keep / don't-touch traps

- **Trajectory capture-and-replay** in the expert exists because curobo trajopt has no uniqueness
  guarantee (qpos drift up to ~1.47 rad flips downstream reachability). Seeding *reduces* drift but
  does not remove the need to replay the exact verified trajectory. Keep it.
- **`_plan_pose_with_shrinking_waypoint` / `_local_waypoint_retry`** stay as the fallback net.
- **`PLACEMENT_STRICT_ORIENTATION_STAGES`** and the `near_contact` / finetune knobs are orthogonal.

---

## 7. Superset caveat (why this path, not the triage gate "#1")

An earlier idea (#1) was to *skip* curobo when the metric says a scene is infeasible. That needs
`rollout-success ⟹ metric-feasible`, and the metric is structurally *tighter* than the rollout on
several axes (the sufficient-not-necessary joint gate; single fixed orientation × single grasp ×
single arm; reduced IK seeds; grid resolution). So a skip-gate can discard real successes.
**Seeding (#5) is robust to all of that** because curobo stays the final arbiter: a wrong seed is
ignored, never a false rejection. That is the core reason this plan pursues seeding over gating.
