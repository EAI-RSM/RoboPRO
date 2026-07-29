---
name: domain_scene
description: "The occluder scene: target/occluder object geometry, ring layout, and the collision-registration traps that make curobo blind to obstacles"
metadata:
  type: project
---

The scene built by `analyze_occluder_visibility.py` (office `put_mouse_on_pad` harness).

**Objects.**
- **Target = `001_bottle` id 9**, 20.6 cm long × 7.7 cm across, scale 0.132. It has 8 contact
  points forming a full 360° azimuthal ring, so `choose_grasp_pose` can pick a clean side grasp.
  (Contrast `047_mouse` id 0: only 2 contact points, and both are the SAME top-down grasp differing
  only in wrist roll — 1 approach direction, no side grasp. That is why the mouse scene is parked;
  see [[archive_planner_comparison]].)
- **Occluder = `029_olive-oil` id 3** (switched 2026-07-17 from `038_milk-box` id 2, which was too
  fat). Ranked by aspect ratio (tallest extent ÷ larger footprint extent), filtered to
  `stable: true`: olive-oil is H 0.305 m, aspect 3.8; milk-box was H 0.254 m, aspect 2.1 — taller
  and ~35% skinnier. Runner-up for max-skinny: `065_soy-sauce` id 0.
- `scale` in this repo is a **direct vertex multiplier** (real size = raw extents × scale), and raw
  extents in `model_dataN.json` are scale-1.0 metres — so aspect ratio is what matters, height is
  freely rescalable.
- All these bottle objects share ONE mesh convention: tall axis is local **Y**, origin at base,
  `center = [0, H/2, 0]`. So the same `OCCLUDER_QPOS` stands any of them upright. Swapping =
  modelname + model_id + `collision/base{id}.glb` path + the half-footprint constant.

**The occluder is NOT round.** The `OCC_HALF_FOOTPRINT = 0.04` **value is correct** (olive-oil id 3
is 0.0795 × 0.0796, half = 0.0398) — it is the *descriptor* in its comment, "~0.08 × 0.08 round,
yaw-invariant", that is wrong. The convex collision proxy's symmetric silhouette is what created
that belief. Measured off the real posed mesh 2026-07-24: cross-section is a rounded
SQUARE (0.0986 m across the posed bbox) that **tapers hard** — 0.0985 m at the body, **0.0387 m at
the neck (z≈0.99)**, gone above z≈1.043. That is up to ~4 cm of phantom obstacle if you extrude the
widest footprint, wider than a 0.03 gripper radius, exactly where a climb-over route passes.
Use the true posed collision mesh per z (see [[tool_clearance_metric]] `--occ-shape mesh`).

**Layout.** Ring construction now lives in `lib/occluder_ring.py`, and `test_ring_config.py`
asserts the formation is byte-identical per (seed, offset-spec) — that identity is what guarantees
the measured scene equals the rolled-out scene, so run it after any change there.
Occluders spawn as a **RING**, controlled by env attrs set BEFORE `setup_demo`:
`spawn_occluder`, `occluder_offset` (ring RADIUS), `num_occluders`, `occluder_angle0` (radians;
0 = bottle 0 in front at −y). Off-table ring positions are silently dropped. Destination pad is
parked at `PAD_XY=(0,−0.28)`. `SPAWN_BACK_FURNITURE` (default True, experiments set False) removes
shelf/cabinet/wooden-box/file-holder for gripper workspace — the only core-repo edit, flag-gated;
drawer/shelf/file-holder tasks must keep it True.

**Registration traps — these make curobo blind to obstacles and look like planner failures:**
1. **An actor created with `create_actor` is invisible to curobo unless appended to
   `self.collision_list`.** The occluder was physically present and camera-occluding but absent
   from curobo's world, so the planner found a "collision-free" path straight through it and the
   arm rammed it. Looks like the planner ignoring obstacles; really a planning-world/physics
   mismatch. `collision_list` is reset each build (`_office_base_task.py` (`benchmark/bench_envs/office/`)) before
   `load_actors`, so no stale accumulation.
2. **`update_world(exclude_obstacles=True)` strips every entry flagged `is_obstacle=True`** — that
   is ALL procedural clutter. `bench_demo_office_clean.yml` sets `enable_collision_metrics: true`
   and `build_cfg` never overrides it, so setup silently ran with clutter absent from curobo's
   world. Symptom: rollouts knock down clutter. **Fix: call `self.update_world()` (full world) at
   the top of `play_once`.** Keep `enable_collision_metrics=true` — physics `check_collisions()` is
   independent of curobo's world and IS the clutter-avoidance measurement. Reusable gotcha for ANY
   script reusing this harness with clutter. Consequence: cluttered scenes now legitimately fail
   to plan more often, because curobo actually sees the obstacles.
3. `env.get_collision_metrics()` → `total_collision_count`,
   `robot_to_static_object` / `target_to_static_object` counts + hit-object NAMES. This is how you
   disambiguate a model-vs-physics gap: `robot_to_static_object > 0` = a robot LINK hit clutter
   (margin/hull issue); `target_to_static_object > 0` = the HELD OBJECT hit it (attach approximation).
4. `self.occluder` LEAKS across reused-env builds (the env is reused for every `setup_demo`) —
   `load_actors` must set `self.occluder = None` up front, or a stale toppled occluder poisons
   later clean builds.

**Stability gates.** The occluder topples ~1 in 4 seeds during physics settling at cd=8 (empirical;
NOT the spawn rotation — `rotate_lim=[0,3.14,0]` is a pure yaw for these meshes, verified 0° tilt
across the range). A toppled object comes to REST so the base motion check reads it "stable" —
hence `OCCLUDER_MAX_TILT_DEG=25°` + `_occluder_tilt_deg()` in `check_stable`, which rejects the
seed and redraws. `--num-seeds N` guarantees N complete trajectory sets via a two-pass loop: pass 1
builds/measures every (offset, density) and gates stability, buffering; ANY rejection discards the
WHOLE seed; pass 2 rolls out and commits only fully-passing seeds. Safety cap `N*20+50` draws.

**Approach/arm geometry.** Each arm's collision-free reach envelope is lopsided toward its own
side — always approach on the arm's own side (right→+x, left→−x); crossing to the far side is
usually IK-infeasible. Any inward-x shift must FLIP SIGN per arm (`x_side − side*INSET`); a raw
`x_side − 0.1` only works for the right gripper. Higher approach z is more IK-reachable.
Orientation is a bigger lever than position.
