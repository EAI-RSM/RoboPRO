---
name: domain_curobo
description: "How curobo behaves in this repo: why it needs subgoals, the batched-IK reachability recipe, tuning knobs and what each failure status actually means"
metadata:
  type: project
---

Durable knowledge about the planner layer. Non-obvious from the code; earned across many sessions.

**Core problem (mentor-confirmed).** `motion_gen` is a *local* trajopt seeded by IK + graph search.
It does NOT reliably route a (possibly long, attached) object around a tall thin obstacle sitting
between start and goal — it either rams through (obstacle absent from the world) or fails
`TRAJOPT_FAIL`/`IK_FAIL` (obstacle present). The accepted RoboTwin fix is to decompose into
explicit subgoals; the base class even ships `get_curobo_target`/`_curobo_escape`. Each `move()`
Action is an independent `plan_single`, so subgoals hand curobo several easy problems instead of
one hard one. (The current work replaces hand-authored subgoals with computed seed trajectories —
see [[tool_seed_from_clearance]].)

**Who owns what.** Grasp ORIENTATION is chosen by `choose_grasp_pose` (score
`0.7*top_down + 0.3*side`, not occlusion- or side-aware), NOT curobo — curobo only *reaches* the
pose it is given. So bad grasp orientation is a RoboTwin selection problem, fixed by forcing a
contact-point id. Also: `choose_grasp_pose` validates a grasp by PLANNING a direct path to it
(`choose_best_pose` → `plan_multi_path`), so a blocked direct plan returns None → "can't find a
valid pre_grasp_pose". Compute waypoints from a GEOMETRIC grasp pose, move there first, then grasp.

**Batched collision-free IK reachability recipe** (took real investigation):
- `IKSolver.solve_batch` gives collision-free IK (joint limits + self + world) when a
  `world_coll_checker` is attached. Reachable-endpoint ≠ collision-free-PATH between endpoints
  (that's trajopt), but the map tells you where subgoals may legally sit.
- **aloha-agilex dual-arm planners are IN-PROCESS** (`communication_flag=False`, because both arms
  share one `curobo.yml`), so `env.robot.left_planner.motion_gen` is directly accessible with the
  scene already in its world. Subprocess/pipe mode only when arms use different ymls.
- The task planner's `motion_gen.ik_solver` uses `use_cuda_graph=True` → batch size LOCKS after
  the first call. For an arbitrary grid, build a FRESH `IKSolver` with `use_cuda_graph=False`,
  reusing `motion_gen.world_coll_checker`. Chunk the grid (~256) or you OOM: two warmed planners
  already hold ~9 GB of a 15.5 GB GPU.
- **Collision-OFF pass = pass `world_model=None`, NOT an empty world.** An empty `WorldConfig`
  keeps the collision cost active and errors `Primitive Collision has no obstacles`; curobo only
  builds that cost when `world_coll_checker is not None` (see `arm_base.py`). `None` → kinematics +
  self-collision only = the reach envelope.
- **Frame chain**, world gripper pose → curobo IK frame: `robot._trans_from_gripper_to_endlink` →
  `planner._trans_from_world_to_base` → aloha-agilex `frame_bias` + per-arm yaw patch (replicate
  `plan_path`'s branch). Self-check: real grasp pose → reachable True; occluder centre → False.
- `gripper_bias: 0.12` in `assets/embodiments/aloha-agilex/config.yml` and
  `_trans_from_gripper_to_endlink` (robot.py) offsets by `[0.12 − gripper_bias, 0, 0]` = **zero**
  → the gripper↔endlink POSITION offset is 0.
- Always release solvers: `del ik; torch.cuda.empty_cache()` in a `finally`. Torch's caching
  allocator holds memory RESERVED FROM THE DRIVER and never returns it without `empty_cache()`, so
  a leaked solver starves the whole machine, not just the process. This caused a 16 h run to die.

**World frame:** floor z=0, **table top z=0.74** (`table_height`), bottle grasp ~0.89. Every EE z
carries the ~0.74 offset — z=0.90 is only ~16 cm above the table, not "high".

**Knobs** (read from env in `planner.py` via `os.environ.get`; the Makefile's exported defaults
MATCH planner.py's, so `make` and raw `python` behave identically unless you override):
`CUROBO_MAX_ATTEMPTS` (24) and the finetune knobs are safe to raise — compute-only cost.
`CUROBO_TRAJOPT_SEEDS` (16) is **memory-bound** (OOM risk on 16 GB), raise cautiously. Raw `python`
does NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the Makefile does.

**Reading failure statuses:**
- `max_attempts` = curobo's per-plan retry budget; each attempt is a full IK→seed/graph→trajopt→
  finetune pipeline that adapts/loosens dt. `finetune_attempts` retries within one attempt's
  finetune stage.
- A printed `FINETUNE_TRAJOPT_FAIL` is the verdict after ALL attempts → geometrically hard, NOT
  attempt-starved. Raising `max_attempts` will not fix it. Real levers: looser `finetune_dt_scale`,
  higher release `dis`, the `approach_axis` straight-line metric.
- `IK_FAIL` = collision at the goal (clutter / held object). `TRAJOPT_FAIL` / `GRAPH_FAIL` = goal
  is fine, the path is blocked. curobo prints the exact status from `planner.py` (`[Error]: CuroboPlanner plan_path failed: <status>`).
- `INVALID_START_STATE_WORLD_COLLISION` = the *start* qpos already collides; no goal-side waypoint
  scoring fixes it.
- `move()` aborts on the FIRST failed plan, so a failed place-descent means the following
  gripper-open never runs → "stops right before placement, released nothing."
- **No scalar plan cost is exposed.** `plan_path` returns only status / `fail_reason` /
  `position_error` / `rotation_error` / trajectory. `PoseCostMetric` is a constraint spec pushed
  *into* the solve, not a queryable objective. The practical route to a real cost is
  `motion_gen.world_coll_checker.get_sphere_distance` (batched signed distance, robot spheres →
  obstacles) combined with `check_ik_batch`.

**`check_success` gotcha** (`put_mouse_on_pad.py`, reused with a bottle target): success = object
CENTER within **±2 cm per axis** of the pad centre AND both grippers open. The pad is 12 cm wide,
so an object clearly placed on the pad but off-centre is labelled FAIL. This, plus `move()`
aborting mid-plan, explains "successful on video but labeled fail" — it is not a labeling bug.

**Attached objects:** `planner.attach_object()` already voxelizes the real mesh into the
`attached_object` link's 60 sphere slots on every motion_gen, and a grid IKSolver built from the
same robot yml can copy that tensor verbatim (`get_link_spheres` → `attach_object(sphere_tensor=)`)
— no bounding-sphere approximation needed. Gotcha: the yml's placeholder spheres have radius
+0.001, the same value as `CUROBO_ATTACH_SPHERE_RADIUS`, so a radius-only "is anything attached"
check passes on an EMPTY gripper. Detect by sphere CENTRES instead (a real attach spreads them
~0.12 m from the link origin; placeholders sit exactly on it).
