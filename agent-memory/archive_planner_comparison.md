---
name: archive_planner_comparison
description: "ARCHIVE — the multi-planner / Hamid comparison line (Jul 2026), the parked mouse scene, and the hand-authored waypoint arc. Not on the working branch; kept for its lessons."
metadata:
  type: project
---

**Dead line of work** (user-confirmed 2026-07-29). None of this is on `peng-research-branch`:
`grep -c plan_algo analyze_occluder_visibility.py` → 0, and `hamid_occluder_expert.py` /
`experiment_free_target.py` do not exist here. Kept only for the facts that would be expensive to
rediscover.

**Where the code is.** `occluder-testbench` @ `91b2d26` = the 5-algorithm bench
(`--plan-algo {subgoal,reachability,baseline,dev,hamid}`, the 2x2 experiment scripts,
`experiment_free_target.py`, `--mouse` reachability). `35-occluder-spawn` @ `7f61a7c` = the
deployment branch, Hamid's single hardcoded planner, no selector.
`backup/pre-merge-35-occluder` @ `35801cb` = the pre-split safety net. What each planner was:
`subgoal` = the hand-authored around-box waypoints; `reachability` = check_ik_batch candidate
search, **side-grasp-ONLY** so it structurally cannot grasp a top-down object; `baseline` =
subgoal with routing suppressed; `dev` = a verbatim copy of dev's `put_mouse_on_pad.play_once`;
`hamid` = the deployment expert vendored as a frozen class.

**Lessons worth keeping:**

- **Attribution: check `git log --format='%an' -- <file>` before attributing anything.**
  `analyze_occluder_visibility.py` was created by the USER (`b3c2e66`, 2026-07-03). Hamid layered 5
  commits on top, growing it 477 → 2583 lines. An earlier memory called it a "from-scratch rewrite"
  and misled a whole session into analyzing the user's own code as someone else's. The load-bearing
  design — waypoints anchored to the occluder/bottle/pad tracing an arc so curobo can route around
  — is the user's; Hamid kept `_box_side_x`, `_around_box_waypoint` and the two-phase structure
  essentially verbatim, including the user's comments.
- **"baseline" was NOT the old planner.** It carried occluder-harness tuning dev never had
  (absolute-height `FORWARD_SUBGOAL_Z=0.85` lift vs dev's relative +0.1;
  `DROP_GRASP_ORIENTATION_CONSTRAINT` grasp vs dev's stock constrained; `constrain="free"` place vs
  `"align"`). So baseline-vs-reachability isolated routing only *within that harness* — it never
  licensed "new beats dev". The general form of this trap is in [[feedback_scientific_rigor]].
- **Parametrize-and-enumerate is not derivation.** Hamid's changes were (1) single predefined
  distances → *ladders* of predefined distances tried in fixed order, first-feasible wins —
  still hand-picked constants, just more of them, producing a ~108-candidate search per seed;
  (2) two extra arc vertices; (3) **~2/3 of the 2100 new lines were execution fidelity, not path
  design** — because `plan_success=True` was lying (bottle on the floor, every stage "successful").
  Trajectory cache+replay (an independent re-plan to the same pose drifts up to 1.47 rad),
  `_object_retained`, grasp-rise verification, descent slicing, landing search. Fair framing: that
  is what makes any waypoint-strategy comparison trustworthy at all.
- **A slow seed ≈ a failing seed.** `_select_pick_place_candidate` is a combinatorial SEARCH that
  returns on the first fully-planning candidate; if none do — exactly the hard seeds — it plans all
  ~108 then falls back to deepest-progress. Long runtime = search exhausted, not wasted retries.
  There was no time budget; a per-rollout wall-clock deadline wrapping
  `robot.left_plan_path`/`right_plan_path` (the single verified choke point for every expensive
  call) was designed but never built.
- **The ramming was never "executes a false path."** Every execution path guards plan status and
  aborts (`move()`, `_replay_planned_move`, descent slices). The
  knock-downs were a **collision-model-vs-physics gap**: curobo executes a path it believes free
  but physically contacts clutter. Candidates: convex-hull clutter meshes under-representing flat
  cartons; `motion_gen_near_contact` using a reduced `collision_activation_distance` by design;
  the final grasp approach running with the TABLE DISABLED; the held object voxelized coarsely.
- **Waypoint placement may not be the binding constraint.** Hamid's own comment records that he
  widened the waypoint orientation and y-offset search and measured NO improvement, concluding
  failures were "pure kinematic unreachability at the position itself." Placement's dominant
  failure was `INVALID_START_STATE_WORLD_COLLISION` — the *start* qpos already collides, which no
  goal-waypoint scoring fixes. And a cost function cannot invent the route: it tunes a human's arc;
  discovering the sequence needs a search over pose *sequences* = a planner. This reasoning is why
  the work moved to computed seeds ([[tool_seed_from_clearance]]).
- **Reading the terminal:** `saving: episode = N index = M` prints with `end="\r"`. A **frozen
  index for minutes = planning-bound** (the real tell); a **slowly ticking index = render-bound**,
  a different problem. `records.jsonl` IS the log (no FileHandler exists); it is opened `"w"` and
  written per completed seed, so a currently-stuck episode is absent until it resolves.
  `ROBOTWIN_LOG_MOVE=1` adds verbose `[play_once]`/`[grasp_verify]`/`[descent-slice]` to stdout only.

**The typical/mouse scene is PARKED** (2026-07-10; all planners failed 100%). Causes in order:
reachability's side-grasp-only search cannot grasp a top-down object at all (architectural);
`FORWARD_SUBGOAL_Z=0.85` is an ABSOLUTE wrist height tuned for the tall bottle, so lifting "to
0.85" doesn't raise a flat mouse grasped at table height; and the binding constraint turned out to
be **spawn geometry** — the mouse at the bottle-tuned back band is IK-MARGINAL, graspable but with
no headroom to lift straight up. Confirmed 2026-07-13: the mouse GRASP pose maps GREEN, so the
`IK_FAIL` is downstream (clutter in the collision world / the attached object on lift-carry-place /
the constrained-orientation approach / the start joint config), not raw grasp reachability. Also
worth knowing: **dev's `put_mouse_on_pad::load_actors` ALREADY spawns a milk-box occluder**
(conditional on clutter, at the mouse–pad midpoint, random id) — so the occluder concept predates
this branch; the contribution is the CONTROLLED occluder + the routing, not the box itself.

**The hand-authored subgoal arc** (superseded by `APPROACH_MODE`/`PLACEMENT_MODE=direct`, which
exist to turn it OFF): 9 subgoals, forward `fwd_pad → fwd_box → fwd_bottle → grasp → lift` mirrored
by backward `bottle → box_mid → pad_high → place`. Two empirical findings survive it —
**orientation is a bigger lever than position**, and **rotating the held bottle FLAT during the
backward carry markedly improved success** ("more of a help than most of the location subgoals").
