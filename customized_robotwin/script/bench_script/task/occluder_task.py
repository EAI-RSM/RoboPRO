"""Runtime assembly of the occluder benchmark task."""

import contextlib
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import transforms3d as t3d
from curobo.types.state import JointState

from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from envs.utils import ArmTag, create_actor, create_box, rand_pose
from lib.occluder_ring import occluder_ring_xy
from lib.planning_tuning import *  # noqa: F403
from lib.scene_build import get_env_class
from lib.scene_constants import *  # noqa: F403
from .grasp_mixin import GraspMixin
from .placement_mixin import PlacementMixin
from .planning_mixin import PlanningMixin
from .pose_geometry import PoseGeometryMixin
from .seeding_mixin import SeedingMixin
from .state_checks_mixin import StateChecksMixin


def make_occluder_task():
    Base = get_env_class("put_mouse_on_pad", bench_subdir="office")

    class OccluderTask(SeedingMixin, PlacementMixin, PlanningMixin,
                       GraspMixin, StateChecksMixin, PoseGeometryMixin, Base):
        # Remove the back furniture (shelf/cabinet/file-holder) for now to free up
        # gripper workspace. Flip back to True (or delete) to restore the full office.
        SPAWN_BACK_FURNITURE = False
        # Lever A: drop the orientation-hold constraint on the final grasp approach so
        # curobo won't reject it with INVALID_PARTIAL_POSE_COST_METRIC. Set False (or
        # delete this flag + the grasp_actor_from_table override below) to restore the
        # stock straight-in, wrist-locked constrained approach.
        DROP_GRASP_ORIENTATION_CONSTRAINT = True
        spawn_occluder = False
        # Occluder ring: num_occluders bottles equally spaced (2*pi/n apart) on a circle of
        # radius occluder_offset centred on the target. occluder_offset is the RING RADIUS
        # (kept under its old name so the reachability/gripper/swept tools that set
        # env.occluder_offset still get a single front box). occluder_angle0 rotates the
        # whole ring; 0 puts occluder k=0 directly in front (-y). num_occluders=1 == the
        # original single front occluder. See occluder_ring_xy().
        occluder_offset = 0.2         # ring radius in metres (target at centre)
        num_occluders = 1
        occluder_angle0 = 0.0         # radians; angle of occluder k=0, measured from -y (front)
        occluder_radii = None         # per-occluder radii (list); None -> occluder_offset for all
        target_model = TARGET_MODEL
        target_id = TARGET_ID
        target_xlim = TARGET_XLIM
        target_ylim = TARGET_YLIM
        fixed_pad_xy = PAD_XY

        def load_actors(self):
            # target: random within the upper-third band (varies per seed -> distribution)
            target_pose = rand_pose(xlim=list(self.target_xlim), ylim=list(self.target_ylim),
                                    qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])
            # scale from the task yaml if present, else None -> model's own scale
            self.target_obj = create_actor(
                scene=self, pose=target_pose, modelname=self.target_model, convex=True,
                model_id=self.target_id,
                scale=self.item_info["scales"].get(self.target_model, {}).get(f"{self.target_id}", None),
            )
            self.target_obj.set_mass(0.05)

            # destination pad (flat; parked at the front so it never conflicts with the occluder)
            px, py = self.fixed_pad_xy
            pad_pose = rand_pose(xlim=[px], ylim=[py], qpos=[1, 0, 0, 0], rotate_rand=False)
            self.color_name, self.color_value = "Gray", (0.5, 0.5, 0.5)
            self.des_obj = create_box(scene=self, pose=pad_pose, half_size=[0.06, 0.06, 0.0005],
                                      color=self.color_value, name="box", is_static=True)
            self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
            self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
            self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
            self.des_obj_pose[2] += 0.02

            # occluder ring: num_occluders tall olive-oil bottles equally spaced (2*pi/n
            # apart) on a circle of radius occluder_offset around the target. k=0 sits
            # directly in front (-y) when occluder_angle0=0; num_occluders=1 reproduces the
            # original single front box. self.occluder aliases ring bottle 0 so the
            # single-box tools (reachability/gripper/swept) keep working unchanged.
            self.occluders = []
            self.occluder = None
            if self.spawn_occluder and self.num_occluders > 0:
                mp = self.target_obj.get_pose().p
                # Per-occluder radii when the caller supplied them (--offsets given a range),
                # else the single scalar radius. The scalar path keeps every external tool
                # that only sets env.occluder_offset (reachability / gripper / swept volume)
                # working unchanged.
                radii = getattr(self, "occluder_radii", None) or self.occluder_offset
                # Drop any ring position whose centre would leave the tabletop -- keep the
                # formation with fewer bottles rather than spawning one off the table.
                ring = occluder_ring_xy(float(mp[0]), float(mp[1]),
                                        radii, self.num_occluders,
                                        self.occluder_angle0,
                                        xlim=TABLE_XLIM, ylim=TABLE_YLIM)
                for ox, oy in ring:
                    occ_pose = rand_pose(xlim=[ox], ylim=[oy], qpos=OCCLUDER_QPOS,
                                         rotate_rand=True, rotate_lim=[0, 3.14, 0])  # random yaw
                    occ = create_actor(
                        scene=self, pose=occ_pose, modelname=OCCLUDER_MODEL, convex=True,
                        model_id=OCCLUDER_ID, scale=[1.0, 1.0, 1.0],
                    )
                    occ.set_mass(0.1)
                    # Register each occluder in curobo's collision world so the planner
                    # actually avoids it. NOT flagged is_obstacle -> it survives the
                    # exclude_obstacles=True pass used in collision-metrics/eval mode
                    # ("... curobo planner skips clutter obstacles"), which only drops
                    # is_obstacle=True procedural clutter.
                    self.collision_list.append({
                        "actor": occ,
                        "collision_path": f"{os.environ['BENCH_ROOT']}/{OCCLUDER_COLLISION}",
                    })
                    self.occluders.append(occ)
                # self.occluder aliases ring bottle 0 for the single-box tools; stays None
                # if every ring position was filtered off-table (an empty formation).
                if self.occluders:
                    self.occluder = self.occluders[0]

        def play_once(self):
            # Expert plan. FORWARD (grasp) below is fixed. BACKWARD (placement) is a list
            # of subgoals returned by self._backward_subgoal_poses() -- EDIT THAT METHOD to
            # add / remove / reorder placement subgoals; each subgoal is one curobo move.
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"

            # Structured failure recorder (replaces having to grep '[play_once] after'
            # lines out of ROBOTWIN_LOG_MOVE logs by hand): the FIRST stage whose
            # checkpoint sees plan_success go False, plus whatever specific CuRobo
            # fail_reason our own plan+replay helpers captured for it (None for
            # stages still going through the standard self.move() path, which
            # doesn't expose a reason). Read by run() after run_rollout() and written
            # into records.jsonl. Reset per-episode since env is reused across seeds.
            self.rollout_failure_stage = None
            self.rollout_failure_reason = None
            self.rollout_candidate_info = None
            self._last_fail_reason = None
            self._grasp_baseline_transform = None
            self._slow_attached_replay = False
            # Richer diagnostics read by run() into records.jsonl alongside the
            # single-value failure/candidate fields above: per-attempt grasp
            # history (which contact points were tried and why each failed, not
            # just the last one), per-check retention drift (position + rotation,
            # not just the pass/fail it produced), and per-slice descent detail
            # (CuRobo's own position/rotation error on each place_actor segment).
            self.rollout_grasp_attempts = []
            self.rollout_retention_checks = []
            self.rollout_descent_slices = []
            # ROBOPRO Phase 4 (APPROACH_MODE A/B): per-plan curobo effort and per-build
            # seed outcomes for this episode. rollout_plan_effort answers "did the seed
            # let curobo converge in fewer attempts"; rollout_seed_stats answers "how
            # often did the seed actually fire" (seeding is opportunistic -- IK is
            # nondeterministic, so build_seed sometimes returns no route for inputs that
            # worked a moment earlier). Both reset per-episode; env is reused across seeds.
            self.rollout_plan_effort = []
            self.rollout_seed_stats = []
            # Per-phase wall-clock timing: each checkpoint(name) records how long the
            # expert spent since the previous checkpoint. Read by run() and written
            # into the per-rollout log file. Reset per-episode (env reused across seeds).
            self.rollout_stage_times = []
            self._stage_clock = time.perf_counter()

            # Re-register the FULL collision world (clutter included) before any
            # planning. setup_demo ran update_world(exclude_obstacles=True) because the
            # bench config sets enable_collision_metrics=true -- that drops every
            # is_obstacle=True procedural clutter object from curobo's world, so the
            # expert would plan straight THROUGH clutter and knock it over (even on
            # "successful" rollouts). The whole point here is to test how well the
            # expert AVOIDS clutter, so curobo must see it: update_world() (no exclude)
            # rebuilds the world from collision_list including clutter. The physics-based
            # collision metrics (check_collisions, gated by enable_collision_metrics) are
            # independent of curobo's world, so they still measure any residual hits.
            self.update_world()

            def checkpoint(name):
                # Diagnostic: the dry-run candidate search only verifies the
                # waypoint/pre_grasp/grasp/lift/placement POSES are independently
                # reachable via a chained last_qpos rollout -- it never calls
                # attach_object/enable_table (which change the live collision world)
                # and never tests place_actor's final descent at all. A candidate can
                # pass the dry run and still die at any of these real-execution-only
                # steps, or at a step the dry run "verified" but which replans from a
                # genuinely different qpos in real execution. This pinpoints which.
                if not self.plan_success and self.rollout_failure_stage is None:
                    self.rollout_failure_stage = name
                    self.rollout_failure_reason = self._last_fail_reason
                now = time.perf_counter()
                self.rollout_stage_times.append(
                    {"stage": name, "seconds": round(now - self._stage_clock, 4)})
                self._stage_clock = now
                if log_move:
                    print(f"[play_once] after {name}: plan_success={self.plan_success}")
                # Optional stage observer (set by tooling, e.g. swept_volume_3d.py, to slice
                # the rollout into named stages). Only read via getattr; never set normally.
                stage_hook = getattr(self, "stage_hook", None)
                if stage_hook is not None:
                    stage_hook(name)

            target_p0 = self.target_obj.get_pose().p        # original target location (pre-grasp)
            self._place_target_y = float(target_p0[1])      # available to backward subgoals
            arm_tag = ArmTag("right" if target_p0[0] > 0 else "left")
            # Retry with a genuinely different contact-point candidate when
            # grasp_verify (below) detects a missed grasp, instead of ending the
            # episode after the first miss. _select_pick_place_candidate already
            # ranks/generates several candidates for the dry-run search --
            # exclude_cp_ids reuses that same ranking rather than re-deriving it.
            # Intermediate failed attempts reset plan_success/rollout_failure_*
            # so a LATER successful attempt isn't left alongside a stale failure
            # recording from an earlier, since-recovered-from miss.
            tried_cp_ids = set()
            candidate_plan = None
            cp_id = None
            grasp_succeeded = False
            retry_reset_pose = list(self.get_arm_pose(arm_tag))
            for grasp_attempt in range(1, GRASP_VERIFY_MAX_CANDIDATES + 1):
                candidate_plan = self._select_pick_place_candidate(arm_tag, exclude_cp_ids=tried_cp_ids)
                cp_id = candidate_plan["contact_point_id"]
                if cp_id is None:
                    # No viable contact-point candidate at all (e.g. the object's
                    # current geometry/pose gives _rank_side_grasp_ids nothing to
                    # rank) -- retrying can't help since nothing about the object
                    # or arm has changed. Executing the grasp anyway would fall
                    # through to an unconstrained default grasp
                    # (grasp_actor_from_table with contact_point_id=None) whose
                    # failure then overwrites the real cause with a generic
                    # grasp/lift error and reason=None. Fail fast with the real
                    # reason instead.
                    self.plan_success = False
                    self._last_fail_reason = "no_ranked_contact_points"
                    checkpoint("grasp_candidate_selection")
                    self.rollout_grasp_attempts.append({
                        "attempt": grasp_attempt,
                        "contact_point_id": None,
                        "success": False,
                        "failed_stage": self.rollout_failure_stage,
                        "fail_reason": self.rollout_failure_reason,
                    })
                    break
                tried_cp_ids.add(cp_id)

                # Belt-and-braces: _candidate_specs already emits grasp_waypoint=None outside
                # 'off' mode. If one ever leaks through again, the `else` below would live-plan
                # and EXECUTE the around-box heuristic, silently putting it back into the
                # waypoint-free modes and invalidating the A/B. Skip it loudly instead.
                if candidate_plan["grasp_waypoint"] is not None and self._approach_mode() != "off":
                    print("\033[93m[waypoint] BUG: a grasp_waypoint survived in "
                          f"APPROACH_MODE={self._approach_mode()}; skipping it\033[0m")
                elif candidate_plan["grasp_waypoint"] is not None:
                    wp_trajectory = candidate_plan.get("grasp_waypoint_trajectory")
                    if wp_trajectory is not None:
                        self._replay_planned_sequence(arm_tag, wp_trajectory)
                    else:
                        self._plan_and_replay_pose(arm_tag, candidate_plan["grasp_waypoint"])
                    checkpoint("waypoint_move")
                self._grasp_via_cached_trajectories(arm_tag, candidate_plan)
                checkpoint("grasp")

                pre_lift_object_z = float(self.target_obj.get_pose().p[2])
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=GRASP_LIFT_HEIGHT))
                checkpoint("lift")
                # Verify the grasp actually captured the object before trusting it. plan_success
                # only reflects whether CuRobo found a valid motion plan for the COMMANDED
                # gripper move -- it says nothing about whether the object moved with it.
                # Confirmed empirically (seeds 11/19/26): the gripper lifted the full commanded
                # GRASP_LIFT_HEIGHT while the object stayed frozen at its original table pose
                # (z-rise 0.0-0.6% of the commanded height) -- a fully missed grasp, not a
                # marginal one. Left unchecked, attach_object below bakes the object's real
                # (still-on-table) pose into CuRobo's collision model as if it were rigidly
                # held by the now-lifted gripper, producing a nonsensical attached-mesh
                # position that immediately collides with the table the instant
                # enable_table(True) runs a few lines down -- misattributed as a placement/
                # pre_beside_box planning bug rather than what it actually is: a failed grasp.
                if self.plan_success:
                    object_rise = float(self.target_obj.get_pose().p[2] - pre_lift_object_z)
                    if object_rise < GRASP_LIFT_HEIGHT * GRASP_VERIFY_MIN_RISE_FRACTION:
                        self.plan_success = False
                        self._last_fail_reason = f"grasp_missed(object_z_rise={object_rise:.3f}m)"
                checkpoint("grasp_verify")
                attempt_record = {
                    "attempt": grasp_attempt,
                    "contact_point_id": cp_id,
                    "success": self.plan_success,
                    "failed_stage": self.rollout_failure_stage,
                    "fail_reason": self.rollout_failure_reason,
                }
                self.rollout_grasp_attempts.append(attempt_record)
                if self.plan_success:
                    grasp_succeeded = True
                    break
                if grasp_attempt < GRASP_VERIFY_MAX_CANDIDATES and cp_id is not None:
                    if log_move:
                        print(f"[grasp_verify] attempt {grasp_attempt} missed (cp_id={cp_id}); "
                              f"retrying with next candidate")
                    self.plan_success = True
                    self.rollout_failure_stage = None
                    self.rollout_failure_reason = None
                    self._last_fail_reason = None
                    # The failed attempt closed the gripper; restore both the
                    # gripper and table state before selecting another contact point. The table must go back to
                    # ENABLED here, not stay disabled: _grasp_via_cached_
                    # trajectories' own waypoint_move/pre_grasp leg expects to
                    # plan WITH table collision active (that's why it only
                    # disables the table partway through, right before the
                    # close final approach) -- leaving it disabled would let
                    # the next candidate's waypoint/pre_grasp plan straight
                    # through the table.
                    self.move(self.open_gripper(arm_tag))
                    self.enable_table(enable=True)
                    # Candidate plans are generated from the live arm state. A
                    # missed grasp leaves the arm at its lifted endpoint, so
                    # selecting the next candidate there made every retry solve
                    # a different, frequently colliding problem. Return to the
                    # common pre-attempt pose before selecting another contact.
                    self._plan_and_replay_pose(arm_tag, retry_reset_pose)
                    attempt_record["reset_success"] = bool(self.plan_success)
                    if not self.plan_success:
                        reset_reason = self._last_fail_reason or "unknown"
                        attempt_record["reset_fail_reason"] = reset_reason
                        self._last_fail_reason = f"grasp_retry_reset_failed:{reset_reason}"
                        checkpoint("grasp_retry_reset")
                        break
                else:
                    break
            if not grasp_succeeded:
                # Retry cleanup can temporarily restore plan_success so the next
                # candidate can run. Exhausting the loop is still a failed grasp.
                self.plan_success = False
                if self._last_fail_reason is None:
                    self._last_fail_reason = "grasp_candidates_exhausted"
                checkpoint("grasp_verify")
                return
            self.attach_object(
                self.target_obj,
                f"{os.environ['BENCH_ROOT']}/assets/objects/{self.target_model}/collision/base{self.target_id}.glb",
                str(arm_tag),
            )
            checkpoint("attach_object")
            # Baseline for _object_retained: the object's full pose relative to the
            # gripper right now, while we still trust the grasp (grasp_verify just
            # passed). Every subsequent replay step compares against this.
            self._grasp_baseline_transform = self._gripper_relative_object_transform(arm_tag)
            self._slow_attached_replay = True
            self.enable_table(enable=True)
            checkpoint("enable_table")

            # Re-select the placement geometry under the REAL attached-object world,
            # but rank candidates by whether the FULL post-attach motion chain plans
            # successfully, not just by endpoint IK. Cache the winning trajectories so
            # these stages are replayed exactly instead of being planned again live.
            # PLACEMENT_MODE=direct: no subgoals, so there is no geometry to re-select --
            # skip the whole attached-world search (it is the scene-specific part: it scores
            # _box_side_x corridors against the PLACE_CLEARANCE_ZS ladder) and hand the lift
            # straight to place_actor below, via the one generic blended intermediate.
            if not candidate_plan["placement_subgoals"]:
                placement_plan = {"placement_subgoals": [], "trajectories": [], "verified": True}
                if log_move:
                    print("[placement] PLACEMENT_MODE=direct -- no placement subgoals; "
                          "lift -> intermediate -> place_actor")
            else:
                _beside_box_pose = candidate_plan["placement_subgoals"][0][1]
                _lift_z = _beside_box_pose[2]
                _quat = list(_beside_box_pose[3:])
                placement_plan = self._select_attached_placement_plan(
                    arm_tag,
                    tgt_y=self._place_target_y,
                    lift_z=_lift_z,
                    quat=_quat,
                )
                if log_move:
                    print(f"[placement] verified x_side={placement_plan['x_side']:.3f} "
                          f"clearance_z={placement_plan['clearance_z']:.2f} "
                          f"escape={placement_plan.get('escape_idx', 0)} "
                          f"quat={placement_plan['quat']}")
            candidate_plan["placement_subgoals"] = placement_plan["placement_subgoals"]
            for stage_name, traj in placement_plan["trajectories"]:
                self._replay_planned_move(arm_tag, traj)
                if self.plan_success and not self._object_retained(arm_tag, context=stage_name):
                    self.plan_success = False
                    self._last_fail_reason = f"object_lost:{stage_name}"
                checkpoint(stage_name)
            # _select_attached_placement_plan's search can exhaust every geometry/
            # escape combo without ever finding one that plans ALL the way through
            # (verified=False): it then falls back to the DEEPEST partial chain it
            # found, whose "trajectories" list is truncated at the stage that kept
            # failing (confirmed empirically: seed 8/9 checkpoint traces jumped
            # straight from an early placement subgoal to place_actor's
            # pre_place_descent, skipping lift_above_box/over_box_to_pad_y/
            # center_over_pad entirely, yet still reported rollout_success=True).
            # The loop above only iterates what's actually IN that truncated list,
            # so it silently "completes" without ever calling checkpoint() for the
            # missing subgoals -- plan_success never goes False. Make the fallback
            # an explicit, correctly-attributed failure instead of a silent skip.
            if self.plan_success and not placement_plan.get("verified", False):
                self.plan_success = False
                self._last_fail_reason = placement_plan.get("fail_reason", "unverified_placement_fallback")
                checkpoint(placement_plan.get("failed_stage") or "placement:unverified_fallback")
            # place_actor lowers the object from the last subgoal onto the pad.
            # constrain="free": check_success is position-only, so alignment buys nothing
            # and was a likely IK_FAIL cause for a tall object held near its top.
            # dis=0.02 (was 0.005): the wide-sweep checkpoints showed 3/3 episodes that
            # reached this step failed here with MotionGenStatus.FINETUNE_TRAJOPT_FAIL --
            # a 5mm final gap puts the held object's collision volume so close to the
            # table that trajopt's finetune can't converge on a smooth, collision-margin
            # -respecting descent. check_success() (put_mouse_on_pad.py) only checks x/y
            # position + gripper state, no z/height requirement, so a looser release
            # height costs nothing and gives finetune room to converge.
            # That alone didn't fix it: [place_actor] logging showed place_pre_pose ->
            # place_pose isn't a pure vertical drop, it also shifts diagonally in x/y,
            # and still hit FINETUNE_TRAJOPT_FAIL. CuRobo's own stacking example
            # (examples/isaac_sim/simple_stacking.py) uses PoseCostMetric.
            # create_grasp_approach_metric for exactly this: a straight-line approach
            # blended into only the last tstep_fraction of the trajectory. Unlike the
            # plain constraint_pose path (why DROP_GRASP_ORIENTATION_CONSTRAINT exists),
            # it sets offset_tstep_fraction >= 0, which skips CuRobo's start-vs-goal
            # orientation match check, so it won't reject a differing start/goal
            # orientation the way constraint_pose did. approach_axis=2 asks for a
            # straight-line descent (z) into both the pre-pose and final pose.
            place_arm_tag, place_actions = self.place_actor(
                self.target_obj, arm_tag=arm_tag, target_pose=self.des_obj_pose,
                constrain="free", pre_dis=0.05, dis=0.02,
            )
            if not place_actions:        # place_actor bails (plan_success already False)
                return                   # -> avoid place_actions[0] IndexError below
            for a in place_actions:
                if a.action == "move":
                    a.args["approach_axis"] = 2
            # None of the above fixed it: a pure-IK check (object attached, live qpos)
            # showed place_pre_pose/place_pose are BOTH independently reachable and
            # collision-free -- the endpoint is fine. So FINETUNE_TRAJOPT_FAIL here is a
            # path-smoothness problem, not a reachability/collision one: center_over_pad
            # -> place_pre_pose is one big trajopt problem (large vertical drop + lateral
            # shift + orientation change all at once) that the optimizer can't
            # interpolate smoothly, even though both ends are individually valid. Same
            # fix pattern as the grasp side: break it into a smaller intermediate step
            # (position+orientation blend) instead of one large jump.
            # The post-lift transit to place_pre_pose, which the two modes reach very differently.
            #
            # SCRIPTED (unchanged): the chain already ended at center_over_pad, directly above
            # the pad, so all that remains is the short blended hop described above.
            #
            # DIRECT: one move, straight from the lift pose to place_pre_pose, seeded.
            # This replaced a two-hop version (blend, then pad) for a reason worth recording.
            # The blend is an INTERMEDIATE, not a destination. With the chain on, that is
            # invisible -- starting above the pad, the blend lands at the pad. With the chain
            # off it starts at the LIFT pose and stops half way, which is what the first smoke
            # run hit: carry_transit planned Success and was the episode's LAST plan, then
            # landing_search_preconditions_not_met(xy_distance=0.206 m). Appending a second hop
            # fixed the reach but left the seed covering only the first half -- and on the
            # rerun that unseeded second hop was exactly where the seeded cell died
            # (FINETUNE_TRAJOPT_FAIL, 24/24 attempts), while its seeded first hop had converged
            # in ONE attempt with the object held to 4e-5 m.
            #
            # Splitting the move is also the wrong shape for the experiment. The seed's whole
            # claim is that it makes the big jump tractable, so the big jump is what has to be
            # attempted -- unseeded in `direct`, seeded in `seed`. Both cells keep the generic
            # replacement for the hand-placed midpoint: _plan_pose_with_local_waypoint_retry's
            # shrinking-waypoint fallback, which breaks a failed move into smaller hops using
            # CuRobo's own reported position error, reads no scene state, and applies equally
            # to both. So the midpoint is not lost, it is earned rather than assumed.
            if candidate_plan["placement_subgoals"]:
                transit_pose = self._verified_intermediate(
                    arm_tag, candidate_plan["placement_subgoals"][-1][1],
                    place_actions[0].target_pose)
            else:
                transit_pose = list(place_actions[0].target_pose)
            # Phase C seed: goal is the pose actually being planned to, which is now the whole
            # transit. None in every other mode, so this call is byte-identical to the unseeded
            # one there. (Scripted's plan-effort label changes from "pose" to "carry_transit";
            # that is a record label only, no behaviour rides on it.)
            _carry_seed = self._get_carry_seed(arm_tag, transit_pose[:3])
            self._plan_and_replay_pose(arm_tag, transit_pose, seed_traj=_carry_seed,
                                       stage_label="carry_transit")
            if self.plan_success and not self._object_retained(arm_tag, context="placement:pre_place_descent"):
                self.plan_success = False
                self._last_fail_reason = "object_lost:placement:pre_place_descent"
            checkpoint("placement:pre_place_descent")
            if log_move:
                # Diagnostic: is the endpoint itself reachable (pure collision-aware IK,
                # no trajectory optimization/smoothness), at the LIVE qpos with the
                # object actually attached right now? Distinguishes "the pose itself is
                # unreachable/colliding" from "the pose is fine but trajopt can't find a
                # smooth path there" -- four different trajopt/collision-margin fixes
                # all failed identically, so this tells us whether to keep tuning the
                # solver or reconsider the target pose itself.
                check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
                move_poses = [a.target_pose for a in place_actions if a.action == "move"]
                ik_ok = check_ik(move_poses)
                print(f"[place_actor] pure-IK reachability (attached, live qpos): {list(ik_ok)}")
            # Forcing CuRobo all the way down to place_actor's one exact final
            # pose (via _execute_actions_via_plan_and_replay, retries 4 -> 10)
            # was the prior approach here; the broad-sweep validation (seeds
            # 8-31) confirmed it made place_actor the dominant remaining
            # failure (100% MotionGenStatus.FINETUNE_TRAJOPT_FAIL), since the
            # attached object's collision margin against the table is
            # tightest exactly at that one pose. Searching a small grid of
            # nearby landing poses instead -- accepting any that plans
            # cleanly within check_success's own tolerance, then letting
            # physics settle the final contact -- avoids ever asking CuRobo
            # to solve that one hardest point.
            self._local_landing_search_and_place(place_arm_tag)
            checkpoint("place_actor")
















































    return OccluderTask
