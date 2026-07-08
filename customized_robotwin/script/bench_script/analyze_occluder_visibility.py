"""
Phase 2 part 1 (#35): deterministic single-occluder spawn + visibility distribution.

Places the target (001_bottle id 9) in the upper third of the reachable region (in
front of the back furniture) and spawns ONE tall milk-box occluder (038_milk-box id 2,
scale 1.0) in a narrow band just in front (-y, robot/camera side) of the target -- no
clutter, no binary search. Then measures the t=0 countertop `visible_fraction`
distribution across seeds, reusing the Phase 1 harness (measurement + plotting).

visible_fraction = visible_target_px(with occluder) / full_target_px(no occluder),
same seed (target pose is fixed before the occluder is added).

USAGE (from the benchmark folder):
    cd benchmark
    source set_env.sh
    export ROBOTWIN_BENCH_TASK=bench
    python script/bench_script/analyze_occluder_visibility.py \
        --seed-start 0 --num-seeds 50 --offsets 0.10 --bins 20 \
        --out-dir ../scripts/validation/results/phase2_occluder

    # narrow-region sweep (a few offsets) to see how distance affects occlusion:
    python script/bench_script/analyze_occluder_visibility.py \
        --num-seeds 40 --offsets 0.07,0.10,0.13 \
        --out-dir ../scripts/validation/results/phase2_occluder

    # re-plot only:
    python script/bench_script/analyze_occluder_visibility.py --plot-only \
        --out-dir ../scripts/validation/results/phase2_occluder
"""
import os
import json
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import transforms3d as t3d

from setup_paths import setup_paths
setup_paths()
# these analysis scripts are bench-only; default the task mode so build_cfg looks
# under bench_task_config/ (explicit ROBOTWIN_BENCH_TASK still wins)
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
os.chdir(robotwin_root)

from envs.utils import rand_pose, create_actor, create_box, ArmTag
from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from visualize_task_scene import get_env_class
# reuse the Phase 1 harness verbatim
from analyze_natural_visibility import (build_cfg, DR_CLEAN, save_overlay, analyze, _resolve_target,
                                        CAMERA, run_rollout, analyze_rollout, effective_out_dir)

# target object: 001_bottle id 9 (8-way grasp ring -> side-reaches around the occluder).
# Swap TARGET_MODEL/TARGET_ID back to 047_mouse id 0 for the stock top-down mouse.
TARGET_MODEL = "001_bottle"
TARGET_ID = 9
# target spawn: back half of the table (back furniture removed). Capped at y=0.20 so it
# stays 0.15 m off the back edge (table y in [-0.35, 0.35]); x kept to [-0.15, 0.15] so it
# stays 0.45 m off each side edge (table x in [-0.6, 0.6]) -> leaves room for the around-
# the-box side waypoint and keeps target + waypoint inside the arm's reach envelope.
TARGET_XLIM = (-0.15, 0.15)
TARGET_YLIM = (0.0, 0.20)
PAD_XY = (0.0, -0.28)          # destination pad parked at the front, out of the occluder zone
OCCLUDER_QPOS = [0.66, 0.66, -0.25, -0.25]   # same upright carton orientation as the stock milk-box

# Two-step (around-the-box) planning: curobo won't reliably route around a tall obstacle
# between start and goal, so before the grasp (and before the place) we send the gripper
# to a waypoint beside the occluder on the arm's own side, then reach in. The waypoint's
# x-offset from the box CENTRE = OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP, i.e. SIDE_WAYPOINT
# _GAP is the clearance from the box EDGE to the gripper (not from centre). If the waypoint
# would leave the reachable x-range we flip to the other (table-centre) side. All tunable.
OCC_HALF_FOOTPRINT = 0.08      # milk-box base half-diagonal (~0.11 x 0.122, incl. yaw)
SIDE_WAYPOINT_GAP = 0.24       # clearance from the box EDGE to the waypoint (gripper)
REACH_X_LIMIT = 0.5
GRASP_CANDIDATE_LIMIT = 4
# The pre_beside_box move (right after lift+attach_object+enable_table) was found
# empirically to fail with MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION in
# 100% of its failures (5/5 in a 24-seed sweep) -- meaning the STARTING qpos for
# that plan is already colliding with the just-re-enabled table, before any target
# pose even matters. A 0.1m lift isn't always enough clearance for the held
# object's true collision volume (voxelized via attach_object) to clear the table
# once its collision is turned back on. Bumping the post-grasp lift height directly
# targets this; must stay consistent across dry-run and real execution.
# 0.15 (first attempt) reduced this failure 5/5 -> 4/24 seeds and raised the
# occluder-present success rate to 5/16. Pushing further to 0.2 was WORSE on both
# counts (4/16 successes, 6 pre_beside_box collisions) -- not monotonic, likely
# because more lift shifts the arm into a different (sometimes worse) posture
# rather than simply buying more clearance. Reverted to the better-validated 0.15;
# some of the 0.2 regression is plausibly sampling noise (documented CuRobo
# run-to-run nondeterminism) rather than a true causal effect, but 0.15 is the
# only value with two consistent, positive data points.
GRASP_LIFT_HEIGHT = 0.15
SIDE_WAYPOINT_GAPS = (0.20, 0.24, 0.28)
# A direct IK-reachability sweep (check_ik_batch diagnostic) showed the waypoint's
# feasibility roughly DOUBLES at +0.15m above grasp height (64%) vs the +0.0/+0.06m
# band this used to test alone (29-36%) -- x/gap and going even higher (+0.30, 54%)
# both matter far less. +0.15 is the empirically-found sweet spot, not a guess.
SIDE_WAYPOINT_Z_LIFTS = (0.0, 0.06, 0.15)
PLACE_CLEARANCE_ZS = (1.05, 1.15, 1.25)
# Both a second ("top_down") waypoint orientation and a +/-0.05m y-offset search were
# tried here and measured empirically to fail at the SAME rate as the baseline (both
# orientations: ~equal IK_FAIL proportion; y-offset: no change in failure proportion
# either) -- i.e. neither is the actual bottleneck (that's pure kinematic unreachability
# at the position itself). Collapsed back to single values so this search space doesn't
# multiply the cost of the (expensive, real CuRobo batch-plan) grasp-pose check below
# for no measured benefit. Re-widen only with new evidence, not speculatively.
WAYPOINT_ORIENTATIONS = ("grasp_aligned",)
WAYPOINT_Y_OFFSETS = (0.0,)
# Ordered chain stages (see _plan_candidate) -- used to rank how far a candidate got
# before failing, so the fallback (used when NO candidate fully plans) can pick the
# one that progressed furthest instead of whichever was generated first. Confirmed via
# a check_ik_batch reach-envelope probe: a fully-reachable contact point (both pre_grasp
# and grasp) can exist among the ones tried, but the old "first-generated" fallback
# picked an unreachable one instead because it happened to rank #1 geometrically.
STAGE_ORDER = ["waypoint", "pre_grasp", "grasp", "lift",
               "beside_box", "lift_above_box", "over_box_to_pad_y", "center_over_pad"]

# Reject a (seed, offset) build if the occluder would sit within OCC_PAD_CLEARANCE
# (edge-to-edge) of the destination pad, so the occluder can never block/overlap the
# target's final placement. Center-to-center threshold = pad_half + occluder_half + gap.
OCC_PAD_CLEARANCE = 0.05
_PAD_HALF = 0.06               # pad half-size (create_box half_size xy)
_OCC_HALF = 0.06               # milk-box base half-extent (~0.11 x 0.122 footprint)
OCC_PAD_MIN_DIST = _PAD_HALF + _OCC_HALF + OCC_PAD_CLEARANCE


def dr_measure(clutter_density):
    """Domain-randomization for the measurement build: optional default (non-curated)
    table clutter at the given density. Density 0 -> clean (occluder only)."""
    if clutter_density and clutter_density > 0:
        return {"cluttered_table": True, "obstacle_density": int(clutter_density),
                "clean_background_rate": 0}
    return dict(DR_CLEAN)


def make_occluder_task():
    Base = get_env_class("put_mouse_on_pad", bench_subdir="office")

    class OccluderTask(Base):
        # Remove the back furniture (shelf/cabinet/file-holder) for now to free up
        # gripper workspace. Flip back to True (or delete) to restore the full office.
        SPAWN_BACK_FURNITURE = False
        # Lever A: drop the orientation-hold constraint on the final grasp approach so
        # curobo won't reject it with INVALID_PARTIAL_POSE_COST_METRIC. Set False (or
        # delete this flag + the grasp_actor_from_table override below) to restore the
        # stock straight-in, wrist-locked constrained approach.
        DROP_GRASP_ORIENTATION_CONSTRAINT = True
        # When True, play_once stops right after the lift (pickup only) -- used by
        # pickup_reachability_map.py to reach the picked-up state, then probe IK.
        PICKUP_ONLY = False
        spawn_occluder = False
        occluder_offset = 0.2         # metres in front (-y) of the target
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

            # occluder: one tall milk-box, deterministically at mouse-x, just in front (-y)
            if self.spawn_occluder:
                mp = self.target_obj.get_pose().p
                ox, oy = float(mp[0]), float(mp[1]) - self.occluder_offset
                occ_pose = rand_pose(xlim=[ox], ylim=[oy], qpos=OCCLUDER_QPOS,
                                     rotate_rand=True, rotate_lim=[0, 3.14, 0])  # random yaw
                self.occluder = create_actor(
                    scene=self, pose=occ_pose, modelname="038_milk-box", convex=True,
                    model_id=2, scale=[1.0, 1.0, 1.0],
                )
                self.occluder.set_mass(0.1)
                # Register the occluder in curobo's collision world so the planner
                # actually avoids it. NOT flagged is_obstacle -> it survives the
                # exclude_obstacles=True pass used in collision-metrics/eval mode
                # ("... curobo planner skips clutter obstacles"), which only drops
                # is_obstacle=True procedural clutter.
                self.collision_list.append({
                    "actor": self.occluder,
                    "collision_path": f"{os.environ['BENCH_ROOT']}/assets/objects/038_milk-box/collision/base2.glb",
                })

        def play_once(self):
            # Expert plan. FORWARD (grasp) below is fixed. BACKWARD (placement) is a list
            # of subgoals returned by self._backward_subgoal_poses() -- EDIT THAT METHOD to
            # add / remove / reorder placement subgoals; each subgoal is one curobo move.
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"

            def checkpoint(name):
                # Diagnostic: the dry-run candidate search only verifies the
                # waypoint/pre_grasp/grasp/lift/placement POSES are independently
                # reachable via a chained last_qpos rollout -- it never calls
                # attach_object/enable_table (which change the live collision world)
                # and never tests place_actor's final descent at all. A candidate can
                # pass the dry run and still die at any of these real-execution-only
                # steps, or at a step the dry run "verified" but which replans from a
                # genuinely different qpos in real execution. This pinpoints which.
                if log_move:
                    print(f"[play_once] after {name}: plan_success={self.plan_success}")

            target_p0 = self.target_obj.get_pose().p        # original target location (pre-grasp)
            self._place_target_y = float(target_p0[1])      # available to backward subgoals
            arm_tag = ArmTag("right" if target_p0[0] > 0 else "left")
            candidate_plan = self._select_pick_place_candidate(arm_tag)
            cp_id = candidate_plan["contact_point_id"]

            if candidate_plan["grasp_waypoint"] is not None:
                wp_trajectory = candidate_plan.get("grasp_waypoint_trajectory")
                if wp_trajectory is not None:
                    self._replay_planned_move(arm_tag, wp_trajectory)
                else:
                    self.move(self.move_to_pose(arm_tag, candidate_plan["grasp_waypoint"]))
                checkpoint("waypoint_move")
            self._grasp_via_cached_trajectories(arm_tag, candidate_plan)
            checkpoint("grasp")

            self.move(self.move_by_displacement(arm_tag=arm_tag, z=GRASP_LIFT_HEIGHT))
            checkpoint("lift")
            if self.PICKUP_ONLY:        # stop at the picked-up state (reachability probe)
                return
            self.attach_object(
                self.target_obj,
                f"{os.environ['BENCH_ROOT']}/assets/objects/{self.target_model}/collision/base{self.target_id}.glb",
                str(arm_tag),
            )
            checkpoint("attach_object")
            self.enable_table(enable=True)
            checkpoint("enable_table")

            # Re-verify the placement subgoals at real-execution time instead of
            # trusting whatever the dry-run candidate search baked in. Diagnostics
            # showed beside_box/lift_above_box failures (seeds 9/11/12) are genuinely
            # UNREACHABLE endpoints (pure-IK False) at every gap/clearance_z in the
            # dry-run's own search -- a kinematic dead zone with the grasp-inherited
            # orientation, not a path-smoothness problem like place_actor was.
            _beside_box_pose = candidate_plan["placement_subgoals"][0][1]
            _lift_z = _beside_box_pose[2]
            _quat = list(_beside_box_pose[3:])
            verified_x_side, verified_clearance_z, verified_quat = self._pick_reachable_placement_geometry(
                arm_tag, self._place_target_y, _lift_z, _quat)
            candidate_plan["placement_subgoals"] = self._backward_subgoal_poses(
                arm_tag, x_side=verified_x_side, clearance_z=verified_clearance_z,
                lift_pose=[0, 0, _lift_z] + verified_quat,
            )
            if log_move:
                print(f"[placement] verified x_side={verified_x_side:.3f} clearance_z={verified_clearance_z:.2f} "
                      f"quat={verified_quat}")

            # Same path-smoothness pattern as place_actor's pre_place_descent: seed 11
            # showed beside_box's pose can be pure-IK verified reachable and STILL fail
            # in real execution -- a large single jump (lift_pose -> beside_box, often
            # combining a big lateral shift with an orientation change) that trajopt
            # can't smoothly interpolate, even though both ends are individually valid.
            # Insert a blended intermediate step for the same reason it helped there.
            mid_to_beside_box = self._verified_intermediate(arm_tag, list(self.get_arm_pose(arm_tag)), candidate_plan["placement_subgoals"][0][1])
            self.move(self.move_to_pose(arm_tag, mid_to_beside_box))
            checkpoint("placement:pre_beside_box")

            prev_pose = None  # overwritten before use -- "beside_box" is always first
            for name, pose in candidate_plan["placement_subgoals"]:
                if name == "lift_above_box":
                    # Same large-jump shape as place_actor's pre_place_descent and the
                    # arm-pose -> beside_box entry: beside_box (~grasp height, ~0.85m)
                    # to lift_above_box (clearance_z, 1.05-1.25m) is another single big
                    # vertical jump with no intermediate waypoint. Seeds 9/12 die here;
                    # unlike seeds 11/19's beside_box failures (confirmed genuinely
                    # unreachable at every geometry combo via pure-IK), lift_above_box's
                    # pose IS pure-IK verified reachable by _pick_reachable_placement_
                    # geometry -- so a failure here is the same path-smoothness problem,
                    # not a dead zone. Apply the identical fix.
                    mid_to_lift_above_box = self._verified_intermediate(arm_tag, prev_pose, pose)
                    self.move(self.move_to_pose(arm_tag, mid_to_lift_above_box))
                    checkpoint("placement:pre_lift_above_box")
                if log_move and self.plan_success:
                    # Same diagnostic as place_actor below: is THIS subgoal reachable
                    # via pure collision-aware IK (object attached, live qpos), before
                    # we attempt the real trajopt move? Confirms whether a failing
                    # subgoal here is a genuine reachability problem or -- like
                    # place_actor -- a path-smoothness problem worth an intermediate
                    # waypoint instead.
                    check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
                    ik_ok = check_ik([pose])
                    print(f"[placement:{name}] pure-IK reachability (attached, live qpos): {list(ik_ok)}")
                self.move(self.move_to_pose(arm_tag, pose))
                checkpoint(f"placement:{name}")
                prev_pose = pose
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
            mid_pose = self._verified_intermediate(arm_tag, candidate_plan["placement_subgoals"][-1][1], place_actions[0].target_pose)
            self.move(self.move_to_pose(arm_tag, mid_pose))
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
            self._execute_actions_via_plan_and_replay(place_arm_tag, place_actions, retries=4)
            checkpoint("place_actor")

        def _blend_pose(self, pose_a, pose_b, t=0.5):
            """Position lerp + quaternion nlerp (normalized linear interpolation --
            a cheap, good-enough slerp approximation for a transit waypoint that
            doesn't need to be exact) between two flat [x,y,z,qw,qx,qy,qz] poses.
            Used to insert an intermediate step in a large single move so trajopt
            gets a smaller, easier motion to solve instead of one big jump."""
            pa, pb = np.asarray(pose_a, dtype=float), np.asarray(pose_b, dtype=float)
            pos = (1 - t) * pa[:3] + t * pb[:3]
            qa, qb = pa[3:], pb[3:]
            if np.dot(qa, qb) < 0:      # quaternion double-cover: align signs before blending
                qb = -qb
            q = (1 - t) * qa + t * qb
            q = q / np.linalg.norm(q)
            return list(pos) + list(q)

        def _verified_intermediate(self, arm_tag, pose_a, pose_b, fractions=(0.5, 0.35, 0.65, 0.25, 0.75)):
            """Same verify-before-committing principle as
            _pick_reachable_placement_geometry, applied to a transit waypoint: a naive
            midpoint blend can itself be unreachable (confirmed empirically -- seed
            11's pre_beside_box waypoint failed at the plain t=0.5 blend). Try a small
            spread of blend fractions and use whichever verifies reachable via
            check_ik_batch, falling back to the plain midpoint if none do."""
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            candidates = [self._blend_pose(pose_a, pose_b, t) for t in fractions]
            ok = check_ik(candidates)
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[intermediate] fractions={fractions} ok={list(ok)}")
            for candidate, reachable in zip(candidates, ok):
                if reachable:
                    return candidate
            return candidates[0]  # none verified reachable; fall back to the plain midpoint

        def _backward_subgoal_poses(self, arm_tag, x_side, clearance_z, lift_pose):
            """Ordered BACKWARD (placement) subgoals for a chosen candidate path."""
            quat = list(lift_pose[3:])
            tgt_y = self._place_target_y
            pad_x, pad_y = self.fixed_pad_xy
            lift_z = float(lift_pose[2])
            subgoals = []
            subgoals.append(("beside_box", [x_side, tgt_y, lift_z, *quat]))
            subgoals.append(("lift_above_box", [x_side, tgt_y, clearance_z, *quat]))
            subgoals.append(("over_box_to_pad_y", [x_side, pad_y, clearance_z, *quat]))
            subgoals.append(("center_over_pad", [pad_x, pad_y, clearance_z, *quat]))
            return subgoals

        def _pick_reachable_placement_geometry(self, arm_tag, tgt_y, lift_z, quat,
                                               gaps=SIDE_WAYPOINT_GAPS, clearance_options=PLACE_CLEARANCE_ZS):
            """beside_box/lift_above_box can be genuinely unreachable (pure-IK False)
            at EVERY gap/clearance_z combo while holding the grasp-inherited
            orientation -- the same kinematic-dead-zone pattern the original side
            -waypoint search hit, which needed an orientation search (not just
            position) to escape. quat comes from choose_best_pose's SHORTEST-PATH
            grasp rotation, which optimizes for reaching the grasp, not for what
            happens after -- so it isn't guaranteed workable for placement either.
            Searches orientation (grasp-inherited vs top_down) x gap for beside_box,
            then clearance_z within the winning orientation for lift_above_box, all
            verified via check_ik_batch instead of trusting the dry-run candidate's
            baked-in geometry. Returns (x_side, clearance_z, quat_to_use)."""
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            quat_options = (quat, GRASP_DIRECTION_DIC["top_down"])
            fallback = None  # best partial match: beside_box verified, but no clearance_z did
            for quat_choice in quat_options:
                x_sides = [self._box_side_x(arm_tag, gap=g) for g in gaps]
                beside_box_ok = check_ik([[x, tgt_y, lift_z, *quat_choice] for x in x_sides])
                if log_move:
                    print(f"[placement] beside_box quat={quat_choice} gaps={gaps} ok={list(beside_box_ok)}")
                for x, reachable in zip(x_sides, beside_box_ok):
                    if not reachable:
                        continue
                    cz_ok = check_ik([[x, tgt_y, cz, *quat_choice] for cz in clearance_options])
                    if log_move:
                        print(f"[placement] lift_above_box x_side={x:.3f} quat={quat_choice} "
                              f"clearance_options={clearance_options} ok={list(cz_ok)}")
                    for cz, cz_reachable in zip(clearance_options, cz_ok):
                        if cz_reachable:
                            return x, cz, quat_choice   # both beside_box AND lift_above_box verified
                    if fallback is None:
                        fallback = (x, clearance_options[0], quat_choice)  # beside_box ok; keep searching
            if fallback is not None:
                return fallback
            # nothing verified at all; fall back to the original defaults
            return self._box_side_x(arm_tag, gap=gaps[0]), clearance_options[0], quat

        def _box_side_x(self, arm_tag, gap=SIDE_WAYPOINT_GAPS[1]):
            """World x beside the occluder on the arm's OWN side (right -> +x, left -> -x),
            OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP from the box centre, clamped to the reach
            limit on that same side (never flipped across to the wrong side)."""
            if self.spawn_occluder and getattr(self, "occluder", None) is not None:
                box_x = float(self.occluder.get_pose().p[0])
            else:
                box_x = float(self.target_obj.get_pose().p[0])
            side = 1.0 if str(arm_tag) == "right" else -1.0
            x = box_x + side * (OCC_HALF_FOOTPRINT + gap)
            return side * REACH_X_LIMIT if abs(x) > REACH_X_LIMIT else x

        def _around_box_waypoint(self, arm_tag, ref_pose, gap=SIDE_WAYPOINT_GAPS[1], z_lift=0.0,
                                  orient="grasp_aligned", y_offset=0.0):
            """Grasp subgoal: a pose beside the occluder (via _box_side_x), near the box's
            horizontal (y) line. Position comes from ref_pose's height + the side-waypoint
            offset, with y = box's y-line + y_offset (searched around 0 -- pinning exactly
            to the box's y-line was found to be IK-unreachable at some seeds regardless of
            x/orientation). Orientation is either ref_pose's own (grasp_aligned -- matches
            the eventual reach-in angle) or a neutral top-down orientation (orient=
            "top_down"). None when no occluder is present."""
            if not (self.spawn_occluder and getattr(self, "occluder", None) is not None):
                return None
            wp = list(ref_pose)
            wp[0] = self._box_side_x(arm_tag, gap=gap)
            wp[1] = float(self.occluder.get_pose().p[1]) + float(y_offset)   # box's y-line +/- offset
            wp[2] += float(z_lift)
            if orient == "top_down":
                wp[3:] = GRASP_DIRECTION_DIC["top_down"]
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[around_box] arm={arm_tag} orient={orient} y_offset={y_offset:.2f} -> waypoint "
                      f"x={wp[0]:.3f} y={wp[1]:.3f} z={wp[2]:.3f}")
            return wp

        def _rank_side_grasp_ids(self, actor, arm_tag, limit=GRASP_CANDIDATE_LIMIT):
            """Rank a few side grasps so we can search across candidates instead of
            committing to exactly one local optimum."""
            side = 1.0 if str(arm_tag) == "right" else -1.0
            cx = float(actor.get_pose().p[0])
            conv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            ranked = []
            for i, cm in actor.iter_contact_points("matrix"):
                if cm is None:
                    continue
                g = np.asarray(cm, dtype=float) @ conv
                R = g[:3, :3]
                approach = R @ np.array([1.0, 0.0, 0.0])      # gripper +x approach (world)
                horiz = abs(float(approach[2]))               # 0 = perfectly horizontal
                if horiz > 0.35:                              # skip clearly tilted/vertical
                    continue
                grasp_x = float(g[0, 3]) + float((R @ np.array([-0.12, 0.0, 0.0]))[0])
                arm_side = side * (grasp_x - cx)              # >0 = gripper on arm's side
                score = arm_side - horiz                      # arm-side + as horizontal as possible
                ranked.append((score, i))
            ranked.sort(reverse=True)
            ids = [i for _, i in ranked[:limit]]
            if not ids:
                fallback = self._pick_side_grasp_id(actor, arm_tag)
                ids = [] if fallback is None else [fallback]
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[grasp_id] arm={arm_tag} candidates={ids}")
            return ids

        def _pick_side_grasp_id(self, actor, arm_tag):
            ranked = self._rank_side_grasp_ids(actor, arm_tag, limit=1)
            return ranked[0] if ranked else None

        def _geometric_grasp_pose(self, actor, cp_id, pre_dis=0.0):
            """Grasp pose for a contact point, computed geometrically (same math as
            get_grasp_pose but WITHOUT choose_best_pose's planning gate). Used only to
            orient/place the side waypoint, so a box-blocked direct plan can't veto it."""
            cm = actor.get_contact_point(cp_id, "matrix")
            if cm is None:
                return None
            conv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            g = np.asarray(cm, dtype=float) @ conv
            R = g[:3, :3]
            p = g[:3, 3] + R @ np.array([-0.12 - pre_dis, 0.0, 0.0])
            q = t3d.quaternions.mat2quat(R)
            return list(p) + list(q)

        def _replay_planned_move(self, arm_tag, cached_result):
            """Execute an ALREADY-PLANNED trajectory (position/velocity arrays from a
            prior plan_path call) directly via take_dense_action, skipping a fresh
            plan_path call entirely. Mirrors left_move_to_pose/right_move_to_pose's
            need_plan/joint_path bookkeeping but never re-plans.

            Why this exists: CuRobo's trajopt has no uniqueness guarantee for a given
            Cartesian target -- re-planning to the SAME pose independently (as
            left_move_to_pose/right_move_to_pose normally do) can converge to a
            genuinely different, equally-valid arm posture. Confirmed empirically
            (qpos drift up to 1.47 rad between a dry-run-verified plan and a fresh
            live re-plan to the identical waypoint pose), and that drift flips
            downstream grasp reachability in ~50% of cases -- explaining most of the
            grasp-stage failures seen despite the dry run having just verified a
            working grasp. Replaying the exact verified trajectory removes the
            divergence: the live qpos after this call is guaranteed to match what the
            dry run assumed, since it's the same trajectory, not a new one."""
            if not self.plan_success:
                return
            if cached_result is None or cached_result.get("status") != "Success":
                self.plan_success = False
                return
            if str(arm_tag) == "left":
                if self.need_plan:
                    self.left_joint_path.append(deepcopy(cached_result))
                else:
                    cached_result = deepcopy(self.left_joint_path[self.left_cnt])
                    self.left_cnt += 1
                control_seq = {"left_arm": cached_result, "left_gripper": None, "right_arm": None, "right_gripper": None}
            else:
                if self.need_plan:
                    self.right_joint_path.append(deepcopy(cached_result))
                else:
                    cached_result = deepcopy(self.right_joint_path[self.right_cnt])
                    self.right_cnt += 1
                control_seq = {"left_arm": None, "left_gripper": None, "right_arm": cached_result, "right_gripper": None}
            self.take_dense_action(control_seq)

        def _execute_actions_via_plan_and_replay(self, arm_tag, actions, retries=1):
            """Plan every 'move' Action in the list upfront (chained through a
            tracked virtual qpos, exactly like _plan_grasp_side), then replay each
            via _replay_planned_move; non-move actions (gripper open/close) execute
            normally through self.move. Extends the same plan-once-then-replay
            pattern already applied to the waypoint and grasp moves to place_actor.
            Unlike those, there's no separate earlier dry-run pass here (place_actor's
            target poses are computed live, not pre-verified) -- so this doesn't
            eliminate a proven duplicate-plan divergence the way it did there.

            retries: place_actor's failures are 100% MotionGenStatus.FINETUNE_TRAJOPT_FAIL
            (confirmed empirically) -- a marginal-feasibility trajopt-difficulty problem,
            not a reachability one. We've directly observed the SAME setup succeed on
            one run and fail on another (CuRobo's trajopt has internal randomized
            seeding), so retrying a failed plan_func call with fresh internal seeding
            has a real chance of finding a solution the first attempt missed. Each
            retry re-attempts ONLY the failed action, not the whole chain."""
            if not self.plan_success:
                return
            plan_func = self._arm_plan_func(arm_tag)
            start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            trajectories = []
            for a in actions:
                if a.action != "move":
                    continue
                for attempt in range(retries):
                    result = plan_func(a.target_pose, last_qpos=qpos,
                                       constraint_pose=a.args.get("constraint_pose"),
                                       approach_axis=a.args.get("approach_axis"))
                    if result.get("status") == "Success":
                        break
                    if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                        print(f"[plan_and_replay] attempt {attempt + 1}/{retries} failed "
                              f"({result.get('fail_reason')}) for target {np.round(np.asarray(a.target_pose, dtype=float), 4)}")
                if result.get("status") != "Success":
                    self.plan_success = False
                    return
                trajectories.append(result)
                qpos = self._roll_qpos_forward(arm_tag, qpos, result)
            for traj in trajectories:
                self._replay_planned_move(arm_tag, traj)
                if not self.plan_success:
                    return
            for a in actions:
                if a.action != "move":
                    self.move((arm_tag, [a]))
                    if not self.plan_success:
                        return

        def _grasp_via_cached_trajectories(self, arm_tag, candidate_plan, gripper_pos=0.0):
            """Replay the dry-run-verified pre_grasp/grasp trajectories (see
            _plan_grasp_side) instead of calling grasp_actor_from_table, which
            independently re-derives and re-plans the grasp via choose_grasp_pose --
            exactly the re-plan-diverges-from-verified-plan failure mode already
            fixed for the waypoint, recurring one step later. Table is disabled
            between the two moves, matching grasp_actor_from_table's own sequencing
            (drop the table right after clearing pre_grasp, before the final
            approach). Falls back to the live re-plan path if no cached trajectory
            exists (e.g. a fallback candidate that never got this far)."""
            pre_traj = candidate_plan.get("pre_grasp_trajectory")
            grasp_traj = candidate_plan.get("grasp_trajectory")
            if pre_traj is None or grasp_traj is None:
                return self.grasp_actor_from_table(
                    self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1,
                    contact_point_id=candidate_plan["contact_point_id"])
            self._replay_planned_move(arm_tag, pre_traj)
            self.enable_table(enable=False)
            self._replay_planned_move(arm_tag, grasp_traj)
            self.move(self.close_gripper(arm_tag, pos=gripper_pos))

        def grasp_actor_from_table(self, actor, arm_tag, pre_grasp_dis=0.1, grasp_dis=0,
                                   gripper_pos=0.0, contact_point_id=None):
            # Same as Office_base_task.grasp_actor_from_table, but (Lever A) strips the
            # orientation-hold constraint_pose from the approach actions before executing.
            if not self.DROP_GRASP_ORIENTATION_CONSTRAINT:
                return super().grasp_actor_from_table(
                    actor, arm_tag, pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis,
                    gripper_pos=gripper_pos, contact_point_id=contact_point_id)
            _, actions = self.grasp_actor(
                actor=actor, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis, gripper_pos=gripper_pos, contact_point_id=contact_point_id)
            if not actions:            # grasp_actor bails (e.g. an earlier move set
                return                 # plan_success=False) -> avoid actions[0] IndexError
            for a in actions:
                if getattr(a, "args", None):
                    a.args.pop("constraint_pose", None)  # -> unconstrained plan in move()
            self.move((arm_tag, [actions[0]]))
            self.enable_table(enable=False)
            self.move((arm_tag, actions[1:]))

        def _arm_plan_func(self, arm_tag):
            return self.robot.left_plan_path if str(arm_tag) == "left" else self.robot.right_plan_path

        def _arm_active_joint_indices(self, arm_tag):
            planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
            return [planner.all_joints.index(name) for name in planner.active_joints_name if name in planner.all_joints]

        def _roll_qpos_forward(self, arm_tag, qpos, plan_result):
            qpos_next = np.array(qpos, dtype=np.float64, copy=True)
            active_indices = self._arm_active_joint_indices(arm_tag)
            qpos_next[active_indices] = np.asarray(plan_result["position"][-1], dtype=np.float64)
            return qpos_next

        def _plan_pose_sequence(self, arm_tag, poses, start_qpos, stage_labels=None):
            """Returns (success, failed_stage_label, fail_reason, final_qpos).
            failed_stage_label/fail_reason are None on success; failed_stage_label is
            the label (from stage_labels, positionally aligned with poses) of the
            first pose in the chain that failed to plan, and fail_reason is CuRobo's
            own MotionGenStatus string for that failure (e.g. IK_FAIL vs a
            collision-specific status). final_qpos is the qpos reached by the last
            successfully-planned pose (== start_qpos if the very first pose failed),
            so a caller can resume planning a follow-on sub-chain from where this one
            left off (e.g. after toggling world state between segments)."""
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            plan_func = self._arm_plan_func(arm_tag)
            for idx, pose in enumerate(poses):
                result = plan_func(pose, last_qpos=qpos)
                if result.get("status") != "Success":
                    label = stage_labels[idx] if stage_labels else f"pose{idx}"
                    return False, label, result.get("fail_reason", "unknown"), qpos
                qpos = self._roll_qpos_forward(arm_tag, qpos, result)
            return True, None, None, qpos

        def _candidate_specs(self, arm_tag, cp_ids):
            for cp_id in cp_ids:
                grasp_pre_pose = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.1)
                grasp_pose = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.0)
                if grasp_pre_pose is None or grasp_pose is None:
                    continue
                lift_pose = list(grasp_pose)
                lift_pose[2] += GRASP_LIFT_HEIGHT
                occluder_present = self.spawn_occluder and getattr(self, "occluder", None) is not None
                gaps = SIDE_WAYPOINT_GAPS if occluder_present else (SIDE_WAYPOINT_GAPS[1],)
                z_lifts = SIDE_WAYPOINT_Z_LIFTS if occluder_present else (0.0,)
                orients = WAYPOINT_ORIENTATIONS if occluder_present else ("grasp_aligned",)
                y_offsets = WAYPOINT_Y_OFFSETS if occluder_present else (0.0,)
                for gap in gaps:
                    x_side = self._box_side_x(arm_tag, gap=gap)
                    for z_lift in z_lifts:
                        for orient in orients:
                            for y_offset in y_offsets:
                                grasp_waypoint = self._around_box_waypoint(
                                    arm_tag, grasp_pre_pose, gap=gap, z_lift=z_lift,
                                    orient=orient, y_offset=y_offset)
                                for clearance_z in PLACE_CLEARANCE_ZS:
                                    placement_subgoals = self._backward_subgoal_poses(
                                        arm_tag,
                                        x_side=x_side,
                                        clearance_z=clearance_z,
                                        lift_pose=lift_pose,
                                    )
                                    yield {
                                        "contact_point_id": cp_id,
                                        "grasp_waypoint": grasp_waypoint,
                                        "lift_pose": lift_pose,
                                        "placement_subgoals": placement_subgoals,
                                        "gap": gap,
                                        "y_offset": y_offset,
                                        "z_lift": z_lift,
                                        "orient": orient,
                                        "clearance_z": clearance_z,
                                    }

        def _plan_grasp_side(self, arm_tag, candidate, cache):
            """Waypoint -> pre_grasp -> grasp -> lift, cached per (cp_id, gap, z_lift,
            orient, y_offset) -- this doesn't depend on clearance_z, but
            _candidate_specs yields one candidate per clearance_z sharing the same
            waypoint/grasp, so without caching this (expensive: get_grasp_pose does a
            real 10-rotation CuRobo batch-plan) work would be redone 3x for nothing.
            Returns (ok, failed_stage, fail_reason, qpos, lift_pose, trajectories).
            trajectories is {"waypoint"/"pre_grasp"/"grasp": <raw plan_path result>},
            captured so real execution can REPLAY these EXACT verified trajectories
            instead of independently re-planning to the same poses: confirmed
            empirically that an independent re-plan to the same Cartesian pose can
            converge to a genuinely different (though equally valid) joint
            configuration -- CuRobo's trajopt has no uniqueness guarantee -- with
            qpos drift up to 1.47 rad between the two, which flips downstream grasp
            reachability in ~50% of cases with large drift. This already fixed the
            waypoint; pre_grasp/grasp trajectories are captured the same way here
            because grasp_actor_from_table's own independent re-plan to the SAME
            pose choose_best_pose already verified is exactly the same failure mode,
            one step later in the chain.
            pre_grasp/grasp trajectories are planned CHAINED (pre_grasp's landed
            qpos feeds grasp's start), matching how they're physically executed one
            after another -- unlike the get_grasp_pose feasibility checks below,
            which mirror choose_grasp_pose's own unchained pose determination."""
            key = (candidate["contact_point_id"], candidate["gap"], candidate["z_lift"],
                  candidate["orient"], candidate["y_offset"])
            if key in cache:
                return cache[key]

            start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            trajectories = {}
            plan_func = self._arm_plan_func(arm_tag)

            if candidate["grasp_waypoint"] is not None:
                waypoint_trajectory = plan_func(candidate["grasp_waypoint"], last_qpos=qpos)
                if waypoint_trajectory.get("status") != "Success":
                    result = (False, "waypoint", waypoint_trajectory.get("fail_reason", "unknown"), None, None, None)
                    cache[key] = result
                    return result
                trajectories["waypoint"] = waypoint_trajectory
                qpos = self._roll_qpos_forward(arm_tag, qpos, waypoint_trajectory)
            grasp_side_start_qpos = qpos  # pre_grasp/grasp poses are both determined from here (unchained)

            # Validate the SAME grasp-pose selection real execution uses
            # (get_grasp_pose -> choose_best_pose's rotation search + batch-plan
            # check, now that choose_best_pose's shortest-plan logic is fixed)
            # instead of the geometric approximation -- a candidate that "passes"
            # here can no longer diverge from what grasp_actor_from_table() actually
            # executes. choose_grasp_pose's own check_pose tests pre_grasp/grasp
            # independently from the SAME starting qpos (it doesn't chain through
            # pre_grasp's landed state) -- mirrored here for consistency.
            cp_id = candidate["contact_point_id"]
            pre_grasp_pose = self.get_grasp_pose(self.target_obj, arm_tag, contact_point_id=cp_id,
                                                 pre_dis=0.1, last_qpos=grasp_side_start_qpos)
            if pre_grasp_pose is None:
                result = (False, "pre_grasp", "no_reachable_rotation", None, None, None)
                cache[key] = result
                return result
            grasp_pose = self.get_grasp_pose(self.target_obj, arm_tag, contact_point_id=cp_id,
                                             pre_dis=0.0, last_qpos=grasp_side_start_qpos)
            if grasp_pose is None:
                result = (False, "grasp", "no_reachable_rotation", None, None, None)
                cache[key] = result
                return result

            # Now plan+capture the ACTUAL trajectories to replay, CHAINED (pre_grasp
            # then grasp from wherever pre_grasp lands), matching physical execution.
            pre_grasp_trajectory = plan_func(pre_grasp_pose, last_qpos=qpos)
            if pre_grasp_trajectory.get("status") != "Success":
                result = (False, "pre_grasp", pre_grasp_trajectory.get("fail_reason", "unknown"), None, None, None)
                cache[key] = result
                return result
            trajectories["pre_grasp"] = pre_grasp_trajectory
            qpos = self._roll_qpos_forward(arm_tag, qpos, pre_grasp_trajectory)

            grasp_trajectory = plan_func(grasp_pose, last_qpos=qpos)
            if grasp_trajectory.get("status") != "Success":
                result = (False, "grasp", grasp_trajectory.get("fail_reason", "unknown"), None, None, None)
                cache[key] = result
                return result
            trajectories["grasp"] = grasp_trajectory
            qpos = self._roll_qpos_forward(arm_tag, qpos, grasp_trajectory)

            lift_pose = list(grasp_pose)
            lift_pose[2] += GRASP_LIFT_HEIGHT
            ok, failed_stage, fail_reason, qpos = self._plan_pose_sequence(
                arm_tag, [lift_pose], qpos, stage_labels=["lift"])
            if not ok:
                result = (ok, failed_stage, fail_reason, None, None, None)
                cache[key] = result
                return result

            result = (True, None, None, qpos, lift_pose, trajectories)
            cache[key] = result
            return result

        def _plan_candidate(self, arm_tag, candidate, grasp_side_cache):
            ok, failed_stage, fail_reason, qpos, lift_pose, trajectories = self._plan_grasp_side(
                arm_tag, candidate, grasp_side_cache)
            if not ok:
                return ok, failed_stage, fail_reason
            # Carried to play_once so real execution can REPLAY these exact verified
            # trajectories for the waypoint/pre_grasp/grasp moves instead of
            # independently re-planning to the same poses (see _plan_grasp_side's
            # docstring for why that matters).
            candidate["grasp_waypoint_trajectory"] = trajectories.get("waypoint")
            candidate["pre_grasp_trajectory"] = trajectories.get("pre_grasp")
            candidate["grasp_trajectory"] = trajectories.get("grasp")

            # Placement subgoal orientation is derived from the grasp orientation --
            # rebuild from the REAL lift_pose above (candidate["placement_subgoals"]
            # was built in _candidate_specs from the geometric approximation) and
            # write the corrected version back onto the candidate so real execution
            # (play_once) uses it too, not just this validation pass.
            x_side = self._box_side_x(arm_tag, gap=candidate["gap"])
            placement_subgoals = self._backward_subgoal_poses(
                arm_tag, x_side=x_side, clearance_z=candidate["clearance_z"], lift_pose=lift_pose)
            candidate["placement_subgoals"] = placement_subgoals
            candidate["lift_pose"] = lift_pose

            # Real execution calls attach_object (holding the target) before planning
            # the placement subgoals; the check above doesn't, so it can wrongly
            # certify a placement pose that's actually blocked by the held object's
            # own collision volume. Confirmed empirically: "beside_box" flips from
            # IK-feasible to infeasible once attached with this same chained qpos.
            # Approximate the held object's pose as lift_pose (the gripper is
            # essentially at the object once it's holding it).
            planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
            object_dict = {
                "name": self.target_obj.get_name(),
                "pose": list(lift_pose),
                "file_path": f"{os.environ['BENCH_ROOT']}/assets/objects/{self.target_model}/collision/base{self.target_id}.glb",
                "scale": self.target_obj.scale,
            }
            placement_poses = [pose for _, pose in placement_subgoals]
            placement_labels = [name for name, _ in placement_subgoals]
            try:
                planner.attach_object(object_dict, qpos, arms_tag=str(arm_tag))
                ok2, failed_stage2, fail_reason2, _ = self._plan_pose_sequence(
                    arm_tag, placement_poses, qpos, stage_labels=placement_labels)
            finally:
                planner.detach_object()
            return ok2, failed_stage2, fail_reason2

        def _select_pick_place_candidate(self, arm_tag):
            cp_ids = self._rank_side_grasp_ids(self.target_obj, arm_tag)
            if not cp_ids:
                return {
                    "contact_point_id": None,
                    "grasp_waypoint": None,
                    "placement_subgoals": self._backward_subgoal_poses(
                        arm_tag,
                        x_side=self._box_side_x(arm_tag),
                        clearance_z=PLACE_CLEARANCE_ZS[1],
                        lift_pose=list(self.get_arm_pose(arm_tag)),
                    ),
                }

            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            fallback = None
            fallback_depth = -1
            fallback_stage = None
            tried = 0
            failure_breakdown = {}
            grasp_side_cache = {}
            for candidate in self._candidate_specs(arm_tag, cp_ids):
                tried += 1
                ok, failed_stage, fail_reason = self._plan_candidate(arm_tag, candidate, grasp_side_cache)
                if ok:
                    if log_move:
                        print(
                            "[candidate] selected "
                            f"cp_id={candidate['contact_point_id']} gap={candidate['gap']:.2f} "
                            f"z_lift={candidate['z_lift']:.2f} orient={candidate['orient']} "
                            f"y_offset={candidate['y_offset']:.2f} "
                            f"clearance_z={candidate['clearance_z']:.2f} (tried {tried})"
                        )
                    return candidate
                # Fallback = the candidate that progressed FURTHEST through the chain
                # before failing, not just the first one generated -- a later cp_id can
                # be far more reachable than the top-ranked one even though it's tried
                # later in the loop.
                depth = STAGE_ORDER.index(failed_stage) if failed_stage in STAGE_ORDER else -1
                if depth > fallback_depth:
                    fallback, fallback_depth, fallback_stage = candidate, depth, failed_stage
                reasons = failure_breakdown.setdefault(failed_stage, {})
                reasons[fail_reason] = reasons.get(fail_reason, 0) + 1

            if fallback is not None and log_move:
                print(
                    f"[candidate] no fully planned candidate found ({tried} tried); "
                    f"using deepest-progress fallback cp_id={fallback['contact_point_id']} "
                    f"(died at '{fallback_stage}', depth {fallback_depth}); "
                    f"failure breakdown (stage -> {{reason: count}}): {failure_breakdown}"
                )
            return fallback

    return OccluderTask


def run(args):
    out_dir = effective_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    offsets = [float(o) for o in args.offsets.split(",")]
    clutter_densities = [int(d) for d in args.clutter_densities.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    rollout = getattr(args, "rollout", False)
    ep_counter = 0   # unique episode id per rollout (-> video/episode{N}.mp4)

    env = make_occluder_task()()
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  offsets={offsets}  "
          f"clutter_densities={clutter_densities}  rollout={rollout}  camera={CAMERA}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        for si, seed in enumerate(seeds):
            # --- denominator: same scene, NO occluder ---
            env.spawn_occluder = False
            try:
                env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed, DR_CLEAN))
                target = _resolve_target(env)
                clean_pose = np.array(target.actor.get_pose().p)
                res_clean = env.measure_target_visibility(target, camera_name=CAMERA)
                full_px = res_clean["visible_pixel_count"]
                if args.save_images:
                    save_overlay(env, res_clean["mask"], img_dir / f"seed{seed:04d}_clean.png",
                                 f"seed {seed} CLEAN (denominator)  full_px={full_px}")
            except Exception as e:
                print(f"[seed {seed}] clean build failed ({type(e).__name__}: {e}); skipping seed")
                safe_close()
                continue
            safe_close()
            if full_px <= 0:
                print(f"[seed {seed}] full_target_px=0 on clean scene; skipping seed.")
                continue

            for off in offsets:
                # per-build coin flip: drop the occluder with prob no_occluder_prob
                # (keyed on seed+offset so the decision is shared across densities)
                show = bool(np.random.default_rng(int(seed) * 1000 + int(round(off * 100))).random()
                            >= args.no_occluder_prob)
                # Reject scenes where the occluder (at target_x, target_y - off) would land
                # too close to / on the destination pad, blocking the target's placement.
                if show:
                    occ_xy = np.array([clean_pose[0], clean_pose[1] - off])
                    pad_dist = float(np.linalg.norm(occ_xy - np.array(PAD_XY)))
                    if pad_dist < OCC_PAD_MIN_DIST:
                        print(f"[seed {seed}] occluder@off={off} is {pad_dist:.3f}m from pad "
                              f"(< {OCC_PAD_MIN_DIST:.3f}m); rejecting this build.")
                        continue
                env.spawn_occluder = show
                env.occluder_offset = off
                for cd in clutter_densities:
                    try:
                        env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed,
                                                   dr_measure(cd)))
                        target = _resolve_target(env)
                        pose_ok = bool(np.allclose(np.array(target.actor.get_pose().p), clean_pose, atol=1e-4))
                        res = env.measure_target_visibility(target, camera_name=CAMERA, denominator=full_px)
                        if args.save_images:
                            tag = "occ" if show else "noocc"
                            save_overlay(
                                env, res["mask"],
                                img_dir / f"seed{seed:04d}_off{off:.2f}_cd{cd:02d}_{tag}.png",
                                f"seed {seed} off={off:.2f} {tag} clut={cd}  "
                                f"vis_px={res['visible_pixel_count']} full={full_px} "
                                f"frac={res['visible_fraction']:.3f} {res['bucket']} pose_match={pose_ok}",
                            )
                    except Exception as e:
                        print(f"[seed {seed} off{off:.2f} cd{cd}] build failed ({type(e).__name__}: {e}); skipping")
                        safe_close()
                        continue
                    safe_close()

                    rec = {"seed": int(seed), "offset": float(off), "full_px": int(full_px),
                           "visible_px": int(res["visible_pixel_count"]),
                           "visible_fraction": float(res["visible_fraction"]),
                           "bucket": res["bucket"], "in_fov": bool(res["in_fov"]),
                           "pose_match": pose_ok, "occluder_shown": show,
                           "clutter_density": int(cd)}
                    # --- expert curobo rollout on the same scene (video + success) ---
                    # env.spawn_occluder / occluder_offset persist, so the rollout
                    # build reproduces the same occluder placement as the measurement.
                    if rollout:
                        rollout_result = run_rollout(env, "put_mouse_on_pad", args.base_config, seed,
                                                     dr_measure(cd), out_dir, ep_counter)
                        success = bool(rollout_result["success"])
                        rec["rollout_success"] = success
                        rec["rollout_ep"] = ep_counter
                        artifact_info = rollout_result.get("artifact_info") or {}
                        rec["rollout_bucket"] = artifact_info.get("bucket", "success" if success else "fail")
                        rec["rollout_video"] = artifact_info.get(
                            "video_relpath", f"{rec['rollout_bucket']}/video/episode{ep_counter}.mp4"
                        )
                        rec["rollout_data"] = artifact_info.get(
                            "hdf5_relpath", f"{rec['rollout_bucket']}/data/episode{ep_counter}.hdf5"
                        )
                        print(f"    seed {seed} off={off:.2f} cd={cd} {res['bucket']}: "
                              f"rollout {'SUCCESS' if success else 'FAIL'} -> "
                              f"{rec['rollout_bucket']}/episode{ep_counter}")
                        ep_counter += 1
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
            print(f"[{si+1}/{len(seeds)}] seed {seed}: full_px={full_px} done")

    safe_close()
    print(f"\nsweep complete -> {jsonl_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--offsets", default="0.2", help="occluder offset(s) in m in front of the target")
    ap.add_argument("--clutter-densities", default="0",
                    help="table clutter density/densities to sweep in the measurement scene, "
                         "comma-separated (0=off). e.g. 0,8,15")
    ap.add_argument("--group-by", default="offset", choices=["offset", "clutter_density"],
                    help="analysis grouping variable for the bucket/histogram figures")
    ap.add_argument("--no-occluder-prob", type=float, default=0.2,
                    help="probability the milk-box occluder is NOT spawned for a build (default 0.2)")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="../scripts/validation/results/phase2_occluder")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--rollout", action="store_true",
                    help="run an expert curobo rollout per scene (writes to <out-dir>_rollout, "
                         "saves videos, and adds success-only distribution + P(success) per bucket)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        run(args)
    group_label = {"offset": "occluder offset (m)",
                   "clutter_density": "table clutter density"}[args.group_by]
    out_dir = effective_out_dir(args)
    analyze_kwargs = dict(group_key=args.group_by, group_label=group_label,
                          suptitle="Visibility with one milk-box occluder (countertop)",
                          bar_title=f"Bucket proportions vs {group_label}")
    if args.rollout:
        analyze_rollout(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)
    else:
        analyze(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)


if __name__ == "__main__":
    main()
