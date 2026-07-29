"""Methods extracted mechanically from analyze_occluder_visibility.py."""

import contextlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import transforms3d as t3d
from curobo.types.state import JointState

from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from envs.utils import ArmTag
from lib.ik_grid import _build_ik_solver, grasp_orientation
from lib.planning_tuning import *  # noqa: F403
from lib.run_io import CLEARANCE_RESULTS_DIR
from lib.scene_constants import *  # noqa: F403
import seed_from_clearance as sfc


class PlacementMixin:
    def _backward_subgoal_poses(self, arm_tag, x_side, clearance_z, lift_pose):
        """Ordered BACKWARD (placement) subgoals for a chosen candidate path.

            Empty in PLACEMENT_MODE=direct. Killed HERE, at the single source every caller
            goes through (_candidate_specs, _plan_candidate, _select_attached_placement_plan
            and its no-geometry fallback, the no-contact-point fallback), rather than at each
            call site -- the same reason the around-box waypoint is suppressed inside
            _candidate_specs: a chain that merely goes unplanned still gets live-re-planned
            downstream by whatever branches on its poses."""
        if self._placement_mode() == "direct":
            return []
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


    def _post_grasp_escape_poses(self, arm_tag, attempts=LOCAL_WAYPOINT_ATTEMPTS):
        """Local escape waypoint candidates from the current held-object pose.

            Candidate zero is `None`, preserving the direct route. The remaining
            candidates are the bounded local waypoint family for this specific
            attached-object transition out of the grasped state.
            """
        current = list(self.get_arm_pose(arm_tag))
        return [None] + self._local_waypoint_candidates(
            arm_tag, current, current, attempts=attempts)


    def _placement_execution_steps(self, arm_tag, placement_subgoals, escape_pose=None):
        """Build the exact post-attach placement chain we want to execute."""
        current_pose = list(self.get_arm_pose(arm_tag))
        steps = []
        if escape_pose is not None:
            steps.append(("placement:post_grasp_escape", escape_pose))
            current_pose = escape_pose
        if placement_subgoals:
            beside_box_pose = placement_subgoals[0][1]
            mid_to_beside_box = self._verified_intermediate(arm_tag, current_pose, beside_box_pose)
            steps.append(("placement:pre_beside_box", mid_to_beside_box))

        prev_pose = None
        for name, pose in placement_subgoals:
            if name == "lift_above_box" and prev_pose is not None:
                mid_to_lift_above_box = self._verified_intermediate(arm_tag, prev_pose, pose)
                steps.append(("placement:pre_lift_above_box", mid_to_lift_above_box))
            steps.append((f"placement:{name}", pose))
            prev_pose = pose
        return steps


    def _select_attached_placement_plan(self, arm_tag, tgt_y, lift_z, quat,
                                        gaps=SIDE_WAYPOINT_GAPS, clearance_options=PLACE_CLEARANCE_ZS,
                                        fallback_gaps=SIDE_WAYPOINT_GAPS_FALLBACK,
                                        clearance_fallback=PLACE_CLEARANCE_ZS_FALLBACK):
        """Pick placement geometry by FULL attached-world motion-plan success.

            Endpoint IK is a useful prefilter, but the remaining failures here have
            increasingly been about the chained motion itself. This helper keeps the
            same geometry search space, then scores candidates by whether the exact
            post-attach execution chain plans successfully from the current live qpos.
            The winning trajectories are cached for immediate replay.

            Every stage this searches over (post_grasp_escape, beside_box,
            lift_above_box, over_box_to_pad_y, center_over_pad, and the blended
            pre_* transit waypoints between them) is a pure carry-the-held-object
            move -- check_success() never looks at orientation en route, only the
            final x/y placement. So both the prefilters and the real trajopt calls
            below use relax_orientation=True (position-only goals): a
            check_ik_batch(relax_orientation=True) sweep confirmed this opens up
            real, currently-dead combos at lift_above_box that strict full-pose IK
            rejects outright."""
        log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
        quat_options = (quat, GRASP_DIRECTION_DIC["top_down"])
        start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
        check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
        fallback = None
        fallback_depth = -1
        failure_breakdown = {}
        escape_poses = self._post_grasp_escape_poses(arm_tag, attempts=LOCAL_WAYPOINT_ATTEMPTS)

        # gaps first, fallback_gaps ONLY if NOTHING in gaps ever verified reachable
        # (the seed-9 case: box far enough from the shoulder that ALL of SIDE_
        # WAYPOINT_GAPS's beside_box targets are IK-infeasible, so the full trajopt
        # chain never even had a chance regardless of clearance/escape/quat).
        for gap_values in (gaps, fallback_gaps):
            any_reachable_x = False
            for quat_choice in quat_options:
                for gap in gap_values:
                    x = self._box_side_x(arm_tag, gap=gap)
                    # Cheap prefilter (no trajopt): beside_box's height (lift_z) and
                    # y (tgt_y) don't depend on clearance_z/escape_idx, so checking
                    # reachability once per (gap, quat) here skips the ENTIRE cz x
                    # escape_idx loop (up to 30 full trajopt chains) for a gap whose
                    # target could never succeed regardless of how it's reached.
                    if not bool(check_ik([[x, tgt_y, lift_z, *quat_choice]], relax_orientation=True)[0]):
                        failure_breakdown[("placement:beside_box", "IK_prefilter_unreachable")] = (
                            failure_breakdown.get(("placement:beside_box", "IK_prefilter_unreachable"), 0) + 1)
                        if log_move:
                            print(f"[placement-plan] gap={gap:.2f} x_side={x:.3f} quat={quat_choice} "
                                  f"skipped: beside_box IK-unreachable (prefilter)")
                        continue
                    any_reachable_x = True
                    for cz in list(clearance_options) + list(clearance_fallback):
                        # Same idea as the beside_box prefilter above, one level
                        # deeper: lift_above_box shares this (x, tgt_y) and only
                        # varies in z (clearance) -- skip a clearance value here
                        # (cheaply) rather than discovering it's dead only after
                        # the full escape_idx loop of trajopt chains.
                        if not bool(check_ik([[x, tgt_y, cz, *quat_choice]], relax_orientation=True)[0]):
                            failure_breakdown[("placement:lift_above_box", "IK_prefilter_unreachable")] = (
                                failure_breakdown.get(("placement:lift_above_box", "IK_prefilter_unreachable"), 0) + 1)
                            if log_move:
                                print(f"[placement-plan] gap={gap:.2f} x_side={x:.3f} clearance_z={cz:.2f} "
                                      f"quat={quat_choice} skipped: lift_above_box IK-unreachable (prefilter)")
                            continue
                        placement_subgoals = self._backward_subgoal_poses(
                            arm_tag,
                            x_side=x,
                            clearance_z=cz,
                            lift_pose=[0, 0, lift_z, *quat_choice],
                        )
                        for escape_idx, escape_pose in enumerate(escape_poses):
                            stage_poses = self._placement_execution_steps(
                                arm_tag, placement_subgoals, escape_pose=escape_pose)
                            ok, failed_stage, fail_reason, trajectories = self._plan_pose_trajectory_sequence(
                                arm_tag, stage_poses, start_qpos, local_attempts=0,
                                retries=PLACEMENT_SEARCH_RETRIES, relax_orientation=True)
                            if ok:
                                return {
                                    "x_side": x,
                                    "clearance_z": cz,
                                    "quat": quat_choice,
                                    "escape_idx": escape_idx,
                                    "placement_subgoals": placement_subgoals,
                                    "trajectories": trajectories,
                                    "verified": True,
                                }
                            depth = STAGE_ORDER.index(failed_stage.split("placement:")[-1]) if (
                                failed_stage and failed_stage.startswith("placement:")
                                and failed_stage.split("placement:")[-1] in STAGE_ORDER
                            ) else -1
                            if depth > fallback_depth:
                                fallback = {
                                    "x_side": x,
                                    "clearance_z": cz,
                                    "quat": quat_choice,
                                    "escape_idx": escape_idx,
                                    "placement_subgoals": placement_subgoals,
                                    "trajectories": trajectories,
                                    "verified": False,
                                    "failed_stage": failed_stage,
                                    "fail_reason": fail_reason,
                                }
                                fallback_depth = depth
                            failure_breakdown[(failed_stage, fail_reason)] = failure_breakdown.get((failed_stage, fail_reason), 0) + 1
                            if log_move:
                                print(f"[placement-plan] x_side={x:.3f} clearance_z={cz:.2f} "
                                      f"escape={escape_idx} quat={quat_choice} failed at "
                                      f"{failed_stage} ({fail_reason})")
            if any_reachable_x or gap_values is not gaps:
                break
            if log_move:
                print(f"[placement-plan] none of gaps={gap_values} verified beside_box-reachable "
                      f"for any quat; trying fallback gaps={fallback_gaps}")
        if log_move and fallback is not None:
            print(f"[placement-plan] no fully planned geometry; using deepest fallback "
                  f"(depth={fallback_depth}) with breakdown={failure_breakdown}")
        if fallback is not None:
            return fallback
        placement_subgoals = self._backward_subgoal_poses(
            arm_tag,
            x_side=self._box_side_x(arm_tag, gap=gaps[0]),
            clearance_z=clearance_options[0],
            lift_pose=[0, 0, lift_z, *quat],
        )
        return {
            "x_side": self._box_side_x(arm_tag, gap=gaps[0]),
            "clearance_z": clearance_options[0],
            "quat": quat,
            "placement_subgoals": placement_subgoals,
            "trajectories": [],
            "verified": False,
            "failed_stage": None,
            "fail_reason": "no_geometry_attempted",
        }


    def _descent_tstep_fraction_for_attempt(self, attempt_idx, total_attempts,
                                           fractions=DESCENT_APPROACH_TSTEP_FRACTIONS):
        """Progressive relaxation schedule across a descent slice's retry
            budget: front-load attempts on the tightest (least detour room)
            fraction, then relax toward CuRobo's own default across the
            remaining attempts -- see DESCENT_APPROACH_TSTEP_FRACTIONS for why
            a single fixed value (either tight or loose) isn't enough on its
            own."""
        tier = min(len(fractions) - 1, attempt_idx * len(fractions) // max(1, total_attempts))
        return fractions[tier]


    def _plan_pose_with_descent_slices(self, arm_tag, target_pose, qpos, stage_label, start_pose,
                                      retries=1, constraint_pose=None, approach_axis=None,
                                      slice_size=DESCENT_SLICE_SIZE, near_contact=False):
        """Deterministic incremental descent, used for place_actor's moves
            instead of the generic scalar-error-informed shrink retry
            (_plan_pose_with_shrinking_waypoint). place_actor's moves are short,
            mostly-straight-line final approaches with KNOWN structure (fixed
            target orientation, held via approach_axis) -- exploiting that
            structure with fixed-size slices along the straight line to the
            target is more appropriate than an adaptive shrink that doesn't know
            it. Plans AND REPLAYS each slice immediately (not a virtual dry-run
            chain), checking _object_retained after every segment -- catches a
            mid-descent drop at the EXACT slice it happens (confirmed
            empirically: seed 27 dropped the object mid-transit while every
            stage still reported plan_success=True) instead of only at the end,
            and naturally preserves whatever slices already succeeded if a
            later one fails, without needing a separate replay-before-return
            step (each slice is already committed by the time the next runs).

            Returns (ok, fail_reason, qpos, placed). placed=True means the
            object has already reached a pose that would pass check_success --
            the caller (_execute_actions_via_plan_and_replay) must skip any
            remaining 'move' actions and go straight to opening the gripper,
            since continuing to descend past an already-good placement could
            only make it worse."""
        plan_func = self._arm_plan_func(arm_tag)
        qpos0 = np.array(qpos, dtype=np.float64, copy=True)
        start = np.asarray(start_pose, dtype=float)
        target = np.asarray(target_pose, dtype=float)
        total_dist = float(np.linalg.norm(target[:3] - start[:3]))
        num_slices = max(1, int(np.ceil(total_dist / slice_size)))
        segment_length = total_dist / num_slices
        # create_grasp_approach_metric's offset_position must stay smaller
        # than the segment it's applied to -- CuRobo's own default (0.05) is
        # LARGER than a 4cm slice, asking it to set up a straight-line
        # pre-approach point farther away than the entire move itself.
        # Confirmed to matter: 6 episodes still failed FINETUNE_TRAJOPT_FAIL
        # with the unscaled 5cm offset on every slice.
        approach_offset = min(0.05, segment_length * 0.8) if approach_axis is not None else 0.05
        log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
        last_reason = "unknown"
        for i in range(1, num_slices + 1):
            # Interpolate BOTH position and orientation (_blend_pose does
            # position lerp + quaternion nlerp) -- copying the target's
            # quaternion onto every slice made the FIRST slice perform the
            # entire orientation change within just one small position step.
            waypoint = np.asarray(self._blend_pose(start, target, i / num_slices), dtype=float)
            total_attempts = max(1, int(retries))
            # Collect every attempt that plans successfully AND passes the
            # path-safety filter, across the whole progressive-relaxation
            # schedule, then replay the SHORTEST accepted one -- not just
            # the first success. Trying every attempt regardless of an
            # early pass (rather than stopping at the first accepted
            # candidate) is what lets a tight-fraction attempt lose to a
            # looser one that happens to find a genuinely shorter path,
            # and is bounded by retries (already small, e.g. 10).
            accepted = []  # list of (path_length, joint_path_length, max_joint_range, result, attempt_idx, attempt_record)
            for attempt in range(total_attempts):
                fraction = self._descent_tstep_fraction_for_attempt(attempt, total_attempts)
                candidate = plan_func(list(waypoint), last_qpos=qpos0,
                                     constraint_pose=constraint_pose, approach_axis=approach_axis,
                                     approach_offset=approach_offset,
                                     tstep_fraction=fraction, near_contact=near_contact)
                candidate["tstep_fraction"] = fraction
                attempt_record = {
                    "stage": stage_label, "slice": i, "num_slices": num_slices,
                    "attempt": attempt, "tstep_fraction": fraction,
                    "near_contact": near_contact,
                    "status": candidate.get("status"),
                    "position_error": candidate.get("position_error"),
                    "rotation_error": candidate.get("rotation_error"),
                }
                if candidate.get("status") != "Success":
                    last_reason = candidate.get("fail_reason", "unknown")
                    attempt_record["fail_reason"] = last_reason
                    self.rollout_descent_slices.append(attempt_record)
                    continue
                metrics = self._trajectory_path_metrics(arm_tag, candidate)
                attempt_record["path_metrics"] = metrics
                if metrics is None:
                    # Nothing to measure (degenerate <2-waypoint plan) --
                    # trust it, there's no path to have gone wrong.
                    accepted.append((0.0, 0.0, 0.0, candidate, attempt, attempt_record))
                    attempt_record["accepted"] = True
                elif (metrics["max_perp_deviation"] <= DESCENT_MAX_PATH_DEVIATION
                        and metrics["path_length_ratio"] <= DESCENT_MAX_PATH_LENGTH_RATIO
                        and metrics["joint_travel_ratio"] <= DESCENT_MAX_JOINT_TRAVEL_RATIO
                        and metrics["max_joint_range"] <= DESCENT_MAX_JOINT_RANGE
                        and metrics["joint_direct_dist"] <= DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT
                        and metrics["joint_path_length"] <= DESCENT_MAX_JOINT_PATH_LENGTH):
                    accepted.append((metrics["path_length"], metrics["joint_path_length"],
                                    metrics["max_joint_range"], candidate, attempt, attempt_record))
                    attempt_record["accepted"] = True
                else:
                    last_reason = (
                        f"path_filter_rejected(dev={metrics['max_perp_deviation']:.3f},"
                        f"len_ratio={metrics['path_length_ratio']:.2f},"
                        f"joint_ratio={metrics['joint_travel_ratio']:.2f},"
                        f"max_joint_range={metrics['max_joint_range']:.3f},"
                        f"joint_dist={metrics['joint_direct_dist']:.3f},"
                        f"joint_path={metrics['joint_path_length']:.3f})")
                    attempt_record["accepted"] = False
                    attempt_record["reject_reason"] = last_reason
                    if log_move:
                        print(f"[descent-slice] {stage_label} slice {i}/{num_slices} attempt "
                              f"{attempt} (tstep_fraction={fraction}): rejected -- {last_reason}")
                self.rollout_descent_slices.append(attempt_record)
            if not accepted:
                if log_move:
                    print(f"[descent-slice] {stage_label} slice {i}/{num_slices} failed "
                          f"({total_attempts} attempts, none accepted; last={last_reason})")
                return False, last_reason, qpos0, False
            # Lexicographic: shortest Cartesian path first, then shortest
            # joint-space path, then smallest single-joint excursion --
            # two candidates can have near-identical Cartesian paths but
            # very different joint motion, so Cartesian length alone
            # isn't enough to prefer the more sensible one.
            accepted.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
            best_path_length, best_joint_path_length, best_max_joint_range, result, best_attempt, best_record = accepted[0]
            best_record["selected"] = True
            if log_move:
                print(f"[descent-slice] {stage_label} slice {i}/{num_slices} succeeded "
                      f"(attempt {best_attempt}, tstep_fraction={result['tstep_fraction']}, "
                      f"path_length={best_path_length:.3f}m, "
                      f"joint_path_length={best_joint_path_length:.3f}rad, "
                      f"{len(accepted)}/{total_attempts} attempts accepted)")
            qpos0 = self._roll_qpos_forward(arm_tag, qpos0, result)
            self._replay_planned_move(arm_tag, result)
            if not self.plan_success:
                return False, self._last_fail_reason, qpos0, False
            # Check arrival BEFORE retention, not as a fallback after a
            # failed retention check: the object settling onto the target
            # surface early (before the gripper's own nominal endpoint)
            # changes the gripper-relative transform exactly like a drop
            # would, but it's the intended outcome -- stop descending
            # further (which could otherwise ram the held object into the
            # surface it already reached) and report this move done, via
            # placed=True (the caller must skip any remaining moves).
            if self._object_near_placement_target():
                if log_move:
                    print(f"[descent-slice] {stage_label} slice {i}/{num_slices}: object "
                          f"already at a placement that would pass check_success -- done")
                return True, None, qpos0, True
            retained, pos_drift, rot_drift = self._object_retained(
                arm_tag, context=f"{stage_label}:slice{i}", return_drift=True)
            if not retained:
                xy_error = self._placement_xy_error()
                supported_contact = (
                    pos_drift <= DESCENT_CONTACT_DRIFT_TOLERANCE
                    and self._object_near_support_height())
                if (supported_contact
                        and np.all(np.abs(xy_error) < CONTACT_RELEASE_XY_TOLERANCE)):
                    self.rollout_descent_slices.append({
                        "stage": f"{stage_label}:contact-release",
                        "dx": float(xy_error[0]), "dy": float(xy_error[1]),
                        "correction_needed": False,
                    })
                    return True, None, qpos0, True
                # Ambiguous-contact requires ALL of: bounded position
                # drift, bounded ROTATION drift (a rotation-only loss --
                # object tipped/rotated loose without much translating --
                # would otherwise sail through on position alone, quietly
                # defeating the retention check's own rotation term), and
                # the object plausibly near the support surface it would
                # be contacting (a genuine drop shows the same 4-15cm
                # position drift while still high above the surface,
                # since it's falling under gravity rather than following
                # the descent's planned trajectory).
                if (pos_drift <= DESCENT_CONTACT_DRIFT_TOLERANCE
                        and rot_drift <= OBJECT_RETENTION_ROTATION_TOLERANCE
                        and self._object_near_support_height()):
                    # Ambiguous: not yet at a passing placement (the check
                    # above already ruled that out) and not within strict
                    # retention tolerance either, but the drift magnitude is
                    # well below anything a genuine drop has shown (30cm-
                    # 1.4m) and consistent with early, imprecise contact.
                    # Don't fail -- let the remaining slices keep trying to
                    # converge on the real target instead of giving up on
                    # a placement that's already known to be too far off.
                    if log_move:
                        print(f"[descent-slice] {stage_label} slice {i}/{num_slices}: drift "
                              f"{pos_drift:.3f}m looks like early contact, not a loss -- continuing")
                else:
                    self.plan_success = False
                    self._last_fail_reason = f"object_lost:{stage_label}:slice{i}"
                    return False, self._last_fail_reason, qpos0, False
            elif log_move:
                print(f"[descent-slice] {stage_label} slice {i}/{num_slices} succeeded")
        return True, None, qpos0, False


    def _local_landing_search_and_place(self, arm_tag):
        """Replaces place_actor's forced descent to one exact final pose
            (move1/move2 via _execute_actions_via_plan_and_replay), which kept
            failing MotionGenStatus.FINETUNE_TRAJOPT_FAIL right at the surface
            -- the attached object's collision margin against the table is
            tightest exactly there, and CuRobo's trajopt struggles to converge
            on ANY smooth path into it, let alone the one exact pose asked for.

            Instead searches a small grid of nearby landing poses -- XY
            offsets within check_success's own tolerance, release heights
            from low to high -- and requires only ONE of them to actually
            plan and pass the same trajectory-safety filters used by
            _plan_pose_with_descent_slices, then replays ONLY that one and
            opens the gripper, letting physics settle the final contact
            instead of commanding CuRobo all the way down to it. Explicitly
            gated on the object being retained and within
            LANDING_SEARCH_TRIGGER_DISTANCE (XY) / LANDING_SEARCH_MAX_Z_
            DISTANCE (vertical) of the target -- this function's
            candidate moves are short by design, so calling it from farther
            away would apply the same short-path filters to a legitimately
            longer approach and reject it.

            Heights are grouped into two tiers ([1cm,3cm] then [5cm,8cm]):
            ALL accepted candidates across an entire tier are compared
            before picking a winner, not just the first height with any
            success -- otherwise a marginal 1cm solve could beat a cleanly
            planned 3cm one that was never even tried, since candidates
            aren't compared across different heights independently.

            Candidate object poses are converted to gripper poses via
            get_place_pose, NOT a fixed offset above the pad -- get_place_pose
            derives the required gripper pose from the ACTUAL current grasp
            transform (actor pose vs. end-effector pose), so it's correct
            regardless of which grasp this particular episode happened to
            use, unlike a fixed "gripper N cm above the pad" assumption which
            would put the object at the wrong height for a different grasp.

            Sets plan_success=False with a descriptive fail_reason if no
            candidate at any height passes; the caller's checkpoint() records
            it as an ordinary failure, same as any other stage."""
        plan_func = self._arm_plan_func(arm_tag)
        log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
        check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
        target = np.asarray(self.des_obj_pose, dtype=float)

        obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
        xy_distance = float(np.hypot(obj_pos[0] - target[0], obj_pos[1] - target[1]))
        z_distance = float(abs(obj_pos[2] - target[2]))
        retained = self._object_retained(arm_tag, context="landing-search:precondition")
        if not retained or xy_distance > LANDING_SEARCH_TRIGGER_DISTANCE or z_distance > LANDING_SEARCH_MAX_Z_DISTANCE:
            self.plan_success = False
            self._last_fail_reason = (f"landing_search_preconditions_not_met(retained={retained},"
                                      f"xy_distance={xy_distance:.3f}m,z_distance={z_distance:.3f}m)")
            if log_move:
                print(f"[landing-search] preconditions not met -- retained={retained} "
                      f"xy_distance={xy_distance:.3f}m (max {LANDING_SEARCH_TRIGGER_DISTANCE}m) "
                      f"z_distance={z_distance:.3f}m (max {LANDING_SEARCH_MAX_Z_DISTANCE}m)")
            return

        # Skip direct 1--5 cm grids: the sweep produced 509 solver failures.
        # Stage at 8 cm, then descend sequentially with the near-contact profile.
        height_tiers = [[max(LANDING_RELEASE_HEIGHTS)]]
        for tier in height_tiers:
            tier_accepted = []  # (path_length, joint_path_length, max_joint_range, result, height, record)
            for height in tier:
                candidates = []  # (dx, dy, obj_pose)
                for dx in LANDING_XY_OFFSETS:
                    for dy in LANDING_XY_OFFSETS:
                        obj_pose = target.copy()
                        obj_pose[0] += dx
                        obj_pose[1] += dy
                        obj_pose[2] = target[2] + height
                        candidates.append((dx, dy, obj_pose))
                # Sort by radius (closest to nominal target first), then
                # by angle -- groups by distance WITHOUT letting ties at
                # the same radius systematically favor one sign over
                # another (plain grid iteration order puts every
                # negative-dx candidate before any positive-dx one at
                # the same radius, since LANDING_XY_OFFSETS lists
                # negatives first and Python's sort is stable).
                candidates.sort(key=lambda c: (np.hypot(c[0], c[1]), np.arctan2(c[1], c[0])))

                gripper_poses = [
                    self.get_place_pose(self.target_obj, arm_tag, obj_pose, constrain="free", pre_dis=0.0)
                    for _, _, obj_pose in candidates
                ]
                ik_ok = list(check_ik(gripper_poses))
                current_ee_pose = np.asarray(self.get_arm_pose(arm_tag), dtype=float)
                qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
                qpos0 = np.array(qpos, dtype=np.float64, copy=True)

                planned = 0
                height_accepted = 0
                for (dx, dy, _), gripper_pose, feasible in zip(candidates, gripper_poses, ik_ok):
                    record = {"stage": "landing-search", "height": height, "dx": dx, "dy": dy,
                              "ik_feasible": bool(feasible)}
                    if not feasible:
                        self.rollout_descent_slices.append(record)
                        continue
                    if planned >= LANDING_MAX_CANDIDATES_PER_HEIGHT:
                        continue  # never attempted -- not a rejection, don't record
                    planned += 1
                    segment_length = float(np.linalg.norm(np.asarray(gripper_pose[:3]) - current_ee_pose[:3]))
                    approach_offset = min(0.05, segment_length * 0.8) if segment_length > 1e-6 else 0.02
                    if height == max(LANDING_RELEASE_HEIGHTS):
                        # This collision-clear target is a staging transit,
                        # not a descent. The approach metric caused the
                        # repeatable 2--2.5x loops seen in landing_v2.
                        candidate_result = plan_func(list(gripper_pose), last_qpos=qpos0)
                    else:
                        candidate_result = plan_func(
                            list(gripper_pose), last_qpos=qpos0, approach_axis=2,
                            approach_offset=approach_offset, tstep_fraction=0.4)
                    record["status"] = candidate_result.get("status")
                    if candidate_result.get("status") != "Success":
                        record["fail_reason"] = candidate_result.get("fail_reason", "unknown")
                        record["accepted"] = False
                        self.rollout_descent_slices.append(record)
                        continue
                    metrics = self._trajectory_path_metrics(arm_tag, candidate_result)
                    record["path_metrics"] = metrics
                    # DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT/PATH_LENGTH
                    # were calibrated for descent slices (~DESCENT_SLICE_
                    # SIZE=4cm each) -- a landing-search candidate can be
                    # up to ~10cm away, which naturally needs more joint
                    # travel even for a perfectly clean plan. Reusing the
                    # unscaled absolute caps rejected almost everything
                    # (confirmed empirically: 1/288 candidates accepted
                    # across a 6-seed sanity run). Scale by how much
                    # longer this candidate's segment is than a nominal
                    # slice, same way approach_offset already scales.
                    joint_scale = max(1.0, segment_length / DESCENT_SLICE_SIZE)
                    max_joint_endpoint_displacement = DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT * joint_scale
                    max_joint_path_length = DESCENT_MAX_JOINT_PATH_LENGTH * joint_scale
                    record["joint_scale"] = joint_scale
                    if metrics is None:
                        tier_accepted.append((0.0, 0.0, 0.0, candidate_result, height, dx, dy, record))
                        record["accepted"] = True
                        height_accepted += 1
                    elif (metrics["max_perp_deviation"] <= DESCENT_MAX_PATH_DEVIATION
                            and metrics["path_length_ratio"] <= DESCENT_MAX_PATH_LENGTH_RATIO
                            and metrics["joint_travel_ratio"] <= DESCENT_MAX_JOINT_TRAVEL_RATIO
                            and metrics["max_joint_range"] <= DESCENT_MAX_JOINT_RANGE
                            and metrics["joint_direct_dist"] <= max_joint_endpoint_displacement
                            and metrics["joint_path_length"] <= max_joint_path_length):
                        tier_accepted.append((metrics["path_length"], metrics["joint_path_length"],
                                              metrics["max_joint_range"], candidate_result, height, dx, dy, record))
                        record["accepted"] = True
                        height_accepted += 1
                    else:
                        record["accepted"] = False
                        record["reject_reason"] = (
                            f"path_filter_rejected(dev={metrics['max_perp_deviation']:.3f},"
                            f"len_ratio={metrics['path_length_ratio']:.2f},"
                            f"joint_ratio={metrics['joint_travel_ratio']:.2f},"
                            f"max_joint_range={metrics['max_joint_range']:.3f},"
                            f"joint_dist={metrics['joint_direct_dist']:.3f}(max={max_joint_endpoint_displacement:.3f}),"
                            f"joint_path={metrics['joint_path_length']:.3f}(max={max_joint_path_length:.3f}))")
                    self.rollout_descent_slices.append(record)
                    if height_accepted >= LANDING_MIN_ACCEPTED_TO_STOP:
                        break
                if log_move:
                    print(f"[landing-search] height={height:.3f}m: {height_accepted} accepted "
                          f"({sum(ik_ok)}/{len(candidates)} IK-feasible, {planned} planned)")

            if not tier_accepted:
                continue

            tier_accepted.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
            best_path_length, best_joint_path_length, _, result, best_height, best_dx, best_dy, best_record = tier_accepted[0]
            best_record["selected"] = True
            if log_move:
                print(f"[landing-search] tier {tier} succeeded: {len(tier_accepted)} candidates "
                      f"accepted across the tier, chose height={best_height:.3f}m "
                      f"path_length={best_path_length:.3f}m joint_path_length={best_joint_path_length:.3f}rad")
            self._replay_planned_move(arm_tag, result)
            if not self.plan_success:
                return
            if not self._object_retained(arm_tag, context="landing-search"):
                self.plan_success = False
                self._last_fail_reason = "object_lost:landing-search"
                return

            release_height = best_height
            descent_failure = None
            if best_height == max(LANDING_RELEASE_HEIGHTS):
                # The collision-clear solve is a staging pose. From this
                # verified state, make only short constrained downward moves.
                for lower_height in sorted(
                        (h for h in LANDING_RELEASE_HEIGHTS if h < best_height), reverse=True):
                    lower_obj_pose = target.copy()
                    lower_obj_pose[:3] += np.array([best_dx, best_dy, lower_height])
                    lower_gripper_pose = self.get_place_pose(
                        self.target_obj, arm_tag, lower_obj_pose,
                        constrain="free", pre_dis=0.0)
                    live_qpos = (self.robot.left_entity.get_qpos() if str(arm_tag) == "left"
                                 else self.robot.right_entity.get_qpos())
                    ok, reason, _, placed = self._plan_pose_with_descent_slices(
                        arm_tag, lower_gripper_pose,
                        np.asarray(live_qpos, dtype=np.float64),
                        f"landing-descent:{lower_height:.3f}",
                        start_pose=self.get_arm_pose(arm_tag),
                        retries=PLACEMENT_SEARCH_RETRIES, approach_axis=2,
                        near_contact=True)
                    if not ok:
                        descent_failure = reason
                        if not self.plan_success:
                            return
                        break
                    release_height = lower_height
                    if placed:
                        break

            self.rollout_descent_slices.append({
                "stage": "landing-release", "height": release_height,
                "dx": best_dx, "dy": best_dy,
                "descent_failure": descent_failure,
            })
            self.move(self.open_gripper(arm_tag))
            return

        self.plan_success = False
        self._last_fail_reason = "no_valid_landing_candidate"
        if log_move:
            print(f"[landing-search] failed: no candidate at any height "
                  f"{LANDING_RELEASE_HEIGHTS} planned and passed the filter")
