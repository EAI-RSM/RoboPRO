"""Methods extracted mechanically from analyze_occluder_visibility.py."""

import os
from copy import deepcopy

import numpy as np

from lib.planning_tuning import *  # noqa: F403
from lib.scene_constants import *  # noqa: F403


class PlanningMixin:
    def _plan_pose_trajectory_sequence(self, arm_tag, stage_poses, start_qpos,
                                       local_attempts=LOCAL_WAYPOINT_ATTEMPTS, retries=1,
                                       relax_orientation=False):
        """Plan a labeled pose chain and keep the exact successful trajectories.

            relax_orientation is withheld for PLACEMENT_STRICT_ORIENTATION_STAGES
            regardless of the caller's request: relaxing orientation on the LAST
            placement subgoal (center_over_pad, the handoff into place_actor's own
            strict, orientation-sensitive final descent) let CuRobo converge to an
            arbitrary held-object orientation there -- confirmed empirically to
            cause catastrophic placement misses (one seed landed the object ~51m
            from the pad, almost certainly a real collision/physics ejection from
            an orientation the attached-object collision approximation never
            validated) despite every stage reporting plan_success=True. Interior
            transit stages stay relaxed (that's where the real IK_FAIL bottleneck
            was, and they're always followed by at least one more stage -- ending
            at this forced-strict one -- so any orientation drift there gets
            corrected before the place_actor handoff."""
        qpos = np.array(start_qpos, dtype=np.float64, copy=True)
        trajectories = []
        start_pose = list(self.get_arm_pose(arm_tag))
        for stage_name, pose in stage_poses:
            stage_relax_orientation = relax_orientation and stage_name not in PLACEMENT_STRICT_ORIENTATION_STAGES
            ok, fail_reason, qpos, stage_trajs = self._plan_pose_with_local_waypoint_retry(
                arm_tag, pose, qpos, stage_name, start_pose=start_pose,
                local_attempts=local_attempts, retries=retries,
                relax_orientation=stage_relax_orientation)
            if not ok:
                return False, stage_name, fail_reason, trajectories
            trajectories.extend(stage_trajs)
            start_pose = pose
        return True, None, None, trajectories


    def _time_stretch_trajectory(self, result, factor):
        """Densify a planned joint path while preserving its geometry."""
        factor = max(1, int(factor))
        positions = np.asarray(result.get("position"))
        if factor == 1 or positions.ndim < 2 or len(positions) < 2:
            return result

        stretched = deepcopy(result)
        sample = np.linspace(0.0, len(positions) - 1,
                             (len(positions) - 1) * factor + 1)
        lo = np.floor(sample).astype(int)
        hi = np.minimum(lo + 1, len(positions) - 1)
        alpha = (sample - lo).reshape((-1,) + (1,) * (positions.ndim - 1))
        for key, derivative_order in (("position", 0), ("velocity", 1),
                                      ("acceleration", 2), ("jerk", 3)):
            values = result.get(key)
            if values is None:
                continue
            values = np.asarray(values)
            if values.ndim == 0 or len(values) != len(positions):
                continue
            blended = values[lo] * (1.0 - alpha) + values[hi] * alpha
            if derivative_order:
                blended = blended / (factor ** derivative_order)
            stretched[key] = blended.astype(values.dtype, copy=False)
        return stretched


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
            self._last_fail_reason = (cached_result or {}).get("fail_reason", "unknown")
            return
        if str(arm_tag) == "left":
            if self.need_plan:
                self.left_joint_path.append(deepcopy(cached_result))
            else:
                cached_result = deepcopy(self.left_joint_path[self.left_cnt])
                self.left_cnt += 1
            replay_result = self._time_stretch_trajectory(
                cached_result, ATTACHED_TRAJECTORY_SLOWDOWN
                if self._slow_attached_replay else 1)
            control_seq = {"left_arm": replay_result, "left_gripper": None, "right_arm": None, "right_gripper": None}
        else:
            if self.need_plan:
                self.right_joint_path.append(deepcopy(cached_result))
            else:
                cached_result = deepcopy(self.right_joint_path[self.right_cnt])
                self.right_cnt += 1
            replay_result = self._time_stretch_trajectory(
                cached_result, ATTACHED_TRAJECTORY_SLOWDOWN
                if self._slow_attached_replay else 1)
            control_seq = {"left_arm": None, "left_gripper": None, "right_arm": replay_result, "right_gripper": None}
        self.take_dense_action(control_seq)


    def _plan_and_replay_pose(self, arm_tag, pose, seed_traj=None, stage_label="pose"):
        """Plan a single pose from the CURRENT live qpos and immediately replay
            it via _replay_planned_move, instead of self.move(self.move_to_pose(...)).
            For a lone plan-then-immediately-execute call like this there's no
            separate earlier verification pass to diverge from (unlike waypoint/
            grasp/place_actor), so this isn't fixing a re-plan-divergence bug here --
            its value is exposing the specific CuRobo fail_reason via
            self._last_fail_reason for the structured failure recorder, and using the
            same execution primitive as the rest of this file for consistency.

            seed_traj: optional clearance-route trajopt seed (Phase C). Warm-starts only the
            DIRECT attempt inside the retry helper; the shrinking-waypoint fallback stays
            unseeded, matching how the approach leg is seeded. None => unseeded, i.e. identical
            to what this did before, which is what makes seeded-vs-unseeded a one-variable
            contrast at this call site.
            stage_label: what the plan-effort recorder files this attempt under. Defaults to the
            historical "pose" so existing call sites keep their labels."""
        if not self.plan_success:
            return
        start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
        ok, fail_reason, _, trajectories = self._plan_pose_with_local_waypoint_retry(
            arm_tag, pose, np.array(start_qpos, dtype=np.float64, copy=True), stage_label,
            seed_traj=seed_traj)
        if not ok:
            self.plan_success = False
            self._last_fail_reason = fail_reason
            return
        self._replay_planned_sequence(arm_tag, trajectories)


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


    def _local_waypoint_candidates(self, arm_tag, start_pose, target_pose, attempts=LOCAL_WAYPOINT_ATTEMPTS):
        """Bridge candidates near the current/start pose for any Cartesian move."""
        try:
            attempts = max(0, int(attempts))
        except (TypeError, ValueError):
            attempts = LOCAL_WAYPOINT_ATTEMPTS
        if attempts == 0:
            return []
        start = np.asarray(start_pose, dtype=float)
        target = np.asarray(target_pose, dtype=float)
        side = 1.0 if str(arm_tag) == "right" else -1.0
        specs = [
            ("offset", 0.00, 0.00, 0.04, 0.00),
            ("offset", 0.04, 0.00, 0.04, 0.00),
            ("offset", 0.08, 0.00, 0.06, 0.00),
            ("offset", 0.00, -0.04, 0.06, 0.00),
            ("offset", 0.04, -0.04, 0.08, 0.00),
            ("offset", 0.08, -0.04, 0.08, 0.00),
            ("offset", 0.00, 0.04, 0.06, 0.00),
            ("blend", 0.00, 0.00, 0.04, 0.35),
            ("blend", 0.04, 0.00, 0.06, 0.35),
            ("blend", 0.00, -0.04, 0.08, 0.35),
            ("offset", 0.10, 0.00, 0.10, 0.00),
            ("offset", 0.10, -0.06, 0.10, 0.00),
            # Added when the bridge search gained an IK prefilter (see
            # _plan_pose_with_local_waypoint_retry): a bigger pure-vertical lift
            # (no lateral offset -- the direction that resolved the escape/
            # pre_beside_box INVALID_START_STATE_WORLD_COLLISION failures) and a
            # later blend fraction, both cheap to add now that infeasible
            # candidates are filtered before spending a trajopt call on them.
            ("offset", 0.00, 0.00, 0.10, 0.00),
            ("blend", 0.00, 0.00, 0.00, 0.50),
        ]
        candidates = []
        for mode, dx_side, dy, dz, blend_t in specs[:attempts]:
            pose = self._blend_pose(start, target, blend_t) if mode == "blend" else list(start)
            pose[0] = float(np.clip(pose[0] + side * dx_side, -REACH_X_LIMIT, REACH_X_LIMIT))
            pose[1] = float(np.clip(pose[1] + dy, -0.33, 0.30))
            pose[2] = float(np.clip(pose[2] + dz, 0.78, 1.30))
            candidates.append(pose)
        return candidates


    def _plan_pose_with_shrinking_waypoint(self, arm_tag, target_pose, qpos, stage_label,
                                          start_pose, constraint_pose=None, approach_axis=None,
                                          relax_orientation=False,
                                          max_iterations=LOCAL_WAYPOINT_ATTEMPTS,
                                          min_distance=WAYPOINT_SHRINK_MIN_DISTANCE):
        """Adaptive waypoint-shrink retry, used when a direct plan to target_pose
            has already failed.

            CuRobo's MotionGenResult sets position_error/rotation_error UNCONDITIONALLY
            even on FINETUNE_TRAJOPT_FAIL (confirmed via source read of motion_gen.py's
            _plan_from_solve_state) -- a failed attempt isn't pure noise, it tells us
            how far off the optimizer's best attempt landed. Instead of only trying
            hand-picked fixed offset directions (the old _local_waypoint_candidates
            bridge pool), each iteration here either (a) retries the ORIGINAL target
            from wherever the previous iteration actually landed, or (b) if that
            fails, shrinks the target toward the current pose and retries that
            smaller hop instead. The shrink amount is informed by position_error when
            CuRobo reports one: back the target off by roughly the residual gap
            (plus a safety margin) instead of blindly halving every time, so a
            near-miss shrinks only a little (another attempt might close a small
            gap) while a wild miss shrinks a lot. Falls back to a plain 50% halving
            when no usable position_error is available (e.g. IK_FAIL, where
            trajopt never ran). Shrinking toward the current pose (by definition
            already reachable) converges toward something plannable even for
            targets that are hard/far, unlike fixed offsets which can themselves be
            unreachable. Stops on success at the real target, on max_iterations, or
            once the remaining hop is already < min_distance from the current pose
            (not worth shrinking further). Note: this does NOT replay CuRobo's own
            failed near-miss trajectory (its full path was never verified
            collision-safe, only its final error magnitude is known) -- it only
            uses the error MAGNITUDE to size a fresh, independently-planned hop."""
        log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
        plan_func = self._arm_plan_func(arm_tag)
        qpos0 = np.array(qpos, dtype=np.float64, copy=True)
        current_pose = np.asarray(start_pose, dtype=float)
        original_target = np.asarray(target_pose, dtype=float)
        goal = original_target.copy()
        trajectories = []
        last_reason = "unknown"
        for iteration in range(1, max(1, int(max_iterations)) + 1):
            result = plan_func(list(goal), last_qpos=qpos0, constraint_pose=constraint_pose,
                               approach_axis=approach_axis, relax_orientation=relax_orientation)
            reached_original = bool(np.allclose(goal[:3], original_target[:3], atol=1e-6))
            if result.get("status") == "Success":
                qpos0 = self._roll_qpos_forward(arm_tag, qpos0, result)
                label = stage_label if reached_original else f"{stage_label}:shrink{iteration}"
                trajectories.append((label, result))
                if reached_original:
                    return True, None, qpos0, trajectories
                if log_move:
                    print(f"[waypoint-shrink] {stage_label} iter {iteration}: reached shrunk "
                          f"intermediate, retrying real target from there")
                current_pose = goal.copy()
                goal = original_target.copy()
                continue
            last_reason = result.get("fail_reason", "unknown")
            remaining_dist = float(np.linalg.norm(goal[:3] - current_pose[:3]))
            if remaining_dist < min_distance:
                if log_move:
                    print(f"[waypoint-shrink] {stage_label} iter {iteration} failed ({last_reason}); "
                          f"remaining distance {remaining_dist:.3f}m < {min_distance}m, giving up")
                break
            position_error = result.get("position_error")
            if position_error is not None and 0 < position_error < remaining_dist:
                # Back off by roughly how far short CuRobo's own best attempt
                # landed (2x margin so we don't retry right at the same edge
                # of infeasibility), instead of blindly halving every time.
                target_remaining = max(min_distance, remaining_dist - 2.0 * position_error)
                shrink_fraction = target_remaining / remaining_dist
            else:
                shrink_fraction = 0.5
            goal[:3] = current_pose[:3] + (goal[:3] - current_pose[:3]) * shrink_fraction
            if log_move:
                print(f"[waypoint-shrink] {stage_label} iter {iteration} failed ({last_reason}, "
                      f"position_error={position_error}); shrinking target toward current pose "
                      f"(fraction={shrink_fraction:.2f}) -> new distance "
                      f"{float(np.linalg.norm(goal[:3] - current_pose[:3])):.3f}m")
        return False, last_reason, qpos0, trajectories


    def _record_plan_effort(self, stage_label, arm_tag, result):
        """ROBOPRO Phase 4: log curobo's effort for ONE plan_path call so the
            APPROACH_MODE A/B can report attempts, not just success rate. Recorded for
            the DIRECT attempt only (the stage the experiment varies); the
            shrinking-waypoint fallback is deliberately not counted here so 'attempts'
            stays comparable across modes. Read by run() into records.jsonl. Never raises
            -- this is measurement, it must not be able to fail a rollout."""
        try:
            if not hasattr(self, "rollout_plan_effort"):
                self.rollout_plan_effort = []
            self.rollout_plan_effort.append({
                "stage": stage_label,
                "arm": str(arm_tag),
                "status": result.get("status"),
                "attempts": int(result.get("attempts", 0) or 0),
                "trajopt_attempts": int(result.get("trajopt_attempts", 0) or 0),
                "seeded": bool(result.get("seeded", False)),
            })
        except Exception:
            pass


    def _plan_pose_with_local_waypoint_retry(self, arm_tag, target_pose, qpos, stage_label,
                                            start_pose=None, retries=1, constraint_pose=None,
                                            approach_axis=None, local_attempts=LOCAL_WAYPOINT_ATTEMPTS,
                                            relax_orientation=False, seed_traj=None):
        """Plan one target; if direct planning fails, try the adaptive
            waypoint-shrink retry (see _plan_pose_with_shrinking_waypoint).

            relax_orientation=True: position-only goal (PoseCostMetric.
            reach_partial_pose, see planner.py's plan_path) -- for transit-only moves
            where the target's orientation doesn't matter, only its position. A
            check_ik_batch(relax_orientation=True) sweep confirmed this opens up
            real, currently-dead combos (e.g. lift_above_box at low clearance_z)."""
        qpos0 = np.array(qpos, dtype=np.float64, copy=True)
        start_pose = list(self.get_arm_pose(arm_tag)) if start_pose is None else list(start_pose)
        last_reason = "unknown"
        plan_func = self._arm_plan_func(arm_tag)
        for _ in range(max(1, int(retries))):
            # ROBOPRO Phase 3: the external seed (if any) warm-starts only the DIRECT attempt; the
            # shrinking-waypoint fallback below stays unseeded (it's the safety net). None => stock.
            result = plan_func(target_pose, last_qpos=qpos0,
                               constraint_pose=constraint_pose, approach_axis=approach_axis,
                               relax_orientation=relax_orientation, seed_traj=seed_traj)
            self._record_plan_effort(stage_label, arm_tag, result)
            if result.get("status") == "Success":
                return True, None, self._roll_qpos_forward(arm_tag, qpos0, result), [(stage_label, result)]
            last_reason = result.get("fail_reason", "unknown")
        if local_attempts <= 0:
            return False, last_reason, qpos0, []
        return self._plan_pose_with_shrinking_waypoint(
            arm_tag, target_pose, qpos0, stage_label, start_pose,
            constraint_pose=constraint_pose, approach_axis=approach_axis,
            relax_orientation=relax_orientation, max_iterations=local_attempts)


    def _replay_planned_sequence(self, arm_tag, trajectories):
        """Replay one planned trajectory or a list of labeled trajectories."""
        if trajectories is None:
            return
        if isinstance(trajectories, dict):
            trajectories = [trajectories]
        for item in trajectories:
            traj = item[1] if isinstance(item, tuple) else item
            self._replay_planned_move(arm_tag, traj)
            if not self.plan_success:
                return


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
        start_pose = list(self.get_arm_pose(arm_tag))
        for idx, pose in enumerate(poses):
            label = stage_labels[idx] if stage_labels else f"pose{idx}"
            ok, fail_reason, qpos, _ = self._plan_pose_with_local_waypoint_retry(
                arm_tag, pose, qpos, label, start_pose=start_pose)
            if not ok:
                return False, label, fail_reason, qpos
            start_pose = pose
        return True, None, None, qpos
