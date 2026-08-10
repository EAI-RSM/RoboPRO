"""Methods extracted mechanically from analyze_occluder_visibility.py."""

import os

import numpy as np
import torch
import transforms3d as t3d
from curobo.types.state import JointState

from lib.planning_tuning import *  # noqa: F403
from lib.scene_constants import *  # noqa: F403


class StateChecksMixin:
    def _gripper_relative_object_transform(self, arm_tag):
        """The held object's FULL pose (position AND orientation) expressed in
            the gripper's LOCAL frame -- stable under gripper translation/rotation
            as long as the object is still rigidly held, unlike a raw world-frame
            offset (which changes with gripper orientation even with zero slip).
            Returns (relative_position, relative_quaternion)."""
        gripper_pose = np.asarray(self.get_arm_pose(arm_tag), dtype=float)
        gripper_rot = t3d.quaternions.quat2mat(gripper_pose[3:])
        object_pose = self.target_obj.get_pose()
        object_pos = np.asarray(object_pose.p, dtype=float)
        object_rot = t3d.quaternions.quat2mat(np.asarray(object_pose.q, dtype=float))
        relative_pos = gripper_rot.T @ (object_pos - gripper_pose[:3])
        relative_quat = t3d.quaternions.mat2quat(gripper_rot.T @ object_rot)
        return relative_pos, relative_quat


    def _object_near_placement_target(self, xy_tolerance=OBJECT_PLACEMENT_XY_TOLERANCE,
                                      z_tolerance=OBJECT_PLACEMENT_Z_TOLERANCE):
        """Has the held object already reached a pose that WOULD PASS
            put_mouse_on_pad.check_success() -- independent x/y error each under
            xy_tolerance (mirroring check_success's own per-axis 0.02 criterion,
            not a looser combined radius)? Used by place_actor's descent to stop
            and open the gripper immediately once this is genuinely true, rather
            than continuing to descend past a placement that's already good.
            Deliberately conservative: an earlier combined-radius version
            accepted seed 9 (dx=3.24cm) and seed 12 (dx=6.11cm) as "arrived" even
            though check_success() would still reject both on at least one
            axis -- see DESCENT_CONTACT_DRIFT_TOLERANCE for how a drift that
            ISN'T yet this precise, but also isn't a genuine loss, is handled."""
        if not getattr(self, "des_obj_pose", None):
            return False
        obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
        target = np.asarray(self.des_obj_pose, dtype=float)
        dx = abs(float(obj_pos[0] - target[0]))
        dy = abs(float(obj_pos[1] - target[1]))
        z_dist = float(obj_pos[2] - target[2])  # positive: object sits above the target resting height
        return dx <= xy_tolerance and dy <= xy_tolerance and -z_tolerance <= z_dist <= z_tolerance


    def _object_near_support_height(self, height_tolerance=DESCENT_CONTACT_HEIGHT_TOLERANCE):
        """Is the held object's world-frame z plausibly at the placement
            surface right now (independent of x/y)? Used to gate
            DESCENT_CONTACT_DRIFT_TOLERANCE's ambiguous-contact branch: a
            gripper-relative drift in that band is only explainable as early
            contact if the object is actually near the resting height it would
            be contacting -- a genuine drop shows the same drift magnitude
            while still high above the surface, since it's falling under
            gravity rather than following the descent's planned trajectory."""
        if not getattr(self, "des_obj_pose", None):
            return False
        obj_z = float(self.target_obj.get_pose().p[2])
        target_z = float(self.des_obj_pose[2])
        return abs(obj_z - target_z) <= height_tolerance


    def _placement_xy_error(self):
        """Signed target-minus-object XY error in world coordinates."""
        obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
        target = np.asarray(self.des_obj_pose, dtype=float)
        return target[:2] - obj_pos[:2]


    def _object_retained(self, arm_tag, tolerance=OBJECT_RETENTION_TOLERANCE,
                         rotation_tolerance=OBJECT_RETENTION_ROTATION_TOLERANCE, context=None,
                         return_drift=False):
        """True if the held object is still where it was (position AND
            orientation) relative to the gripper when _grasp_baseline_transform
            was captured (right after attach_object in play_once) -- i.e. the
            grasp hasn't slipped, rotated loose, or been dropped since. Confirmed
            empirically (seed 27): every placement stage reported
            plan_success=True while the object physically fell mid-transit, since
            plan_success only reflects the ARM's motion plan, never the HELD
            OBJECT's actual state. Returns True (nothing to check) if no baseline
            has been captured yet.

            Strict everywhere, including place_actor's descent -- callers there
            should check _object_near_placement_target FIRST and short-circuit
            before ever calling this, since a legitimate early placement also
            changes the gripper-relative transform and would otherwise look
            identical to a drop."""
        if self._grasp_baseline_transform is None:
            return (True, 0.0, 0.0) if return_drift else True
        baseline_pos, baseline_quat = self._grasp_baseline_transform
        current_pos, current_quat = self._gripper_relative_object_transform(arm_tag)
        pos_drift = float(np.linalg.norm(current_pos - baseline_pos))
        rot_dot = float(np.clip(abs(np.dot(baseline_quat, current_quat)), -1.0, 1.0))
        rot_drift = float(2.0 * np.arccos(rot_dot))
        retained = pos_drift <= tolerance and rot_drift <= rotation_tolerance
        self.rollout_retention_checks.append({
            "context": context, "pos_drift": pos_drift, "rot_drift": rot_drift, "retained": retained,
        })
        if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
            print(f"[object-retained] context={context} pos_drift={pos_drift:.4f}m (tol={tolerance}) "
                  f"rot_drift={rot_drift:.4f}rad (tol={rotation_tolerance}) retained={retained}")
        if return_drift:
            return retained, pos_drift, rot_drift
        return retained


    def _trajectory_path_metrics(self, arm_tag, result):
        """Safety-filter metrics for a planned descent-slice trajectory,
            via the arm's planner's own forward kinematics. Used by
            _plan_pose_with_descent_slices to catch CuRobo trajopt solves
            that are technically valid but take an implausibly circuitous
            path for what should be a tiny straight-line slice -- see
            DESCENT_MAX_PATH_DEVIATION and friends for the empirical case
            that motivated this (a 3cm slice's solve swinging the gripper
            56cm away before snapping back). Returns None if the trajectory
            has fewer than 2 waypoints (nothing to check).

            - max_perp_deviation: max distance any waypoint's EE position
              strays from the closest point on the FINITE segment between
              the trajectory's own start/end EE pose (projection clamped to
              [0, line_len] -- an unclamped projection would report
              near-zero deviation for a trajectory that overshoots straight
              past the target along the same axis).
            - path_length_ratio: actual Cartesian path length (sum of
              consecutive EE waypoint distances) over the straight-line
              distance -- catches back-and-forth paths that stay close to
              the line but travel much farther than the direct distance.
            - joint_path_length / joint_direct_dist: absolute joint-space
              path length and direct start->end joint-space distance
              (radians, L2 norm) -- exposed directly (not just as a ratio)
              so a monotonic jump to a distant IK branch can be caught by
              its own absolute magnitude, since such a jump has
              joint_travel_ratio near 1.0 regardless of how far it goes.
            - joint_travel_ratio: joint_path_length / joint_direct_dist --
              catches joint-space WINDING an EE-only check can miss (a
              redundant/near-singular arm can reach nearly the same EE pose
              via very different, looping joint paths). A near-zero
              joint_direct_dist (or line_len for path_length_ratio) with a
              non-negligible numerator returns inf rather than silently
              collapsing to a "perfect" 1.0 -- a real loop with almost
              identical start/end shouldn't look the same as "nothing
              moved."
            - max_joint_range: the largest single joint's (max-min)
              excursion across the trajectory, in radians -- flags one
              joint spinning through an implausible range even when the
              aggregate joint_travel_ratio looks acceptable."""
        positions = result.get("position")
        if positions is None or len(positions) < 2:
            return None
        planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
        joint_state = JointState.from_position(
            torch.tensor(positions, dtype=torch.float32).cuda(),
            joint_names=planner.active_joints_name,
        )
        kin_state = planner.motion_gen.compute_kinematics(joint_state)
        ee_pos = np.array(kin_state.ee_pos_seq.to("cpu"), dtype=float)
        start_pos, end_pos = ee_pos[0], ee_pos[-1]
        line_vec = end_pos - start_pos
        line_len = float(np.linalg.norm(line_vec))
        if line_len < 1e-6:
            max_perp_deviation = float(np.max(np.linalg.norm(ee_pos - start_pos, axis=1)))
        else:
            line_unit = line_vec / line_len
            rel = ee_pos - start_pos
            t = np.clip(rel @ line_unit, 0.0, line_len)
            closest = start_pos + np.outer(t, line_unit)
            max_perp_deviation = float(np.max(np.linalg.norm(ee_pos - closest, axis=1)))
        path_length = float(np.sum(np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)))
        if line_len > 1e-6:
            path_length_ratio = path_length / line_len
        elif path_length > 1e-6:
            path_length_ratio = float("inf")
        else:
            path_length_ratio = 1.0

        joint_pos = np.asarray(positions, dtype=float)
        joint_path_length = float(np.sum(np.linalg.norm(np.diff(joint_pos, axis=0), axis=1)))
        joint_direct_dist = float(np.linalg.norm(joint_pos[-1] - joint_pos[0]))
        if joint_direct_dist > 1e-6:
            joint_travel_ratio = joint_path_length / joint_direct_dist
        elif joint_path_length > 1e-6:
            joint_travel_ratio = float("inf")
        else:
            joint_travel_ratio = 1.0
        max_joint_range = float(np.max(joint_pos.max(axis=0) - joint_pos.min(axis=0)))

        return {
            "max_perp_deviation": max_perp_deviation,
            "path_length": path_length,
            "path_length_ratio": path_length_ratio,
            "joint_path_length": joint_path_length,
            "joint_direct_dist": joint_direct_dist,
            "joint_travel_ratio": joint_travel_ratio,
            "max_joint_range": max_joint_range,
        }
