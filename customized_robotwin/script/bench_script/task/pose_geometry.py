"""Methods extracted mechanically from analyze_occluder_visibility.py."""

import os

import numpy as np

from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from lib.planning_tuning import *  # noqa: F403
from lib.scene_constants import *  # noqa: F403


class PoseGeometryMixin:
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
            _select_attached_placement_plan, applied to a transit waypoint: a naive
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
