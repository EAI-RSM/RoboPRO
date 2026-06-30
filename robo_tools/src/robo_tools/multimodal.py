"""Multimodal action generation — parameterize the classical planner to produce
several DISTINCT behaviors (modes) for the SAME task and the SAME initial scene.

Motivation: RoboPRO collects ~1 demo per scene, so a policy conditioned on the
scene only ever sees one way to solve it — the action distribution looks unimodal.
To study / train multimodal behavior we need many *different* valid solutions of
the same scene. This generates them by parameterizing the planner.

Mechanism (general — NO per-task code):
  The scripted expert decides grasp / transport / place as a stream of `Action`
  objects (EE target poses) handed to `env.move()`. A *mode* re-parameterizes the
  grasp->place TRANSPORT only: it injects one mode-specific WAYPOINT before the
  placement, leaving the grasp and place ENDPOINTS untouched. So the task still
  succeeds (same endpoints) but the arm takes a visibly different path. It works
  for any pick-and-place task because it wraps the common `place_actor` helper.

This mirrors robo_negative.TargetedRuntime (attach + method wrapping), but instead
of desyncing the planner to cause *failures*, it parameterizes the planner to
produce *diverse successes*.
"""
import numpy as np

# A mode is one injected transport waypoint, expressed relative to the straight
# grasp->place line:
#   lift_m    : height of the waypoint above the higher endpoint (arc height)
#   lateral_m : signed perpendicular offset (swing the path to one side)
#   waypoint  : False => inject nothing => the planner's default straight transport
MODES = {
    "direct":      {"lift_m": 0.00, "lateral_m":  0.00, "waypoint": False,
                    "desc": "planner default: straight grasp->place transport"},
    "high_arc":    {"lift_m": 0.22, "lateral_m":  0.00, "waypoint": True,
                    "desc": "lift high and arc over the top"},
    "swing_left":  {"lift_m": 0.08, "lateral_m":  0.18, "waypoint": True,
                    "desc": "swing wide to the +lateral side"},
    "swing_right": {"lift_m": 0.08, "lateral_m": -0.18, "waypoint": True,
                    "desc": "swing wide to the -lateral side"},
}
MODE_ORDER = ["direct", "high_arc", "swing_left", "swing_right"]


class MultimodalRuntime:
    """Attach to a bench env to make its planner take a mode-specific transport
    path. Endpoints (grasp, place) are unchanged, so success is preserved."""

    def __init__(self, mode: dict, name: str = ""):
        self.mode = mode
        self.name = name
        self.injected_waypoints = []  # poses injected, for logging / inspection

    def attach(self, env):
        # imported here so the module is importable without the sim on the path
        from envs.utils.action import Action, ArmTag

        original_place = env.place_actor

        def wrapped_place(actor, arm_tag, target_pose, *a, **kw):
            arm_tag, actions = original_place(actor, arm_tag, target_pose, *a, **kw)
            # actions == [] when planning already failed; nothing to detour
            if not actions or not self.mode.get("waypoint"):
                return arm_tag, actions
            arm = ArmTag(arm_tag).arm
            cur = np.asarray(
                env.robot.get_left_ee_pose() if arm == "left" else env.robot.get_right_ee_pose(),
                dtype=np.float64)                       # EE pose at the start of transport
            place_pre = np.asarray(actions[0].target_pose, dtype=np.float64)  # above the destination
            wp = self._waypoint(cur, place_pre)
            self.injected_waypoints.append(wp)
            # detour: current -> WAYPOINT -> place_pre -> place -> open
            return arm_tag, [Action(arm_tag, "move", target_pose=wp)] + list(actions)

        env.place_actor = wrapped_place
        env.multimodal = self

    def _waypoint(self, cur, place_pre):
        """A 7-DoF EE waypoint on the way from `cur` to `place_pre`, lifted and
        swung per the mode. Keeps the current grasp orientation so the held object
        stays put through the detour."""
        start, end = cur[:3], place_pre[:3]
        pos = 0.5 * (start + end)
        pos[2] = max(start[2], end[2]) + float(self.mode["lift_m"])
        d = end[:2] - start[:2]
        perp = np.array([-d[1], d[0]], dtype=np.float64)  # perpendicular to transport (xy)
        n = float(np.linalg.norm(perp))
        if n > 1e-6:
            pos[:2] += (perp / n) * float(self.mode["lateral_m"])
        quat = cur[3:7]  # hold the grasp orientation
        return [float(pos[0]), float(pos[1]), float(pos[2]),
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


# ======================================================================
# Registry — which tasks have been verified for multimodal collection (and which
# modes). Unregistered tasks are SKIPPED by the pipeline (slow, audited rollout),
# mirroring core.SUPPORTED_TASKS for the negative perturbations.
# ======================================================================
MULTIMODAL_TASKS: dict[str, dict] = {
    # Verified 2026-06-30: bread realized all 4 modes; stapler realized direct+swing_left.
    "put_bread_on_board_ks":        {"bench_subdir": "kitchens", "modes": frozenset(MODE_ORDER),
                                     "clean_config": "bench_demo_kitchens_clean"},
    "put_stapler_on_book":          {"bench_subdir": "office",   "modes": frozenset(MODE_ORDER),
                                     "clean_config": "bench_demo_office_clean"},
    "move_hamburger_onto_plate_ks": {"bench_subdir": "kitchens", "modes": frozenset(MODE_ORDER),
                                     "clean_config": "bench_demo_kitchens_clean"},
}


def multimodal_entry(task_name: str) -> dict | None:
    return MULTIMODAL_TASKS.get(task_name)
