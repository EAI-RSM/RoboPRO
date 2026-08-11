"""Planning-free nominal waypoint chain for the selected stock task."""

from dataclasses import dataclass

import numpy as np
import transforms3d as t3d

from .task_roles import SUPPORTED_TASK, resolve_task_roles


LEG_KINDS = {"pre_grasp", "grasp", "lift", "carry", "pre_place", "place"}


@dataclass(frozen=True)
class Waypoint:
    xyz: tuple
    quat: tuple
    kind: str | None
    arm: str
    gripper_state: str


@dataclass(frozen=True)
class CanonicalLeg:
    start_xyz: tuple
    goal_xyz: tuple
    kind: str
    gripper_state: str


def _waypoint(pose, kind, arm, gripper_state):
    pose = np.asarray(pose, dtype=float).reshape(7)
    return Waypoint(
        tuple(float(v) for v in pose[:3]),
        tuple(float(v) for v in pose[3:]),
        kind,
        arm,
        gripper_state,
    )


def canonical_waypoints(env, task_name):
    """Return the nominal, not expert-realized, cup-on-coaster path.

    Contact point zero is deliberately fixed so scene tightness cannot influence waypoint
    selection through a planner.  The holding legs retain the nominal grasp transform, but the
    held cup never enters metric occupancy or clearance.
    """
    if task_name != SUPPORTED_TASK:
        raise ValueError(f"no waypoint adapter for {task_name!r}")
    roles = resolve_task_roles(env, task_name)
    arm = str(env.side_to_place)
    if arm not in {"left", "right"}:
        raise ValueError(f"invalid acting arm from side_to_place: {arm!r}")

    home = np.asarray(
        env.robot.get_left_ee_pose() if arm == "left" else env.robot.get_right_ee_pose(),
        dtype=float,
    ).reshape(7)
    contact = np.asarray(roles.target.actor.get_contact_point(0, "matrix"), dtype=float)
    if contact.shape != (4, 4):
        raise ValueError("cup contact point 0 did not produce a 4x4 world matrix")

    # Same fixed contact-frame remapping and 0.12 m tool offset as Base_Task.get_grasp_pose,
    # without its planner-backed choose_best_pose call.
    contact_to_gripper = np.eye(4)
    contact_to_gripper[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    grasp_frame = contact @ contact_to_gripper
    rotation = grasp_frame[:3, :3]
    contact_xyz = grasp_frame[:3, 3]
    pre_grasp_xyz = contact_xyz + rotation @ np.array([-0.22, 0.0, 0.0])
    grasp_xyz = contact_xyz + rotation @ np.array([-0.12, 0.0, 0.0])
    grasp_quat = t3d.quaternions.mat2quat(rotation)

    target_xyz = np.asarray(roles.target.actor.get_pose().p, dtype=float)
    destination_pose = np.asarray(env.des_obj_pose, dtype=float).reshape(7)
    grasp_bias = target_xyz - grasp_xyz
    approach = grasp_bias / np.linalg.norm(grasp_bias)
    # put_cup_on_coaster.play_once uses pre_dis=0.07 and dis=0.005.  Keeping the
    # grasp transform fixed gives a deterministic nominal placement proxy.
    pre_place_xyz = destination_pose[:3] - grasp_bias - 0.07 * approach
    place_xyz = destination_pose[:3] - grasp_bias - 0.005 * approach

    return [
        _waypoint(home, None, arm, "empty"),
        _waypoint(np.r_[pre_grasp_xyz, grasp_quat], "pre_grasp", arm, "empty"),
        _waypoint(np.r_[grasp_xyz, grasp_quat], "grasp", arm, "empty"),
        _waypoint(np.r_[pre_place_xyz, grasp_quat], "carry", arm, "holding"),
        _waypoint(np.r_[place_xyz, grasp_quat], "place", arm, "holding"),
    ]


def canonical_legs(waypoints):
    """Turn the ordered chain into metric requests and leg metadata."""
    if len(waypoints) < 2:
        raise ValueError("a canonical path needs at least two waypoints")
    legs = []
    for start, goal in zip(waypoints, waypoints[1:]):
        if goal.kind not in LEG_KINDS:
            raise ValueError(f"invalid canonical leg kind: {goal.kind!r}")
        if start.arm != goal.arm:
            raise ValueError("canonical leg changes acting arm")
        if goal.gripper_state not in {"empty", "holding"}:
            raise ValueError(f"invalid gripper state: {goal.gripper_state!r}")
        legs.append(
            CanonicalLeg(start.xyz, goal.xyz, goal.kind, goal.gripper_state)
        )
    return legs
