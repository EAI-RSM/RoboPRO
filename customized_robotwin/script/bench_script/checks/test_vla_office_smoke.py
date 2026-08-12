#!/usr/bin/env python3
"""CPU checks for the staged office-scene VLA smoke path."""

from __future__ import annotations

import os
import sys
import tempfile
import types
from argparse import Namespace
from pathlib import Path

import numpy as np

BENCH_SCRIPT_ROOT = Path(__file__).resolve().parent.parent
if str(BENCH_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPT_ROOT))

import vla_rollout as rollout

# Bench_base_task imports the real Robot class eagerly, which imports CuRobo and
# initializes CUDA. This test exercises take_action with a fake robot only.
robot_module = types.ModuleType("envs.robot")
robot_module.Robot = object
sys.modules["envs.robot"] = robot_module

from bench_envs._bench_base_task import Bench_base_task
from bench_envs.eval_video import select_eval_video_camera
from bench_envs.study.put_cup_on_coaster import put_cup_on_coaster
from lib.scene_build import DR_CLEAN, build_cfg
from lib.task_roles import resolve_task_roles
from lib.waypoints import canonical_waypoints


def _fail(*_args, **_kwargs):
    raise AssertionError("office mode touched occluder-ring code")


def test_office_scene_selection() -> None:
    args = Namespace(
        scene="office",
        offsets="invalid-on-purpose",
        num_occluders="invalid-on-purpose",
        clutter_densities="invalid-on-purpose",
        no_occluder_prob=0.0,
        random_ring_rotation=True,
        task_name="unused",
        bench_subdir="unused",
    )

    original_parse_offsets = rollout.parse_offset_specs
    original_parse_counts = rollout.parse_count_choices
    original_draw_ring = rollout.draw_ring_config
    try:
        rollout.parse_offset_specs = _fail
        rollout.parse_count_choices = _fail
        rollout.draw_ring_config = _fail
        assert rollout._scene_sweep(args) == ([None], [], [0])
        assert rollout._scene_parameters(args, 7, None, []) == (
            None,
            False,
            None,
            0,
            [],
        )
    finally:
        rollout.parse_offset_specs = original_parse_offsets
        rollout.parse_count_choices = original_parse_counts
        rollout.draw_ring_config = original_draw_ring

    office_env = object()
    original_get_env_class = rollout.get_env_class
    try:
        def fake_get_env_class(task_name, bench_subdir=None):
            assert task_name == "put_mouse_on_pad"
            assert bench_subdir == "office"
            return lambda: office_env

        rollout.get_env_class = fake_get_env_class
        assert rollout._make_env(args) is office_env
    finally:
        rollout.get_env_class = original_get_env_class

    assert (
        rollout._instruction_for(None, "office")
        == "Put the mouse onto the mouse pad."
    )
    assert (
        rollout._instruction_for("Pinned instruction.", "office")
        == "Pinned instruction."
    )
    assert rollout._instruction_for(None, "occluder") == rollout.OCCLUDER_INSTRUCTION


def test_stock_task_selection_and_outcome() -> None:
    args = Namespace(
        scene="task",
        task_name="put_cup_on_coaster",
        bench_subdir="study",
        clutter_densities="0,10",
        offsets="invalid-on-purpose",
        num_occluders="invalid-on-purpose",
        no_occluder_prob=0.0,
        random_ring_rotation=True,
    )
    assert rollout._task_context(args) == ("put_cup_on_coaster", "study")
    assert rollout._scene_sweep(args) == ([None], [], [0, 10])
    assert rollout._scene_parameters(args, 7, None, []) == (
        None,
        False,
        None,
        0,
        [],
    )

    task_env = object()
    original_get_env_class = rollout.get_env_class
    try:
        def fake_get_env_class(task_name, bench_subdir=None):
            assert task_name == "put_cup_on_coaster"
            assert bench_subdir == "study"
            return lambda: task_env

        rollout.get_env_class = fake_get_env_class
        assert rollout._make_env(args) is task_env
    finally:
        rollout.get_env_class = original_get_env_class

    instruction = rollout._instruction_for(None, "task", "put_cup_on_coaster")
    assert instruction == "Place the cup onto the coaster on the table."
    assert rollout._hard_success(True, {"is_collision": False}) is True
    assert rollout._hard_success(True, {"is_collision": True}) is False
    assert rollout._hard_success(False, {"is_collision": False}) is False
    for malformed in (None, {}, {"is_collision": None}):
        try:
            rollout._hard_success(True, malformed)
        except (RuntimeError, TypeError):
            pass
        else:
            raise AssertionError(f"accepted malformed collision metrics: {malformed!r}")


class _TaskPose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)


class _TaskActor:
    def __init__(self, name, p, contact=None):
        self._name = name
        self._pose = _TaskPose(p)
        self._contact = contact
        self.scale = 1.0

    def get_name(self):
        return self._name

    def get_pose(self):
        return self._pose

    def get_contact_point(self, index, ret):
        assert index == 0 and ret == "matrix"
        return self._contact.copy()


class _TaskRobot:
    def update_world(self, collision_dict):
        self.collision_dict = collision_dict

    def get_left_ee_pose(self):
        return [-0.4, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]

    def get_right_ee_pose(self):
        return [0.4, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]


def _task_api_env(root):
    for name, model_id in (("021_cup", "2"), ("019_coaster", "0"), ("block", "3")):
        path = root / "assets" / "objects" / name / "collision"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"base{model_id}.glb").write_bytes(b"mesh")
    contact = np.eye(4)
    contact[:3, 3] = [0.2, -0.1, 0.9]
    target = _TaskActor("021_cup", [0.2, -0.1, 0.82], contact)
    destination = _TaskActor("019_coaster", [-0.2, 0.15, 0.82])
    obstacle = _TaskActor("block", [0.0, 0.0, 0.85])
    return Namespace(
        target_obj=target,
        target_name="021_cup",
        target_id=2,
        des_obj=destination,
        des_obj_id=0,
        des_obj_pose=[-0.2, 0.10, 0.82, 1.0, 0.0, 0.0, 0.0],
        side_to_place="right",
        robot=_TaskRobot(),
        collision_list=[
            {"actor": target, "collision_path": str(root / "assets/objects/021_cup/collision/base2.glb")},
            {"actor": destination, "collision_path": str(root / "assets/objects/019_coaster/collision/base0.glb")},
            {"actor": obstacle, "collision_path": str(root / "assets/objects/block/collision/base3.glb"), "is_obstacle": True},
        ],
        cuboid_collision_list=[],
        seed=0,
    )


def test_live_task_roles_and_waypoints() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        previous = os.environ.get("BENCH_ROOT")
        os.environ["BENCH_ROOT"] = str(root)
        try:
            env = _task_api_env(root)
            roles = resolve_task_roles(env, "put_cup_on_coaster")
            first = canonical_waypoints(env, "put_cup_on_coaster")
            second = canonical_waypoints(env, "put_cup_on_coaster")
            Bench_base_task.update_world(env, exclude_obstacles=True)
            excluded_count = env._last_curobo_collision_world_entry_count
            Bench_base_task.update_world(env, exclude_obstacles=False)
            included_count = env._last_curobo_collision_world_entry_count
        finally:
            if previous is None:
                os.environ.pop("BENCH_ROOT", None)
            else:
                os.environ["BENCH_ROOT"] = previous
    assert roles.target.actor is env.target_obj
    assert roles.destination.actor is env.des_obj
    assert [role.name for role in roles.obstacles] == ["block"]
    assert first == second
    assert first[0].arm == "right"
    assert excluded_count == 2
    assert included_count == 3
    assert len(env.robot.collision_dict["mesh"]) == included_count


def test_policy_camera_config() -> None:
    cfg = build_cfg(
        "put_mouse_on_pad",
        "bench_demo_office_clean",
        0,
        DR_CLEAN,
        mode="policy",
        eval_video_camera="countertop_camera",
    )
    assert cfg["eval_video_camera"] == "countertop_camera"
    assert cfg["domain_randomization"]["cluttered_table"] is False
    assert cfg["domain_randomization"]["obstacle_density"] == 0

    unpinned = build_cfg(
        "put_mouse_on_pad",
        "bench_demo_office_clean",
        0,
        DR_CLEAN,
        mode="policy",
    )
    assert "eval_video_camera" not in unpinned


def test_video_camera_selection() -> None:
    observation = {
        "demo_camera": object(),
        "countertop_camera": object(),
        "head_camera": object(),
    }
    assert (
        select_eval_video_camera(observation, "countertop_camera")
        == "countertop_camera"
    )
    assert select_eval_video_camera(observation) == "demo_camera"

    try:
        select_eval_video_camera(observation, "missing_camera")
    except KeyError as exc:
        assert "missing_camera" in str(exc)
    else:
        raise AssertionError("missing requested camera did not fail")


def test_cup_coaster_success_uses_absolute_xy_error() -> None:
    class FakePose:
        def __init__(self, xy):
            self.p = np.array([xy[0], xy[1], 0.0], dtype=float)

    class FakeActor:
        def __init__(self, xy):
            self.xy = xy

        def get_pose(self):
            return FakePose(self.xy)

    class FakeRobot:
        def __init__(self, *, left_open=True, right_open=True):
            self.left_open = left_open
            self.right_open = right_open

        def is_left_gripper_open(self):
            return self.left_open

        def is_right_gripper_open(self):
            return self.right_open

    env = put_cup_on_coaster.__new__(put_cup_on_coaster)
    env.des_obj = FakeActor((0.0, 0.0))
    env.robot = FakeRobot()

    for target_xy in ((0.03, 0.0), (-0.03, 0.0)):
        env.target_obj = FakeActor(target_xy)
        assert not env.check_success()

    env.target_obj = FakeActor((0.019, -0.019))
    assert env.check_success()

    env.robot = FakeRobot(left_open=False)
    assert not env.check_success()


def _run_fake_qpos_action(*, build_planner: bool):
    class FakePlanner:
        def __init__(self):
            self.paths = []

        def TOPP(self, path, timestep, verbose):
            self.paths.append((path.copy(), timestep, verbose))
            return (
                np.array([0.0, timestep]),
                path.copy(),
                np.full_like(path, 0.25),
                np.zeros_like(path),
                timestep,
            )

    class FakeRobot:
        def __init__(self):
            self.build_planner = build_planner
            self.left_mplib_planner = FakePlanner()
            self.right_mplib_planner = FakePlanner()
            self.arm_commands = {"left": [], "right": []}

        def get_left_arm_jointState(self):
            return [0.0] * 7

        def get_right_arm_jointState(self):
            return [0.0] * 7

        def get_left_gripper_val(self):
            return 0.0

        def get_right_gripper_val(self):
            return 0.0

        def set_arm_joints(self, position, velocity, arm_tag):
            self.arm_commands[arm_tag].append(
                (np.asarray(position).copy(), np.asarray(velocity).copy())
            )

        def set_gripper(self, _value, _arm_tag):
            pass

    class FakeScene:
        def step(self):
            pass

    env = Bench_base_task.__new__(Bench_base_task)
    env.take_action_cnt = 0
    env.step_lim = 10
    env.eval_success = False
    env.eval_video_path = None
    env.render_freq = 0
    env.enable_collision_metrics = False
    env.robot = FakeRobot()
    env.scene = FakeScene()
    env._update_render = lambda: None
    env.check_success = lambda: False

    target = np.arange(1.0, 15.0)
    env.take_action(target, action_type="qpos")
    return env.robot, target


def test_planner_free_qpos_moves_arms() -> None:
    robot, target = _run_fake_qpos_action(build_planner=False)

    for arm_tag, expected_target in (
        ("left", target[:6]),
        ("right", target[7:13]),
    ):
        positions = np.array(
            [position for position, _velocity in robot.arm_commands[arm_tag]]
        )
        assert positions.shape == (50, 6)
        assert np.allclose(positions[0], 0.0)
        assert np.allclose(positions[-1], expected_target)
        assert np.any(np.diff(positions, axis=0) != 0.0)

    assert robot.left_mplib_planner.paths == []
    assert robot.right_mplib_planner.paths == []


def test_planner_qpos_keeps_topp_path() -> None:
    robot, target = _run_fake_qpos_action(build_planner=True)

    for planner, expected_target in (
        (robot.left_mplib_planner, target[:6]),
        (robot.right_mplib_planner, target[7:13]),
    ):
        assert len(planner.paths) == 1
        path, timestep, verbose = planner.paths[0]
        assert np.allclose(path[0], 0.0)
        assert np.allclose(path[1], expected_target)
        assert timestep == 1 / 250
        assert verbose is True

    for arm_tag in ("left", "right"):
        velocities = np.array(
            [velocity for _position, velocity in robot.arm_commands[arm_tag]]
        )
        assert velocities.shape == (2, 6)
        assert np.allclose(velocities, 0.25)


def main() -> None:
    test_office_scene_selection()
    test_stock_task_selection_and_outcome()
    test_live_task_roles_and_waypoints()
    test_policy_camera_config()
    test_video_camera_selection()
    test_cup_coaster_success_uses_absolute_xy_error()
    test_planner_free_qpos_moves_arms()
    test_planner_qpos_keeps_topp_path()
    print("PASS: stock/office scenes, strict HSR, countertop video, and qpos actions")


if __name__ == "__main__":
    main()
