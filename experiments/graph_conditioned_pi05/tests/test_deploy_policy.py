import numpy as np

from customized_robotwin.policy.pi05.deploy_policy import (
    _record_motion_or_terminate,
    _truncate_actions_to_step_limit,
)
from experiments.graph_conditioned_pi05.graph_replanning import (
    GraphControllerState,
    GraspSubstage,
    MotionStallDetector,
)
from experiments.graph_conditioned_pi05.simulator_evidence import EffectorEvidence


class Task:
    def __init__(self, count=None, limit=None):
        if count is not None:
            self.take_action_cnt = count
        if limit is not None:
            self.step_lim = limit


def test_action_chunk_is_truncated_to_remaining_episode_steps():
    actions = np.zeros((50, 14), dtype=np.float32)
    assert _truncate_actions_to_step_limit(Task(598, 600), actions).shape == (2, 14)
    assert _truncate_actions_to_step_limit(Task(600, 600), actions).shape == (0, 14)
    assert _truncate_actions_to_step_limit(Task(601, 600), actions).shape == (0, 14)


def test_missing_step_budget_preserves_actions():
    actions = np.zeros((50, 14), dtype=np.float32)
    assert _truncate_actions_to_step_limit(Task(), actions) is actions


def _evidence(left_x, right_x=1.0):
    class Evidence:
        left = EffectorEvidence(tcp_position_world=(left_x, 0.0, 0.0))
        right = EffectorEvidence(tcp_position_world=(right_x, 0.0, 0.0))

    return Evidence()


def test_motion_stall_terminates_after_horizon_without_significant_motion():
    task = Task(50, 600)
    controller = GraphControllerState(
        grasp_substage=GraspSubstage.MOVE_DOWN,
        motion_detector=MotionStallDetector(horizon=3, min_displacement_m=0.002),
    )
    assert not _record_motion_or_terminate(task, controller, _evidence(0.0))
    assert not _record_motion_or_terminate(task, controller, _evidence(0.0005))
    assert not _record_motion_or_terminate(task, controller, _evidence(0.0010))
    assert _record_motion_or_terminate(task, controller, _evidence(0.0015))
    assert task._graph_termination_reason == (
        "graph_motion_stall:grasp:move_down"
    )


def test_significant_motion_and_replan_reset_prevent_stall():
    task = Task(50, 600)
    controller = GraphControllerState(
        motion_detector=MotionStallDetector(horizon=3, min_displacement_m=0.002)
    )
    for position in (0.0, 0.001, 0.0021, 0.003):
        assert not _record_motion_or_terminate(task, controller, _evidence(position))
    assert not _record_motion_or_terminate(
        task, controller, _evidence(0.003), reset=True
    )
    assert controller.motion_detector.has_observation
    assert not hasattr(task, "_graph_termination_reason")


def main():
    test_action_chunk_is_truncated_to_remaining_episode_steps()
    test_missing_step_budget_preserves_actions()
    test_motion_stall_terminates_after_horizon_without_significant_motion()
    test_significant_motion_and_replan_reset_prevent_stall()
    print("4 deploy-policy checks passed")


if __name__ == "__main__":
    main()
