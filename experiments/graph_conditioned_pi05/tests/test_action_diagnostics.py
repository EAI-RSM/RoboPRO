from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.graph_conditioned_pi05.action_diagnostics import ActionTraceRecorder


def _record(recorder, frame, *, held=False, contact=False):
    raw = np.arange(14, dtype=np.float32)
    executed = raw.copy()
    if frame == 2:
        executed[6] = -1.0
    measured = np.zeros(14, dtype=np.float32)
    recorder.record(
        frame=frame,
        prompt="Task objective: test\nCurrent stage: pick it up",
        phase="grasp",
        raw_action=raw,
        executed_action=executed,
        observation={"joint_action": {"vector": measured}},
        evidence={
            "target_left_distance": 0.2 / frame,
            "target_right_distance": 0.3,
            "target_left_contact": contact,
            "target_right_contact": False,
            "held_by_left": held,
            "held_by_right": False,
            "held_arm": "left" if held else "",
        },
    )


def test_summary_preserves_control_and_evidence_events():
    recorder = ActionTraceRecorder()
    _record(recorder, 1)
    _record(recorder, 2, contact=True, held=True)
    summary = recorder.summary()
    assert summary["frame_count"] == 2
    assert summary["min_target_left_distance"] == 0.1
    assert summary["left_first_contact_frame"] == 2
    assert summary["left_first_held_frame"] == 2
    assert summary["left_gripper_override_count"] == 1
    assert summary["right_gripper_override_count"] == 0


def test_npz_round_trip_keeps_actions_and_text():
    recorder = ActionTraceRecorder()
    _record(recorder, 1)
    with TemporaryDirectory() as directory:
        path = recorder.save_npz(Path(directory) / "trace.npz")
        with np.load(path) as trace:
            assert trace["raw_action"].shape == (1, 14)
            assert trace["executed_action"].shape == (1, 14)
            assert trace["prompt"][0].startswith("Task objective:")
            assert trace["phase"].tolist() == ["grasp"]


def test_empty_trace_has_a_safe_summary_and_archive():
    recorder = ActionTraceRecorder()
    assert recorder.summary() == {"frame_count": 0}
    with TemporaryDirectory() as directory:
        path = recorder.save_npz(Path(directory) / "empty.npz")
        with np.load(path) as trace:
            assert trace.files == []


def main():
    test_summary_preserves_control_and_evidence_events()
    test_npz_round_trip_keeps_actions_and_text()
    test_empty_trace_has_a_safe_summary_and_archive()
    print("3 action-diagnostics checks passed")


if __name__ == "__main__":
    main()
