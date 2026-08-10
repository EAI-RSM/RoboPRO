#!/usr/bin/env python3
"""CPU checks for crash-safe VLA records, resume, and periodic reports."""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import vla_rollout as rollout
from lib.run_io import atomic_write_json
from lib.vla_reporting import read_episode_records, write_rollout_reports


def _args(**overrides):
    values = {
        "scene": "task",
        "task_name": "put_cup_on_coaster",
        "bench_subdir": "study",
        "base_config": "bench_demo_study_clean",
        "seed_start": 3000,
        "num_seeds": 10,
        "rollouts_per_density": 1000,
        "replicate": 0,
        "offsets": "0.2",
        "num_occluders": "1",
        "random_ring_rotation": False,
        "pad_shift_y": 0.0,
        "clutter_densities": "6,10,15",
        "no_occluder_prob": 0.0,
        "instruction": None,
        "max_steps": 600,
        "pi0_step": 50,
        "report_every": 10,
        "resume_dir": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _record(episode, density, config_hash, *, hard_success):
    collision = not hard_success and episode % 2 == 0
    return {
        "schema": "robopro.vla-rollout.v3",
        "episode": episode,
        "record_id": f"rollout-{config_hash[:12]}-{episode:06d}",
        "run_config_sha256": config_hash,
        "seed": 3000 + episode,
        "replicate": 0,
        "task": "put_cup_on_coaster",
        "bench_subdir": "study",
        "base_config": "bench_demo_study_clean",
        "dr_settings": {
            "cluttered_table": True,
            "obstacle_density": density,
            "clean_background_rate": 0,
        },
        "clutter_density": density,
        "clutter_count": density,
        "scene_id": f"scene-{episode}",
        "scene_fingerprint": f"fingerprint-{episode}",
        "acting_arm": "left" if episode % 2 else "right",
        "task_success": bool(hard_success or collision),
        "hard_success": bool(hard_success),
        "steps_taken": 20 + episode,
        "step_lim": 600,
        "wall_seconds": 3.0 + episode,
        "timing_seconds": {"scene_setup": 0.5, "policy": 2.0 + episode},
        "video_relpath": (
            f"hard_success/video/episode{episode}.mp4"
            if hard_success else f"hard_fail/video/episode{episode}.mp4"
        ),
        "video_camera": "countertop_camera",
        "policy_error": None,
        "failure_reason": None if hard_success else "step_limit_reached",
        "instruction": "Place the cup onto the coaster on the table.",
        "checkpoint": rollout.CHECKPOINT,
        "scene_code_version": "scene-code",
        "collision_metrics": {
            "is_collision": collision,
            "robot_to_furniture": int(collision),
            "robot_to_static_object": 0,
            "target_to_static_object": 0,
            "total_collision_count": int(collision),
        },
    }


def test_round_robin_target():
    args = _args()
    densities = rollout._clutter_densities(args.clutter_densities)
    assert rollout._target_rollouts(args, densities) == 3000
    assert [rollout._task_density(densities, index) for index in range(7)] == [
        6, 10, 15, 6, 10, 15, 6,
    ]
    print("  [1] 1000/density yields 3000 rollouts in d6,d10,d15 order       PASS")


def test_video_uses_hard_outcome_bucket():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        source = run_dir / "video" / "episode0.mp4"
        source.parent.mkdir()
        source.write_bytes(b"complete-video")
        relative = rollout._bucket_video(run_dir, 0, hard_success=False)
        assert relative == "hard_fail/video/episode0.mp4"
        assert (run_dir / relative).read_bytes() == b"complete-video"
    print("  [2] finalized video is durable and grouped by hard outcome          PASS")


def test_atomic_records_resume_and_reports():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "episodes").mkdir()
        args = _args(resume_dir=str(run_dir))
        densities = rollout._clutter_densities(args.clutter_densities)
        config = rollout._run_config(args, densities, "scene-code")
        atomic_write_json(run_dir / "config.json", config)

        records = [
            _record(index, density, config["config_sha256"], hard_success=index % 2 == 0)
            for index, density in enumerate((6, 10, 15, 6))
        ]
        # Model a power loss after the fourth atomic episode file but before its
        # JSONL append. Resume must recover it from episodes/, then rebuild JSONL.
        for record in records[:3]:
            rollout._commit_record(run_dir, record)
        atomic_write_json(run_dir / "episodes" / "episode000003.json", records[3])

        restored_config, restored = rollout._restore_run(args, run_dir)
        assert restored_config == config
        assert restored == records
        jsonl = [
            json.loads(line)
            for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert jsonl == records

        summary = write_rollout_reports(run_dir)
        assert summary["n_episodes"] == 4
        assert summary["hard_success_rate"] == 0.5
        assert summary["records_by_density"] == {"6": 2, "10": 1, "15": 1}
        for filename in (
            "records.csv",
            "metric_scene_manifest.jsonl",
            "summary.json",
            "running_hsr.png",
            "outcome_diagnostics.png",
            "report_state.json",
        ):
            path = run_dir / filename
            assert path.is_file() and path.stat().st_size > 0
        assert read_episode_records(run_dir) == records
    print("  [3] atomic episode recovery, JSONL reconciliation, resume, reports PASS")


def main():
    print("VLA long-run durability/reporting -- CPU checks")
    test_round_robin_target()
    test_video_uses_hard_outcome_bucket()
    test_atomic_records_resume_and_reports()
    print("ALL PASS")


if __name__ == "__main__":
    main()
