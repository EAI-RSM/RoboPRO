#!/usr/bin/env python3
"""Focused CPU checks for per-rollout geometric-route visualization plumbing."""

import copy
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

from lib import metric_viz
import visualize_task_metric_routes as viz
from lib.geometric_metric import LegResult
from lib.metric_buckets import load_bucket_spec
from lib.run_io import atomic_write_json


SPEC_PATH = Path(__file__).with_name("bucket_spec.json")
KINDS = ("pre_grasp", "grasp", "carry", "place")


def _raises(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def _metric(values, *, episode=0):
    minimum = min(values)
    legs = []
    for index, (kind, value) in enumerate(zip(KINDS, values)):
        unbounded = math.isinf(value)
        legs.append(
            {
                "index": index,
                "kind": kind,
                "gripper_state": "empty" if index < 2 else "holding",
                "merged": True,
                "reason": None,
                "eps_geom": None if unbounded else float(value),
                "eps_geom_unbounded": unbounded,
                "start_xyz": [float(index), 0.0, 0.9],
                "goal_xyz": [float(index + 1), 0.0, 0.9],
                "bottleneck_xyz": None if unbounded else [float(index), 0.0, 0.9],
            }
        )
    return {
        "schema": "robopro.task-metric.v1",
        "status": "ok",
        "scene_id": f"scene-{episode}",
        "scene_fingerprint": f"fingerprint-{episode}",
        "scene_code_version": "scene-code",
        "rollout_episode": episode,
        "task": "put_cup_on_coaster",
        "seed": 3000 + episode,
        "replicate": 0,
        "bench_subdir": "study",
        "base_config": "bench_demo_study_clean",
        "obstacle_density": 6,
        "clutter_count": 6,
        "checkpoint": "checkpoint",
        "instruction": "instruction",
        "arm": "right",
        "n_free": 123,
        "gripper_reference_radius_m": 0.03,
        "eps_geom_min": None if math.isinf(minimum) else float(minimum),
        "eps_geom_min_unbounded": math.isinf(minimum),
        "legs": legs,
    }


def _rollout(episode=0, density=6):
    return {
        "episode": episode,
        "seed": 3000 + episode,
        "replicate": 0,
        "task": "put_cup_on_coaster",
        "bench_subdir": "study",
        "base_config": "bench_demo_study_clean",
        "scene_id": f"scene-{episode}",
        "scene_fingerprint": f"fingerprint-{episode}",
        "scene_code_version": "scene-code",
        "clutter_density": density,
        "clutter_count": density,
        "checkpoint": "checkpoint",
        "instruction": "instruction",
        "acting_arm": "right",
        "task_success": False,
        "hard_success": False,
        "collision_metrics": {"is_collision": False},
        "video_relpath": f"hard_fail/video/episode{episode}.mp4",
    }


def _job(episode=0, density=6):
    return {
        "rollout_episode": episode,
        "seed": 3000 + episode,
        "replicate": 0,
        "task": "put_cup_on_coaster",
        "bench_subdir": "study",
        "base_config": "bench_demo_study_clean",
        "expected_scene_id": f"scene-{episode}",
        "expected_scene_fingerprint": f"fingerprint-{episode}",
        "expected_scene_code_version": "scene-code",
        "obstacle_density": density,
        "expected_clutter_count": density,
        "checkpoint": "checkpoint",
        "instruction": "instruction",
        "expected_acting_arm": "right",
    }


def test_minimum_selection_and_plot_calls():
    assert viz.minimum_leg_indices(_metric([0.1, 0.2, 0.3, 0.4])) == [0]
    assert viz.minimum_leg_indices(_metric([0.2, 0.1, 0.1, 0.4])) == [1, 2]
    assert viz.minimum_leg_indices(_metric([math.inf] * 4)) == [0, 1, 2, 3]

    calls = []
    original = viz._metric_path3d

    def fake_renderer(out_dir, _args, _foots, _occ_ps, *_positional, **kwargs):
        calls.append((_positional[4], _positional[5], kwargs))
        (Path(out_dir) / f"{kwargs['stem']}.png").write_bytes(b"png")
        return None

    viz._metric_path3d = fake_renderer
    try:
        args = SimpleNamespace(seed=3000, arm="right", occ_shape="mesh", gripper_r=0.03)
        result = LegResult(
            math.inf, True, None, [(0.0, 0.0, 0.9)],
            [0.0, 0.0, 0.9], [1.0, 0.0, 0.9], 10, None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            viz._atomic_route_plot(
                Path(tmp), args, [], [], result, "pre_grasp", None, 0, 6,
                "high_clearance",
            )
        assert len(calls) == 1
        assert math.isinf(calls[0][0]) and calls[0][1] is True
        assert calls[0][2]["route_label"] == "geometric representative path"
        assert calls[0][2]["metric_label"] == "eps_geom"
        # No table measurement -> the framing arguments stay off, so the renderer keeps its
        # historical data-bounding-cube behaviour rather than framing on a surface it never saw.
        assert calls[0][2]["frame_xy"] is None and calls[0][2]["ground_z"] is None
    finally:
        viz._metric_path3d = original
    print("  [1] unique/tied/+inf minima and semantic plot call plumbing          PASS")


def test_tabletop_framing_forwarded():
    """The framing/annotation arguments reach the renderer, and only where they are meaningful."""
    calls = []
    original = viz._metric_path3d

    def fake_renderer(out_dir, _args, _foots, _occ_ps, *_positional, **kwargs):
        calls.append(kwargs)
        (Path(out_dir) / f"{kwargs['stem']}.png").write_bytes(b"png")
        return None

    viz._metric_path3d = fake_renderer
    try:
        args = SimpleNamespace(seed=3000, arm="right", occ_shape="mesh", gripper_r=0.03)
        result = LegResult(
            0.15, True, (0.0, 0.0, 0.93), [(0.0, 0.0, 0.92)],
            [0.0, 0.0, 1.02], [0.0, 0.0, 0.92], 10, None,
        )
        table_bbox = ((-0.6, -0.35, 0.0), (0.6, 0.35, 0.74))
        destination_p = [0.24, -0.21, 0.74]
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("grasp", "carry"):
                viz._atomic_route_plot(
                    Path(tmp), args, [], [], result, kind, [0.01, 0.03, 0.74], 0, 6,
                    "high_clearance", table_bbox=table_bbox, destination_p=destination_p,
                )
        assert len(calls) == 2
        for kwargs in calls:
            assert kwargs["frame_xy"] == ((-0.6, 0.6), (-0.35, 0.35))
            assert kwargs["ground_z"] == 0.74      # table TOP, not the floor-level bbox minimum
            assert kwargs["view"] == (22, -72)
            assert kwargs["dest_xyz"] == destination_p
            assert kwargs["target_label"] == "target cup (spawn)"
            assert kwargs["geom_label"] == "obstacle"
        # The tool-offset connector is a grasp-side claim only; the cup has left its spawn by carry.
        assert calls[0]["tool_link_xyz"] == result.goal_xyz
        assert calls[1]["tool_link_xyz"] is None
    finally:
        viz._metric_path3d = original
    print("  [1b] tabletop framing/annotation forwarded, grasp-side connector only PASS")


def test_renderer_sphere_guards_and_defaults():
    calls = []
    original_sphere = metric_viz._draw_eps_sphere
    original_distance = metric_viz.surface_distance_to_occluders
    metric_viz._draw_eps_sphere = lambda _ax, center, radius: calls.append((center, radius))
    metric_viz.surface_distance_to_occluders = lambda _foots, _point: 0.09
    args = SimpleNamespace(seed=7, arm="left", occ_shape="mesh", gripper_r=0.03)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            metric_viz._metric_path3d(
                out, args, [], [], [0, 0, 0.9], [1, 0, 0.9], [0.5, 0, 0.9],
                [[0, 0, 0.9], [1, 0, 0.9]], 0.1, True,
            )
            assert (out / "metric_path3d_seed7_left.png").is_file()
            metric_viz._metric_path3d(
                out, args, [], [], [0, 0, 0.9], [1, 0, 0.9], None,
                [[0, 0, 0.9]], math.inf, True, stem="unbounded",
            )
            metric_viz._metric_path3d(
                out, args, [], [], [0, 0, 0.9], [1, 0, 0.9], None,
                None, 0.0, False, stem="inaccessible",
            )
        assert len(calls) == 1 and calls[0][1] == 0.1
    finally:
        metric_viz._draw_eps_sphere = original_sphere
        metric_viz.surface_distance_to_occluders = original_distance
    print("  [2] renderer defaults remain compatible; +inf/inaccessible skip sphere PASS")


def test_regeneration_and_source_mismatch_rejection():
    committed = _metric([0.1, 0.2, 0.3, 0.4])
    assert viz.validate_regenerated_metric(committed, copy.deepcopy(committed)) == [0]
    mutations = [
        lambda row: row.update(scene_fingerprint="different"),
        lambda row: row["legs"][0].update(eps_geom=0.11),
        lambda row: row["legs"][0].update(bottleneck_xyz=[0.01, 0.0, 0.9]),
        lambda row: row["legs"][0].update(kind="wrong"),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(committed)
        mutate(changed)
        _raises(lambda changed=changed: viz.validate_regenerated_metric(committed, changed))

    rollout, job = _rollout(), _job()
    viz.validate_source_pair(job, rollout, committed)
    changed_rollout = dict(rollout, scene_id="wrong")
    _raises(lambda: viz.validate_source_pair(job, changed_rollout, committed))
    _raises(lambda: viz._hard_outcome(dict(rollout, hard_success=True)))
    print("  [3] identity, epsilon, bottleneck, leg, source, and outcome mismatches reject PASS")


def test_stratified_selection_reuses_frozen_buckets():
    spec = load_bucket_spec(SPEC_PATH)
    values = (0.05, 0.09, 0.13, 0.17)
    metrics = []
    rollouts = {}
    episode = 0
    for density in (6, 10, 15):
        for value in values:
            for _ in range(2):
                row = _metric([value, value + 0.01, value + 0.02, value + 0.03], episode=episode)
                row["obstacle_density"] = density
                row["clutter_count"] = density
                metrics.append(row)
                rollouts[episode] = _rollout(episode, density)
                episode += 1
    assert viz.select_stratified(metrics, rollouts, spec, 2) == list(range(24))
    _raises(lambda: viz.select_stratified(metrics[:16], rollouts, spec, 2))

    called = []
    original = viz.assign_metric_record
    viz.assign_metric_record = lambda row, passed: (
        called.append((row, passed)) or original(row, passed)
    )
    try:
        viz.select_stratified(metrics, rollouts, spec, 1)
    finally:
        viz.assign_metric_record = original
    assert called and all(passed is spec for _, passed in called)
    assert viz.select_first(3000, 50) == list(range(50))
    _raises(lambda: viz.select_first(20, 50))
    print("  [4] first-50 and stratified selectors reuse valid source/bucket scope PASS")


def _visual_record(episode=0):
    return {
        "schema": viz.SCHEMA,
        "status": "ok",
        "visualization_config_sha256": "config-hash",
        "source_record_sha256": {"manifest_row": "a", "rollout_record": "b", "metric_record": "c"},
        "episode": episode,
        "scene_id": f"scene-{episode}",
        "seed": 3000 + episode,
        "clutter_density": 6,
        "clearance_bucket": "low_clearance",
        "hard_outcome": "hard_fail",
        "video_relpath": f"hard_fail/video/episode{episode}.mp4",
        "figures": [
            {
                "kind": "pre_grasp",
                "is_minimum": True,
                "figure_relpath": f"figures/episode{episode:06d}_seed{3000 + episode}_pre_grasp.png",
            },
            {
                "kind": "carry",
                "is_minimum": False,
                "figure_relpath": f"figures/episode{episode:06d}_seed{3000 + episode}_carry.png",
            },
        ],
    }


def test_visual_record_resume_and_indexes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rollout = root / "rollout"
        out = root / "visuals"
        video = rollout / "hard_fail/video/episode0.mp4"
        figure = out / "figures/episode000000_seed3000_pre_grasp.png"
        carry_figure = out / "figures/episode000000_seed3000_carry.png"
        video.parent.mkdir(parents=True)
        figure.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        figure.write_bytes(b"png")
        carry_figure.write_bytes(b"png")

        # A staged/orphan figure is not a commit and is safely ignored on resume.
        assert viz.read_visual_records(out, "config-hash", 1, {}) == []
        episode_path = out / "episodes/episode000000.json"
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        record = _visual_record()
        atomic_write_json(episode_path, record)
        expected_hashes = {0: record["source_record_sha256"]}
        assert viz.read_visual_records(out, "config-hash", 1, expected_hashes) == [record]
        _raises(lambda: viz.read_visual_records(
            out, "config-hash", 1,
            {0: {"manifest_row": "changed", "rollout_record": "b", "metric_record": "c"}},
        ))

        first = viz.regenerate_indexes(out, rollout, [record], [0], 3000)
        second = viz.regenerate_indexes(out, rollout, [record], [0], 3000)
        assert first == second
        video_link = out / first[0]["video_index_link"]
        figure_link = out / first[0]["figure_index_links"][0]
        assert video_link.is_symlink() and video_link.resolve() == video.resolve()
        assert figure_link.is_symlink() and figure_link.resolve() == figure.resolve()
        report = json.loads((out / "report_state.json").read_text())
        assert report["processing_complete"] is True and report["figure_count"] == 2
        assert report["target_episodes"] == 1
        assert report["source_target_episodes"] == 3000
        # Every leg is drawn, but the by-leg tally must still name only the binding leg.
        assert report["figure_counts_by_minimum_leg"] == {"pre_grasp": 1}

        conflict_out = root / "conflict"
        conflict_link = (
            conflict_out / "by_density/d6/low_clearance/hard_fail"
            / "episode000000_seed3000_video.mp4"
        )
        conflict_link.parent.mkdir(parents=True)
        conflict_link.write_bytes(b"unrelated")
        conflict_figure = conflict_out / record["figures"][0]["figure_relpath"]
        conflict_figure.parent.mkdir(parents=True)
        conflict_figure.write_bytes(b"png")
        _raises(lambda: viz.regenerate_indexes(conflict_out, rollout, [record], [0], 3000))

        figure.unlink()
        _raises(lambda: viz.read_visual_records(out, "config-hash", 1, expected_hashes))
    print("  [5] source-bound resume and video/PNG indexes are valid and idempotent PASS")


def main():
    print("task metric per-rollout route visualization -- CPU checks")
    test_minimum_selection_and_plot_calls()
    test_tabletop_framing_forwarded()
    test_renderer_sphere_guards_and_defaults()
    test_regeneration_and_source_mismatch_rejection()
    test_stratified_selection_reuses_frozen_buckets()
    test_visual_record_resume_and_indexes()
    print("ALL PASS")


if __name__ == "__main__":
    main()
