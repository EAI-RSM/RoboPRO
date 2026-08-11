#!/usr/bin/env python3
"""CPU checks for strict task-metric/outcome joining and association artifacts."""

import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

from analyze_metric_correlation import (
    DEFAULT_BUCKET_SPEC,
    index_videos,
    join_records,
    run,
    summarize,
    write_correlation_reports,
)
from lib.run_io import atomic_write_json
from lib.metric_buckets import load_bucket_spec
from lib.scene_provenance import fingerprint
from lib.vla_reporting import metric_scene_manifest, sync_records_jsonl
from task_metric import (
    _commit_metric_record,
    _read_committed_metric_records,
    _read_scene_manifest,
    _regenerate_postprocess_reports,
    _validate_stored_config,
)


def _rollout(episode, *, eps_id=None, success=False, collision=False):
    scene_id = eps_id or f"scene-{episode}"
    return {
        "schema": "robopro.vla-rollout.v3",
        "episode": episode,
        "seed": 3000 + episode,
        "replicate": 0,
        "task": "put_cup_on_coaster",
        "bench_subdir": "study",
        "base_config": "bench_demo_study_clean",
        "dr_settings": {
            "cluttered_table": True,
            "obstacle_density": (6, 10, 15)[episode % 3],
            "clean_background_rate": 0,
        },
        "clutter_density": (6, 10, 15)[episode % 3],
        "clutter_count": (6, 10, 15)[episode % 3],
        "scene_id": scene_id,
        "scene_fingerprint": f"fingerprint-{episode}",
        "scene_code_version": "scene-code",
        "checkpoint": "mzxuan/robopro_jax_30000",
        "instruction": "Place the cup onto the coaster on the table.",
        "acting_arm": "right" if episode % 2 == 0 else "left",
        "task_success": bool(success),
        "hard_success": bool(success and not collision),
        "steps_taken": 130 if success else 600,
        "video_relpath": (
            f"hard_success/video/episode{episode}.mp4"
            if success and not collision else f"hard_fail/video/episode{episode}.mp4"
        ),
        "collision_metrics": {"is_collision": bool(collision)},
    }


def _metric(rollout, eps):
    kinds = ("pre_grasp", "grasp", "carry", "place")
    return {
        "schema": "robopro.task-metric.v1",
        "status": "ok",
        "rollout_episode": rollout["episode"],
        "scene_id": rollout["scene_id"],
        "scene_fingerprint": rollout["scene_fingerprint"],
        "task": rollout["task"],
        "seed": rollout["seed"],
        "replicate": rollout["replicate"],
        "bench_subdir": rollout["bench_subdir"],
        "base_config": rollout["base_config"],
        "obstacle_density": rollout["clutter_density"],
        "clutter_count": rollout["clutter_count"],
        "checkpoint": rollout["checkpoint"],
        "scene_code_version": rollout["scene_code_version"],
        "instruction": rollout["instruction"],
        "arm": rollout["acting_arm"],
        "gripper_reference_radius_m": 0.03,
        "eps_geom_min": eps,
        "eps_geom_min_unbounded": False,
        "legs": [
            {
                "index": index,
                "kind": kind,
                "eps_geom": eps + index * 0.01,
                "eps_geom_unbounded": False,
            }
            for index, kind in enumerate(kinds)
        ],
    }


def test_manifest_is_outcome_blind():
    rows = [_rollout(0, success=True), _rollout(1, success=False)]
    manifest = metric_scene_manifest(rows)
    assert len(manifest) == 2
    forbidden = {"task_success", "hard_success", "collision_metrics", "video_relpath"}
    assert all(not (forbidden & set(row)) for row in manifest)
    assert manifest[0]["expected_scene_fingerprint"] == "fingerprint-0"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "manifest.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
        )
        resolved, jobs = _read_scene_manifest(path)
        assert resolved == path.resolve() and len(jobs) == 2
        bad = dict(manifest[0], hard_success=True)
        path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        try:
            _read_scene_manifest(path)
        except ValueError as exc:
            assert "outcome-blind schema" in str(exc)
        else:
            raise AssertionError("task_metric accepted outcome data in its scene manifest")


def test_strict_join_and_statistics():
    spec = load_bucket_spec(DEFAULT_BUCKET_SPEC)
    rollouts = [
        _rollout(0, success=False),
        _rollout(1, success=False),
        _rollout(2, success=True),
        _rollout(3, success=True),
    ]
    metrics = [
        _metric(rollouts[0], 0.05),
        _metric(rollouts[1], 0.09),
        _metric(rollouts[2], 0.13),
        _metric(rollouts[3], 0.17),
    ]
    joined = join_records(rollouts, metrics, spec)
    assert [row["clearance_bucket"] for row in joined] == [
        "very_low_clearance", "low_clearance", "medium_clearance", "high_clearance"
    ]
    summary = summarize(joined, spec, resamples=200, seed=7)
    assert summary["analysis_hierarchy"]["primary_density_order"] == [6, 10, 15]
    assert summary["by_density"]["6"]["association"]["spearman_rho"] > 0.8
    assert summary["secondary_pooled"]["hard_success_rate"] == 0.5
    assert summary["secondary_pooled"]["association"]["spearman_rho"] > 0.8
    assert summary["secondary_pooled"]["association"]["valid_bootstrap_resamples"] > 0

    partial = join_records(rollouts, metrics[:2], spec, require_complete=False)
    provisional = summarize(
        partial,
        spec,
        resamples=0,
        seed=7,
        require_complete=False,
        target_n=len(rollouts),
    )
    assert provisional["provisional"] is True
    assert provisional["processing_complete"] is False
    assert provisional["secondary_pooled"]["association"]["deferred_until_complete"] is True

    try:
        join_records(rollouts, metrics[:-1], spec)
    except ValueError as exc:
        assert "not one-to-one" in str(exc)
    else:
        raise AssertionError("missing metric record was accepted")

    mismatched = [dict(row) for row in metrics]
    mismatched[0]["scene_fingerprint"] = "wrong"
    try:
        join_records(rollouts, mismatched, spec)
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("fingerprint mismatch was accepted")

    duplicated = metrics + [dict(metrics[0])]
    try:
        join_records(rollouts, duplicated, spec)
    except ValueError as exc:
        assert "duplicate metric scene_id" in str(exc)
    else:
        raise AssertionError("duplicate metric record was accepted")


def test_non_destructive_video_index():
    spec = load_bucket_spec(DEFAULT_BUCKET_SPEC)
    rollout = _rollout(0, success=True)
    joined = join_records([rollout], [_metric(rollout, 0.05)], spec)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rollout_dir = root / "rollout"
        source = rollout_dir / rollout["video_relpath"]
        source.parent.mkdir(parents=True)
        source.write_bytes(b"video")
        out_dir = root / "analysis"
        out_dir.mkdir()
        index = index_videos(out_dir, rollout_dir, joined)
        link = out_dir / index[0]["index_link"]
        assert link.is_symlink() and link.read_bytes() == b"video"
        assert source.read_bytes() == b"video"
        assert json.loads((out_dir / "video_index.json").read_text())[0]["video_available"]
        # Regeneration is idempotent; existing correct symlinks are reused.
        second = index_videos(out_dir, rollout_dir, joined)
        assert second == index


def test_metric_commit_repair_and_provisional_reporting():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        metric_dir = root / "metric_postprocess"
        rollout_dir = root / "rollout"
        (rollout_dir / "episodes").mkdir(parents=True)
        config = {
            "schema": "robopro.task-metric-postprocess-config.v1",
            "value": 1,
            "metric": {"gate_tau_sweep": [0.5, 1.0]},
        }
        config["config_sha256"] = fingerprint(config)
        atomic_write_json(metric_dir / "config.json", config)
        assert _validate_stored_config(metric_dir / "config.json", config) == config
        changed = {
            "schema": config["schema"],
            "value": 2,
            "metric": config["metric"],
        }
        changed["config_sha256"] = fingerprint(changed)
        try:
            _validate_stored_config(metric_dir / "config.json", changed)
        except ValueError as exc:
            assert "immutable config" in str(exc)
        else:
            raise AssertionError("metric resume accepted changed immutable settings")
        jobs = []
        metrics = []
        for episode in range(12):
            # The first report deliberately contains an all-failure bucket.
            # This reproduces the n=10 live smoke where Wilson roundoff at p=0
            # previously produced a negative Matplotlib error-bar length.
            rollout = _rollout(episode, success=episode in {3, 4, 5, 7})
            atomic_write_json(
                rollout_dir / "episodes" / f"episode{episode:06d}.json", rollout
            )
            video = rollout_dir / rollout["video_relpath"]
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
            jobs.append(
                {
                    "expected_scene_id": rollout["scene_id"],
                    "expected_scene_fingerprint": rollout["scene_fingerprint"],
                }
            )
            metric = _metric(rollout, 0.05 + episode * 0.01)
            metric["metric_sequence"] = episode
            metric["metric_run_config_sha256"] = "config-hash"
            metrics.append(metric)

        for metric in metrics[:10]:
            _commit_metric_record(metric_dir, metric)
        # Simulate power loss after a partial JSONL append. Atomic episode files
        # remain authoritative and repair the convenience stream on resume.
        with (metric_dir / "records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write('{"partial"')
        repaired = _read_committed_metric_records(metric_dir, jobs, "config-hash")
        sync_records_jsonl(metric_dir, repaired)
        assert len(repaired) == 10
        assert len((metric_dir / "records.jsonl").read_text().splitlines()) == 10

        integrated = _regenerate_postprocess_reports(
            SimpleNamespace(
                rollout_run=str(rollout_dir),
                bucket_spec=str(DEFAULT_BUCKET_SPEC),
                bootstrap_resamples=100,
                bootstrap_seed=3,
            ),
            metric_dir,
            repaired,
            complete=False,
            raise_errors=True,
        )
        assert integrated["n_joined"] == 10 and integrated["provisional"] is True
        for filename in (
            "eps_geom_min_distribution.png",
            "eps_geom_by_leg.png",
            "eps_geom_min_by_scene.png",
            "hsr_by_clearance_bucket.png",
            "eps_geom_by_outcome.png",
            "report_index.json",
            "video_index.json",
        ):
            assert (metric_dir / filename).is_file()
        assert [
            row["clutter_density"]
            for row in json.loads((metric_dir / "report_index.json").read_text())[
                "primary_reports"
            ]
        ] == [6, 10, 15]
        for density in (6, 10, 15):
            density_dir = metric_dir / "by_density" / f"d{density}"
            for filename in (
                "summary.json", "distribution_summary.json", "source_records.json",
                "joined_records.jsonl", "joined_records.csv",
                "eps_geom_min_distribution.png", "eps_geom_by_leg.png",
                "eps_geom_min_by_scene.png",
                "hsr_by_clearance_bucket.png", "eps_geom_by_outcome.png",
            ):
                assert (density_dir / filename).is_file()

        out_dir = metric_dir / "analysis"
        summary = write_correlation_reports(
            out_dir,
            rollout_dir,
            [metric_dir / "records.jsonl"],
            DEFAULT_BUCKET_SPEC,
            require_complete=False,
            bootstrap_resamples=0,
            bootstrap_seed=3,
        )
        assert summary["n_joined"] == 10 and summary["target_n"] == 12
        assert summary["provisional"] is True
        assert summary["video_index"] == {
            "available": 10,
            "missing": 0,
            "organization": "clearance_bucket / hard_outcome",
            "non_destructive_symlinks": True,
        }
        # A second every-10 refresh must reuse links without failing.
        repeated = write_correlation_reports(
            out_dir,
            rollout_dir,
            [metric_dir / "records.jsonl"],
            DEFAULT_BUCKET_SPEC,
            require_complete=False,
            bootstrap_resamples=0,
            bootstrap_seed=3,
        )
        assert repeated["n_joined"] == 10


def test_end_to_end_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rollout_dir = root / "rollout"
        episodes_dir = rollout_dir / "episodes"
        episodes_dir.mkdir(parents=True)
        metrics = []
        for episode, success, eps in ((0, False, 0.05), (1, True, 0.16)):
            rollout = _rollout(episode, success=success)
            atomic_write_json(episodes_dir / f"episode{episode:06d}.json", rollout)
            video = rollout_dir / rollout["video_relpath"]
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
            metrics.append(_metric(rollout, eps))
        metric_path = root / "metrics.jsonl"
        metric_path.write_text(
            "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
        )
        out = run(
            SimpleNamespace(
                rollout_run=str(rollout_dir),
                metric_records=[str(metric_path)],
                bucket_spec=str(DEFAULT_BUCKET_SPEC),
                bootstrap_resamples=100,
                bootstrap_seed=3,
                out_dir=str(root / "analysis"),
            )
        )
        expected = {
            "joined_records.jsonl", "joined_records.csv", "association_summary.json",
            "hsr_by_clearance_bucket.png", "eps_geom_by_outcome.png",
            "report_index.json", "video_index.json", "timings.json",
        }
        assert expected <= {path.name for path in out.iterdir()}
        assert (out / "by_density" / "d6" / "summary.json").is_file()
        assert (out / "by_density" / "d10" / "summary.json").is_file()


def main():
    test_manifest_is_outcome_blind()
    test_strict_join_and_statistics()
    test_non_destructive_video_index()
    test_metric_commit_repair_and_provisional_reporting()
    test_end_to_end_artifacts()
    print("task metric correlation tests: PASS")


if __name__ == "__main__":
    main()
