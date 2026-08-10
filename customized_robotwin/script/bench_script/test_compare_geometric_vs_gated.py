#!/usr/bin/env python3
"""CPU checks for compare_geometric_vs_gated.py's analysis and artifact path."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compare_geometric_vs_gated import (
    AlignmentError,
    SceneNotAlignableError,
    _read_records,
    _restore_resume_config,
    _ring_config,
    _scene_record,
    aligned_widest_pair,
    build_aligned_comparison,
    false_keep_overlap,
    save_route_overlay,
    summarize_records,
    write_reports,
)
from lib.geometric_metric import _GeometricVolume
from lib.labeling import BEYOND, FREE
from lib.metric_config import SeedMetricConfig


def _record(seed, gated, geom, gated_merged=True, geom_merged=True, hits=0, route=10):
    return {
        "status": "ok",
        "seed": seed,
        "merged_gated": gated_merged,
        "merged_geom": geom_merged,
        "eps_gated": float(gated),
        "eps_geom": float(geom),
        "merged_gated_notarget_common": gated_merged,
        "merged_geom_notarget_common": geom_merged,
        "eps_gated_notarget_common": float(gated),
        "eps_geom_notarget_common": float(geom),
        "geom_route_voxels": route,
        "false_keep_route_voxels": hits,
    }


def test_summary_and_reports():
    records = [_record(i, i + 1, 2 * (i + 1), hits=i, route=20) for i in range(10)]
    records[3]["merged_gated"] = False
    records[3]["merged_gated_notarget_common"] = False
    records[3]["eps_gated"] = 0.0
    records[3]["eps_gated_notarget_common"] = 0.0
    summary = summarize_records(records)
    assert summary["completed_scenes"] == 10
    assert summary["sufficient_for_rank_gate"]
    assert summary["spearman_all"] > 0.8
    assert summary["rank_gate_pass"]
    assert summary["production_target_masked"]["spearman_all"] > 0.8
    assert summary["isolation_no_target"]["spearman_all"] > 0.8
    assert summary["gated_inaccessible_geom_merged_seeds"] == [3]
    assert summary["false_keep_route_voxels"] == sum(range(10))
    assert summary["route_overlay_gate"] == "PENDING_USER_INSPECTION"

    incomplete = records + [
        {"status": "not_alignable", "seed": 10, "reason": "no warm field"},
        {"status": "error", "seed": 11, "error": "RuntimeError: failed"},
    ]
    incomplete_summary = summarize_records(incomplete)
    assert incomplete_summary["not_alignable_scenes"] == 1
    assert incomplete_summary["not_alignable_seeds"] == [10]
    assert incomplete_summary["error_scenes"] == 1
    assert incomplete_summary["error_seeds"] == [11]

    with tempfile.TemporaryDirectory() as tmp:
        written = write_reports(Path(tmp), records, {"arm": "right"})
        assert written["rank_gate_pass"]
        assert (Path(tmp) / "summary.json").is_file()
        assert (Path(tmp) / "eps_geom_vs_gated_scatter.png").is_file()


def test_aligned_relaxation_invariant_and_preconditions():
    edt = np.array([[[5.0, 1.0, 5.0], [5.0, 5.0, 5.0]]])
    gated = np.full(edt.shape, BEYOND, dtype=np.int8)
    gated[0, 0, :] = FREE
    geom = np.full(edt.shape, FREE, dtype=np.int8)
    qvol = np.zeros(edt.shape + (1,), dtype=float)
    result = aligned_widest_pair(
        gated, geom, edt, edt.copy(), qvol,
        (0, 0, 0), (0, 0, 2), 0.35,
    )
    assert result["gated"]["eps"] == 1.0
    assert result["geom"]["eps"] == 5.0

    stricter_geom = geom.copy()
    stricter_geom[0, 0, 1] = BEYOND
    try:
        aligned_widest_pair(
            geom, stricter_geom, edt, edt, qvol,
            (0, 0, 0), (0, 0, 2), 0.35,
        )
    except ValueError as exc:
        assert "not a superset" in str(exc)
    else:
        raise AssertionError("FREE-set violation was not rejected")

    changed_edt = edt.copy()
    changed_edt[0, 0, 1] += 0.01
    try:
        aligned_widest_pair(
            gated, geom, edt, changed_edt, qvol,
            (0, 0, 0), (0, 0, 2), 0.35,
        )
    except ValueError as exc:
        assert "EDTs differ" in str(exc)
    else:
        raise AssertionError("EDT mismatch was not rejected")


def test_full_aligned_comparison_prefers_climb():
    xs = np.arange(5, dtype=float) * 0.1
    ys = np.array([0.0])
    zs = np.array([0.8, 0.9])
    XX, YY = np.meshgrid(xs, ys)
    label = np.full((2, 1, 5), FREE, dtype=np.int8)
    edt = np.ones(label.shape, dtype=float)
    edt[1] = 10.0
    zeros = np.zeros(label.shape, dtype=bool)
    volume = _GeometricVolume(
        xs, ys, zs, XX, YY, label.copy(), edt.copy(),
        prune_mask=zeros, occ_mask=zeros, target_mask=zeros,
    )
    gated = SimpleNamespace(
        XX=XX, YY=YY, zs=zs, label=label, edt=edt,
        q_warm_3d=np.zeros(label.shape + (1,), dtype=float),
        merged=True, eps_gated=1.0, reason=None,
        seed_start=(0, 0, 0), seed_goal=(0, 0, 4),
    )
    cfg = SeedMetricConfig(
        xmin=0.0, xmax=0.4, ymin=0.0, ymax=0.0, res=0.1,
        zmin=0.8, zmax=0.9, zres=0.1, seed_snap=0.01,
    )
    result = build_aligned_comparison(
        gated, volume, np.array([0.0, 0.0, 0.8]),
        np.array([0.4, 0.0, 0.8]), cfg,
    )
    assert result["target"]["geom"]["eps"] >= result["target"]["gated"]["eps"]
    assert result["target"]["geom"]["route_bfs_diagnostics"]["zmin_fraction"] == 1.0
    assert result["target"]["geom"]["route_diagnostics"]["z_max"] == 0.9
    overlap = (0, len(result["target"]["geom"]["route_world"]), 0.0)
    record = _scene_record(
        0, "right", gated, result, overlap,
        np.array([0.0, 0.0, 0.8]), np.array([0.4, 0.0, 0.8]), 1.0,
    )
    assert record["eps_geom"] >= record["eps_gated"]
    json.dumps(record, allow_nan=False)


def test_expected_gated_early_exit_is_not_a_fatal_alignment_error():
    grid = np.zeros((1, 1), dtype=float)
    label = np.full((1, 1, 1), FREE, dtype=np.int8)
    gated = SimpleNamespace(
        XX=grid, YY=grid, zs=np.array([0.8]), label=label,
        edt=None, q_warm_3d=None,
        reason="no converged warm branch on any FREE voxel",
    )
    zeros = np.zeros(label.shape, dtype=bool)
    volume = _GeometricVolume(
        np.array([0.0]), np.array([0.0]), np.array([0.8]), grid, grid,
        label.copy(), np.ones(label.shape),
        prune_mask=zeros, occ_mask=zeros, target_mask=zeros,
    )
    try:
        build_aligned_comparison(
            gated, volume, np.array([0.0, 0.0, 0.8]),
            np.array([0.0, 0.0, 0.8]), SeedMetricConfig(),
        )
    except SceneNotAlignableError as exc:
        assert "no converged warm branch" in str(exc)
        assert not isinstance(exc, AlignmentError)
    else:
        raise AssertionError("expected gated early exit was not rejected")


def test_resume_restores_config_and_skips_existing_records():
    config = {
        "seeds": [0, 1, 2],
        "arm": "right",
        "base_config": "bench_demo_office_clean",
        "offset": 0.2,
        "num_occluders": 4,
        "occluder_angle0": 0.0,
        "reach_cache_dir": "/tmp/reach",
        "reach_mode": "occupancy",
        "false_keep_mask": "/tmp/false_keep_mask.npz",
        "metric": {
            "res": 0.01,
            "gate_tau_sweep": [0.5, 1.0],
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        records = [_record(0, 0.05, 0.06)]
        (root / "records.jsonl").write_text(
            json.dumps(records[0]) + "\n", encoding="utf-8"
        )
        args = SimpleNamespace(resume_dir=str(root))
        out_dir, restored, loaded = _restore_resume_config(args)
        assert out_dir == root.resolve()
        assert restored == config
        assert loaded == records
        assert args.seeds == [0, 1, 2]
        assert args.gate_tau_sweep == [0.5, 1.0]
        assert _read_records(root / "records.jsonl") == records


def test_diverse_ring_config_is_deterministic_and_in_range():
    args = SimpleNamespace(
        offsets="0.1-0.25",
        offset=0.2,
        num_occluders="2,3,4,5",
        random_ring_rotation=True,
        occluder_angle0=0.0,
    )
    draws = [_ring_config(args, seed) for seed in range(100)]
    assert draws == [_ring_config(args, seed) for seed in range(100)]
    assert {draw["count"] for draw in draws} == {2, 3, 4, 5}
    assert all(
        len(draw["radii_m"]) == draw["count"]
        and all(0.1 <= radius <= 0.25 for radius in draw["radii_m"])
        for draw in draws
    )
    assert len({draw["angle0_rad"] for draw in draws}) > 1


def test_false_keep_overlap():
    data = {
        "mask": np.zeros((2, 2, 3), dtype=bool),
        "xs": np.array([0.0, 0.1, 0.2]),
        "ys": np.array([0.0, 0.1]),
        "zs": np.array([0.8, 0.9]),
    }
    data["mask"][0, 0, 1] = True
    hits, total, fraction = false_keep_overlap(
        [(0.0, 0.0, 0.8), (0.1, 0.0, 0.8), (0.2, 0.1, 0.9)], data
    )
    assert (hits, total, fraction) == (1, 3, 1 / 3)


def test_route_overlay():
    XX, YY = np.meshgrid(np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1]))
    label = np.full((2, 2, 3), 2, dtype=np.int8)
    gated = SimpleNamespace(
        start_xyz=np.array([0.0, 0.0, 0.8]),
        goal_xyz=np.array([0.2, 0.1, 0.8]),
        XX=XX,
        YY=YY,
        zs=np.array([0.8, 0.9]),
        label=label,
        route_world=[(0.0, 0.0, 0.8), (0.1, 0.0, 0.9), (0.2, 0.1, 0.8)],
        eps_gated=0.05,
        merged=True,
    )
    geom = SimpleNamespace(
        start_xyz=np.array([0.0, 0.0, 0.8]),
        goal_xyz=np.array([0.2, 0.1, 0.8]),
        route_world=[(0.0, 0.0, 0.8), (0.1, 0.1, 0.9), (0.2, 0.1, 0.8)],
        eps_star=0.07,
        merged=True,
    )
    args = SimpleNamespace(
        seed=2, arm="right", res=0.1, occ_shape="mesh"
    )
    with tempfile.TemporaryDirectory() as tmp:
        save_route_overlay(
            Path(tmp), args, gated, geom, [], np.array([0.0, 0.0, 0.8])
        )
        assert (Path(tmp) / "route_overlay_seed2_right.png").is_file()


if __name__ == "__main__":
    test_summary_and_reports()
    test_aligned_relaxation_invariant_and_preconditions()
    test_full_aligned_comparison_prefers_climb()
    test_expected_gated_early_exit_is_not_a_fatal_alignment_error()
    test_resume_restores_config_and_skips_existing_records()
    test_diverse_ring_config_is_deterministic_and_in_range()
    test_false_keep_overlap()
    test_route_overlay()
    print("compare geometric-vs-gated tests: PASS")
