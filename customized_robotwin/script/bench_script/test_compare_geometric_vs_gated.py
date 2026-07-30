#!/usr/bin/env python3
"""CPU checks for compare_geometric_vs_gated.py's analysis and artifact path."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compare_geometric_vs_gated import (
    false_keep_overlap,
    save_route_overlay,
    summarize_records,
    write_reports,
)


def _record(seed, gated, geom, gated_merged=True, geom_merged=True, hits=0, route=10):
    return {
        "status": "ok",
        "seed": seed,
        "merged_gated": gated_merged,
        "merged_geom": geom_merged,
        "eps_gated": float(gated),
        "eps_geom": float(geom),
        "geom_route_voxels": route,
        "false_keep_route_voxels": hits,
    }


def test_summary_and_reports():
    records = [_record(i, i + 1, 2 * (i + 1), hits=i, route=20) for i in range(10)]
    records[3]["merged_gated"] = False
    records[3]["eps_gated"] = 0.0
    summary = summarize_records(records)
    assert summary["completed_scenes"] == 10
    assert summary["sufficient_for_rank_gate"]
    assert summary["spearman_all"] > 0.8
    assert summary["rank_gate_pass"]
    assert summary["gated_inaccessible_geom_merged_seeds"] == [3]
    assert summary["false_keep_route_voxels"] == sum(range(10))
    assert summary["route_overlay_gate"] == "PENDING_USER_INSPECTION"

    with tempfile.TemporaryDirectory() as tmp:
        written = write_reports(Path(tmp), records, {"arm": "right"})
        assert written["rank_gate_pass"]
        assert (Path(tmp) / "summary.json").is_file()
        assert (Path(tmp) / "eps_geom_vs_gated_scatter.png").is_file()


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
    test_false_keep_overlap()
    test_route_overlay()
    print("compare geometric-vs-gated tests: PASS")
