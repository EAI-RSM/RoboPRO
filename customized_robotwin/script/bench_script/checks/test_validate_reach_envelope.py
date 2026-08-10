#!/usr/bin/env python3
"""CPU checks for validate_reach_envelope.py false-keep accounting and figure output."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lib.labeling import BEYOND, FREE, OBSTACLE
from validate_reach_envelope import false_keep_summary, plot_false_keeps, save_false_keep_mask


def test_false_keep_summary():
    zs = np.array([0.8, 0.9])
    prune = np.array([
        [[True, False], [False, False]],
        [[False, True], [False, False]],
    ])
    label = np.array([
        [[BEYOND, BEYOND], [OBSTACLE, BEYOND]],
        [[FREE, BEYOND], [BEYOND, FREE]],
    ])

    mask, result = false_keep_summary(label, prune, zs)
    expected = np.array([
        [[False, True], [False, True]],
        [[False, False], [True, False]],
    ])
    assert np.array_equal(mask, expected), mask
    assert result["false_keep_count"] == 3
    assert result["kept_cells_solved"] == 6
    assert result["voxels_total"] == 8
    assert result["false_keep_fraction_of_kept"] == 0.5
    assert result["false_keep_fraction_of_grid"] == 3 / 8
    assert [row["false_keep_cells"] for row in result["per_z"]] == [2, 1]
    empty_slice_prune = prune.copy()
    empty_slice_prune[1] = True
    _, with_empty_slice = false_keep_summary(label, empty_slice_prune, zs)
    assert with_empty_slice["per_z"][1]["false_keep_fraction"] is None
    json.dumps(with_empty_slice, allow_nan=False)
    print("  [1] false-keep counts exclude unsolved pruned cells                  PASS")


def test_false_keep_figure():
    xs = np.array([-0.1, 0.1])
    ys = np.array([-0.2, 0.2])
    zs = np.array([0.8, 0.9])
    prune = np.zeros((2, 2, 2), dtype=bool)
    label = np.full((2, 2, 2), FREE, dtype=np.int8)
    label[0, 0, 0] = BEYOND
    mask, result = false_keep_summary(label, prune, zs)
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        cache_dir.mkdir()
        np.savez(cache_dir / "reach_envelope_right.npz", base_world=np.array([0.3, -0.2, 0.9]))
        args = SimpleNamespace(seed=1, arm="right", reach_mode="occupancy",
                               reach_cache_dir=str(cache_dir))
        save_false_keep_mask(Path(tmp), mask, xs, ys, zs, args.arm, args.reach_mode)
        plot_false_keeps(Path(tmp), args, xs, ys, zs, prune, mask, result)
        out = Path(tmp) / "false_keep_spatial_seed1_right.png"
        assert out.exists() and out.stat().st_size > 0
        saved = np.load(Path(tmp) / "false_keep_mask.npz")
        assert np.array_equal(saved["false_keep_mask"], mask)
        assert np.array_equal(saved["xs"], xs) and np.array_equal(saved["ys"], ys)
        assert np.array_equal(saved["zs"], zs)
        assert str(saved["arm"]) == "right" and str(saved["reach_mode"]) == "occupancy"
    print("  [2] spatial/z-profile figure + Stage-3 mask artifact are written     PASS")


def main():
    print("reach-envelope false-keep -- CPU checks")
    test_false_keep_summary()
    test_false_keep_figure()
    print("ALL PASS")


if __name__ == "__main__":
    main()
