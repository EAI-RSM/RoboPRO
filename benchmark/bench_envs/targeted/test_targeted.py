"""Unit tests for the pure PoC modules (sampler, labels). No sim imports.

Run:  python benchmark/bench_envs/targeted/test_targeted.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # flat import: skip the sim-heavy bench_envs __init__
import labels  # noqa: E402
import sampler  # noqa: E402


def test_sampler_deterministic_and_isolated():
    a = sampler.sample_shift_params("shift_object", 7, 3)
    np.random.seed(123)  # touching the global stream must not change draws
    np.random.rand(100)
    b = sampler.sample_shift_params("shift_object", 7, 3)
    assert a == b, "sampler must be deterministic and isolated from global np.random"


def test_sampler_grid_coverage():
    classes, bins = set(), set()
    for i in range(15):
        p = sampler.sample_shift_params("shift_target", 0, i)
        classes.add(p["target_class"])
        bins.add(p["magnitude_bin_index"])
        lo, hi = p["magnitude_bin_cm"]
        assert lo <= p["magnitude_cm"] < hi
        assert p["sign"] in (-1, 1)
    assert classes == set(sampler.PERCEPTUAL_CLASSES)
    assert bins == set(range(len(sampler.MAGNITUDE_BINS_CM)))


def test_construct_world_shift_by_class():
    depth = np.array([0.0, 1.0, 0.0])
    lateral = np.cross([0.0, 0.0, 1.0], depth)
    base = {"magnitude_cm": 4.0, "sign": 1, "mixed_angle_deg": 40.0}

    s = sampler.construct_world_shift({**base, "target_class": "depth"}, depth, lateral)
    assert np.allclose(s, [0.0, 0.04, 0.0])

    s = sampler.construct_world_shift({**base, "target_class": "lateral", "sign": -1}, depth, lateral)
    assert np.allclose(s, [0.04, 0.0, 0.0])

    s = sampler.construct_world_shift({**base, "target_class": "mixed"}, depth, lateral)
    assert abs(np.linalg.norm(s) - 0.04) < 1e-9 and s[2] == 0.0
    d = labels.camera_frame_decomposition(s, depth, lateral)
    assert d["posthoc_dominant_axis"] == "mixed", "40-degree off-axis shift must read as mixed"


def test_decomposition_consistency():
    depth = np.array([0.0, 1.0, 0.0])
    lateral = np.cross([0.0, 0.0, 1.0], depth)
    d = labels.camera_frame_decomposition([0.0, 0.05, 0.0], depth, lateral)
    assert abs(d["depth_cm"] - 5.0) < 1e-9 and d["posthoc_dominant_axis"] == "depth"
    d = labels.camera_frame_decomposition([-0.03, 0.0, 0.0], depth, lateral)
    assert abs(d["lateral_cm"] - 3.0) < 1e-9 and d["posthoc_dominant_axis"] == "lateral"


def test_derive_outcome_precedence():
    f = labels.derive_outcome
    assert f(plan_success=False, success=False, is_collision=True,
             object_moved_cm=None, place_error_cm=None)[0] == "plan_aborted"
    assert f(plan_success=True, success=True, is_collision=False,
             object_moved_cm=30, place_error_cm=0.2)[0] == "clean_success"
    assert f(plan_success=True, success=True, is_collision=True,
             object_moved_cm=30, place_error_cm=0.2)[0] == "success_with_collision"
    assert f(plan_success=True, success=False, is_collision=False,
             object_moved_cm=0.5, place_error_cm=25)[0] == "empty_grasp"
    assert f(plan_success=True, success=False, is_collision=True,
             object_moved_cm=20, place_error_cm=8)[0] == "collision_with_object"
    assert f(plan_success=True, success=False, is_collision=False,
             object_moved_cm=20, place_error_cm=8)[0] == "wrong_place_pose"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
