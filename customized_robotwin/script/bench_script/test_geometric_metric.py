#!/usr/bin/env python3
"""CPU checks for the envelope-only geometric eps* implementation."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import lib.geometric_metric as gm
from lib.labeling import BEYOND, FREE
from lib.metric_config import SeedMetricConfig
from lib.obstacles import occluder_clearance_3d
from lib.scene_constants import OCC_HALF_FOOTPRINT
from lib.widest_path import (
    reconstruct_clearance_preferred_path_3d,
    reconstruct_widest_path_3d,
    widest_path_eps_3d,
)


class FakeActor:
    def __init__(self, p, q=(1, 0, 0, 0), scale=1.0):
        self._pose = type("Pose", (), {
            "p": np.asarray(p, dtype=float),
            "q": np.asarray(q, dtype=float),
        })()
        self.scale = scale

    def get_pose(self):
        return self._pose


def test_cpu_only_import():
    script = r"""
import importlib.abc
import sys

class BlockGpu(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch.") or fullname == "curobo" or fullname.startswith("curobo."):
            raise RuntimeError("GPU stack import blocked: " + fullname)
        return None

sys.meta_path.insert(0, BlockGpu())
import lib.geometric_metric
assert "torch" not in sys.modules
assert not any(name == "curobo" or name.startswith("curobo.") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).parent)
    print("  [1] fresh import succeeds with torch/curobo blocked                    PASS")


def test_target_blocks_label_not_edt():
    import trimesh

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mesh_dir = root / "assets" / "objects" / "target" / "collision"
        mesh_dir.mkdir(parents=True)
        trimesh.creation.box(extents=(0.12, 0.12, 0.12)).export(mesh_dir / "base0.glb")

        target = FakeActor((0.0, 0.0, 0.84))
        env = type("Env", (), {
            "target_obj": target,
            "target_model": "target",
            "target_id": 0,
            "collision_list": [],
            "occluders": [],
        })()
        cfg = SeedMetricConfig(
            xmin=-0.1, xmax=0.1, ymin=-0.1, ymax=0.1, res=0.1,
            zmin=0.78, zmax=0.9, zres=0.06, seed_snap=0.2,
        )
        xs, ys, zs, XX, YY = gm.build_grid(cfg)
        cache = root / "cache"
        cache.mkdir()
        np.savez(
            cache / "reach_envelope_right.npz",
            base_world=np.zeros(3),
            reach_radius=np.float64(10.0),
            occupancy_prune=np.zeros((len(zs), len(ys), len(xs)), dtype=bool),
            xs=xs, ys=ys, zs=zs, n_feasible=np.int64(1),
        )

        previous = os.environ.get("BENCH_ROOT")
        os.environ["BENCH_ROOT"] = str(root)
        try:
            volume = gm._build_geometric_volume(env, "right", cfg, cache, "occupancy")
        finally:
            if previous is None:
                os.environ.pop("BENCH_ROOT", None)
            else:
                os.environ["BENCH_ROOT"] = previous

        iz = int(np.argmin(np.abs(zs - 0.84)))
        iy = int(np.argmin(np.abs(ys)))
        ix = int(np.argmin(np.abs(xs)))
        assert volume.label[iz, iy, ix] == BEYOND, "target centre remained traversable"
        assert gm._label_for(volume, mask_target=False)[iz, iy, ix] == FREE
        no_target_edt = occluder_clearance_3d(
            [], [], XX, YY, zs, cfg.res, cfg.zres,
            OCC_HALF_FOOTPRINT, shape=cfg.occ_shape,
        )
        assert np.array_equal(volume.edt, no_target_edt), "target leaked into obstacle EDT"
    print("  [2] target centre blocked in label while obstacle EDT is unchanged    PASS")


def test_two_legs_share_one_volume():
    cfg = SeedMetricConfig(
        xmin=0.0, xmax=0.2, ymin=0.0, ymax=0.0, res=0.1,
        zmin=0.8, zmax=0.8, zres=0.1, seed_snap=0.01,
    )
    xs, ys, zs, XX, YY = gm.build_grid(cfg)
    volume = gm._GeometricVolume(
        xs, ys, zs, XX, YY,
        np.full((1, 1, 3), FREE, dtype=np.int8),
        np.full((1, 1, 3), np.inf, dtype=float),
    )
    calls = {"volume": 0}
    original = gm._build_geometric_volume

    def build_once(*_args, **_kwargs):
        calls["volume"] += 1
        return volume

    gm._build_geometric_volume = build_once
    try:
        results = gm.geometric_eps(
            object(), "right",
            [((0.0, 0.0, 0.8), (0.2, 0.0, 0.8)),
             ((0.2, 0.0, 0.8), (0.1, 0.0, 0.8))],
            cfg=cfg, reach_cache_dir="unused",
        )
    finally:
        gm._build_geometric_volume = original

    assert calls["volume"] == 1
    assert len(results) == 2
    assert all(result.merged and result.reason is None for result in results)
    assert all(result.route_world and result.n_free == 3 for result in results)
    required = {
        "eps_star", "merged", "bottleneck_xyz", "route_world",
        "start_xyz", "goal_xyz", "n_free", "reason",
    }
    assert required == set(results[0].__dataclass_fields__)
    print("  [3] two-leg call builds one volume and returns complete LegResults    PASS")


def test_clearance_preferred_route_climbs_without_changing_eps():
    label = np.full((2, 1, 5), FREE, dtype=np.int8)
    edt = np.ones(label.shape, dtype=float)
    edt[1, :, :] = 10.0
    seed_a, seed_b = (0, 0, 0), (0, 0, 4)
    eps_star, _bottleneck, merged = widest_path_eps_3d(
        label, edt, None, seed_a, seed_b, 0.35
    )
    assert merged and eps_star == 1.0

    bfs = reconstruct_widest_path_3d(
        label == FREE, edt, None, seed_a, seed_b, eps_star, 0.35
    )
    preferred = reconstruct_clearance_preferred_path_3d(
        label == FREE, edt, seed_a, seed_b, eps_star, res=0.1, zres=0.1
    )
    assert all(voxel[0] == 0 for voxel in bfs)
    assert any(voxel[0] == 1 for voxel in preferred)
    assert preferred[0] == seed_a and preferred[-1] == seed_b
    assert all(edt[voxel] >= eps_star for voxel in preferred)

    eps_after, _bottleneck, merged_after = widest_path_eps_3d(
        label, edt, None, seed_a, seed_b, 0.35
    )
    assert merged_after and eps_after == eps_star
    print("  [4] preferred representative climbs with eps* unchanged              PASS")


def main():
    print("geometric eps* -- CPU checks")
    test_cpu_only_import()
    test_target_blocks_label_not_edt()
    test_two_legs_share_one_volume()
    test_clearance_preferred_route_climbs_without_changing_eps()
    print("ALL PASS")


if __name__ == "__main__":
    main()
