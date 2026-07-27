#!/usr/bin/env python3
"""Checks for the scene-obstacle clearance set (clearance_metric_3d --obstacles).

Three things worth proving without a GPU or a live scene:

  1. occluder_mask_3d's bounding-box clip is EXACT. It is a pure speedup, so its mask must be
     byte-identical to the full-grid contains_points reference for every slice -- a silent
     off-by-one there would carve phantom gaps in the clearance field and make eps* optimistic.
  2. scene_obstacle_entries picks the right actors: clutter and occluders in, the target and
     the pad out, articulated (per-link) entries out.
  3. _load_collision_mesh handles both forms the registry uses -- a single mesh file and a
     directory of convex parts (which must come back as their union).

Run:  python test_obstacle_set.py
"""
import os
import sys
import tempfile

import numpy as np

import clearance_metric_3d as cm


class FakeActor:
    """Minimal stand-in for a SAPIEN actor: pose (p, q) + scale, which is all the footprint
    builder reads."""
    def __init__(self, p=(0, 0, 0), q=(1, 0, 0, 0), scale=1.0):
        self._p, self._q, self.scale = np.asarray(p, float), np.asarray(q, float), scale

    def get_pose(self):
        return type("P", (), {"p": self._p, "q": self._q})()


class FakeEnv:
    def __init__(self, collision_list, target=None, pad=None, occluders=None):
        self.collision_list = collision_list
        if target is not None:
            self.target_obj = target
        if pad is not None:
            self.des_obj = pad
        self.occluders = occluders or []


def _grid(res=0.01, xlim=(-0.3, 0.3), ylim=(-0.25, 0.25)):
    xs = np.arange(xlim[0], xlim[1] + 1e-9, res)
    ys = np.arange(ylim[0], ylim[1] + 1e-9, res)
    XX, YY = np.meshgrid(xs, ys)
    return XX, YY


def _box_foot(cx, cy, zlo, zhi, hx, hy):
    """A footprint dict whose mesh is an axis-aligned box -- gives exact, checkable slices."""
    import trimesh
    V = np.array([[cx - hx, cy - hy, zlo], [cx + hx, cy - hy, zlo],
                  [cx + hx, cy + hy, zlo], [cx - hx, cy + hy, zlo],
                  [cx - hx, cy - hy, zhi], [cx + hx, cy - hy, zhi],
                  [cx + hx, cy + hy, zhi], [cx - hx, cy + hy, zhi]], dtype=float)
    F = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    poly = np.array([[cx - hx, cy - hy], [cx + hx, cy - hy],
                     [cx + hx, cy + hy], [cx - hx, cy + hy]], dtype=float)
    return dict(poly=poly, zlo=zlo, zhi=zhi, mesh=mesh,
                center=np.array([cx, cy, 0.5 * (zlo + zhi)]))


def _reference_mask(foots, XX, YY, zs, shape):
    """Full-grid contains_points, i.e. the pre-bbox behaviour, recomputed here as the oracle."""
    from matplotlib.path import Path as MplPath
    ny, nx = XX.shape
    cols = np.column_stack([XX.ravel(), YY.ravel()])
    mask = np.zeros((len(zs), ny, nx), dtype=bool)
    if shape == "mesh":
        for iz, z in enumerate(zs):
            for f in foots:
                loops = cm.occluder_slice_polys(f, float(z))
                if not loops:
                    continue
                cp = MplPath.make_compound_path(*[MplPath(l, closed=True) for l in loops])
                hit = cp.contains_points(cols).reshape(ny, nx)
                if not hit.any():
                    for l in loops:
                        c = l.mean(axis=0)
                        iy, ix = np.unravel_index(
                            int(np.argmin((XX - c[0]) ** 2 + (YY - c[1]) ** 2)), XX.shape)
                        hit[iy, ix] = True
                mask[iz] |= hit
        return mask
    inxy = np.zeros((len(foots), ny, nx), dtype=bool)
    for i, f in enumerate(foots):
        inxy[i] = MplPath(f["poly"]).contains_points(cols).reshape(ny, nx)
    for iz, z in enumerate(zs):
        for i, f in enumerate(foots):
            if f["zlo"] - 1e-9 <= z <= f["zhi"] + 1e-9:
                mask[iz] |= inxy[i]
    return mask


def test_bbox_clip_is_exact():
    XX, YY = _grid()
    zs = np.arange(0.78, 1.10, 0.02)
    # a spread of sizes and positions, incl. one at the grid edge and one sub-cell thin sliver
    foots = [
        _box_foot(0.00, 0.00, 0.80, 1.00, 0.05, 0.05),
        _box_foot(-0.22, 0.18, 0.78, 0.92, 0.03, 0.04),
        _box_foot(0.24, -0.20, 0.82, 1.05, 0.02, 0.02),
        _box_foot(0.10, 0.10, 0.85, 0.95, 0.002, 0.002),   # thinner than a cell -> rescue path
        _box_foot(-0.30, -0.25, 0.79, 0.90, 0.02, 0.02),   # flush against the grid corner
    ]
    for shape in ("mesh", "extruded"):
        got = cm.occluder_mask_3d(foots, XX, YY, zs, shape=shape)
        ref = _reference_mask(foots, XX, YY, zs, shape)
        assert got is not None, shape
        assert got.shape == ref.shape, (got.shape, ref.shape)
        assert np.array_equal(got, ref), (
            f"bbox-clipped mask differs from full-grid reference ({shape}): "
            f"{int((got != ref).sum())} voxel(s)")
        assert ref.any(), f"{shape} reference is empty -- the test would prove nothing"
    print("  [1] bbox clip byte-identical to full-grid reference (mesh + extruded)  PASS")


def test_bbox_clip_on_irregular_grid():
    """A descending axis must fall back to the full-grid path rather than silently mis-clip."""
    XX, YY = _grid()
    XX = XX[:, ::-1]                      # x now descends -> `regular` is False
    zs = np.arange(0.80, 0.95, 0.02)
    foots = [_box_foot(0.05, 0.05, 0.80, 0.94, 0.04, 0.04)]
    got = cm.occluder_mask_3d(foots, XX, YY, zs, shape="mesh")
    ref = _reference_mask(foots, XX, YY, zs, "mesh")
    assert np.array_equal(got, ref), "irregular-grid fallback diverged"
    print("  [2] irregular grid falls back to full-grid, still exact                PASS")


def test_scene_obstacle_entries():
    tgt, pad = FakeActor(), FakeActor()
    clutter_a, clutter_b, occ, arti = FakeActor(), FakeActor(), FakeActor(), FakeActor()
    env = FakeEnv(
        collision_list=[
            {"actor": clutter_a, "collision_path": "/x/a.glb", "is_obstacle": True},
            {"actor": clutter_b, "collision_path": "/x/b.glb", "is_obstacle": True},
            {"actor": occ, "collision_path": "/x/occ.glb"},          # occluder: no is_obstacle
            {"actor": tgt, "collision_path": "/x/target.glb"},       # must be excluded
            {"actor": pad, "collision_path": "/x/pad.glb"},          # must be excluded
            {"actor": arti, "collision_path": "/x/arm", "link": "l0"},  # per-link: excluded
            {"actor": None, "collision_path": "/x/none.glb"},
            {"actor": FakeActor(), "collision_path": ""},
        ],
        target=tgt, pad=pad, occluders=[occ],
    )
    got = cm.scene_obstacle_entries(env, "all")
    assert [p for _, p in got] == ["/x/a.glb", "/x/b.glb", "/x/occ.glb"], got
    actors = [a for a, _ in got]
    assert tgt not in actors and pad not in actors, "target/pad leaked into the obstacle set"
    assert arti not in actors, "articulated entry leaked in"

    os.environ.setdefault("BENCH_ROOT", "/tmp/bench_root_stub")
    only = cm.scene_obstacle_entries(env, "occluders")
    assert [a for a, _ in only] == [occ], only
    assert len(only) == 1, "occluders mode must ignore collision_list"
    print("  [3] entry selection: clutter+occluder in; target/pad/articulated out   PASS")


def test_load_collision_mesh():
    import trimesh
    cm._MESH_CACHE.clear()
    with tempfile.TemporaryDirectory() as td:
        # single file
        f = os.path.join(td, "one.obj")
        trimesh.creation.box(extents=(0.1, 0.1, 0.2)).export(f)
        got = cm._load_collision_mesh(f)
        assert got is not None and len(got[0]) == 8, got
        assert cm._load_collision_mesh(f) is got, "second call should hit the cache"

        # directory of convex parts -> union
        d = os.path.join(td, "parts")
        os.makedirs(d)
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(os.path.join(d, "p0.obj"))
        b1 = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        b1.apply_translation((0.5, 0, 0))
        b1.export(os.path.join(d, "p1.obj"))
        gotd = cm._load_collision_mesh(d)
        assert gotd is not None, "directory form returned nothing"
        V = gotd[0]
        assert len(V) == 16, f"parts not concatenated (got {len(V)} verts)"
        assert V[:, 0].max() > 0.4, "second part missing from the union"

        assert cm._load_collision_mesh(os.path.join(td, "missing.glb")) is None
    print("  [4] mesh loader: file, cache, directory-of-parts union, missing->None  PASS")


def test_centers_track_the_set():
    foots = [_box_foot(0.1, 0.2, 0.8, 0.9, 0.02, 0.02),
             _box_foot(-0.1, 0.0, 0.8, 0.9, 0.02, 0.02)]
    c = cm.obstacle_centers(foots)
    assert len(c) == 2 and np.allclose(c[0][:2], [0.1, 0.2]), c
    assert cm.obstacle_centers([]) == [] and cm.obstacle_centers(None) == []
    print("  [5] obstacle_centers tracks the footprint list                         PASS")


def main():
    print("scene-obstacle clearance set -- checks")
    test_bbox_clip_is_exact()
    test_bbox_clip_on_irregular_grid()
    test_scene_obstacle_entries()
    test_load_collision_mesh()
    test_centers_track_the_set()
    print("ALL PASS")


if __name__ == "__main__":
    sys.exit(main())
