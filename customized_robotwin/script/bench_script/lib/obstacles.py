"""Scene-obstacle geometry, masks, and clearance fields."""

import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from .scene_constants import OCCLUDER_COLLISION


_MESH_CACHE: dict = {}


def occluder_footprint_polys(env):
    """True 2D footprints of the occluders, for BOTH the distance calc and the drawing.

    Each occluder's COLLISION mesh -- the exact geometry curobo collides against (base<id>.glb,
    used with convex=True) -- is transformed by the occluder's ACTUAL world pose (position +
    orientation, including the random yaw), projected to xy, and convex-hulled. So the footprint
    is orientation-faithful: a non-round bottle comes out as a correctly-rotated polygon, and it
    matches the same mesh that produced the FREE/OBSTACLE labels (no shape mismatch between the
    clearance value and the reachability).

    Returns a list of (K,2) world-xy hull polygons; None if the mesh can't be loaded (the caller
    then falls back to the OCC_HALF_FOOTPRINT circle)."""
    occs = getattr(env, "occluders", None)
    if not occs:
        return []
    try:
        import trimesh
        import transforms3d as t3d
        from scipy.spatial import ConvexHull
        path = os.path.join(os.environ["BENCH_ROOT"], OCCLUDER_COLLISION)
        V = np.asarray(trimesh.load(path, force="mesh").vertices)   # same mesh for every bottle
    except Exception as e:
        print(f"[footprint] could not load occluder collision mesh ({e}); falling back to circle")
        return None
    polys = []
    for occ in occs:
        pose = occ.get_pose()
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=float))   # SAPIEN quat is wxyz
        xy = (V @ R.T + np.asarray(pose.p, dtype=float))[:, :2]         # posed world footprint (xy)
        try:
            polys.append(xy[ConvexHull(xy).vertices])
        except Exception:
            polys.append(None)
    return polys


def occluder_clearance(polys, occ_ps, XX, YY, res, r_foot):
    """Clearance (m) from each grid cell to the nearest OCCLUDER footprint -- not the table /
    furniture / target that also sit in curobo's world.

    Primary path: rasterise the TRUE posed mesh footprints (polys) onto the grid and take the
    Euclidean distance transform -> distance to the real oriented shape, 0 inside a footprint.
    Fallback (polys is None): analytic distance to circles of radius r_foot around occ_ps.

    Measuring to the occluders alone means a table-blocked cell reads FAR here (the table never
    contributes clearance) and we never measure to the reach boundary either. Full-arm/EE-point
    caveat still applies (addendum 2 §5): this is the EE control-point distance to the footprint;
    routability stays body-aware via the IK-FREE node set. No occluders -> +inf."""
    if polys is not None and any(p is not None for p in polys):
        from matplotlib.path import Path as MplPath
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        mask = np.zeros(XX.size, dtype=bool)
        for p in polys:
            if p is not None:
                mask |= MplPath(p).contains_points(pts)
        mask = mask.reshape(XX.shape)
        if not mask.any():
            return np.full(XX.shape, np.inf, dtype=float)
        return distance_transform_edt(~mask, sampling=res)
    if len(occ_ps) == 0:
        return np.full(XX.shape, np.inf, dtype=float)
    d = np.full(XX.shape, np.inf, dtype=float)
    for op in occ_ps:
        d = np.minimum(d, np.hypot(XX - op[0], YY - op[1]) - r_foot)
    return np.clip(d, 0.0, None)


def _load_collision_mesh(path):
    """The collision mesh at `path`, in the actor's own local frame, cached by path.

    Two forms, matching _bench_base_task.update_world: a single mesh FILE (base<id>.glb for
    the convex assets, coacd_collision.obj for the objaverse ones), or a DIRECTORY of convex
    parts, which we concatenate into one mesh -- the union is what the actor collides as.
    Returns (vertices, faces) or None. Never raises; a missing asset just drops that obstacle
    from the metric (and is reported by the caller)."""
    if path in _MESH_CACHE:
        return _MESH_CACHE[path]
    out = None
    try:
        import trimesh
        if os.path.isdir(path):
            parts = []
            for fn in sorted(os.listdir(path)):
                if fn.lower().endswith((".obj", ".glb", ".ply", ".stl")):
                    try:
                        parts.append(trimesh.load(os.path.join(path, fn), force="mesh"))
                    except Exception:
                        continue
            m = trimesh.util.concatenate(parts) if parts else None
        else:
            m = trimesh.load(path, force="mesh")
        if m is not None and len(getattr(m, "vertices", [])):
            out = (np.asarray(m.vertices, dtype=float), np.asarray(m.faces))
    except Exception:
        out = None
    _MESH_CACHE[path] = out
    return out


def scene_obstacle_entries(env, obstacles="all"):
    """The (actor, collision_path) pairs the clearance metric should treat as obstacles.

    obstacles="occluders": the curated olive-oil ring only (env.occluders) -- the pre-2026-07-27
      behaviour, kept so old eps* numbers can be reproduced.
    obstacles="all": everything in env.collision_list, which is the SAME registry curobo's
      update_world consumes, minus the two actors that must never be obstacles -- the target
      being grasped and the destination pad. This picks up procedural table clutter (registered
      with its own per-object collision mesh, is_obstacle=True) and the occluders (registered
      without the flag) in one pass, so the metric's world matches the planner's.

    The table and walls are NOT here: the office base task puts them in cuboid_collision_list,
    a separate list. That is load-bearing -- a tabletop in the obstacle set would swamp the EDT
    and drive eps* to zero everywhere."""
    if obstacles == "occluders":
        occs = getattr(env, "occluders", None) or []
        path = os.path.join(os.environ["BENCH_ROOT"], OCCLUDER_COLLISION)
        return [(o, path) for o in occs]

    skip = set()
    for attr in ("target_obj", "des_obj"):
        a = getattr(env, attr, None)
        if a is not None:
            skip.add(id(a))
    out = []
    for info in (getattr(env, "collision_list", None) or []):
        actor = info.get("actor")
        path = info.get("collision_path")
        if actor is None or not path or id(actor) in skip:
            continue
        # Articulated entries carry a "link" and are posed per-link; the metric's rigid
        # pose-the-whole-mesh transform would place them wrong, so leave them out rather
        # than put a misplaced solid in the field.
        if "link" in info:
            continue
        out.append((actor, path))
    return out


def occluder_footprints_3d(env, obstacles="all"):
    """Per obstacle, the posed COLLISION geometry -- the exact mesh files curobo's world model
    loads for these actors (see _bench_base_task.update_world). `obstacles` selects the set:
    "all" (scene obstacles: clutter + occluders) or "occluders" (the curated ring only); see
    scene_obstacle_entries. We keep three things:

      * "mesh"  the full posed triangle mesh in WORLD coordinates -- the ground truth. Used by the
                --occ-shape mesh path (true per-height cross-sections) and by the 3D render.
      * "poly"  the convex-hull xy footprint (the mesh's widest silhouette from above), and
      * zlo/zhi the mesh's true world z-range.

    "poly" EXTRUDED over [zlo, zhi] is the OLD occluder solid (--occ-shape extruded): correct in
    plan view, but it keeps the widest cross-section all the way to the cap, so it over-fills the
    bottle's NECK badly (olive-oil: ~0.098 m wide at the body, ~0.019 m at the neck -> up to ~4 cm of
    phantom obstacle near the top, larger than the gripper half-width). --occ-shape mesh removes that
    by re-cutting the real mesh at every z. Either way, above zhi there is no occluder at all, so the
    gripper is free to pass over -- the whole point of going 3D.

    "center" is the actor's world spawn position (marker only). Returns a list of
    dict(poly=(K,2)|None, zlo, zhi, mesh=Trimesh|None, center=(3,)); None if NOTHING could be
    loaded (caller falls back to cylinders). An empty list means there were no obstacles."""
    entries = scene_obstacle_entries(env, obstacles)
    if not entries:
        return []
    try:
        import trimesh
        import transforms3d as t3d
        from scipy.spatial import ConvexHull
    except Exception as e:
        print(f"[footprint] geometry deps unavailable ({e}); falling back to cylinders")
        return None
    out, missed = [], 0
    for actor, path in entries:
        loaded = _load_collision_mesh(path)
        if loaded is None:
            missed += 1
            continue
        V0, F0 = loaded
        pose = actor.get_pose()
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=float))   # SAPIEN quat is wxyz
        try:                                    # curobo scales the same mesh by the actor's scale
            sc = np.asarray(actor.scale, dtype=float)
        except Exception:
            sc = 1.0
        Vw = (V0 * sc) @ R.T + np.asarray(pose.p, dtype=float)          # (V,3) posed world verts
        try:
            poly = Vw[:, :2][ConvexHull(Vw[:, :2]).vertices]
        except Exception:
            poly = None
        try:
            mesh = trimesh.Trimesh(vertices=Vw, faces=F0, process=False)
        except Exception:
            mesh = None
        out.append(dict(poly=poly, zlo=float(Vw[:, 2].min()), zhi=float(Vw[:, 2].max()),
                        mesh=mesh, center=np.asarray(pose.p, dtype=float)))
    if missed:
        # Loud: a dropped obstacle is a hole in the clearance field, and eps* would come out
        # optimistic through a gap that is not really there.
        print(f"\033[93m[footprint] {missed}/{len(entries)} obstacle mesh(es) failed to load and "
              f"are NOT in the clearance field -- eps* is optimistic by that much\033[0m")
    if not out:
        return None
    return out


def obstacle_centers(foots):
    """World centres of the posed obstacles, for the cylinder fallback and the plot markers.
    Derived from the footprints so it always matches the obstacle set actually used."""
    return [np.asarray(f["center"], dtype=float) for f in (foots or [])]


def occluder_slice_polys(f, z):
    """The TRUE cross-section of one posed occluder mesh at height z: the closed world-xy loops where
    the plane z cuts the mesh (outer boundary first, any inner loops after). This is what makes the
    occluder solid taper -- a fat body low down, a thin neck up top -- instead of being a prism.
    Returns [] when the plane misses the mesh (above the cap / below the base) or on any failure."""
    m = f.get("mesh")
    if m is None or not (f["zlo"] - 1e-9 <= z <= f["zhi"] + 1e-9):
        return []
    try:
        sec = m.section(plane_origin=[0.0, 0.0, float(z)], plane_normal=[0.0, 0.0, 1.0])
    except Exception:
        return []
    if sec is None:
        return []
    return [np.asarray(loop)[:, :2] for loop in sec.discrete if len(loop) >= 3]


def occluder_mask_3d(foots, XX, YY, zs, shape="mesh"):
    """Voxel occupancy (nz,ny,nx bool) of the occluder solid, in one of two geometries:

      shape="mesh"     : re-cut the posed COLLISION mesh at every z and fill the resulting loops --
                         the bottle tapers, so the neck is thin and the cap is gone. Faithful to the
                         geometry curobo collides against, up to grid sampling.
      shape="extruded" : the old solid -- the widest xy footprint held constant over [zlo, zhi].

    Thin-feature guard (mesh path): a neck narrower than one grid cell can slip between sample points
    and vanish, which would read as free space. Whenever a slice has real cross-section loops but
    stamps no cell, we stamp the cell nearest each loop's centroid so the occluder never disappears.
    Returns None if no footprint geometry is available at all."""
    from matplotlib.path import Path as MplPath
    ny, nx = XX.shape
    nz = len(zs)
    cols = np.column_stack([XX.ravel(), YY.ravel()])
    mask = np.zeros((nz, ny, nx), dtype=bool)

    # Per-obstacle bounding-box clip. contains_points over the FULL grid costs
    # O(nz x n_obstacles x ny x nx), which is fine for one bottle and not fine for a cluttered
    # table. Each cross-section only ever occupies its own xy bbox, so we test just that
    # sub-grid: the cost becomes proportional to the obstacles' combined area, not the table's.
    # Only valid on a regular ascending grid, which is how run() builds it; otherwise fall back
    # to the full-grid test.
    xs1d, ys1d = XX[0, :], YY[:, 0]
    regular = (nx > 1 and ny > 1
               and bool(np.all(np.diff(xs1d) > 0)) and bool(np.all(np.diff(ys1d) > 0)))

    def _stamp(loops, closed=True):
        """Boolean (ny,nx) coverage of one slice's loops, evaluated on the bbox sub-grid.

        closed=True is for mesh cross-sections, whose loops repeat their first vertex at the
        end. Do NOT pass it for a plain polygon: Path(verts, closed=True) turns the LAST vertex
        into the CLOSEPOLY marker, so a 4-corner hull would silently become a triangle."""
        cp = MplPath.make_compound_path(*[MplPath(l, closed=closed) for l in loops])
        hit = np.zeros((ny, nx), dtype=bool)
        if not regular:
            return cp.contains_points(cols).reshape(ny, nx)
        allpts = np.vstack(loops)
        # +-1 cell of slack so a loop edge falling between samples still covers its cell
        ix0 = max(0, int(np.searchsorted(xs1d, allpts[:, 0].min())) - 1)
        ix1 = min(nx, int(np.searchsorted(xs1d, allpts[:, 0].max())) + 1)
        iy0 = max(0, int(np.searchsorted(ys1d, allpts[:, 1].min())) - 1)
        iy1 = min(ny, int(np.searchsorted(ys1d, allpts[:, 1].max())) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            return hit
        sub = np.column_stack([XX[iy0:iy1, ix0:ix1].ravel(), YY[iy0:iy1, ix0:ix1].ravel()])
        hit[iy0:iy1, ix0:ix1] = cp.contains_points(sub).reshape(iy1 - iy0, ix1 - ix0)
        return hit

    if shape == "mesh" and foots and any(f.get("mesh") is not None for f in foots):
        rescued = 0
        for iz, z in enumerate(zs):
            for f in foots:
                loops = occluder_slice_polys(f, float(z))
                if not loops:
                    continue
                # compound path so an inner loop punches a hole rather than filling it
                hit = _stamp(loops)
                if not hit.any():                       # thin-feature rescue (see docstring)
                    for l in loops:
                        c = l.mean(axis=0)
                        iy, ix = np.unravel_index(int(np.argmin((XX - c[0]) ** 2 + (YY - c[1]) ** 2)),
                                                  XX.shape)
                        hit[iy, ix] = True
                    rescued += 1
                mask[iz] |= hit
        if rescued:
            print(f"[edt] thin-feature rescue on {rescued} slice(s): the cross-section was narrower "
                  f"than the grid, so a single centre cell was stamped")
        return mask

    if not (foots and any(f["poly"] is not None for f in foots)):
        return None
    inxy = np.zeros((len(foots), ny, nx), dtype=bool)      # per-occluder in-footprint (xy), cached
    for i, f in enumerate(foots):
        if f["poly"] is not None:
            inxy[i] = _stamp([np.asarray(f["poly"], dtype=float)], closed=False)
    for iz, z in enumerate(zs):
        for i, f in enumerate(foots):
            if f["poly"] is not None and f["zlo"] - 1e-9 <= z <= f["zhi"] + 1e-9:
                mask[iz] |= inxy[i]
    return mask


def occluder_clearance_3d(foots, occ_ps, XX, YY, zs, res, zres, r_foot, shape="mesh"):
    """3D clearance (m) from each voxel to the nearest OCCLUDER -- not the table / furniture / target
    that also sit in curobo's world. Primary path: rasterise the occluder solid (see
    occluder_mask_3d; --occ-shape picks the true tapering mesh or the old extruded footprint) into a
    (nz,ny,nx) mask, then the anisotropic 3D Euclidean distance transform (sampling (zres,res,res))
    -> 0 inside a bottle, growing outward, and UNBOUNDED above the bottle top (no occluder there =
    free to pass over). Fallback (no footprint geometry): vertical cylinders radius r_foot.
    No occluders -> +inf everywhere. Returns (nz, ny, nx) float.

    Grid caveat (both shapes): the EDT measures voxel-centre to voxel-centre, so the returned
    clearance sits up to about half a voxel (res/2 in x,y; zres/2 in z) ABOVE the true point-to-
    surface distance. eps* inherits that bias -- it is slightly optimistic, by construction."""
    ny, nx = XX.shape
    nz = len(zs)
    mask = occluder_mask_3d(foots, XX, YY, zs, shape=shape)
    if mask is not None:
        if not mask.any():
            return np.full((nz, ny, nx), np.inf, dtype=float)
        return distance_transform_edt(~mask, sampling=(zres, res, res))
    if len(occ_ps) == 0:
        return np.full((nz, ny, nx), np.inf, dtype=float)
    d = np.full((nz, ny, nx), np.inf, dtype=float)             # vertical cylinders (xy distance only)
    for op in occ_ps:
        dxy = np.hypot(XX - op[0], YY - op[1]) - r_foot
        d = np.minimum(d, np.broadcast_to(dxy, (nz, ny, nx)))
    return np.clip(d, 0.0, None)


def surface_distance_to_occluders(foots, pt):
    """Exact continuous distance (m) from a world point to the nearest occluder SURFACE, measured on
    the real posed triangle mesh -- no grid, no extrusion. This is the honest yardstick for the eps*
    sphere: eps* itself comes from the voxel EDT, so comparing the two shows how much the grid costs
    us. Returns None if no mesh is available."""
    best = None
    for f in (foots or []):
        m = f.get("mesh")
        if m is None:
            continue
        try:
            import trimesh
            d = float(trimesh.proximity.closest_point(m, np.asarray([pt], dtype=float))[1][0])
        except Exception:
            continue
        best = d if best is None else min(best, d)
    return best
