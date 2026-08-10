"""Shared plotting helpers for benchmark analysis tools."""

import numpy as np

import numpy as np

# Four azimuths so the depth ordering of the path vs the box is readable.
VIEWS = [("iso", 26, -60), ("front", 8, -90), ("side", 8, 0), ("top", 78, -90)]


def _box_wireframe(ax, box_p, half, height):
    """Milk-box occluder as a wireframe. box_p[2] is the box BASE (sits on the table top),
    so the box spans [z, z + height]. `half` is the base half-diagonal including yaw, so
    this is a conservative axis-aligned envelope, not the exact yawed footprint."""
    x, y, z0 = float(box_p[0]), float(box_p[1]), float(box_p[2])
    z1 = z0 + height
    xs = [x - half, x + half]
    ys = [y - half, y + half]
    # 4 verticals
    for xi in xs:
        for yi in ys:
            ax.plot([xi, xi], [yi, yi], [z0, z1], color="dimgray", lw=1.2, alpha=0.9)
    # top + bottom rectangles
    for zi in (z0, z1):
        ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]],
                [ys[0], ys[0], ys[1], ys[1], ys[0]],
                [zi] * 5, color="dimgray", lw=1.2, alpha=0.9)
    ax.plot([], [], [], color="dimgray", lw=1.2, label="milk-box occluder")


def _write_video(env, args):
    """Close the env and merge the captured frames into <run_dir>/video/episode{seed}.mp4.
    Same close -> merge -> drop-cache order visualize_task_scene.py uses; the merge only
    has frames to work with when save_data was on (i.e. not --no-video)."""
    try:
        env.close_env(clear_cache=True)
    except Exception as e:
        print(f"[video] close_env failed ({type(e).__name__}: {e})")
        return
    if not args.save_video:
        return
    try:
        env.merge_pkl_to_hdf5_video()
        env.remove_data_cache()
    except Exception as e:
        # a missing video must not sink an otherwise-good figure run
        print(f"[video] merge failed ({type(e).__name__}: {e}); figures are unaffected")


def _draw_occluder_solids_3d(ax, foots, shape):
    """Draw the occluders in 3D as the SAME solid the clearance field was measured against: the true
    posed collision mesh under --occ-shape mesh, the extruded footprint prism under extruded. Keeping
    the picture and the metric on one geometry is what makes the eps* sphere's 'just touches' claim
    checkable by eye."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for f in (foots or []):
        m = f.get("mesh")
        if shape == "mesh" and m is not None:
            tris = np.asarray(m.vertices)[np.asarray(m.faces)]
            ax.add_collection3d(Poly3DCollection(tris, facecolor="red", alpha=0.10, edgecolor="none"))
            continue
        if f["poly"] is None:
            continue
        p, zlo, zhi = f["poly"], f["zlo"], f["zhi"]
        bottom = [(x, y, zlo) for x, y in p]
        top = [(x, y, zhi) for x, y in p]
        faces = [bottom, top] + [[bottom[i], bottom[(i + 1) % len(p)], top[(i + 1) % len(p)], top[i]]
                                 for i in range(len(p))]
        ax.add_collection3d(Poly3DCollection(faces, facecolor="red", alpha=0.12, edgecolor="red", lw=0.5))


def _draw_ground_plane(ax, xlim, ylim, z):
    """The support surface (table top) as a translucent quad at height z, so the obstacles read as
    standing ON something instead of hovering. Drawn BEFORE the occluder solids: matplotlib
    depth-sorts each Poly3DCollection independently, and adding the lowest surface first is what
    keeps the painter's order right for a plane that sits under everything else."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    quad = [[(xlim[0], ylim[0], z), (xlim[1], ylim[0], z),
             (xlim[1], ylim[1], z), (xlim[0], ylim[1], z)]]
    ax.add_collection3d(Poly3DCollection(quad, facecolor="#c8b89a", alpha=0.35,
                                         edgecolor="#8a7a5a", lw=1.0, zsort="min"))
    ax.plot([], [], [], "-", color="#8a7a5a", lw=1.2, label=f"table top (z = {z:.2f} m)")


def _draw_eps_sphere(ax, centre, radius, n=48):
    """Wireframe sphere of radius eps* centred on the bottleneck voxel -- the 3D twin of the 2D eps*
    circle. Its surface is the set of points exactly eps* from the bottleneck, so with the axes at
    equal aspect it should just KISS the nearest occluder: eps* IS that distance. A sphere that
    visibly bites into a bottle, or floats clear of every bottle, means the clearance field and the
    drawn geometry have drifted apart."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#00bcd4", lw=0.6, alpha=0.55, rstride=3, cstride=3)
    ax.plot([centre[0]], [centre[1]], [centre[2]], ".", color="#00bcd4", ms=1,
            label=f"eps* sphere (r = {radius:.3f} m)")


def _equal_aspect_3d(ax, pts, pad=0.02):
    """Force a TRUE 1:1:1 data aspect over the bounding box of `pts` (list of (3,) points). Without
    this matplotlib stretches each axis independently, the eps* sphere renders as an ellipsoid and
    'just touching' becomes unreadable."""
    P = np.asarray([p for p in pts if p is not None], dtype=float)
    if P.size == 0:
        return
    lo, hi = P.min(axis=0) - pad, P.max(axis=0) + pad
    span = float(max(hi - lo))
    mid = 0.5 * (lo + hi)
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
    ax.set_box_aspect((1, 1, 1))


def _true_aspect_3d(ax, lo, hi):
    """Frame the axes on an EXPLICIT world box, still at a true 1:1:1 data aspect.

    _equal_aspect_3d gets 1:1:1 by padding every axis out to the LONGEST span. That is right for
    roughly cubic data and wasteful otherwise: a 1.2 x 0.7 x 0.3 m tabletop gets a 1.2 m z axis, so
    the scene collapses into the middle third of an empty cube with no tick near it. Setting the box
    aspect to the box's OWN spans keeps one metre the same rendered length on all three axes -- the
    eps* sphere still renders as a sphere, so 'just kisses the nearest obstacle' stays checkable --
    without the padding. Caller supplies the box, so the frame is the same in every figure."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    span = hi - lo
    if not (np.all(np.isfinite(span)) and np.all(span > 0)):
        return False
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(span))
    return True


def _line_axis(g_xy, p_xy):
    """Unit direction + length of the grasp->pad line in the xy plane (the profile/side-view axis)."""
    d = np.asarray(p_xy, float) - np.asarray(g_xy, float)
    L = float(np.hypot(*d))
    u = d / L if L > 1e-9 else np.array([1.0, 0.0])
    return np.asarray(g_xy, float), u, L


def _scene_anchor_markers(ax, tgt_p=None, ee_xyz=None, arm=""):
    """The two scene anchors every plan-view figure needs to be readable: where the TARGET BOTTLE
    actually spawned (the thing being picked -- distinct from the 'grasp seed', which is that pose
    snapped to the nearest FREE voxel) and where the acting GRIPPER currently is (its rest pose, i.e.
    the end the route has to start from). Both are plotted in the xy plane."""
    if tgt_p is not None:
        ax.plot(tgt_p[0], tgt_p[1], "*", color="blue", ms=17, mec="k", mew=0.6,
                label="target bottle (spawn)")
    if ee_xyz is not None:
        ax.plot(ee_xyz[0], ee_xyz[1], "^", color="darkorange", ms=12, mec="k", mew=0.6,
                label=f"gripper now ({arm})")
