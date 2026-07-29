"""Shared plotting helpers for benchmark analysis tools."""

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
