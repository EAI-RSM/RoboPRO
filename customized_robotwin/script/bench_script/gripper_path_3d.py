#!/usr/bin/env python3
"""
3D gripper path (polyline) for the occluder rollout -- issue #35 visualisation.

Runs the REAL expert rollout (OccluderTask.play_once) on a chosen seed with the milk-box
occluder ON and NO table clutter (density 0), sampling the gripper TCP pose after every
executed physics step via `env.step_hook` (added in _base_task.take_dense_action). The
result is plotted as a 3D polyline coloured by time, with the scene landmarks marked:

    * milk-box occluder  -- grey wireframe box (footprint OCC_HALF_FOOTPRINT, height OCC_HEIGHT)
    * bottle start       -- blue dot at the target's t=0 pose (BEFORE the rollout moves it)
    * destination pad    -- magenta square at PAD_XY

A rollout that fails is still plotted: the path is drawn as far as it got and the last
sampled point is marked with a red X ("rollout failed here"). The SUCCESS/FAILED verdict
goes in the title, so a failed frame is never mistaken for a complete path.

Only the ACTIVE arm is drawn. The task is single-armed but which arm it picks varies with
the bottle's x, so we sample both TCPs and keep whichever actually moved (longest path).

Four viewing angles are saved per seed -- a single static 3D projection is hard to read.
An mp4 of the rollout is saved alongside them (--no-video to skip).

OUTPUT LAYOUT: every run writes to its OWN timestamped folder, so re-running never
overwrites an earlier run's figures:

    <out-dir>/<YYYYmmdd-HHMMSS>/grippath_seed0001_FAILED_iso.png
                               video/episode1.mp4

USAGE (from customized_robotwin, env sourced + ROBOTWIN_BENCH_TASK=bench):
    python script/bench_script/gripper_path_3d.py --seed 1
    python script/bench_script/gripper_path_3d.py --seed 7 --offset 0.2 --stride 5
    python script/bench_script/gripper_path_3d.py --seed 1 --no-video
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setup_paths import setup_paths
setup_paths()

from analyze_occluder_visibility import (make_occluder_task, PAD_XY,  # noqa: E402
                                         OCC_HALF_FOOTPRINT)
from analyze_natural_visibility import build_cfg, DR_CLEAN  # noqa: E402
from reachability_view import OCC_HEIGHT  # noqa: E402  (milk-box height, 0.2542 m)

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


def _plot_path(seed, pts, ok_overall, box_p, tgt_p, pad_xy, arm_name, run_dir, args):
    """3D polyline coloured by time (viridis: dark = start, bright = end)."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    # segment i spans pts[i] -> pts[i+1]; colour by normalised time at the segment start
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    tnorm = np.linspace(0.0, 1.0, len(segs))

    for tag, elev, azim in VIEWS:
        fig = plt.figure(figsize=(9.5, 8))
        ax = fig.add_subplot(111, projection="3d")

        lc = Line3DCollection(segs, cmap="viridis", norm=plt.Normalize(0, 1), lw=2.0)
        lc.set_array(tnorm)
        ax.add_collection3d(lc)

        _box_wireframe(ax, box_p, OCC_HALF_FOOTPRINT, OCC_HEIGHT)
        ax.scatter([tgt_p[0]], [tgt_p[1]], [tgt_p[2]], color="tab:blue", s=70,
                   marker="o", depthshade=False, ec="black", label="bottle start")
        # pad is flat on the table; draw it at the box base height (= table top)
        ax.scatter([pad_xy[0]], [pad_xy[1]], [float(box_p[2])], color="magenta", s=80,
                   marker="s", depthshade=False, ec="black", label="pad (destination)")

        ax.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], color="black", s=45,
                   marker="^", depthshade=False, label="gripper start")
        if ok_overall:
            ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], color="tab:green", s=70,
                       marker="*", depthshade=False, ec="black", label="gripper end")
        else:
            ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], color="red", s=110,
                       marker="X", depthshade=False, ec="black", label="rollout failed here")

        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_xlim(args.xmin, args.xmax); ax.set_ylim(args.ymin, args.ymax)
        ax.set_zlim(args.zmin, args.zmax)
        try:
            ax.set_box_aspect((args.xmax - args.xmin, args.ymax - args.ymin,
                               args.zmax - args.zmin))
        except Exception:
            pass   # older matplotlib: fall back to the default (non-equal) aspect
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper left", fontsize=8)

        sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.10)
        cb.set_label("rollout progress (0 = start, 1 = end)")

        ax.set_title(f"seed {seed}  |  {arm_name} gripper path  |  "
                     f"rollout {'SUCCESS' if ok_overall else 'FAILED'}\n"
                     f"occluder offset {args.offset:.2f} m, no clutter (density 0)  |  "
                     f"{len(pts)} samples (stride {args.stride})  |  view: {tag}")

        vtag = "SUCCESS" if ok_overall else "FAILED"
        p = out / f"grippath_seed{seed:04d}_{vtag}_{tag}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"saved {p}")


def run(args):
    # One folder per run, so re-running a seed never clobbers the previous figures/video.
    run_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] writing to {run_dir}")

    env = make_occluder_task()()
    env.spawn_occluder = True
    env.occluder_offset = args.offset

    # DR_CLEAN == {"cluttered_table": False, "obstacle_density": 0, ...} -> occluder only.
    # rollout=True is required: it sets need_plan=True so play_once actually PLANS. With the
    # default (rollout=False) the env is a t=0 measurement build that replays a joint-path
    # cache nothing ever filled -> IndexError on the first move.
    # save_path=run_dir -> the mp4 lands in <run_dir>/video/episode{seed}.mp4.
    cfg = build_cfg("put_mouse_on_pad", args.base_config, args.seed, DR_CLEAN,
                    rollout=True, ep_num=args.seed,
                    save_path=(str(run_dir) if args.save_video else None))
    if not args.save_video:
        cfg["save_data"] = False      # execute the plan but skip frame capture (faster)
    env.setup_demo(**cfg)

    box_p = np.array(env.occluder.get_pose().p)
    tgt_p = np.array(env.target_obj.get_pose().p)   # BEFORE the rollout moves the bottle
    print(f"[seed {args.seed}] box={np.round(box_p, 3)}  bottle={np.round(tgt_p, 3)}  "
          f"pad={PAD_XY}")

    # Sample BOTH TCPs every executed physics step; pick the arm that actually moved.
    left, right = [], []

    def hook(control_idx):
        if control_idx % args.stride:
            return
        left.append(env.robot.get_left_tcp_pose()[:3])
        right.append(env.robot.get_right_tcp_pose()[:3])

    env.step_hook = hook
    try:
        env.play_once()
        ok_overall = bool(getattr(env, "plan_success", False))
    except Exception as e:
        print(f"[seed {args.seed}] rollout raised ({type(e).__name__}: {e}); "
              f"plotting the path up to the failure")
        ok_overall = False
    env.step_hook = None

    L, R = np.array(left, dtype=float), np.array(right, dtype=float)
    if len(L) < 2:
        print(f"[seed {args.seed}] only {len(L)} sample(s) captured -- nothing to plot "
              f"(the rollout failed before any arm motion). Try another seed.")
        _write_video(env, args)
        return

    # path length decides the active arm; the idle arm holds its home pose and moves ~0
    dl = float(np.linalg.norm(np.diff(L, axis=0), axis=1).sum())
    dr = float(np.linalg.norm(np.diff(R, axis=0), axis=1).sum())
    arm_name, pts = ("left", L) if dl >= dr else ("right", R)
    print(f"[seed {args.seed}] rollout {'SUCCESS' if ok_overall else 'FAILED'}; "
          f"{len(pts)} samples; path length left={dl:.3f} m right={dr:.3f} m "
          f"-> plotting {arm_name} arm")

    _plot_path(args.seed, pts, ok_overall, box_p, tgt_p, np.array(PAD_XY), arm_name,
               run_dir, args)
    _write_video(env, args)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed", type=int, default=1, help="single seed to roll out and plot")
    ap.add_argument("--offset", type=float, default=0.2,
                    help="occluder offset in front of the target (m)")
    ap.add_argument("--stride", type=int, default=5,
                    help="sample every Nth physics step (1 = every step; the sim runs "
                         "fast enough that stride 1 yields ~100k near-identical poses)")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.45)
    ap.add_argument("--ymax", type=float, default=0.45)
    ap.add_argument("--zmin", type=float, default=0.70, help="table top sits at ~0.742 m")
    ap.add_argument("--zmax", type=float, default=1.20)
    ap.add_argument("--out-dir", default="../scripts/validation/results/gripper_path_3d",
                    help="repo-root results location (same convention as other bench scripts). "
                         "Each run lands in its own <out-dir>/<timestamp>/ subfolder.")
    ap.add_argument("--no-video", dest="save_video", action="store_false",
                    help="skip the rollout mp4 (faster: no frame capture). By default the "
                         "video is written to <out-dir>/<timestamp>/video/episode{seed}.mp4")
    ap.set_defaults(save_video=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
