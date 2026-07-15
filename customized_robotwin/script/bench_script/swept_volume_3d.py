#!/usr/bin/env python3
"""
3D SWEPT VOLUME of the arm over an occluder rollout (issue #35 visualisation).

Companion to gripper_path_3d.py: that script draws where the gripper TCP went (a line),
this one draws the space the whole ARM occupied over the rollout (a volume).

Runs the REAL expert rollout (OccluderTask.play_once) on a chosen seed with the milk-box
occluder ON and NO table clutter (density 0). Via `env.step_hook` (in take_dense_action)
it samples every arm LINK's pose each executed physics step, transforms that link's
collision spheres into the world, and stamps them into a boolean occupancy grid. The union
over all steps IS the swept volume; it is rendered as a marching-cubes isosurface with the
scene landmarks marked (milk-box wireframe, bottle start, pad) and the TCP path drawn
inside it for continuity with gripper_path_3d.py.

WHAT THE VOLUME ACTUALLY IS -- READ THIS BEFORE INTERPRETING THE FIGURE:
The spheres come from the embodiment's curobo collision model
(assets/embodiments/aloha-agilex/collision_aloha_{left,right}.yml), i.e. the SAME geometry
CuRobo plans against: 36 spheres for the left arm, 33 for the right. That model is a
deliberately CONSERVATIVE, padded approximation of the real meshes, so this volume is
"the space the PLANNER believes the arm sweeps", NOT a literal geometric sweep of the
mesh -- it is somewhat inflated. That is usually what you want when explaining planner
behaviour, but it is not a precise geometric measurement, and the reported volume number
inherits the same padding.

Robot geometry only: the carried bottle is NOT included (the yml's attached_object spheres
are placeholders, radius 0.001), so the volume shows what the ARM sweeps, not the load.

Only the ACTIVE arm is stamped -- the task is single-armed, but which arm it picks varies
with the bottle's x, so both TCPs are sampled and whichever actually moved wins.

A rollout that fails is still plotted (volume as far as it got, last TCP point marked with
a red X). The SUCCESS/FAILED verdict goes in every title.

OUTPUT LAYOUT: every run writes to its OWN timestamped folder, so re-running never
overwrites an earlier run:

    <out-dir>/<YYYYmmdd-HHMMSS>/sweptvol_seed0001_SUCCESS_iso.png
                               sweptvol_seed0001.npz     (grid + landmarks, for re-render)
                               video/episode1.mp4

USAGE (from customized_robotwin, env sourced + ROBOTWIN_BENCH_TASK=bench):
    python script/bench_script/swept_volume_3d.py --seed 1
    python script/bench_script/swept_volume_3d.py --seed 1 --res 0.005      # finer grid
    python script/bench_script/swept_volume_3d.py --seed 1 --gripper-only   # links 6/7/8
    python script/bench_script/swept_volume_3d.py --seed 1 --no-video
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import transforms3d as t3d
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setup_paths import setup_paths
setup_paths()

from analyze_occluder_visibility import (make_occluder_task, PAD_XY,  # noqa: E402
                                         OCC_HALF_FOOTPRINT)
from analyze_natural_visibility import build_cfg, DR_CLEAN  # noqa: E402
from reachability_view import OCC_HEIGHT  # noqa: E402  (milk-box height, 0.2542 m)
# Landmark drawing / view angles / video+close handling are shared with the polyline script
# so the two figures stay visually comparable and the conventions live in one place.
from gripper_path_3d import VIEWS, _box_wireframe, _write_video  # noqa: E402

# Per-arm curobo collision model. The LEFT file defines both arms, the RIGHT file only fr_*,
# so each arm is read from its own canonical file rather than assuming one covers both.
SPHERE_YML = {"left": "collision_aloha_left.yml", "right": "collision_aloha_right.yml"}
LINK_PREFIX = {"left": "fl_", "right": "fr_"}
# ee_link is *_link6; 7/8 are the fingers. Used by --gripper-only.
GRIPPER_SUFFIXES = ("link6", "link7", "link8")


def _load_spheres(side, gripper_only):
    """{link_name: [(center(3,), radius), ...]} in LINK-LOCAL frames, for one arm.

    Drops `attached_object` (placeholder spheres for a held object -- we stamp robot
    geometry only) and the wrist camera link (not arm structure)."""
    path = (Path(os.environ["BENCH_ROOT"]) / "assets" / "embodiments" / "aloha-agilex"
            / SPHERE_YML[side])
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)["collision_spheres"]
    prefix = LINK_PREFIX[side]
    out = {}
    for link, spheres in raw.items():
        if not link.startswith(prefix):
            continue                      # other arm, attached_object, cameras
        if gripper_only and not link.endswith(GRIPPER_SUFFIXES):
            continue
        out[link] = [(np.asarray(s["center"], dtype=float), float(s["radius"]))
                     for s in spheres]
    if not out:
        raise SystemExit(f"no spheres found for {side} arm in {path}")
    return out


def _arm_links(env, side):
    """{link_name: sapien link} for one arm's links, keyed to match the sphere model."""
    ent = env.robot.left_entity if side == "left" else env.robot.right_entity
    prefix = LINK_PREFIX[side]
    return {l.get_name(): l for l in ent.get_links() if l.get_name().startswith(prefix)}


def _stamp_sphere(grid, origin, res, c, r):
    """OR one world-frame sphere into the occupancy grid, touching only the cells in its
    own index bounding box (so cost scales with the sphere, not the grid).

    Returns False if the sphere fell entirely outside the grid -- the caller counts these,
    because a clipped sphere is a silently MISSING chunk of swept volume (the arm's base
    links sit outside the default table-top bounds)."""
    lo = np.maximum(np.floor((c - r - origin) / res).astype(int), 0)
    hi = np.minimum(np.ceil((c + r - origin) / res).astype(int) + 1, np.array(grid.shape))
    if np.any(lo >= hi):
        return False                                  # sphere is outside the grid
    # cell CENTRES along each axis within the bbox
    ax = [origin[i] + (np.arange(lo[i], hi[i]) + 0.5) * res for i in range(3)]
    X, Y, Z = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")
    inside = ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) <= r * r
    grid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside
    return True


def _build_volume(poses, spheres, args):
    """Stamp every sampled link pose's spheres into one boolean grid.

    poses: {link_name: [(p(3,), q(4,)), ...]} world poses per sampled step.
    Returns (grid (nx,ny,nz) bool, origin (3,))."""
    origin = np.array([args.xmin, args.ymin, args.zmin], dtype=float)
    shape = tuple(int(np.ceil((hi - lo) / args.res)) for lo, hi in
                  ((args.xmin, args.xmax), (args.ymin, args.ymax), (args.zmin, args.zmax)))
    grid = np.zeros(shape, dtype=bool)
    n = 0
    clipped = {}
    for link, seq in poses.items():
        local = spheres.get(link)
        if not local:
            continue                                  # link has no spheres in the model
        for p, q in seq:
            R = t3d.quaternions.quat2mat(q)           # sapien q is (w,x,y,z), as t3d expects
            for c_local, r in local:
                if not _stamp_sphere(grid, origin, args.res, p + R @ c_local, r):
                    clipped[link] = clipped.get(link, 0) + 1
                n += 1
    print(f"[volume] stamped {n} spheres into a {shape[0]}x{shape[1]}x{shape[2]} grid "
          f"@ {args.res * 100:.1f} cm")
    # A clipped sphere is swept volume the figure does NOT show. Name the links so it is
    # obvious whether it's just the fixed base (harmless) or a moving link (widen --x/y/zmin).
    if clipped:
        tot = sum(clipped.values())
        detail = ", ".join(f"{k}={v}" for k, v in sorted(clipped.items()))
        print(f"[volume] WARNING: {tot}/{n} spheres ({100.0 * tot / n:.1f}%) fell OUTSIDE the "
              f"grid and are missing from the volume: {detail}")
        print(f"[volume]          grid bounds x[{args.xmin},{args.xmax}] "
              f"y[{args.ymin},{args.ymax}] z[{args.zmin},{args.zmax}] -- widen to include them")
    return grid, origin


def _plot_volume(seed, grid, origin, pts, ok_overall, box_p, tgt_p, pad_xy, arm_name,
                 run_dir, args):
    """Swept volume as a marching-cubes isosurface, with the TCP path drawn inside it."""
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    if grid.sum() == 0:
        print("isosurface skipped: empty volume"); return
    try:
        verts, faces, _, _ = measure.marching_cubes(grid.astype(np.float32), level=0.5,
                                                    spacing=(args.res,) * 3)
    except (ValueError, RuntimeError) as e:
        print(f"isosurface skipped: {e}"); return
    # grid is indexed (x,y,z), so verts columns are already (x,y,z) in index*spacing
    tri = (verts + origin)[faces]                     # (nfaces, 3, 3)

    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    tnorm = np.linspace(0.0, 1.0, len(segs))
    vol_l = float(grid.sum()) * (args.res ** 3) * 1000.0     # m^3 -> litres

    out = Path(run_dir)
    for tag, elev, azim in VIEWS:
        fig = plt.figure(figsize=(9.5, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.add_collection3d(Poly3DCollection(tri, alpha=0.18, facecolor="tab:blue",
                                             edgecolor="none"))
        lc = Line3DCollection(segs, cmap="viridis", norm=plt.Normalize(0, 1), lw=2.0)
        lc.set_array(tnorm)
        ax.add_collection3d(lc)

        _box_wireframe(ax, box_p, OCC_HALF_FOOTPRINT, OCC_HEIGHT)
        ax.scatter([tgt_p[0]], [tgt_p[1]], [tgt_p[2]], color="tab:blue", s=70, marker="o",
                   depthshade=False, ec="black", label="bottle start")
        ax.scatter([pad_xy[0]], [pad_xy[1]], [float(box_p[2])], color="magenta", s=80,
                   marker="s", depthshade=False, ec="black", label="pad (destination)")
        if not ok_overall:
            ax.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], color="red", s=110,
                       marker="X", depthshade=False, ec="black", label="rollout failed here")
        ax.plot([], [], [], color="tab:blue", lw=6, alpha=0.3,
                label=f"swept volume ({vol_l:.1f} L)")

        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_xlim(args.xmin, args.xmax); ax.set_ylim(args.ymin, args.ymax)
        ax.set_zlim(args.zmin, args.zmax)
        try:
            ax.set_box_aspect((args.xmax - args.xmin, args.ymax - args.ymin,
                               args.zmax - args.zmin))
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper left", fontsize=8)

        scope = "gripper (links 6/7/8)" if args.gripper_only else "full arm"
        ax.set_title(f"seed {seed}  |  {arm_name} {scope} swept volume  |  "
                     f"rollout {'SUCCESS' if ok_overall else 'FAILED'}\n"
                     f"space the PLANNER models the arm sweeping (padded CuRobo spheres, "
                     f"not exact mesh)\n"
                     f"occluder offset {args.offset:.2f} m, no clutter (density 0)  |  "
                     f"{args.res * 100:.1f} cm grid  |  view: {tag}")

        vtag = "SUCCESS" if ok_overall else "FAILED"
        p = out / f"sweptvol_seed{seed:04d}_{vtag}_{tag}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"saved {p}")


def run(args):
    # One folder per run, so re-running a seed never clobbers the previous run.
    run_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] writing to {run_dir}")

    env = make_occluder_task()()
    env.spawn_occluder = True
    env.occluder_offset = args.offset

    # rollout=True -> need_plan=True so play_once actually PLANS (the default measurement
    # build replays a joint-path cache nothing filled). save_path -> video/episode{seed}.mp4.
    cfg = build_cfg("put_mouse_on_pad", args.base_config, args.seed, DR_CLEAN,
                    rollout=True, ep_num=args.seed,
                    save_path=(str(run_dir) if args.save_video else None))
    if not args.save_video:
        cfg["save_data"] = False
    env.setup_demo(**cfg)

    box_p = np.array(env.occluder.get_pose().p)
    tgt_p = np.array(env.target_obj.get_pose().p)     # BEFORE the rollout moves the bottle
    print(f"[seed {args.seed}] box={np.round(box_p, 3)}  bottle={np.round(tgt_p, 3)}  "
          f"pad={PAD_XY}")

    # The active arm isn't known until the rollout has run, so sample BOTH arms' link poses
    # and both TCPs, then stamp only the arm that actually moved.
    links = {s: _arm_links(env, s) for s in ("left", "right")}
    poses = {s: {name: [] for name in links[s]} for s in ("left", "right")}
    tcps = {"left": [], "right": []}

    def hook(control_idx):
        if control_idx % args.stride:
            return
        for s in ("left", "right"):
            for name, link in links[s].items():
                lp = link.get_pose()
                poses[s][name].append((np.asarray(lp.p, dtype=float),
                                       np.asarray(lp.q, dtype=float)))
        tcps["left"].append(env.robot.get_left_tcp_pose()[:3])
        tcps["right"].append(env.robot.get_right_tcp_pose()[:3])

    env.step_hook = hook
    try:
        env.play_once()
        ok_overall = bool(getattr(env, "plan_success", False))
    except Exception as e:
        print(f"[seed {args.seed}] rollout raised ({type(e).__name__}: {e}); "
              f"plotting what it swept up to the failure")
        ok_overall = False
    env.step_hook = None

    L, R = np.array(tcps["left"], dtype=float), np.array(tcps["right"], dtype=float)
    if len(L) < 2:
        print(f"[seed {args.seed}] only {len(L)} sample(s) captured -- nothing to plot "
              f"(the rollout failed before any arm motion). Try another seed.")
        _write_video(env, args)
        return

    dl = float(np.linalg.norm(np.diff(L, axis=0), axis=1).sum())
    dr = float(np.linalg.norm(np.diff(R, axis=0), axis=1).sum())
    arm_name, pts = ("left", L) if dl >= dr else ("right", R)
    print(f"[seed {args.seed}] rollout {'SUCCESS' if ok_overall else 'FAILED'}; "
          f"{len(pts)} samples; path length left={dl:.3f} m right={dr:.3f} m "
          f"-> stamping {arm_name} arm")

    spheres = _load_spheres(arm_name, args.gripper_only)
    print(f"[volume] {arm_name} model: {len(spheres)} links, "
          f"{sum(len(v) for v in spheres.values())} spheres/step"
          f"{' (gripper only)' if args.gripper_only else ''}")
    grid, origin = _build_volume(poses[arm_name], spheres, args)
    vol_l = float(grid.sum()) * (args.res ** 3) * 1000.0
    print(f"[volume] occupied {int(grid.sum())} cells = {vol_l:.2f} L")

    # cache the grid so the figure can be re-rendered without paying for another rollout
    npz = Path(run_dir) / f"sweptvol_seed{args.seed:04d}.npz"
    np.savez_compressed(npz, grid=grid, origin=origin, res=args.res, pts=pts,
                        box_p=box_p, tgt_p=tgt_p, pad_xy=np.array(PAD_XY),
                        ok=ok_overall, arm=arm_name)
    print(f"saved {npz}")

    _plot_volume(args.seed, grid, origin, pts, ok_overall, box_p, tgt_p,
                 np.array(PAD_XY), arm_name, run_dir, args)
    _write_video(env, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed", type=int, default=1, help="single seed to roll out and plot")
    ap.add_argument("--offset", type=float, default=0.2,
                    help="occluder offset in front of the target (m)")
    ap.add_argument("--stride", type=int, default=5,
                    help="sample every Nth physics step (1 = every step)")
    ap.add_argument("--res", type=float, default=0.01,
                    help="voxel size (m). 0.005 gives a finer surface at ~8x the stamping cost.")
    ap.add_argument("--gripper-only", action="store_true",
                    help="stamp only the gripper links (6/7/8) instead of the whole arm")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.45)
    ap.add_argument("--ymax", type=float, default=0.45)
    ap.add_argument("--zmin", type=float, default=0.70, help="table top sits at ~0.742 m")
    ap.add_argument("--zmax", type=float, default=1.20)
    ap.add_argument("--out-dir", default="../scripts/validation/results/swept_volume_3d",
                    help="repo-root results location. Each run lands in its own "
                         "<out-dir>/<timestamp>/ subfolder.")
    ap.add_argument("--no-video", dest="save_video", action="store_false",
                    help="skip the rollout mp4 (faster: no frame capture)")
    ap.set_defaults(save_video=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
