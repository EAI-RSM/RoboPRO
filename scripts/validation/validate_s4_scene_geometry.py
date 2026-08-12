#!/usr/bin/env python3
"""Build one live ring scene and verify the requested collision-mesh gap."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOMIZED_ROOT = REPO_ROOT / "customized_robotwin"
BENCH_ROOT = REPO_ROOT / "benchmark"
BENCH_SCRIPT_ROOT = CUSTOMIZED_ROOT / "script/bench_script"
if str(CUSTOMIZED_ROOT) not in sys.path:
    sys.path.insert(0, str(CUSTOMIZED_ROOT))
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
if str(BENCH_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPT_ROOT))

os.environ.setdefault("ROBOTWIN_ROOT", str(CUSTOMIZED_ROOT))
os.environ.setdefault("BENCH_ROOT", str(BENCH_ROOT))
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

if importlib.util.find_spec("bench_envs.office.put_mouse_on_pad") is None:
    raise RuntimeError(
        f"cannot resolve bench_envs.office.put_mouse_on_pad under {BENCH_ROOT}"
    )

from lib.obstacles import _load_collision_mesh  # noqa: E402
from lib.run_io import Timings, atomic_write_json  # noqa: E402
from lib.scene_build import DR_CLEAN, build_cfg  # noqa: E402
from lib.visibility import CAMERA, _resolve_target, save_overlay  # noqa: E402
from task.occluder_task import make_occluder_task  # noqa: E402


def _posed_collision_vertices(info):
    loaded = _load_collision_mesh(info["collision_path"])
    if loaded is None:
        raise RuntimeError(f"could not load collision mesh: {info['collision_path']}")
    vertices, _faces = loaded
    actor = info["actor"]
    pose = info.get("pose") or actor.get_pose()

    import transforms3d as t3d

    rotation = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=float))
    try:
        scale = np.asarray(actor.scale, dtype=float)
    except Exception:
        scale = 1.0
    return (vertices * scale) @ rotation.T + np.asarray(pose.p, dtype=float)


def _xy_hull(vertices):
    from scipy.spatial import ConvexHull
    from shapely.geometry import Polygon

    xy = np.asarray(vertices[:, :2], dtype=float)
    return Polygon(xy[ConvexHull(xy).vertices])


def _collision_info(env, actor):
    for info in env.collision_list:
        if info.get("actor") is actor:
            return info
    raise RuntimeError(f"actor {actor.get_name()!r} is absent from collision_list")


def run(args):
    run_dir = (
        Path(args.out_dir).resolve()
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"writing -> {run_dir}")

    timings = Timings()
    env = make_occluder_task()()
    env.spawn_occluder = True
    env.occluder_offset = float(args.gap)
    env.occluder_radii = [float(args.gap)]
    env.num_occluders = 1
    env.occluder_angle0 = 0.0

    try:
        with timings.section("scene_setup"):
            os.chdir(CUSTOMIZED_ROOT)
            env.setup_demo(
                **build_cfg(
                    "put_mouse_on_pad",
                    args.base_config,
                    args.seed,
                    DR_CLEAN,
                )
            )
        if len(env.occluders) != 1:
            raise RuntimeError(f"expected one live occluder, found {len(env.occluders)}")

        with timings.section("collision_mesh_gap"):
            target = _resolve_target(env)
            target_path = (
                Path(os.environ["BENCH_ROOT"])
                / "assets/objects"
                / env.target_model
                / "collision"
                / f"base{env.target_id}.glb"
            )
            target_vertices = _posed_collision_vertices(
                {"actor": target, "collision_path": str(target_path)}
            )
            occluder = env.occluders[0]
            occluder_vertices = _posed_collision_vertices(
                _collision_info(env, occluder)
            )
            target_hull = _xy_hull(target_vertices)
            occluder_hull = _xy_hull(occluder_vertices)
            measured_gap = float(target_hull.distance(occluder_hull))
            z_overlap = float(
                min(target_vertices[:, 2].max(), occluder_vertices[:, 2].max())
                - max(target_vertices[:, 2].min(), occluder_vertices[:, 2].min())
            )
            error = abs(measured_gap - args.gap)

        with timings.section("initialized_scene_image"):
            visibility = env.measure_target_visibility(target, camera_name=CAMERA)
            save_overlay(
                env,
                visibility["mask"],
                run_dir / "initialized_scene.png",
                (
                    f"S4 geometry seed={args.seed} requested={args.gap:.3f}m "
                    f"measured={measured_gap:.3f}m visible_px="
                    f"{visibility['visible_pixel_count']}"
                ),
            )

        passed = z_overlap > 0.0 and error <= args.tolerance
        result = {
            "seed": int(args.seed),
            "requested_gap_m": float(args.gap),
            "measured_xy_collision_hull_gap_m": measured_gap,
            "absolute_error_m": error,
            "tolerance_m": float(args.tolerance),
            "target_occluder_z_overlap_m": z_overlap,
            "target_center_world": np.asarray(target.actor.get_pose().p).tolist(),
            "occluder_center_world": np.asarray(occluder.get_pose().p).tolist(),
            "visibility_target_ids": visibility["target_ids"],
            "visibility_pixel_count": visibility["visible_pixel_count"],
            "passed": passed,
        }
        atomic_write_json(run_dir / "geometry_result.json", result)
        print(
            f"requested={args.gap:.6f}m measured={measured_gap:.6f}m "
            f"error={error:.6f}m tolerance={args.tolerance:.6f}m "
            f"z_overlap={z_overlap:.6f}m"
        )
        if not passed:
            raise RuntimeError(
                "built-scene collision geometry did not reproduce the requested gap; "
                f"see {run_dir / 'geometry_result.json'}"
            )
    finally:
        try:
            env.close_env()
        finally:
            timings.save(run_dir)

    print(f"S4 scene geometry: PASS -> {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gap", type=float, default=0.10)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--base-config", default="bench_demo_office_clean")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "scripts/validation/results/s4_make_it_run/geometry"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
