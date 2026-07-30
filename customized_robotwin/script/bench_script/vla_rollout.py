#!/usr/bin/env python3
"""Run the pi05 policy directly on deterministic occluder-ring scenes.

This validation driver intentionally bypasses ``play_once`` and the expert
feasibility gate. It saves one video and one JSONL record per policy episode.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from setup_paths import setup_paths

setup_paths()
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

from lib.model_client import ModelClient
from lib.occluder_ring import (
    draw_ring_config,
    occluder_ring_xy,
    parse_count_choices,
    parse_offset_specs,
)
from lib.run_io import Timings, _Tee, _prune_empty_topdirs
from lib.scene_build import build_cfg, dr_measure
from lib.scene_constants import OCC_PAD_MIN_DIST, PAD_XY, TABLE_XLIM, TABLE_YLIM
from task.occluder_task import make_occluder_task
from envs.utils.create_actor import UnStableError

robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
policy_root = robotwin_root / "policy"
if str(policy_root) not in sys.path:
    sys.path.insert(0, str(policy_root))
os.chdir(robotwin_root)

POLICY_NAME = "pi05"
CHECKPOINT_ID = 30000
VIDEO_SIZE = "320x240"


def _load_instruction_pool() -> list[str]:
    path = Path(os.environ["BENCH_ROOT"]) / "bench_task_config" / "instruction_bank.json"
    bank = json.loads(path.read_text(encoding="utf-8"))
    pool = bank.get("put_mouse_on_pad", [])
    if not pool:
        raise RuntimeError(f"put_mouse_on_pad has no instructions in {path}")
    return [str(item) for item in pool]


def _start_ffmpeg(video_path: Path) -> subprocess.Popen:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            VIDEO_SIZE,
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )


def _safe_close_env(env) -> None:
    try:
        env.close_env()
    except Exception:
        pass


def _finish_ffmpeg(env, ffmpeg: subprocess.Popen | None) -> str | None:
    if ffmpeg is None:
        return None
    try:
        env._del_eval_video_ffmpeg()
        return None
    except Exception as exc:
        try:
            if ffmpeg.stdin and not ffmpeg.stdin.closed:
                ffmpeg.stdin.close()
            ffmpeg.wait(timeout=10)
        except Exception:
            try:
                ffmpeg.kill()
            except Exception:
                pass
        return f"ffmpeg finalize failed: {type(exc).__name__}: {exc}"


def _bucket_video(out_dir: Path, episode: int, success: bool) -> str | None:
    source = out_dir / "video" / f"episode{episode}.mp4"
    if not source.is_file() or source.stat().st_size == 0:
        return None
    destination = out_dir / ("success" if success else "fail") / "video" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    _prune_empty_topdirs(out_dir)
    return destination.relative_to(out_dir).as_posix()


def _instruction_for(seed: int, pinned: str | None, pool: list[str]) -> str:
    return pinned if pinned is not None else random.Random(int(seed)).choice(pool)


def _set_ring(env, *, show: bool, offset: float, count: int,
              angle0: float, radii: list[float]) -> None:
    env.spawn_occluder = show
    env.occluder_offset = offset
    env.num_occluders = count
    env.occluder_angle0 = angle0
    env.occluder_radii = list(radii)


def _pad_distance(env, radii: list[float], count: int, angle0: float) -> float:
    target_xy = np.asarray(env.target_obj.get_pose().p[:2], dtype=float)
    ring_xys = occluder_ring_xy(
        float(target_xy[0]),
        float(target_xy[1]),
        radii,
        count,
        angle0,
        xlim=TABLE_XLIM,
        ylim=TABLE_YLIM,
    )
    return min(
        (float(np.linalg.norm(np.asarray(xy) - np.asarray(PAD_XY))) for xy in ring_xys),
        default=float("inf"),
    )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "log").mkdir(parents=True, exist_ok=True)
    (out_dir / "video").mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    offset_specs = parse_offset_specs(args.offsets)
    count_choices = parse_count_choices(args.num_occluders)
    clutter_densities = [
        int(item.strip()) for item in str(args.clutter_densities).split(",") if item.strip()
    ]
    if not clutter_densities:
        raise SystemExit("--clutter-densities parsed to nothing")

    instruction_pool = _load_instruction_pool()
    policy_eval = importlib.import_module("pi05.deploy_policy").eval
    timings = Timings()
    env = make_occluder_task()()
    model: ModelClient | None = None
    episode = 0
    produced = 0
    draw = args.seed_start
    max_draws = args.num_seeds * 20 + 50

    print(
        f"seeds from {args.seed_start}, want {args.num_seeds} stable seed(s), "
        f"offsets={args.offsets}, n_occluders={count_choices}, "
        f"clutter_densities={clutter_densities}"
    )
    print(f"writing -> {records_path}")

    try:
        with timings.section("model_connect"):
            model = ModelClient(port=args.port, pi0_step=args.pi0_step)

        with records_path.open("w", encoding="utf-8") as records:
            while produced < args.num_seeds and (draw - args.seed_start) < max_draws:
                seed = draw
                draw += 1
                seed_episode_count = 0

                for spec in offset_specs:
                    offset = spec[2]
                    show = bool(
                        np.random.default_rng(
                            int(seed) * 1000 + int(round(offset * 100))
                        ).random()
                        >= args.no_occluder_prob
                    )
                    angle0, count, radii = draw_ring_config(
                        seed, spec, count_choices, args.random_ring_rotation
                    )

                    for clutter_density in clutter_densities:
                        instruction = _instruction_for(
                            seed, args.instruction, instruction_pool
                        )
                        log_path = out_dir / "log" / f"episode{episode}.log"
                        video_path = out_dir / "video" / f"episode{episode}.mp4"
                        episode_started = time.perf_counter()
                        setup_ok = False
                        ffmpeg = None
                        policy_error = None
                        ffmpeg_error = None
                        collision_metrics = None
                        steps_taken = 0
                        step_lim = args.max_steps
                        success = False

                        with log_path.open("w", encoding="utf-8") as log:
                            log.write(
                                f"# episode{episode} seed={seed} offset={offset:.3f} "
                                f"clutter_density={clutter_density} show={show} "
                                f"count={count if show else 0} "
                                f"radii={[round(r, 4) for r in radii] if show else []} "
                                f"angle0={angle0:.4f}\n"
                                f"# instruction: {instruction}\n\n"
                            )
                            log.flush()

                            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
                                 contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                                _set_ring(
                                    env,
                                    show=show,
                                    offset=offset,
                                    count=count,
                                    angle0=angle0,
                                    radii=radii,
                                )
                                try:
                                    with timings.section(f"episode{episode}_scene_setup"):
                                        cfg = build_cfg(
                                            "put_mouse_on_pad",
                                            args.base_config,
                                            seed,
                                            dr_measure(clutter_density),
                                            ep_num=episode,
                                            save_path=out_dir / "video",
                                            mode="policy",
                                        )
                                        env.setup_demo(**cfg)
                                    setup_ok = True
                                except UnStableError as exc:
                                    print(
                                        f"[seed {seed}] scene unstable "
                                        f"({type(exc).__name__}: {exc}); drawing another seed"
                                    )
                                except Exception as exc:
                                    print(
                                        f"[seed {seed}] fatal scene-build error "
                                        f"({type(exc).__name__}: {exc})"
                                    )
                                    raise

                                if setup_ok and show:
                                    pad_distance = _pad_distance(env, radii, count, angle0)
                                    if pad_distance < OCC_PAD_MIN_DIST:
                                        print(
                                            f"[seed {seed}] ring is {pad_distance:.3f}m from pad "
                                            f"(< {OCC_PAD_MIN_DIST:.3f}m); drawing another seed"
                                        )
                                        setup_ok = False

                                if setup_ok:
                                    if args.max_steps is not None:
                                        env.step_lim = args.max_steps
                                    step_lim = int(env.step_lim)
                                    env.set_instruction(instruction=instruction)
                                    try:
                                        ffmpeg = _start_ffmpeg(video_path)
                                        env._set_eval_video_ffmpeg(ffmpeg)
                                        # Use the dynamic proxy, not call(), so the
                                        # client-side observation-window mirror resets.
                                        model.reset_model()
                                        with timings.section(f"episode{episode}_policy"):
                                            while env.take_action_cnt < env.step_lim:
                                                observation = env.get_obs()
                                                policy_eval(env, model, observation)
                                                if env.eval_success:
                                                    break
                                        success = bool(env.check_success())
                                    except Exception as exc:
                                        policy_error = f"{type(exc).__name__}: {exc}"
                                        print(f"[episode {episode}] policy failed: {policy_error}")
                                        traceback.print_exc()
                                    finally:
                                        ffmpeg_error = _finish_ffmpeg(env, ffmpeg)

                                    steps_taken = int(env.take_action_cnt)
                                    try:
                                        collision_metrics = env.get_collision_metrics()
                                    except Exception:
                                        collision_metrics = None

                        if not setup_ok:
                            _safe_close_env(env)
                            continue

                        failure_reason = policy_error
                        if failure_reason is None and not success:
                            failure_reason = "step_limit_reached"
                        if ffmpeg_error:
                            failure_reason = (
                                f"{failure_reason}; {ffmpeg_error}"
                                if failure_reason
                                else ffmpeg_error
                            )

                        _safe_close_env(env)
                        video_relpath = _bucket_video(out_dir, episode, success)
                        if video_relpath is None:
                            video_missing = "video missing or empty"
                            failure_reason = (
                                f"{failure_reason}; {video_missing}"
                                if failure_reason
                                else video_missing
                            )

                        record = {
                            "episode": int(episode),
                            "seed": int(seed),
                            "offset": float(offset),
                            "num_occluders": int(count) if show else 0,
                            "occluder_radii": (
                                [round(float(radius), 4) for radius in radii] if show else []
                            ),
                            "occluder_angle0": (
                                round(float(angle0), 4) if show else None
                            ),
                            "occluder_shown": bool(show),
                            "clutter_density": int(clutter_density),
                            "instruction": instruction,
                            "success": bool(success),
                            "steps_taken": int(steps_taken),
                            "step_lim": int(step_lim),
                            "wall_seconds": float(time.perf_counter() - episode_started),
                            "video_relpath": video_relpath,
                            "failure_reason": failure_reason,
                            "policy": POLICY_NAME,
                            "checkpoint_id": CHECKPOINT_ID,
                            "collision_metrics": collision_metrics,
                        }
                        records.write(json.dumps(record) + "\n")
                        records.flush()
                        print(
                            f"[episode {episode}] seed={seed} "
                            f"{'SUCCESS' if success else 'FAIL'} steps={steps_taken}/{step_lim} "
                            f"video={video_relpath}"
                        )
                        episode += 1
                        seed_episode_count += 1

                        # The server closes a client socket after any RPC error.
                        # Reconnect before the next episode; the loaded model stays alive.
                        if policy_error is not None:
                            model.close()
                            with timings.section("model_reconnect"):
                                model = ModelClient(port=args.port, pi0_step=args.pi0_step)

                if seed_episode_count:
                    produced += 1
                    print(
                        f"[{produced}/{args.num_seeds}] seed {seed}: "
                        f"{seed_episode_count} episode(s) complete"
                    )

            if produced < args.num_seeds:
                print(
                    f"WARNING: only {produced}/{args.num_seeds} stable seeds after "
                    f"{draw - args.seed_start} draws (hit safety cap)"
                )
    finally:
        _safe_close_env(env)
        if model is not None:
            model.close()
        timings.save(out_dir)

    print(f"rollout validation ready -> {records_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="model-server RPC port")
    parser.add_argument("--pi0-step", type=int, default=50,
                        help="actions executed per inference chunk (1-50)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override the task's configured 600-step limit")
    parser.add_argument("--base-config", default="bench_demo_office_clean")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--offsets", default="0.2",
                        help="fixed/ranged ring radii, e.g. 0.2 or 0.10-0.25")
    parser.add_argument("--num-occluders", default="1",
                        help="fixed count or comma-separated count choices")
    parser.add_argument(
        "--random-ring-rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--clutter-densities", default="0")
    parser.add_argument("--no-occluder-prob", type=float, default=0.2)
    parser.add_argument("--instruction", default=None,
                        help="pin the instruction instead of sampling by seed")
    parser.add_argument(
        "--out-dir",
        default="../scripts/validation/results/vla_occluder_rollout",
    )
    parser.add_argument("--run-type", default="vla")
    args = parser.parse_args()

    if not 1 <= args.pi0_step <= 50:
        parser.error("--pi0-step must be between 1 and 50")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.num_seeds <= 0:
        parser.error("--num-seeds must be positive")
    if not 0.0 <= args.no_occluder_prob <= 1.0:
        parser.error("--no-occluder-prob must be between 0 and 1")

    args.out_dir = str(
        Path(args.out_dir)
        / args.run_type
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    print(f"[run] writing to {args.out_dir}")
    run(args)


if __name__ == "__main__":
    main()
