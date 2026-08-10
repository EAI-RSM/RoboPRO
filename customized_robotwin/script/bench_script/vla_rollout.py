#!/usr/bin/env python3
"""Run pi05 directly on stock RoboPRO tasks or deterministic occluder scenes.

This validation driver intentionally bypasses ``play_once`` and the expert
feasibility gate. It saves one video and one JSONL record per policy episode.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
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
from lib.run_io import (
    Timings,
    _Tee,
    _prune_empty_topdirs,
    append_jsonl_fsync,
    atomic_write_json,
)
from lib.scene_build import DR_CLEAN, build_cfg, dr_measure, get_env_class
from lib.scene_constants import OCC_PAD_MIN_DIST, PAD_XY, TABLE_XLIM, TABLE_YLIM
from lib.scene_provenance import (
    fingerprint,
    hash_files,
    task_scene_code_version,
    task_scene_identity,
)
from lib.task_roles import resolve_task_roles
from lib.vla_reporting import read_episode_records, sync_records_jsonl, write_rollout_reports
from lib.waypoints import canonical_waypoints
from envs.utils.create_actor import UnStableError

robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
policy_root = robotwin_root / "policy"
if str(policy_root) not in sys.path:
    sys.path.insert(0, str(policy_root))
os.chdir(robotwin_root)

POLICY_NAME = "pi05"
CHECKPOINT_ID = 30000
CHECKPOINT = "mzxuan/robopro_jax_30000"
VIDEO_SIZE = "320x240"
OFFICE_INSTRUCTION = "Put the mouse onto the mouse pad."
OCCLUDER_INSTRUCTION = (
    "Put the bottle onto the mouse pad."
    "The bottle must be placed upright on the pad near the center."
    "You will fail even if the bottle is on the pad but not close enough to the center."
    "Do not place the bottle on it's side or upside down."
)
RUN_CONFIG_SCHEMA = "robopro.vla-rollout-config.v1"
RESUME_FIELDS = (
    "scene",
    "task_name",
    "bench_subdir",
    "base_config",
    "seed_start",
    "num_seeds",
    "rollouts_per_density",
    "replicate",
    "offsets",
    "num_occluders",
    "random_ring_rotation",
    "pad_shift_y",
    "clutter_densities",
    "no_occluder_prob",
    "instruction",
    "max_steps",
    "pi0_step",
)


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


def _bucket_video(out_dir: Path, episode: int, hard_success: bool) -> str | None:
    source = out_dir / "video" / f"episode{episode}.mp4"
    if not source.is_file() or source.stat().st_size == 0:
        return None
    destination = out_dir / (
        "hard_success" if hard_success else "hard_fail"
    ) / "video" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream:
        os.fsync(stream.fileno())
    source.replace(destination)
    try:
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass
    _prune_empty_topdirs(out_dir)
    return destination.relative_to(out_dir).as_posix()


def _instruction_for(
    pinned: str | None,
    scene: str,
    task_name: str | None = None,
) -> str:
    if pinned is not None:
        return pinned
    if scene == "office":
        return OFFICE_INSTRUCTION
    if scene == "occluder":
        return OCCLUDER_INSTRUCTION
    if not task_name:
        raise ValueError("task mode requires a task name for instruction lookup")
    path = Path(os.environ["BENCH_ROOT"]) / "bench_task_config" / "instruction_bank.json"
    bank = json.loads(path.read_text(encoding="utf-8"))
    instructions = bank.get(task_name)
    if not isinstance(instructions, list) or not instructions:
        raise RuntimeError(f"{task_name!r} has no instructions in {path}")
    return str(instructions[0])


def _task_context(args: argparse.Namespace) -> tuple[str, str]:
    if args.scene == "task":
        return str(args.task_name), str(args.bench_subdir)
    return "put_mouse_on_pad", "office"


def _make_env(args: argparse.Namespace):
    task_name, bench_subdir = _task_context(args)
    if args.scene != "occluder":
        return get_env_class(task_name, bench_subdir=bench_subdir)()
    from task.occluder_task import make_occluder_task
    return make_occluder_task()()


def _clutter_densities(value) -> list[int]:
    densities = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not densities:
        raise SystemExit("--clutter-densities parsed to nothing")
    if any(density < 0 for density in densities):
        raise SystemExit("--clutter-densities cannot contain negative values")
    return densities


def _scene_sweep(args: argparse.Namespace):
    if args.scene == "office":
        return [None], [], [0]
    if args.scene == "task":
        return [None], [], _clutter_densities(args.clutter_densities)

    offset_specs = parse_offset_specs(args.offsets)
    count_choices = parse_count_choices(args.num_occluders)
    clutter_densities = _clutter_densities(args.clutter_densities)
    return offset_specs, count_choices, clutter_densities


def _scene_parameters(
    args: argparse.Namespace,
    seed: int,
    spec,
    count_choices: list[int],
):
    if args.scene != "occluder":
        return None, False, None, 0, []

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
    return offset, show, angle0, count, radii


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


def _hard_success(task_success: bool, collision_metrics) -> bool:
    if not isinstance(task_success, (bool, np.bool_)):
        raise TypeError("task success must be Boolean")
    if not isinstance(collision_metrics, dict):
        raise RuntimeError("collision metrics are required for hard success")
    if "is_collision" not in collision_metrics:
        raise RuntimeError("collision metrics are missing is_collision")
    is_collision = collision_metrics["is_collision"]
    if not isinstance(is_collision, (bool, np.bool_)):
        raise TypeError("collision_metrics.is_collision must be Boolean")
    return bool(task_success) and not bool(is_collision)


def _task_final_state(env):
    target_pose = env.target_obj.get_pose()
    destination_pose = env.des_obj.get_pose()
    target_xy = np.asarray(target_pose.p[:2], dtype=float)
    destination_xy = np.asarray(destination_pose.p[:2], dtype=float)
    return {
        "target_pose": {
            "p": [float(value) for value in np.asarray(target_pose.p, dtype=float)],
            "q": [float(value) for value in np.asarray(target_pose.q, dtype=float)],
        },
        "destination_pose": {
            "p": [float(value) for value in np.asarray(destination_pose.p, dtype=float)],
            "q": [float(value) for value in np.asarray(destination_pose.q, dtype=float)],
        },
        "xy_error_m": [float(value) for value in target_xy - destination_xy],
        "xy_l2_error_m": float(np.linalg.norm(target_xy - destination_xy)),
        "left_gripper_open": bool(env.robot.is_left_gripper_open()),
        "right_gripper_open": bool(env.robot.is_right_gripper_open()),
    }


def _target_rollouts(args, clutter_densities):
    if args.scene != "task":
        return int(args.num_seeds)
    per_density = (
        int(args.rollouts_per_density)
        if args.rollouts_per_density is not None
        else int(args.num_seeds)
    )
    return per_density * len(clutter_densities)


def _task_density(clutter_densities, completed_rollouts):
    return int(clutter_densities[int(completed_rollouts) % len(clutter_densities)])


def _rollout_code_version():
    return hash_files(
        [
            Path(__file__),
            robotwin_root / "script" / "bench_script" / "lib" / "scene_build.py",
            robotwin_root / "policy" / "pi05" / "deploy_policy.py",
            Path(os.environ["BENCH_ROOT"]) / "bench_envs" / "_bench_base_task.py",
        ]
    )


def _run_config(args, clutter_densities, scene_code_version, rollout_code_version=None):
    arguments = {name: getattr(args, name) for name in RESUME_FIELDS}
    config = {
        "schema": RUN_CONFIG_SCHEMA,
        "arguments": arguments,
        "density_cycle": [int(value) for value in clutter_densities],
        "rollouts_per_density": (
            int(args.rollouts_per_density)
            if args.rollouts_per_density is not None
            else int(args.num_seeds)
        ) if args.scene == "task" else None,
        "target_rollouts": _target_rollouts(args, clutter_densities),
        "schedule": (
            "round_robin_density_with_new_seed_per_rollout"
            if args.scene == "task"
            else "legacy_scene_sweep"
        ),
        "policy": POLICY_NAME,
        "checkpoint_id": CHECKPOINT_ID,
        "checkpoint": CHECKPOINT,
        "scene_code_version": scene_code_version,
        "rollout_code_version": (
            rollout_code_version
            if rollout_code_version is not None
            else _rollout_code_version()
        ),
        "video_camera": "countertop_camera",
        "report_every": int(args.report_every),
    }
    config["config_sha256"] = fingerprint(config)
    return config


def _restore_run(args, out_dir):
    if not args.resume_dir:
        return None, []
    config_path = out_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"resume directory lacks config.json: {out_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != RUN_CONFIG_SCHEMA:
        raise ValueError(f"unsupported resume config schema in {config_path}")
    stored_hash = config.get("config_sha256")
    unhashed = {key: value for key, value in config.items() if key != "config_sha256"}
    if stored_hash != fingerprint(unhashed):
        raise ValueError(f"resume config hash mismatch: {config_path}")
    for name in RESUME_FIELDS:
        setattr(args, name, config["arguments"][name])
    args.report_every = int(config["report_every"])
    if args.scene != "task":
        raise ValueError("crash-safe resume is currently limited to --scene task")
    records = read_episode_records(out_dir)
    cycle = [int(value) for value in config["density_cycle"]]
    if len(records) > int(config["target_rollouts"]):
        raise ValueError("resume run contains more records than its declared target")
    for index, record in enumerate(records):
        expected_density = cycle[index % len(cycle)]
        if int(record["clutter_density"]) != expected_density:
            raise ValueError(
                f"resume density order breaks at episode {index}: "
                f"expected d{expected_density}, got d{record['clutter_density']}"
            )
        if record.get("run_config_sha256") != stored_hash:
            raise ValueError(f"episode {index} does not match the resume config hash")
    sync_records_jsonl(out_dir, records)
    return config, records


def _commit_record(out_dir, record):
    episode = int(record["episode"])
    episode_path = Path(out_dir) / "episodes" / f"episode{episode:06d}.json"
    if episode_path.exists():
        raise FileExistsError(f"refusing to overwrite committed record {episode_path}")
    atomic_write_json(episode_path, record)
    append_jsonl_fsync(Path(out_dir) / "records.jsonl", record)


def _regenerate_reports(out_dir):
    try:
        summary = write_rollout_reports(out_dir)
        print(
            f"[report] regenerated at n={summary['n_episodes']} "
            f"HSR={summary['hard_success_rate']}"
        )
        return summary
    except Exception as exc:
        print(f"[report] WARNING: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


def _run_metric_postprocess(args, out_dir):
    if not args.postprocess_metrics:
        return
    if args.scene != "task":
        raise ValueError("--postprocess-metrics is only valid with --scene task")
    command = [
        sys.executable,
        str(Path(__file__).with_name("task_metric.py")),
        "--rollout-run",
        str(Path(out_dir).resolve()),
        "--report-every",
        str(int(args.report_every)),
        "--no-scene-images",
    ]
    print("[postprocess] starting/resuming integrated task metrics")
    subprocess.run(command, check=True)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "log").mkdir(parents=True, exist_ok=True)
    (out_dir / "video").mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    stored_config, existing_records = _restore_run(args, out_dir)
    offset_specs, count_choices, clutter_densities = _scene_sweep(args)
    if len(clutter_densities) != len(set(clutter_densities)):
        raise ValueError("clutter density cycle cannot contain duplicates")
    scene_code_version = (
        task_scene_code_version(args.base_config) if args.scene == "task" else None
    )
    config = _run_config(
        args, clutter_densities, scene_code_version, _rollout_code_version()
    )
    if stored_config is None:
        atomic_write_json(out_dir / "config.json", config)
    elif config != stored_config:
        raise ValueError("restored rollout configuration does not match config.json")
    run_config_sha256 = config["config_sha256"]
    target_rollouts = int(config["target_rollouts"])
    run_instance_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    if len(existing_records) == target_rollouts:
        summary = _regenerate_reports(out_dir)
        if summary is None or not summary.get("collection_complete"):
            raise RuntimeError("completed rollout records did not regenerate a complete summary")
        print(f"[resume] rollout collection already complete at n={target_rollouts}")
        _run_metric_postprocess(args, out_dir)
        return

    policy_eval = importlib.import_module("pi05.deploy_policy").eval
    timings = Timings()
    task_name, bench_subdir = _task_context(args)
    env = _make_env(args)
    model: ModelClient | None = None
    episode = len(existing_records)
    produced = len(existing_records) if args.scene == "task" else 0
    draw = (
        max(int(record["seed"]) for record in existing_records) + 1
        if existing_records
        else int(args.seed_start)
    )
    max_draws = target_rollouts * 20 + 50

    if args.scene == "office":
        print(
            f"default office scene: seeds from {args.seed_start}, "
            f"want {args.num_seeds} stable seed(s), no clutter or occluders"
        )
    elif args.scene == "task":
        print(
            f"stock task: {bench_subdir}/{task_name}, seeds from {args.seed_start}, "
            f"want {config['rollouts_per_density']} rollout(s) per density "
            f"({target_rollouts} total), density_cycle={clutter_densities}"
        )
    else:
        print(
            f"occluder scene: seeds from {args.seed_start}, "
            f"want {args.num_seeds} stable seed(s), offsets={args.offsets}, "
            f"n_occluders={count_choices}, clutter_densities={clutter_densities}"
        )
    print(f"writing -> {records_path}")
    if existing_records:
        print(
            f"[resume] {len(existing_records)}/{target_rollouts} committed; "
            f"continuing with episode {episode}, seed {draw}"
        )

    try:
        with timings.section("model_connect"):
            model = ModelClient(port=args.port, pi0_step=args.pi0_step)

        with contextlib.nullcontext():
            while produced < target_rollouts and (draw - args.seed_start) < max_draws:
                seed = draw
                draw += 1
                seed_episode_count = 0

                for spec in offset_specs:
                    offset, show, angle0, count, radii = _scene_parameters(
                        args, seed, spec, count_choices
                    )

                    episode_densities = (
                        [_task_density(clutter_densities, produced)]
                        if args.scene == "task"
                        else clutter_densities
                    )
                    for clutter_density in episode_densities:
                        instruction = _instruction_for(
                            args.instruction, args.scene, task_name
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
                        identity = None
                        final_state = None
                        scene_setup_seconds = None
                        policy_seconds = None

                        with log_path.open("w", encoding="utf-8") as log:
                            if args.scene == "office":
                                log.write(
                                    f"# episode{episode} seed={seed} scene=office "
                                    "clutter_density=0 occluders=0 "
                                    f"# instruction: {instruction}\n\n"
                                )
                            elif args.scene == "occluder":
                                log.write(
                                    f"# episode{episode} seed={seed} scene=occluder "
                                    f"offset={offset:.3f} "
                                    f"clutter_density={clutter_density} show={show} "
                                    f"count={count if show else 0} "
                                    f"radii="
                                    f"{[round(r, 4) for r in radii] if show else []} "
                                    f"angle0={angle0:.4f}\n"
                                    f"# instruction: {instruction}\n\n"
                                )
                            else:
                                log.write(
                                    f"# episode{episode} seed={seed} scene=task "
                                    f"task={task_name} bench_subdir={bench_subdir} "
                                    f"base_config={args.base_config} "
                                    f"clutter_density={clutter_density}\n"
                                    f"# instruction: {instruction}\n\n"
                                )
                            log.flush()

                            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
                                 contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                                if args.scene == "occluder":
                                    # load_actors reads fixed_pad_xy, so the pad spawns
                                    # (and registers its prohibited area) at the shifted y.
                                    env.fixed_pad_xy = (
                                        PAD_XY[0], PAD_XY[1] + args.pad_shift_y
                                    )
                                    _set_ring(
                                        env,
                                        show=show,
                                        offset=offset,
                                        count=count,
                                        angle0=angle0,
                                        radii=radii,
                                    )
                                try:
                                    scene_setup_started = time.perf_counter()
                                    with timings.section(f"episode{episode}_scene_setup"):
                                        cfg = build_cfg(
                                            task_name,
                                            args.base_config,
                                            seed,
                                            dr_measure(clutter_density),
                                            ep_num=episode,
                                            save_path=out_dir / "video",
                                            mode="policy",
                                            eval_video_camera="countertop_camera",
                                        )
                                        if args.scene == "office":
                                            cfg["domain_randomization"].update(DR_CLEAN)
                                        if not cfg.get("enable_collision_metrics", False):
                                            raise RuntimeError(
                                                f"{args.base_config} must enable collision metrics"
                                            )
                                        env.setup_demo(**cfg)
                                        if (
                                            getattr(env, "eval_video_camera", None)
                                            != "countertop_camera"
                                        ):
                                            raise RuntimeError(
                                                "task environment did not retain the requested "
                                                "countertop_camera video view"
                                            )
                                        if args.scene == "task":
                                            roles = resolve_task_roles(env, task_name)
                                            waypoints = canonical_waypoints(env, task_name)
                                            identity = task_scene_identity(
                                                env,
                                                task=task_name,
                                                seed=seed,
                                                replicate=args.replicate,
                                                bench_subdir=bench_subdir,
                                                base_config=args.base_config,
                                                dr_settings=cfg["domain_randomization"],
                                                checkpoint=CHECKPOINT,
                                                scene_code_version=scene_code_version,
                                                roles=roles,
                                                acting_arm=waypoints[0].arm,
                                                instruction=instruction,
                                            )
                                    scene_setup_seconds = (
                                        time.perf_counter() - scene_setup_started
                                    )
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

                                if setup_ok and args.scene == "occluder" and show:
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
                                        policy_started = time.perf_counter()
                                        try:
                                            with timings.section(f"episode{episode}_policy"):
                                                while env.take_action_cnt < env.step_lim:
                                                    observation = env.get_obs()
                                                    policy_eval(env, model, observation)
                                                    if env.eval_success:
                                                        break
                                        finally:
                                            policy_seconds = (
                                                time.perf_counter() - policy_started
                                            )
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
                                    except Exception as exc:
                                        raise RuntimeError(
                                            "collision metrics are required for hard success"
                                        ) from exc
                                    if args.scene == "task":
                                        final_state = _task_final_state(env)

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

                        hard_success = _hard_success(success, collision_metrics)
                        _safe_close_env(env)
                        video_relpath = _bucket_video(
                            out_dir, episode, hard_success
                        )
                        if video_relpath is None:
                            video_missing = "video missing or empty"
                            failure_reason = (
                                f"{failure_reason}; {video_missing}"
                                if failure_reason
                                else video_missing
                            )

                        record = {
                            "schema": "robopro.vla-rollout.v3",
                            "episode": int(episode),
                            "record_id": (
                                f"rollout-{run_config_sha256[:12]}-{episode:06d}"
                            ),
                            "run_instance_id": run_instance_id,
                            "run_config_sha256": run_config_sha256,
                            "seed": int(seed),
                            "replicate": int(args.replicate),
                            "scene": args.scene,
                            "task": task_name,
                            "bench_subdir": bench_subdir,
                            "base_config": args.base_config,
                            "dr_settings": cfg["domain_randomization"],
                            "scene_id": (
                                identity["scene_id"] if identity is not None else None
                            ),
                            "scene_fingerprint": (
                                identity["scene_fingerprint"]
                                if identity is not None else None
                            ),
                            "scene_fingerprint_source": (
                                identity["scene_fingerprint_source"]
                                if identity is not None else None
                            ),
                            "scene_code_version": scene_code_version,
                            "rollout_code_version": config["rollout_code_version"],
                            "acting_arm": (
                                identity["acting_arm"] if identity is not None else None
                            ),
                            "offset": float(offset) if offset is not None else None,
                            "num_occluders": int(count) if show else 0,
                            "occluder_radii": (
                                [round(float(radius), 4) for radius in radii] if show else []
                            ),
                            "occluder_angle0": (
                                round(float(angle0), 4) if show else None
                            ),
                            "occluder_shown": bool(show),
                            "clutter_density": int(clutter_density),
                            "density_cycle_index": (
                                int(episode % len(clutter_densities))
                                if args.scene == "task" else None
                            ),
                            "clutter_count": (
                                identity["clutter_count"]
                                if identity is not None
                                else len(getattr(env, "cluttered_objs", []) or [])
                            ),
                            "instruction": instruction,
                            "instruction_source": (
                                "cli" if args.instruction is not None
                                else "instruction_bank[0]" if args.scene == "task"
                                else "scene_default"
                            ),
                            "task_success": bool(success),
                            "success": bool(success),
                            "hard_success": hard_success,
                            "steps_taken": int(steps_taken),
                            "step_lim": int(step_lim),
                            "wall_seconds": float(time.perf_counter() - episode_started),
                            "timing_seconds": {
                                "scene_setup": scene_setup_seconds,
                                "policy": policy_seconds,
                            },
                            "video_relpath": video_relpath,
                            "video_camera": "countertop_camera",
                            "policy_error": policy_error,
                            "failure_reason": failure_reason,
                            "policy": POLICY_NAME,
                            "checkpoint_id": CHECKPOINT_ID,
                            "checkpoint": CHECKPOINT,
                            "collision_metrics": collision_metrics,
                            "final_state": final_state,
                        }
                        _commit_record(out_dir, record)
                        print(
                            f"[episode {episode}] seed={seed} "
                            f"{'SUCCESS' if success else 'FAIL'} steps={steps_taken}/{step_lim} "
                            f"video={video_relpath}"
                        )
                        episode += 1
                        seed_episode_count += 1

                        if episode % args.report_every == 0:
                            _regenerate_reports(out_dir)

                        # The server closes a client socket after any RPC error.
                        # Reconnect before the next episode; the loaded model stays alive.
                        if policy_error is not None:
                            model.close()
                            with timings.section("model_reconnect"):
                                model = ModelClient(port=args.port, pi0_step=args.pi0_step)

                if seed_episode_count:
                    produced += (
                        seed_episode_count if args.scene == "task" else 1
                    )
                    print(
                        f"[{produced}/{target_rollouts}] seed {seed}: "
                        f"{seed_episode_count} episode(s) complete"
                    )

            if produced < target_rollouts:
                print(
                    f"WARNING: only {produced}/{target_rollouts} completed units after "
                    f"{draw - args.seed_start} draws (hit safety cap)"
                )
    finally:
        _safe_close_env(env)
        if model is not None:
            model.close()
        timings.save(out_dir, filename=f"timings_session_{run_instance_id}.json")
        summary = _regenerate_reports(out_dir)

    if summary is not None:
        print(
            f"hard success: {summary['n_hard_success']}/{summary['n_episodes']} "
            f"(non_degenerate={summary['hard_success_non_degenerate']})"
        )
    if summary is not None and summary.get("collection_complete"):
        _run_metric_postprocess(args, out_dir)
    print(f"rollout validation ready -> {records_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="model-server RPC port")
    parser.add_argument("--pi0-step", type=int, default=50,
                        help="actions executed per inference chunk (1-50)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override the task's configured 600-step limit")
    parser.add_argument(
        "--scene",
        choices=("office", "occluder", "task"),
        default="occluder",
        help="office control, custom bottle/occluder scene, or a stock benchmark task",
    )
    parser.add_argument("--task-name", default="put_cup_on_coaster")
    parser.add_argument(
        "--bench-subdir",
        choices=("study", "office", "kitchenl", "kitchens"),
        default="study",
    )
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument(
        "--rollouts-per-density",
        type=int,
        default=None,
        help="stock-task rollout count for each density; densities run round-robin "
             "in the supplied order (preferred for the long association run)",
    )
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--offsets", default="0.2",
                        help="fixed/ranged ring radii, e.g. 0.2 or 0.10-0.25")
    parser.add_argument("--num-occluders", default="1",
                        help="fixed count or comma-separated count choices")
    parser.add_argument(
        "--random-ring-rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pad-shift-y", type=float, default=0.0,
        help="move the occluder-scene pad this far in +y, away from the robot "
             "(0.0 = stock pad at y=-0.28)",
    )
    parser.add_argument("--clutter-densities", default="0")
    parser.add_argument("--no-occluder-prob", type=float, default=0.0)
    parser.add_argument("--instruction", default=None,
                        help="override the scene's default instruction")
    parser.add_argument(
        "--out-dir",
        default="../scripts/validation/results/vla_occluder_rollout",
    )
    parser.add_argument("--run-type", default="vla")
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="resume an interrupted stock-task run from its exact timestamped directory",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=10,
        help="atomically regenerate CSV, summary, and figures every N committed rollouts",
    )
    parser.add_argument(
        "--postprocess-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after stock-task rollout teardown, automatically start/resume crash-safe "
            "metric processing and clearance-bucket reports"
        ),
    )
    args = parser.parse_args()

    if not 1 <= args.pi0_step <= 50:
        parser.error("--pi0-step must be between 1 and 50")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not args.resume_dir and args.scene == "task" and args.max_steps is None:
        parser.error("--max-steps is required for stock-task rollouts")
    if args.num_seeds <= 0:
        parser.error("--num-seeds must be positive")
    if args.rollouts_per_density is not None and args.rollouts_per_density <= 0:
        parser.error("--rollouts-per-density must be positive")
    if (
        not args.resume_dir
        and args.rollouts_per_density is not None
        and args.scene != "task"
    ):
        parser.error("--rollouts-per-density is only valid with --scene task")
    if not 0.0 <= args.no_occluder_prob <= 1.0:
        parser.error("--no-occluder-prob must be between 0 and 1")
    if args.replicate < 0:
        parser.error("--replicate cannot be negative")
    if args.report_every <= 0:
        parser.error("--report-every must be positive")
    if args.postprocess_metrics and args.scene != "task":
        parser.error("--postprocess-metrics is only valid with --scene task")
    if args.base_config is None:
        args.base_config = (
            f"bench_demo_{args.bench_subdir}_clean"
            if args.scene == "task"
            else "bench_demo_office_clean"
        )

    if args.resume_dir:
        args.out_dir = str(Path(args.resume_dir).resolve())
        print(f"[run] resuming {args.out_dir}")
    else:
        args.out_dir = str(
            Path(args.out_dir)
            / args.run_type
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        print(f"[run] writing to {args.out_dir}")
    run(args)


if __name__ == "__main__":
    main()
