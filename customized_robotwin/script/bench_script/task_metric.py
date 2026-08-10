#!/usr/bin/env python3
"""Generate planning-free geometric metric records for put_cup_on_coaster scenes.

This is the Stage C metric producer.  It builds stock Study scenes without an expert or policy,
computes every canonical leg in one geometric_eps call, and writes no outcome/HSR fields.
"""

import argparse
import gc
import json
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from setup_paths import setup_paths

setup_paths()
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

from lib.geometric_metric import geometric_eps
from lib.metric_config import SeedMetricConfig
from analyze_metric_correlation import (
    DEFAULT_BUCKET_SPEC,
    write_correlation_reports,
)
from analyze_metric_distribution import write_distribution_reports
from lib.run_io import (
    CLEARANCE_RESULTS_DIR,
    Timings,
    append_jsonl_fsync,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from lib.scene_build import build_cfg, get_env_class
from lib.scene_provenance import (
    fingerprint,
    hash_files,
    task_scene_code_version,
    task_scene_identity,
)
from lib.task_roles import SUPPORTED_TASK, resolve_task_roles
from lib.waypoints import canonical_legs, canonical_waypoints


RESULTS_DIR = CLEARANCE_RESULTS_DIR.parent / "task_metric"
REACH_CACHE_DIR = CLEARANCE_RESULTS_DIR / "_reach_cache_geometric_stage1"
DEFAULT_CHECKPOINT = "mzxuan/robopro_jax_30000"
DEFAULT_SCENE_CAMERAS = ("countertop_camera", "demo_camera", "demo_camera_2")
SCENE_MANIFEST_SCHEMA = "robopro.metric-scene-manifest.v1"
POSTPROCESS_CONFIG_SCHEMA = "robopro.task-metric-postprocess-config.v1"


def _json_value(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _save_scene_images(env, out_dir, seed, camera_names):
    """Render the initialized scene once and save the requested static camera views."""
    available = list(getattr(env.cameras, "static_camera_name", None) or [])
    missing = [name for name in camera_names if name not in available]
    if missing:
        raise ValueError(
            f"scene camera(s) {missing} are unavailable; available static cameras: {available}"
        )
    env._update_render()
    env.cameras.update_picture(camera_names=list(camera_names))
    rgb = env.cameras.get_rgb(camera_names=list(camera_names))
    image_dir = Path(out_dir) / "scene_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in camera_names:
        image = np.asarray(rgb[name]["rgb"])
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        path = image_dir / f"seed{int(seed):04d}_{name}.png"
        Image.fromarray(image).save(path)
        paths.append(str(path.relative_to(out_dir)))
        print(f"[scene] saved {path}")
    return paths


def _metric_record(env, args, job, scene_cfg, roles, waypoints, leg_specs, results,
                   code_version, wall_seconds, scene_images):
    if len(leg_specs) != len(results):
        raise RuntimeError("geometric metric returned the wrong number of legs")
    if results and len({result.n_free for result in results}) != 1:
        raise RuntimeError("canonical legs did not share one geometric volume")

    per_leg = []
    raw_values = []
    for index, (leg, result) in enumerate(zip(leg_specs, results)):
        eps = float(result.eps_star)
        raw_values.append(eps)
        rho = eps / args.gripper_r
        per_leg.append(
            {
                "index": index,
                "kind": leg.kind,
                "gripper_state": leg.gripper_state,
                "start_xyz": list(leg.start_xyz),
                "goal_xyz": list(leg.goal_xyz),
                "eps_geom": _json_value(eps),
                "eps_geom_unbounded": bool(np.isposinf(eps)),
                "rho_geom": _json_value(rho),
                "rho_geom_unbounded": bool(np.isposinf(rho)),
                "merged": bool(result.merged),
                "reason": result.reason,
                "bottleneck_xyz": (
                    None if result.bottleneck_xyz is None
                    else [float(v) for v in result.bottleneck_xyz]
                ),
            }
        )
    eps_min = min(raw_values)

    identity = task_scene_identity(
        env,
        task=job["task"],
        seed=job["seed"],
        replicate=job["replicate"],
        bench_subdir=job["bench_subdir"],
        base_config=job["base_config"],
        dr_settings=scene_cfg["domain_randomization"],
        checkpoint=job["checkpoint"],
        scene_code_version=code_version,
        roles=roles,
        acting_arm=waypoints[0].arm,
        instruction=job["instruction"],
    )
    return {
        "schema": "robopro.task-metric.v1",
        "status": "ok",
        "scene_id": identity["scene_id"],
        "scene_fingerprint": identity["scene_fingerprint"],
        "scene_fingerprint_source": identity["scene_fingerprint_source"],
        "rollout_episode": job["rollout_episode"],
        "task": job["task"],
        "seed": int(job["seed"]),
        "replicate": int(job["replicate"]),
        "bench_subdir": job["bench_subdir"],
        "base_config": job["base_config"],
        "dr_settings": scene_cfg["domain_randomization"],
        "obstacle_density": identity["obstacle_density"],
        "clutter_count": identity["clutter_count"],
        "clutter": identity["clutter"],
        "roles": {
            "target": identity["target"],
            "destination": identity["destination"],
        },
        "instruction": job["instruction"],
        "checkpoint": job["checkpoint"],
        "scene_code_version": code_version,
        "arm": waypoints[0].arm,
        "scene_images": list(scene_images),
        "gripper_reference_radius_m": float(args.gripper_r),
        "legs": per_leg,
        "eps_geom_min": _json_value(eps_min),
        "eps_geom_min_unbounded": bool(np.isposinf(eps_min)),
        "n_free": int(results[0].n_free),
        "wall_seconds": float(wall_seconds),
    }


def _dr_overrides(args):
    if args.obstacle_density is None:
        return {}
    return {
        "cluttered_table": True,
        "obstacle_density": int(args.obstacle_density),
        "clean_background_rate": 0,
    }




def _default_instruction(task_name):
    path = Path(os.environ["BENCH_ROOT"]) / "bench_task_config" / "instruction_bank.json"
    bank = json.loads(path.read_text(encoding="utf-8"))
    instructions = bank.get(task_name)
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f"{task_name!r} has no instruction-bank entry in {path}")
    return str(instructions[0])


def _read_scene_manifest(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"metric scene manifest not found: {path}")
    required = {
        "schema", "rollout_episode", "seed", "replicate", "task", "bench_subdir",
        "base_config", "dr_settings", "obstacle_density", "expected_clutter_count",
        "expected_scene_id", "expected_scene_fingerprint",
        "expected_scene_code_version", "checkpoint", "instruction",
        "expected_acting_arm",
    }
    jobs = []
    seen_episodes = set()
    seen_scene_ids = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid scene manifest JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or row.get("schema") != SCENE_MANIFEST_SCHEMA:
                raise ValueError(f"invalid scene manifest record at {path}:{line_number}")
            if set(row) != required:
                raise ValueError(
                    f"scene manifest record at {path}:{line_number} has fields "
                    f"{sorted(set(row) - required)} beyond the outcome-blind schema or misses "
                    f"{sorted(required - set(row))}"
                )
            episode = int(row["rollout_episode"])
            scene_id = str(row["expected_scene_id"])
            if episode in seen_episodes or scene_id in seen_scene_ids:
                raise ValueError(f"duplicate episode or scene_id at {path}:{line_number}")
            seen_episodes.add(episode)
            seen_scene_ids.add(scene_id)
            if row["task"] != SUPPORTED_TASK or row["bench_subdir"] != "study":
                raise ValueError(f"unsupported manifest task/domain at {path}:{line_number}")
            if not isinstance(row["dr_settings"], dict):
                raise ValueError(f"manifest dr_settings must be an object at {path}:{line_number}")
            if int(row["dr_settings"].get("obstacle_density", -1)) != int(row["obstacle_density"]):
                raise ValueError(f"manifest density fields disagree at {path}:{line_number}")
            jobs.append(dict(row))
    if not jobs:
        raise ValueError(f"scene manifest is empty: {path}")
    jobs.sort(key=lambda row: int(row["rollout_episode"]))
    if [int(row["rollout_episode"]) for row in jobs] != list(range(len(jobs))):
        raise ValueError("scene manifest episodes must be contiguous from zero")
    return path, jobs


def _metric_jobs(args):
    if args.scene_manifest:
        return _read_scene_manifest(args.scene_manifest)
    instruction = args.instruction or _default_instruction(args.task_name)
    jobs = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        jobs.append(
            {
                "rollout_episode": None,
                "seed": int(seed),
                "replicate": int(args.replicate),
                "task": args.task_name,
                "bench_subdir": args.bench_subdir,
                "base_config": args.base_config,
                "dr_settings": None,
                "obstacle_density": args.obstacle_density,
                "expected_clutter_count": None,
                "expected_scene_id": None,
                "expected_scene_fingerprint": None,
                "expected_scene_code_version": None,
                "checkpoint": args.checkpoint,
                "instruction": instruction,
                "expected_acting_arm": None,
            }
        )
    return None, jobs


def _validate_expected_identity(record, job):
    checks = {
        "scene_id": job["expected_scene_id"],
        "scene_fingerprint": job["expected_scene_fingerprint"],
        "scene_code_version": job["expected_scene_code_version"],
        "clutter_count": job["expected_clutter_count"],
        "arm": job["expected_acting_arm"],
    }
    mismatches = {
        field: {"expected": expected, "actual": record.get(field)}
        for field, expected in checks.items()
        if expected is not None and record.get(field) != expected
    }
    if mismatches:
        raise ValueError(
            f"regenerated scene does not match rollout episode {job['rollout_episode']}: "
            f"{mismatches}"
        )


def _metric_code_version():
    root = Path(__file__).resolve().parent
    return hash_files(
        [
            Path(__file__),
            root / "lib" / "geometric_metric.py",
            root / "lib" / "metric_buckets.py",
            root / "lib" / "task_roles.py",
            root / "lib" / "waypoints.py",
        ]
    )


def _metric_run_config(args, metric_cfg, manifest_path, jobs):
    rollout_dir = Path(args.rollout_run).resolve() if args.rollout_run else None
    rollout_config_path = rollout_dir / "config.json" if rollout_dir else None
    rollout_config = (
        json.loads(rollout_config_path.read_text(encoding="utf-8"))
        if rollout_config_path is not None and rollout_config_path.is_file()
        else None
    )
    scene_images_enabled = not args.no_scene_images and rollout_dir is None
    config = {
        "schema": POSTPROCESS_CONFIG_SCHEMA,
        "task": SUPPORTED_TASK,
        "target_metrics": len(jobs),
        "source_rollout": (
            None if rollout_dir is None else {
                "path": str(rollout_dir),
                "config_sha256": rollout_config.get("config_sha256") if rollout_config else None,
            }
        ),
        "scene_manifest": (
            None if manifest_path is None else {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "records": len(jobs),
            }
        ),
        "standalone_generation": (
            None if manifest_path is not None else {
                "base_config": args.base_config,
                "seed_start": int(args.seed_start),
                "num_seeds": int(args.num_seeds),
                "replicate": int(args.replicate),
                "checkpoint": args.checkpoint,
                "scene_code_version": task_scene_code_version(args.base_config),
                "dr_overrides": _dr_overrides(args),
            }
        ),
        "metric": asdict(metric_cfg),
        "metric_code_version": _metric_code_version(),
        "reach_cache_dir": str(Path(args.reach_cache_dir).resolve()),
        "reach_mode": args.reach_mode,
        "gripper_reference_radius_m": args.gripper_r,
        "bucket_spec": {
            "path": str(Path(args.bucket_spec).resolve()),
            "sha256": sha256_file(args.bucket_spec),
        },
        "report_every": int(args.report_every),
        "final_bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "metric_generation_outcome_blind": True,
        "analysis_loads_outcomes": rollout_dir is not None,
        "scene_images_enabled": scene_images_enabled,
        "scene_cameras": list(args.scene_cameras) if scene_images_enabled else [],
    }
    config["config_sha256"] = fingerprint(config)
    return config


def _validate_stored_config(path, expected):
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    if stored.get("schema") != POSTPROCESS_CONFIG_SCHEMA:
        raise ValueError(f"unsupported metric resume config schema in {path}")
    stored_hash = stored.get("config_sha256")
    unhashed = {key: value for key, value in stored.items() if key != "config_sha256"}
    if stored_hash != fingerprint(unhashed):
        raise ValueError(f"metric resume config hash mismatch: {path}")
    if stored != expected:
        raise ValueError(
            "metric resume configuration differs from the immutable config; "
            "use the original rollout, bucket spec, and metric settings"
        )
    return stored


def _read_committed_metric_records(out_dir, jobs, config_sha256):
    episodes_dir = Path(out_dir) / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    records = []
    seen_scenes = set()
    for path in sorted(episodes_dir.glob("episode*.json")):
        suffix = path.stem.removeprefix("episode")
        if not suffix.isdigit():
            raise ValueError(f"malformed metric episode filename: {path.name}")
        sequence = int(suffix)
        if sequence >= len(jobs):
            raise ValueError(f"metric episode is outside the declared manifest: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "robopro.task-metric.v1" or record.get("status") != "ok":
            raise ValueError(f"invalid committed metric record: {path}")
        if int(record.get("metric_sequence", -1)) != sequence:
            raise ValueError(f"metric sequence does not match filename: {path}")
        if record.get("metric_run_config_sha256") != config_sha256:
            raise ValueError(f"metric episode does not match immutable config: {path}")
        job = jobs[sequence]
        if record.get("scene_id") != job.get("expected_scene_id") and job.get("expected_scene_id"):
            raise ValueError(f"committed metric scene_id mismatch: {path}")
        if (
            job.get("expected_scene_fingerprint")
            and record.get("scene_fingerprint") != job["expected_scene_fingerprint"]
        ):
            raise ValueError(f"committed metric fingerprint mismatch: {path}")
        scene_id = record.get("scene_id")
        if scene_id in seen_scenes:
            raise ValueError(f"duplicate committed metric scene_id: {scene_id}")
        seen_scenes.add(scene_id)
        records.append(record)
    records.sort(key=lambda row: int(row["metric_sequence"]))
    return records


def _sync_metric_jsonl(out_dir, records):
    text = "".join(
        json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        for record in records
    )
    atomic_write_text(Path(out_dir) / "records.jsonl", text)


def _commit_metric_record(out_dir, record):
    sequence = int(record["metric_sequence"])
    episode_path = Path(out_dir) / "episodes" / f"episode{sequence:06d}.json"
    if episode_path.exists():
        raise FileExistsError(f"refusing to overwrite committed metric {episode_path}")
    atomic_write_json(episode_path, record)
    append_jsonl_fsync(Path(out_dir) / "records.jsonl", record)


def _regenerate_postprocess_reports(args, out_dir, records, *, complete, raise_errors):
    if not records or not args.rollout_run:
        return None
    try:
        records_path = Path(out_dir) / "records.jsonl"
        config_path = Path(out_dir) / "config.json"
        provenance = {
            "records_path": str(records_path.resolve()),
            "records_sha256": sha256_file(records_path),
            "record_count": len(records),
            "source_config_path": str(config_path.resolve()),
            "source_config_sha256": sha256_file(config_path),
            "outcome_data_loaded": False,
            "provisional": not complete,
        }
        write_distribution_reports(out_dir, records, provenance)
        summary = write_correlation_reports(
            out_dir,
            args.rollout_run,
            [records_path],
            args.bucket_spec,
            require_complete=complete,
            bootstrap_resamples=args.bootstrap_resamples if complete else 0,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(
            f"[postprocess] regenerated at {len(records)}/{summary['target_n']} metrics; "
            f"provisional={summary['provisional']}"
        )
        return summary
    except Exception as exc:
        print(f"[postprocess] WARNING: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if raise_errors:
            raise
        return None


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    print(f"[run] outputs: {out_dir}")
    records_path = out_dir / "records.jsonl"
    timings = Timings()
    metric_cfg = SeedMetricConfig.from_env(SeedMetricConfig.from_args(args))
    manifest_path, jobs = _metric_jobs(args)
    if args.rollout_run:
        rollout_dir = Path(args.rollout_run)
        rollout_config = json.loads(
            (rollout_dir / "config.json").read_text(encoding="utf-8")
        )
        rollout_hash = rollout_config.get("config_sha256")
        rollout_unhashed = {
            key: value for key, value in rollout_config.items()
            if key != "config_sha256"
        }
        if rollout_hash != fingerprint(rollout_unhashed):
            raise ValueError(f"source rollout config hash mismatch: {rollout_dir}")
        target = int(rollout_config.get("target_rollouts", -1))
        committed = len(list((rollout_dir / "episodes").glob("episode*.json")))
        if target <= 0 or committed != target or len(jobs) != target:
            raise ValueError(
                "integrated metric processing requires a complete rollout run: "
                f"target={target}, committed={committed}, manifest={len(jobs)}"
            )
    config = _metric_run_config(args, metric_cfg, manifest_path, jobs)
    config_path = out_dir / "config.json"
    if config_path.is_file():
        _validate_stored_config(config_path, config)
    else:
        atomic_write_json(config_path, config)
    records = _read_committed_metric_records(
        out_dir, jobs, config["config_sha256"]
    )
    _sync_metric_jsonl(out_dir, records)
    completed_sequences = {int(record["metric_sequence"]) for record in records}
    if records:
        print(
            f"[resume] {len(records)}/{len(jobs)} metrics committed; "
            "records.jsonl repaired from atomic episode files"
        )

    try:
        for sequence, job in enumerate(jobs):
            if sequence in completed_sequences:
                continue
            seed = int(job["seed"])
            print(f"[scene] {sequence + 1}/{len(jobs)} seed={seed}")
            started = time.perf_counter()
            env = None
            try:
                env_class = get_env_class(job["task"], bench_subdir=job["bench_subdir"])
                code_version = task_scene_code_version(job["base_config"])
                dr_overrides = (
                    dict(job["dr_settings"])
                    if job["dr_settings"] is not None
                    else _dr_overrides(args)
                )
                scene_cfg = build_cfg(
                    job["task"],
                    job["base_config"],
                    seed,
                    dr_overrides,
                    mode="measure",
                )
                dr = scene_cfg["domain_randomization"]
                if int(dr.get("obstacle_density", 0)) > 0 and dr.get("clean_background_rate") != 0:
                    raise ValueError(
                        "nonzero-density metric scenes require clean_background_rate=0 so clutter fires"
                    )
                with timings.section(f"seed_{seed}_scene_setup"):
                    env = env_class()
                    env.setup_demo(**scene_cfg)
                    table_z_bias = float(getattr(env, "table_z_bias", np.nan))
                    if not np.isfinite(table_z_bias) or abs(table_z_bias) > 1e-12:
                        raise ValueError(
                            f"table_z_bias must be 0 for this study, got {table_z_bias}"
                        )
                    roles = resolve_task_roles(env, job["task"])
                    env.target_collision_path = roles.target.collision_path
                    waypoints = canonical_waypoints(env, job["task"])
                    leg_specs = canonical_legs(waypoints)

                scene_images = []
                if config["scene_images_enabled"]:
                    with timings.section(f"seed_{seed}_scene_images"):
                        scene_images = _save_scene_images(
                            env, out_dir, seed, args.scene_cameras
                        )

                with timings.section(f"seed_{seed}_geometric"):
                    results = geometric_eps(
                        env,
                        waypoints[0].arm,
                        [(leg.start_xyz, leg.goal_xyz) for leg in leg_specs],
                        cfg=metric_cfg,
                        reach_cache_dir=args.reach_cache_dir,
                        reach_mode=args.reach_mode,
                        mask_target=True,
                    )
                with timings.section(f"seed_{seed}_record"):
                    record = _metric_record(
                        env,
                        args,
                        job,
                        scene_cfg,
                        roles,
                        waypoints,
                        leg_specs,
                        results,
                        code_version,
                        time.perf_counter() - started,
                        scene_images,
                    )
                    _validate_expected_identity(record, job)
                    record["metric_sequence"] = int(sequence)
                    record["metric_run_config_sha256"] = config["config_sha256"]
                    record["metric_code_version"] = config["metric_code_version"]
                    _commit_metric_record(out_dir, record)
                    records.append(record)
                    records.sort(key=lambda row: int(row["metric_sequence"]))
                value = "+inf" if record["eps_geom_min_unbounded"] else f"{record['eps_geom_min']:.4f}"
                print(
                    f"[scene] {record['scene_id']} arm={record['arm']} "
                    f"clutter={record['clutter_count']} eps_geom_min={value}"
                )
                if (
                    len(records) % args.report_every == 0
                    and len(records) < len(jobs)
                ):
                    _regenerate_postprocess_reports(
                        args, out_dir, records, complete=False, raise_errors=False
                    )
            finally:
                if env is not None:
                    try:
                        env.close_env()
                    except Exception:
                        pass
                del env
                gc.collect()
    except BaseException:
        records = _read_committed_metric_records(
            out_dir, jobs, config["config_sha256"]
        )
        _sync_metric_jsonl(out_dir, records)
        timings.save(out_dir)
        _regenerate_postprocess_reports(
            args,
            out_dir,
            records,
            complete=len(records) == len(jobs),
            raise_errors=False,
        )
        raise
    _sync_metric_jsonl(out_dir, records)
    timings.save(out_dir)
    _regenerate_postprocess_reports(
        args, out_dir, records, complete=True, raise_errors=True
    )
    print(f"[run] wrote {records_path}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", choices=[SUPPORTED_TASK], default=SUPPORTED_TASK)
    parser.add_argument("--bench-subdir", choices=["study"], default="study")
    parser.add_argument("--base-config", default="bench_demo_study_d10")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument(
        "--obstacle-density",
        type=int,
        help="override the base config density; omitted means use the config exactly",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--scene-manifest",
        help=(
            "outcome-blind metric_scene_manifest.jsonl emitted by a VLA run; "
            "when supplied it is authoritative for seed, density, and scene identity"
        ),
    )
    parser.add_argument(
        "--rollout-run",
        help=(
            "completed VLA rollout directory to post-process; uses its exact scene "
            "manifest and resumes in <rollout-run>/metric_postprocess"
        ),
    )
    parser.add_argument(
        "--instruction",
        help="instruction included in the scene fingerprint (default: instruction_bank[0])",
    )
    parser.add_argument(
        "--scene-cameras",
        default=",".join(DEFAULT_SCENE_CAMERAS),
        help="comma-separated static cameras saved as scene PNGs",
    )
    parser.add_argument(
        "--no-scene-images",
        action="store_true",
        help="disable the default initialized-scene PNGs",
    )
    parser.add_argument("--gripper-r", type=float, default=0.03)
    parser.add_argument("--reach-mode", choices=["occupancy", "sphere"], default="occupancy")
    parser.add_argument("--reach-cache-dir", default=str(REACH_CACHE_DIR))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--bucket-spec", default=str(DEFAULT_BUCKET_SPEC))
    parser.add_argument(
        "--report-every",
        type=int,
        default=10,
        help="regenerate provisional metric/correlation plots every N committed metrics",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--res", type=float, default=None)
    parser.add_argument("--zmin", type=float, default=None)
    parser.add_argument("--zmax", type=float, default=None)
    parser.add_argument("--zres", type=float, default=None)
    parser.add_argument("--seed-snap", type=float, default=None)
    parser.add_argument("--occ-shape", choices=["mesh", "extruded"], default=None)
    parser.add_argument("--obstacles", choices=["all", "occluders"], default="all")
    args = parser.parse_args()
    if args.num_seeds <= 0:
        parser.error("--num-seeds must be positive")
    if args.gripper_r <= 0:
        parser.error("--gripper-r must be positive")
    if args.report_every <= 0:
        parser.error("--report-every must be positive")
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")
    if args.rollout_run and args.scene_manifest:
        parser.error("--rollout-run supplies its own --scene-manifest")
    args.scene_cameras = tuple(
        name.strip() for name in args.scene_cameras.split(",") if name.strip()
    )
    if not args.no_scene_images and not args.scene_cameras:
        parser.error("--scene-cameras parsed to nothing")
    if args.rollout_run:
        rollout_run = Path(args.rollout_run).resolve()
        manifest = rollout_run / "metric_scene_manifest.jsonl"
        if not manifest.is_file():
            parser.error(f"rollout run lacks metric_scene_manifest.jsonl: {rollout_run}")
        args.rollout_run = str(rollout_run)
        args.scene_manifest = str(manifest)
        args.out_dir = str(rollout_run / "metric_postprocess")
        # The rollout already has a countertop video for every scene.  Avoid 9000
        # redundant static PNGs and keep metric regeneration outcome-independent.
        args.no_scene_images = True
    else:
        args.out_dir = str(
            Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    run(args)


if __name__ == "__main__":
    main()
