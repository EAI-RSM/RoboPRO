"""Canonical scene records shared by independent metric and rollout runs."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCENE_FINGERPRINT_SCHEMA = "robopro.scene-fingerprint.v1"


def _canonical(value):
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("scene fingerprints cannot contain non-finite values")
        return round(value, 9)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def canonical_json(value):
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_files(paths):
    digest = hashlib.sha256()
    for path in sorted(Path(path).resolve() for path in paths):
        if not path.is_file():
            raise FileNotFoundError(f"scene-version source does not exist: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def actor_snapshot(actor, collision_path=None, model_id=None):
    pose = actor.get_pose()
    out = {
        "name": str(actor.get_name()),
        "model_id": None if model_id is None else str(model_id),
        "pose": {
            "p": [float(v) for v in np.asarray(pose.p, dtype=float)],
            "q": [float(v) for v in np.asarray(pose.q, dtype=float)],
        },
    }
    if collision_path is not None:
        out["collision_path"] = str(Path(collision_path).resolve())
    scale = getattr(actor, "scale", None)
    if scale is not None:
        out["scale"] = np.asarray(scale, dtype=float).reshape(-1).tolist()
    return _canonical(out)


def scene_id(intent):
    return f"scene-{fingerprint(intent)[:20]}"


def scene_fingerprint(payload):
    wrapped = {"schema": SCENE_FINGERPRINT_SCHEMA, "scene": payload}
    return fingerprint(wrapped), _canonical(wrapped)


def task_scene_code_version(base_config):
    """Hash the selected Study task's scene-generating sources and configuration."""
    bench = Path(os.environ["BENCH_ROOT"])
    return hash_files(
        [
            bench / "bench_envs" / "study" / "put_cup_on_coaster.py",
            bench / "bench_envs" / "study" / "_study_base_task.py",
            bench / "bench_envs" / "_bench_base_task.py",
            bench / "bench_envs" / "utils" / "scene_gen_utils.py",
            bench / "bench_task_config" / "task_objects.yml",
            bench / "bench_task_config" / f"{base_config}.yml",
        ]
    )


def _clutter_snapshots(env):
    rows = []
    for info in getattr(env, "collision_list", None) or []:
        if info.get("is_obstacle") is not True:
            continue
        rows.append(actor_snapshot(info["actor"], info.get("collision_path")))
    return sorted(
        rows,
        key=lambda row: (row["name"], row.get("collision_path", ""), row["pose"]["p"]),
    )


def _obstacle_snapshots(roles):
    return sorted(
        [actor_snapshot(role.actor, role.collision_path) for role in roles.obstacles],
        key=lambda row: (row["name"], row.get("collision_path", ""), row["pose"]["p"]),
    )


def task_scene_identity(
    env,
    *,
    task,
    seed,
    replicate,
    bench_subdir,
    base_config,
    dr_settings,
    checkpoint,
    scene_code_version,
    roles,
    acting_arm,
    instruction,
):
    """Build the exact identity shared by independent metric and rollout runs."""
    intent = {
        "task": task,
        "seed": int(seed),
        "replicate": int(replicate),
        "bench_subdir": bench_subdir,
        "base_config": base_config,
        "dr_settings": dr_settings,
        "checkpoint": checkpoint,
        "scene_code_version": scene_code_version,
    }
    sid = scene_id(intent)
    clutter = _clutter_snapshots(env)
    target = actor_snapshot(
        roles.target.actor, roles.target.collision_path, roles.target.model_id
    )
    destination = actor_snapshot(
        roles.destination.actor,
        roles.destination.collision_path,
        roles.destination.model_id,
    )
    obstacle_density = int(dr_settings.get("obstacle_density", 0))
    fingerprint_payload = {
        **intent,
        "scene_id": sid,
        "configured_obstacle_density": obstacle_density,
        "realized_clutter_count": len(clutter),
        "clutter": clutter,
        "metric_obstacles": _obstacle_snapshots(roles),
        "target": target,
        "destination": destination,
        "destination_pose": [
            float(value) for value in np.asarray(env.des_obj_pose, dtype=float)
        ],
        "instruction": None if instruction is None else str(instruction),
        "acting_arm": acting_arm,
    }
    scene_hash, fingerprint_source = scene_fingerprint(fingerprint_payload)
    return {
        "scene_id": sid,
        "scene_fingerprint": scene_hash,
        "scene_fingerprint_source": fingerprint_source,
        "scene_code_version": scene_code_version,
        "obstacle_density": obstacle_density,
        "clutter_count": len(clutter),
        "clutter": clutter,
        "metric_obstacles": fingerprint_payload["metric_obstacles"],
        "target": target,
        "destination": destination,
        "acting_arm": acting_arm,
    }
