#!/usr/bin/env python3
"""Inventory research-showcase evidence without modifying source artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return decode(value.item())
    if isinstance(value, np.ndarray):
        return [decode(item) for item in value.tolist()]
    return value


def video_metadata(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)["streams"][0]
        numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": float(numerator) / float(denominator),
            "frame_count": int(stream["nb_frames"]),
            "duration_seconds": float(stream["duration"]),
        }
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError, ZeroDivisionError) as error:
        return {"error": str(error)}


def relation_summary(state: h5py.Group, name: str) -> dict[str, Any]:
    if name not in state:
        return {"available": False}
    values = np.asarray(state[name], dtype=bool)
    flattened = values.reshape(values.shape[0], -1)
    active_per_frame = flattened.sum(axis=1)
    changed = np.flatnonzero(np.any(flattened[1:] != flattened[:-1], axis=1)) + 1
    result: dict[str, Any] = {
        "available": True,
        "shape": list(values.shape),
        "active_count_min": int(active_per_frame.min(initial=0)),
        "active_count_max": int(active_per_frame.max(initial=0)),
        "change_frames": changed.astype(int).tolist(),
    }
    evaluated_name = f"{name}_evaluated"
    if evaluated_name in state:
        evaluated = np.asarray(state[evaluated_name], dtype=bool)
        result["evaluated_frames"] = np.flatnonzero(evaluated).astype(int).tolist()
    return result


def action_summary(actions: h5py.Group) -> list[dict[str, Any]]:
    if "action_ids" not in actions:
        return []
    count = len(actions["action_ids"])
    field_map = {
        "action_id": "action_ids", "action_type": "action_types", "arm": "arms",
        "phase": "execution_phases", "status": "statuses", "start_frame": "start_frame",
        "end_frame": "end_frame", "recorded_frame_count": "recorded_frame_count",
        "provenance": "provenance", "observed_effects": "observed_effects_json",
        "tool_call": "tool_calls_json",
    }
    rows = []
    for index in range(count):
        row = {}
        for output_name, dataset_name in field_map.items():
            if dataset_name not in actions:
                continue
            value = decode(actions[dataset_name][index])
            if output_name in {"observed_effects", "tool_call"} and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            row[output_name] = value
        rows.append(row)
    return rows


def inspect_episode(hdf5_path: Path, video_path: Path, requested_relations: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hdf5": str(hdf5_path.relative_to(REPO_ROOT)),
        "video": str(video_path.relative_to(REPO_ROOT)),
        "source_exists": {"hdf5": hdf5_path.is_file(), "video": video_path.is_file()},
    }
    if not hdf5_path.is_file():
        return result
    with h5py.File(hdf5_path, "r") as root:
        support = root.get("benchmark_support")
        cameras = {
            name: int(group["rgb"].shape[0])
            for name, group in root.get("observation", {}).items()
            if isinstance(group, h5py.Group) and "rgb" in group
        }
        result.update({
            "attributes": {key: decode(value) for key, value in root.attrs.items()},
            "hdf5_frame_count": max(cameras.values(), default=0),
            "cameras": cameras,
        })
        if support is None:
            result["error"] = "benchmark_support group is missing"
            return result
        catalog = support.get("object_catalog")
        result["objects"] = [] if catalog is None else [
            {"id": int(object_id), "name": decode(name), "role": decode(role)}
            for object_id, name, role in zip(catalog["object_ids"], catalog["names"], catalog["roles"])
        ]
        state = support.get("relation_state")
        result["relations"] = {} if state is None else {
            name: relation_summary(state, name) for name in sorted(requested_relations)
        }
        result["relation_metadata"] = {} if state is None else {
            key: decode(state[key][()]) for key in (
                "implemented_relation_names", "visible_to_camera_names",
                "reachable_by_effector_names",
            ) if key in state
        }
        result["actions"] = action_summary(support["action_nodes"]) if "action_nodes" in support else []
        contract = support.get("policy_action_contract")
        result["policy_action_contract"] = {} if contract is None else {
            key: decode(dataset[()]) for key, dataset in contract.items()
            if key in {"version", "provider_name", "provider_kind", "action_representation"}
        }
    if video_path.is_file():
        result["video_metadata"] = video_metadata(video_path)
        video_frames = result["video_metadata"].get("frame_count")
        result["frame_alignment"] = {
            "aligned": video_frames == result["hdf5_frame_count"],
            "hdf5_frames": result["hdf5_frame_count"],
            "video_frames": video_frames,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config/claims.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/research_feature_showcase/evidence_inventory.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    config = json.loads(config_path.read_text())
    source_root = REPO_ROOT / config["source_root"]

    episode_relations: dict[str, set[str]] = {}
    for claim in config["claims"]:
        for episode in claim.get("episodes", [claim.get("episode")]):
            if episode:
                episode_relations.setdefault(episode, set()).update(claim.get("relations", []))

    episodes = {}
    for episode, relations in sorted(episode_relations.items()):
        episode_root = source_root / episode
        episodes[episode] = inspect_episode(
            episode_root / "data/episode0.hdf5",
            episode_root / "video/episode0.mp4",
            relations,
        )

    inventory = {
        "inventory_schema_version": "1.0.0",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "claims": config["claims"],
        "limitations": config.get("limitations", []),
        "episodes": episodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Wrote {output_path}")
    for name, episode in episodes.items():
        alignment = episode.get("frame_alignment", {})
        print(f"  {name}: HDF5={alignment.get('hdf5_frames')} video={alignment.get('video_frames')} aligned={alignment.get('aligned')}")


if __name__ == "__main__":
    main()
