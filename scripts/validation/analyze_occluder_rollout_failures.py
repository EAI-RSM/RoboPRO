#!/usr/bin/env python3
"""Summarize occluder-rollout results by coarse failure mode."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def load_records(results_dir: Path) -> list[dict]:
    path = results_dir / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify_episode(record: dict, hdf5_path: Path) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        names = ast.literal_eval(f["object_state/name"][0].decode("utf-8"))
        pos = f["object_state/position"][:]
        bottle_idx = names.index("001_bottle")
        pad_idx = names.index("box")

        bottle_disp = float(np.linalg.norm(pos[-1, bottle_idx] - pos[0, bottle_idx]))
        final_pad_xy_dist = float(np.linalg.norm(pos[-1, bottle_idx, :2] - pos[0, pad_idx, :2]))
        left_open = float(f["endpose/left_gripper"][-1]) > 0.9
        right_open = float(f["endpose/right_gripper"][-1]) > 0.9

    if record.get("rollout_success"):
        mode = "success"
    elif bottle_disp < 0.02:
        mode = "no_pick_or_early_plan_fail"
    elif bottle_disp < 0.15:
        mode = "picked_or_lifted_but_no_transport"
    elif final_pad_xy_dist < 0.06:
        mode = "near_pad_but_failed_release_or_alignment"
    else:
        mode = "transport_failed_far_from_pad"

    return {
        "ep": record["rollout_ep"],
        "seed": record["seed"],
        "success": bool(record.get("rollout_success")),
        "occluder_shown": bool(record.get("occluder_shown")),
        "visible_fraction": float(record["visible_fraction"]),
        "mode": mode,
        "bottle_disp": bottle_disp,
        "final_pad_xy_dist": final_pad_xy_dist,
        "grippers_open": left_open and right_open,
        "artifact": str(hdf5_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="scripts/validation/results/phase2_occluder_rollout",
        help="Directory containing records.jsonl plus success/fail subfolders.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_records(results_dir)
    rows = []
    for record in records:
        rel = record.get("rollout_data")
        if not rel:
            continue
        hdf5_path = results_dir / rel
        if not hdf5_path.exists():
            continue
        rows.append(classify_episode(record, hdf5_path))

    counts = Counter(row["mode"] for row in rows)
    print("Failure-mode summary")
    for mode, count in sorted(counts.items()):
        print(f"  {mode}: {count}")

    print("\nEpisodes")
    for row in sorted(rows, key=lambda item: (item["mode"], item["ep"])):
        print(
            f"  ep={row['ep']:>2} seed={row['seed']:>2} mode={row['mode']:<38} "
            f"vis={row['visible_fraction']:.3f} disp={row['bottle_disp']:.3f} "
            f"pad_xy={row['final_pad_xy_dist']:.3f} occ={int(row['occluder_shown'])}"
        )


if __name__ == "__main__":
    main()
