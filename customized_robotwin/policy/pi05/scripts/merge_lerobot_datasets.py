"""Merge multiple LeRobot v2.1 datasets into one combined dataset.

Takes any number of existing LeRobot dataset directories and produces a single
combined dataset by re-indexing episodes, frames, and task_indices sequentially.
Image data embedded in parquet columns is preserved as-is.

Usage (run from customized_robotwin/policy/pi05/):

    # Merge 5 rollout datasets into one
    HF_LEROBOT_HOME=/shared_work/joshua/RoboPRO/lerobot \
    uv run scripts/merge_lerobot_datasets.py \
        --input-repos \
            put_bottle_in_fridge_rollout_d15 \
            put_sauce_can_in_cabinet_rollout_d15 \
            pick_bottle_from_fridge_rollout_d15 \
            put_bottle_in_basket_rollout_d15 \
            pick_can_from_cabinet_rollout_d15 \
        --output-repo rollout_all_5tasks_d15

    # Point at a different lerobot home:
    --lerobot-home /shared_work/joshua/RoboPRO/lerobot
"""

import argparse
import json
import shutil
from pathlib import Path

import os

import pandas as pd

_DEFAULT_LEROBOT_HOME = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, records: list[dict]):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def merge(input_repos: list[str], output_repo: str, lerobot_home: Path):
    out_dir = lerobot_home / output_repo
    if out_dir.exists():
        print(f"Removing existing {out_dir}")
        shutil.rmtree(out_dir)
    (out_dir / "data" / "chunk-000").mkdir(parents=True)
    (out_dir / "meta").mkdir(parents=True)

    # ── Pass 1: collect metadata from all source datasets ──────────────────
    all_tasks: dict[str, int] = {}      # task_string → new global task_index
    combined_episodes: list[dict] = []
    combined_stats: list[dict] = []

    ep_offset = 0      # cumulative episode index
    frame_offset = 0   # cumulative global frame index
    ref_info = None    # info.json from first dataset (fps, features, etc.)

    for repo_id in input_repos:
        src = lerobot_home / repo_id
        if not src.exists():
            raise FileNotFoundError(f"Source dataset not found: {src}")

        info = json.loads((src / "meta" / "info.json").read_text())
        if ref_info is None:
            ref_info = info

        # Build task_index remapping for this source dataset
        src_tasks = load_jsonl(src / "meta" / "tasks.jsonl")  # [{task_index, task}, ...]
        local_to_global: dict[int, int] = {}
        for t in src_tasks:
            task_str = t["task"]
            if task_str not in all_tasks:
                all_tasks[task_str] = len(all_tasks)
            local_to_global[t["task_index"]] = all_tasks[task_str]

        # Remap episodes.jsonl
        src_episodes = load_jsonl(src / "meta" / "episodes.jsonl")
        for ep in src_episodes:
            new_ep = dict(ep)
            new_ep["episode_index"] = ep["episode_index"] + ep_offset
            combined_episodes.append(new_ep)

        # Remap episodes_stats.jsonl if present
        stats_path = src / "meta" / "episodes_stats.jsonl"
        if stats_path.exists():
            for stat in load_jsonl(stats_path):
                new_stat = dict(stat)
                new_stat["episode_index"] = stat["episode_index"] + ep_offset
                combined_stats.append(new_stat)

        # ── Pass 2: rewrite parquet files ──────────────────────────────────
        parquets = sorted((src / "data" / "chunk-000").glob("episode_*.parquet"))
        print(f"  {repo_id}: {len(parquets)} episodes, task_map={local_to_global}")

        for pq_path in parquets:
            df = pd.read_parquet(pq_path)

            # Remap indices
            df["episode_index"] = df["episode_index"] + ep_offset
            df["index"] = df["index"] + frame_offset
            df["task_index"] = df["task_index"].map(local_to_global)

            # Determine new episode number from remapped column
            new_ep_idx = int(df["episode_index"].iloc[0])
            out_path = out_dir / "data" / "chunk-000" / f"episode_{new_ep_idx:06d}.parquet"
            df.to_parquet(out_path, index=False)

        ep_offset += info["total_episodes"]
        frame_offset += info["total_frames"]

    # ── Write combined metadata ─────────────────────────────────────────────
    combined_tasks = [{"task_index": v, "task": k} for k, v in sorted(all_tasks.items(), key=lambda x: x[1])]

    save_jsonl(out_dir / "meta" / "tasks.jsonl", combined_tasks)
    save_jsonl(out_dir / "meta" / "episodes.jsonl", combined_episodes)
    if combined_stats:
        save_jsonl(out_dir / "meta" / "episodes_stats.jsonl", combined_stats)

    # Build combined info.json from the first dataset's template
    out_info = dict(ref_info)
    out_info["total_episodes"] = ep_offset
    out_info["total_frames"] = frame_offset
    out_info["total_tasks"] = len(all_tasks)
    out_info["splits"] = {"train": f"0:{ep_offset}"}
    (out_dir / "meta" / "info.json").write_text(json.dumps(out_info, indent=4))

    print(f"\nMerged dataset saved to {out_dir}")
    print(f"  Total episodes : {ep_offset}")
    print(f"  Total frames   : {frame_offset}")
    print(f"  Tasks ({len(all_tasks)}):")
    for task_str, idx in all_tasks.items():
        print(f"    [{idx}] {task_str}")


def main():
    parser = argparse.ArgumentParser(description="Merge LeRobot v2.1 datasets")
    parser.add_argument("--input-repos", nargs="+", required=True,
                        help="repo_ids of source datasets (relative to HF_LEROBOT_HOME)")
    parser.add_argument("--output-repo", required=True,
                        help="repo_id for the merged output dataset")
    parser.add_argument("--lerobot-home", type=Path, default=None,
                        help="Override HF_LEROBOT_HOME (default: use env var)")
    args = parser.parse_args()

    lerobot_home = args.lerobot_home or _DEFAULT_LEROBOT_HOME
    print(f"HF_LEROBOT_HOME: {lerobot_home}")
    print(f"Merging: {args.input_repos} → {args.output_repo}\n")

    merge(args.input_repos, args.output_repo, Path(lerobot_home))


if __name__ == "__main__":
    main()
