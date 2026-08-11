"""Validate graph-conditioned pi0.5 inputs against schema-1.9 episodes."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean

import h5py

from .contract import RetrievalContract
from .graph_retriever import HDF5GraphRetriever
from .graph_serializer import serialize_facts


FRAME_ALIGNED_PATHS = (
    "observation/countertop_camera/rgb",
    "observation/right_camera/rgb",
    "observation/left_camera/rgb",
    "joint_action/vector",
    "benchmark_support/object_state/pose_world",
    "benchmark_support/relation_state/near",
    "benchmark_support/relation_state/blocks",
    "benchmark_support/relation_state/occludes",
    "benchmark_support/action_nodes/active",
)


def discover_files(inputs: list[str]) -> list[Path]:
    files: set[Path] = set()
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("episode*.hdf5"))
        else:
            files.update(Path(item).resolve() for item in glob.glob(value, recursive=True))
    return sorted(files)


def validate_episode(path: Path, contract: RetrievalContract, frame_stride: int) -> dict:
    failures: list[str] = []
    token_counts: list[int] = []
    retrieved_counts: list[int] = []
    selected_counts: list[int] = []
    dropped_counts: list[int] = []
    with h5py.File(path, "r") as root:
        schema = str(root.attrs.get("schema_version", ""))
        task = str(root.attrs.get("task_name", ""))
        success = bool(root.attrs.get("success", False))
        if schema != "1.9.0":
            failures.append(f"expected schema 1.9.0, got {schema!r}")
        missing = [name for name in FRAME_ALIGNED_PATHS if name not in root]
        failures.extend(f"missing required dataset {name}" for name in missing)
        frame_count = 0
        if not missing:
            lengths = {name: int(root[name].shape[0]) for name in FRAME_ALIGNED_PATHS}
            frame_count = lengths[FRAME_ALIGNED_PATHS[0]]
            if len(set(lengths.values())) != 1:
                failures.append(f"frame-axis mismatch: {lengths}")
        if failures:
            return {
                "file": str(path), "task": task, "schema": schema,
                "success": success, "frame_count": frame_count,
                "status": "FAIL", "failures": failures,
            }

        retriever = HDF5GraphRetriever(root, contract)
        frames = list(range(0, frame_count, frame_stride))
        if frames[-1] != frame_count - 1:
            frames.append(frame_count - 1)
        for frame in frames:
            facts = retriever.retrieve_frame(frame)
            repeated = retriever.retrieve_frame(frame)
            if facts != repeated:
                failures.append(f"retrieval is non-deterministic at frame {frame}")
                continue
            if any(fact.relation.startswith("action") for fact in facts):
                failures.append(f"expert action-node leakage at frame {frame}")
            graph = serialize_facts(facts, contract.graph_token_budget)
            if graph.token_count > contract.graph_token_budget:
                failures.append(f"token budget exceeded at frame {frame}")
            token_counts.append(graph.token_count)
            retrieved_counts.append(len(facts))
            selected_counts.append(len(graph.selected_facts))
            dropped_counts.append(graph.dropped_fact_count)

    def summary(values: list[int]) -> dict:
        return {
            "min": min(values) if values else 0,
            "mean": round(mean(values), 3) if values else 0,
            "max": max(values) if values else 0,
        }

    return {
        "file": str(path),
        "task": task,
        "schema": schema,
        "success": success,
        "frame_count": frame_count,
        "sampled_frame_count": len(token_counts),
        "retrieved_fact_count": summary(retrieved_counts),
        "selected_fact_count": summary(selected_counts),
        "dropped_fact_count": summary(dropped_counts),
        "graph_token_count": summary(token_counts),
        "token_budget": contract.graph_token_budget,
        "action_history_included": contract.include_action_history,
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="HDF5 files, directories, or glob patterns")
    parser.add_argument("--graph-token-budget", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--default-camera", default="countertop_camera")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    files = discover_files(args.inputs)
    if not files:
        parser.error("No HDF5 episode files found")
    contract = RetrievalContract(
        graph_token_budget=args.graph_token_budget,
        default_camera=args.default_camera,
    )
    episodes = [validate_episode(path, contract, args.frame_stride) for path in files]
    report = {
        "protocol": "graph_conditioned_pi05_poc_v1",
        "conditions": ["visual_only", "visual_retrieved_graph"],
        "episode_count": len(episodes),
        "pass_count": sum(item["status"] == "PASS" for item in episodes),
        "status": "PASS" if all(item["status"] == "PASS" for item in episodes) else "FAIL",
        "contract": {
            "graph_token_budget": contract.graph_token_budget,
            "default_camera": contract.default_camera,
            "max_hops": contract.max_hops,
            "include_action_history": contract.include_action_history,
            "include_invalid": contract.include_invalid,
        },
        "episodes": episodes,
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
