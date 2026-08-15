"""Behavior-neutral, per-action diagnostics for policy evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from experiments.graph_conditioned_pi05.live_adapter import (
    LiveGraphRetriever,
    task_goal_from_env,
)


GRIPPER_INDICES = {"left": 6, "right": 13}
EFFECTOR_IDS = {"left": -2, "right": -3}
GRIPPER_CLOSED_THRESHOLD = 0.2


def graph_evidence(task_env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Read target/effector geometry and contact evidence from an observation."""
    support = observation.get("benchmark_support") or {}
    relations = support.get("relation_state")
    objects = support.get("object_state")
    if relations is None or objects is None:
        return {}
    catalog = task_env._get_benchmark_object_catalog()
    retriever = LiveGraphRetriever(catalog, relations, objects)
    goal = task_goal_from_env(task_env, catalog)
    target_id = goal.target_ids[0] if len(goal.target_ids) == 1 else None
    if target_id is None:
        return {}
    index = {int(value): i for i, value in enumerate(retriever.object_ids)}
    target_index = index.get(int(target_id))
    target_pose = retriever._poses_world.get(int(target_id))
    held = np.asarray(relations.get("held_by", ()), dtype=np.bool_)
    held_valid = retriever._valid("held_by", held) if held.ndim == 2 else None
    raw_contact = np.asarray(relations.get("raw_contact", ()), dtype=np.bool_)
    result: dict[str, Any] = {"target_id": int(target_id), "held_arm": ""}
    for arm_index, arm in enumerate(("left", "right")):
        effector_pose = retriever._poses_world.get(EFFECTOR_IDS[arm])
        result[f"target_{arm}_distance"] = (
            float(np.linalg.norm(target_pose[:3] - effector_pose[:3]))
            if target_pose is not None and effector_pose is not None else np.nan
        )
        effector_index = index.get(EFFECTOR_IDS[arm])
        result[f"target_{arm}_contact"] = bool(
            target_index is not None
            and effector_index is not None
            and raw_contact.ndim == 2
            and raw_contact[target_index, effector_index]
        )
        is_held = bool(
            target_index is not None
            and held.ndim == 2
            and target_index < held.shape[0]
            and arm_index < held.shape[1]
            and held[target_index, arm_index]
            and (held_valid is None or held_valid[target_index, arm_index])
        )
        result[f"held_by_{arm}"] = is_held
        if is_held and not result["held_arm"]:
            result["held_arm"] = arm
    return result


class ActionTraceRecorder:
    """Collect a compact row per executed action and serialize it as NPZ."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(
        self,
        *,
        frame: int,
        prompt: str,
        phase: str,
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        observation: dict[str, Any],
        evidence: dict[str, Any] | None = None,
    ) -> None:
        raw = np.asarray(raw_action, dtype=np.float32).copy()
        executed = np.asarray(executed_action, dtype=np.float32).copy()
        measured = np.asarray(
            observation.get("joint_action", {}).get("vector", ()), dtype=np.float32
        )
        row = {
            "frame": int(frame), "prompt": str(prompt), "phase": str(phase),
            "raw_action": raw, "executed_action": executed,
        }
        for arm, action_index in GRIPPER_INDICES.items():
            row[f"raw_{arm}_gripper"] = float(raw[action_index])
            row[f"executed_{arm}_gripper"] = float(executed[action_index])
            row[f"measured_{arm}_gripper"] = (
                float(measured[action_index]) if measured.size > action_index else np.nan
            )
        row.update(evidence or {})
        self.rows.append(row)

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"frame_count": 0}
        result: dict[str, Any] = {"frame_count": len(self.rows)}
        for arm in ("left", "right"):
            distance = np.asarray([
                row.get(f"target_{arm}_distance", np.nan) for row in self.rows
            ], dtype=float)
            raw = np.asarray([row[f"raw_{arm}_gripper"] for row in self.rows])
            executed = np.asarray([row[f"executed_{arm}_gripper"] for row in self.rows])
            # Match Robot.is_{left,right}_gripper_close().
            close = executed < GRIPPER_CLOSED_THRESHOLD
            held_frames = [
                row["frame"] for row in self.rows if row.get(f"held_by_{arm}", False)
            ]
            contact_frames = [
                row["frame"] for row in self.rows
                if row.get(f"target_{arm}_contact", False)
            ]
            result.update({
                f"min_target_{arm}_distance": (
                    float(np.nanmin(distance)) if np.isfinite(distance).any() else None
                ),
                f"{arm}_close_command_count": int(close.sum()),
                f"{arm}_close_command_fraction": float(close.mean()),
                f"{arm}_ever_closed": bool(close.any()),
                f"{arm}_first_contact_frame": contact_frames[0] if contact_frames else None,
                f"{arm}_first_held_frame": held_frames[0] if held_frames else None,
                f"{arm}_gripper_override_count": int(np.count_nonzero(raw != executed)),
            })
        return result

    def save_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in self.rows for key in row})
        arrays = {}
        for key in keys:
            values = [row.get(key, np.nan) for row in self.rows]
            if key in {"prompt", "phase", "held_arm"}:
                arrays[key] = np.asarray(values, dtype=str)
            else:
                arrays[key] = np.asarray(values)
        np.savez_compressed(path, **arrays)
        return path
