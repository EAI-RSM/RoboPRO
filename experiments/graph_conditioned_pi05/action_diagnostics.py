"""Behavior-neutral, per-action diagnostics for policy evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.graph_conditioned_pi05.contract import RetrievalContract
from experiments.graph_conditioned_pi05.live_adapter import (
    LiveGraphContext,
    build_live_graph_context,
)
from experiments.graph_conditioned_pi05.simulator_evidence import (
    SimulatorEvidence,
    extract_simulator_evidence,
)


GRIPPER_INDICES = {"left": 6, "right": 13}
GRIPPER_CLOSED_THRESHOLD = 0.2


def graph_evidence(
    task_env: Any,
    observation: dict[str, Any],
    context: LiveGraphContext | None = None,
    evidence: SimulatorEvidence | None = None,
) -> dict[str, Any]:
    """Read target/effector geometry and contact evidence from an observation."""
    if evidence is None and context is None:
        support = observation.get("benchmark_support") or {}
        if support.get("relation_state") is None or support.get("object_state") is None:
            return {}
        context = build_live_graph_context(
            task_env, observation, RetrievalContract()
        )
    evidence = evidence or extract_simulator_evidence(context)
    return evidence.diagnostic_dict()


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
        action_intent: dict[str, Any] | None = None,
    ) -> None:
        raw = np.asarray(raw_action, dtype=np.float32).copy()
        executed = np.asarray(executed_action, dtype=np.float32).copy()
        measured = np.asarray(
            observation.get("joint_action", {}).get("vector", ()), dtype=np.float32
        )
        row = {
            "frame": int(frame), "prompt": str(prompt), "phase": str(phase),
            "raw_action": raw, "executed_action": executed,
            "action_intent": json.dumps(action_intent, sort_keys=True),
        }
        for arm, action_index in GRIPPER_INDICES.items():
            row[f"raw_{arm}_gripper"] = float(raw[action_index])
            row[f"executed_{arm}_gripper"] = float(executed[action_index])
            row[f"measured_{arm}_gripper"] = (
                float(measured[action_index]) if measured.size > action_index else np.nan
            )
        row.update(evidence or {})
        for arm in ("left", "right"):
            for family in ("selected", "orientation_best", "joint_best"):
                key = f"target_{arm}_{family}_candidate_index"
                current = int(row.get(key, -1))
                previous = int(self.rows[-1].get(key, -1)) if self.rows else -1
                row[f"target_{arm}_{family}_candidate_changed"] = bool(
                    current >= 0 and previous >= 0 and current != previous
                )
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
            if key in {
                "prompt", "phase", "held_arm", "action_intent",
                "left_tcp_pose_source", "right_tcp_pose_source",
                "grasp_reference_target_dis_source",
                "target_left_joint_best_selection_status",
                "target_right_joint_best_selection_status",
            }:
                arrays[key] = np.asarray(values, dtype=str)
            else:
                arrays[key] = np.asarray(values)
        np.savez_compressed(path, **arrays)
        return path
