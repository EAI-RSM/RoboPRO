"""Typed, single-pass simulator evidence for control and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .graph_replanning import ActionGraphState, Evidence, GraspSubstage

if TYPE_CHECKING:
    from .live_adapter import LiveGraphContext


RELEASE_READY_VERTICAL_CLEARANCE_M = 0.10
EFFECTOR_IDS = {"left": -2, "right": -3}
GRASP_ALIGNMENT_DISTANCE_M = 0.12
GRASP_CLOSE_PREFERRED_DISTANCE_M = 0.08
GRASP_CLOSE_MAX_DISTANCE_M = 0.10
GRASP_HEIGHT_BAND_HALF_WIDTH_M = 0.02
# Generic parallel-jaw tolerance, not fit to any single episode: this is a
# starting point pending real-batch validation, not a calibrated value.
GRASP_ORIENTATION_MAX_ERROR_DEG = 20.0


def _aggregate(values: np.ndarray, valid: np.ndarray) -> Evidence:
    values = np.asarray(values, dtype=np.bool_)
    valid = np.asarray(valid, dtype=np.bool_)
    if values.shape != valid.shape:
        raise ValueError("relation values and validity mask are not aligned")
    if not np.any(valid):
        return Evidence.UNKNOWN
    return Evidence.TRUE if np.any(values & valid) else Evidence.FALSE


def _quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest rotation angle between two wxyz quaternions, in degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return float("nan")
    cos_half_angle = np.clip(abs(np.dot(a / norm_a, b / norm_b)), -1.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(cos_half_angle)))


def _min_orientation_error_deg(
    effector_quat: np.ndarray | None,
    reference_quats: tuple[tuple[float, float, float, float], ...],
) -> float:
    """Smallest angular error to any annotated valid grasp orientation.

    Returns NaN (not a large number) when no reference is available, so
    callers can distinguish "no annotation to check against" from "checked
    and misaligned" -- the former must not block a substage that worked
    fine before this check existed.
    """
    if effector_quat is None or not reference_quats:
        return float("nan")
    errors = [_quaternion_angle_deg(effector_quat, ref) for ref in reference_quats]
    errors = [error for error in errors if np.isfinite(error)]
    return float(min(errors)) if errors else float("nan")


@dataclass(frozen=True)
class EffectorEvidence:
    tcp_position_world: tuple[float, float, float] | None = None
    target_distance_m: float = np.nan
    target_vertical_offset_m: float = np.nan
    grasp_height_aligned: bool = False
    target_orientation_error_deg: float = np.nan
    # Fails open (True) when no reference orientation is available, so
    # objects without annotated grasp geometry behave exactly as before
    # this check existed.
    grasp_orientation_aligned: bool = True
    target_contact: bool = False
    target_held: bool = False


@dataclass(frozen=True)
class LiveTaskState:
    """Compatibility projection of evidence used by legacy adapter callers."""

    target_held: bool
    held_arm: str | None
    target_inside_destination: bool
    release_ready: bool


@dataclass(frozen=True)
class SimulatorEvidence:
    """All action-relevant evidence derived from one observation frame."""

    target_id: int | None
    held: Evidence
    held_arm: str | None
    goal_satisfied: Evidence
    path_blocked: Evidence
    reachable: Evidence
    visible: Evidence
    target_inside_destination: bool
    release_ready: bool
    left: EffectorEvidence
    right: EffectorEvidence
    grasp_substage: GraspSubstage
    grasp_arm: str | None
    grasp_close_immediate: bool
    grasp_height_half_width_m: float

    def action_graph_state(self) -> ActionGraphState:
        held_contact = Evidence.UNKNOWN
        if self.held is Evidence.FALSE:
            held_contact = Evidence.FALSE
        elif self.held is Evidence.TRUE and self.held_arm is not None:
            held_contact = (
                Evidence.TRUE
                if getattr(self, self.held_arm).target_contact
                else Evidence.FALSE
            )
        return ActionGraphState(
            held=self.held,
            held_contact=held_contact,
            release_ready=Evidence.TRUE if self.release_ready else Evidence.FALSE,
            goal_satisfied=self.goal_satisfied,
            path_blocked=self.path_blocked,
            reachable=self.reachable,
            visible=self.visible,
            held_arm=self.held_arm,
            grasp_substage=self.grasp_substage,
            grasp_arm=self.grasp_arm,
            grasp_close_immediate=self.grasp_close_immediate,
        )

    def live_task_state(self) -> LiveTaskState:
        return LiveTaskState(
            target_held=self.held_arm is not None,
            held_arm=self.held_arm,
            target_inside_destination=self.target_inside_destination,
            release_ready=self.release_ready,
        )

    def diagnostic_dict(self) -> dict[str, Any]:
        if self.target_id is None:
            return {}
        result: dict[str, Any] = {
            "target_id": self.target_id,
            "held_arm": self.held_arm or "",
            "grasp_close_immediate": self.grasp_close_immediate,
            "grasp_height_half_width": self.grasp_height_half_width_m,
        }
        for arm in ("left", "right"):
            effector = getattr(self, arm)
            result.update({
                f"{arm}_tcp_position_world": effector.tcp_position_world,
                f"target_{arm}_distance": effector.target_distance_m,
                f"target_{arm}_vertical_offset": effector.target_vertical_offset_m,
                f"target_{arm}_grasp_height_aligned": effector.grasp_height_aligned,
                f"target_{arm}_orientation_error_deg": effector.target_orientation_error_deg,
                f"target_{arm}_grasp_orientation_aligned": effector.grasp_orientation_aligned,
                f"target_{arm}_contact": effector.target_contact,
                f"held_by_{arm}": effector.target_held,
            })
        return result


def extract_simulator_evidence(context: "LiveGraphContext") -> SimulatorEvidence:
    """Interpret graph arrays and geometry once for all downstream consumers."""
    relations = context.relation_state
    objects = context.object_state
    retriever = context.retriever
    goal = context.goal
    target_indices = [
        context.index_by_id[value]
        for value in goal.target_ids
        if value in context.index_by_id
    ]
    destination_indices = [
        context.index_by_id[value]
        for value in goal.destination_ids
        if value in context.index_by_id
    ]
    if not target_indices or not destination_indices:
        raise ValueError("task goal IDs are absent from the live relation state")

    held_values = np.asarray(relations.get("held_by", ()), dtype=np.bool_)
    held_valid = (
        retriever._valid("held_by", held_values)
        if held_values.ndim == 2 else np.zeros_like(held_values)
    )
    held = _aggregate(held_values[target_indices], held_valid[target_indices])
    held_arm = None
    if held is Evidence.TRUE:
        for target_index in target_indices:
            active = np.flatnonzero(
                held_values[target_index] & held_valid[target_index]
            )
            if len(active):
                name = retriever.effector_names[int(active[0])].lower()
                held_arm = (
                    "left" if "left" in name
                    else "right" if "right" in name
                    else None
                )
                break

    goal_values = np.asarray(relations.get(goal.relation, ()), dtype=np.bool_)
    if goal_values.ndim == 2:
        goal_valid = retriever._valid(goal.relation, goal_values)
        selection = np.ix_(target_indices, destination_indices)
        goal_satisfied = _aggregate(
            goal_values[selection], goal_valid[selection]
        )
    else:
        goal_satisfied = Evidence.UNKNOWN

    reachable_values = np.asarray(
        relations.get("reachable_by", ()), dtype=np.bool_
    )
    reachable = Evidence.UNKNOWN
    if reachable_values.ndim == 2:
        reachable_valid = retriever._valid("reachable_by", reachable_values)
        reachable = _aggregate(
            reachable_values[target_indices], reachable_valid[target_indices]
        )

    visible_values = np.asarray(relations.get("visible_to", ()), dtype=np.bool_)
    visible = Evidence.UNKNOWN
    if (
        visible_values.ndim == 2
        and context.contract.default_camera in retriever.camera_names
    ):
        camera_index = retriever.camera_names.index(context.contract.default_camera)
        visible_valid = retriever._valid("visible_to", visible_values)
        visible = _aggregate(
            visible_values[target_indices, camera_index],
            visible_valid[target_indices, camera_index],
        )

    blocked_values = np.asarray(relations.get("blocks", ()), dtype=np.bool_)
    path_blocked = Evidence.UNKNOWN
    if blocked_values.ndim == 2:
        blocked_valid = retriever._valid("blocks", blocked_values)
        path_blocked = _aggregate(
            blocked_values[:, target_indices], blocked_valid[:, target_indices]
        )

    inside_destination = False
    inside = np.asarray(relations.get("in", ()), dtype=np.bool_)
    containment_valid = np.asarray(
        relations.get("containment_valid", np.zeros_like(inside)), dtype=np.bool_
    )
    if inside.ndim == 2 and containment_valid.shape == inside.shape:
        selection = np.ix_(target_indices, destination_indices)
        inside_destination = bool(
            np.any(inside[selection] & containment_valid[selection])
        )

    target_id = goal.target_ids[0] if len(goal.target_ids) == 1 else None
    raw_contact = np.asarray(relations.get("raw_contact", ()), dtype=np.bool_)
    target_pose = retriever._poses_world.get(target_id) if target_id is not None else None
    target_bounds = retriever._aabb_bounds.get(target_id)
    target_height_center = (
        float(target_bounds[1][2])
        if target_bounds is not None else
        float(target_pose[2]) if target_pose is not None else np.nan
    )
    target_height_half_width = GRASP_HEIGHT_BAND_HALF_WIDTH_M
    effectors = {}
    target_index = context.index_by_id.get(target_id) if target_id is not None else None
    for arm_index, arm in enumerate(("left", "right")):
        effector_index = context.index_by_id.get(EFFECTOR_IDS[arm])
        effector_pose = retriever._poses_world.get(EFFECTOR_IDS[arm])
        vertical_offset = (
            float(effector_pose[2] - target_height_center)
            if effector_pose is not None and np.isfinite(target_height_center)
            else np.nan
        )
        effector_quat = (
            effector_pose[3:7]
            if effector_pose is not None and effector_pose.shape[0] >= 7
            else None
        )
        orientation_error = _min_orientation_error_deg(
            effector_quat, context.target_grasp_orientations_wxyz
        )
        effectors[arm] = EffectorEvidence(
            tcp_position_world=(
                tuple(float(value) for value in effector_pose[:3])
                if effector_pose is not None else None
            ),
            target_distance_m=(
                float(np.linalg.norm(target_pose[:3] - effector_pose[:3]))
                if target_pose is not None and effector_pose is not None else np.nan
            ),
            target_vertical_offset_m=vertical_offset,
            grasp_height_aligned=bool(
                np.isfinite(vertical_offset)
                and abs(vertical_offset) <= target_height_half_width
            ),
            target_orientation_error_deg=orientation_error,
            grasp_orientation_aligned=bool(
                not np.isfinite(orientation_error)
                or orientation_error <= GRASP_ORIENTATION_MAX_ERROR_DEG
            ),
            target_contact=bool(
                target_index is not None
                and effector_index is not None
                and raw_contact.ndim == 2
                and raw_contact[target_index, effector_index]
            ),
            target_held=bool(
                target_index is not None
                and held_values.ndim == 2
                and target_index < held_values.shape[0]
                and arm_index < held_values.shape[1]
                and held_values[target_index, arm_index]
                and held_valid[target_index, arm_index]
            ),
        )

    release_ready = _release_ready(context, target_id, held_arm)
    finite_arms = [
        arm for arm in ("left", "right")
        if np.isfinite(effectors[arm].target_distance_m)
    ]
    close_arms = [
        arm for arm in finite_arms
        if effectors[arm].grasp_height_aligned
        and effectors[arm].grasp_orientation_aligned
        and effectors[arm].target_distance_m <= GRASP_CLOSE_MAX_DISTANCE_M
    ]
    vicinity_arms = [
        arm for arm in finite_arms
        if effectors[arm].grasp_height_aligned
        and effectors[arm].target_distance_m <= GRASP_ALIGNMENT_DISTANCE_M
    ]
    vertical_correction_arms = [
        arm for arm in finite_arms
        if not effectors[arm].grasp_height_aligned
        and effectors[arm].target_distance_m <= GRASP_ALIGNMENT_DISTANCE_M
    ]
    if close_arms:
        grasp_substage = GraspSubstage.CLOSE
        grasp_arm = min(
            close_arms, key=lambda arm: effectors[arm].target_distance_m
        )
        grasp_close_immediate = bool(
            effectors[grasp_arm].target_contact
            or effectors[grasp_arm].target_distance_m
            <= GRASP_CLOSE_PREFERRED_DISTANCE_M
        )
    elif vertical_correction_arms:
        grasp_arm = min(
            vertical_correction_arms,
            key=lambda arm: effectors[arm].target_distance_m,
        )
        grasp_substage = (
            GraspSubstage.MOVE_DOWN
            if effectors[grasp_arm].target_vertical_offset_m > 0
            else GraspSubstage.MOVE_UP
        )
        grasp_close_immediate = False
    elif vicinity_arms:
        grasp_substage = GraspSubstage.MOVE_CLOSER
        grasp_arm = min(
            vicinity_arms, key=lambda arm: effectors[arm].target_distance_m
        )
        grasp_close_immediate = False
    else:
        grasp_substage = GraspSubstage.ALIGN
        # Before geometric alignment, use the collision-aware arm from grasp_intent().
        grasp_arm = None
        grasp_close_immediate = False
    return SimulatorEvidence(
        target_id=target_id,
        held=held,
        held_arm=held_arm,
        goal_satisfied=goal_satisfied,
        path_blocked=path_blocked,
        reachable=reachable,
        visible=visible,
        target_inside_destination=inside_destination,
        release_ready=release_ready,
        left=effectors["left"],
        right=effectors["right"],
        grasp_substage=grasp_substage,
        grasp_arm=grasp_arm,
        grasp_close_immediate=grasp_close_immediate,
        grasp_height_half_width_m=target_height_half_width,
    )


def _release_ready(
    context: "LiveGraphContext", target_id: int | None, held_arm: str | None
) -> bool:
    destination_ids = context.goal.destination_ids
    if target_id is None or len(destination_ids) != 1:
        return False
    objects = context.object_state
    object_ids = np.asarray(objects["object_ids"], dtype=np.int64)
    aabb_lower = np.asarray(objects.get("aabb_lower", ()), dtype=np.float64)
    aabb_upper = np.asarray(objects.get("aabb_upper", ()), dtype=np.float64)
    has_aabb = np.asarray(objects.get("has_aabb", ()), dtype=np.bool_)
    object_index = {
        int(object_id): index for index, object_id in enumerate(object_ids)
    }
    target_index = object_index.get(target_id)
    destination_index = object_index.get(destination_ids[0])
    if not (
        target_index is not None
        and destination_index is not None
        and aabb_lower.ndim == 2
        and aabb_upper.shape == aabb_lower.shape
        and len(has_aabb) == len(object_ids)
        and bool(has_aabb[target_index])
        and bool(has_aabb[destination_index])
    ):
        return False
    target_center = 0.5 * (aabb_lower[target_index] + aabb_upper[target_index])
    destination_min = aabb_lower[destination_index]
    destination_max = aabb_upper[destination_index]
    horizontally_ready = bool(
        np.all(target_center[:2] >= destination_min[:2])
        and np.all(target_center[:2] <= destination_max[:2])
    )
    vertically_ready = bool(
        target_center[2] >= destination_min[2] - 0.02
        and aabb_lower[target_index, 2]
        <= destination_max[2] + RELEASE_READY_VERTICAL_CLEARANCE_M
    )
    return held_arm is not None and horizontally_ready and vertically_ready
