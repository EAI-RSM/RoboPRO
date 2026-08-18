"""Typed, single-pass simulator evidence for control and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import transforms3d as t3d

from .graph_replanning import ActionGraphState, Evidence, GraspSubstage

if TYPE_CHECKING:
    from .live_adapter import LiveGraphContext


RELEASE_READY_VERTICAL_CLEARANCE_M = 0.10
EFFECTOR_IDS = {"left": -2, "right": -3}
GRASP_ALIGNMENT_DISTANCE_M = 0.12
GRASP_CLOSE_PREFERRED_DISTANCE_M = 0.08
GRASP_CLOSE_MAX_DISTANCE_M = 0.10
# Symmetric tolerance around the grasp-height reference. When an annotated
# grasp pose is available (context.left/right_reference_grasp_heights_m),
# the reference is that pose's actual world height -- not the object's
# AABB top surface,
# which for a tall object can sit several cm from the real grasp point,
# needing a large systematic margin just to compensate for that mismatch.
# With the real reference, this tolerance mostly needs to absorb policy
# jitter and geometric imprecision (observed on the order of 1-4mm), not a
# structural offset -- 3cm is a reasonable starting point given that, not
# an empirically tuned value; widen/narrow based on real-batch data once
# available. Falls back to the old AABB-top-based reference (and this same
# tolerance) when no annotation is available, unchanged from before.
GRASP_HEIGHT_BAND_HALF_WIDTH_M = 0.03
# Generic parallel-jaw tolerance, not fit to any single episode: this is a
# starting point pending real-batch validation, not a calibrated value.
GRASP_ORIENTATION_MAX_ERROR_DEG = 20.0
# RoboTwin's own candidate count for the rotate_lim arc
# (customized_robotwin/envs/robot/robot.py: create_target_pose_list uses
# CONFIGS.ROTATE_NUM, defined as 10 in customized_robotwin/envs/_GLOBAL_CONFIGS.py).
# Duplicated rather than imported: importing that module pulls in
# customized_robotwin.envs.utils (sapien), which isn't installed in this
# experiment's lightweight test environment and would break every existing
# test here. Keep this in sync with ROTATE_NUM if it ever changes.
GRASP_ORIENTATION_ARC_SAMPLES = 10


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


def _rotate_quat_about_own_local_axis(
    quat_wxyz: tuple[float, float, float, float], local_axis, angle_rad: float
) -> tuple[float, float, float, float]:
    """Rotate a wxyz quaternion by ``angle_rad`` about ITS OWN local axis.

    Matches ``rotate_along_axis(..., axis_type="target")`` in
    customized_robotwin/envs/utils/transforms.py: the axis is expressed in
    the orientation's own frame, converted to world frame, and the rotation
    is composed on the left (i.e. this is a world-frame rotation of the
    orientation about an axis attached to that same orientation).
    """
    rotation = t3d.quaternions.quat2mat(np.asarray(quat_wxyz, dtype=np.float64))
    world_axis = rotation @ np.asarray(local_axis, dtype=np.float64)
    delta = t3d.axangles.axangle2mat(world_axis, float(angle_rad))
    new_rotation = delta @ rotation
    return tuple(float(value) for value in t3d.quaternions.mat2quat(new_rotation))


def _rotate_pose_about_point(
    position_world,
    orientation_wxyz: tuple[float, float, float, float],
    pivot_world,
    local_axis,
    angle_rad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Rotate a world pose (position AND orientation) about ``pivot_world``
    -- which may be a different point than ``position_world`` itself --
    using an axis expressed in the pose's own current frame.

    Matches ``rotate_along_axis(..., axis_type="target")`` in
    customized_robotwin/envs/utils/transforms.py exactly: BOTH the offset
    from the pivot to the position, and the orientation, are transformed by
    the identical world-frame delta rotation. When ``pivot_world ==
    position_world`` this reduces to a pure self-rotation (position
    unchanged) -- the case ``_rotate_quat_about_own_local_axis`` already
    handles for the finger-swap flip.
    """
    position_world = np.asarray(position_world, dtype=np.float64)
    pivot_world = np.asarray(pivot_world, dtype=np.float64)
    rotation = t3d.quaternions.quat2mat(np.asarray(orientation_wxyz, dtype=np.float64))
    world_axis = rotation @ np.asarray(local_axis, dtype=np.float64)
    delta = t3d.axangles.axangle2mat(world_axis, float(angle_rad))
    new_rotation = delta @ rotation
    new_position = delta @ (position_world - pivot_world) + pivot_world
    return (
        tuple(float(value) for value in new_position),
        tuple(float(value) for value in t3d.quaternions.mat2quat(new_rotation)),
    )


def expand_grasp_pose_family(
    seed_position_world: tuple[float, float, float],
    seed_orientation_wxyz: tuple[float, float, float, float],
    contact_center_world: tuple[float, float, float],
    rotate_lim_rad: tuple[float, float],
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...]:
    """Every (position, orientation) RoboTwin would treat as an equally
    valid candidate grasp pose for this seed -- not just its orientation,
    and not just the seed's own unrotated position.

    Reproduces two symmetries of RoboTwin's own grasp-pose search
    (customized_robotwin/envs/robot/robot.py: create_target_pose_list) and
    of the aloha-agilex parallel-jaw gripper's own geometry:

    - A ``rotate_lim``-radian arc about the seed's own local Y (jaw-closing)
      axis. ``create_target_pose_list`` searches exactly this per-arm arc
      (read live from the embodiment config, not hardcoded -- callers pass
      each arm's own limit, since RoboTwin's is arm-specific) for a
      reachable grasp pose. Critically, this rotates the seed's POSITION
      too, as an offset from ``contact_center_world`` (the raw annotated
      contact point, NOT the seed's own position) -- an earlier version of
      this code rotated only the orientation and left the seed's position
      fixed, which is wrong: the offset from the contact center to the
      seed has magnitude ``GRASP_APPROACH_STANDOFF_M`` (12cm), so rotating
      it can move the candidate's height by up to ``STANDOFF * sin(theta)``
      -- several cm for even a ~1 radian arc, comfortably larger than a
      few-cm height tolerance.
    - A 180-degree flip about EACH resulting arc candidate's own local X
      (approach) axis, not the seed's -- 3D rotations don't commute, so
      flipping the seed and then rotating it is a different orientation
      from rotating the seed and then flipping that specific result (they
      only coincide at theta=0). Unlike the arc rotation, this genuinely IS
      a self-rotation (about the candidate's own position, not the contact
      center), so it leaves position unchanged.

    Public (not module-private) because callers need to build a separate
    expanded family per arm -- see
    ``LiveGraphContext.left_reference_orientations_wxyz`` /
    ``right_reference_orientations_wxyz`` and the analogous per-arm height
    tuples.

    Deliberately NOT reproduced:
    - ``create_target_pose_list``'s position-dependent ``towards`` sign
      disambiguation, which independently resolves each candidate's
      rotation direction (``+theta`` vs. ``-theta``) based on which side of
      the contact center it ends up on. This function always walks the
      configured ``rotate_lim`` direction only (never the mirrored sign),
      matching how the earlier, orientation-only version of this code
      already treated this same gap -- an acknowledged limitation, not
      newly introduced or silently widened here.
    - ``choose_grasp_pose``'s arm-mirrored preferred-direction scoring
      (``GRASP_DIRECTION_DIC["top_down_little_left"/"top_down_little_right"]``,
      blended with a task-specific side preference) used to RANK candidate
      contact points per arm before reachability filtering. That's a soft
      preference over which contact point RoboTwin would rather use, not a
      hard validity boundary like the two symmetries above, and needs
      per-task metadata this module doesn't currently extract.

    Both are acknowledged gaps, not silent assumptions. If live smoke-test
    data ever shows a known-good grasp reading a large error, these are the
    first two places to look.
    """
    # Deliberately NOT sorted into ascending order: create_target_pose_list
    # never reorders rotate_lim either, it uses the signed step directly
    # (rotate_step = (rotate_lim[1] - rotate_lim[0]) / ROTATE_NUM), so a
    # hypothetical reversed config (e.g. (1.0, 0.0)) walks 1.0, 0.9, ..., 0.1
    # -- excluding 0.0, not 1.0. Sorting first would silently swap which
    # endpoint is excluded relative to what RoboTwin actually configured.
    # Every embodiment config in this repo happens to be ascending already,
    # so this has no effect today, but preserves parity if that ever changes.
    lower, upper = float(rotate_lim_rad[0]), float(rotate_lim_rad[1])
    if upper != lower:
        # Matches create_target_pose_list exactly: step * i for i in
        # range(ROTATE_NUM), a half-open grid that never reaches the
        # configured endpoint (e.g. rotate_lim=(0,1) samples 0.0..0.9, never
        # 1.0). Using np.linspace(..., inclusive) here would add a pose
        # RoboTwin never actually generates, artificially shrinking the
        # reported error for anything near that boundary.
        step = (upper - lower) / GRASP_ORIENTATION_ARC_SAMPLES
        thetas = lower + step * np.arange(GRASP_ORIENTATION_ARC_SAMPLES)
    else:
        thetas = np.array([lower])
    # The flip must be applied to EACH rotated arc candidate, not to the
    # unrotated seed before rotating -- 3D rotations don't commute, so
    # rotate-then-flip and flip-then-rotate are different orientations in
    # general (they only coincide at theta=0). The physically-intended
    # symmetry is "this specific candidate's own finger-swapped variant",
    # i.e. flip about THAT candidate's own local X, not the seed's.
    family = []
    for theta in thetas:
        arc_position, arc_orientation = _rotate_pose_about_point(
            seed_position_world,
            seed_orientation_wxyz,
            contact_center_world,
            (0.0, 1.0, 0.0),
            float(theta),
        )
        family.append((arc_position, arc_orientation))
        flipped_orientation = _rotate_quat_about_own_local_axis(
            arc_orientation, (1.0, 0.0, 0.0), np.pi
        )
        family.append((arc_position, flipped_orientation))
    return tuple(family)


def _min_orientation_error_deg(
    effector_quat: np.ndarray | None,
    reference_quats: tuple[tuple[float, float, float, float], ...],
) -> float:
    """Smallest angular error to any already-symmetry-expanded reference.

    ``reference_quats`` is expected to already be one arm's full expanded
    orientation family (the orientations from ``expand_grasp_pose_family``,
    per-arm since RoboTwin's own candidate search is arm-specific) -- this
    function itself does no expansion, just the comparison.

    Returns NaN (not a large number) when no reference is available, so
    callers can distinguish "no annotation to check against" from "checked
    and misaligned" -- the former must not block a substage that worked
    fine before this check existed.
    """
    if effector_quat is None or not reference_quats:
        return float("nan")
    errors = [
        _quaternion_angle_deg(effector_quat, candidate)
        for candidate in reference_quats
    ]
    errors = [error for error in errors if np.isfinite(error)]
    return float(min(errors)) if errors else float("nan")


@dataclass(frozen=True)
class EffectorEvidence:
    tcp_position_world: tuple[float, float, float] | None = None
    # Known reference mismatch, not yet corrected: this is
    # norm(target_object_pose - TCP_pose) -- distance to the object's own
    # pose/center -- while target_vertical_offset_m below is measured
    # against the annotated grasp-pose candidates (contact point + 12cm
    # standoff + rotate_lim arc, see expand_grasp_pose_family), not the
    # object center. A grasp point can sit well off-center (e.g. a tall or
    # asymmetric object), so "close by this distance" and "aligned by that
    # height" are not guaranteed to describe the same physical point.
    # Unifying both around one grasp-pose reference (3D vector to the
    # nearest candidate, decomposed into vertical/horizontal/3D error)
    # would be more coherent, but is deliberately deferred to a later
    # treatment rather than folded into the height-reference fix above.
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
    # Why left/right's orientation reference tuples are what they are --
    # see live_adapter.OrientationReferenceStatus. Exported so a batch that
    # silently never resolved the target actor is visible in the trace
    # rather than looking identical to "no annotation, as expected."
    orientation_reference_status: str = "target_unresolved"
    orientation_reference_count: int = 0

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
            "orientation_reference_available": (
                self.orientation_reference_status == "available"
            ),
            "orientation_reference_count": self.orientation_reference_count,
            "orientation_reference_status": self.orientation_reference_status,
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
    aabb_fallback_heights_m = (
        (target_height_center,) if np.isfinite(target_height_center) else ()
    )
    effectors = {}
    target_index = context.index_by_id.get(target_id) if target_id is not None else None
    for arm_index, arm in enumerate(("left", "right")):
        effector_index = context.index_by_id.get(EFFECTOR_IDS[arm])
        effector_pose = retriever._poses_world.get(EFFECTOR_IDS[arm])
        # Prefer this arm's own annotated grasp-height family (rotate_lim,
        # hence the reachable height range, is arm-specific -- see
        # expand_grasp_pose_family) over the AABB-top proxy; fall back to
        # the old reference (unchanged) when no annotation is available,
        # rather than treating an empty tuple as "aligned anywhere."
        height_candidates_m = (
            context.left_reference_grasp_heights_m
            if arm == "left"
            else context.right_reference_grasp_heights_m
        ) or aabb_fallback_heights_m
        # Aligned if within tolerance of ANY candidate grasp height (multiple
        # annotated contact points, or points along the rotate_lim arc, may
        # sit at different heights); the correction direction (move
        # up/down) follows whichever candidate is nearest when none are
        # within tolerance.
        height_offsets = (
            [float(effector_pose[2] - height) for height in height_candidates_m]
            if effector_pose is not None
            else []
        )
        vertical_offset = (
            min(height_offsets, key=abs) if height_offsets else np.nan
        )
        height_aligned = any(
            abs(offset) <= target_height_half_width for offset in height_offsets
        )
        effector_quat = (
            effector_pose[3:7]
            if effector_pose is not None and effector_pose.shape[0] >= 7
            else None
        )
        orientation_error = _min_orientation_error_deg(
            effector_quat,
            context.left_reference_orientations_wxyz
            if arm == "left"
            else context.right_reference_orientations_wxyz,
        )
        effectors[arm] = EffectorEvidence(
            tcp_position_world=(
                tuple(float(value) for value in effector_pose[:3])
                if effector_pose is not None else None
            ),
            # Object-center reference (see the EffectorEvidence.target_distance_m
            # comment for why this is inconsistent with vertical_offset's
            # grasp-pose reference below).
            target_distance_m=(
                float(np.linalg.norm(target_pose[:3] - effector_pose[:3]))
                if target_pose is not None and effector_pose is not None else np.nan
            ),
            target_vertical_offset_m=vertical_offset,
            grasp_height_aligned=bool(height_aligned),
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
        # Orientation is diagnostics-only for now: recorded on every
        # EffectorEvidence and exported to the trace, but deliberately not
        # gating CLOSE yet. There is no corrective instruction for a
        # misaligned orientation (no ROTATE_FOR_GRASP substage), and the
        # existing fallback substages (move_closer/align) give actively
        # wrong advice for an orientation-only defect -- gating on it before
        # a real batch confirms it predicts outcomes risks trading a known
        # failure mode for an unvalidated, possibly worse one.
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
        orientation_reference_status=context.orientation_reference_status,
        orientation_reference_count=context.orientation_reference_count,
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
