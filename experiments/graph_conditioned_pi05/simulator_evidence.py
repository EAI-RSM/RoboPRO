"""Typed, single-pass simulator evidence for control and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import transforms3d as t3d

from .graph_replanning import ActionGraphState, Evidence, GraspSubstage

if TYPE_CHECKING:
    from .live_adapter import LiveGraphContext


PLACEMENT_HORIZONTAL_MARGIN_M = 0.01
PLACEMENT_RELEASE_ABOVE_RIM_CLEARANCE_M = 0.05
PLACEMENT_ON_VERTICAL_TOLERANCE_M = 0.02
EFFECTOR_IDS = {"left": -2, "right": -3}
GRASP_ALIGNMENT_DISTANCE_M = 0.12
GRASP_CLOSE_PREFERRED_DISTANCE_M = 0.08
GRASP_CLOSE_MAX_DISTANCE_M = 0.10
ANNOTATED_GRASP_CLOSE_MAX_DISTANCE_M = 0.05
# Annotated closure requires the TCP to lie within this signed interval along
# the selected candidate's local approach axis, in addition to the 5 cm total
# distance and lateral-centering limits.
ANNOTATED_GRASP_APPROACH_ERROR_MIN_M = 0.005
ANNOTATED_GRASP_APPROACH_ERROR_MAX_M = 0.020
# Maximum displacement orthogonal to the annotated local approach axis.
# This prevents incidental off-center contact from bypassing grasp alignment.
ANNOTATED_GRASP_LATERAL_ERROR_MAX_M = 0.015
# Contact-calibrated gripper-center TCP offset along annotated local -X.
GRASP_APPROACH_STANDOFF_M = 0.05
# Symmetric tolerance around the grasp-position reference. When an annotated
# grasp pose is available (context.left/right_reference_grasp_positions_m),
# the reference is that pose's actual world position -- not the object's
# AABB top surface, which for a tall object can sit several cm from the
# real grasp point, needing a large systematic margin just to compensate
# for that mismatch. With the real reference, this tolerance mostly needs
# to absorb policy jitter and geometric imprecision (observed on the order
# of 1-4mm), not a structural offset -- 3cm is a reasonable starting point
# given that, not an empirically tuned value; widen/narrow based on
# real-batch data once available. Falls back to the object's own (x, y)
# with the old AABB-top-preferring z (and this same tolerance) when no
# annotation is available -- height's fallback is unchanged from before;
# distance/horizontal-offset's fallback now shares that same point too
# (previously distance fell back to the raw pose center even when height's
# fallback preferred the AABB top, a second, narrower instance of the
# reference-mismatch this file's distance/vertical-offset fix addresses).
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
# create_target_pose_list (customized_robotwin/envs/robot/robot.py) always
# calls rotate_along_axis(..., towards=[0, -1, 0]): whichever sign of theta
# would place the candidate on the wrong side of the contact center (a
# negative dot product against this vector) is rejected in favor of the
# other sign. Not arm- or task-specific, so a single shared constant.
GRASP_TOWARDS_AXIS = (0.0, -1.0, 0.0)


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
    towards=None,
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

    ``towards``, when given, reproduces ``rotate_along_axis``'s sign
    disambiguation exactly: compute the ``+angle_rad`` candidate first: if
    its position, relative to the pivot, has a negative dot product with
    ``towards``, recompute the WHOLE candidate (position and orientation)
    using ``-angle_rad`` instead. ``create_target_pose_list`` always calls
    this with ``towards=[0,-1,0]`` (see ``GRASP_TOWARDS_AXIS`` below) --
    the configured ``rotate_lim`` angle is a magnitude, not a guaranteed
    sign, so the direction actually walked can flip per contact
    point/object-center geometry.
    """
    position_world = np.asarray(position_world, dtype=np.float64)
    pivot_world = np.asarray(pivot_world, dtype=np.float64)
    rotation = t3d.quaternions.quat2mat(np.asarray(orientation_wxyz, dtype=np.float64))
    world_axis = rotation @ np.asarray(local_axis, dtype=np.float64)
    angle_rad = float(angle_rad)
    delta = t3d.axangles.axangle2mat(world_axis, angle_rad)
    new_position = delta @ (position_world - pivot_world) + pivot_world
    if towards is not None:
        towards_dot = np.dot(new_position - pivot_world, np.asarray(towards, dtype=np.float64))
        if towards_dot < 0:
            delta = t3d.axangles.axangle2mat(world_axis, -angle_rad)
            new_position = delta @ (position_world - pivot_world) + pivot_world
    new_rotation = delta @ rotation
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

    Reproduces three behaviors of RoboTwin's own grasp-pose search
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
      seed has magnitude ``GRASP_APPROACH_STANDOFF_M`` (5cm), so rotating
      it can move the candidate's height by up to ``STANDOFF * sin(theta)``
      -- several cm for even a ~1 radian arc, comfortably larger than a
      few-cm height tolerance.
    - ``create_target_pose_list``'s ``towards=[0,-1,0]`` sign
      disambiguation (``GRASP_TOWARDS_AXIS`` below): for each arc candidate,
      RoboTwin computes the ``+theta`` position first and, only if it ends
      up on the wrong side of the contact center (a negative dot product
      with ``towards``), recomputes the WHOLE candidate -- position AND
      orientation -- using ``-theta`` instead. An earlier version of this
      code always used the configured ``+theta`` directly; since the
      rotated offset has magnitude ``GRASP_APPROACH_STANDOFF_M`` (5cm),
      picking the wrong sign can move a candidate's height by up to twice
      its unrotated-seed-relative displacement -- several cm, again
      comfortably larger than the height tolerance, and specifically what
      controls whether a given arc sample is even a candidate RoboTwin
      would generate at all.
    - A 180-degree flip about EACH resulting arc candidate's own local X
      (approach) axis, not the seed's -- 3D rotations don't commute, so
      flipping the seed and then rotating it is a different orientation
      from rotating the seed and then flipping that specific result (they
      only coincide at theta=0). Unlike the arc rotation, this genuinely IS
      a self-rotation (about the candidate's own position, not the contact
      center), so it leaves position unchanged, and RoboTwin's own
      ``towards`` sign selection has no bearing on it either.

    Public (not module-private) because callers need to build a separate
    expanded family per arm -- see
    ``LiveGraphContext.left_reference_orientations_wxyz`` /
    ``right_reference_orientations_wxyz`` and the analogous per-arm height
    tuples.

    Deliberately NOT reproduced:
    - ``choose_grasp_pose``'s arm-mirrored preferred-direction scoring
      (``GRASP_DIRECTION_DIC["top_down_little_left"/"top_down_little_right"]``,
      blended with a task-specific side preference) used to RANK candidate
      contact points per arm before reachability filtering. That's a soft
      preference over which contact point RoboTwin would rather use, not a
      hard validity boundary like the symmetries above, and needs
      per-task metadata this module doesn't currently extract.

    This is an acknowledged gap, not a silent assumption. If live
    smoke-test data ever shows a known-good grasp reading a large error,
    this is the first place to look.
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
            towards=GRASP_TOWARDS_AXIS,
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
class CandidateMatch:
    index: int = -1
    contact_point_index: int = -1
    arc_sample_index: int = -1
    finger_flip_index: int = -1
    position_world: tuple[float, float, float] | None = None
    orientation_wxyz: tuple[float, float, float, float] | None = None
    error_world: tuple[float, float, float] | None = None
    error_local: tuple[float, float, float] | None = None
    distance_m: float = np.nan
    orientation_error_deg: float = np.nan


def _candidate_match(index, positions, orientations, metadata, deltas, orientation_errors):
    """Materialize one coupled pose candidate for diagnostics only."""
    if index is None or index < 0 or index >= len(positions):
        return CandidateMatch()
    position = tuple(float(value) for value in positions[index])
    orientation = (
        tuple(float(value) for value in orientations[index])
        if index < len(orientations) else None
    )
    identity = metadata[index] if index < len(metadata) else (-1, -1, -1)
    error_world = tuple(float(value) for value in deltas[index])
    error_local = None
    if orientation is not None:
        rotation = t3d.quaternions.quat2mat(orientation)
        error_local = tuple(float(value) for value in rotation.T @ np.asarray(error_world))
    orientation_error = (
        float(orientation_errors[index])
        if index < len(orientation_errors) else np.nan
    )
    return CandidateMatch(
        index=int(index),
        contact_point_index=int(identity[0]),
        arc_sample_index=int(identity[1]),
        finger_flip_index=int(identity[2]),
        position_world=position,
        orientation_wxyz=orientation,
        error_world=error_world,
        error_local=error_local,
        distance_m=float(np.linalg.norm(error_world)),
        orientation_error_deg=orientation_error,
    )


@dataclass(frozen=True)
class EffectorEvidence:
    tcp_position_world: tuple[float, float, float] | None = None
    # target_distance_m and target_vertical_offset_m are now measured
    # against the SAME reference point: whichever candidate in this arm's
    # expanded grasp-pose family (or the AABB/pose fallback, if no
    # annotation is available) is nearest to the TCP in full 3D. Before
    # this fix, distance was measured to the object's own pose/center while
    # vertical offset was measured to the annotated grasp-pose candidates
    # independently -- for an object whose grasp point sits well off-center,
    # that let a fully height- and orientation-aligned TCP still read as
    # "not close enough" against an unrelated reference (confirmed in a
    # live batch: seed 40002 stalled at 10.36cm object-center distance while
    # height/orientation were aligned for 300+ frames). target_horizontal_offset_m
    # is the same nearest candidate's horizontal (x/y) component, exported
    # for diagnostics -- nothing currently gates on it.
    target_distance_m: float = np.nan
    target_vertical_offset_m: float = np.nan
    target_horizontal_offset_m: float = np.nan
    selected_candidate_index: int = -1
    selected_contact_point_index: int = -1
    selected_arc_sample_index: int = -1
    selected_finger_flip_index: int = -1
    selected_candidate_position_world: tuple[float, float, float] | None = None
    selected_candidate_orientation_wxyz: tuple[float, float, float, float] | None = None
    selected_candidate_error_world: tuple[float, float, float] | None = None
    selected_candidate_error_local: tuple[float, float, float] | None = None
    selected_candidate_orientation_error_deg: float = np.nan
    orientation_best_candidate: CandidateMatch | None = None
    joint_best_candidate: CandidateMatch | None = None
    joint_best_selection_status: str = "unavailable"
    grasp_height_aligned: bool = False
    target_orientation_error_deg: float = np.nan
    # Fails open (True) when no reference orientation is available, so
    # objects without annotated grasp geometry behave exactly as before
    # this check existed.
    grasp_orientation_aligned: bool = True
    target_contact: bool = False
    target_held: bool = False



def _close_geometry_ready(effector: EffectorEvidence) -> bool:
    """Authorize closure without changing unannotated-object fallback."""
    if not (effector.grasp_height_aligned and effector.grasp_orientation_aligned):
        return False
    if effector.joint_best_selection_status != "orientation_band_then_nearest":
        return effector.target_distance_m <= GRASP_CLOSE_MAX_DISTANCE_M
    match = effector.joint_best_candidate
    error_local = match.error_local if match is not None else None
    if error_local is None:
        return False
    lateral_error_m = float(np.linalg.norm(error_local[1:]))
    if lateral_error_m > ANNOTATED_GRASP_LATERAL_ERROR_MAX_M:
        return False
    return bool(
        effector.target_contact
        or (
            ANNOTATED_GRASP_APPROACH_ERROR_MIN_M
            <= error_local[0]
            <= ANNOTATED_GRASP_APPROACH_ERROR_MAX_M
            and effector.target_distance_m
            <= ANNOTATED_GRASP_CLOSE_MAX_DISTANCE_M
        )
    )


@dataclass(frozen=True)
class LiveTaskState:
    """Compatibility projection of evidence used by legacy adapter callers."""

    target_held: bool
    held_arm: str | None
    target_inside_destination: bool
    release_ready: bool


@dataclass(frozen=True)
class PlacementGeometry:
    """Footprint-aware target geometry relative to its destination."""

    stage: str = "align_destination"
    aligned: bool = False
    descent_ready: bool = False
    offset_x_m: float = np.nan
    offset_y_m: float = np.nan
    horizontal_safe_margin_m: float = np.nan
    target_bottom_to_destination_rim_m: float = np.nan


def placement_geometry_from_bounds(
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    destination_lower: np.ndarray,
    destination_upper: np.ndarray,
    relation: str,
) -> PlacementGeometry:
    """Classify placement using the target footprint, not only its center."""
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (target_lower, target_upper, destination_lower, destination_upper)
    )
    if relation not in {"in", "on"} or any(
        value.shape != (3,) or not np.all(np.isfinite(value)) for value in arrays
    ):
        return PlacementGeometry()
    target_lower, target_upper, destination_lower, destination_upper = arrays
    if np.any(target_upper < target_lower) or np.any(destination_upper < destination_lower):
        return PlacementGeometry()

    target_center = 0.5 * (target_lower + target_upper)
    destination_center = 0.5 * (destination_lower + destination_upper)
    target_half_footprint = 0.5 * (target_upper[:2] - target_lower[:2])
    safe_lower = destination_lower[:2] + target_half_footprint + PLACEMENT_HORIZONTAL_MARGIN_M
    safe_upper = destination_upper[:2] - target_half_footprint - PLACEMENT_HORIZONTAL_MARGIN_M
    clearances = np.concatenate(
        (target_center[:2] - safe_lower, safe_upper - target_center[:2])
    )
    safe_margin = float(np.min(clearances))
    aligned = bool(np.all(safe_lower <= safe_upper) and safe_margin >= 0.0)
    bottom_to_rim = float(target_lower[2] - destination_upper[2])
    if relation == "in":
        descent_ready = (
            aligned
            and bottom_to_rim <= PLACEMENT_RELEASE_ABOVE_RIM_CLEARANCE_M
        )
    else:
        descent_ready = aligned and abs(bottom_to_rim) <= PLACEMENT_ON_VERTICAL_TOLERANCE_M
    stage = "release_ready" if descent_ready else (
        "final_descent" if aligned else "align_destination"
    )
    return PlacementGeometry(
        stage=stage,
        aligned=aligned,
        descent_ready=descent_ready,
        offset_x_m=float(target_center[0] - destination_center[0]),
        offset_y_m=float(target_center[1] - destination_center[1]),
        horizontal_safe_margin_m=safe_margin,
        target_bottom_to_destination_rim_m=bottom_to_rim,
    )


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
    placement_geometry: PlacementGeometry
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
        grasp_effector = (
            getattr(self, self.grasp_arm) if self.grasp_arm is not None else None
        )
        grasp_lateral_error_m = np.nan
        if (
            grasp_effector is not None
            and grasp_effector.joint_best_candidate is not None
            and grasp_effector.joint_best_candidate.error_local is not None
        ):
            grasp_lateral_error_m = float(np.linalg.norm(
                grasp_effector.joint_best_candidate.error_local[1:]
            ))
        return ActionGraphState(
            held=self.held,
            held_contact=held_contact,
            destination_aligned=(
                Evidence.TRUE if self.placement_geometry.aligned else Evidence.FALSE
            ),
            release_ready=Evidence.TRUE if self.release_ready else Evidence.FALSE,
            goal_satisfied=self.goal_satisfied,
            path_blocked=self.path_blocked,
            reachable=self.reachable,
            visible=self.visible,
            held_arm=self.held_arm,
            grasp_substage=self.grasp_substage,
            grasp_arm=self.grasp_arm,
            grasp_close_immediate=self.grasp_close_immediate,
            grasp_target_contact=(
                grasp_effector.target_contact if grasp_effector is not None else False
            ),
            grasp_height_aligned=(
                grasp_effector.grasp_height_aligned
                if grasp_effector is not None else False
            ),
            grasp_orientation_aligned=(
                grasp_effector.grasp_orientation_aligned
                if grasp_effector is not None else False
            ),
            grasp_lateral_error_m=grasp_lateral_error_m,
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
            "placement_stage": self.placement_geometry.stage,
            "destination_aligned": self.placement_geometry.aligned,
            "target_to_destination_dx_m": self.placement_geometry.offset_x_m,
            "target_to_destination_dy_m": self.placement_geometry.offset_y_m,
            "horizontal_safe_margin_m": self.placement_geometry.horizontal_safe_margin_m,
            "target_bottom_to_destination_rim_m": self.placement_geometry.target_bottom_to_destination_rim_m,
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
                f"target_{arm}_horizontal_offset": effector.target_horizontal_offset_m,
                f"target_{arm}_selected_candidate_index": effector.selected_candidate_index,
                f"target_{arm}_selected_contact_point_index": effector.selected_contact_point_index,
                f"target_{arm}_selected_arc_sample_index": effector.selected_arc_sample_index,
                f"target_{arm}_selected_finger_flip_index": effector.selected_finger_flip_index,
                f"target_{arm}_selected_candidate_position_world": effector.selected_candidate_position_world,
                f"target_{arm}_selected_candidate_orientation_wxyz": effector.selected_candidate_orientation_wxyz,
                f"target_{arm}_selected_candidate_error_world": effector.selected_candidate_error_world,
                f"target_{arm}_selected_candidate_error_local": effector.selected_candidate_error_local,
                f"target_{arm}_selected_candidate_orientation_error_deg": effector.selected_candidate_orientation_error_deg,
                f"target_{arm}_orientation_best_candidate_index": (effector.orientation_best_candidate or CandidateMatch()).index,
                f"target_{arm}_orientation_best_contact_point_index": (effector.orientation_best_candidate or CandidateMatch()).contact_point_index,
                f"target_{arm}_orientation_best_arc_sample_index": (effector.orientation_best_candidate or CandidateMatch()).arc_sample_index,
                f"target_{arm}_orientation_best_finger_flip_index": (effector.orientation_best_candidate or CandidateMatch()).finger_flip_index,
                f"target_{arm}_orientation_best_position_world": (effector.orientation_best_candidate or CandidateMatch()).position_world,
                f"target_{arm}_orientation_best_orientation_wxyz": (effector.orientation_best_candidate or CandidateMatch()).orientation_wxyz,
                f"target_{arm}_orientation_best_error_world": (effector.orientation_best_candidate or CandidateMatch()).error_world,
                f"target_{arm}_orientation_best_error_local": (effector.orientation_best_candidate or CandidateMatch()).error_local,
                f"target_{arm}_orientation_best_distance": (effector.orientation_best_candidate or CandidateMatch()).distance_m,
                f"target_{arm}_orientation_best_orientation_error_deg": (effector.orientation_best_candidate or CandidateMatch()).orientation_error_deg,
                f"target_{arm}_joint_best_candidate_index": (effector.joint_best_candidate or CandidateMatch()).index,
                f"target_{arm}_joint_best_contact_point_index": (effector.joint_best_candidate or CandidateMatch()).contact_point_index,
                f"target_{arm}_joint_best_arc_sample_index": (effector.joint_best_candidate or CandidateMatch()).arc_sample_index,
                f"target_{arm}_joint_best_finger_flip_index": (effector.joint_best_candidate or CandidateMatch()).finger_flip_index,
                f"target_{arm}_joint_best_position_world": (effector.joint_best_candidate or CandidateMatch()).position_world,
                f"target_{arm}_joint_best_orientation_wxyz": (effector.joint_best_candidate or CandidateMatch()).orientation_wxyz,
                f"target_{arm}_joint_best_error_world": (effector.joint_best_candidate or CandidateMatch()).error_world,
                f"target_{arm}_joint_best_error_local": (effector.joint_best_candidate or CandidateMatch()).error_local,
                f"target_{arm}_joint_best_distance": (effector.joint_best_candidate or CandidateMatch()).distance_m,
                f"target_{arm}_joint_best_orientation_error_deg": (effector.joint_best_candidate or CandidateMatch()).orientation_error_deg,
                f"target_{arm}_joint_best_selection_status": effector.joint_best_selection_status,
                f"{arm}_tcp_pose_source": f"task_env.robot.get_{arm}_tcp_pose",
                f"{arm}_tcp_relation_object_id": EFFECTOR_IDS[arm],
                "grasp_reference_target_dis_m": 0.0,
                "grasp_reference_approach_standoff_m": GRASP_APPROACH_STANDOFF_M,
                "annotated_grasp_close_max_distance_m": ANNOTATED_GRASP_CLOSE_MAX_DISTANCE_M,
                "annotated_grasp_approach_error_min_m": ANNOTATED_GRASP_APPROACH_ERROR_MIN_M,
                "annotated_grasp_approach_error_max_m": ANNOTATED_GRASP_APPROACH_ERROR_MAX_M,
                "annotated_grasp_lateral_error_max_m": ANNOTATED_GRASP_LATERAL_ERROR_MAX_M,
                "grasp_reference_target_dis_source": "grasp_actor default grasp_dis/choose_grasp_pose target_dis",
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
    # Fallback candidate when no annotated grasp-pose family is available:
    # the object's own (x, y), with z preferring the AABB top over the pose
    # center -- exactly the old height-only fallback's z, but now a full
    # position so distance and vertical offset share this same one point
    # too, instead of distance silently reverting to a different (pose-only)
    # reference than height's own fallback.
    fallback_position_m = (
        (float(target_pose[0]), float(target_pose[1]), target_height_center)
        if target_pose is not None and np.isfinite(target_height_center)
        else None
    )
    aabb_fallback_positions_m = (
        (fallback_position_m,) if fallback_position_m is not None else ()
    )
    effectors = {}
    target_index = context.index_by_id.get(target_id) if target_id is not None else None
    for arm_index, arm in enumerate(("left", "right")):
        effector_index = context.index_by_id.get(EFFECTOR_IDS[arm])
        effector_pose = retriever._poses_world.get(EFFECTOR_IDS[arm])
        # Prefer this arm's own annotated grasp-pose family (rotate_lim,
        # hence the reachable position range, is arm-specific -- see
        # expand_grasp_pose_family) over the AABB/pose fallback.
        position_candidates_m = (
            context.left_reference_grasp_positions_m
            if arm == "left"
            else context.right_reference_grasp_positions_m
        ) or aabb_fallback_positions_m
        orientation_candidates = (
            context.left_reference_orientations_wxyz
            if arm == "left" else context.right_reference_orientations_wxyz
        )
        candidate_metadata = (
            context.left_reference_candidate_metadata
            if arm == "left" else context.right_reference_candidate_metadata
        )
        # Pick ONE reference candidate -- whichever is nearest in full 3D --
        # and derive distance, vertical offset, AND horizontal offset from
        # THAT SAME candidate. Before this fix, distance picked the object's
        # own pose/center while vertical offset independently picked
        # whichever candidate's height matched best; for an object whose
        # annotated grasp point sits well off-center, that let a TCP read as
        # height- and orientation-aligned while distance (measured to an
        # unrelated point) never converged -- confirmed in a live batch
        # (seed 40002: 300+ frames of aligned height/orientation, stalled at
        # 10.36cm object-center distance, 3.6mm over the close threshold).
        deltas_m = (
            [
                np.asarray(effector_pose[:3], dtype=np.float64)
                - np.asarray(candidate, dtype=np.float64)
                for candidate in position_candidates_m
            ]
            if effector_pose is not None
            else []
        )
        distances_m = [float(np.linalg.norm(delta)) for delta in deltas_m]
        nearest_index = int(np.argmin(distances_m)) if distances_m else None
        target_distance = distances_m[nearest_index] if nearest_index is not None else np.nan
        vertical_offset = (
            float(deltas_m[nearest_index][2]) if nearest_index is not None else np.nan
        )
        horizontal_offset = (
            float(np.linalg.norm(deltas_m[nearest_index][:2]))
            if nearest_index is not None else np.nan
        )
        selected_position = (
            tuple(float(value) for value in position_candidates_m[nearest_index])
            if nearest_index is not None else None
        )
        selected_orientation = (
            tuple(float(value) for value in orientation_candidates[nearest_index])
            if nearest_index is not None and nearest_index < len(orientation_candidates)
            else None
        )
        selected_metadata = (
            candidate_metadata[nearest_index]
            if nearest_index is not None and nearest_index < len(candidate_metadata)
            else (-1, -1, -1)
        )
        selected_error_world = (
            tuple(float(value) for value in deltas_m[nearest_index])
            if nearest_index is not None else None
        )
        selected_error_local = None
        if selected_error_world is not None and selected_orientation is not None:
            candidate_rotation = t3d.quaternions.quat2mat(selected_orientation)
            selected_error_local = tuple(
                float(value)
                for value in candidate_rotation.T @ np.asarray(selected_error_world)
            )
        height_aligned = (
            nearest_index is not None
            and abs(vertical_offset) <= target_height_half_width
        )
        effector_quat = (
            effector_pose[3:7]
            if effector_pose is not None and effector_pose.shape[0] >= 7
            else None
        )
        candidate_orientation_errors = [
            _quaternion_angle_deg(effector_quat, candidate)
            for candidate in orientation_candidates
        ] if effector_quat is not None else []
        finite_orientation_indices = [
            index for index, error in enumerate(candidate_orientation_errors)
            if np.isfinite(error)
        ]
        orientation_best_index = (
            min(finite_orientation_indices, key=lambda index: candidate_orientation_errors[index])
            if finite_orientation_indices else None
        )
        orientation_eligible_indices = [
            index for index in finite_orientation_indices
            if candidate_orientation_errors[index] <= GRASP_ORIENTATION_MAX_ERROR_DEG
        ]
        if orientation_eligible_indices:
            joint_best_index = min(
                orientation_eligible_indices, key=lambda index: distances_m[index]
            )
            joint_best_status = "orientation_band_then_nearest"
        elif orientation_best_index is not None:
            joint_best_index = orientation_best_index
            joint_best_status = "fallback_min_orientation"
        elif nearest_index is not None:
            joint_best_index = nearest_index
            joint_best_status = "position_only_fallback"
        else:
            joint_best_index = None
            joint_best_status = "unavailable"
        orientation_best_match = _candidate_match(
            orientation_best_index, position_candidates_m, orientation_candidates,
            candidate_metadata, deltas_m, candidate_orientation_errors,
        )
        joint_best_match = _candidate_match(
            joint_best_index, position_candidates_m, orientation_candidates,
            candidate_metadata, deltas_m, candidate_orientation_errors,
        )
        # Behavior-changing v5 control reference: use the coherent joint
        # candidate, not the independently nearest position. Position-best
        # remains exported above for comparison only.
        target_distance = joint_best_match.distance_m
        vertical_offset = (
            joint_best_match.error_world[2]
            if joint_best_match.error_world is not None else np.nan
        )
        horizontal_offset = (
            float(np.linalg.norm(joint_best_match.error_world[:2]))
            if joint_best_match.error_world is not None else np.nan
        )
        height_aligned = bool(
            np.isfinite(vertical_offset)
            and abs(vertical_offset) <= target_height_half_width
        )
        control_orientation_aligned = joint_best_status in (
            "orientation_band_then_nearest", "position_only_fallback"
        )
        orientation_error = (
            min(candidate_orientation_errors[index] for index in finite_orientation_indices)
            if finite_orientation_indices else np.nan
        )
        selected_orientation_error = (
            _quaternion_angle_deg(effector_quat, selected_orientation)
            if effector_quat is not None and selected_orientation is not None
            else np.nan
        )
        effectors[arm] = EffectorEvidence(
            tcp_position_world=(
                tuple(float(value) for value in effector_pose[:3])
                if effector_pose is not None else None
            ),
            target_distance_m=target_distance,
            target_vertical_offset_m=vertical_offset,
            target_horizontal_offset_m=horizontal_offset,
            selected_candidate_index=(nearest_index if nearest_index is not None else -1),
            selected_contact_point_index=int(selected_metadata[0]),
            selected_arc_sample_index=int(selected_metadata[1]),
            selected_finger_flip_index=int(selected_metadata[2]),
            selected_candidate_position_world=selected_position,
            selected_candidate_orientation_wxyz=selected_orientation,
            selected_candidate_error_world=selected_error_world,
            selected_candidate_error_local=selected_error_local,
            selected_candidate_orientation_error_deg=selected_orientation_error,
            orientation_best_candidate=orientation_best_match,
            joint_best_candidate=joint_best_match,
            joint_best_selection_status=joint_best_status,
            grasp_height_aligned=bool(height_aligned),
            target_orientation_error_deg=orientation_error,
            grasp_orientation_aligned=bool(control_orientation_aligned),
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

    placement_geometry = _placement_geometry(context, target_id)
    release_ready = held_arm is not None and placement_geometry.descent_ready
    finite_arms = [
        arm for arm in ("left", "right")
        if np.isfinite(effectors[arm].target_distance_m)
    ]
    close_arms = [arm for arm in finite_arms if _close_geometry_ready(effectors[arm])]
    approach_arms = [
        arm for arm in finite_arms
        if effectors[arm].grasp_orientation_aligned
        and effectors[arm].target_distance_m <= GRASP_ALIGNMENT_DISTANCE_M
    ]
    orientation_misaligned_arms = [
        arm for arm in finite_arms
        if not effectors[arm].grasp_orientation_aligned
        and effectors[arm].target_distance_m <= GRASP_ALIGNMENT_DISTANCE_M
    ]
    if close_arms:
        grasp_substage = GraspSubstage.CLOSE
        grasp_arm = min(
            close_arms, key=lambda arm: effectors[arm].target_distance_m
        )
        close_effector = effectors[grasp_arm]
        annotated_close = (
            close_effector.joint_best_selection_status
            == "orientation_band_then_nearest"
        )
        # Raw contact is not sufficient to accelerate annotated closure.
        # GraphControllerState grants immediate CLOSE only after contact
        # persists with valid height, orientation, and lateral alignment.
        grasp_close_immediate = bool(
            not annotated_close
            and close_effector.target_distance_m
            <= GRASP_CLOSE_PREFERRED_DISTANCE_M
        )
    elif approach_arms:
        grasp_substage = GraspSubstage.GRASP_APPROACH
        grasp_arm = min(
            approach_arms, key=lambda arm: effectors[arm].target_distance_m
        )
        grasp_close_immediate = False
    elif orientation_misaligned_arms:
        grasp_substage = GraspSubstage.ORIENTATION_ALIGN
        grasp_arm = min(
            orientation_misaligned_arms,
            key=lambda arm: effectors[arm].target_distance_m,
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
        placement_geometry=placement_geometry,
        left=effectors["left"],
        right=effectors["right"],
        grasp_substage=grasp_substage,
        grasp_arm=grasp_arm,
        grasp_close_immediate=grasp_close_immediate,
        grasp_height_half_width_m=target_height_half_width,
        orientation_reference_status=context.orientation_reference_status,
        orientation_reference_count=context.orientation_reference_count,
    )


def _placement_geometry(
    context: "LiveGraphContext", target_id: int | None
) -> PlacementGeometry:
    destination_ids = context.goal.destination_ids
    if target_id is None or len(destination_ids) != 1:
        return PlacementGeometry()
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
        return PlacementGeometry()
    return placement_geometry_from_bounds(
        aabb_lower[target_index],
        aabb_upper[target_index],
        aabb_lower[destination_index],
        aabb_upper[destination_index],
        context.goal.relation,
    )
