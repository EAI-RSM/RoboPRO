"""Live observation adapter for the graph-conditioned pi0.5 POC."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Any, Iterable

import numpy as np
import transforms3d as t3d

from .action_intent import (
    ActionIntent,
    IntentOperation,
    MotionDirection,
    PlacementRelation,
    PlacementSubstage,
)

from .contract import (
    INVERSE_RELATIONS,
    RELATION_PRIORITY,
    SYMMETRIC_RELATIONS,
    VALIDITY_FIELD,
    GraphFact,
    GraphNode,
    InputCondition,
    RetrievalContract,
    stable_aliases,
)
from .graph_serializer import PackedItem
from .graph_replanning import ActionGraphState, Evidence, GraspSubstage, TaskGoal
from .simulator_evidence import (
    GRASP_APPROACH_STANDOFF_M,
    LiveTaskState,
    SimulatorEvidence,
    expand_grasp_pose_family,
    extract_simulator_evidence,
    placement_geometry_from_bounds,
)


IMMINENT_GRIPPER_OBSTACLE_CLEARANCE_M = 0.20


def _resolve_wrapped_actor(task_env: Any, name: str) -> Any | None:
    """Find the wrapped ``Actor`` (with annotated grasp geometry) for ``name``.

    ``task_env.scene.get_all_actors()`` returns raw SAPIEN entities, which lack
    ``iter_contact_points()``. The task-specific wrapped ``Actor`` objects (the
    ones annotated with per-object contact-point poses) live as plain instance
    attributes on the task (e.g. ``self.bottle``), so they have to be found by
    scanning ``__dict__`` and matching on name -- there is no id-keyed lookup.
    """
    for value in vars(task_env).values():
        if (
            hasattr(value, "iter_contact_points")
            and hasattr(value, "get_name")
            and value.get_name() == name
        ):
            return value
    return None


# RoboTwin's own ``get_grasp_pose`` (customized_robotwin/envs/_base_task.py)
# does not use an object's raw annotated contact-point orientation as the
# grasp orientation directly -- it right-multiplies the contact point's
# world transform by this fixed matrix (a valid rotation, exactly 120
# degrees) before reading off the quaternion that actually gets used as an
# IK/grasp target. Any code that wants "the object's valid grasp
# orientation" must apply the same transform, not the raw contact-point
# pose, or it silently compares against a reference ~120 degrees away from
# the real one. Duplicated here rather than imported from _base_task.py to
# keep this experiment decoupled from core simulator harness internals --
# if RoboTwin's own transform ever changes, this constant must follow it.
CONTACT_POINT_TO_GRASP_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# Sanity-check tolerances for a contact-point pose matrix -- generous
# margin for typical float32 mesh/annotation precision (~1e-6), not a
# physically calibrated bound. A matrix failing these isn't a valid rigid
# transform at all (degenerate, a reflection, or outright garbage), so no
# quaternion derived from it can mean anything, regardless of what
# mat2quat happens to return for it.
CONTACT_MATRIX_ORTHONORMALITY_ATOL = 1e-3
CONTACT_MATRIX_DETERMINANT_ATOL = 1e-3
CONTACT_MATRIX_HOMOGENEOUS_ROW_ATOL = 1e-6


@dataclass(frozen=True)
class GraspPose:
    position_world: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    # The raw annotated contact point's own position (before the approach
    # standoff is added) -- the pivot create_target_pose_list rotates
    # position_world around for each rotate_lim candidate. Needed because
    # that rotation is NOT a self-rotation of position_world: the offset
    # from this center to position_world has magnitude
    # GRASP_APPROACH_STANDOFF_M, so rotating it moves position_world along
    # an arc of that radius, not just changing its orientation in place.
    contact_center_world: tuple[float, float, float]
    contact_point_index: int = -1


def grasp_pose_from_contact_matrix(contact_matrix: Any) -> GraspPose | None:
    """One annotated contact point's world-frame pose -> its actual grasp
    pose (position AND orientation), reproducing get_grasp_pose's full
    formula -- not just the orientation remap.

    Shared by any caller (this task or another) that needs "the valid grasp
    pose for this object", not just the sauce-can gate -- callers should
    reuse this rather than re-deriving the contact-to-grasp transform
    inline, so it can't silently drift out of sync in two places.

    Never raises, and never returns a pose for input that isn't actually a
    valid rigid transform: ``transforms3d.quaternions.mat2quat`` does not
    validate its input -- it raises ``LinAlgError`` for a non-finite matrix
    (which would otherwise propagate out of ``build_live_graph_context`` and
    halt live evaluation on a single bad annotation), and silently returns a
    normalized-looking but physically meaningless quaternion for a
    reflection, a degenerate matrix, or an arbitrary non-rotation matrix
    alike (which would otherwise read as a perfectly ordinary, seemingly
    valid ``AVAILABLE`` reference pose). A malformed matrix returns None
    instead, same as an absent one -- callers already treat an all-None
    contact-point set as ``ANNOTATION_INVALID``.
    """
    if contact_matrix is None:
        return None
    try:
        matrix = np.asarray(contact_matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            return None
        if not np.allclose(
            matrix[3, :], (0.0, 0.0, 0.0, 1.0),
            atol=CONTACT_MATRIX_HOMOGENEOUS_ROW_ATOL,
        ):
            return None
        rotation = matrix[:3, :3]
        if not np.allclose(
            rotation @ rotation.T, np.eye(3), atol=CONTACT_MATRIX_ORTHONORMALITY_ATOL
        ):
            return None
        if abs(np.linalg.det(rotation) - 1.0) > CONTACT_MATRIX_DETERMINANT_ATOL:
            return None
        grasp_matrix = matrix @ CONTACT_POINT_TO_GRASP_ROTATION
        grasp_rotation = grasp_matrix[:3, :3]
        contact_center = grasp_matrix[:3, 3]
        position = contact_center + grasp_rotation @ np.array(
            [-GRASP_APPROACH_STANDOFF_M, 0.0, 0.0]
        )
        quat = t3d.quaternions.mat2quat(grasp_rotation)
        quat = tuple(float(value) for value in quat)
        position = tuple(float(value) for value in position)
        contact_center = tuple(float(value) for value in contact_center)
        if len(quat) != 4 or not all(np.isfinite(quat)):
            return None
        if len(position) != 3 or not all(np.isfinite(position)):
            return None
        if len(contact_center) != 3 or not all(np.isfinite(contact_center)):
            return None
        return GraspPose(
            position_world=position,
            orientation_wxyz=quat,
            contact_center_world=contact_center,
        )
    except Exception:
        return None


def grasp_quat_wxyz_from_contact_matrix(
    contact_matrix: Any,
) -> tuple[float, float, float, float] | None:
    """Orientation-only view of ``grasp_pose_from_contact_matrix``, kept for
    existing callers that only need orientation. Prefer
    ``grasp_pose_from_contact_matrix`` directly for new code that needs the
    position too, so validation only happens in one place.
    """
    pose = grasp_pose_from_contact_matrix(contact_matrix)
    return pose.orientation_wxyz if pose is not None else None


class OrientationReferenceStatus(str, Enum):
    """Why ``target_grasp_contact_orientations_wxyz`` returned what it did.

    A bare empty tuple collapses several very different situations into one
    indistinguishable "no reference" result -- a genuinely un-annotated
    object looks identical to a bug in actor resolution or a malformed
    annotation. That's fine for control (fail-open either way), but fatal
    for trusting the diagnostic data this treatment exists to collect: a
    silent 100%-``target_unresolved`` batch would look, from the trace
    alone, exactly like a real "this task has no annotated objects" result.
    """

    AVAILABLE = "available"
    # The goal itself didn't resolve to exactly one target id (task_goal_from_env
    # / get_role_names) -- a target name was never even looked up.
    TARGET_UNRESOLVED = "target_unresolved"
    # The target's name resolved, but _resolve_wrapped_actor couldn't find its
    # wrapped Actor object at all. Distinct from ANNOTATION_MISSING: this
    # points at actor-lookup/resolution logic, not at the object's own
    # (possibly genuinely absent) annotation.
    ACTOR_UNRESOLVED = "actor_unresolved"
    # The actor resolved, but has zero annotated contact points.
    ANNOTATION_MISSING = "annotation_missing"
    ANNOTATION_INVALID = "annotation_invalid"
    EXTRACTION_ERROR = "extraction_error"


@dataclass(frozen=True)
class OrientationReference:
    orientations_wxyz: tuple[tuple[float, float, float, float], ...] = ()
    status: OrientationReferenceStatus = OrientationReferenceStatus.TARGET_UNRESOLVED


@dataclass(frozen=True)
class GraspPoseReference:
    poses: tuple[GraspPose, ...] = ()
    status: OrientationReferenceStatus = OrientationReferenceStatus.TARGET_UNRESOLVED


def target_grasp_poses(
    task_env: Any, target_name: str | None
) -> GraspPoseReference:
    """Every annotated contact point's actual grasp pose (position AND
    orientation) for the target -- the one canonical resolution both the
    orientation check and the height reference draw from, so they can't
    independently resolve the same actor two different ways and drift.

    Pure geometry read off the object's static ``contact_points_pose``
    annotation (``Actor.get_contact_point``) plus RoboTwin's own
    contact-to-grasp transform -- this does not call
    ``get_grasp_pose``/``choose_grasp_pose`` and triggers no motion planning,
    so it is safe to call every observation frame. An empty result (status
    != AVAILABLE) means "no reference available" and callers must treat
    that as such -- but which status it is matters for diagnosing whether
    that's expected (ANNOTATION_MISSING) or a bug (ACTOR_UNRESOLVED/
    EXTRACTION_ERROR).
    """
    if not target_name:
        return GraspPoseReference((), OrientationReferenceStatus.TARGET_UNRESOLVED)
    actor = _resolve_wrapped_actor(task_env, target_name)
    if actor is None:
        return GraspPoseReference((), OrientationReferenceStatus.ACTOR_UNRESOLVED)
    try:
        contact_points = list(actor.iter_contact_points("matrix"))
    except Exception:
        return GraspPoseReference((), OrientationReferenceStatus.EXTRACTION_ERROR)
    if not contact_points:
        return GraspPoseReference((), OrientationReferenceStatus.ANNOTATION_MISSING)
    poses = []
    for contact_point_index, matrix in contact_points:
        pose = grasp_pose_from_contact_matrix(matrix)
        if pose is not None:
            poses.append(GraspPose(
                position_world=pose.position_world,
                orientation_wxyz=pose.orientation_wxyz,
                contact_center_world=pose.contact_center_world,
                contact_point_index=int(contact_point_index),
            ))
    if not poses:
        return GraspPoseReference((), OrientationReferenceStatus.ANNOTATION_INVALID)
    return GraspPoseReference(tuple(poses), OrientationReferenceStatus.AVAILABLE)


def target_grasp_contact_orientations_wxyz(
    task_env: Any, target_name: str | None
) -> OrientationReference:
    """Orientation-only view of ``target_grasp_poses``, kept for existing
    callers/tests. Prefer ``target_grasp_poses`` directly for new code that
    needs position too, so actor resolution only happens in one place.
    """
    reference = target_grasp_poses(task_env, target_name)
    return OrientationReference(
        tuple(pose.orientation_wxyz for pose in reference.poses),
        reference.status,
    )


def _decode(values: Any) -> list[str]:
    return [
        value.decode("utf-8", errors="replace")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in np.asarray(values).tolist()
    ]


@dataclass(frozen=True)
class PreparedInstruction:
    instruction: str
    retrieved_node_count: int
    selected_node_count: int
    dropped_node_count: int
    retrieved_fact_count: int
    selected_fact_count: int
    dropped_fact_count: int
    graph_token_count: int
    full_prompt_token_count_estimate: int
    destination_seed_available: bool
    prompt_phase: str = "grasp"
    action_intent: ActionIntent | None = None


@dataclass(frozen=True)
class LiveGraphContext:
    """One parsed view of a single live observation, shared by its consumers."""

    catalog: tuple[dict[str, Any], ...]
    relation_state: dict[str, Any]
    object_state: dict[str, Any]
    goal: TaskGoal
    retriever: "LiveGraphRetriever"
    index_by_id: dict[int, int]
    contract: RetrievalContract
    # Each arm's own fully symmetry-expanded set of valid grasp poses for
    # the target -- genuinely separate per arm, not a shared tuple.
    # RoboTwin's own candidate search is arm-specific (rotate_lim can differ
    # per arm, and its preferred-direction scoring is arm-mirrored -- see
    # expand_grasp_pose_family's docstring for what is and isn't reproduced
    # here), so a pose valid for one arm's search is not automatically
    # valid for the other's. This applies to position too, not just
    # orientation: rotate_lim rotates the seed's POSITION as an offset from
    # the raw contact point, not just its orientation in place, so the set
    # of reachable positions along the arc is exactly as arm-specific as the
    # set of reachable orientations.
    left_reference_orientations_wxyz: tuple[tuple[float, float, float, float], ...] = ()
    right_reference_orientations_wxyz: tuple[tuple[float, float, float, float], ...] = ()
    # Full candidate positions, not just heights: simulator_evidence derives
    # distance, vertical offset, AND horizontal offset from the SAME nearest
    # candidate in this tuple, rather than each metric picking its own
    # best-fit candidate independently (which is how a height-aligned,
    # orientation-aligned TCP could still read as "not close enough" against
    # an unrelated object-center distance -- see simulator_evidence.py).
    left_reference_grasp_positions_m: tuple[tuple[float, float, float], ...] = ()
    right_reference_grasp_positions_m: tuple[tuple[float, float, float], ...] = ()
    # Parallel identity metadata: (contact point, arc sample, finger flip).
    left_reference_candidate_metadata: tuple[tuple[int, int, int], ...] = ()
    right_reference_candidate_metadata: tuple[tuple[int, int, int], ...] = ()
    # Why the tuples above are empty (or aren't) -- see
    # OrientationReferenceStatus. Exported to the trace precisely so a
    # smoke test can't silently look like it exercised orientation/height
    # checking when actor resolution or annotation extraction was actually
    # failing the whole time.
    orientation_reference_status: str = OrientationReferenceStatus.TARGET_UNRESOLVED.value
    orientation_reference_count: int = 0


def task_goal_from_env(task_env: Any, catalog: Iterable[dict[str, Any]]) -> TaskGoal:
    """Resolve explicit object IDs and the placement relation for control."""
    catalog = tuple(catalog)
    roles = task_env.get_role_names() if hasattr(task_env, "get_role_names") else {}
    roles = roles if isinstance(roles, dict) else {}
    target_ids = set()
    if roles.get("target_id") is not None:
        target_ids.add(int(roles["target_id"]))
    target_ids.update(int(value) for value in roles.get("target_ids", ()))
    if not target_ids:
        target_ids.update(
            int(entry["object_id"]) for entry in catalog if bool(entry.get("is_target"))
        )
    destination_ids = destination_ids_from_task(task_env, catalog)
    relation = roles.get("goal_relation") or getattr(
        task_env, "benchmark_goal_relation", None
    )
    if relation is None:
        instruction = str(task_env.get_instruction()).lower()
        relation = "on" if re.search(r"\b(?:on|onto)\b", instruction) else "in"
    return TaskGoal(tuple(sorted(target_ids)), destination_ids, str(relation).lower())


def build_live_graph_context(
    task_env: Any,
    observation: dict[str, Any],
    contract: RetrievalContract,
) -> LiveGraphContext:
    """Parse graph support exactly once for one observation frame."""
    support = observation.get("benchmark_support") or {}
    relation_state = support.get("relation_state")
    object_state = support.get("object_state")
    if relation_state is None or object_state is None:
        raise ValueError("Live observation is missing benchmark graph support")
    catalog = tuple(task_env._get_benchmark_object_catalog())
    goal = task_goal_from_env(task_env, catalog)
    retriever = LiveGraphRetriever(
        catalog, relation_state, object_state, contract
    )
    retriever.is_target = np.isin(retriever.object_ids, goal.target_ids)
    target_name = None
    if len(goal.target_ids) == 1:
        target_name = next(
            (
                entry.get("name") for entry in catalog
                if int(entry.get("object_id", -(2**62))) == goal.target_ids[0]
            ),
            None,
        )
    # Resolved once, not once per arm: this determines
    # orientation_reference_status/_count, and both arms must report the
    # same underlying resolution outcome even though their expanded pose
    # families (rotate_lim can differ per arm, and affects both the
    # orientation and the position reference) do not.
    reference = target_grasp_poses(task_env, target_name)
    left_orientations, left_positions, left_metadata = _expand_arm_reference_poses(
        reference.poses, task_env, "left"
    )
    right_orientations, right_positions, right_metadata = _expand_arm_reference_poses(
        reference.poses, task_env, "right"
    )
    return LiveGraphContext(
        catalog=catalog,
        relation_state=relation_state,
        object_state=object_state,
        goal=goal,
        retriever=retriever,
        index_by_id={
            int(object_id): index
            for index, object_id in enumerate(retriever.object_ids)
        },
        contract=contract,
        left_reference_orientations_wxyz=left_orientations,
        right_reference_orientations_wxyz=right_orientations,
        left_reference_grasp_positions_m=left_positions,
        right_reference_grasp_positions_m=right_positions,
        left_reference_candidate_metadata=left_metadata,
        right_reference_candidate_metadata=right_metadata,
        orientation_reference_status=reference.status.value,
        orientation_reference_count=len(reference.poses),
    )


def _effector_rotate_lim_rad(task_env: Any, arm: str) -> tuple[float, float]:
    """One arm's jaw-axis rotation tolerance, read live from the robot object.

    ``Robot.__init__`` (customized_robotwin/envs/robot/robot.py) sets
    ``left_rotate_lim``/``right_rotate_lim`` from the embodiment config
    (``rotate_lim`` in e.g. benchmark/assets/embodiments/*/config.yml),
    defaulting to ``[0, 0]`` -- no arc -- when unconfigured. Read live and
    per-arm rather than hardcoded/shared, since RoboTwin's own is arm-
    specific (it can genuinely differ between arms for other embodiments,
    even though it happens to be equal for both arms of aloha-agilex).
    """
    robot = getattr(task_env, "robot", None)
    limits = getattr(robot, f"{arm}_rotate_lim", None) if robot is not None else None
    if limits is not None and len(limits) == 2 and all(np.isfinite(limits)):
        return (float(limits[0]), float(limits[1]))
    return (0.0, 0.0)


def _expand_arm_reference_poses(
    poses: tuple[GraspPose, ...],
    task_env: Any,
    arm: str,
) -> tuple[
    tuple[tuple[float, float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int, int], ...],
]:
    """One arm's full expanded family of valid grasp poses, split into the
    orientation tuple and the full position tuple callers actually consume.

    Each annotated contact point's grasp pose (already resolved once by the
    caller, not re-derived per arm) is expanded through that arm's own
    rotate_lim arc (genuinely arm-specific, and -- see
    ``expand_grasp_pose_family`` -- affecting both orientation and position,
    since the arc rotates the seed's position as an offset from the raw
    contact point, not just its orientation in place) and the
    approach-axis flip (arm-independent, a property of the gripper, and
    position-neutral since it's a true self-rotation).

    Returns full (x, y, z) positions, not just their z component: callers
    that only care about height can still take ``position[2]``, but
    simulator_evidence also needs the full position to measure distance
    against the same candidate it measures height against, instead of an
    unrelated object-center reference.
    """
    rotate_lim = _effector_rotate_lim_rad(task_env, arm)
    orientations = []
    positions = []
    metadata = []
    for contact_index, pose in enumerate(poses):
        family = expand_grasp_pose_family(
            pose.position_world, pose.orientation_wxyz, pose.contact_center_world,
            rotate_lim,
        )
        for family_index, (position, orientation) in enumerate(family):
            orientations.append(orientation)
            positions.append(position)
            annotation_index = (
                pose.contact_point_index if pose.contact_point_index >= 0 else contact_index
            )
            metadata.append((annotation_index, family_index // 2, family_index % 2))
    return tuple(orientations), tuple(positions), tuple(metadata)


def action_graph_state(
    task_env: Any,
    observation: dict[str, Any],
    contract: RetrievalContract,
    context: LiveGraphContext | None = None,
    evidence: SimulatorEvidence | None = None,
) -> ActionGraphState:
    """Extract action-relevant graph predicates with three-valued validity."""
    context = context or build_live_graph_context(task_env, observation, contract)
    evidence = evidence or extract_simulator_evidence(context)
    return evidence.action_graph_state()


def _round1(value: float) -> float:
    rounded = round(float(value), 1)
    return 0.0 if rounded == 0 else rounded


def _round2(value: float) -> float:
    rounded = round(float(value), 2)
    return 0.0 if rounded == 0 else rounded


_SIMULATOR_LABEL_PREFIXES = ("task_", "target_", "object_", "model_")


def vla_label_from_catalog_entry(entry: dict[str, Any]) -> str:
    """Return a natural label while keeping simulator identity out of prompts.

    A semantic label that differs from the simulator name is treated as an
    explicit catalog override. Otherwise, only known infrastructure prefixes
    and a trailing numeric instance identifier are removed. Directional and
    descriptive words such as ``left``, ``right``, and colors are preserved;
    aliases remain the source of identity when normalized labels are equal.
    """
    name = str(entry.get("name") or "").strip()
    semantic = str(entry.get("semantic_label") or "").strip()
    value = semantic if semantic and semantic != name else name or semantic
    if not value:
        return "object"

    if not (semantic and semantic != name):
        lowered = value.lower()
        for prefix in _SIMULATOR_LABEL_PREFIXES:
            if lowered.startswith(prefix):
                value = value[len(prefix):]
                break
        value = re.sub(r"[_-]\d+$", "", value)

    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if value and not value.isdigit() else "object"


def _rotation_matrix_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("robot-base quaternion must be four finite wxyz values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("robot-base quaternion has zero norm")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def goal_geometry_pack_item(
    target_alias: str,
    destination_alias: str,
    target_world: np.ndarray,
    destination_world: np.ndarray,
    robot_pose_world: np.ndarray,
    containment: str,
) -> PackedItem:
    """Build mandatory task geometry in the robot-base x/y/z frame.

    Robot-base +x is described as forward, +y as left, and +z as up.
    The delta points from the target to the destination, i.e. the direction
    the target must move to satisfy the placement goal.
    """
    target = np.asarray(target_world, dtype=np.float64)
    destination = np.asarray(destination_world, dtype=np.float64)
    robot_pose = np.asarray(robot_pose_world, dtype=np.float64)
    if target.shape != (3,) or destination.shape != (3,) or robot_pose.shape != (7,):
        raise ValueError("goal geometry requires target/destination xyz and robot xyz+wxyz")
    if not all(np.all(np.isfinite(value)) for value in (target, destination, robot_pose)):
        raise ValueError("goal geometry poses must be finite")
    rotation_world_from_base = _rotation_matrix_from_wxyz(robot_pose[3:7])
    forward, left, up = rotation_world_from_base.T @ (destination - target)

    def component(value: float, positive: str, negative: str) -> str:
        direction = positive if value >= 0 else negative
        return f"{direction} {abs(value):.2f}m"

    text = (
        f"{target_alias} to {destination_alias} (robot base): "
        f"{component(forward, 'forward', 'backward')}, "
        f"{component(left, 'left', 'right')}, "
        f"{component(up, 'up', 'down')}; "
        f"{containment}."
    )
    return PackedItem(
        text=text,
        rank=len(RELATION_PRIORITY),  # below held_by, above all other relations
        section="goal",
        requires=(target_alias, destination_alias),
        mandatory=True,
    )


def live_task_state(
    task_env: Any,
    observation: dict[str, Any],
    contract: RetrievalContract,
    context: LiveGraphContext | None = None,
    evidence: SimulatorEvidence | None = None,
) -> LiveTaskState:
    """Read event predicates used to interrupt an open-loop action chunk."""
    context = context or build_live_graph_context(task_env, observation, contract)
    evidence = evidence or extract_simulator_evidence(context)
    return evidence.live_task_state()


def keep_active_gripper_closed(action: np.ndarray, held_arm: str | None) -> np.ndarray:
    """Latch the grasp command while transporting an already-held target."""
    result = np.asarray(action, dtype=np.float64).copy()
    if result.shape != (14,):
        raise ValueError(f"expected a 14-dimensional pi0.5 action, got {result.shape}")
    if held_arm == "left":
        result[6] = 0.0
    elif held_arm == "right":
        result[13] = 0.0
    elif held_arm is not None:
        raise ValueError(f"unknown held arm: {held_arm}")
    return result


def keep_active_gripper_open(action: np.ndarray, arm: str | None) -> np.ndarray:
    """Keep an approaching gripper open until guarded close is authorized."""
    result = np.asarray(action, dtype=np.float64).copy()
    if result.shape != (14,):
        raise ValueError(f"expected a 14-dimensional pi0.5 action, got {result.shape}")
    if arm == "left":
        result[6] = 1.0
    elif arm == "right":
        result[13] = 1.0
    elif arm is not None:
        raise ValueError(f"unknown approach arm: {arm}")
    return result


def placement_intent(
    retriever: "LiveGraphRetriever",
    destination_object_ids: Iterable[int],
    relation: str = "in",
) -> ActionIntent:
    """Build structured target-to-destination placement intent."""
    destination_ids = tuple(map(int, destination_object_ids))
    target_ids = [
        int(object_id)
        for object_id, flag in zip(retriever.object_ids, retriever.is_target)
        if bool(flag)
    ]
    if len(target_ids) != 1 or len(destination_ids) != 1:
        raise ValueError("compact placement guidance requires one target and one destination")
    target_id, destination_id = target_ids[0], destination_ids[0]
    target_pose = retriever._poses_world.get(target_id)
    destination_pose = retriever._poses_world.get(destination_id)
    robot_pose = retriever._poses_world.get(-1)
    if target_pose is None or destination_pose is None or robot_pose is None:
        raise ValueError("compact placement guidance requires target, destination, and robot poses")
    rotation_world_from_base = _rotation_matrix_from_wxyz(robot_pose[3:7])
    forward, left, _ = rotation_world_from_base.T @ (
        destination_pose[:3] - target_pose[:3]
    )
    # Suppress centimeter-scale jitter and express only coarse task geometry.
    directions = []
    if abs(forward) >= 0.05:
        directions.append(
            MotionDirection.FORWARD if forward > 0 else MotionDirection.BACKWARD
        )
    if abs(left) >= 0.05:
        directions.append(MotionDirection.LEFT if left > 0 else MotionDirection.RIGHT)
    destination_label = retriever._label_by_id.get(destination_id, "destination")
    target_bounds = retriever._aabb_bounds.get(target_id)
    destination_bounds = retriever._aabb_bounds.get(destination_id)
    placement_substage = PlacementSubstage.ALIGN_DESTINATION
    if target_bounds is not None and destination_bounds is not None:
        geometry = placement_geometry_from_bounds(
            target_bounds[0], target_bounds[1],
            destination_bounds[0], destination_bounds[1], relation,
        )
        if geometry.aligned:
            placement_substage = PlacementSubstage.FINAL_DESCENT
    return ActionIntent(
        operation=IntentOperation.PLACE,
        target_label=retriever._label_by_id.get(target_id, "object"),
        destination_label=destination_label,
        placement_relation=PlacementRelation(relation),
        motion_directions=tuple(directions),
        placement_substage=placement_substage,
    )


def compact_placement_hint(
    retriever: "LiveGraphRetriever",
    destination_object_ids: Iterable[int],
    relation: str = "in",
) -> str:
    """Compatibility wrapper rendering structured placement intent."""
    return placement_intent(
        retriever, destination_object_ids, relation
    ).render_stage_instruction()


def grasp_intent(
    retriever: "LiveGraphRetriever",
    target_id: int,
) -> ActionIntent:
    """Build grasp intent from validated arm-specific obstacle evidence."""
    target_indices = np.flatnonzero(retriever.object_ids == int(target_id))
    by_effector = np.asarray(
        retriever.state.get("blocks_by_effector", ()), dtype=np.bool_
    )
    valid = np.asarray(
        retriever.state.get(
            "blocks_by_effector_valid", np.zeros_like(by_effector)
        ),
        dtype=np.bool_,
    )
    target_label = retriever._label_by_id[int(target_id)]
    fallback = ActionIntent(IntentOperation.GRASP, target_label)
    if (
        len(target_indices) != 1
        or by_effector.ndim != 3
        or valid.shape != by_effector.shape
        or by_effector.shape[2] != len(retriever.blocks_effector_names)
    ):
        return fallback

    target_index = int(target_indices[0])
    blocks = np.asarray(retriever.state.get("blocks", ()), dtype=np.bool_)
    blocks_valid = np.asarray(
        retriever.state.get("blocks_valid", np.zeros_like(blocks)),
        dtype=np.bool_,
    )
    if blocks.ndim != 2 or blocks_valid.shape != blocks.shape:
        return fallback
    blocker_indices = np.flatnonzero(
        blocks[:, target_index] & blocks_valid[:, target_index]
    )
    if not len(blocker_indices):
        return fallback

    evidence = []
    for effector_index in range(by_effector.shape[2]):
        corridor_valid = valid[
            blocker_indices, target_index, effector_index
        ]
        if not np.all(corridor_valid):
            evidence.append(Evidence.UNKNOWN)
        elif np.any(
            by_effector[blocker_indices, target_index, effector_index]
        ):
            evidence.append(Evidence.TRUE)
        else:
            evidence.append(Evidence.FALSE)

    if evidence.count(Evidence.FALSE) != 1:
        return fallback
    clear_index = evidence.index(Evidence.FALSE)
    if any(
        state is not Evidence.TRUE
        for index, state in enumerate(evidence)
        if index != clear_index
    ):
        return fallback
    effector_name = retriever.blocks_effector_names[clear_index].lower()
    side = (
        "left" if "left" in effector_name
        else "right" if "right" in effector_name
        else ""
    )
    if not side:
        return fallback
    approach = ActionIntent(
        IntentOperation.GRASP, target_label, preferred_arm=side
    )
    blocked_index = next(
        index for index, state in enumerate(evidence)
        if state is Evidence.TRUE
    )
    active_blockers = blocker_indices[
        by_effector[blocker_indices, target_index, blocked_index]
        & valid[blocker_indices, target_index, blocked_index]
    ]
    blocked_name = retriever.blocks_effector_names[blocked_index].lower()
    blocked_side = (
        "left" if "left" in blocked_name
        else "right" if "right" in blocked_name
        else ""
    )
    blocked_effector_id = -2 if blocked_side == "left" else -3
    blocked_effector_pose = retriever._poses_world.get(blocked_effector_id)
    if not blocked_side or blocked_effector_pose is None or not len(active_blockers):
        return approach
    positioned = [
        int(index) for index in active_blockers
        if int(retriever.object_ids[index]) in retriever._aabb_bounds
    ]
    if not positioned:
        return approach

    def point_aabb_clearance(index: int) -> float:
        lower, upper = retriever._aabb_bounds[
            int(retriever.object_ids[index])
        ]
        point = blocked_effector_pose[:3]
        separation = np.maximum(np.maximum(lower - point, point - upper), 0.0)
        return float(np.linalg.norm(separation))

    blocker_index = min(
        positioned,
        key=point_aabb_clearance,
    )
    if point_aabb_clearance(blocker_index) > IMMINENT_GRIPPER_OBSTACLE_CLEARANCE_M:
        return approach
    blocker_id = int(retriever.object_ids[blocker_index])
    blocker_label = retriever._label_by_id.get(blocker_id, "obstacle")
    return ActionIntent(
        IntentOperation.GRASP,
        target_label,
        preferred_arm=side,
        blocked_arm=blocked_side,
        obstacle_label=blocker_label,
        collision_imminent=True,
    )


def compact_grasp_hint(
    retriever: "LiveGraphRetriever",
    target_id: int,
) -> str:
    """Compatibility wrapper rendering structured grasp intent."""
    return grasp_intent(retriever, target_id).render_stage_instruction()


def action_intent_for_phase(
    phase: str,
    retriever: "LiveGraphRetriever",
    target_id: int,
    destination_id: int,
    relation: str,
    grasp_substage: GraspSubstage = GraspSubstage.ALIGN,
    grasp_arm: str | None = None,
) -> ActionIntent:
    """Create one validated atomic intent for the controller's current phase."""
    if phase == "grasp":
        intent = grasp_intent(retriever, target_id)
        return replace(
            intent,
            grasp_substage=grasp_substage,
            preferred_arm=(grasp_arm or intent.preferred_arm),
        )
    if phase == "placement":
        return placement_intent(retriever, (destination_id,), relation)
    if phase == "release":
        return ActionIntent(
            operation=IntentOperation.RELEASE,
            target_label=retriever._label_by_id.get(target_id, "object"),
            destination_label=retriever._label_by_id.get(
                destination_id, "destination"
            ),
            placement_relation=PlacementRelation(relation),
        )
    raise ValueError(f"unknown graph prompt phase: {phase}")


class LiveGraphRetriever:
    """Retrieve current-frame nodes and facts from an observation snapshot.

    The object catalog comes directly from the task environment. Action nodes
    and action outcomes are deliberately not accepted by this interface.
    """

    def __init__(
        self,
        object_catalog: list[dict[str, Any]],
        relation_state: dict[str, Any],
        object_state: dict[str, Any],
        contract: RetrievalContract | None = None,
    ):
        self.contract = contract or RetrievalContract()
        self.state = relation_state
        self.object_ids = np.asarray(
            [entry["object_id"] for entry in object_catalog], dtype=np.int64
        )
        relation_ids = np.asarray(relation_state["object_ids"], dtype=np.int64)
        if not np.array_equal(self.object_ids, relation_ids):
            raise ValueError("Object catalog and live relation-state IDs are not aligned")
        object_state_ids = np.asarray(object_state["object_ids"], dtype=np.int64)
        if not np.array_equal(self.object_ids, object_state_ids):
            raise ValueError("Object catalog and live object-state IDs are not aligned")

        self.labels = [
            vla_label_from_catalog_entry(entry)
            for entry in object_catalog
        ]
        self.kinds = [entry.get("entity_kind", "object") for entry in object_catalog]
        self.is_target = np.asarray(
            [bool(entry.get("is_target")) for entry in object_catalog], dtype=np.bool_
        )
        self.effector_names = _decode(relation_state["held_by_effector_names"])
        self.reachable_names = _decode(relation_state["reachable_by_effector_names"])
        self.camera_names = _decode(relation_state["visible_to_camera_names"])
        self.blocks_effector_names = _decode(
            relation_state.get("blocks_effector_names", [])
        )

        self._label_by_id = {
            int(object_id): label for object_id, label in zip(self.object_ids.tolist(), self.labels)
        }
        self._kind_by_id = {
            int(object_id): kind for object_id, kind in zip(self.object_ids.tolist(), self.kinds)
        }
        self._id_by_name = {
            entry.get("name"): int(entry["object_id"])
            for entry in object_catalog
            if entry.get("name")
        }

        n = len(self.object_ids)
        is_present = np.asarray(
            object_state.get("is_present", np.ones(n, dtype=np.bool_)), dtype=np.bool_
        )
        pose_world = np.asarray(object_state["pose_world"], dtype=np.float64)
        has_aabb = np.asarray(
            object_state.get("has_aabb", np.zeros(n, dtype=np.bool_)), dtype=np.bool_
        )
        aabb_lower = np.asarray(
            object_state.get("aabb_lower", np.zeros((n, 3))), dtype=np.float64
        )
        aabb_upper = np.asarray(
            object_state.get("aabb_upper", np.zeros((n, 3))), dtype=np.float64
        )

        self._positions: dict[int, tuple[float, float, float]] = {}
        self._poses_world: dict[int, np.ndarray] = {}
        self._bbox_sizes: dict[int, tuple[float, float, float] | None] = {}
        self._aabb_bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for index, object_id in enumerate(self.object_ids.tolist()):
            object_id = int(object_id)
            if index < len(is_present) and not bool(is_present[index]):
                continue
            self._poses_world[object_id] = pose_world[index, :7].copy()
            self._positions[object_id] = tuple(
                _round1(value) for value in pose_world[index, :3]
            )
            if index < len(has_aabb) and bool(has_aabb[index]):
                self._aabb_bounds[object_id] = (
                    aabb_lower[index].copy(),
                    aabb_upper[index].copy(),
                )
                self._bbox_sizes[object_id] = tuple(
                    _round2(upper - lower)
                    for lower, upper in zip(aabb_lower[index], aabb_upper[index])
                )
            else:
                self._bbox_sizes[object_id] = None

    def seed_indices(self, destination_object_ids: Iterable[int] = ()) -> set[int]:
        seeds = set(np.flatnonzero(self.is_target).astype(int).tolist())
        index_by_id = {
            int(object_id): index for index, object_id in enumerate(self.object_ids)
        }
        for object_id in (-2, -3, *destination_object_ids):
            index = index_by_id.get(int(object_id))
            if index is not None:
                seeds.add(index)
        return seeds

    def _nodes_for(self, object_ids: set[int]) -> list[GraphNode]:
        nodes = []
        for object_id in sorted(object_ids):
            position = self._positions.get(object_id)
            if position is None:
                continue  # object absent this frame; nothing grounded to declare
            nodes.append(
                GraphNode(
                    object_id=object_id,
                    label=(
                        "left end effector" if object_id == -2
                        else "right end effector" if object_id == -3
                        else self._label_by_id.get(object_id, str(object_id))
                    ),
                    kind=self._kind_by_id.get(object_id, "object"),
                    position=position,
                    bbox_size=self._bbox_sizes.get(object_id),
                    alias=self._aliases[object_id],
                )
            )
        alias_order = {"T": 0, "D": 1, "O": 2, "L": 3, "R": 4}
        return sorted(
            nodes,
            key=lambda node: (
                alias_order.get(node.alias[0], 5),
                int(node.alias[1:]) if node.alias[1:].isdigit() else 0,
                node.object_id,
            ),
        )

    def _valid(self, relation: str, values: np.ndarray) -> np.ndarray:
        validity_name = VALIDITY_FIELD.get(relation)
        if validity_name is None or validity_name not in self.state:
            return np.ones(values.shape, dtype=np.bool_)
        valid = np.asarray(self.state[validity_name], dtype=np.bool_)
        if valid.shape != values.shape:
            raise ValueError(
                f"{validity_name} shape {valid.shape} does not match "
                f"{relation} shape {values.shape}"
            )
        return valid

    def retrieve(
        self, destination_object_ids: Iterable[int] = ()
    ) -> tuple[list[GraphNode], list[GraphFact]]:
        destination_object_ids = tuple(int(value) for value in destination_object_ids)
        seeds = self.seed_indices(destination_object_ids)
        self._aliases = stable_aliases(
            self.object_ids, self.is_target, destination_object_ids
        )
        self.labels = [self._aliases[int(object_id)] for object_id in self.object_ids]
        # Seed nodes are always declared, even if a target ends up with zero
        # true relations this frame.
        participating_ids = {int(self.object_ids[index]) for index in seeds}
        facts: list[GraphFact] = []
        for priority, relation in enumerate(RELATION_PRIORITY):
            if relation in INVERSE_RELATIONS and not self.contract.include_inverse_relations:
                continue
            if relation not in self.state:
                continue
            values = np.asarray(self.state[relation], dtype=np.bool_)
            if relation == "occludes":
                facts.extend(self._camera_facts(priority, relation, values, seeds, participating_ids))
            elif relation in {"held_by", "reachable_by", "visible_to"}:
                facts.extend(self._bipartite_facts(priority, relation, values, seeds, participating_ids))
            elif values.ndim == 2:
                facts.extend(self._binary_facts(priority, relation, values, seeds, participating_ids))
        unique = {fact.key(): fact for fact in facts}
        ordered_facts = sorted(unique.values())
        return self._nodes_for(participating_ids), ordered_facts

    def goal_items(self, destination_object_ids: Iterable[int]) -> list[PackedItem]:
        """Return mandatory target-to-destination placement geometry."""
        if not hasattr(self, "_aliases"):
            raise RuntimeError("retrieve() must be called before goal_items()")
        destination_object_ids = tuple(map(int, destination_object_ids))
        if not destination_object_ids:
            return []
        robot_pose = self._poses_world.get(-1)
        if robot_pose is None:
            raise ValueError("Live object state is missing the robot-base pose")
        index_by_id = {
            int(object_id): index for index, object_id in enumerate(self.object_ids)
        }
        inside = np.asarray(self.state.get("in", ()), dtype=np.bool_)
        inside_valid = np.asarray(
            self.state.get("containment_valid", np.zeros_like(inside)), dtype=np.bool_
        )
        result = []
        targets = [
            int(object_id) for object_id, flag in zip(self.object_ids, self.is_target)
            if bool(flag)
        ]
        for target_id in targets:
            for destination_id in destination_object_ids:
                target_pose = self._poses_world.get(target_id)
                destination_pose = self._poses_world.get(destination_id)
                if target_pose is None or destination_pose is None:
                    raise ValueError("target or destination pose is unavailable for goal geometry")
                source_index = index_by_id[target_id]
                destination_index = index_by_id[destination_id]
                containment = "outside"
                if (
                    inside.ndim == 2
                    and inside_valid.shape == inside.shape
                    and bool(inside_valid[source_index, destination_index])
                    and bool(inside[source_index, destination_index])
                ):
                    containment = "inside"
                result.append(
                    goal_geometry_pack_item(
                        self._aliases[target_id],
                        self._aliases[destination_id],
                        target_pose[:3],
                        destination_pose[:3],
                        robot_pose,
                        containment,
                    )
                )
        return result

    def _binary_facts(
        self,
        priority: int,
        relation: str,
        values: np.ndarray,
        seeds: set[int],
        participating_ids: set[int],
    ) -> list[GraphFact]:
        valid = self._valid(relation, values)
        result = []
        for source, destination in np.argwhere(np.logical_and(values, valid)):
            source, destination = int(source), int(destination)
            if source not in seeds and destination not in seeds:
                continue
            if relation in SYMMETRIC_RELATIONS and destination <= source:
                continue
            qualifier = ""
            if relation == "blocks" and "blocks_by_effector" in self.state:
                per_effector = np.asarray(
                    self.state["blocks_by_effector"][source, destination], dtype=np.bool_
                )
                if "blocks_by_effector_valid" in self.state:
                    per_effector = np.logical_and(
                        per_effector,
                        np.asarray(
                            self.state["blocks_by_effector_valid"][source, destination],
                            dtype=np.bool_,
                        ),
                    )
                qualifier = ",".join(
                    self._effector_alias(name)
                    for name, flag in zip(self.blocks_effector_names, per_effector)
                    if flag
                )
            participating_ids.add(int(self.object_ids[source]))
            participating_ids.add(int(self.object_ids[destination]))
            result.append(
                GraphFact(
                    priority,
                    relation,
                    self.labels[source],
                    self.labels[destination],
                    qualifier,
                    tuple(
                        dict.fromkeys(
                            (self.labels[source], self.labels[destination])
                            + tuple(value for value in qualifier.split(",") if value)
                        )
                    ),
                )
            )
        return result

    def _bipartite_facts(
        self,
        priority: int,
        relation: str,
        values: np.ndarray,
        seeds: set[int],
        participating_ids: set[int],
    ) -> list[GraphFact]:
        valid = self._valid(relation, values)
        labels = (
            self.camera_names
            if relation == "visible_to"
            else self.effector_names
            if relation == "held_by"
            else self.reachable_names
        )
        result = []
        for source, destination in np.argwhere(np.logical_and(values, valid)):
            source, destination = int(source), int(destination)
            if source not in seeds or destination >= len(labels):
                continue
            if relation == "visible_to" and labels[destination] != self.contract.default_camera:
                continue
            destination_label = labels[destination]
            rendered_destination = (
                destination_label
                if relation == "visible_to"
                else self._effector_alias(destination_label)
            )
            participating_ids.add(int(self.object_ids[source]))
            resolved_id = self._id_by_name.get(destination_label)
            if resolved_id is not None:
                participating_ids.add(resolved_id)
            result.append(
                GraphFact(
                    priority,
                    relation,
                    self.labels[source],
                    rendered_destination,
                    "",
                    (self.labels[source],)
                    if relation == "visible_to"
                    else (self.labels[source], rendered_destination),
                )
            )
        return result

    def _camera_facts(
        self,
        priority: int,
        relation: str,
        values: np.ndarray,
        seeds: set[int],
        participating_ids: set[int],
    ) -> list[GraphFact]:
        if self.contract.default_camera not in self.camera_names:
            return []
        camera_index = self.camera_names.index(self.contract.default_camera)
        matrix = values[:, :, camera_index]
        valid = self._valid(relation, values)[:, :, camera_index]
        result = []
        for source, destination in np.argwhere(np.logical_and(matrix, valid)):
            source, destination = int(source), int(destination)
            if source not in seeds and destination not in seeds:
                continue
            participating_ids.add(int(self.object_ids[source]))
            participating_ids.add(int(self.object_ids[destination]))
            result.append(
                GraphFact(
                    priority,
                    relation,
                    self.labels[source],
                    self.labels[destination],
                    self.contract.default_camera,
                    (self.labels[source], self.labels[destination]),
                )
            )
        return result

    @staticmethod
    def _effector_alias(name: str) -> str:
        lowered = name.lower()
        if "left" in lowered:
            return "L"
        if "right" in lowered:
            return "R"
        return name


def destination_ids_from_task(
    task_env: Any, object_catalog: Iterable[dict[str, Any]] = ()
) -> tuple[int, ...]:
    """Resolve destination roles from task metadata, never action nodes.

    Concrete IDs are authoritative. Older task environments may only expose
    destination names, which are accepted only when a name identifies exactly
    one catalog entry.
    """
    try:
        roles = task_env.get_role_names()
    except (AttributeError, TypeError):
        return ()
    if not isinstance(roles, dict):
        return ()

    resolved = set()
    destination = roles.get("destination_id")
    if destination is not None:
        resolved.add(int(destination))
    resolved.update(int(value) for value in (roles.get("destination_ids") or ()))
    if resolved:
        return tuple(sorted(resolved))

    ids_by_name: dict[str, set[int]] = {}
    for entry in object_catalog:
        object_id = int(entry["object_id"])
        for field in ("name", "semantic_label"):
            value = entry.get(field)
            if value:
                ids_by_name.setdefault(str(value), set()).add(object_id)

    for name in roles.get("destination_object_names") or ():
        matches = ids_by_name.get(str(name), set())
        if len(matches) > 1:
            raise ValueError(
                f"Destination name {name!r} is ambiguous in the object catalog; "
                "the task must expose destination_ids"
            )
        resolved.update(matches)
    return tuple(sorted(resolved))


def prepare_instruction(
    task_env: Any,
    model: Any,
    observation: dict[str, Any],
    condition: InputCondition | str,
    contract: RetrievalContract,
    previous_phase: str = "grasp",
    grasp_substage: GraspSubstage | str = GraspSubstage.ALIGN,
    grasp_arm: str | None = None,
) -> PreparedInstruction:
    base_instruction = str(task_env.get_instruction())
    grasp_substage = GraspSubstage(grasp_substage)
    condition = InputCondition(condition)
    if condition is InputCondition.VISUAL_ONLY:
        return PreparedInstruction(
            instruction=base_instruction,
            retrieved_node_count=0,
            selected_node_count=0,
            dropped_node_count=0,
            retrieved_fact_count=0,
            selected_fact_count=0,
            dropped_fact_count=0,
            graph_token_count=0,
            full_prompt_token_count_estimate=0,
            destination_seed_available=False,
            prompt_phase="grasp",
        )

    support = observation.get("benchmark_support") or {}
    relation_state = support.get("relation_state")
    if relation_state is None:
        raise ValueError("Live observation is missing benchmark_support/relation_state")
    object_state = support.get("object_state")
    if object_state is None:
        raise ValueError("Live observation is missing benchmark_support/object_state")
    catalog = task_env._get_benchmark_object_catalog()
    goal = task_goal_from_env(task_env, catalog)
    destination_ids = goal.destination_ids
    retriever = LiveGraphRetriever(catalog, relation_state, object_state, contract)
    retriever.is_target = np.isin(retriever.object_ids, goal.target_ids)
    nodes, facts = retriever.retrieve(destination_ids)
    if previous_phase not in {"grasp", "placement", "release"}:
        raise ValueError(f"unknown graph prompt phase: {previous_phase}")
    target_count = int(np.count_nonzero(retriever.is_target))
    guidance_supported = target_count == 1 and len(destination_ids) == 1
    prompt_phase = previous_phase
    if not guidance_supported:
        prompt_phase = "grasp"
    # Separate the persistent objective from one atomic current-stage command.
    # Use the same two-field natural-language schema in every supported phase.
    guidance_items = []
    action_intent = None
    if guidance_supported:
        target_id = next(
            int(object_id)
            for object_id, is_target in zip(retriever.object_ids, retriever.is_target)
            if bool(is_target)
        )
        action_intent = action_intent_for_phase(
            prompt_phase,
            retriever,
            target_id,
            destination_ids[0],
            goal.relation,
            grasp_substage=grasp_substage,
            grasp_arm=grasp_arm,
        )
        guidance = action_intent.render_stage_instruction()
        guidance_items.append(
            PackedItem(
                text=f"Current stage: {guidance}",
                rank=len(RELATION_PRIORITY),
                section="goal",
                mandatory=True,
            )
        )
    if not guidance_items:
        return PreparedInstruction(
            instruction=base_instruction,
            retrieved_node_count=0,
            selected_node_count=0,
            dropped_node_count=0,
            retrieved_fact_count=0,
            selected_fact_count=0,
            dropped_fact_count=0,
            graph_token_count=0,
            full_prompt_token_count_estimate=0,
            destination_seed_available=bool(destination_ids),
            prompt_phase=prompt_phase,
            action_intent=action_intent,
        )
    items = [
        {
            "text": item.text,
            "rank": item.rank,
            "section": item.section,
            "provides": item.provides,
            "requires": list(item.requires),
            "mandatory": item.mandatory,
        }
        for item in guidance_items
    ]
    response = model.fit_graph_prompt(
        {
            "instruction": f"Task objective: {base_instruction}",
            "state": observation["joint_action"]["vector"],
            "items": items,
            "graph_token_budget": contract.graph_token_budget,
        }
    )
    selected_node_count = int(response["selected_node_count"])
    selected_fact_count = int(response["selected_fact_count"])
    return PreparedInstruction(
        instruction=str(response["instruction"]),
        retrieved_node_count=0,
        selected_node_count=selected_node_count,
        dropped_node_count=0,
        retrieved_fact_count=len(guidance_items),
        selected_fact_count=selected_fact_count,
        dropped_fact_count=len(guidance_items) - selected_fact_count,
        graph_token_count=int(response["graph_token_count"]),
        full_prompt_token_count_estimate=int(response["full_prompt_token_count_estimate"]),
        destination_seed_available=bool(destination_ids),
        prompt_phase=prompt_phase,
        action_intent=action_intent,
    )
