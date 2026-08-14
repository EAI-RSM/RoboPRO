"""Live observation adapter for the graph-conditioned pi0.5 POC."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

import numpy as np

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
from .graph_replanning import ActionGraphState, Evidence, TaskGoal


RELEASE_READY_VERTICAL_CLEARANCE_M = 0.10
IMMINENT_GRIPPER_OBSTACLE_CLEARANCE_M = 0.20


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


@dataclass(frozen=True)
class LiveTaskState:
    target_held: bool
    held_arm: str | None
    target_inside_destination: bool
    release_ready: bool


def _aggregate_evidence(values: np.ndarray, valid: np.ndarray) -> Evidence:
    """Reduce a relevant relation slice without converting unknown to false."""
    values = np.asarray(values, dtype=np.bool_)
    valid = np.asarray(valid, dtype=np.bool_)
    if values.shape != valid.shape:
        raise ValueError("relation values and validity mask are not aligned")
    if not np.any(valid):
        return Evidence.UNKNOWN
    return Evidence.TRUE if np.any(values & valid) else Evidence.FALSE


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


def action_graph_state(
    task_env: Any,
    observation: dict[str, Any],
    contract: RetrievalContract,
) -> ActionGraphState:
    """Extract action-relevant graph predicates with three-valued validity."""
    support = observation.get("benchmark_support") or {}
    relation_state = support.get("relation_state")
    object_state = support.get("object_state")
    if relation_state is None or object_state is None:
        raise ValueError("Live observation is missing benchmark graph support")
    catalog = task_env._get_benchmark_object_catalog()
    retriever = LiveGraphRetriever(catalog, relation_state, object_state, contract)
    goal = task_goal_from_env(task_env, catalog)
    index_by_id = {
        int(object_id): index for index, object_id in enumerate(retriever.object_ids)
    }
    target_indices = [index_by_id[value] for value in goal.target_ids if value in index_by_id]
    destination_indices = [
        index_by_id[value] for value in goal.destination_ids if value in index_by_id
    ]
    if not target_indices or not destination_indices:
        raise ValueError("task goal IDs are absent from the live relation state")

    held = np.asarray(relation_state.get("held_by", ()), dtype=np.bool_)
    held_valid = retriever._valid("held_by", held) if held.ndim == 2 else np.zeros_like(held)
    held_evidence = _aggregate_evidence(held[target_indices], held_valid[target_indices])
    held_arm = None
    if held_evidence is Evidence.TRUE:
        for target_index in target_indices:
            active = np.flatnonzero(held[target_index] & held_valid[target_index])
            if len(active):
                name = retriever.effector_names[int(active[0])].lower()
                held_arm = "left" if "left" in name else "right" if "right" in name else None
                break

    goal_values = np.asarray(relation_state.get(goal.relation, ()), dtype=np.bool_)
    if goal_values.ndim == 2:
        goal_valid = retriever._valid(goal.relation, goal_values)
        selection = np.ix_(target_indices, destination_indices)
        goal_evidence = _aggregate_evidence(goal_values[selection], goal_valid[selection])
    else:
        goal_evidence = Evidence.UNKNOWN

    reachable = np.asarray(relation_state.get("reachable_by", ()), dtype=np.bool_)
    reachable_evidence = Evidence.UNKNOWN
    if reachable.ndim == 2:
        reachable_valid = retriever._valid("reachable_by", reachable)
        reachable_evidence = _aggregate_evidence(
            reachable[target_indices], reachable_valid[target_indices]
        )

    visible = np.asarray(relation_state.get("visible_to", ()), dtype=np.bool_)
    visible_evidence = Evidence.UNKNOWN
    if visible.ndim == 2 and contract.default_camera in retriever.camera_names:
        camera_index = retriever.camera_names.index(contract.default_camera)
        visible_valid = retriever._valid("visible_to", visible)
        visible_evidence = _aggregate_evidence(
            visible[target_indices, camera_index],
            visible_valid[target_indices, camera_index],
        )

    blocks = np.asarray(relation_state.get("blocks", ()), dtype=np.bool_)
    blocked_evidence = Evidence.UNKNOWN
    if blocks.ndim == 2:
        blocks_valid = retriever._valid("blocks", blocks)
        blocked_evidence = _aggregate_evidence(
            blocks[:, target_indices], blocks_valid[:, target_indices]
        )

    legacy = live_task_state(task_env, observation, contract)
    return ActionGraphState(
        held=held_evidence,
        release_ready=(
            Evidence.TRUE if legacy.release_ready else Evidence.FALSE
        ),
        goal_satisfied=goal_evidence,
        path_blocked=blocked_evidence,
        reachable=reachable_evidence,
        visible=visible_evidence,
        held_arm=held_arm,
    )


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


def target_is_held(retriever: "LiveGraphRetriever") -> bool:
    """Return whether any target has valid current-frame gripper evidence."""
    held = np.asarray(retriever.state.get("held_by", ()), dtype=np.bool_)
    if held.ndim != 2:
        return False
    valid = retriever._valid("held_by", held)
    for index, is_target in enumerate(retriever.is_target):
        if bool(is_target) and index < held.shape[0] and np.any(held[index] & valid[index]):
            return True
    return False


def live_task_state(
    task_env: Any,
    observation: dict[str, Any],
    contract: RetrievalContract,
) -> LiveTaskState:
    """Read event predicates used to interrupt an open-loop action chunk."""
    support = observation.get("benchmark_support") or {}
    relation_state = support.get("relation_state")
    object_state = support.get("object_state")
    if relation_state is None or object_state is None:
        raise ValueError("Live observation is missing benchmark graph support")
    catalog = task_env._get_benchmark_object_catalog()
    goal = task_goal_from_env(task_env, catalog)
    destination_ids = goal.destination_ids
    retriever = LiveGraphRetriever(catalog, relation_state, object_state, contract)
    retriever.is_target = np.isin(retriever.object_ids, goal.target_ids)

    held = np.asarray(relation_state.get("held_by", ()), dtype=np.bool_)
    held_valid = retriever._valid("held_by", held) if held.ndim == 2 else held
    held_arm = None
    if held.ndim == 2:
        for target_index in np.flatnonzero(retriever.is_target):
            active = np.flatnonzero(held[target_index] & held_valid[target_index])
            if len(active):
                name = retriever.effector_names[int(active[0])].lower()
                held_arm = "left" if "left" in name else "right" if "right" in name else None
                break

    inside_destination = False
    release_ready = False
    inside = np.asarray(relation_state.get("in", ()), dtype=np.bool_)
    valid = np.asarray(
        relation_state.get("containment_valid", np.zeros_like(inside)), dtype=np.bool_
    )
    if inside.ndim == 2 and valid.shape == inside.shape:
        index_by_id = {
            int(object_id): index for index, object_id in enumerate(retriever.object_ids)
        }
        for target_index in np.flatnonzero(retriever.is_target):
            for destination_id in destination_ids:
                destination_index = index_by_id.get(int(destination_id))
                if (
                    destination_index is not None
                    and bool(valid[target_index, destination_index])
                    and bool(inside[target_index, destination_index])
                ):
                    inside_destination = True
                    break

    # A held object may remain just above an open container because the policy
    # expects gravity to finish insertion after release. Requiring full center
    # containment before opening therefore deadlocks. Permit release when the
    # target is horizontally over the destination and its bottom is close
    # enough for gravity to finish insertion. Full `in` remains the success
    # test; this threshold only decides when opening is safe.
    object_ids = np.asarray(object_state["object_ids"], dtype=np.int64)
    aabb_lower = np.asarray(object_state.get("aabb_lower", ()), dtype=np.float64)
    aabb_upper = np.asarray(object_state.get("aabb_upper", ()), dtype=np.float64)
    has_aabb = np.asarray(object_state.get("has_aabb", ()), dtype=np.bool_)
    object_index = {int(object_id): index for index, object_id in enumerate(object_ids)}
    target_ids = [
        int(object_id)
        for object_id, flag in zip(retriever.object_ids, retriever.is_target)
        if bool(flag)
    ]
    if len(target_ids) == 1 and len(destination_ids) == 1:
        target_index = object_index.get(target_ids[0])
        destination_index = object_index.get(destination_ids[0])
        if (
            target_index is not None
            and destination_index is not None
            and aabb_lower.ndim == 2
            and aabb_upper.shape == aabb_lower.shape
            and len(has_aabb) == len(object_ids)
            and bool(has_aabb[target_index])
            and bool(has_aabb[destination_index])
        ):
            target_center = 0.5 * (
                aabb_lower[target_index] + aabb_upper[target_index]
            )
            destination_min = aabb_lower[destination_index]
            destination_max = aabb_upper[destination_index]
            horizontally_ready = bool(
                np.all(target_center[:2] >= destination_min[:2])
                and np.all(target_center[:2] <= destination_max[:2])
            )
            vertical_ready = bool(
                target_center[2] >= destination_min[2] - 0.02
                and aabb_lower[target_index, 2]
                <= destination_max[2] + RELEASE_READY_VERTICAL_CLEARANCE_M
            )
            release_ready = (
                held_arm is not None
                and horizontally_ready
                and vertical_ready
            )

    return LiveTaskState(
        held_arm is not None,
        held_arm,
        inside_destination,
        release_ready,
    )


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


def compact_placement_hint(
    retriever: "LiveGraphRetriever",
    destination_object_ids: Iterable[int],
    relation: str = "in",
) -> str:
    """Describe target-to-destination motion in short instruction-like language."""
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
        directions.append("forward" if forward > 0 else "backward")
    if abs(left) >= 0.05:
        directions.append("left" if left > 0 else "right")
    motion = " and ".join(directions)
    destination_label = retriever._label_by_id.get(destination_id, "destination")
    preposition = "onto" if relation == "on" else "into"
    if motion:
        return (
            f"Keep holding the object. Move it {motion} "
            f"{preposition} the {destination_label}."
        )
    return f"Place the held object {preposition} the {destination_label}."


def compact_grasp_hint(
    retriever: "LiveGraphRetriever",
    target_id: int,
) -> str:
    """Prefer the gripper whose validated straight target corridor is clear."""
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
    fallback = f"Pick up the {target_label}."
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
    approach = (
        f"Use the {side} gripper to approach the {target_label} "
        f"and pick it up."
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
    return (
        f"Collision risk: the {blocker_label} blocks the {blocked_side} gripper. "
        f"{approach}"
    )


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
) -> PreparedInstruction:
    base_instruction = str(task_env.get_instruction())
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
    if guidance_supported:
        target_id = next(
            int(object_id)
            for object_id, is_target in zip(retriever.object_ids, retriever.is_target)
            if bool(is_target)
        )
        destination_label = retriever._label_by_id[destination_ids[0]]
        guidance = {
            "grasp": compact_grasp_hint(retriever, target_id),
            "placement": compact_placement_hint(
                retriever, destination_ids, relation=goal.relation
            ),
            "release": (
                f"Release the held object "
                f"{'on' if goal.relation == 'on' else 'in'} the {destination_label}."
            ),
        }[prompt_phase]
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
    )
