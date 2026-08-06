"""Live observation adapter for the graph-conditioned pi0.5 POC."""

from __future__ import annotations

from dataclasses import dataclass
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
)
from .graph_serializer import fact_pack_item, node_pack_item


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


def _round1(value: float) -> float:
    rounded = round(float(value), 1)
    return 0.0 if rounded == 0 else rounded


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
            f"{entry.get('name') or entry.get('semantic_label') or 'object'}#{int(object_id)}"
            for entry, object_id in zip(object_catalog, self.object_ids)
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
        self._bbox_sizes: dict[int, tuple[float, float, float] | None] = {}
        for index, object_id in enumerate(self.object_ids.tolist()):
            object_id = int(object_id)
            if index < len(is_present) and not bool(is_present[index]):
                continue
            self._positions[object_id] = tuple(
                _round1(value) for value in pose_world[index, :3]
            )
            if index < len(has_aabb) and bool(has_aabb[index]):
                self._bbox_sizes[object_id] = tuple(
                    _round1(upper - lower)
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
                    label=self._label_by_id.get(object_id, str(object_id)),
                    kind=self._kind_by_id.get(object_id, "object"),
                    position=position,
                    bbox_size=self._bbox_sizes.get(object_id),
                )
            )
        return nodes

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
        seeds = self.seed_indices(destination_object_ids)
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
                    name
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
            participating_ids.add(int(self.object_ids[source]))
            resolved_id = self._id_by_name.get(destination_label)
            if resolved_id is not None:
                participating_ids.add(resolved_id)
            result.append(
                GraphFact(priority, relation, self.labels[source], destination_label)
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
                )
            )
        return result


def destination_ids_from_task(task_env: Any) -> tuple[int, ...]:
    """Read destination roles only from task metadata, never action nodes."""
    try:
        roles = task_env.get_role_names()
    except (AttributeError, TypeError):
        return ()
    destination = roles.get("destination_id") if isinstance(roles, dict) else None
    return () if destination is None else (int(destination),)


def prepare_instruction(
    task_env: Any,
    model: Any,
    observation: dict[str, Any],
    condition: InputCondition | str,
    contract: RetrievalContract,
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
        )

    support = observation.get("benchmark_support") or {}
    relation_state = support.get("relation_state")
    if relation_state is None:
        raise ValueError("Live observation is missing benchmark_support/relation_state")
    object_state = support.get("object_state")
    if object_state is None:
        raise ValueError("Live observation is missing benchmark_support/object_state")
    catalog = task_env._get_benchmark_object_catalog()
    destination_ids = destination_ids_from_task(task_env)
    nodes, facts = LiveGraphRetriever(catalog, relation_state, object_state, contract).retrieve(
        destination_ids
    )
    # Each item carries its own rank so the server-side packer (real
    # tokenizer, same knapsack) can reproduce the exact same
    # priority-maximizing selection it would make offline.
    items = [
        {"text": item.text, "rank": item.rank, "section": item.section}
        for item in (
            [node_pack_item(node) for node in nodes]
            + [fact_pack_item(fact) for fact in facts]
        )
    ]
    response = model.fit_graph_prompt(
        {
            "instruction": base_instruction,
            "state": observation["joint_action"]["vector"],
            "items": items,
            "graph_token_budget": contract.graph_token_budget,
        }
    )
    selected_node_count = int(response["selected_node_count"])
    selected_fact_count = int(response["selected_fact_count"])
    return PreparedInstruction(
        instruction=str(response["instruction"]),
        retrieved_node_count=len(nodes),
        selected_node_count=selected_node_count,
        dropped_node_count=len(nodes) - selected_node_count,
        retrieved_fact_count=len(facts),
        selected_fact_count=selected_fact_count,
        dropped_fact_count=len(facts) - selected_fact_count,
        graph_token_count=int(response["graph_token_count"]),
        full_prompt_token_count_estimate=int(response["full_prompt_token_count_estimate"]),
        destination_seed_available=bool(destination_ids),
    )
