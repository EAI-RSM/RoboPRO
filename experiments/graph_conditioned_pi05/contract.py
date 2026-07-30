"""Frozen protocol for the graph-conditioned pi0.5 proof of concept."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InputCondition(str, Enum):
    VISUAL_ONLY = "visual_only"
    VISUAL_RETRIEVED_GRAPH = "visual_retrieved_graph"


RELATION_PRIORITY = (
    "held_by",
    "reachable_by",
    "blocks",
    "occludes",
    "visible_to",
    "in",
    "contains",
    "on",
    "supports",
    "intentional_contact_with",
    "robot_collision_with",
    "unexpected_collision_with",
    "static_contact_with",
    "near",
    "part_of",
)

VALIDITY_FIELD = {
    "held_by": "held_by_valid",
    "reachable_by": "reachable_by_valid",
    "blocks": "blocks_valid",
    "occludes": "occludes_valid",
    "visible_to": "visible_to_valid",
    "in": "containment_valid",
    "contains": "contains_valid",
    "intentional_contact_with": "contact_semantics_valid",
    "robot_collision_with": "contact_semantics_valid",
    "unexpected_collision_with": "contact_semantics_valid",
    "static_contact_with": "contact_semantics_valid",
    "part_of": "part_of_valid",
}

INVERSE_RELATIONS = {"contains", "supports"}
SYMMETRIC_RELATIONS = {
    "near",
    "intentional_contact_with",
    "robot_collision_with",
    "unexpected_collision_with",
    "static_contact_with",
}


@dataclass(frozen=True, order=True)
class GraphFact:
    priority: int
    relation: str
    source: str
    destination: str
    qualifier: str = ""

    def key(self) -> tuple[str, str, str, str]:
        return self.relation, self.source, self.destination, self.qualifier


@dataclass(frozen=True)
class RetrievalContract:
    graph_token_budget: int = 120
    max_hops: int = 1
    default_camera: str = "countertop_camera"
    include_action_history: bool = False
    include_invalid: bool = False
    include_inverse_relations: bool = False

    def __post_init__(self) -> None:
        if self.graph_token_budget < 1:
            raise ValueError("graph_token_budget must be positive")
        if self.max_hops != 1:
            raise ValueError("The POC contract supports exactly one retrieval hop")
        if self.include_action_history:
            raise ValueError(
                "Online policy action nodes are not implemented; expert action nodes "
                "must not be exposed as input"
            )
        if self.include_invalid:
            raise ValueError("Invalid graph facts must be omitted, not labeled false")


PROTOCOL_INVARIANTS = (
    "Both conditions use identical images, state, instruction, seeds, checkpoint, and action chunking.",
    "The only treatment difference is retrieved graph text appended to the instruction.",
    "Future expert action nodes and outcomes are never policy inputs.",
    "Only true facts with valid evidence are serialized.",
    "Retrieval and serialization are deterministic for a fixed frame and contract.",
    "Primary metrics remain success, collision, hard success, and collision count/category.",
)
