"""Frozen protocol for the graph-conditioned pi0.5 proof of concept."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class InputCondition(str, Enum):
    VISUAL_ONLY = "visual_only"
    VISUAL_RETRIEVED_GRAPH = "visual_retrieved_graph"


GRAPH_TREATMENT_VERSION = "graph_delta_annotated_grasp_orientation_diagnostics_v1"


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
    required_aliases: tuple[str, ...] = ()

    def key(self) -> tuple[str, str, str, str]:
        return self.relation, self.source, self.destination, self.qualifier


@dataclass(frozen=True)
class GraphNode:
    """A retrieved graph participant, declared once with grounding attributes.

    `position` is the node's world-frame 3D center, rounded to 1 decimal
    place. `bbox_size` is the node's axis-aligned world-frame extent
    (upper - lower corner, rounded to 2 decimal places); it is None for node
    kinds where no collision geometry is available (e.g. end effectors).
    Declaring these lets the model distinguish two same-name objects (e.g.
    two "bowl" instances from a cluttered-table distractor draw) by where
    they actually are, instead of only by an opaque catalog ID it was never
    trained to ground visually.
    """

    object_id: int
    label: str
    kind: str
    position: tuple[float, float, float]
    bbox_size: tuple[float, float, float] | None = None
    alias: str = ""


def stable_aliases(
    object_ids: Iterable[int],
    is_target: Iterable[bool],
    destination_ids: Iterable[int] = (),
) -> dict[int, str]:
    """Assign deterministic role-aware aliases from the complete catalog."""
    ids = [int(value) for value in object_ids]
    targets = {object_id for object_id, flag in zip(ids, is_target) if flag}
    destinations = {int(value) for value in destination_ids} - targets
    aliases: dict[int, str] = {}
    if -2 in ids:
        aliases[-2] = "L"
    if -3 in ids:
        aliases[-3] = "R"
    for prefix, members in (("T", targets), ("D", destinations)):
        for index, object_id in enumerate(sorted(members), 1):
            aliases[object_id] = f"{prefix}{index}"
    others = sorted(object_id for object_id in ids if object_id not in aliases)
    for index, object_id in enumerate(others, 1):
        aliases[object_id] = f"O{index}"
    return aliases


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
    "Both conditions use identical tasks, seeds, images, robot state, checkpoint, action representation, nominal chunk horizon, and evaluation criteria.",
    "The graph-aware treatment may use graph-derived phase instructions, task-relevant event-triggered replanning, and active-gripper protection.",
    "Prompt phases, executed chunk lengths, and chunk-interruption counts are treatment outputs and must be logged rather than claimed as controlled variables.",
    "Reported effects compare the complete graph-aware planning system with visual-only rollout; they do not isolate graph prose alone.",
    "Future expert action nodes and outcomes are never policy inputs.",
    "Only true facts with valid evidence are serialized.",
    "Retrieval and serialization are deterministic for a fixed frame and contract.",
    "Primary metrics remain success, collision, hard success, and collision count/category.",
)
