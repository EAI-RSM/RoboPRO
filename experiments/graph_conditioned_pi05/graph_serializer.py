"""Natural-language serialization for retrieved graph nodes and facts.

The packer here (`pack_items` / `_knapsack_select`) is the single shared
selection algorithm for both the offline/approximate path (this module,
`conservative_token_count`) and the live path, which calls it from
`customized_robotwin/policy/pi05/pi_model.py` with the real checkpoint
tokenizer as `token_counter`. Keeping one implementation means a change to
selection semantics (separator, header text, tie-breaking) only has to be
made once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable, Iterable, Mapping, Sequence

from .contract import RELATION_PRIORITY, GraphFact, GraphNode


TokenCounter = Callable[[str], int]


RELATION_TEMPLATES: Mapping[str, str] = {
    "held_by": "{source} is held by {destination}{qualifier}.",
    "reachable_by": "{source} is reachable by {destination}{qualifier}.",
    "blocks": "{source} blocks {destination}{qualifier}.",
    "occludes": "{source} occludes {destination}{qualifier}.",
    "visible_to": "{source} is visible to {destination}{qualifier}.",
    "in": "{source} is inside {destination}{qualifier}.",
    "contains": "{source} contains {destination}{qualifier}.",
    "on": "{source} is on top of {destination}{qualifier}.",
    "supports": "{source} supports {destination}{qualifier}.",
    "intentional_contact_with": "{source} is intentionally in contact with {destination}{qualifier}.",
    "robot_collision_with": "{source} is colliding with {destination}{qualifier}.",
    "unexpected_collision_with": "{source} is unexpectedly colliding with {destination}{qualifier}.",
    "static_contact_with": "{source} is in static contact with {destination}{qualifier}.",
    "near": "{source} is near {destination}{qualifier}.",
    "part_of": "{source} is part of {destination}{qualifier}.",
}


@dataclass(frozen=True)
class NaturalLanguageGraphRenderer:
    """Render typed graph facts as reusable, deterministic English sentences."""

    templates: Mapping[str, str] = field(default_factory=lambda: RELATION_TEMPLATES)

    @staticmethod
    def _natural_list(value: str) -> str:
        items = [item.strip() for item in value.split(",") if item.strip()]
        if len(items) < 2:
            return items[0] if items else ""
        return ", ".join(items[:-1]) + f" and {items[-1]}"

    def _qualifier(self, fact: GraphFact) -> str:
        value = self._natural_list(fact.qualifier)
        if not value:
            return ""
        if fact.relation == "blocks":
            return f" for {value}"
        if fact.relation == "occludes":
            return f" from {value}"
        return f" with respect to {value}"

    def render_fact(self, fact: GraphFact) -> str:
        template = self.templates.get(fact.relation)
        if template is None:
            relation = fact.relation.replace("_", " ")
            template = f"{{source}} {relation} {{destination}}{{qualifier}}."
        return template.format(
            source=fact.source,
            destination=fact.destination,
            qualifier=self._qualifier(fact),
        )

    def render_facts(self, facts: Iterable[GraphFact]) -> tuple[str, ...]:
        return tuple(self.render_fact(fact) for fact in facts)


DEFAULT_GRAPH_RENDERER = NaturalLanguageGraphRenderer()


def conservative_token_count(text: str) -> int:
    """Dependency-free upper-biased counter used only before the real tokenizer adapter.

    Phase 3 must inject the checkpoint PaliGemma tokenizer. Counting punctuation
    separately makes this safer than whitespace counting for compact graph text.
    """
    return len(re.findall(r"[A-Za-z0-9_#.-]+|[^\w\s]", text))


def format_fact(fact: GraphFact) -> str:
    """Backward-compatible entry point for the default natural-language renderer."""
    return DEFAULT_GRAPH_RENDERER.render_fact(fact)


def _format_number(value: float) -> str:
    rounded = round(float(value), 1)
    if rounded == 0:
        rounded = 0.0  # normalize away "-0.0"
    return f"{rounded:.1f}"


def _format_vector(values: Sequence[float]) -> str:
    return "(" + ", ".join(_format_number(v) for v in values) + ")"


def format_node(node: GraphNode) -> str:
    """Render a node's unique identifier and grounding attributes.

    Position and bounding-box size are reported at 1-decimal precision so a
    model that has never seen a catalog ID before still gets a concrete,
    if coarse, spatial handle for telling apart two same-named objects.
    """
    position = _format_vector(node.position)
    if node.bbox_size is not None:
        size = _format_vector(node.bbox_size)
        return f"{node.label} is at {position} with bounding-box size {size}."
    return f"{node.label} is at {position}."


NODE_HEADER = "Nodes:"
FACT_HEADER = "Scene graph:"

# Ranks are exponentially spaced (see `_rank_value`) so that within the
# "maximize total information under budget" objective, a single
# higher-ranked item always outvalues *any* realistic combination of
# lower-ranked ones. This is what keeps "fixed priority order" meaningful:
# without it, a pile of cheap, unimportant facts could crowd out one
# expensive, important fact purely because they happen to fit.
_NUM_RELATION_TIERS = len(RELATION_PRIORITY)
NODE_RANK = _NUM_RELATION_TIERS + 1  # above every relation tier
_TIER_BASE = 1_000_000


def relation_rank(priority: int) -> int:
    """Map a RELATION_PRIORITY index to a rank where higher = more important."""
    return _NUM_RELATION_TIERS - priority


def _rank_value(rank: int) -> int:
    return _TIER_BASE ** max(rank, 0)


@dataclass(frozen=True)
class PackedItem:
    text: str
    rank: int
    section: str  # "node" or "fact"
    origin: object = None


def node_pack_item(node: GraphNode) -> PackedItem:
    return PackedItem(format_node(node), NODE_RANK, "node", origin=node)


def fact_pack_item(fact: GraphFact) -> PackedItem:
    return PackedItem(format_fact(fact), relation_rank(fact.priority), "fact", origin=fact)


def _knapsack_select(weights: Sequence[int], values: Sequence[int], capacity: int) -> list[int]:
    """0/1 knapsack: indices maximizing sum(values) s.t. sum(weights) <= capacity."""
    n = len(weights)
    capacity = max(capacity, 0)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        row, prev = dp[i], dp[i - 1]
        for c in range(capacity + 1):
            best = prev[c]
            if w <= c:
                candidate = prev[c - w] + v
                if candidate > best:
                    best = candidate
            row[c] = best
    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected.append(i - 1)
            c -= weights[i - 1]
    selected.reverse()
    return selected


@dataclass(frozen=True)
class PackedGraph:
    text: str
    token_count: int
    selected: tuple[PackedItem, ...]
    dropped_count: int


def _render(selected: Sequence[PackedItem]) -> str:
    node_texts = [item.text for item in selected if item.section == "node"]
    fact_texts = [item.text for item in selected if item.section == "fact"]
    sections = []
    if node_texts:
        sections.append(NODE_HEADER + " " + " ".join(node_texts))
    if fact_texts:
        sections.append(FACT_HEADER + " " + " ".join(fact_texts))
    return " ".join(sections)


def pack_items(
    items: Sequence[PackedItem],
    token_budget: int,
    token_counter: TokenCounter = conservative_token_count,
) -> PackedGraph:
    """Select the value-maximizing subset of `items` that fits `token_budget`.

    Selection is a 0/1 knapsack rather than a first-fit walk in priority
    order: a plain "skip whatever doesn't fit, keep trying smaller/later
    items" walk can let several cheap, low-priority facts crowd out one
    expensive, high-priority fact purely from iteration order — which
    undermines "priority order" instead of maximizing information under it.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if not items:
        return PackedGraph("", 0, (), 0)

    order = sorted(range(len(items)), key=lambda i: (-items[i].rank, items[i].text))
    weights = [max(token_counter(items[i].text), 1) for i in order]
    values = [_rank_value(items[i].rank) for i in order]

    header_cost = 0
    if any(items[i].section == "node" for i in order):
        header_cost += token_counter(NODE_HEADER)
    if any(items[i].section == "fact" for i in order):
        header_cost += token_counter(FACT_HEADER)
    capacity = max(token_budget - header_cost, 0)

    chosen = _knapsack_select(weights, values, capacity)
    chosen_original = sorted(order[i] for i in chosen)
    selected = tuple(items[i] for i in chosen_original)

    text = _render(selected)
    count = token_counter(text) if text else 0
    return PackedGraph(text, count, selected, len(items) - len(selected))


@dataclass(frozen=True)
class SerializedGraph:
    text: str
    token_count: int
    selected_nodes: tuple[GraphNode, ...]
    selected_facts: tuple[GraphFact, ...]
    dropped_node_count: int
    dropped_fact_count: int


def serialize_graph(
    nodes: Iterable[GraphNode],
    facts: Iterable[GraphFact],
    token_budget: int,
    token_counter: TokenCounter = conservative_token_count,
) -> SerializedGraph:
    node_list = sorted(nodes, key=lambda n: n.object_id)
    fact_list = sorted(facts)
    items = [node_pack_item(n) for n in node_list] + [fact_pack_item(f) for f in fact_list]
    packed = pack_items(items, token_budget, token_counter)
    selected_nodes = tuple(item.origin for item in packed.selected if item.section == "node")
    selected_facts = tuple(item.origin for item in packed.selected if item.section == "fact")
    return SerializedGraph(
        text=packed.text,
        token_count=packed.token_count,
        selected_nodes=selected_nodes,
        selected_facts=selected_facts,
        dropped_node_count=len(node_list) - len(selected_nodes),
        dropped_fact_count=len(fact_list) - len(selected_facts),
    )


def serialize_facts(
    facts: Iterable[GraphFact],
    token_budget: int,
    token_counter: TokenCounter = conservative_token_count,
) -> SerializedGraph:
    """Facts-only convenience wrapper (offline/validation path with no nodes)."""
    return serialize_graph((), facts, token_budget, token_counter)


def augment_instruction(instruction: str, graph: SerializedGraph) -> str:
    instruction = instruction.strip()
    if not graph.text:
        return instruction
    return f"{instruction}\n{graph.text}"
