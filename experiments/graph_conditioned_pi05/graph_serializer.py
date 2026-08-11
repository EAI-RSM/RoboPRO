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


def _format_number(value: float, decimals: int = 1) -> str:
    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0  # normalize away "-0.0"
    return f"{rounded:.{decimals}f}"


def _format_vector(values: Sequence[float], decimals: int = 1) -> str:
    return "(" + ", ".join(_format_number(v, decimals) for v in values) + ")"


def format_node(node: GraphNode) -> str:
    """Render a node's unique identifier and grounding attributes.

    Position is reported at 1-decimal precision. Bounding-box size uses 2
    decimals so centimeter-scale objects do not acquire zero-width dimensions.
    Together these give a model that has never seen a catalog ID a concrete
    spatial handle for telling apart two same-named objects.
    """
    # Goal-role nodes use task-relative geometry supplied by the mandatory
    # Goal section. Repeating their absolute world positions is both less
    # actionable and expensive in the small pi0.5 prompt budget.
    role_node = node.alias.startswith(("T", "D"))
    position = _format_vector(node.position)
    if node.bbox_size is not None:
        size = _format_vector(node.bbox_size, decimals=2)
        if role_node:
            return f"{node.alias} = {node.label}, size {size}."
        return f"{node.alias or node.label} = {node.label} at {position}, size {size}."
    if role_node:
        return f"{node.alias} = {node.label}."
    return f"{node.alias or node.label} = {node.label} at {position}."


NODE_HEADER = "Nodes:"
FACT_HEADER = "Relations:"

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
    provides: str = ""
    requires: tuple[str, ...] = ()
    mandatory: bool = False


def node_pack_item(node: GraphNode) -> PackedItem:
    # Target and destination seeds define the task grounding. They must not
    # disappear merely because no relation involving them is true in the
    # current frame (a destination commonly has no target relation before the
    # object is placed). Other nodes remain dependency-selected by facts.
    mandatory = node.alias.startswith(("T", "D"))
    return PackedItem(
        format_node(node), NODE_RANK, "node", origin=node,
        provides=node.alias, mandatory=mandatory,
    )


def fact_pack_item(fact: GraphFact) -> PackedItem:
    return PackedItem(
        format_fact(fact), relation_rank(fact.priority), "fact", origin=fact,
        requires=fact.required_aliases,
    )


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
    goal_texts = [item.text for item in selected if item.section == "goal"]
    fact_texts = [item.text for item in selected if item.section == "fact"]
    sections = []
    if node_texts:
        sections.append(NODE_HEADER + " " + " ".join(node_texts))
    if goal_texts:
        # Goal items are already imperative natural-language sentences. A
        # structural header adds distribution shift without useful grounding.
        sections.append(" ".join(goal_texts))
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

    node_by_alias = {
        item.provides: index for index, item in enumerate(items)
        if item.section == "node" and item.provides
    }
    facts = sorted(
        (index for index, item in enumerate(items) if item.section == "fact"),
        key=lambda i: (-items[i].rank, items[i].text),
    )
    # One-hop retrieved graphs are small. Keeping exact candidate closures
    # avoids tokenizer separator approximations and charges shared nodes once.
    # Keying by dependency closure and exact token count prunes equivalent
    # fact subsets, bounding the search by aliases x budget rather than 2^facts.
    mandatory_indices = tuple(
        index for index, item in enumerate(items) if item.mandatory
    )
    for index in mandatory_indices:
        missing = [
            alias for alias in items[index].requires if alias not in node_by_alias
        ]
        if missing:
            raise ValueError(
                f"mandatory graph item requires undeclared aliases: {missing}"
            )
    mandatory_aliases = frozenset(
        alias
        for index in mandatory_indices
        for alias in ((items[index].provides,) + items[index].requires)
        if alias
    )
    mandatory_text = _render([items[index] for index in mandatory_indices])
    mandatory_count = token_counter(mandatory_text) if mandatory_text else 0
    if mandatory_count > token_budget:
        raise ValueError(
            "graph token budget is too small for mandatory target/destination "
            f"context: need {mandatory_count}, have {token_budget}"
        )

    candidates: dict[
        tuple[frozenset[str], int], tuple[int, tuple[int, ...], tuple[int, ...]]
    ] = {
        (mandatory_aliases, mandatory_count): (0, (), mandatory_indices)
    }
    for fact_index in facts:
        updated = dict(candidates)
        for (old_required, _), (value, chosen_facts, _) in candidates.items():
            required = old_required | frozenset(items[fact_index].requires)
            if any(alias not in node_by_alias for alias in required):
                continue
            chosen = tuple(sorted((*chosen_facts, fact_index)))
            indices = tuple(
                sorted(
                    set(mandatory_indices)
                    | set(chosen)
                    | {node_by_alias[a] for a in required}
                )
            )
            rendered = _render([items[index] for index in indices])
            count = token_counter(rendered)
            if count > token_budget:
                continue
            candidate = (
                value + _rank_value(items[fact_index].rank), chosen, indices
            )
            key = (required, count)
            previous = updated.get(key)
            if previous is None or (candidate[0], len(candidate[1])) > (
                previous[0], len(previous[1])
            ):
                updated[key] = candidate
        candidates = updated
    _, _, selected_indices = max(
        candidates.values(),
        key=lambda result: (result[0], len(result[1]), tuple(-i for i in result[2])),
    )
    selected = tuple(items[index] for index in selected_indices)

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
    alias_order = {"T": 0, "D": 1, "O": 2, "L": 3, "R": 4}
    node_list = sorted(
        nodes,
        key=lambda node: (
            alias_order.get((node.alias or "O")[0], 5),
            int(node.alias[1:]) if node.alias[1:].isdigit() else 0,
            node.object_id,
        ),
    )
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
