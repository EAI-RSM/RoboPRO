"""Compact serialization for retrieved graph facts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

from .contract import GraphFact


TokenCounter = Callable[[str], int]


def conservative_token_count(text: str) -> int:
    """Dependency-free upper-biased counter used only before the real tokenizer adapter.

    Phase 3 must inject the checkpoint PaliGemma tokenizer. Counting punctuation
    separately makes this safer than whitespace counting for compact graph text.
    """
    return len(re.findall(r"[A-Za-z0-9_#.-]+|[^\w\s]", text))


@dataclass(frozen=True)
class SerializedGraph:
    text: str
    token_count: int
    selected_facts: tuple[GraphFact, ...]
    dropped_fact_count: int


def format_fact(fact: GraphFact) -> str:
    qualifier = f"@{fact.qualifier}" if fact.qualifier else ""
    return f"{fact.relation}{qualifier}({fact.source},{fact.destination})"


def serialize_facts(
    facts: Iterable[GraphFact],
    token_budget: int,
    token_counter: TokenCounter = conservative_token_count,
) -> SerializedGraph:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    ordered = tuple(sorted(facts))
    selected: list[GraphFact] = []
    header = "Scene graph:"
    text = header
    for fact in ordered:
        candidate_facts = selected + [fact]
        candidate = header + " " + "; ".join(format_fact(item) for item in candidate_facts)
        if token_counter(candidate) <= token_budget:
            selected.append(fact)
            text = candidate
    count = token_counter(text)
    if count > token_budget:
        raise RuntimeError("Serialized graph exceeds its token budget")
    return SerializedGraph(
        text=text,
        token_count=count,
        selected_facts=tuple(selected),
        dropped_fact_count=len(ordered) - len(selected),
    )


def augment_instruction(instruction: str, graph: SerializedGraph) -> str:
    instruction = instruction.strip()
    return f"{instruction}\n{graph.text}" if graph.selected_facts else instruction
