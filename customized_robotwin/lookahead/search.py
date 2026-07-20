"""Search strategies over the sim's future-tree.

A :class:`SearchStrategy` explores candidate chunk sequences from a root snapshot,
branching with :mod:`lookahead.rollback` (snapshot / restore) and ranking branches
with a :class:`~lookahead.fitness.Fitness`. The output is a :class:`SearchResult`
carrying the winning path's **raw action sequence** (the concatenated committed
chunks) — NOT the modes/noise that produced it — so a downstream replay policy can
reproduce the trajectory by pure action playback.

:class:`BeamSearch` is the default. :class:`MonteCarloSearch` and
:class:`FullTreeSearch` are provided as simple alternative strategies / extension
points.

No jax / no policy imports.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import rollback
from .candidates import CandidateSource
from .fitness import Fitness


@dataclass
class SearchResult:
    """The winning branch of a search.

    Attributes
    ----------
    actions:
        The committed raw action sequence, ``float`` array ``[T, action_dim]`` =
        every action row executed along the winning path (concatenated chunks).
        This is what a replay policy plays back.
    score:
        The winning outcome's fitness score (lower = better), as a tuple.
    candidate_indices:
        The candidate index chosen at each committed step (length = number of
        committed chunks). Provenance only — replay uses ``actions``, not these.
    outcome:
        The terminal outcome dict from the fitness at the winning leaf.
    success:
        ``outcome["success"]`` convenience copy.
    depth:
        Number of committed chunks along the winning path.
    root_fingerprint, terminal_fingerprint:
        State fingerprints at t0 and at the scored terminal, for replay verification.
    """

    actions: np.ndarray
    score: Tuple[float, ...]
    candidate_indices: List[int]
    outcome: Dict[str, Any]
    success: bool
    depth: int
    root_fingerprint: np.ndarray
    terminal_fingerprint: np.ndarray


@dataclass(order=True)
class _Node:
    """Internal beam/tree node. Ordered by ``score`` for cheap ``min``/sort."""

    score: Tuple[float, ...]
    snapshot: Dict[str, Any] = field(compare=False)
    actions: List[np.ndarray] = field(compare=False)
    indices: List[int] = field(compare=False)
    outcome: Dict[str, Any] = field(compare=False)
    success: bool = field(compare=False)


def _stack(rows: List[np.ndarray]) -> np.ndarray:
    """Concatenate committed action rows into a single ``[T, action_dim]`` array."""
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(np.stack([np.asarray(r, dtype=np.float32) for r in rows]))


def _result_from_node(node: _Node, root_fp: np.ndarray) -> SearchResult:
    return SearchResult(
        actions=_stack(node.actions),
        score=tuple(node.score),
        candidate_indices=list(node.indices),
        outcome=node.outcome,
        success=bool(node.success),
        depth=len(node.indices),
        root_fingerprint=np.asarray(root_fp, dtype=np.float64),
        terminal_fingerprint=np.asarray(node.snapshot["fingerprint"], dtype=np.float64),
    )


class SearchStrategy(ABC):
    """Explore the future-tree from a root snapshot and return the best branch."""

    @abstractmethod
    def search(self, task: Any, root_snapshot: Dict[str, Any],
               candidates: CandidateSource, fitness: Fitness,
               depth: Optional[int] = None,
               context: Optional[Dict[str, Any]] = None) -> SearchResult:
        """Run the search. ``context`` defaults to ``fitness.make_context(task)``
        captured at the root; ``depth`` overrides the strategy's default depth."""


class BeamSearch(SearchStrategy):
    """Beam search over committed chunks (the default strategy).

    At each depth every beam node proposes ``k`` candidate chunks; each candidate is
    branched (restore node -> apply chunk -> score outcome -> snapshot), and the top
    ``width`` expansions by fitness are kept. The best-scoring node found is the
    winner. When ``stop_on_success`` (default) a fully successful best node ends the
    search early. The committed RAW ACTIONS of the winner are recorded (not modes).

    Reasonable defaults: ``width=3``, ``k=6``, ``depth=4``.
    """

    def __init__(self, width: int = 3, k: int = 6, depth: int = 4,
                 stop_on_success: bool = True) -> None:
        self.width = int(width)
        self.k = int(k)
        self.depth = int(depth)
        self.stop_on_success = bool(stop_on_success)

    def search(self, task: Any, root_snapshot: Dict[str, Any],
               candidates: CandidateSource, fitness: Fitness,
               depth: Optional[int] = None,
               context: Optional[Dict[str, Any]] = None) -> SearchResult:
        depth = self.depth if depth is None else int(depth)
        ctx = fitness.make_context(task) if context is None else context
        root_fp = np.asarray(root_snapshot["fingerprint"], dtype=np.float64)

        # seed the beam with the root (no actions committed yet)
        rollback.restore(task, root_snapshot)
        root_outcome = fitness.outcome(task, ctx)
        beam: List[_Node] = [
            _Node(score=fitness.score(root_outcome), snapshot=root_snapshot,
                  actions=[], indices=[], outcome=root_outcome,
                  success=bool(root_outcome.get("success", False)))
        ]
        best: _Node = beam[0]

        for _d in range(depth):
            expansions: List[_Node] = []
            for node in beam:
                if node.success:
                    # already-successful nodes are carried forward unchanged
                    expansions.append(node)
                    continue
                rollback.restore(task, node.snapshot)
                cand_chunks = candidates.propose(task, self.k)
                for ci, chunk in enumerate(cand_chunks):
                    rollback.restore(task, node.snapshot)
                    rollback.apply_chunk(task, chunk)
                    outcome = fitness.outcome(task, ctx)
                    snap = rollback.snapshot(task)
                    expansions.append(_Node(
                        score=fitness.score(outcome), snapshot=snap,
                        actions=node.actions + [np.asarray(r) for r in chunk],
                        indices=node.indices + [ci], outcome=outcome,
                        success=bool(outcome.get("success", False))))
            if not expansions:
                break
            expansions.sort()  # ascending score (lower = better)
            beam = expansions[:self.width]
            if beam[0].score < best.score:
                best = beam[0]
            if self.stop_on_success and best.success:
                break

        # leave the task in the scored terminal state of the winner
        rollback.restore(task, best.snapshot)
        return _result_from_node(best, root_fp)


class MonteCarloSearch(SearchStrategy):
    """Random-rollout search: ``n_samples`` independent depth-long chains.

    Each sample restores the root and, at every step, proposes ``k`` candidates and
    commits a uniformly random one; the best terminal (by fitness) wins. A cheap,
    unbiased baseline / extension point next to :class:`BeamSearch`.
    """

    def __init__(self, n_samples: int = 8, k: int = 6, depth: int = 4,
                 seed: int = 0, stop_on_success: bool = True) -> None:
        self.n_samples = int(n_samples)
        self.k = int(k)
        self.depth = int(depth)
        self.seed = int(seed)
        self.stop_on_success = bool(stop_on_success)

    def search(self, task: Any, root_snapshot: Dict[str, Any],
               candidates: CandidateSource, fitness: Fitness,
               depth: Optional[int] = None,
               context: Optional[Dict[str, Any]] = None) -> SearchResult:
        depth = self.depth if depth is None else int(depth)
        ctx = fitness.make_context(task) if context is None else context
        root_fp = np.asarray(root_snapshot["fingerprint"], dtype=np.float64)
        rng = np.random.default_rng(self.seed)

        rollback.restore(task, root_snapshot)
        root_outcome = fitness.outcome(task, ctx)
        best = _Node(score=fitness.score(root_outcome), snapshot=root_snapshot,
                     actions=[], indices=[], outcome=root_outcome,
                     success=bool(root_outcome.get("success", False)))

        for _s in range(self.n_samples):
            rollback.restore(task, root_snapshot)
            actions: List[np.ndarray] = []
            indices: List[int] = []
            outcome = root_outcome
            success = False
            for _d in range(depth):
                if task.check_success():
                    success = True
                    break
                chunks = candidates.propose(task, self.k)
                ci = int(rng.integers(len(chunks)))
                rollback.apply_chunk(task, chunks[ci])
                actions += [np.asarray(r) for r in chunks[ci]]
                indices.append(ci)
                outcome = fitness.outcome(task, ctx)
                success = bool(outcome.get("success", False))
                if self.stop_on_success and success:
                    break
            node = _Node(score=fitness.score(outcome), snapshot=rollback.snapshot(task),
                         actions=actions, indices=indices, outcome=outcome, success=success)
            if node.score < best.score:
                best = node
            if self.stop_on_success and best.success:
                break

        rollback.restore(task, best.snapshot)
        return _result_from_node(best, root_fp)


class FullTreeSearch(SearchStrategy):
    """Exhaustive DFS over the full ``k**depth`` future-tree.

    Enumerates every candidate sequence and keeps the best terminal. Correct but
    exponential — guarded by ``max_leaves`` (raises if the tree is too large). Use
    only with small ``k``/``depth``; :class:`BeamSearch` is the practical default.
    """

    def __init__(self, k: int = 3, depth: int = 3, max_leaves: int = 512,
                 stop_on_success: bool = True) -> None:
        self.k = int(k)
        self.depth = int(depth)
        self.max_leaves = int(max_leaves)
        self.stop_on_success = bool(stop_on_success)

    def search(self, task: Any, root_snapshot: Dict[str, Any],
               candidates: CandidateSource, fitness: Fitness,
               depth: Optional[int] = None,
               context: Optional[Dict[str, Any]] = None) -> SearchResult:
        depth = self.depth if depth is None else int(depth)
        n_leaves = self.k ** depth
        if n_leaves > self.max_leaves:
            raise ValueError(
                f"FullTreeSearch would explore {n_leaves} leaves (k={self.k}, "
                f"depth={depth}) > max_leaves={self.max_leaves}; lower k/depth or "
                f"use BeamSearch.")
        ctx = fitness.make_context(task) if context is None else context
        root_fp = np.asarray(root_snapshot["fingerprint"], dtype=np.float64)

        rollback.restore(task, root_snapshot)
        root_outcome = fitness.outcome(task, ctx)
        best = _Node(score=fitness.score(root_outcome), snapshot=root_snapshot,
                     actions=[], indices=[], outcome=root_outcome,
                     success=bool(root_outcome.get("success", False)))

        def dfs(snap: Dict[str, Any], actions: List[np.ndarray],
                indices: List[int], d: int) -> None:
            nonlocal best
            if d == depth:
                return
            rollback.restore(task, snap)
            if self.stop_on_success and task.check_success():
                return
            for ci, chunk in enumerate(candidates.propose(task, self.k)):
                rollback.restore(task, snap)
                rollback.apply_chunk(task, chunk)
                outcome = fitness.outcome(task, ctx)
                child_snap = rollback.snapshot(task)
                node = _Node(score=fitness.score(outcome), snapshot=child_snap,
                             actions=actions + [np.asarray(r) for r in chunk],
                             indices=indices + [ci], outcome=outcome,
                             success=bool(outcome.get("success", False)))
                if node.score < best.score:
                    best = node
                dfs(child_snap, node.actions, node.indices, d + 1)

        dfs(root_snapshot, [], [], 0)
        rollback.restore(task, best.snapshot)
        return _result_from_node(best, root_fp)


# Registry so a CLI ``--strategy <name>`` resolves a strategy class.
STRATEGY_REGISTRY = {
    "beam": BeamSearch,
    "montecarlo": MonteCarloSearch,
    "fulltree": FullTreeSearch,
}


def build_strategy(name: str, **kwargs: Any) -> SearchStrategy:
    """Instantiate a registered strategy by name (for CLI use).

    Drops ``None`` kwargs (so class defaults apply) and silently ignores kwargs the
    chosen strategy does not accept — e.g. ``width`` is meaningful for BeamSearch but
    not MonteCarloSearch, so a single CLI call can target any strategy.
    """
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"unknown strategy '{name}'; choices: {sorted(STRATEGY_REGISTRY)}")
    import inspect
    cls = STRATEGY_REGISTRY[name]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in kwargs.items() if v is not None and k in accepted}
    return cls(**kwargs)
