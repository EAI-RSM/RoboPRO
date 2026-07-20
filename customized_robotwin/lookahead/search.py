"""Search strategies over the sim's future-tree.

A :class:`SearchStrategy` explores candidate chunk sequences from a root snapshot,
branching with :mod:`lookahead.rollback` (snapshot / restore) and ranking branches
with a :class:`~lookahead.fitness.Fitness`. It returns the **top-M distinct plans**
(``List[SearchResult]``, best first) — each carrying its winning path's **raw action
sequence** (the concatenated committed chunks), NOT the modes/noise that produced it
— so a downstream replay policy can reproduce each trajectory by pure action
playback.

``m`` defaults to 1, in which case the returned list holds exactly the single best
plan (identical to the pre-top-M behaviour). The top-M are drawn from ALL scored
expansions seen during the search and deduped by the committed candidate-index path,
so near-identical plans are not returned twice.

:class:`BeamSearch` is the default. :class:`MonteCarloSearch` and
:class:`FullTreeSearch` are provided as simple alternative strategies / extension
points.

No jax / no policy imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import rollback
from .candidates import CandidateSource
from .fitness import Fitness


@dataclass
class SearchResult:
    """One ranked branch of a search.

    Attributes
    ----------
    actions:
        The committed raw action sequence, ``float`` array ``[T, action_dim]`` =
        every action row executed along this path (concatenated chunks). This is
        what a replay policy plays back.
    score:
        This outcome's fitness score (lower = better), as a tuple.
    candidate_indices:
        The candidate index chosen at each committed step (length = number of
        committed chunks). Provenance + the dedup key; replay uses ``actions``.
    outcome:
        The terminal outcome dict from the fitness at this leaf.
    success:
        ``outcome["success"]`` convenience copy.
    depth:
        Number of committed chunks along this path.
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


def _rank_nodes(nodes: List[_Node], m: int) -> List[_Node]:
    """Top-``m`` DISTINCT nodes by score (best first).

    Distinctness is by the committed candidate-index path, so two paths with the
    same commit sequence collapse to their best-scoring instance and near-identical
    plans are not returned twice. Ties keep the first (best) occurrence. The result
    is never empty (there is always at least the root).
    """
    m = max(1, int(m))
    seen: set = set()
    ranked: List[_Node] = []
    for node in sorted(nodes):                      # ascending score
        key = tuple(node.indices)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(node)
        if len(ranked) >= m:
            break
    return ranked


class SearchStrategy(ABC):
    """Explore the future-tree from a root snapshot and return the top-M branches."""

    @abstractmethod
    def search(self, task: Any, root_snapshot: Dict[str, Any],
               candidates: CandidateSource, fitness: Fitness,
               depth: Optional[int] = None,
               context: Optional[Dict[str, Any]] = None,
               m: int = 1) -> List[SearchResult]:
        """Run the search and return the top-``m`` distinct plans (best first).

        ``context`` defaults to ``fitness.make_context(task)`` captured at the root;
        ``depth`` overrides the strategy's default depth. With ``m == 1`` the list
        holds exactly the single best plan. The task is left restored at the best
        (rank-0) plan's scored terminal state.
        """


class BeamSearch(SearchStrategy):
    """Beam search over committed chunks (the default strategy).

    At each depth every beam node proposes ``k`` candidate chunks; each candidate is
    branched (restore node -> apply chunk -> score outcome -> snapshot) and every
    scored child is retained in a pool. The top ``width`` expansions by fitness
    survive into the next beam. The top-M are drawn from the FULL pool of scored
    expansions (not just the pruned survivors), deduped by candidate-index path.
    When ``stop_on_success`` (default) the depth loop stops once the global-best node
    is a success, but the returned list is still filled from everything scored so
    far. The committed RAW ACTIONS of each ranked node are recorded (not modes).

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
               context: Optional[Dict[str, Any]] = None,
               m: int = 1) -> List[SearchResult]:
        depth = self.depth if depth is None else int(depth)
        ctx = fitness.make_context(task) if context is None else context
        root_fp = np.asarray(root_snapshot["fingerprint"], dtype=np.float64)

        rollback.restore(task, root_snapshot)
        root_outcome = fitness.outcome(task, ctx)
        root_node = _Node(score=fitness.score(root_outcome), snapshot=root_snapshot,
                          actions=[], indices=[], outcome=root_outcome,
                          success=bool(root_outcome.get("success", False)))
        beam: List[_Node] = [root_node]
        pool: List[_Node] = [root_node]  # every scored node ever, for the top-M

        for _d in range(depth):
            expansions: List[_Node] = []
            for node in beam:
                if node.success:
                    expansions.append(node)      # survive in the beam (already pooled)
                    continue
                rollback.restore(task, node.snapshot)
                cand_chunks = candidates.propose(task, self.k)
                for ci, chunk in enumerate(cand_chunks):
                    rollback.restore(task, node.snapshot)
                    rollback.apply_chunk(task, chunk)
                    outcome = fitness.outcome(task, ctx)
                    child = _Node(
                        score=fitness.score(outcome), snapshot=rollback.snapshot(task),
                        actions=node.actions + [np.asarray(r) for r in chunk],
                        indices=node.indices + [ci], outcome=outcome,
                        success=bool(outcome.get("success", False)))
                    expansions.append(child)
                    pool.append(child)
            if not expansions:
                break
            expansions.sort()                      # ascending score (lower = better)
            beam = expansions[:self.width]
            if self.stop_on_success and min(pool).success:
                break

        ranked = _rank_nodes(pool, m)
        rollback.restore(task, ranked[0].snapshot)
        return [_result_from_node(n, root_fp) for n in ranked]


class MonteCarloSearch(SearchStrategy):
    """Random-rollout search: up to ``n_samples`` independent depth-long chains.

    Each sample restores the root and, at every step, proposes ``k`` candidates and
    commits a uniformly random one; every terminal is pooled and the top-M distinct
    are returned. When ``stop_on_success`` it stops early once a success has been
    found AND at least ``m`` distinct plans are in the pool. A cheap, unbiased
    baseline / extension point next to :class:`BeamSearch`.
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
               context: Optional[Dict[str, Any]] = None,
               m: int = 1) -> List[SearchResult]:
        depth = self.depth if depth is None else int(depth)
        ctx = fitness.make_context(task) if context is None else context
        root_fp = np.asarray(root_snapshot["fingerprint"], dtype=np.float64)
        rng = np.random.default_rng(self.seed)

        rollback.restore(task, root_snapshot)
        root_outcome = fitness.outcome(task, ctx)
        pool: List[_Node] = [
            _Node(score=fitness.score(root_outcome), snapshot=root_snapshot,
                  actions=[], indices=[], outcome=root_outcome,
                  success=bool(root_outcome.get("success", False)))]
        keys: set = {()}

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
            pool.append(_Node(score=fitness.score(outcome),
                              snapshot=rollback.snapshot(task), actions=actions,
                              indices=indices, outcome=outcome, success=success))
            keys.add(tuple(indices))
            if self.stop_on_success and any(n.success for n in pool) and len(keys) >= max(1, int(m)):
                break

        ranked = _rank_nodes(pool, m)
        rollback.restore(task, ranked[0].snapshot)
        return [_result_from_node(n, root_fp) for n in ranked]


class FullTreeSearch(SearchStrategy):
    """Exhaustive DFS over the full ``k**depth`` future-tree.

    Enumerates every candidate sequence, pools every scored node and returns the
    top-M distinct. Correct but exponential — guarded by ``max_leaves`` (raises if
    the tree is too large). Use only with small ``k``/``depth``; :class:`BeamSearch`
    is the practical default.
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
               context: Optional[Dict[str, Any]] = None,
               m: int = 1) -> List[SearchResult]:
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
        pool: List[_Node] = [
            _Node(score=fitness.score(root_outcome), snapshot=root_snapshot,
                  actions=[], indices=[], outcome=root_outcome,
                  success=bool(root_outcome.get("success", False)))]

        def dfs(snap: Dict[str, Any], actions: List[np.ndarray],
                indices: List[int], d: int) -> None:
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
                pool.append(node)
                dfs(child_snap, node.actions, node.indices, d + 1)

        dfs(root_snapshot, [], [], 0)
        ranked = _rank_nodes(pool, m)
        rollback.restore(task, ranked[0].snapshot)
        return [_result_from_node(n, root_fp) for n in ranked]


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
