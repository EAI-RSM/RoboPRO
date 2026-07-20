"""lookahead-search: decouple SEARCH from EXECUTION.

A *search* uses the simulator's snapshot / rollback ("branch & rollback") to explore
a task with a candidate policy (or a planner), scores branches with a fitness
function, and emits a **motion plan = the winning trajectory's RAW ACTION
SEQUENCE**. A separate *replay policy* (``policy/replay``) — compatible with the
existing eval and data-collection pipelines — just replays those raw actions.
Because raw-action replay involves no policy re-inference, it is deterministic:
it reproduces the trajectory exactly and lets you enrich observations / annotations
at replay time.

Modules
-------
rollback   snapshot / restore / rollout core + the ``TaskAdapter`` contract
candidates ``CandidateSource`` (default: reuse a deploy-policy model)
fitness    ``Fitness`` (default: ``OracleFitness``, lower = better)
search     ``SearchStrategy`` (default: ``BeamSearch``) -> ``SearchResult``
plan       ``TaskSpec`` / ``Plan`` dataclasses + JSON save/load
pins       ``PinPolicy`` — configurable replay-time verification pins

The pure modules (``rollback``, ``fitness``, ``search``, ``plan``, ``pins``) have no
jax dependency; only ``candidates`` / ``run_search`` touch a policy model.
"""

from __future__ import annotations

from . import fitness, pins, plan, rollback, search  # noqa: F401
from .candidates import CandidateSource, PolicyCandidates
from .fitness import Fitness, OracleFitness, SuccessFitness, build_fitness
from .pins import Pin, PinPolicy, PinViolation, default_policy
from .plan import Plan, TaskSpec
from .plan import load as load_plan
from .plan import save as save_plan
from .rollback import TaskAdapter, apply_chunk, restore, settle, snapshot, state_fingerprint
from .search import (BeamSearch, FullTreeSearch, MonteCarloSearch, SearchResult,
                     SearchStrategy, build_strategy)

__all__ = [
    "rollback", "candidates", "fitness", "search", "plan", "pins",
    "TaskAdapter", "snapshot", "restore", "apply_chunk", "settle", "state_fingerprint",
    "CandidateSource", "PolicyCandidates",
    "Fitness", "OracleFitness", "SuccessFitness", "build_fitness",
    "SearchStrategy", "BeamSearch", "MonteCarloSearch", "FullTreeSearch",
    "SearchResult", "build_strategy",
    "TaskSpec", "Plan", "save_plan", "load_plan",
    "Pin", "PinPolicy", "PinViolation", "default_policy",
]
