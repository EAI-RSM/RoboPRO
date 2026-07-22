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
trace      ``SearchTrace`` — committed plan + per-decision candidate fans, the
           policy-free artifact shared by viz / replay / Q-data
pins       ``PinPolicy`` — configurable replay-time verification pins
viz        ghost-futures branching videos: ``capture_trace`` (policy stage) /
           ``render_trace`` (deterministic playback) / ``compose_scene_video``

The pure modules (``rollback``, ``fitness``, ``search``, ``plan``, ``pins``) have no
jax dependency; only ``candidates`` / ``run_search`` touch a policy model.
"""

from __future__ import annotations

from . import fitness, pins, plan, rollback, search, trace  # noqa: F401
from . import viz  # noqa: F401 - ghost-futures branching-video capture/render/composite (numpy+cv2, no sim/jax at import)
from .candidates import (CandidateSource, PolicyCandidates, mode_propose,
                         noise_propose)
from .fitness import Fitness, OracleFitness, SuccessFitness, build_fitness
from .pins import Pin, PinPolicy, PinViolation, default_policy
from .plan import Plan, TaskSpec
from .plan import load as load_plan
from .plan import save as save_plan
from .rollback import TaskAdapter, apply_chunk, restore, settle, snapshot, state_fingerprint
from .search import (BeamSearch, FullTreeSearch, MonteCarloSearch, SearchResult,
                     SearchStrategy, build_strategy)
from .trace import SearchTrace
from .trace import load as load_trace
from .trace import save as save_trace

__all__ = [
    "rollback", "candidates", "fitness", "search", "plan", "trace", "pins", "viz",
    "TaskAdapter", "snapshot", "restore", "apply_chunk", "settle", "state_fingerprint",
    "CandidateSource", "PolicyCandidates", "noise_propose", "mode_propose",
    "Fitness", "OracleFitness", "SuccessFitness", "build_fitness",
    "SearchStrategy", "BeamSearch", "MonteCarloSearch", "FullTreeSearch",
    "SearchResult", "build_strategy",
    "TaskSpec", "Plan", "save_plan", "load_plan",
    "SearchTrace", "save_trace", "load_trace",
    "Pin", "PinPolicy", "PinViolation", "default_policy",
]
