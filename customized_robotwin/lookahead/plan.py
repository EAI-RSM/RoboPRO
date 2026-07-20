"""Motion-plan data model + JSON (de)serialisation.

A :class:`Plan` is a self-contained, portable, inspectable record of a searched
trajectory: the replay-hash task spec (the properties that determine the trajectory),
the winning **raw action sequence** (inline nested lists), the t0/terminal state
fingerprints (the replay arbiter) and search/fitness metadata. It is the hand-off
artifact between the search side (:mod:`lookahead.run_search`) and the replay policy
(``policy/replay``).

Pure module: no jax, no sim, no ``datetime.now`` inside serialisation (timestamps
are passed in by the caller so these functions stay deterministic and testable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

PLAN_FORMAT = "lookahead_plan_v1"


def _to_jsonable(x: Any) -> Any:
    """Recursively convert numpy types / arrays to plain JSON-serialisable values."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


@dataclass
class TaskSpec:
    """The plan's replay hash — the properties that determine the trajectory.

    These are what a replay must match to reproduce the exact task spec; changing any
    of them changes the outcome, so a mismatch means the plan must be re-searched (the
    replay :class:`~lookahead.pins.PinPolicy` verifies them).

    Fields
    ------
    task_name, task_config, seed:
        Scene identity.
    embodiment:
        Robot embodiment descriptor (type list + resolved name), a dict.
    domain_randomization:
        The FULL domain_randomization dict from the task config (physics-changing
        knobs like cluttered_table / random_table_height live here).
    control:
        Action semantics: ``{action_dim, chunk, control_freq, ...}`` (may also carry
        physics params such as ``physics_timestep``).
    provenance:
        Harness provenance: ``{robotwin_commit, bench_config_hash, created_at, ...}``.
    """

    task_name: str
    task_config: str
    seed: int
    embodiment: Dict[str, Any] = field(default_factory=dict)
    domain_randomization: Dict[str, Any] = field(default_factory=dict)
    control: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSpec":
        return cls(
            task_name=d["task_name"],
            task_config=d["task_config"],
            seed=int(d["seed"]),
            embodiment=dict(d.get("embodiment", {})),
            domain_randomization=dict(d.get("domain_randomization", {})),
            control=dict(d.get("control", {})),
            provenance=dict(d.get("provenance", {})),
        )


@dataclass
class Plan:
    """A searched motion plan = task spec + raw actions + fingerprints + meta.

    Fields
    ------
    task_spec:
        The replay-hash :class:`TaskSpec`.
    actions:
        Raw action sequence as nested lists ``[[...row...], ...]`` (inline, portable).
    fingerprints:
        ``{"t0": [...], "terminal": [...]}`` — state fingerprints for pin checks.
    meta:
        Free-form search metadata: ``{strategy, fitness, candidate_policy, score,
        outcome, ...}``.
    format:
        Schema tag (``lookahead_plan_v1``).
    """

    task_spec: TaskSpec
    actions: List[List[float]]
    fingerprints: Dict[str, List[float]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    format: str = PLAN_FORMAT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "task_spec": self.task_spec.to_dict(),
            "actions": _to_jsonable(self.actions),
            "fingerprints": _to_jsonable(self.fingerprints),
            "meta": _to_jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Plan":
        return cls(
            task_spec=TaskSpec.from_dict(d["task_spec"]),
            actions=[list(map(float, row)) for row in d.get("actions", [])],
            fingerprints={k: list(map(float, v))
                          for k, v in d.get("fingerprints", {}).items()},
            meta=dict(d.get("meta", {})),
            format=d.get("format", PLAN_FORMAT),
        )

    def actions_array(self) -> np.ndarray:
        """The raw actions as a ``float32`` array ``[T, action_dim]``."""
        if not self.actions:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(self.actions, dtype=np.float32)


def save(plan: Plan, path: str) -> None:
    """Write ``plan`` to ``path`` as pretty JSON (creates parent dirs)."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2)


def load(path: str) -> Plan:
    """Read a :class:`Plan` from a JSON file written by :func:`save`."""
    with open(path, "r", encoding="utf-8") as f:
        return Plan.from_dict(json.load(f))
