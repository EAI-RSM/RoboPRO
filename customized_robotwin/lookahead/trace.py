"""Search-trace data model — the shared, POLICY-FREE artifact of one search.

A :class:`SearchTrace` is a superset of a :class:`~lookahead.plan.Plan`: besides
the committed winning path it records, for every decision point along the
committed spine, the K candidate futures' RAW held-to-terminal action rows and
their scored outcomes/tiers. It is produced ONCE by the policy+sim stage
(:func:`lookahead.viz.capture_trace`, or ``run_search --dump-trace``) and then
consumed by deterministic, policy-free stages — exactly like replay consumes a
plan:

* ghost-futures visualization (:func:`lookahead.viz.render_trace` — viz never
  re-runs the search and never touches a policy model),
* raw-action replay (:meth:`SearchTrace.committed_plan` -> ``policy/replay``),
* Q-/preference-data extraction (per-decision candidate actions + outcomes).

On-disk format (``lookahead_trace_v1``): an inspectable JSON file (task spec, the
committed plan INLINE — the trace JSON alone is a valid replay hand-off — and the
per-decision fan entries WITHOUT their bulky arrays) plus a companion ``.npz``
next to it (same stem) holding each non-chosen ghost's float32 action rows under
``fan_t{decision}_k{mode}``. The CHOSEN entry's actions are never duplicated on
disk: in memory they always reference the committed continuation
(``committed.actions[d * chunk:]``) and are rebuilt on :func:`load`.

Pure module: no jax, no sim, no cv2.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from .plan import Plan, TaskSpec, _to_jsonable

TRACE_FORMAT = "lookahead_trace_v1"

# One fan entry (dict) per candidate index per decision:
#   {"mode": int, "actions": float32 [T, adim] raw held-to-terminal rows,
#    "tier": str (HARD/SOFT/COLLIDED_FAIL/FAIL), "outcome": dict, "chosen": bool}
FanEntry = Dict[str, Any]


@dataclass
class SearchTrace:
    """One search's full decision record: committed plan + per-decision K-fans.

    Fields
    ------
    task_spec:
        The replay-hash :class:`~lookahead.plan.TaskSpec` (scene identity +
        embodiment + domain randomization + control semantics + provenance).
    chunk:
        Action rows per committed chunk (the decision granularity).
    horizon:
        Search horizon in chunks (ghosts were held to at most this depth).
    committed:
        The winning path as a full :class:`~lookahead.plan.Plan` — raw actions
        inline, ``meta`` carrying ``candidate_indices`` / ``outcome`` /
        ``success`` / ``score``.
    fans:
        ``fans[d]`` = the K :data:`FanEntry` dicts of decision ``d`` (one per
        candidate index, exactly one with ``chosen=True``).
    meta:
        Free-form capture metadata (strategy/fitness/candidate policy/
        instruction, ...).
    """

    task_spec: TaskSpec
    chunk: int
    horizon: int
    committed: Plan
    fans: List[List[FanEntry]]
    meta: Dict[str, Any] = field(default_factory=dict)
    format: str = TRACE_FORMAT

    def committed_plan(self) -> Plan:
        """The winning path as a standalone Plan (replay / ``plan.save`` off this)."""
        return self.committed

    @property
    def candidate_indices(self) -> List[int]:
        """The committed candidate index at each decision (from the plan meta)."""
        return [int(i) for i in self.committed.meta.get("candidate_indices", [])]

    @property
    def n_decisions(self) -> int:
        return len(self.fans)

    def spine_chunks(self) -> List[np.ndarray]:
        """The committed raw actions split into per-decision chunks."""
        acts = self.committed.actions_array()
        return [acts[d * self.chunk:(d + 1) * self.chunk]
                for d in range(len(self.fans))]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-side view: fan entries WITHOUT their action arrays (see save)."""
        return {
            "format": self.format,
            "task_spec": self.task_spec.to_dict(),
            "chunk": int(self.chunk),
            "horizon": int(self.horizon),
            "committed": self.committed.to_dict(),
            "fans": [[{"mode": int(e["mode"]), "tier": str(e["tier"]),
                       "chosen": bool(e["chosen"]),
                       "n_rows": int(np.asarray(e["actions"]).shape[0])
                       if e.get("actions") is not None else 0,
                       "outcome": _to_jsonable(e.get("outcome"))}
                      for e in fan] for fan in self.fans],
            "meta": _to_jsonable(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SearchTrace":
        """Rebuild from the JSON dict; fan ``actions`` are None until attached
        (:func:`load` attaches them from the companion npz / the committed plan)."""
        return cls(
            task_spec=TaskSpec.from_dict(d["task_spec"]),
            chunk=int(d["chunk"]),
            horizon=int(d["horizon"]),
            committed=Plan.from_dict(d["committed"]),
            fans=[[{"mode": int(e["mode"]), "tier": str(e["tier"]),
                    "chosen": bool(e["chosen"]), "n_rows": int(e.get("n_rows", 0)),
                    "outcome": e.get("outcome"), "actions": None}
                   for e in fan] for fan in d.get("fans", [])],
            meta=dict(d.get("meta", {})),
            format=d.get("format", TRACE_FORMAT),
        )


def _npz_path(path: str) -> str:
    """The companion array file: same stem as the trace JSON, ``.npz`` suffix."""
    return os.path.splitext(path)[0] + ".npz"


def save(trace: SearchTrace, path: str) -> None:
    """Write ``trace`` to ``path`` (pretty JSON) + its companion ``.npz``.

    The npz holds each NON-chosen ghost's raw action rows (float32, compressed)
    under ``fan_t{decision}_k{mode}``; chosen entries are rebuilt on load from the
    committed plan. Creates parent dirs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for d, fan in enumerate(trace.fans):
        for e in fan:
            if not e["chosen"]:
                arrays[f"fan_t{d}_k{int(e['mode'])}"] = \
                    np.asarray(e["actions"], dtype=np.float32)
    np.savez_compressed(_npz_path(path), **arrays)
    doc = trace.to_dict()
    doc["arrays"] = os.path.basename(_npz_path(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def load(path: str) -> SearchTrace:
    """Read a :class:`SearchTrace` written by :func:`save` and attach the arrays.

    Non-chosen fan actions come from the companion npz (validated against the
    JSON's ``n_rows``); each chosen entry's actions are re-referenced to the
    committed continuation ``committed.actions[d * chunk:]``.
    """
    with open(path, "r", encoding="utf-8") as f:
        trace = SearchTrace.from_dict(json.load(f))
    acts = trace.committed.actions_array()
    with np.load(_npz_path(path)) as z:
        for d, fan in enumerate(trace.fans):
            for e in fan:
                if e["chosen"]:
                    e["actions"] = acts[d * trace.chunk:]
                    continue
                rows = np.asarray(z[f"fan_t{d}_k{e['mode']}"], dtype=np.float32)
                if e.get("n_rows") and int(e["n_rows"]) != rows.shape[0]:
                    raise ValueError(
                        f"{path}: fan_t{d}_k{e['mode']} has {rows.shape[0]} rows, "
                        f"JSON says {e['n_rows']}")
                e["actions"] = rows
    return trace
