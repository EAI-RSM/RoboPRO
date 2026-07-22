"""Fitness / outcome scoring for lookahead search.

A :class:`Fitness` turns a rolled-forward sim state into (a) an ``outcome`` dict of
task-relevant features and (b) a ``score`` that ranks branches (LOWER = BETTER, so a
search simply minimises it). The default :class:`OracleFitness` ports the privileged
sim oracle used across the reference scripts (``motivation_run.oracle_key`` /
``outcome_features``): a 4-tier success/collision label with a distance-to-goal
tiebreak.

Everything here is optional-by-design so target-less tasks (e.g. ``open_drawer``,
no movable target) still work: the collision (clutter-displacement) proxy and the
distance-to-goal term are only computed when the task actually exposes a
``target_obj`` / ``des_obj``; otherwise the fitness degrades gracefully to a
success-only ranking.

No jax / no policy imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# A non-robot actor that moves > DISP_THRESH from its root position is treated as
# disturbed/collided (a displacement proxy, NOT a force-contact event).
DISP_THRESH = 0.03

# Ranking of the 4 outcome tiers (lower = better).
_TIER_RANK = {"HARD": 0, "SOFT": 1}


# --------------------------------------------------------------------- geometry helpers
def _target_pose(task: Any) -> Optional[np.ndarray]:
    """7D pose [x,y,z,qw,qx,qy,qz] of the task's movable target, or None.

    Tasks with no movable target (``target_obj`` absent) return None; success is
    then driven purely by ``check_success``.
    """
    t = getattr(task, "target_obj", None)
    if t is None:
        return None
    p = t.get_pose()
    return np.concatenate([np.asarray(p.p), np.asarray(p.q)]).astype(np.float64)


def _des_xy(task: Any) -> Optional[np.ndarray]:
    """xyz of the destination object, or None when the task has no destination."""
    d = getattr(task, "des_obj", None)
    if d is not None:
        return np.asarray(d.get_pose().p, dtype=np.float64)
    # some tasks expose a precomputed destination pose instead
    dp = getattr(task, "des_obj_pose", None)
    if dp is not None:
        return np.asarray(dp[:3], dtype=np.float64)
    return None


def _clutter_positions(task: Any, exclude_names: Sequence[str]) -> Dict[int, np.ndarray]:
    """id -> xyz for every non-robot actor whose name is not target/destination.

    Returns {} when the task does not expose the object-pose API, so the collision
    proxy simply becomes inactive rather than raising.
    """
    if not (hasattr(task, "get_object_poses") and hasattr(task, "get_actor_id_map")):
        return {}
    ids, poses = task.get_object_poses()
    id_map = task.get_actor_id_map()
    out: Dict[int, np.ndarray] = {}
    exclude = set(exclude_names)
    for k, i in enumerate(ids):
        if id_map.get(int(i), "") in exclude:
            continue
        out[int(i)] = np.asarray(poses[k, :3], dtype=np.float64)
    return out


def _max_clutter_disp(task: Any, baseline: Dict[int, np.ndarray],
                      exclude_names: Sequence[str]) -> float:
    """Largest displacement (m) of any baseline clutter object from its root xyz."""
    now = _clutter_positions(task, exclude_names)
    disps = [float(np.linalg.norm(now[i] - baseline[i])) for i in baseline if i in now]
    return max(disps) if disps else 0.0


def tier_of(success: bool, collided: bool) -> str:
    """4 outcome tiers from the two flags (matches ``motivation_run.tier_of``)."""
    if success:
        return "HARD" if not collided else "SOFT"
    return "COLLIDED_FAIL" if collided else "FAIL"


# --------------------------------------------------------------------- Fitness ABC
class Fitness(ABC):
    """Maps a rolled-forward sim state to an outcome dict and a minimised score.

    Score convention: **lower is better**. A score is a tuple so lexicographic
    comparison expresses a priority order (primary tier, then a numeric tiebreak);
    a search only ever needs ``min``.
    """

    def make_context(self, task: Any) -> Dict[str, Any]:
        """Build the once-per-search context (captured at the ROOT state).

        Default: record the target/destination exclusion set, destination xy and
        the baseline clutter positions so displacement is measured from t0. Safe on
        target-less tasks (fields become None / empty).
        """
        exclude: List[str] = []
        for attr in ("target_obj", "des_obj"):
            obj = getattr(task, attr, None)
            if obj is not None:
                try:
                    exclude.append(obj.get_name())
                except Exception:  # noqa: BLE001 - name is best-effort
                    pass
        return {
            "exclude_names": exclude,
            "des_xy": _des_xy(task),
            "baseline_clutter": _clutter_positions(task, exclude),
            "disp_thresh": DISP_THRESH,
        }

    @abstractmethod
    def outcome(self, task: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Task-relevant features of the CURRENT sim state given the root context."""

    @abstractmethod
    def score(self, outcome: Dict[str, Any]) -> Tuple[float, ...]:
        """Rank key for an outcome (lower = better)."""


# --------------------------------------------------------------------- default oracle
class OracleFitness(Fitness):
    """Privileged sim oracle: ``(tier_rank, dist_to_goal)`` minimised.

    * ``tier`` in {HARD, SOFT, COLLIDED_FAIL, FAIL} from ``check_success`` +
      collision. Collision is the repo's EXACT metric,
      ``get_collision_metrics()["is_collision"]`` -- the same signal the eval
      pipeline uses -- so ``HARD`` (success & not collided) is identical to the
      eval's ``hard_success``. The clutter-displacement proxy is used only as a
      fallback when collision metrics are not enabled on the task.
    * primary key = tier rank (HARD < SOFT < any fail),
    * tiebreak = Euclidean distance from the target to the destination xy.

    Parameters
    ----------
    disp_thresh:
        Fallback-only clutter-displacement threshold (m) above which a branch is
        "collided" when the repo collision metric is unavailable.
    use_dist_tiebreak:
        When True (default) and a target+destination exist, ties within a tier are
        broken by proximity to the goal. When False, or on target-less tasks, the
        score is success/tier-only.
    """

    def __init__(self, disp_thresh: float = DISP_THRESH,
                 use_dist_tiebreak: bool = True) -> None:
        self.disp_thresh = float(disp_thresh)
        self.use_dist_tiebreak = bool(use_dist_tiebreak)

    def outcome(self, task: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
        success = bool(task.check_success())
        exclude = ctx.get("exclude_names", [])
        baseline = ctx.get("baseline_clutter", {})
        des_xy = ctx.get("des_xy", None)

        # Collision: use the repo's EXACT metric -- get_collision_metrics()["is_collision"]
        # -- the identical definition the eval pipeline scores with, so search pruning and
        # eval scoring agree (HARD tier == eval hard_success). Requires the task to have
        # collision metrics enabled (the lookahead search env sets enable_collision_metrics
        # and the rollback snapshots/restores the accumulator so it is per-branch correct).
        # Falls back to the clutter-displacement proxy only when the repo metric is
        # unavailable (e.g. unit tests / target-less setups with no collision metrics).
        cm = None
        if (getattr(task, "enable_collision_metrics", False)
                and hasattr(task, "get_collision_metrics")
                and hasattr(task, "robot_link_names")):
            try:
                cm = task.get_collision_metrics()
            except Exception:  # noqa: BLE001 - fall back to the proxy on any failure
                cm = None
        if cm is not None:
            collided = bool(cm["is_collision"])
            collision_count = int(cm["total_collision_count"])
            max_disp = None
        elif baseline:
            max_disp = _max_clutter_disp(task, baseline, exclude)
            collided = max_disp > self.disp_thresh
            collision_count = None
        else:
            max_disp, collided, collision_count = 0.0, False, None

        tp = _target_pose(task)
        if tp is not None and des_xy is not None:
            dist = float(np.linalg.norm(tp[:3] - np.asarray(des_xy)))
        else:
            dist = None

        return {
            "success": success,
            "collided": bool(collided),          # get_collision_metrics (eval-identical) when available
            "is_collision": bool(collided),
            "tier": tier_of(success, collided),  # HARD == success & not is_collision == eval hard_success
            "collision_count": collision_count,
            "max_clutter_disp": None if max_disp is None else float(max_disp),
            "dist_to_goal": dist,
            "target_pose": None if tp is None else tp.tolist(),
        }

    def score(self, outcome: Dict[str, Any]) -> Tuple[float, ...]:
        rank = _TIER_RANK.get(outcome["tier"], 2)
        dist = outcome.get("dist_to_goal")
        if self.use_dist_tiebreak and dist is not None:
            return (float(rank), float(dist))
        return (float(rank),)


class SuccessFitness(Fitness):
    """Minimal success-only fitness (``0`` if success else ``1``).

    Useful for tasks with no geometric goal, or when a plain success search is all
    that is wanted. No collision proxy, no distance term.
    """

    def outcome(self, task: Any, ctx: Dict[str, Any]) -> Dict[str, Any]:
        success = bool(task.check_success())
        return {"success": success, "tier": "HARD" if success else "FAIL"}

    def score(self, outcome: Dict[str, Any]) -> Tuple[float, ...]:
        return (0.0,) if outcome["success"] else (1.0,)


# Registry so a CLI ``--fitness <name>`` can resolve a fitness by string.
FITNESS_REGISTRY = {
    "oracle": OracleFitness,
    "success": SuccessFitness,
}


def build_fitness(name: str, **kwargs: Any) -> Fitness:
    """Instantiate a registered fitness by name (for CLI use)."""
    if name not in FITNESS_REGISTRY:
        raise KeyError(f"unknown fitness '{name}'; choices: {sorted(FITNESS_REGISTRY)}")
    return FITNESS_REGISTRY[name](**kwargs)
