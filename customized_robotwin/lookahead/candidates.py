"""Candidate next-chunk sources for lookahead search.

A :class:`CandidateSource` proposes K candidate next-chunks at the CURRENT task
state (without mutating the sim). The search branches each candidate, rolls it
forward and scores it.

The default :class:`PolicyCandidates` is **policy-agnostic**: it renders the
observation into the model and delegates the actual sampling to a ``propose_fn``.
When no ``propose_fn`` is given it uses :func:`noise_propose` — K flow-matching
NOISE re-samples, which works for any single-mode flow/diffusion policy. A policy
with a richer candidate mechanism (e.g. a WTA policy's latent action MODES) opts in
by providing its own proposer; :func:`mode_propose` is a ready-made helper such a
policy can use. The search core never mentions "modes": that concept lives entirely
in the policy layer behind the uniform ``propose_fn(model, obs, k) -> list[chunks]``
interface.

This is the only core module that touches a policy model, and it does so purely by
duck typing (``model.update_observation_window`` / ``model.observation_window`` /
``model.policy.infer`` / ``model.get_action``) — there is NO jax import here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import numpy as np

# A proposer turns the current (rendered) observation into K candidate chunks.
ProposeFn = Callable[[Any, Any, int], List[np.ndarray]]

# Default flow-matching noise shape (action horizon x action dim). Proposers accept
# overrides, so a policy with a different shape can pass its own.
AH_DEFAULT, AD_DEFAULT = 50, 32


# ------------------------------------------------------------------ inference helper
def _infer(model: Any, obs: Any, z: Optional[int] = None,
           noise: Optional[np.ndarray] = None) -> np.ndarray:
    """One action chunk from the model. Prefers ``policy.infer`` (mode/noise control);
    falls back to ``model.get_action`` when the model exposes no such knobs.

    ``z`` selects a latent mode via ``policy._sample_kwargs`` (ignored by single-mode
    policies); ``noise`` is the flow-matching noise. Both are optional so this helper
    serves the noise and mode proposers alike.
    """
    policy = getattr(model, "policy", None)
    if policy is not None and hasattr(policy, "infer"):
        try:
            policy._sample_kwargs = {} if z is None else {"z": int(z)}
        except Exception:  # noqa: BLE001 - single-mode policies lack this attr
            pass
        try:
            out = policy.infer(obs, noise=noise)
        except TypeError:
            out = policy.infer(obs)          # single-mode policy: no noise kwarg
        return np.asarray(out["actions"])
    return np.asarray(model.get_action())    # last-resort deterministic fallback


# ------------------------------------------------------------------ proposers
def noise_propose(model: Any, obs: Any, k: int, *,
                  action_horizon: int = AH_DEFAULT, action_dim: int = AD_DEFAULT,
                  seed: int = 0) -> List[np.ndarray]:
    """Policy-agnostic default: ``k`` flow-matching NOISE re-samples (single mode).

    Draws ``k`` independent noises (deterministic, seeded) and infers a chunk for
    each. Works for any flow/diffusion policy; a single-mode policy's outcome
    diversity comes purely from the noise.
    """
    rng = np.random.default_rng(seed)
    return [_infer(model, obs, z=None,
                   noise=rng.standard_normal((action_horizon, action_dim)).astype(np.float32))
            for _ in range(k)]


def mode_propose(model: Any, obs: Any, k: int, *,
                 action_horizon: int = AH_DEFAULT, action_dim: int = AD_DEFAULT,
                 seed: int = 0) -> List[np.ndarray]:
    """WTA-style helper: ``k`` latent action MODES ``z = 0..k-1`` under one fixed noise.

    A LIBRARY helper for policies that expose latent action modes (e.g. a WTA head).
    The single shared noise isolates the mode as the only varying factor. NOT used by
    the search core or ``run_search`` — a policy opts into it via its own
    ``propose_candidates`` hook (see the module docstring).
    """
    fixed = np.random.default_rng(seed).standard_normal((action_horizon, action_dim)).astype(np.float32)
    return [_infer(model, obs, z=z, noise=fixed) for z in range(k)]


# ------------------------------------------------------------------ sources
class CandidateSource(ABC):
    """Proposes K candidate next-chunks at the task's current state.

    Implementations MUST NOT mutate the sim (they only read the observation and
    sample). Each returned chunk is a ``float`` array of shape ``[chunk, action_dim]``
    ready to feed to ``rollback.apply_chunk``.
    """

    @abstractmethod
    def propose(self, task: Any, k: int) -> List[np.ndarray]:
        """Return ``k`` candidate chunks sampled from the current observation."""


class PolicyCandidates(CandidateSource):
    """K candidate chunks from a deploy-policy model, via a pluggable proposer.

    Parameters
    ----------
    model:
        A loaded deploy-policy model (``deploy_policy.get_model(...)``). Must expose
        ``update_observation_window(rgb, state)`` + ``observation_window`` and either
        ``policy.infer(obs, noise=...)`` or ``get_action()``.
    encode_obs:
        The policy module's ``encode_obs`` — turns ``task.get_obs()`` into
        ``(rgb_list, state)``.
    k:
        Default number of candidates (the search passes its own ``k`` to
        :meth:`propose`; this is a convenience default).
    propose_fn:
        ``propose_fn(model, obs, k) -> list[chunks]``. Defaults to
        :func:`noise_propose` (the policy-agnostic fallback). A policy overrides it
        with its own mechanism (e.g. :func:`mode_propose` for a WTA policy).
    chunk:
        Number of leading action rows kept from each sampled chunk (the executed
        horizon, e.g. ``pi0_step``).
    """

    def __init__(self, model: Any, encode_obs: Callable[[dict], Any], k: int = 6,
                 propose_fn: Optional[ProposeFn] = None, chunk: int = 50) -> None:
        self.model = model
        self.encode_obs = encode_obs
        self.k = int(k)
        self.propose_fn: ProposeFn = propose_fn or noise_propose
        self.chunk = int(chunk)

    def _render(self, task: Any) -> None:
        """Push the current observation into the model window (no sim mutation)."""
        rgb, state = self.encode_obs(task.get_obs())
        self.model.update_observation_window(rgb, state)

    def propose(self, task: Any, k: int) -> List[np.ndarray]:
        self._render(task)
        obs = getattr(self.model, "observation_window", None)
        chunks = self.propose_fn(self.model, obs, k)
        return [np.asarray(c)[:self.chunk] for c in chunks]


class CuroboCandidates(CandidateSource):
    """Seam for a CuRobo (motion-planner) candidate source.

    A planner variant would sample K goal/grasp hypotheses and return their joint
    trajectories as chunks. Deliberately unimplemented — wire a planner here without
    touching the search core.
    """

    def propose(self, task: Any, k: int) -> List[np.ndarray]:  # pragma: no cover
        raise NotImplementedError(
            "CuroboCandidates is a planned extension point; implement propose() to "
            "return K planner trajectories as [chunk, action_dim] arrays.")
