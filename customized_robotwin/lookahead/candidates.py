"""Candidate next-chunk sources for lookahead search.

A :class:`CandidateSource` proposes K candidate next-chunks at the CURRENT task
state (without mutating the sim). The search branches each candidate, rolls it
forward and scores it. The default :class:`PolicyCandidates` reuses a deploy-policy
model (pi05 / a WTA variant) and draws K chunks either by latent MODE (``z``) or by
K different flow-matching NOISE samples — mirroring
``outcome_diversity.sample_chunk`` / ``run_branch``.

This is the only core module that touches a policy model, and it does so purely by
duck typing (``model.get_action`` / ``model.policy.infer`` / ``model.observation_window``)
— there is NO jax import here, so the module byte-compiles and imports on the client
side. A ``CuroboCandidates`` seam is left for a planner-based source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

import numpy as np


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
    """K candidate chunks from a deploy-policy model.

    Parameters
    ----------
    model:
        A loaded deploy-policy model (``deploy_policy.get_model(...)``). Must expose
        ``update_observation_window(rgb, state)`` and either
        ``policy.infer(obs, noise=...)`` (preferred; enables mode/noise control) or
        ``get_action()`` (deterministic fallback).
    encode_obs:
        The policy module's ``encode_obs`` — turns ``task.get_obs()`` into
        ``(rgb_list, state)``.
    mode:
        ``"modes"`` samples latent modes ``z = 0..k-1`` under a single fixed noise
        (isolates a WTA policy's modes); ``"noise"`` fixes ``z`` and draws ``k``
        different noises (outcome diversity of a single-mode policy).
    chunk:
        Number of leading action rows to keep from each sampled chunk (the executed
        horizon, e.g. ``pi0_step``).
    action_horizon, action_dim:
        Shape of the flow-matching noise the model consumes (default 50 x 32).
    noise_seed:
        Base seed for reproducible noise draws.
    """

    def __init__(self, model: Any, encode_obs: Callable[[dict], Any],
                 mode: str = "modes", chunk: int = 50,
                 action_horizon: int = 50, action_dim: int = 32,
                 noise_seed: int = 0) -> None:
        if mode not in ("modes", "noise"):
            raise ValueError(f"mode must be 'modes' or 'noise', got {mode!r}")
        self.model = model
        self.encode_obs = encode_obs
        self.mode = mode
        self.chunk = int(chunk)
        self.ah = int(action_horizon)
        self.ad = int(action_dim)
        self.noise_seed = int(noise_seed)
        self._call = 0  # advances the noise stream so repeated calls differ

    # -- internals -------------------------------------------------------------
    def _render(self, task: Any) -> None:
        """Push the current observation into the model window (no sim mutation)."""
        rgb, state = self.encode_obs(task.get_obs())
        self.model.update_observation_window(rgb, state)

    def _noises(self, k: int, salt: int) -> List[np.ndarray]:
        rng = np.random.default_rng(self.noise_seed + salt)
        return [rng.standard_normal((self.ah, self.ad)).astype(np.float32) for _ in range(k)]

    def _sample(self, z: Optional[int], noise: Optional[np.ndarray]) -> np.ndarray:
        """One chunk from the model. Prefers ``policy.infer`` (mode/noise control);
        falls back to ``get_action`` when the model exposes no such knobs."""
        policy = getattr(self.model, "policy", None)
        obs = getattr(self.model, "observation_window", None)
        if policy is not None and hasattr(policy, "infer"):
            # WTA-style knob: pick the latent mode via _sample_kwargs
            try:
                policy._sample_kwargs = {} if z is None else {"z": int(z)}
            except Exception:  # noqa: BLE001 - single-mode policies lack this attr
                pass
            try:
                out = policy.infer(obs, noise=noise)
            except TypeError:
                # single-mode policy: no noise kwarg
                out = policy.infer(obs)
            return np.asarray(out["actions"])
        # last-resort deterministic fallback
        return np.asarray(self.model.get_action())

    # -- public ---------------------------------------------------------------
    def propose(self, task: Any, k: int) -> List[np.ndarray]:
        self._render(task)
        salt = self._call * 1000
        self._call += 1
        if self.mode == "modes":
            zs: List[Optional[int]] = list(range(k))
            fixed = self._noises(1, salt)[0]
            noises = [fixed] * k
        else:  # "noise"
            zs = [0] * k
            noises = self._noises(k, salt)
        return [self._sample(zs[i], noises[i])[:self.chunk] for i in range(k)]


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
