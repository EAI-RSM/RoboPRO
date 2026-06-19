"""Adjoint Matching core (model-agnostic) for fine-tuning a flow policy to a reward.

Implements plain Adjoint Matching (Domingo-Enrich et al., 2024) — the reward-only
half of QAM (no learned Q; the reward gradient is analytic, e.g. differentiable
proximity).  Ported from the reference QAM ``adj_matching`` / ``actor_loss``
(qam/agents/qam.py), generalized to take velocity-field *callables* so the same
code drives a toy MLP flow and pi05's transformer.

Convention (τ): t=0 is noise, t=1 is data — the QAM/AM convention.  pi05 uses the
opposite (t=1 noise → 0 data); wire it in with the adapter

    v_τ(x, t) = −v_pi05(x, 1 − t)

since both use the same linear interpolation x_t = (1−t)·noise + t·data.

Velocity callables take ``v(x, t)`` with x [..., A], t [..., 1] (any leading
batch dims; obs is closed over).  ``reward_grad_fn(x) -> ∇_x reward`` gives the
reward to MAXIMIZE evaluated at the generated terminal sample.

LoRA usage (pi05): non-residual mode with
    v_slow = base (LoRA off),  v_fast = base + LoRA (LoRA on)
trains the LoRA-induced velocity delta (v_fast − v_slow) onto the adjoint.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class AdjointConfig:
    flow_steps: int = 10         # SDE steps for the adjoint unroll (matches pi05 sample_actions / QAM)
    inv_temp: float = 1.0        # reward-gradient strength (1/KL-temperature)
    residual: bool = False       # False: v_fast is the full field (LoRA); True: v_fast is a residual


def _sigma(t: jnp.ndarray, h: float) -> jnp.ndarray:
    """Memoryless-SOC noise schedule for the linear-interpolation flow."""
    return jnp.sqrt(2.0 * (1.0 - t + h) / (t + h))


def adjoint_unroll(
    v_slow: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    v_fast: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    reward_grad_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    rng: jax.Array,
    cfg: AdjointConfig,
):
    """Roll out the controlled SDE from noise x0, then back-integrate the adjoint.

    Returns (xs [N,B,A], adjs [N,B,A], ts [N,B,1]) where N = flow_steps.
    """
    n = cfg.flow_steps
    h = 1.0 / n
    x = x0
    xs = [x]
    ts = []
    keys = jax.random.split(rng, n)
    for i in range(n):
        t = (i / n) * jnp.ones_like(x[..., :1])
        sig = _sigma(t, h)
        noise = jax.random.normal(keys[i], x.shape)
        if i != n - 1:
            v = v_fast(x, t) + (v_slow(x, t) if cfg.residual else 0.0)
            x = x + h * (2.0 * v - x / (t + h)) + jnp.sqrt(h) * sig * noise
        else:  # last step: ODE with the base field (AM paper)
            x = x + h * v_slow(x, t)
        xs.append(x)
        ts.append(t)

    # Terminal adjoint: −∇_x(reward)·inv_temp  (QAM sign; reward is maximized).
    adj = -reward_grad_fn(xs[-1]) * cfg.inv_temp
    adjs = []
    for i in reversed(range(n)):
        t = (i / n) * jnp.ones_like(x[..., :1])

        def fn(xi):                                   # base-field drift (memoryless)
            return 2.0 * v_slow(xi, t + h) - xi / (t + h)

        vjp = jax.vjp(fn, xs[i])[1](adj)[0]           # [∂fn/∂x]ᵀ adj
        adj = adj + h * vjp
        adjs.append(adj)

    return jnp.stack(xs[:-1], 0), jnp.stack(list(reversed(adjs)), 0), jnp.stack(ts, 0)


def adjoint_loss(
    v_slow: Callable,
    v_fast: Callable,
    reward_grad_fn: Callable,
    x0: jnp.ndarray,
    rng: jax.Array,
    cfg: AdjointConfig,
) -> jnp.ndarray:
    """Adjoint-matching loss for the fast/control field. Scalar."""
    xs, adjs, ts = adjoint_unroll(v_slow, v_fast, reward_grad_fn, x0, rng, cfg)
    h = 1.0 / cfg.flow_steps
    sig = _sigma(ts, h)
    vf = v_fast(xs, ts)
    if cfg.residual:
        per = jnp.square(vf * 2.0 / sig + sig * adjs).sum(-1)
    else:
        vb = v_slow(xs, ts)
        per = jnp.square((vf - vb) * 2.0 / sig + sig * adjs).sum(-1)
    return jnp.mean(jnp.sum(per, axis=0))   # sum over flow steps, mean over batch
