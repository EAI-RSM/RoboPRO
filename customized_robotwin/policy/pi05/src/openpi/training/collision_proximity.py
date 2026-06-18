"""Differentiable proximity reward for adjoint-matching fine-tuning of pi05.

Shared between the offline relabel (scripts/relabel_actions_jax.py) and the
adjoint-matching trainer.  Provides a pure-JAX, batched, per-sample collision
cost that is differentiable w.r.t. a *normalized* pi05 action chunk — i.e. it
yields ∇_a r at arbitrary generated actions, which is exactly what Adjoint
Matching's terminal adjoint needs.

Reward  r(a) = −proximity_cost(a)   (higher = safer).  Only the gradient ∇_a r
is used by the adjoint, so the absolute scale folds into the AM temperature.

Geometry (matches collect_rollout_proximity_client / relabel_actions_jax):
    clearance = nearest_obstacle_point_dist − sphere_radius
    cost      = Σ squared-hinge(margin − clearance), depth-weighted
Sphere FK comes from fk_basis.npz via collision_fk.make_sphere_fn (validated).

Action-space note: pi05 acts in normalized "pi" space (quantile norm + the
AlohaInputs adapt_to_pi joint flip).  ``unnormalize_to_physical`` maps a
normalized action chunk back to physical aloha joint radians (grippers passed
through, ignored by FK) so the sphere FK sees the right configs.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training.collision_fk import make_sphere_fn

# AlohaInputs adapt_to_pi joint-flip mask (its own inverse; grippers at 6,13 are 1).
JOINT_FLIP = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1], dtype=np.float32)


def build_joint_affine(norm_stats, use_quantiles: bool):
    """Fold (un-normalize → un-flip) into a single affine on the first 14 dims.

    q_phys = action_norm[..., :14] * scale + bias  (aloha physical radians).
    norm_stats is the "actions" NormStats (has .q01/.q99 or .mean/.std).
    Returns (scale[14], bias[14]) as jnp.float32.
    """
    flip = JOINT_FLIP
    if use_quantiles:
        q01 = np.asarray(norm_stats.q01[:14], np.float32)
        q99 = np.asarray(norm_stats.q99[:14], np.float32)
        D = (q99 - q01) + 1e-6
        A, B = D / 2.0, D / 2.0 + q01          # x_phys = (x+1)/2*D + q01
    else:
        A = np.asarray(norm_stats.std[:14], np.float32) + 1e-6
        B = np.asarray(norm_stats.mean[:14], np.float32)
    return jnp.asarray(flip * A), jnp.asarray(flip * B)


def unnormalize_to_physical(action_norm: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """[..., A>=14] normalized pi-space action -> [..., 14] physical aloha joints."""
    return action_norm[..., :14] * scale + bias


def _nearest_dist_batched(centers: jnp.ndarray, P: jnp.ndarray, b_block: int) -> jnp.ndarray:
    """Per-sample nearest-obstacle-point distance, chunked over batch to bound memory.

    centers [B,H,S,3], P [B,M,3] (per-sample obstacle points, padded with a far
    sentinel) -> d_nn [B,H,S].  ||c−p||² via the expanded form; jnp.min over P is
    differentiable (gradient flows to the nearest point).
    """
    B = centers.shape[0]
    outs = []
    for s in range(0, B, b_block):
        c = centers[s:s + b_block]                       # [b,H,S,3]
        p = P[s:s + b_block]                             # [b,M,3]
        c2 = (c ** 2).sum(-1)                            # [b,H,S]
        p2 = (p ** 2).sum(-1)                            # [b,M]
        cp = jnp.einsum("bhsi,bmi->bhsm", c, p)          # [b,H,S,M]
        d2 = c2[..., None] + p2[:, None, None, :] - 2.0 * cp
        outs.append(jnp.sqrt(jnp.clip(d2.min(-1), 1e-12, None)))
    return jnp.concatenate(outs, axis=0)                 # [B,H,S]


def make_proximity_cost(
    fk_basis_path: str,
    *,
    margin: float = 0.03,
    depth_gain: float = 1.0,
    beta: float = 1.0,
    b_block: int = 8,
) -> tuple[Callable, np.ndarray]:
    """Build cost_fn(q_phys[B,H,14], P[B,M,3]) -> per-sample cost[B], differentiable.

    Returns (cost_fn, radii).  Reward for adjoint matching is r = −cost.
    """
    sphere_fn, radii = make_sphere_fn(fk_basis_path)
    radii_j = jnp.asarray(radii, jnp.float32)

    def cost_fn(q_phys: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        B, H, _ = q_phys.shape
        centers = sphere_fn(q_phys.reshape(B * H, 14)).reshape(B, H, -1, 3)  # [B,H,S,3]
        d_nn = _nearest_dist_batched(centers, P, b_block)                    # [B,H,S]
        h = jnp.clip(margin - (d_nn - radii_j[None, None, :]), 0.0, None)
        return ((h ** 2) * (1.0 + depth_gain * h / margin)).sum(axis=(1, 2)) / (2.0 * beta)

    return cost_fn, radii


def make_action_reward(
    fk_basis_path: str,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    **cost_kwargs,
) -> Callable:
    """Reward as a function of the NORMALIZED action chunk (for the AM adjoint).

    reward_fn(action_norm[B,H,A], P[B,M,3]) -> r[B]  (= −proximity_cost).
    ∇_{action_norm} reward_fn is the terminal-adjoint signal for adjoint matching.
    """
    cost_fn, _ = make_proximity_cost(fk_basis_path, **cost_kwargs)

    def reward_fn(action_norm: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
        q_phys = unnormalize_to_physical(action_norm, scale, bias)
        return -cost_fn(q_phys, P)

    return reward_fn
