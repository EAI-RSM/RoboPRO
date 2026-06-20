"""Online action-chunk perturbation for collision-aware BC.

At TRAINING time, take the dataset's action chunk (the BC target) and — only when the
conditioning frame is collision-free — push its in-contact future steps away from the
obstacle point cloud, using the SAME contact-gated smooth push as the offline relabel
(collision_relabel.smooth_push), but computed live per batch. Observations are never
touched; only the target chunk is corrected, so there is no image/action inconsistency.

Gate (per the design):
    * conditioning frame in contact  -> DON'T perturb (plain BC on the original chunk).
    * conditioning frame collision-free -> push the chunk's in-contact spheres to a
      depth-scaled standoff, smooth in time, grippers frozen, |dq| capped per arm.

The perturbation is a TARGET generator: it is wrapped in stop_gradient, so no policy
gradient flows through the inner optimization.

Batched + jittable: the inner push is a fixed-length optax.adam scan (default 12 steps),
vmapped implicitly over the batch via the batched nearest-distance.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training.collision_fk import make_sphere_fn
from openpi.training.collision_proximity import _nearest_dist_batched
from openpi.training.collision_relabel import ARM_MASK, LEFT, RIGHT


def make_online_perturb(
    fk_basis_path: str,
    scale: jnp.ndarray,            # [14] norm->phys affine (from build_joint_affine)
    bias: jnp.ndarray,             # [14]
    *,
    push: float = 0.02,            # push contacting spheres this far beyond measured (m)
    contact_clearance: float = 0.05,  # at a contact step, spheres under this are active (m)
    depth_gain: float = 1.0,       # deeper penetration -> pushed harder
    beta: float = 0.02,            # push-cost temperature (matches smooth relabel)
    w_push: float = 4.0,
    lam_vel: float = 6.0,
    lam_acc: float = 12.0,
    mu_anchor: float = 0.02,
    mu_end: float = 4.0,
    max_dq: float = 0.30,
    end_tau: float = 6.0,
    inner_steps: int = 12,
    inner_lr: float = 0.02,
    b_block: int = 8,
) -> Callable:
    """Returns perturb(actions_norm[B,H,A], P[B,M,3], contact_chunk[B,H]) -> safe[B,H,A].

    `contact_chunk[b,j]` is the per-step contact flag aligned with the action chunk;
    `contact_chunk[:,0]` is the conditioning-frame gate.
    """
    sphere_fn, radii = make_sphere_fn(fk_basis_path)
    radii_j = jnp.asarray(radii, jnp.float32)
    arm = jnp.asarray(ARM_MASK, jnp.float32)               # [14]
    scale = jnp.asarray(scale, jnp.float32)
    bias = jnp.asarray(bias, jnp.float32)

    def _clearance(q_phys, P):
        B, H, _ = q_phys.shape
        centers = sphere_fn(q_phys.reshape(B * H, 14)).reshape(B, H, -1, 3)   # [B,H,S,3]
        d_nn = _nearest_dist_batched(centers, P, b_block)                     # [B,H,S]
        return d_nn - radii_j[None, None, :]                                  # [B,H,S]

    def perturb(actions_norm: jnp.ndarray, P: jnp.ndarray, contact_chunk: jnp.ndarray) -> jnp.ndarray:
        B, H, A = actions_norm.shape
        q_phys = actions_norm[..., :14] * scale + bias                       # [B,H,14]

        d0 = _clearance(q_phys, P)                                           # [B,H,S]
        # contact_chunk[b,j] = stored (recorded) per-step contact flag, aligned with the
        # action chunk; active = at a recorded-contact step, spheres within contact_clearance.
        active = ((contact_chunk[..., None] > 0.5) & (d0 < contact_clearance)).astype(jnp.float32)
        pen = jnp.clip(-d0, 0.0, None)
        target = d0 + push + depth_gain * pen                                # [B,H,S]

        # Gate on the conditioning frame: perturb only if step 0 is NOT a recorded contact.
        gate = (contact_chunk[:, 0] < 0.5).astype(jnp.float32)               # [B] 1=perturb (clean frame)
        idx = jnp.arange(H)
        # Anchor ONLY the START (continuity from the current state); the chunk END is
        # free to land wherever the safe+smooth correction puts it (the last action need
        # not rejoin the original — there is no fixed trajectory after the chunk).
        w_end = jnp.exp(-idx / end_tau)                                      # [H]

        def loss(dd):                                                        # dd [B,H,14]
            dd = dd * arm
            clr = _clearance(q_phys + dd, P)                                 # [B,H,S]
            h = jnp.clip(target - clr, 0.0, None) * active
            push_c = (h ** 2).sum() / (2.0 * beta)
            vel = ((dd[:, 1:] - dd[:, :-1]) ** 2).sum()
            acc = ((dd[:, 2:] - 2 * dd[:, 1:-1] + dd[:, :-2]) ** 2).sum()
            anchor = (dd ** 2).sum()
            end = (w_end[None, :, None] * dd ** 2).sum()
            return w_push * push_c + lam_vel * vel + lam_acc * acc + mu_anchor * anchor + mu_end * end

        opt = optax.adam(inner_lr)
        dd0 = jnp.zeros_like(q_phys)
        st0 = opt.init(dd0)

        def body(carry, _):
            dd, st = carry
            g = jax.grad(loss)(dd)
            upd, st = opt.update(g, st)
            dd = optax.apply_updates(dd, upd) * arm
            return (dd, st), None

        (dd, _), _ = jax.lax.scan(body, (dd0, st0), None, length=inner_steps)

        # cap |dq| per arm side to max_dq (uniform rescale, like the offline relabel)
        for sl in (LEFT, RIGHT):
            m = jnp.abs(dd[:, :, sl]).max(axis=(1, 2), keepdims=True)        # [B,1,1]
            s = jnp.where(m > max_dq, max_dq / jnp.clip(m, 1e-9, None), 1.0)
            dd = dd.at[:, :, sl].multiply(s)

        dd = dd * gate[:, None, None]                                       # only clean-frame samples
        safe_q = q_phys + dd
        safe_joints = (safe_q - bias) / scale                               # back to normalized
        out = actions_norm.at[..., :14].set(safe_joints)
        return jax.lax.stop_gradient(out)

    return perturb
