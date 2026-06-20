"""Shared collision-geometry core: scene point-cloud prep + contact-gated SMOOTH push.

Extracted so it can be reused by:
  * scripts/convert_collision_am_to_lerobot.py  (obstacle-point prep)
  * scripts/relabel_actions_smooth.py           (offline smooth relabel — deprecated)
  * the ONLINE action-chunk perturbation used during training (train-time relabel)

The optimization (`smooth_push`) is identical to the offline smooth relabel: at the
spheres in contact, push their clearance out to a depth-scaled target, while keeping
the correction smooth in time (velocity + acceleration), trust-region-anchored, and
rejoining the original at the ends. Grippers are frozen (ARM_MASK).

Sphere FK comes from collision_fk.make_sphere_fn (validated against the torch FK).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Arm joints only (aloha 14-dof: grippers at indices 6 and 13 are frozen).
ARM_MASK = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], dtype=np.float32)
LEFT = slice(0, 6)
RIGHT = slice(7, 13)


# ── scene point-cloud preparation (numpy) ────────────────────────────────────
def load_obstacle_points(scene_dir: Path, exclude_tags: set[str],
                         exclude_names=frozenset(), include_names=frozenset()) -> np.ndarray:
    """Per-object surface samples minus excluded-tag / excluded-name objects."""
    z = np.load(Path(scene_dir) / "scene.npz")
    samples, obj_id = z["samples"].astype(np.float32), z["obj_id"]
    objects = json.loads((Path(scene_dir) / "objects.json").read_text())
    excluded = {o["per_scene_id"] for o in objects
                if (set(o["tags"]) & exclude_tags or o["name"] in exclude_names)
                and o["name"] not in include_names}
    keep = ~np.isin(obj_id, sorted(excluded))
    return samples[keep]


def prefilter_points(P: np.ndarray, centers0: np.ndarray, buffer: float) -> np.ndarray:
    """Keep only obstacle points inside the arm's swept bounding box + buffer."""
    flat = np.asarray(centers0).reshape(-1, 3)
    lo = flat.min(0) - buffer
    hi = flat.max(0) + buffer
    keep = ((P >= lo) & (P <= hi)).all(-1)
    return P[keep]


def voxel_downsample(P: np.ndarray, voxel: float) -> np.ndarray:
    """One point per occupied voxel (margin >> voxel, so clearance is preserved)."""
    if voxel <= 0 or len(P) == 0:
        return P
    keys = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return P[np.sort(idx)]


# ── differentiable clearance / push cost (JAX) ───────────────────────────────
def nearest_dist(centers: jnp.ndarray, P: jnp.ndarray, t_block: int = 50) -> jnp.ndarray:
    """Nearest-obstacle-point distance per sphere, chunked over time to bound memory.

    centers [T,S,3], P [M,3] -> d_nn [T,S]. jnp.min over P is differentiable.
    """
    T = centers.shape[0]
    outs = []
    for s in range(0, T, t_block):
        c = centers[s:s + t_block]                                   # [t,S,3]
        d2 = ((c[:, :, None, :] - P[None, None, :, :]) ** 2).sum(-1)  # [t,S,M]
        outs.append(jnp.sqrt(jnp.clip(d2.min(-1), 1e-12, None)))
    return jnp.concatenate(outs, axis=0)                             # [T,S]


def contact_push_cost(centers, radii, P, target, active, beta, t_block):
    """Push active spheres' clearance up to a per-sphere target.  -> scalar.

    cost = Σ_{active} relu(target − clearance(q))² / (2β); target/active are constants.
    """
    d_nn = nearest_dist(centers, P, t_block)             # [T,S]
    clearance = d_nn - radii[None, :]                    # [T,S]
    h = jnp.clip(target - clearance, 0.0, None) * active
    return (h ** 2).sum() / (2.0 * beta)


def smooth_push(q0, sphere_fn, radii, P, active, target, *, steps, lr, beta,
                w_push, lam_vel, lam_acc, mu_anchor, mu_end, max_dq, t_block, end_tau):
    """Contact-gated SMOOTH push: move active spheres to `target` clearance with a
    correction field d=(q−q0) that is smooth in time (vel+acc), trust-region anchored,
    and rejoins the original at the ends. Grippers frozen. Returns (q_safe, d).
    """
    q0 = jnp.asarray(q0, jnp.float32)
    radii = jnp.asarray(radii, jnp.float32)
    P = jnp.asarray(P, jnp.float32)
    active = jnp.asarray(active, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    T = q0.shape[0]
    arm = jnp.asarray(ARM_MASK)[None, :]
    idx = jnp.arange(T)
    w_end = (jnp.exp(-idx / end_tau) + jnp.exp(-(T - 1 - idx) / end_tau))[:, None]

    def loss_fn(d):
        d = d * arm
        centers = sphere_fn(q0 + d)
        push = contact_push_cost(centers, radii, P, target, active, beta, t_block)
        vel = ((d[1:] - d[:-1]) ** 2).sum()
        acc = ((d[2:] - 2 * d[1:-1] + d[:-2]) ** 2).sum()
        anchor = (d ** 2).sum()
        end = (w_end * d ** 2).sum()
        return (w_push * push + lam_vel * vel + lam_acc * acc
                + mu_anchor * anchor + mu_end * end)

    opt = optax.adam(lr)
    d = jnp.zeros_like(q0)
    opt_state = opt.init(d)

    @jax.jit
    def step(d, opt_state):
        g = jax.grad(loss_fn)(d)
        updates, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(d, updates) * arm, opt_state

    for _ in range(steps):
        d, opt_state = step(d, opt_state)

    d = np.array(d * arm, dtype=np.float32)
    for sl in (LEFT, RIGHT):
        m = float(np.abs(d[:, sl]).max())
        if m > max_dq:
            d[:, sl] *= max_dq / m
    return np.asarray(q0) + d, d
