"""Pure-JAX forward kinematics for the aloha-agilex dual arm.

Ported from the numpy ``SerializedArm`` in
``customized_robotwin/script/relabel_collision_dataset.py`` and the FK basis
serialized by ``collect_rollout_proximity_client.py`` (``fk_basis.npz``).

Provides ``make_fk_fn`` → a pure JAX callable

    fk_fn(q: [N, 14]) -> x_links: [N, 16, 3]

mapping 14-D physical joint configs to the 16 per-link *centre* positions in
world frame, in the SAME order and frame the collector used when it recorded
``link_distances/delta`` (``delta = obstacle_surface − link_center``).  This is
exactly the ``x_links`` that ``collision_critic.collision_loss`` expects.

Joint layout of the 14-D action/state vector (see MOTORS in
convert_rollout_to_lerobot.py):
    [0:6]  left arm joints    [6]  left gripper
    [7:13] right arm joints   [13] right gripper
Only the 12 arm joints drive FK; the 2 gripper columns are ignored.

Link order — the order ``collect_rollout_proximity_client`` writes into
``link_distances.attrs['link_names']`` for this embodiment (interleaved L/R for
the 6 chain links, then the 4 end-effector links):

    fl_link1 fr_link1 fl_link2 fr_link2 fl_link3 fr_link3
    fl_link4 fr_link4 fl_link5 fr_link5 fl_link6 fr_link6
    fl_link7 fl_link8 fr_link7 fr_link8

Per-arm link→transform mapping (uniform):
    linkN, N∈1..6  → chain transform Ts[N-1]
    link7          → Ts[5] @ ee_offsets[0]
    link8          → Ts[5] @ ee_offsets[1]
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

# Default 16-link order as written by collect_rollout_proximity_client.py for
# the aloha-agilex embodiment.  fk_fn output rows follow this order so they line
# up with the dataset's link_distances[..., l, :] rows.
DEFAULT_LINK_ORDER = (
    "fl_link1", "fr_link1", "fl_link2", "fr_link2", "fl_link3", "fr_link3",
    "fl_link4", "fr_link4", "fl_link5", "fr_link5", "fl_link6", "fr_link6",
    "fl_link7", "fl_link8", "fr_link7", "fr_link8",
)

# Action-vector slices for the two arms (6 revolute joints each).
LEFT_ARM_SLICE = slice(0, 6)
RIGHT_ARM_SLICE = slice(7, 13)


def _skew(a: jnp.ndarray) -> jnp.ndarray:
    """Skew-symmetric matrix of a 3-vector (3,) -> (3, 3)."""
    return jnp.array(
        [[0.0, -a[2], a[1]],
         [a[2], 0.0, -a[0]],
         [-a[1], a[0], 0.0]],
        dtype=a.dtype,
    )


def _rot_about_axis(axis: jnp.ndarray, point: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """(N,) joint angles -> (N, 4, 4) homogeneous rotations about line (point, axis).

    Rodrigues' formula, matching ``_rot_about_axis_np`` in relabel_collision_dataset.
    """
    K = _skew(axis)                                  # (3, 3)
    s = jnp.sin(q)[:, None, None]                    # (N, 1, 1)
    c = jnp.cos(q)[:, None, None]                    # (N, 1, 1)
    R = jnp.eye(3, dtype=q.dtype) + s * K + (1.0 - c) * (K @ K)   # (N, 3, 3)
    t = point - jnp.einsum("nij,j->ni", R, point)    # (N, 3)
    N = q.shape[0]
    M = jnp.zeros((N, 4, 4), dtype=q.dtype)
    M = M.at[:, :3, :3].set(R)
    M = M.at[:, :3, 3].set(t)
    M = M.at[:, 3, 3].set(1.0)
    return M


def _chain_transforms(params: dict, Q: jnp.ndarray) -> list[jnp.ndarray]:
    """Q (N, 6) -> list of 6 (N, 4, 4) world transforms for one arm."""
    N = Q.shape[0]
    T = jnp.broadcast_to(params["T_base"], (N, 4, 4))
    out = []
    for i in range(6):
        M = _rot_about_axis(params["axes"][i], params["points"][i], Q[:, i])
        T = T @ params["O"][i] @ M
        out.append(T)
    return out


def _load_side(z, side: str) -> dict:
    g = lambda k: jnp.asarray(z[f"{side}_{k}"], dtype=jnp.float32)
    return {
        "T_base": g("T_base"),        # (4, 4)
        "O": g("O"),                  # (6, 4, 4)
        "axes": g("axes"),            # (6, 3)
        "points": g("points"),        # (6, 3)
        "ee_offsets": g("ee_offsets"),  # (2, 4, 4)
    }


def _link_spec(name: str) -> tuple[str, str, int]:
    """'fl_link3' -> ('left', 'chain', 2);  'fr_link7' -> ('right', 'ee', 0)."""
    side = "left" if name.startswith("fl") else "right"
    num = int(name.split("link")[-1])
    if num <= 6:
        return side, "chain", num - 1
    return side, "ee", num - 7


def _slot_transforms(params: dict, Q: jnp.ndarray) -> jnp.ndarray:
    """Q (N, 6) -> (N, 8, 4, 4): the 6 chain frames + 2 end-effector frames.

    Slot index convention: 0..5 = chain links, 6 = ee_offsets[0], 7 = ee_offsets[1].
    """
    Ts = _chain_transforms(params, Q)               # list of 6 (N, 4, 4)
    ee0 = Ts[5] @ params["ee_offsets"][0]            # (N, 4, 4)
    ee1 = Ts[5] @ params["ee_offsets"][1]
    return jnp.stack([*Ts, ee0, ee1], axis=1)        # (N, 8, 4, 4)


def make_sphere_fn(
    fk_basis_path: str,
) -> tuple[Callable[[jnp.ndarray], jnp.ndarray], np.ndarray]:
    """Build a pure-JAX sphere_fn(q[N, 14]) -> centers[N, S, 3] and radii[S].

    Ports ``SerializedArm.sphere_centers``: every collision sphere is rigidly
    attached to a chain/ee slot frame; its world centre is Tm @ local_centre.
    S = 35 (left) + 32 (right) = 67 for aloha-agilex.  Left spheres come first,
    then right — matching ``np.concatenate([cL, cR], axis=1)`` in the labeler.
    """
    z = np.load(fk_basis_path, allow_pickle=False)
    sides_params = {"left": _load_side(z, "left"), "right": _load_side(z, "right")}

    side_local = {}   # side -> (local_centers [S_side,3], slot_idx [S_side], radii [S_side])
    radii_all = []
    for side in ("left", "right"):
        slot_type = [str(x) for x in z[f"{side}_sph_slot_type"]]
        slot_idx_raw = z[f"{side}_sph_slot_idx"].astype(int)
        slot = np.array(
            [si if t == "chain" else 6 + si for t, si in zip(slot_type, slot_idx_raw)],
            dtype=np.int32,
        )
        local = jnp.asarray(z[f"{side}_sph_center"], dtype=jnp.float32)   # (S_side, 3)
        side_local[side] = (local, jnp.asarray(slot))
        radii_all.append(z[f"{side}_sph_radius"].astype(np.float32))
    radii = np.concatenate(radii_all)   # (S,)

    def sphere_fn(q: jnp.ndarray) -> jnp.ndarray:
        cols = []
        for side, qslice in (("left", LEFT_ARM_SLICE), ("right", RIGHT_ARM_SLICE)):
            slotTs = _slot_transforms(sides_params[side], q[:, qslice])   # (N,8,4,4)
            local, slot = side_local[side]
            Tm = slotTs[:, slot]                                          # (N,S_side,4,4)
            c = jnp.einsum("nsij,sj->nsi", Tm[..., :3, :3], local) + Tm[..., :3, 3]
            cols.append(c)
        return jnp.concatenate(cols, axis=1)   # (N, S, 3)

    return sphere_fn, radii


def make_fk_fn(
    fk_basis_path: str,
    link_order: tuple[str, ...] = DEFAULT_LINK_ORDER,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a pure-JAX fk_fn(q[N, 14]) -> x_links[N, L, 3] from an fk_basis.npz.

    L = len(link_order) (16 for aloha-agilex).  Output row l is the world-frame
    centre of link_order[l], matching the dataset's link_distances row order.
    """
    z = np.load(fk_basis_path, allow_pickle=False)
    sides = {"left": _load_side(z, "left"), "right": _load_side(z, "right")}
    specs = [_link_spec(n) for n in link_order]

    def fk_fn(q: jnp.ndarray) -> jnp.ndarray:
        qL = q[:, LEFT_ARM_SLICE]    # (N, 6)
        qR = q[:, RIGHT_ARM_SLICE]   # (N, 6)
        Ts = {
            "left": _chain_transforms(sides["left"], qL),
            "right": _chain_transforms(sides["right"], qR),
        }
        cols = []
        for side, kind, idx in specs:
            if kind == "chain":
                c = Ts[side][idx][:, :3, 3]
            else:  # end-effector link: offset from the last chain frame
                T = Ts[side][5] @ sides[side]["ee_offsets"][idx]
                c = T[:, :3, 3]
            cols.append(c)
        return jnp.stack(cols, axis=1)   # (N, L, 3)

    return fk_fn
