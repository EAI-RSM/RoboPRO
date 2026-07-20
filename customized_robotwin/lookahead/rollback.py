"""Snapshot / rollback core for lookahead search ("branch & rollback").

This module consolidates the exact-state save/restore primitives that make a
future-tree search possible on the RoboTwin simulator: PhysX ``pack()``/``unpack()``
(exact rigid-body + articulation state) *plus* the controller drive targets that
``pack()`` does NOT capture. Restoring both, and clearing the latched
``eval_success`` flag, lets the search branch a candidate chunk, roll it forward,
score the outcome, then roll back and try another candidate deterministically.

The primitives were validated in
``policy/pi05/scripts/outcome_diversity.py`` (err=0 restores); they are lifted
here verbatim in behaviour and given a documented :class:`TaskAdapter` contract so
the search core stays decoupled from any particular RoboTwin task class.

No jax / no policy imports: this module only needs numpy and the ``task`` object.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

# A raw action = one control-step command row (whatever width take_action expects,
# e.g. 14 for a dual-arm aloha qpos action, or the model's padded 32-d row — the
# task slices what it needs). A chunk = a sequence of such rows.
Action = np.ndarray
Chunk = np.ndarray


@runtime_checkable
class TaskAdapter(Protocol):
    """Structural contract a RoboTwin task must satisfy to be searched.

    Every concrete RoboTwin task (``envs/*.py`` subclassing ``Base_Task``) already
    satisfies this; the Protocol documents *exactly* what the search core touches so
    the core can be reused with any sim exposing the same surface. Only the members
    below are used by :mod:`lookahead.rollback` / :mod:`lookahead.search`.

    Required attributes / methods
    -----------------------------
    scene:
        SAPIEN scene handle. Must provide ``get_physx_system()`` (with ``.pack()``
        / ``.unpack(blob)``), ``get_all_articulations()``, ``get_all_actors()`` and
        ``step()``.
    eval_success: bool
        Latched python success flag. Set True inside ``take_action`` on the first
        ``check_success`` and never auto-reset; :func:`restore` clears it.
    take_action(action) -> None:
        Apply one raw action row (drives the controllers + steps physics).
    get_obs() -> dict:
        Current observation dict (cameras + ``joint_action``), the policy input.
    check_success() -> bool:
        Task success predicate (unlatched; recomputed from live state).
    _update_render() -> None:
        Refresh camera / render state (used by :func:`settle`).
    """

    scene: Any
    eval_success: bool

    def take_action(self, action: Action) -> None: ...
    def get_obs(self) -> Dict[str, Any]: ...
    def check_success(self) -> bool: ...
    def _update_render(self) -> None: ...


# ------------------------------------------------------------------ drive targets
def snapshot_drive_targets(task: TaskAdapter) -> List[List[Tuple[Any, Any]]]:
    """Capture every articulation joint's (position, velocity) drive target.

    ``PhysxSystem.pack()`` saves dynamic state (poses, velocities) but NOT the
    controller drive targets, so they must be snapshotted separately or a restored
    branch would keep driving toward the wrong target.
    """
    targets: List[List[Tuple[Any, Any]]] = []
    for art in task.scene.get_all_articulations():
        targets.append([(j.get_drive_target(), j.get_drive_velocity_target())
                        for j in art.get_active_joints()])
    return targets


def restore_drive_targets(task: TaskAdapter,
                          targets: Sequence[Sequence[Tuple[Any, Any]]]) -> None:
    """Restore drive targets captured by :func:`snapshot_drive_targets`."""
    for art, art_targets in zip(task.scene.get_all_articulations(), targets):
        for j, (pos_t, vel_t) in zip(art.get_active_joints(), art_targets):
            j.set_drive_target(pos_t)
            j.set_drive_velocity_target(vel_t)


# ------------------------------------------------------------------ fingerprint
def state_fingerprint(task: TaskAdapter) -> np.ndarray:
    """Deterministic float64 vector summarising the full sim state.

    Concatenates every actor pose (p, q) and every articulation
    (root pose, qpos, qvel). Two states with an equal fingerprint (within atol)
    are the same scene configuration; this is the arbiter the replay pins use to
    prove a from-start raw-action replay reproduced the searched trajectory.
    """
    parts: List[np.ndarray] = []
    for a in task.scene.get_all_actors():
        p = a.get_pose()
        parts.append(np.concatenate([np.asarray(p.p), np.asarray(p.q)]))
    for art in task.scene.get_all_articulations():
        rp = art.get_root_pose()
        parts.append(np.concatenate([
            np.asarray(rp.p), np.asarray(rp.q),
            np.asarray(art.get_qpos()), np.asarray(art.get_qvel())]))
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts).astype(np.float64)


# ------------------------------------------------------------------ snapshot / restore
def snapshot(task: TaskAdapter) -> Dict[str, Any]:
    """Full branch point: physx pack + drive targets + a state fingerprint.

    Returns an in-memory dict ``{"physx", "drive", "fingerprint"}``. The physx blob
    is an opaque SAPIEN handle (never serialised into a Plan — only the fingerprint
    is portable). Restoring this exact dict reproduces the scored state.
    """
    return {
        "physx": task.scene.get_physx_system().pack(),
        "drive": snapshot_drive_targets(task),
        "fingerprint": state_fingerprint(task),
    }


def restore(task: TaskAdapter, snap: Dict[str, Any]) -> None:
    """Inverse of :func:`snapshot`.

    Unpacks physx state, restores the controller drive targets, and CRUCIALLY
    clears ``task.eval_success``. ``eval_success`` is a latched flag: once any
    branch reaches success, ``take_action`` becomes a no-op for every subsequent
    branch until it is reset — so failing to clear it silently corrupts the search.
    """
    task.scene.get_physx_system().unpack(snap["physx"])
    restore_drive_targets(task, snap["drive"])
    task.eval_success = False


# ------------------------------------------------------------------ rollout helpers
def apply_chunk(task: TaskAdapter, chunk: Chunk) -> None:
    """Execute a chunk = sequence of raw action rows via ``take_action``.

    Once ``check_success`` latches ``eval_success`` mid-chunk, ``take_action``
    ignores the remaining rows (both here and on a fresh replay), so a chunk that
    reaches success stops advancing at the same row on replay — the stored raw
    actions remain faithful.
    """
    for a in chunk:
        task.take_action(a)


def settle(task: TaskAdapter, n: int) -> None:
    """Step physics ``n`` times with no new drive targets.

    Lets a placement / gripper-open release stabilise (or dynamics quiesce) without
    issuing new commands — physics only, plus a render refresh so any recorded
    frames stay in sync.
    """
    for _ in range(int(n)):
        task.scene.step()
        task._update_render()
