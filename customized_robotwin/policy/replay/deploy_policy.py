"""Replay policy — deterministic playback of a searched motion Plan.

Implements the standard RoboTwin policy interface (``get_model`` / ``eval`` /
``reset_model`` + ``encode_obs``) so ``--policy_name replay`` plugs into BOTH
``script/eval_policy.py`` and ``script/collect_rollout_client.py`` unmodified. Unlike
a normal policy it runs NO inference: :class:`ReplayModel` streams the plan's stored
raw action chunks straight to ``take_action``. Because raw-action playback is
deterministic, it reproduces the searched trajectory exactly and lets you enrich
observations / annotations at replay time.

Verification is governed by a configurable :class:`~lookahead.pins.PinPolicy`
(the ``pins:`` block of ``deploy_policy.yml``): the replay checks that the scene it
replays into matches the scene the plan was searched in (provenance, task spec,
control, and — the gray-zone arbiter — the t0 state fingerprint). Any single pin can
be set to ``enforce`` / ``warn`` / ``off``.

Seed / plan resolution: RoboTwin does NOT store the episode seed on the task
(``getattr(task, "seed", None)`` is ``None``), so the model selects the plan for a
configured seed (``seed`` in yml / usr_args), or a single ``plan_path``. If a future
env DOES expose ``task.seed``, :meth:`ReplayModel.select_for_task` will honour it.
"""

from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np

# Make the customized_robotwin root importable so ``lookahead`` resolves regardless
# of the caller's cwd (eval_policy / collect_rollout_client both append "./" too).
_CRW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _CRW_ROOT not in sys.path:
    sys.path.append(_CRW_ROOT)

from lookahead import pins as pins_mod
from lookahead import plan as plan_mod
from lookahead import rollback as rb


# --------------------------------------------------------------------- obs encoding
def encode_obs(observation: Dict[str, Any]):
    """Match the pi05 obs encoding so the interface is identical.

    Replay does not consume the observation for action selection, but keeping the
    same signature means ``eval`` mirrors the standard policies and downstream code
    that expects ``encode_obs`` keeps working.
    """
    input_rgb_arr = [
        observation["observation"]["countertop_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


# --------------------------------------------------------------------- provenance helpers
def _git_head(repo_dir: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def _config_hash(task_config: Optional[str]) -> Optional[str]:
    if not task_config:
        return None
    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench" and os.getenv("BENCH_ROOT"):
        path = os.path.join(os.environ["BENCH_ROOT"], "bench_task_config", f"{task_config}.yml")
    else:
        path = os.path.join("task_config", f"{task_config}.yml")
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------- the model
class ReplayModel:
    """Streams a plan's raw action chunks and runs the PinPolicy checks.

    Parameters (from ``usr_args`` / yml)
    ------------------------------------
    plan_dir:
        Directory of ``seed<N>.json`` plans (a plan store keyed by seed). Optional.
    plan_path:
        A single plan json (used when no per-seed store is given). Optional.
    seed:
        The configured seed to select from ``plan_dir`` when the task does not
        expose its own seed.
    chunk:
        Rows emitted per ``get_action`` (defaults to the plan's control chunk).
    pins:
        A :class:`~lookahead.pins.PinPolicy` config dict (see yml ``pins:`` block).
    current:
        ``{task_name, task_config, embodiment, chunk, robotwin_commit,
        bench_config_hash}`` describing the CURRENT harness, for the static pins.
    """

    # local ReplayModel exposes this; the collect_rollout_client TCP proxy hides any
    # underscore attribute (raises AttributeError), so ``getattr(model,
    # "_is_local_replay", False)`` cleanly distinguishes the two — task-dependent
    # pins only run on the local model.
    _is_local_replay = True

    def __init__(self, plan_dir: Optional[str] = None, plan_path: Optional[str] = None,
                 seed: Optional[int] = None, chunk: Optional[int] = None,
                 pin_policy: Optional[pins_mod.PinPolicy] = None,
                 current: Optional[Dict[str, Any]] = None) -> None:
        self.plan_dir = plan_dir
        self.plan_path = plan_path
        self.seed = None if seed is None else int(seed)
        self.forced_chunk = None if chunk is None else int(chunk)
        self.pins = pin_policy or pins_mod.default_policy()
        self.current = current or {}

        # index the plan store: seed -> path
        self._store: Dict[int, str] = {}
        if plan_dir and os.path.isdir(plan_dir):
            for p in glob.glob(os.path.join(plan_dir, "seed*.json")):
                stem = os.path.splitext(os.path.basename(p))[0]  # "seed40000"
                try:
                    self._store[int(stem[len("seed"):])] = p
                except ValueError:
                    continue

        # mirrored for the ModelClient proxy / interface parity
        self.observation_window = None

        # per-episode replay state
        self.plan: Optional[plan_mod.Plan] = None
        self.actions: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.chunk: int = self.forced_chunk or 1
        self.cursor: int = 0
        self._t0_checked: bool = False
        self._exhausted: bool = False

        self.select(self.seed)

    # -- plan selection -------------------------------------------------------
    def _resolve_path(self, seed: Optional[int]) -> Optional[str]:
        if seed is not None and seed in self._store:
            return self._store[seed]
        if self.plan_path:
            return self.plan_path
        if len(self._store) == 1:  # single plan in the store -> use it
            return next(iter(self._store.values()))
        return None

    def select(self, seed: Optional[int]) -> None:
        """Load and arm the plan for ``seed`` (or the single configured plan)."""
        path = self._resolve_path(seed)
        self.cursor = 0
        self._t0_checked = False
        self._exhausted = False
        if path is None:
            self.plan = None
            self.actions = np.zeros((0, 0), dtype=np.float32)
            print(f"\033[93m[replay] no plan for seed={seed} "
                  f"(plan_dir={self.plan_dir}, plan_path={self.plan_path})\033[0m")
            return
        self.plan = plan_mod.load(path)
        self.actions = self.plan.actions_array()
        self.chunk = self.forced_chunk or int(
            self.plan.task_spec.control.get("chunk", 1)) or 1
        print(f"\033[36m[replay] armed plan {os.path.basename(path)}: "
              f"{len(self.actions)} actions, chunk={self.chunk}\033[0m")
        self._check_static()

    def select_for_task(self, task: Any) -> None:
        """Re-select using the task's own seed if it exposes one; else keep the
        configured selection. (RoboTwin currently never sets ``task.seed``.)"""
        task_seed = getattr(task, "seed", None)
        if task_seed is not None and int(task_seed) != (self.seed or -1):
            self.seed = int(task_seed)
            self.select(self.seed)

    # -- pin checks -----------------------------------------------------------
    def _check_static(self) -> None:
        """Non-sim pins: task spec, control, provenance (best-effort on current)."""
        if self.plan is None:
            return
        spec = self.plan.task_spec
        cur = self.current
        # task identity
        if cur.get("task_name") is not None:
            self.pins.check("task_config", (spec.task_name, spec.task_config),
                            (cur.get("task_name"), cur.get("task_config")))
        # control / action semantics (chunk)
        if cur.get("chunk") is not None:
            self.pins.check("control", int(spec.control.get("chunk", 0)), int(cur["chunk"]))
        # embodiment (only when the current embodiment is known)
        if cur.get("embodiment") is not None:
            self.pins.check("embodiment", spec.embodiment.get("embodiment"),
                            cur.get("embodiment"))
        # provenance: commit + config hash (skip when current is unavailable to
        # avoid false alarms in a non-git checkout)
        prov = spec.provenance
        if cur.get("robotwin_commit") is not None and prov.get("robotwin_commit") is not None:
            self.pins.check("provenance", prov.get("robotwin_commit"), cur.get("robotwin_commit"))
        if cur.get("bench_config_hash") is not None and prov.get("bench_config_hash") is not None:
            self.pins.check("provenance", prov.get("bench_config_hash"), cur.get("bench_config_hash"))

    def check_t0(self, task: Any) -> None:
        """Sim pin: the live scene's t0 fingerprint vs the plan's (the gray-zone
        arbiter — a 'free' knob that perturbs the seeded scene is caught here)."""
        if self.plan is None or self._t0_checked:
            return
        expected = self.plan.fingerprints.get("t0")
        if expected is not None:
            self.pins.check("fingerprint_t0", expected, rb.state_fingerprint(task).tolist())
        self._t0_checked = True

    def check_terminal(self, task: Any) -> None:
        """Optional end-of-episode pins: terminal fingerprint + success flag."""
        if self.plan is None:
            return
        expected_fp = self.plan.fingerprints.get("terminal")
        if expected_fp is not None:
            self.pins.check("fingerprint_terminal", expected_fp, rb.state_fingerprint(task).tolist())
        expected_success = self.plan.meta.get("success")
        if expected_success is not None:
            self.pins.check("success", bool(expected_success), bool(task.check_success()))

    # -- policy interface -----------------------------------------------------
    def set_language(self, instruction: Any) -> None:  # no-op (no inference)
        self.instruction = instruction

    def update_observation_window(self, img_arr: Any, state: Any) -> None:  # no-op
        self.observation_window = True

    def get_action(self) -> np.ndarray:
        """Return the next chunk of raw actions (``[<=chunk, action_dim]``).

        Advances the cursor. Returns an empty array once the plan is exhausted; the
        caller (``eval``) uses that to end the episode.
        """
        if self.actions.size == 0 or self.cursor >= len(self.actions):
            self._exhausted = True
            return np.zeros((0, 0), dtype=np.float32)
        nxt = self.actions[self.cursor:self.cursor + self.chunk]
        self.cursor += self.chunk
        return nxt

    def reset_obsrvationwindows(self) -> None:
        """Reset the cursor and re-select the plan for the configured seed."""
        self.observation_window = None
        self.select(self.seed)

    # collect_rollout_client / server proxy call reset_model; keep both names
    def reset_model(self) -> None:
        self.reset_obsrvationwindows()


# --------------------------------------------------------------------- interface fns
def get_model(usr_args: Dict[str, Any]) -> ReplayModel:
    """Build a :class:`ReplayModel` from the deploy config (yml + overrides)."""
    pin_policy = pins_mod.PinPolicy.from_dict(usr_args.get("pins"))
    task_config = usr_args.get("task_config")
    current = {
        "task_name": usr_args.get("task_name"),
        "task_config": task_config,
        "embodiment": usr_args.get("embodiment"),
        "chunk": usr_args.get("pi0_step", usr_args.get("chunk")),
        "robotwin_commit": _git_head(_CRW_ROOT),
        "bench_config_hash": _config_hash(task_config),
    }
    return ReplayModel(
        plan_dir=usr_args.get("plan_dir"),
        plan_path=usr_args.get("plan_path"),
        seed=usr_args.get("seed"),
        chunk=usr_args.get("chunk"),
        pin_policy=pin_policy,
        current=current,
    )


def eval(TASK_ENV, model, observation) -> None:
    """Replay one chunk of stored raw actions into the env (no inference).

    Mirrors the pi05 ``eval`` control flow but pulls actions from the plan. On the
    first call of an episode it runs the t0 fingerprint pin (local model only). When
    the plan is exhausted it runs the optional terminal pins and force-ends the
    episode (sets ``take_action_cnt = step_lim``) so the harness loop terminates
    cleanly without polluting the trajectory.
    """
    local = getattr(model, "_is_local_replay", False)  # False for the TCP proxy

    if local:
        # lazily honour a task-provided seed (no-op on stock RoboTwin) then t0 pin
        if not model._t0_checked:
            model.select_for_task(TASK_ENV)
            model.check_t0(TASK_ENV)

    actions = model.get_action()
    if actions is None or len(actions) == 0:
        if local:
            model.check_terminal(TASK_ENV)
        # end the episode: the eval / collect loops both gate on take_action_cnt
        if getattr(TASK_ENV, "step_lim", None) is not None:
            TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return

    for action in actions:
        TASK_ENV.take_action(action)


def reset_model(model) -> None:
    """Reset the cursor + re-select the plan (and re-run the static pins)."""
    model.reset_obsrvationwindows()
