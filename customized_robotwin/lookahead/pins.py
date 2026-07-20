"""Configurable verification pins for plan replay.

When a replay policy loads a :class:`~lookahead.plan.Plan`, it should verify that
the scene it is about to replay into actually matches the scene the plan was
searched in — otherwise the raw-action playback is meaningless. A :class:`Pin`
controls how strictly one named property is checked:

* ``enforce`` (default) — raise :class:`PinViolation` on mismatch,
* ``warn``             — log a clear warning and continue,
* ``off``              — skip the check entirely.

This is THE knob the user asked for: ship a sensible default policy, but make it
trivial to relax any single pin (e.g. ``provenance: off`` to stop pinning the
harness commit hash while iterating). Numeric pins (fingerprints) compare with a
per-pin ``atol``.

Pure module: no jax, no sim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("lookahead.pins")

# Named pins and their default level/atol. fingerprints compare numerically.
PIN_NAMES = (
    "provenance",        # harness commit / bench_config_hash
    "task_config",       # task_name + task_config + domain_randomization
    "embodiment",        # robot embodiment
    "control",           # action_dim / chunk / control_freq
    "fingerprint_t0",    # numeric: scene state at t0 (the gray-zone arbiter)
    "fingerprint_terminal",  # numeric: scene state at the scored terminal
    "success",           # bool: did the replay reach the plan's success flag
)

_NUMERIC_PINS = frozenset({"fingerprint_t0", "fingerprint_terminal"})


class PinViolation(Exception):
    """Raised by an ``enforce`` pin when expected != actual."""


@dataclass
class Pin:
    """One verification pin: how strictly to check a single named property."""

    level: str = "enforce"          # "enforce" | "warn" | "off"
    atol: float = 1e-5              # tolerance for numeric pins

    def __post_init__(self) -> None:
        if self.level not in ("enforce", "warn", "off"):
            raise ValueError(f"Pin.level must be enforce|warn|off, got {self.level!r}")
        self.atol = float(self.atol)

    @classmethod
    def from_config(cls, cfg: Any, default_atol: float = 1e-5) -> "Pin":
        """Build a Pin from a bare level string (``"off"``) or a dict
        (``{level, atol}``). ``None`` -> default (enforce)."""
        if cfg is None:
            return cls(atol=default_atol)
        if isinstance(cfg, str):
            return cls(level=cfg, atol=default_atol)
        if isinstance(cfg, dict):
            return cls(level=cfg.get("level", "enforce"),
                       atol=float(cfg.get("atol", default_atol)))
        raise TypeError(f"pin config must be str|dict|None, got {type(cfg).__name__}")


def _matches(expected: Any, actual: Any, atol: float) -> bool:
    """True if ``actual`` equals ``expected`` (numeric arrays via ``atol``)."""
    exp_arr = _as_float_array(expected)
    act_arr = _as_float_array(actual)
    if exp_arr is not None and act_arr is not None:
        if exp_arr.shape != act_arr.shape:
            return False
        return bool(np.allclose(exp_arr, act_arr, atol=atol, rtol=0.0))
    return expected == actual


def _as_float_array(x: Any) -> Optional[np.ndarray]:
    """Return a float array view of ``x`` if it is numeric array-like, else None."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64) if x.dtype != object else None
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(
            isinstance(v, (int, float, np.integer, np.floating)) for v in x):
        return np.asarray(x, dtype=np.float64)
    return None


class PinPolicy:
    """A named set of :class:`Pin` s with a uniform :meth:`check` entry point."""

    def __init__(self, pins: Optional[Dict[str, Pin]] = None) -> None:
        # start from defaults, then overlay any provided pins
        self.pins: Dict[str, Pin] = {name: Pin(level=_default_level(name), atol=_default_atol(name))
                                     for name in PIN_NAMES}
        if pins:
            self.pins.update(pins)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PinPolicy":
        """Build a policy from a config dict (e.g. a yaml ``pins:`` block).

        Each key is a pin name; each value is a bare level string or a
        ``{level, atol}`` dict. Unspecified pins keep their defaults. A missing /
        empty dict yields the all-``enforce`` default policy.
        """
        pins: Dict[str, Pin] = {}
        for name, cfg in (d or {}).items():
            if name not in PIN_NAMES:
                raise KeyError(f"unknown pin '{name}'; choices: {PIN_NAMES}")
            pins[name] = Pin.from_config(cfg, default_atol=_default_atol(name))
        return cls(pins)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: {"level": p.level, "atol": p.atol} for name, p in self.pins.items()}

    def get(self, name: str) -> Pin:
        if name not in self.pins:
            raise KeyError(f"unknown pin '{name}'; choices: {tuple(self.pins)}")
        return self.pins[name]

    def check(self, name: str, expected: Any, actual: Any) -> bool:
        """Verify one property under its pin's level.

        Returns True if it matched (or the pin is ``off``/``warn`` on a mismatch).
        Raises :class:`PinViolation` only for an ``enforce`` mismatch.
        """
        pin = self.get(name)
        if pin.level == "off":
            return True
        ok = _matches(expected, actual, pin.atol)
        if ok:
            return True
        msg = (f"pin '{name}' mismatch: expected {_brief(expected)} != "
               f"actual {_brief(actual)}"
               + (f" (atol={pin.atol})" if name in _NUMERIC_PINS else ""))
        if pin.level == "enforce":
            raise PinViolation(msg)
        logger.warning("[lookahead.pins] %s", msg)
        return False


def _default_atol(name: str) -> float:
    """Default atol per pin (fingerprint_t0 gets the documented 1e-5)."""
    return 1e-5


def _default_level(name: str) -> str:
    """Default level per pin. The TERMINAL fingerprint drifts at physics-jitter
    level over long rollouts, so it defaults to ``warn`` (informational). The t0
    fingerprint — the seeded-scene arbiter — and every other pin default to
    ``enforce``."""
    return "warn" if name == "fingerprint_terminal" else "enforce"


def _brief(x: Any) -> str:
    """Compact repr for a pin value (summarise large numeric arrays)."""
    arr = _as_float_array(x)
    if arr is not None and arr.size > 6:
        return f"<array shape={arr.shape} norm={float(np.linalg.norm(arr)):.4f}>"
    return repr(x)


def default_policy() -> PinPolicy:
    """The shipped default: every pin ``enforce`` except ``fingerprint_terminal``,
    which defaults to ``warn`` (it drifts at jitter level over long rollouts — the
    t0 pin is the seeded-scene arbiter). Fingerprint atol 1e-5.

    Relax individual pins via :meth:`PinPolicy.from_dict`, e.g.
    ``PinPolicy.from_dict({"provenance": "off"})``.
    """
    return PinPolicy()
