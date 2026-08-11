"""Single source of truth for clearance-route metric configuration."""

import os
from dataclasses import dataclass, field, fields, replace


def _env_value(raw, current):
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (list, tuple)):
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    return raw.strip()


@dataclass
class SeedMetricConfig:
    """Metric knobs shared by the standalone tool and rollout seed builder.

    The 1.23 m ceiling is the general clearance-volume limit: it retains
    headroom over the olive-oil occluder while avoiding the unused upper
    slices from the former 1.4 m standalone default.
    """

    xmin: float = -0.6
    xmax: float = 0.6
    ymin: float = -0.35
    ymax: float = 0.35
    res: float = 0.01
    zmin: float = 0.78
    zmax: float = 1.23
    zres: float = 0.03
    gate_tau: float = 0.35
    gate_tau_sweep: list[float] = field(
        default_factory=lambda: [0.5, 0.7, 1.0, 1.5, 2.0]
    )
    seed_snap: float = 0.10
    warm_seeds: int = 8
    ik_seeds: int = 30
    chunk: int = 256
    occ_shape: str = "mesh"
    obstacles: str = "all"
    free_only: bool = False

    @classmethod
    def from_args(cls, args):
        """Overlay explicitly supplied argparse values on the field defaults."""
        values = {}
        for field in fields(cls):
            value = getattr(args, field.name, None)
            if value is not None:
                values[field.name] = value
        return cls(**values)

    @classmethod
    def from_env(cls, base=None):
        """Overlay present ``SEED_<FIELD>`` variables on ``base`` or defaults."""
        config = base or cls()
        values = {}
        for field in fields(cls):
            key = f"SEED_{field.name.upper()}"
            if key in os.environ:
                values[field.name] = _env_value(os.environ[key], getattr(config, field.name))
        return replace(config, **values)
