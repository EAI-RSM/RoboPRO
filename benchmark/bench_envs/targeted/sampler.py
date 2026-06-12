"""Stratified perturbation sampling — isolated RNG, class-by-construction directions.

Design refs: TARGETED_DATA_COLLECTION.md §3.2 (class-by-construction) and §3.3
(stratified magnitude bins, isolated per-episode RNG that never touches the
global np.random stream scene generation consumes).
"""
import numpy as np

LABEL_SCHEMA_VERSION = 1

MAGNITUDE_BINS_CM = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 8.0))
PERCEPTUAL_CLASSES = ("depth", "lateral", "mixed")

# Stable per-type RNG stream codes. Never hash() — it is salted per process.
PTYPE_CODES = {"shift_object": 1, "shift_target": 2, "shift_obstacle": 3, "hide_obstacle": 4}


def sample_shift_params(ptype: str, scene_seed: int, episode_index: int) -> dict:
    """Sample (class, magnitude, sign, mixed angle) for one perturbed episode.

    Deterministic given (ptype, scene_seed, episode_index). Class and magnitude
    bin are assigned by round-robin quota so small PoC batches cover the full
    class x bin grid; only the within-bin magnitude, the sign and the mixed
    angle are random draws.
    """
    rng = np.random.default_rng([PTYPE_CODES[ptype], int(scene_seed), int(episode_index)])
    target_class = PERCEPTUAL_CLASSES[episode_index % len(PERCEPTUAL_CLASSES)]
    bin_index = (episode_index // len(PERCEPTUAL_CLASSES)) % len(MAGNITUDE_BINS_CM)
    lo, hi = MAGNITUDE_BINS_CM[bin_index]
    return {
        "target_class": target_class,
        "magnitude_bin_index": int(bin_index),
        "magnitude_bin_cm": [lo, hi],
        "magnitude_cm": float(rng.uniform(lo, hi)),
        "sign": int(rng.choice([-1, 1])),
        # Deliberately off-axis angle for the `mixed` class (between 25 and 65
        # degrees off the depth axis, either side).
        "mixed_angle_deg": float(rng.uniform(25.0, 65.0)) * (-1 if rng.random() < 0.5 else 1),
    }


def construct_world_shift(params: dict, depth_axis: np.ndarray, lateral_axis: np.ndarray) -> np.ndarray:
    """Build the commanded world-frame shift (z = 0) from sampled params and the
    table-plane camera axes. Returns a (3,) vector in meters."""
    depth = np.asarray(depth_axis, dtype=np.float64)
    lateral = np.asarray(lateral_axis, dtype=np.float64)
    cls = params["target_class"]
    if cls == "depth":
        direction = depth.copy()
    elif cls == "lateral":
        direction = lateral.copy()
    elif cls == "mixed":
        a = np.deg2rad(params["mixed_angle_deg"])
        direction = np.cos(a) * depth + np.sin(a) * lateral
    else:
        raise ValueError(f"unknown perceptual class: {cls}")
    direction = direction / np.linalg.norm(direction)
    shift = params["sign"] * (params["magnitude_cm"] / 100.0) * direction
    return np.array([shift[0], shift[1], 0.0], dtype=np.float64)
