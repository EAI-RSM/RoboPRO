"""Shared benchmark scene identities, bounds, and geometry constants."""


TARGET_MODEL = "001_bottle"


TARGET_ID = 9


TARGET_XLIM = (-0.15, 0.15)


TARGET_YLIM = (0.0, 0.20)


TABLE_XLIM = (-0.6, 0.6)


TABLE_YLIM = (-0.35, 0.35)


PAD_XY = (0.0, -0.28)          # destination pad parked at the front, out of the occluder zone


# ---------------------------------------------------------------- occluder asset selection
# Which object the occluders are made of. `olive_oil` (default) is the current ring benchmark
# and reproduces the previous constants EXACTLY, so leaving OCCLUDER_ASSET unset changes
# nothing. `milk_box` restores the obstacle from the ORIGINAL pre-July single-box scene
# (038_milk-box id 2, the one the old baseline scored 0/16 against), so that scene can be
# rebuilt to test the direct solver on it.
#
# Only the ASSET and its footprint constants switch here. Ring geometry, APPROACH_MODE,
# PLACEMENT_MODE and every planner path are untouched -- the original scene is reproduced by
# running this asset with ONE occluder and no ring rotation (OCCLUDER_COUNTS=1,
# RANDOM_RING_ROTATION=0), which `occluder_ring_xy` places directly in front (-y).
#
# Footprint numbers are the originals recovered from b3c2e66: the milk box is a rectangular
# carton (~0.11 x 0.122) and genuinely bigger than the round olive-oil bottle (~0.08), so the
# two constants below MUST track the asset or _box_side_x and the pad-distance scene filter
# would size themselves against the wrong object.
_OCCLUDER_ASSETS = {
    #                  model            id  half_footprint  half_extent
    "olive_oil": ("029_olive-oil", 3, 0.04, 0.04),   # round ~0.08 x 0.08, yaw-invariant
    "milk_box":  ("038_milk-box",  2, 0.08, 0.06),   # carton ~0.11 x 0.122, half-diagonal incl. yaw
}

import os as _os  # noqa: E402  (module is a constants table; keep the import next to its use)

OCCLUDER_ASSET = _os.environ.get("OCCLUDER_ASSET", "olive_oil").strip().lower()
if OCCLUDER_ASSET not in _OCCLUDER_ASSETS:
    raise ValueError(
        f"OCCLUDER_ASSET={OCCLUDER_ASSET!r} is not one of {sorted(_OCCLUDER_ASSETS)}")

OCCLUDER_MODEL, OCCLUDER_ID, OCC_HALF_FOOTPRINT, _OCC_HALF = _OCCLUDER_ASSETS[OCCLUDER_ASSET]


OCCLUDER_COLLISION = f"assets/objects/{OCCLUDER_MODEL}/collision/base{OCCLUDER_ID}.glb"


OCCLUDER_QPOS = [0.66, 0.66, -0.25, -0.25]   # upright carton orientation; same for both assets


OCC_PAD_CLEARANCE = 0.05


_PAD_HALF = 0.06               # pad half-size (create_box half_size xy)


OCC_PAD_MIN_DIST = _PAD_HALF + _OCC_HALF + OCC_PAD_CLEARANCE
