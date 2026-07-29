"""Shared benchmark scene identities, bounds, and geometry constants."""


TARGET_MODEL = "001_bottle"


TARGET_ID = 9


TARGET_XLIM = (-0.15, 0.15)


TARGET_YLIM = (0.0, 0.20)


TABLE_XLIM = (-0.6, 0.6)


TABLE_YLIM = (-0.35, 0.35)


PAD_XY = (0.0, -0.28)          # destination pad parked at the front, out of the occluder zone


OCCLUDER_MODEL = "029_olive-oil"


OCCLUDER_ID = 3


OCCLUDER_COLLISION = f"assets/objects/{OCCLUDER_MODEL}/collision/base{OCCLUDER_ID}.glb"


OCCLUDER_QPOS = [0.66, 0.66, -0.25, -0.25]   # same upright orientation the stock milk-box used


OCC_HALF_FOOTPRINT = 0.04      # olive-oil base half-diagonal (~0.08 x 0.08 round, yaw-invariant)


OCC_PAD_CLEARANCE = 0.05


_PAD_HALF = 0.06               # pad half-size (create_box half_size xy)


_OCC_HALF = 0.04               # olive-oil base half-extent (~0.08 x 0.08 round footprint)


OCC_PAD_MIN_DIST = _PAD_HALF + _OCC_HALF + OCC_PAD_CLEARANCE
