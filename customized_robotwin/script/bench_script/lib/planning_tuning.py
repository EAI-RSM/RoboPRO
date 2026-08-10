"""Environment-configurable planning and execution tuning."""

import os


SIDE_WAYPOINT_GAP = 0.24       # clearance from the box EDGE to the waypoint (gripper)


REACH_X_LIMIT = 0.5


# All of the target's contact points, not a subset. A smaller limit combined with the
# arm-side ranking in _rank_side_grasp_ids discarded half the bottle's 8 (yaw-variant)
# grasps, and the occluder ring is placed at a random rotation -- so the one gap in the
# ring could sit on exactly the side that was never tried. Scene-specific, so it is off
# for the baseline.
GRASP_CANDIDATE_LIMIT = 8


GRASP_LIFT_HEIGHT = 0.15


GRASP_VERIFY_MIN_RISE_FRACTION = 0.5


OBJECT_RETENTION_TOLERANCE = 0.03


OBJECT_RETENTION_ROTATION_TOLERANCE = 0.2


OBJECT_PLACEMENT_XY_TOLERANCE = 0.018


OBJECT_PLACEMENT_Z_TOLERANCE = 0.03


CONTACT_RELEASE_XY_TOLERANCE = 0.02


DESCENT_CONTACT_DRIFT_TOLERANCE = 0.15


DESCENT_CONTACT_HEIGHT_TOLERANCE = 0.06


DESCENT_APPROACH_TSTEP_FRACTIONS = [0.2, 0.4, 0.6, 0.8]


DESCENT_MAX_PATH_DEVIATION = 0.08


DESCENT_MAX_PATH_LENGTH_RATIO = 3.0


DESCENT_MAX_JOINT_TRAVEL_RATIO = 3.0


DESCENT_MAX_JOINT_RANGE = 1.0


DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT = 0.5


DESCENT_MAX_JOINT_PATH_LENGTH = 0.6


LANDING_XY_OFFSETS = [-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015]


LANDING_RELEASE_HEIGHTS = [0.01, 0.03, 0.05, 0.08]


LANDING_MAX_CANDIDATES_PER_HEIGHT = 12


LANDING_MIN_ACCEPTED_TO_STOP = 3


LANDING_SEARCH_TRIGGER_DISTANCE = float(os.environ.get("LANDING_SEARCH_TRIGGER_DISTANCE", "0.10"))


LANDING_SEARCH_MAX_Z_DISTANCE = float(os.environ.get("LANDING_SEARCH_MAX_Z_DISTANCE", "0.15"))


GRASP_VERIFY_MAX_CANDIDATES = 3


ATTACHED_TRAJECTORY_SLOWDOWN = max(
    1, int(os.environ.get("ATTACHED_TRAJECTORY_SLOWDOWN", "2")))


DESCENT_SLICE_SIZE = 0.04


SIDE_WAYPOINT_GAPS = (0.20, 0.24, 0.28)


SIDE_WAYPOINT_GAPS_FALLBACK = (0.16, 0.12, 0.08)


PLACEMENT_SEARCH_RETRIES = 3


SIDE_WAYPOINT_Z_LIFTS = (0.0, 0.06, 0.15)


PLACE_CLEARANCE_ZS = (1.05, 1.15, 1.25)


PLACE_CLEARANCE_ZS_FALLBACK = (0.95, 0.90, 0.85)


PLACEMENT_STRICT_ORIENTATION_STAGES = ("placement:center_over_pad",)


LOCAL_WAYPOINT_ATTEMPTS = int(os.environ.get(
    "LOCAL_WAYPOINT_ATTEMPTS", os.environ.get("POST_GRASP_ESCAPE_ATTEMPTS", "5")))


WAYPOINT_SHRINK_MIN_DISTANCE = float(os.environ.get("WAYPOINT_SHRINK_MIN_DISTANCE", "0.05"))


WAYPOINT_ORIENTATIONS = ("grasp_aligned",)


WAYPOINT_Y_OFFSETS = (0.0,)


STAGE_ORDER = ["waypoint", "pre_grasp", "grasp", "lift", "post_grasp_escape",
               "pre_beside_box", "beside_box", "pre_lift_above_box",
               "lift_above_box", "over_box_to_pad_y", "center_over_pad"]
