"""Pure label-derivation functions — no sim imports, unit-testable.

Layer 1 (cause) is recorded by the runner; this module derives the layer-2
consistency check (camera-frame decomposition of the achieved shift) and the
layer-3 mechanical outcome. Design ref: TARGETED_DATA_COLLECTION.md §3.2.
"""
import numpy as np

LABEL_SCHEMA_VERSION = 1

# Object moved less than this from its (post-shift) start pose => the grasp
# never picked it up.
GRASP_MOVED_THRESHOLD_CM = 3.0
# Mirrors put_mouse_on_pad.check_success epsilon (per-axis 2 cm on the live pad).
PLACE_SUCCESS_EPS_CM = 2.0
# Achieved-shift dominance threshold for the post-hoc consistency check. This
# is NOT the label (labels are by construction) — see §3.2. Must exceed
# cos(25 deg) ~ 0.906 so constructed `mixed` shifts (25-65 deg off the depth
# axis) never read as dominant: 0.92 maps to <23 deg reading depth and >67 deg
# reading lateral.
DOMINANCE_FRACTION = 0.92

OUTCOMES = (
    "clean_success",
    "success_with_collision",
    "empty_grasp",
    "wrong_place_pose",
    "collision_with_object",
    "plan_aborted",
)


def camera_frame_decomposition(shift_world, depth_axis, lateral_axis) -> dict:
    """Decompose a world-frame shift onto the table-plane camera axes.

    Returns cm components plus the post-hoc dominance classification recorded
    as a consistency check against the constructed class.
    """
    s = np.asarray(shift_world, dtype=np.float64)
    depth_cm = float(np.dot(s[:2], np.asarray(depth_axis)[:2]) * 100.0)
    lateral_cm = float(np.dot(s[:2], np.asarray(lateral_axis)[:2]) * 100.0)
    vertical_cm = float(s[2] * 100.0)
    horizontal_norm = float(np.hypot(depth_cm, lateral_cm))
    if horizontal_norm < 1e-9:
        dominant = "none"
    elif abs(depth_cm) >= DOMINANCE_FRACTION * horizontal_norm:
        dominant = "depth"
    elif abs(lateral_cm) >= DOMINANCE_FRACTION * horizontal_norm:
        dominant = "lateral"
    else:
        dominant = "mixed"
    return {
        "depth_cm": depth_cm,
        "lateral_cm": lateral_cm,
        "vertical_cm": vertical_cm,
        "posthoc_dominant_axis": dominant,
    }


def derive_outcome(*, plan_success: bool, success: bool, is_collision: bool,
                   object_moved_cm: float | None, place_error_cm: float | None) -> tuple[str, str | None]:
    """Layer-3 mechanical outcome from sim signals. Success outcomes are
    first-class; object identity / planner visibility are separate attrs and
    never baked into the label (§3.2)."""
    if not plan_success:
        return "plan_aborted", "executor aborted: a motion plan failed"
    if success:
        if is_collision:
            return "success_with_collision", None
        return "clean_success", None
    if object_moved_cm is not None and object_moved_cm < GRASP_MOVED_THRESHOLD_CM:
        return "empty_grasp", (
            f"object moved only {object_moved_cm:.1f} cm from its start pose "
            f"(< {GRASP_MOVED_THRESHOLD_CM} cm threshold)")
    if is_collision:
        return "collision_with_object", "episode failed and a collision was recorded"
    detail = None
    if place_error_cm is not None:
        detail = f"object moved but ended {place_error_cm:.1f} cm (xy) from the live target"
    return "wrong_place_pose", detail


# ======================================================================
# Annotation (failure_annotation.md): WHAT (flag-set) · WHY (cause, known
# by construction) · quality. Pure; derived offline from a recorded
# episode dict (signals + contact log + the intervention we applied).
# ======================================================================

# WHAT vocabulary — co-occurring flags, NOT a precedence ladder.
WHAT_FLAGS = ("grasp_failure", "placement_failure", "collision", "planning_failure")

# WHY: the cause, known by construction from the intervention we applied.
WHY_BY_PTYPE = {
    "shift_object": ("object_mislocalized", "target"),
    "shift_target": ("destination_mislocalized", "destination"),
    "shift_obstacle": ("obstacle_mislocalized", "obstacle"),
    "hide_obstacle": ("object_undetected", "obstacle"),
}


def manifestations(*, plan_success: bool, success: bool, is_collision: bool,
                   object_moved_cm: float | None, place_error_cm: float | None) -> list:
    """WHAT as a flag-set: flags co-occur (a mislocalized grasp that also trips
    the planner is {grasp_failure, planning_failure}), so nothing is masked by a
    single precedence-ordered outcome."""
    flags = []
    if not success:
        if object_moved_cm is not None and object_moved_cm < GRASP_MOVED_THRESHOLD_CM:
            flags.append("grasp_failure")
        elif place_error_cm is not None and place_error_cm > PLACE_SUCCESS_EPS_CM:
            flags.append("placement_failure")
    if is_collision:
        flags.append("collision")
    if not plan_success:
        flags.append("planning_failure")
    return flags


def quality(*, success: bool, what: list) -> str:
    """good = success, no adverse manifestation (incl. fault absorbed);
    suboptimal = success but something happened (e.g. collision); bad = failed."""
    if not success:
        return "bad"
    return "suboptimal" if what else "good"


def annotate(record: dict) -> dict:
    """Top-level annotation from a recorded episode dict. Pure / offline:
    reads `signals`, `contact_log`, `perturbation_type`, etc."""
    sig = record.get("signals") or {}
    cm = sig.get("collision_metrics") or {}
    contact_frames = [c["frame_idx"] for c in record.get("contact_log", []) if c.get("contacts")]
    is_coll = bool(cm.get("is_collision")) or bool(contact_frames)
    success = bool(sig.get("success_flag"))
    what = manifestations(
        plan_success=bool(sig.get("plan_success")),
        success=success,
        is_collision=is_coll,
        object_moved_cm=sig.get("object_moved_cm"),
        place_error_cm=sig.get("place_error_cm"),
    )
    why_val, why_role = WHY_BY_PTYPE.get(record.get("perturbation_type"), ("none", None))
    shift_frame = record.get("shift_frame_idx")
    return {
        "what": what,                                   # flag-set (possibly empty)
        "what_frames": {"collision": contact_frames},   # per-timestep where available
        "why": why_val,
        "why_role": why_role,
        "why_axis": record.get("perceptual_failure_class"),
        "why_active_from_frame": int(shift_frame) if shift_frame is not None else 0,
        "quality": quality(success=success, what=what),
        "task_success": success,
    }
