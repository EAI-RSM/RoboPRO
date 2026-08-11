"""Camera selection for benchmark evaluation videos."""

LEGACY_CAMERA_ORDER = ("demo_camera", "countertop_camera", "head_camera")


def select_eval_video_camera(observation, requested_camera=None):
    if requested_camera is not None:
        if requested_camera not in observation:
            raise KeyError(
                f"requested eval video camera {requested_camera!r} is unavailable"
            )
        return requested_camera
    return next(
        (camera for camera in LEGACY_CAMERA_ORDER if camera in observation),
        None,
    )
