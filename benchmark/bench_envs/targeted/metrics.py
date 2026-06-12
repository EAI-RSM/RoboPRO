"""Offline task-progress / reward metrics from logged per-frame sim state.

The simulation loop logs ONLY raw actor poses (group `targeted_state` in the
episode HDF5, written by runtime.py + hdf5_annot.write_frame_state). Every scalar
metric — task progress, clearances, reward/value signals — is computed HERE,
offline, so adding or changing a metric never requires re-simulating. Metric
choice is task-type dependent (failure_annotation.md).

Used by the visualizer and by any analysis pass. Pure: numpy + h5py only.
"""
import json

import numpy as np

# task_name -> family; unknown tasks default to pick_and_place.
TASK_FAMILY = {"put_mouse_on_pad": "pick_and_place"}


def task_family(task_name: str) -> str:
    return TASK_FAMILY.get(task_name, "pick_and_place")


def load_frame_state(hdf5_path):
    """Load the raw per-frame state written by hdf5_annot.write_frame_state, plus
    the per-frame end-effector poses from the standard `endpose` group. Returns
    {names, pos[T,A,3], quat[T,A,4], frame_idx[T], entities, ee} or None."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        if "targeted_state" not in f:
            return None
        g = f["targeted_state"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in g["actor_names"][:]]
        ent = json.loads(g.attrs["entities_json"]) if "entities_json" in g.attrs else None
        ee = None
        if "endpose" in f:
            ee = {"left": f["endpose/left_endpose"][:], "right": f["endpose/right_endpose"][:]}
        return {"names": names, "pos": g["actor_pos"][:], "quat": g["actor_quat"][:],
                "frame_idx": g["frame_idx"][:], "entities": ent, "ee": ee}


def _idx(state, name):
    return state["names"].index(name) if name and name in state["names"] else None


def object_to_destination_l2_cm(state, target, destination):
    """Pick-and-place progress: L2 distance object<->destination per frame (cm)."""
    ti, di = _idx(state, target), _idx(state, destination)
    if ti is None or di is None:
        return None
    return np.linalg.norm(state["pos"][:, ti, :] - state["pos"][:, di, :], axis=1) * 100.0


# Safety = 1 − proximity-of-an-end-effector-to-any-non-target-object scaled by EE
# speed, in [0,1] (1 = clear, dips toward 0 near objects / at speed). Beyond
# `cutoff_cm` the proximity term is 0. All params are task-dependent.
SAFETY_PARAMS = {
    "pick_and_place": {
        "cutoff_cm": 15.0,         # >= this EE->object distance => 0 proximity term
        "ee_tip_offset_cm": 16.0,  # drop the flange pose to approximate the gripper tip
        "v_ref_cm": 0.9,           # EE displacement/frame that saturates the speed factor
        "speed_floor": 0.65,       # fraction of the proximity term kept at zero speed
        "structural": ("ground", "table", "wall"),  # never count as hazards
    },
}


def compute_safety(record: dict, state: dict):
    """Per-frame safety in [0,1] (1 = clear): 1 − proximity of either end-effector
    to the nearest non-target object (target/destination/structural excluded),
    scaled by EE speed. Task-dependent (SAFETY_PARAMS). Returns {metric, unit,
    values, cutoff_cm} or None."""
    p = SAFETY_PARAMS.get(task_family(record.get("task_name")))
    if p is None or not state.get("ee"):
        return None
    ent = state.get("entities") or record.get("progress_entities") or {}
    exclude = set(p["structural"]) | {ent.get("target"), ent.get("destination")}
    haz_idx = [i for i, n in enumerate(state["names"]) if n not in exclude]
    if not haz_idx:
        return None
    haz = state["pos"][:, haz_idx, :]                         # [T, H, 3]
    T = haz.shape[0]
    cutoff = p["cutoff_cm"] / 100.0
    vref = p["v_ref_cm"] / 100.0
    floor = p["speed_floor"]
    tip = np.array([0.0, 0.0, -p["ee_tip_offset_cm"] / 100.0])  # flange -> approx tip
    risk = np.zeros(T)
    for arm in ("left", "right"):
        if state["ee"].get(arm) is None:
            continue
        ee = np.asarray(state["ee"][arm])[:, :3] + tip        # [T, 3]
        d = np.linalg.norm(haz - ee[:, None, :], axis=2)      # [T, H]
        nearest = np.nanmin(d, axis=1)
        nearest = np.where(np.isnan(nearest), cutoff * 10, nearest)
        prox = np.clip((cutoff - nearest) / cutoff, 0.0, 1.0)
        spd = np.zeros(T)
        spd[1:] = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        vfac = np.clip(spd / vref, 0.0, 1.0)
        risk = np.maximum(risk, prox * (floor + (1.0 - floor) * vfac))
    # Invert: 1 = fully safe (clear), dips toward 0 as the EE nears objects / moves fast.
    safety = [round(1.0 - min(1.0, float(x)), 3) for x in risk]
    return {"metric": "safety", "unit": "", "cutoff_cm": p["cutoff_cm"], "values": safety}


def compute_progress(record: dict, state: dict):
    """Task-type-dependent task-progress signal from raw state, normalized to
    0–100%: the start distance maps to 0%, reaching the destination (distance 0)
    maps to 100%. The raw distance (cm) is returned too for reference. Returns
    {metric, unit, family, values[0..100], distance_cm, reference_cm} or None."""
    fam = task_family(record.get("task_name"))
    ent = state.get("entities") or record.get("progress_entities") or {}
    if fam == "pick_and_place":
        dist = object_to_destination_l2_cm(state, ent.get("target"), ent.get("destination"))
        if dist is not None and len(dist):
            ref = float(dist[0]) if float(dist[0]) > 1e-6 else (float(np.max(dist)) or 1.0)
            pct = [round(max(0.0, min(100.0, (1.0 - float(v) / ref) * 100.0)), 2) for v in dist]
            return {"metric": "task progress", "unit": "%", "family": fam, "values": pct,
                    "distance_cm": [round(float(v), 3) for v in dist], "reference_cm": round(ref, 2)}
    return None
