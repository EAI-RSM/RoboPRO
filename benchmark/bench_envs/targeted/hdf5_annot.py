"""Append the targeted label record to a finished episode HDF5 (component 0.4).

The full record is stored as one JSON attribute (lossless), plus flat scalar
attributes for the fields consumers filter on. No upstream pkl2hdf5 changes.
"""
import json
from pathlib import Path

FLAT_KEYS = (
    "label_schema_version", "task_name", "task_config", "scene_seed",
    "executor_type", "pair_id", "pair_role", "perturbation_type", "trigger",
    "scene_kind", "perceptual_failure_class", "status", "outcome",
    "perturbed_actor", "shift_frame_idx", "instruction",
)


def annotate(hdf5_path, record: dict) -> bool:
    import h5py

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        return False
    with h5py.File(hdf5_path, "a") as f:
        g = f.require_group("targeted_labels")
        g.attrs["record_json"] = json.dumps(record)
        for k in FLAT_KEYS:
            v = record.get(k)
            if v is not None:
                g.attrs[k] = v
    return True


def write_frame_state(hdf5_path, frame_state: list, entities: dict | None = None) -> bool:
    """Write the raw per-frame actor poses logged during the rollout into the
    episode HDF5 (group `targeted_state`), aligned 1:1 with the saved frames.

    Stores ONLY raw state (compact float32 + gzip); scalar metrics are computed
    offline from it (targeted/metrics.py). `entities` maps task roles
    (target/destination/…) to actor names so offline code knows which is which.
    """
    import h5py
    import numpy as np

    hdf5_path = Path(hdf5_path)
    if not frame_state or not hdf5_path.exists():
        return False
    names = sorted({n for fr in frame_state for n in fr["poses"]})
    col = {n: i for i, n in enumerate(names)}
    T, A = len(frame_state), len(names)
    pos = np.full((T, A, 3), np.nan, dtype=np.float32)
    quat = np.full((T, A, 4), np.nan, dtype=np.float32)
    fidx = np.zeros(T, dtype=np.int32)
    for t, fr in enumerate(frame_state):
        fidx[t] = fr.get("frame_idx", t)
        for n, v in fr["poses"].items():
            a = col[n]
            pos[t, a] = v[:3]
            quat[t, a] = v[3:7]
    with h5py.File(hdf5_path, "a") as f:
        if "targeted_state" in f:
            del f["targeted_state"]
        g = f.require_group("targeted_state")
        g.create_dataset("frame_idx", data=fidx)
        g.create_dataset("actor_names", data=np.array(names, dtype=h5py.string_dtype()))
        g.create_dataset("actor_pos", data=pos, compression="gzip")
        g.create_dataset("actor_quat", data=quat, compression="gzip")
        if entities:
            g.attrs["entities_json"] = json.dumps(entities)
    return True
