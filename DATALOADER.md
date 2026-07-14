# Dataloader contract

What a training dataloader needs to consume the generated grounding data. **No
visualization tools are required** — everything below is data files plus one
self-contained engine module (`visualization/masking_resolve.py`, which depends
only on `numpy` / `h5py` / `cv2` / `json`).

## Three files per episode

Each run directory is `data/dataset/<task>/<config>/` and holds, per episode `i`:

| File | Contents |
| --- | --- |
| `data/episode{i}.hdf5` | all per-frame signals: `observation/<cam>/{rgb,depth,actor_segmentation,intrinsic_cv,extrinsic_cv,cam2world_gl}`, `actor_bbox/{id,pose,aabb_min,aabb_max,local_min,local_max}`, `link_bbox/…`, `object_poses`, `joint_action`, `endpose`, per-frame `contact`/`collision` |
| `scene_info.json` | keyed by `episode_{i}`: `actor_id_map` (mask id → name), `role_names`, `object_pose_ids` (column order of `object_poses`) |
| `masking/episode{i}.json` | the grounding labels: `stages` (each with `target_id` + `bin`) and `frame_stage` (maps every frame → its stage). Written at collection time. |

Supporting (optional): `instructions/episode{i}.json` (language), `scene/episode{i}/`
(sampled scene geometry), `video/`, `_traj_data/` (bit-exact replay).

## Building the training targets

The grounding masks use exactly **two classes per stage**: `target` (red) and
`bin` (green). For object bins that's a plain `seg == id`; for region bins
(onto-table / next-to) the mask is *computed* from geometry (with a clearance
padding around other objects), so import the engine rather than reimplementing it:

```python
import sys; sys.path.insert(0, "visualization")
from masking_resolve import build_masks, build_obbs, load_masking, load_sidecar
import h5py, numpy as np

run, ep, frame = "data/dataset/put_cup_on_coaster/datagen_template", 0, 100

# per-frame TARGET + BIN boolean masks (H,W) — the two supervision channels
m = build_masks(run, ep, frame, cam="countertop_camera")
target_mask, bin_mask, stage = m["target"], m["bin"], m["stage"]

# per-object oriented 3D boxes at a frame (grasp-relevant). Each entry has
# 'obb' = {center, half_size, quat, corners} (world frame) + the world AABB.
with h5py.File(f"{run}/data/episode{ep}.hdf5") as f:
    actor_bbox = {k: np.asarray(f["actor_bbox"][k]) for k in
                  ("id","pose","aabb_min","aabb_max","local_min","local_max")}
_, id_map, _, _ = load_sidecar(run, ep)
obbs = build_obbs(actor_bbox, frame, id_map)     # same call works on link_bbox
```

Notes:
- `build_masks` rebuilds region bins with the same object padding the review
  panels use (constant `TABLE_OBJ_PAD`, also recorded per-bin as `obj_pad`), so
  training masks match the visualization 1:1.
- `obb` is `None` for a body with no local extents (older data) — fall back to the
  AABB (`aabb_min`/`aabb_max`) in that case.
- Point clouds are **not** stored; rebuild them from `depth` + `intrinsic_cv` /
  `extrinsic_cv` on demand (see `backproject_full` in the same module).
- Cameras: `countertop_camera` + the two wrist cams are the training set; every
  camera is captured, so the choice is made at load time.
