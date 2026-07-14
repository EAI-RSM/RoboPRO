# Dataloader contract

Everything a training dataloader needs is **in the files** — the grounding masks and the
oriented 3D boxes are written at generation time, so the dataloader just **reads**. No
helper functions or extra modules are required.

## Files per episode

Run directory: `data/dataset/<task>/<config>/`, per episode `i`:

| File | Contents |
| --- | --- |
| `data/episode{i}.hdf5` | all per-frame signals (see below) |
| `scene_info.json` | keyed by `episode_{i}`: `actor_id_map` (mask id → name), `role_names`, `object_pose_ids` |
| `masking/episode{i}.json` | metadata only: stage timeline (`stages`, `frame_stage`), target/bin names + ids. **Not needed to build masks** — the masks are baked into the HDF5. |

## Inside the HDF5

Per camera, per frame (`observation/<cam>/…`):
- `rgb`, `depth`, `intrinsic_cv`, `extrinsic_cv`, `cam2world_gl`
- `actor_segmentation` — raw object-id map (uint16, PNG-encoded)
- **`grounding_mask`** — the training target: a uint8 label map, **`0`=background, `1`=target (red), `2`=bin (green)**, PNG-encoded like the segmentation. Padding/region logic is already applied. Decode exactly like `actor_segmentation`.

Per object, per frame:
- **`actor_bbox`** — the oriented 3D box: `obb_center` (T,N,3), `obb_half` (T,N,3), `obb_quat` (T,N,4, wxyz), `id` (T,N). (No AABB — derive one by bounding the box if you need it.)
- **`link_bbox`** — same fields for articulation parts (drawer/fridge/microwave doors + interiors).
- `object_poses` (T,N,7), `joint_action`, `endpose`, per-frame `contact`/`collision`.

## Reading it (plain numpy — no repo modules needed)

```python
import h5py, cv2, numpy as np

with h5py.File("data/dataset/put_cup_on_coaster/datagen_template/data/episode0.hdf5") as f:
    g = f["observation/countertop_camera"]

    # grounding mask for frame t: 0=bg, 1=target, 2=bin
    t = 100
    gm = cv2.imdecode(np.frombuffer(bytes(g["grounding_mask"][t]), np.uint8), cv2.IMREAD_UNCHANGED)
    target_mask, bin_mask = (gm == 1), (gm == 2)

    # oriented 3D boxes at frame t
    ab = f["actor_bbox"]
    ids   = ab["id"][t]                 # (N,)
    center= ab["obb_center"][t]         # (N,3) world-frame box center
    half  = ab["obb_half"][t]           # (N,3) half extents in the box frame
    quat  = ab["obb_quat"][t]           # (N,4) orientation (wxyz)
    # 8 corners of object j:  center[j] + (signs*half[j]) @ R(quat[j]).T
```

That's the whole contract: **read `grounding_mask` for the masks, read `obb_*` for the boxes.**

## Optional convenience (not required)

`visualization/masking_resolve.py` is a standalone module (numpy/h5py/cv2 only) with
reference helpers if you'd rather call them than read arrays — `build_masks(run_dir, ep,
frame)` and `build_obbs(actor_bbox, k, id_map)`. They produce exactly what's baked into
the HDF5, so use them only if you want to, e.g., re-derive masks with a different padding.

## Notes

- Point clouds are not stored — rebuild from `depth` + `intrinsic_cv`/`extrinsic_cv`.
- Cameras: `countertop_camera` + the two wrist cams are the training set; every camera is
  captured (with its own `grounding_mask`), so the choice is made at load time.
- The bin padding is fixed at generation time (config knob `table_obj_pad`, default 5 cm).
