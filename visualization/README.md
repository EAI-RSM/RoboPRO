# visualization/ — inspect & export collected episodes

Standalone tools (need only the `robopro` env: numpy/h5py/opencv); they import no
engine code and never modify the dataset. Everything they produce is **derived** from
the episode HDF5 + `scene_info.json` sidecar — generate it when you need it, delete it
freely.

`<run_dir>` below = one (task, config) output directory, e.g.
`data/dataset/put_cup_on_coaster/mmz_template`.

## Quick look at an episode

```bash
python visualization/inspect_hdf5.py <run_dir>/data/episode0.hdf5
    # HDF5 tree: datasets, shapes, dtypes, attrs

python visualization/viz_episode.py <run_dir> 0 [--cam head_camera|cam1,cam2|all] [--frames 6]
    # -> <run_dir>/viz/episode0/: panel.png (rows=frames, cols=RGB|depth|seg|
    #    role overlay|2D boxes), legend.json, role-colored .ply clouds, topdowns
```

Role colors everywhere: target=red, destination=green, obstacles=orange, robot=blue.

## Export derived data (point clouds / boxes / panel)

Everything except `bbox3d_exact` is computed from stored depth + camera matrices +
masks. `bbox3d_exact` reads the physics boxes recorded at collection time
(`data_type.actor_bbox: true`).

```bash
python visualization/export.py <run_dir> 0 \
    --what pcd,bbox2d,bbox3d,bbox3d_exact,panel \  # or 'all'; this is also the default
    --cam head_camera \                            # comma list or 'all'
    --frames first,mid,last                        # or 'all', 'every:5', '0,40,80'
```

Outputs under `<run_dir>/export/episode0/`:

| flag | files | content |
|---|---|---|
| `pcd` | `pcd_<cam>_f<k>.ply` | dense role-colored point cloud (+ exact-box wireframes when available), open in MeshLab. Honors `--frames`. |
| `bbox2d` | `bbox2d.json` | per frame, per object: pixel box + id + name + role |
| `bbox3d` | `bbox3d.json` | per frame, per object: world AABB of its **visible surface** (what the camera sees) |
| `bbox3d_exact` | `bbox3d_exact.json` | per frame, per object: **exact full-extent** world AABB + pose, from physx (incl. occluded parts). Needs `actor_bbox` in the HDF5. |
| `panel` | `panel[_<cam>].png` | quick-look grid — **always 6 rows** (first, last, 4 evenly spaced between, independent of `--frames`), cols = RGB \| depth \| seg \| role overlay \| 2D boxes |
| always | `meta.json` | id→name map, role→ids, units, `actor_bbox_available` flag |

Masks themselves live in the HDF5 (`actor_segmentation`) and the panel renders them —
so the exporter no longer dumps standalone mask/overlay PNGs. Point clouds are rebuilt
from depth on demand (and by the training dataloader), so no `.npz` is written either.

**`bbox3d` vs `bbox3d_exact`:** visible-surface boxes hug only the pixels a camera can
see (a half-occluded mug → box around the visible half); exact boxes are the physics
engine's true full-size box for every object, occluded or not. Exact needs
`data_type.actor_bbox: true` at collection — it can't be reconstructed afterward.

(Per-episode `.mp4`s are written automatically during collection, next to each run's
data — no tool needed to view runs.)

## Timing a collection run

`customized_robotwin/time_run.sh` (bash, lives next to `collect_data.sh`):

```bash
cd customized_robotwin
bash time_run.sh <task> <config> [gpu]      # wall time, sec/episode, dataset projection
```

## flag_timeline.py — contact/collision timelines

Per-episode contact/collision frame ranges with video timecodes, peak impulses,
per-window object pairs, and each hit object's t0→window-end pose movement:

```bash
python visualization/flag_timeline.py <run_dir> [ep] [--min-impulse 0.01]
```

Default shows ALL contacts (incl. zero-impulse resting); `--min-impulse` hides
light touches. Full label semantics: DATA_GEN.md §5.2.
