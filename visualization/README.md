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

## Export derived data (point clouds / boxes / masks)

Nothing below needs a collection-time flag — it's all computed from stored
depth + camera matrices + masks:

```bash
python visualization/export.py <run_dir> 0 \
    --what pcd,bbox2d,bbox3d,masks,overlay \   # or 'all'; default pcd,bbox2d,bbox3d
    --cam head_camera \                        # comma list or 'all'
    --frames first,mid,last                    # or 'all', 'every:5', '0,40,80'
```

Outputs under `<run_dir>/export/episode0/`:

| flag | files | content |
|---|---|---|
| `pcd` | `pcd_<cam>_f<k>.ply` + `.npz` | dense labeled cloud; npz = `xyz` (float32, m, world), `rgb`, `seg_id` — training-ready |
| `bbox2d` | `bbox2d.json` | per frame, per object: pixel box + id + name + role |
| `bbox3d` | `bbox3d.json` | per frame, per object: world-frame AABB of its *visible* surface |
| `masks` | `masks_*.png` (+`_color`) | raw uint16 id masks + colorized preview |
| `overlay` | `overlay_*.png` | role-colored RGB |
| always | `meta.json` | id→name map, role→ids, units, settings |

**Exact (full-extent) 3D boxes** are the one thing that can't be derived afterwards —
enable `data_type.actor_bbox: true` in the collection config and they're recorded
into the HDF5 (`actor_bbox/{id,pose,aabb_min,aabb_max}`) by the physics engine.

(Per-episode `.mp4`s are written automatically during collection, next to each run's
data — no tool needed to view runs.)

## Timing a collection run

`customized_robotwin/time_run.sh` (bash, lives next to `collect_data.sh`):

```bash
cd customized_robotwin
bash time_run.sh <task> <config> [gpu]      # wall time, sec/episode, dataset projection
```
