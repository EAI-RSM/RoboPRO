# mmz_tools — sample inspection & visualization

Standalone helpers (need only the `robopro` env: numpy/h5py/opencv). They touch no engine code.

```bash
# from customized_robotwin/, after collecting with an mmz_* config:
python ../mmz_tools/inspect_hdf5.py data/mmz_samples/<task>/<config>/data/episode0.hdf5
python ../mmz_tools/viz_episode.py  data/mmz_samples/<task>/<config> 0 [--cam head_camera] [--frames 6]
```

Outputs land in `<run_dir>/viz/episode<N>/`:
`panel.png` (RGB | depth | actor-seg | role overlay per sampled frame), `legend.json`
(role→ids, id→name map, coverage report), `pcd_stored_f*.ply` (the stored 1024-pt cloud),
`pcd_depth_f*.ply` (dense cloud rebuilt from depth, role-colored), `topdown_f*.png`.

Role colors everywhere: **target = red, destination = green, obstacles = orange, other = gray**.

View `.ply` files with MeshLab (`sudo apt install meshlab`) or any point-cloud viewer;
the `topdown_*.png` files need no viewer.
