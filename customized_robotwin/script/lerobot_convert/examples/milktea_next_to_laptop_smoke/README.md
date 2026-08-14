# put_milktea_next_to_laptop — scene-preserving LeRobot v2.1

Converted from RoboTwin HDF5 at `milktea_next_to_laptop_38scene_400roll_jax30000` **1:1 at 30fps**
(one frame = one raw row = one policy command).
Single task: **put the milk tea next to the laptop**. 14-dim dual-arm joint+gripper
state; `action` is the next-step state. Cameras: countertop, left, right.

- **2 episodes** across **1 scenes** (1 success /
  1 fail).
- A *scene* is a `tier/seedN` folder (clean=1 scenes per tier). Every scene
  mixes success and fail draws; the grouping is preserved via the
  `scene_index / scene_id / tier / seed / draw` columns (in every parquet and in
  `meta/episodes.jsonl`), and `meta/scenes.jsonl` maps each scene to its episode
  indices and success/fail split.
- Unsuccessful episodes: kept full length (no truncation).

## Layout
- `data/chunk-*/episode_*.parquet`
- `videos/chunk-*/observation.images.{countertop,left,right}/episode_*.mp4`
- `meta/` — info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, modality.json, **scenes.jsonl**
- `manifests/` — the original `manifest*.json` copied verbatim

## Notes
- `success` is read from the HDF5 root attr (authoritative). The copied
  `manifest*.json` join by their `hdf5` path field; their own tier/seed labels
  are an inconsistent clean/high/mid labelling and are NOT used for scene ids.
- Per-frame `collision`/`contact` (+ impulses/pairs) come from the simulator.
- No `norm_stats.json` is emitted — normalization stats are model-specific and
  out of scope for this conversion.
