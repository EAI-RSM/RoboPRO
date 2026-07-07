# Data-Gen Branch Handoff (`data-gen`)

**From:** Mehrdad · **Date:** 2026-07-07
**Base:** `origin/main_rep_learn`

This branch adds a grounding **data-generation pipeline** on top of RoboPRO's
`customized_robotwin/` collector: per-object masks + depth + role annotations in every
episode, plus a **collision/outcome labeling layer** used to generate positive
(collision-free) and negative (deliberate-collision) demonstrations.

**Division of labor going forward:**

- ✅ **KEEP AS-IS (§2):** the perception/data plumbing — raw-id masks, ID→name/role maps,
  PNG compression, depth-based 3D lift, viz tools. Verified end-to-end; please build on
  top rather than modify.
- 🔁 **YOURS TO REDESIGN (§3):** the labeling layer — how collision is interpreted, the
  4-way label taxonomy, and the ±/filtering scheme. §3 documents exactly what we built
  and how each piece can be reverted or remapped so you can implement your own logic.
- The old ±-specific config files were **removed**; one commented example remains:
  `benchmark/bench_task_config/mmz_template.yml`.

Quick start for tools/collection commands: `mmz_tools/README.md`.

---

## 1. Change map (every file we touched)

| File | What changed | Layer |
|---|---|---|
| `customized_robotwin/envs/camera/camera.py` | segmentation returns **raw uint16 ids** (was palette-colorized RGB) | keep |
| `customized_robotwin/envs/_base_task.py` | + `get_actor_id_map()`, + `get_role_names()` | keep |
| `customized_robotwin/envs/utils/pkl2hdf5.py` | + `seg_encoding()` — PNG-compresses uint16 masks into HDF5 | keep |
| `customized_robotwin/script/collect_data.py` | sidecar writes (id map + roles) **[keep]**; outcome measurement, `keep_labels` filter, `seed_require_collision_free`, `replan_on_collect`, robustness guards, HDF5 outcome attrs **[label layer]** | both |
| `benchmark/bench_envs/office/_office_base_task.py` | + `planner_exclude_obstacles` decoupling | label |
| `benchmark/bench_envs/study/_study_base_task.py` | same | label |
| `benchmark/bench_envs/kitchenl/_kitchen_base_large.py` | same | label |
| `benchmark/bench_envs/kitchens/_kitchens_base_task.py` | same | label |
| `mmz_tools/` (inspect_hdf5, viz_episode, labels, time_run, README) | standalone inspection/viz/audit/timing tools | keep (labels.py tracks taxonomy) |
| `benchmark/bench_task_config/mmz_template.yml` | single commented example config | reference |

Everything is **additive and default-preserving**: any knob absent from a config ⇒ stock
RoboPRO behavior. Stock `bench_*` configs are unaffected.

---

## 2. ✅ KEEP AS-IS — the perception / data plumbing

These are verified end-to-end (masks↔ids↔names↔roles cross-checked visually and
programmatically across office/study/kitchen tasks; depth→3D lift metrically validated —
e.g. a reconstructed cup measures 4.3×4.4×6.9 cm at the true table height).

### 2.1 Raw-ID segmentation (`camera.py`)
Stock RoboPRO saved segmentation as **palette-colorized RGB images** — object identity
was irreversibly destroyed (you can't reliably map colors back to objects). Now
`_get_segmentation` returns the raw SAPIEN id planes as **uint16**:
mesh level = `seg_labels[..., 0]`, actor level = `seg_labels[..., 1]`.
Masks become usable training supervision. Configs enable it via
`data_type.actor_segmentation: true` (mesh level exists but stays off — sub-part ids
have no name map).

### 2.2 ID → name → role resolution (`_base_task.py` + sidecar)
- `get_actor_id_map()` → `{per_scene_id: name}` for every actor **including articulation
  links** (furniture decomposed into named links; unnamed robot articulations prefixed
  `robot/<link>`; ~70 entries typical).
- `get_role_names()` → the task's designated `target`/`destination` names **and their
  exact instance ids** (`target_id`/`destination_id` via `per_scene_id`). The ids
  disambiguate the target from same-model clutter (two `021_cup`s in one scene: id 7 =
  target, id 8 = clutter). Name keys retained as fallback for older data.
- Both are written into each episode's `scene_info.json` by `collect_data.py`.
- Obstacle convention (working definition): every known object that isn't
  target/destination/robot/scene-shell counts as an obstacle; spawned-clutter vs
  environment-object identity is preserved in the data so the split can be redefined
  later without re-collecting.

### 2.3 PNG-compressed masks (`pkl2hdf5.py`)
Raw uint16 masks at 7 cams × full trajectory = **370–450 MB/episode**. `seg_encoding()`
PNG-compresses each mask frame (lossless) before HDF5 write → **~37 MB/episode (~11×)**.
Decode on read with any PNG decoder (`cv2.imdecode` → uint16).

### 2.4 Cameras & 3D lift policy
- All cameras stay **D435 (320×240)** = identical to the benchmark's eval cameras — no
  train/test distribution mismatch. Extra viewpoints are inspection-only.
- `data_type.pointcloud` stays **off**: the stock cloud is head-cam-only, FPS-downsampled
  to 1024 points. Dense labeled clouds are instead **rebuilt from depth + stored
  intrinsics/extrinsics** (that's also how the method itself lifts masks to 3D).
  `viz_episode.py` shows the reference implementation, incl. 2D boxes (from masks) and
  3D visible-surface AABBs (from masked depth points).

### 2.5 Tools (`mmz_tools/`)
`inspect_hdf5.py` (tree/shapes/attrs) · `viz_episode.py` (panel: RGB | depth | seg |
role-overlay | 2D-boxes; + labeled .ply clouds + top-down; role colors target=red,
dest=green, obstacle=orange, robot=blue) · `labels.py` (outcome audit) ·
`time_run.sh` (throughput → dataset projection). Standalone; no engine imports.

---

## 3. 🔁 THE LABELING LAYER — what we built, and how to replace it

Our team's 4-way labeling isn't general enough for everyone — this section is the map
for changing it. Two design principles worth keeping regardless of taxonomy:
**(a) labels are MEASURED, never assumed** (from the live scene after each episode, not
from the run type); **(b) every knob is config-gated & absent ⇒ legacy** (so reverting
is often just "don't set the key").

### 3.1 Collision signal (what "collision" currently means)
`benchmark/bench_envs/_bench_base_task.py` accumulates collision metrics during rollout
(`get_collision_metrics()` → contacts, per-object events, `is_collision` flag), gated by
`enable_collision_metrics: true`. **Thresholds in code: 2 cm / 0.2 rad / 10 N** (note:
the paper states 1 cm / 0.1 rad — if your logic depends on thresholds, decide which is
canonical). This is the raw signal your interpretation layer sits on; we only *consumed*
it, we didn't change how it's computed.

### 3.2 The 4-way outcome (in `collect_data.py`)
After each episode, while the scene is alive, we compute:

```
success   = TASK_ENV.check_success()                      # task completion (collision-BLIND by design)
collision = get_collision_metrics()["is_collision"]       # any collision event
label     = success × collision  →  1 success | 2 success_with_accident
                                    3 crashed_and_failed | 4 failed_no_accident
```

The full block (label, label_code, success, collision, plan_success,
`planner_blind_to_obstacles`, `has_trajectory`, full collision-metrics blob) is written to
`scene_info.json` per episode, and **stamped into each kept HDF5 as root attrs**
(`label`, `label_code`, `success`, `collision`, `planner_blind`, `seed`, `task`,
`config`, `collision_metrics_json`) so every file is self-describing for a dataloader.
**To change the taxonomy:** replace this outcome block — everything downstream
(filtering §3.3, attrs, `labels.py`) keys off `label`, so it's one choke point.
Keeping success and collision as *separate decoupled axes* is recommended — collapsing
them is what loses information.

### 3.3 Filtering: `keep_labels` (config)
`keep_labels: [<label>, ...]` — after the outcome is measured, episodes whose label isn't
listed are **deleted** (HDF5/video removed; `scene_info.json` keeps their outcome + a
`kept: false` flag as the audit trail). Absent ⇒ legacy keep-success behavior.
Our old scheme: positives kept `[success]`, negatives kept
`[success_with_accident, crashed_and_failed]`. **Your mapping can be anything** — e.g.
keep everything and route by label in the dataloader instead of filtering at collection.

### 3.4 Planner blindness: `planner_exclude_obstacles` (the ± mechanism)
Stock RoboPRO **coupled** planner obstacle-awareness to `enable_collision_metrics`
(metrics on ⇒ `update_world(exclude_obstacles=True)` ⇒ CuRobo blind to clutter) —
identical block in all 4 scene base classes. We decoupled it:

```python
exclude_obs = self.planner_exclude_obstacles   # new config key, default None
if exclude_obs is None:
    exclude_obs = self.enable_collision_metrics   # legacy coupling preserved
```

- `false` ⇒ planner sees & avoids clutter *while metrics record* (obstacle-aware positives)
- `true`  ⇒ planner plans smooth-but-blind ⇒ natural collisions (negatives)
- absent ⇒ exact stock behavior.

**Revert = delete the key from configs** (code can stay; it's inert when unset).
The 4 files: `office/_office_base_task.py`, `study/_study_base_task.py`,
`kitchenl/_kitchen_base_large.py`, `kitchens/_kitchens_base_task.py` — one init line +
one condition each.

### 3.5 Seed machinery (exact counts & matched pairs)
- **Stock trick:** seed search loops `while suc_num < episode_num`, accepting seeds where
  `plan_success and check_success()` — that's how RoboPRO's 16K lands on exact counts.
- **`seed_require_collision_free: true`** (ours): acceptance additionally requires **zero
  collision** → banks exactly `episode_num` *pure* successes. Costlier at high clutter
  density (more rejects). If your positives tolerate contact, just don't set it.
- **Matched pairs:** run A writes `seed.txt`; copy it into run B's dir and set
  `use_seed: true` ⇒ identical scenes/instructions, differing only in planner settings.
- **`replan_on_collect: true`**: re-plans at collection instead of replaying the
  seed-search trajectory — **required** for blind runs (the cached trajectories are the
  *avoiding* paths; replaying them would never collide).

### 3.6 Robustness guards (keep these regardless of taxonomy)
Collision-heavy collection exposed two fatal crash modes: (1) a blind plan yields **no
frames** → the pkl→HDF5 merge exploded on the missing cache; (2) CuRobo
`attach_object`→trimesh crashes on some meshes mid-rollout. Fixes in `collect_data.py`:
per-episode try/except (logs, records `label:"error"`, cleans up, **continues the run**)
+ a `has_frames` guard (no-frame episodes are labeled but produce no HDF5). One bad
episode no longer kills a batch. Also: per-episode banners + end-of-run
`COLLECTION SUMMARY` (label counts, kept, filtered, errors).

### 3.7 Bonus signal you get for free
For each negative, the collision metrics blob names **which objects the blind path
actually hit** — free causal "this obstacle mattered" labels, much stronger than
"everything near the trajectory." Whatever the new taxonomy, consider preserving this.

---

## 4. Practical notes

- Routing: run from `customized_robotwin/` with `ROBOTWIN_BENCH_TASK=bench` +
  `source set_env.sh`; configs resolve by *name* from `benchmark/bench_task_config/`.
- Collection is **resumable**: existing `episodeN.hdf5` skipped, `seed.txt` continues.
  Corollary: after changing what gets collected, delete old episodes (or rename the
  config) or collection silently no-ops.
- Known role-semantics gaps (open design points): some kitchen tasks hold the target in
  task-specific attrs (viz falls back to `_get_target_object_names()`);
  `put_can_next_to_basket` has **no destination object** — its goal is a pose next to an
  anchor. Destination semantics for such tasks are still an open question.
- Throughput anchor (1× RTX 4080, d10, put_cup_on_coaster): aware+pure-S ≈ **82 s/kept
  episode** (seed-search rejects included); blind seed-replay ≈ **37 s/attempt** (~40%
  collided). Episode ≈ 37 MB. `mmz_tools/time_run.sh` re-measures and projects.
