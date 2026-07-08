# Data-Gen Branch Handoff (`data-gen`)

**From:** Mehrdad · **Date:** 2026-07-07
**Base:** `origin/main_rep_learn`

This branch adds a grounding **data-generation pipeline** on top of RoboPRO's
`customized_robotwin/` collector: per-object masks + depth + role annotations in every
episode, ready for large-scale collection.

**Division of labor going forward:**

- ✅ **KEEP AS-IS (§2):** the perception/data plumbing — raw-id masks, ID→name/role maps,
  PNG compression, depth-based 3D lift, viz tools. Verified end-to-end; please build on
  top rather than modify.
- 🔁 **YOURS TO BUILD (§3):** the labeling / collision-interpretation layer. We prototyped
  one (4-way outcome labels + filtering + planner-blindness for negative samples) and then
  **removed it from the branch** so you start from a clean slate — §3 summarizes what it
  was and where to find the full reference implementation in git history.
- One commented example config remains: `benchmark/bench_task_config/mmz_template.yml`.

Quick start for tools/collection commands: `mmz_tools/README.md`.

---

## 1. Change map (every file this branch touches)

| File | What changed |
|---|---|
| `customized_robotwin/envs/camera/camera.py` | segmentation returns **raw uint16 ids** (was palette-colorized RGB) |
| `customized_robotwin/envs/_base_task.py` | + `get_actor_id_map()`, + `get_role_names()` |
| `customized_robotwin/envs/utils/pkl2hdf5.py` | + `seg_encoding()` — PNG-compresses uint16 masks into HDF5 |
| `customized_robotwin/script/collect_data.py` | writes id-map/role sidecar into `scene_info.json`; organized per-episode output + end-of-run summary; per-episode crash guards (one bad episode no longer kills a run) |
| `mmz_tools/` (inspect_hdf5, viz_episode, time_run, README) | standalone inspection/viz/timing tools |
| `benchmark/bench_task_config/mmz_template.yml` | single commented example config |

Nothing outside these files is modified — the `benchmark/bench_envs/` scene classes and
all stock `bench_*` configs are **untouched stock RoboPRO**.

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

### 2.5 Collection robustness (`collect_data.py`)
Per-episode try/except (a CuRobo/mesh crash is logged, cleaned up, and the run
continues) + a no-frames guard (a plan that produces no executable motion yields no
HDF5 instead of crashing the pkl→HDF5 merge). Generic safety — worth keeping under any
labeling scheme. Also: per-episode banners + an end-of-run `COLLECTION SUMMARY`
(`mmz_tools/time_run.sh` parses those summary lines).

### 2.6 Tools (`mmz_tools/`)
`inspect_hdf5.py` (tree/shapes/attrs) · `viz_episode.py` (panel: RGB | depth | seg |
role-overlay | 2D-boxes; + labeled .ply clouds + top-down; role colors target=red,
dest=green, obstacle=orange, robot=blue) · `time_run.sh` (throughput → dataset
projection). Standalone; no engine imports.

---

## 3. 🔁 THE LABELING LAYER — removed; yours to design

**What we had (brief):** a prototype for positive/negative contrast data — a config knob
that made the CuRobo planner *blind* to clutter (its smooth path then naturally collides
→ negative samples on the same seeds as the positives), a **measured** per-episode
outcome (task-success × collision → 4 labels: success / success-with-accident /
crashed-and-failed / failed-no-accident), a `keep_labels` config filter that kept only
the wanted labels per run, a collision-free seed-acceptance mode for exact positive
counts, and the outcome stamped into each HDF5's attrs.

**It was removed from this branch** (the 4-way taxonomy and the hard ± split don't
generalize across teams). The full working implementation is preserved in git history at
commit **`e248b2d`**:

```bash
git diff e248b2d^..e248b2d -- customized_robotwin/script/collect_data.py benchmark/bench_envs/
```

**Recommendation from what worked:** make the new design a **config knob + filter
system** — the collector measures a per-episode outcome (possibly with more than 4
labels), and each team's config declares which labels to keep, so every team maps
outcomes to their own ±/curriculum without forking the collector. Two things proved
valuable regardless of taxonomy: measure labels from the live scene (never assume them
from the run type), and keep task-success vs collision as separate axes. Stamping the
outcome into HDF5 attrs made every file self-describing for dataloaders.

**Facts you'll need:** the collision signal lives in
`benchmark/bench_envs/_bench_base_task.py` (`get_collision_metrics()` → contacts,
per-object events, `is_collision`), gated by `enable_collision_metrics`. Thresholds in
code: **2 cm / 0.2 rad / 10 N**.
⚠️ Stock coupling: `enable_collision_metrics: true` **also makes the planner blind to
clutter** (`update_world(exclude_obstacles=True)` in the four scene base classes) — you
will likely want to decouple those two (our removed version did it with one extra config
key; see the reference commit). Bonus signal worth keeping in mind: for collision
episodes, the metrics blob names **which objects were actually hit** — free causal
"this obstacle mattered" labels.

---

## 4. Practical notes

- Routing: run from `customized_robotwin/` with `ROBOTWIN_BENCH_TASK=bench` +
  `source set_env.sh`; configs resolve by *name* from `benchmark/bench_task_config/`.
- Collection is **resumable**: existing `episodeN.hdf5` skipped, `seed.txt` continues.
  Corollary: after changing what gets collected, delete old episodes (or rename the
  config) or collection silently no-ops.
- Same scenes across two runs: stock `use_seed: true` + copying `seed.txt` from a
  finished run reproduces identical scenes/instructions (useful for matched-pair designs).
- Known role-semantics gaps (open design points): some kitchen tasks hold the target in
  task-specific attrs (viz falls back to `_get_target_object_names()`);
  `put_can_next_to_basket` has **no destination object** — its goal is a pose next to an
  anchor. Destination semantics for such tasks are still an open question.
- Throughput anchor (1× RTX 4080, d10 clutter, put_cup_on_coaster): ≈ **82 s/kept
  episode** including seed-search rejects; seed replay without search ≈ 37 s/episode.
  Episode ≈ 37 MB. `mmz_tools/time_run.sh` re-measures and projects.
