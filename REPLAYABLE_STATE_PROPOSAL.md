# Proposal: Replayable State Capture for RoboPRO

**Status:** Draft / RFC
**Author:** (you)
**Date:** 2026-06-25
**Scope:** Data-collection format for `customized_robotwin` demos and the RoboPRO benchmark

---

## 1. Thesis (one sentence)

Stop recording a fixed, presumed set of semantics at collection time; instead record a
**minimal, replayable state trace** of the simulation, and treat every semantic the
benchmark will ever need — new camera views, depth, segmentation, contact/collision
labels, success criteria, re-worded language, keypoints, relative-pose features — as a
**deterministic offline projection** of that trace.

> **Capture is not interpretation.** Today we conflate them. This proposal separates them.

---

## 2. Problem: the recollection tax

Every time the benchmark needs a semantic that wasn't anticipated at collection time, the
only recourse today is to **re-run the simulator and re-collect**. That is expensive,
non-reproducible (re-collection re-rolls RNG unless carefully pinned), and it scales with
the number of semantics × tasks × episodes.

This is structural, not incidental. The current pipeline writes a **pre-decided** view of
each episode:

- Per frame, [`get_obs()`](customized_robotwin/envs/_base_task.py#L449) captures rendered
  RGB (+ optional depth/segmentation/pointcloud) from a **fixed camera set**, plus robot
  proprioception (`joint_action`, `endpose`). It is serialized via
  [`pkl2hdf5.py`](customized_robotwin/envs/utils/pkl2hdf5.py) into `episode{N}.hdf5`
  (JPEG-encoded frames) + an MP4.
- The modalities are toggled **once**, at collection, by the `data_type:` flags
  ([`demo_clean.yml:22-31`](customized_robotwin/task_config/demo_clean.yml)): `rgb`,
  `depth`, `pointcloud`, `mesh_segmentation`, `actor_segmentation`, `endpose`, `qpos`,
  `third_view`. Anything left off is **gone**.
- Episode metadata ([`collect_data.py:259-274`](customized_robotwin/script/collect_data.py))
  records clutter types, texture IDs, and a task-specific `info` blob — but **no per-frame
  ground-truth object poses, no articulation state, no actions, no contacts, no physics
  properties**.

So the data is **presumptuous**: it assumes you already know which cameras, which
modalities, and which task semantics matter. The moment that assumption changes — a new
viewpoint, a segmentation channel, a contact-aware reward, a relabeled instruction — the
information needed to derive it was never written, and you recollect.

What we discard is precisely the cheap part. What we keep (pixels) is the expensive part.

---

## 3. Core idea: record sufficient statistics, project semantics offline

A SAPIEN episode is a deterministic function of:

```
(asset set, scene init, seed, task config)  →  trajectory of {actor poses, articulation qpos/qvel, contacts}
```

If we log the **trajectory of state** (the right-hand side) together with the **scene
manifest** (the left-hand side), we can reconstruct the simulation offline well enough to
**re-derive any state- or render-defined semantic**, without re-running data collection.

Concretely, two tiers of semantics, each with a clean derivation path:

### Tier A — render-derived semantics (the workhorse)

SAPIEN separates physics from rendering. Given a state snapshot we can `set_pose` /
`set_qpos` every entity, place a camera **anywhere**, and render **on demand without
stepping physics**:

- arbitrary camera poses & intrinsics (views never collected),
- RGB, depth, surface normals, point clouds,
- mesh-level and actor-level **segmentation** (SAPIEN emits per-pixel `mesh_id`/`actor_id`
  for free at render time — see [`camera.py:378-410`](customized_robotwin/envs/camera/camera.py)).

These are **exact** (rendering is deterministic given poses), not approximate, because we
never touch the physics solver.

### Tier B — physics-derived semantics (event logs)

Contacts, impulses, and forces are *step quantities* that poses alone cannot reconstruct.
[`scene.get_contacts()`](customized_robotwin/envs/_base_task.py#L1072) is only meaningful
during simulation. So these are **logged as sparse events at collection time** (cheap),
rather than recomputed by re-simulating. This is also the robust choice given the
determinism caveat in §7.

### Precedent: this already works in-repo

`robo_negative` is a working existence proof of "log state, derive semantics offline":

- It logs a per-frame pose trace for **all** actors into the `targeted_state` HDF5 group —
  `actor_pos[T,A,3]`, `actor_quat[T,A,4]`, `actor_names`, `frame_idx`, plus an
  `entities_json` role map ([`robo_negative/__init__.py:582-589, 644-676`](robo_negative/src/robo_negative/__init__.py)).
- It logs sparse contacts to `contact_log`.
- Its labels are **pure offline functions** over the logged signals —
  `derive_outcome()`, `annotate()`, `compute_progress()`, `compute_safety()`
  ([`robo_negative/__init__.py:171-355`](robo_negative/src/robo_negative/__init__.py)) —
  versioned and unit-tested.

This proposal **generalizes that side-channel into first-class, complete capture for all
collection** (clean demos, perturbation sweeps, negative data), and adds the scene manifest
needed for true offline re-rendering.

---

## 4. What "full replayable state" actually is

The sufficient statistics, split into a once-per-episode **manifest** and a per-frame
**state trace**, plus sparse **event logs**.

### 4.1 Episode manifest (written once)

Everything needed to reconstruct an empty, correctly-populated scene before replaying motion:

| Field | Source | Notes |
|---|---|---|
| `seed`, `task_name`, full `task_config` | [`_base_task.py:59-61`](customized_robotwin/envs/_base_task.py#L59) | deterministic scene recreation |
| Embodiment config (URDF/SRDF paths, `robot_pose`, joint stiffness/damping, gripper mimic) | [`robot.py:33-124`](customized_robotwin/envs/robot/robot.py) | which robot, where, with what drive params |
| Asset manifest: every object `{modelname, model_id, mesh path, scale}` **+ content hash** | [`create_actor.py:550-600`](customized_robotwin/envs/utils/create_actor.py) | meshes referenced by path today; pin by hash |
| Scene init: table/wall poses, `table_z_bias`, wall/table **texture IDs**, ambient + directional + point **lights** | [`_base_task.py:233-317`](customized_robotwin/envs/_base_task.py#L233) | lighting/textures drive vision OOD |
| Clutter manifest (types, indices, placements) | [`_base_task.py:343-377`](customized_robotwin/envs/_base_task.py#L343) | already partly in `scene_info.json` |
| Camera rig (intrinsics, mount links) | [`camera.py:296-322`](customized_robotwin/envs/camera/camera.py) | so offline cameras match collected ones |
| `sim` params: timestep `1/250`, gravity, solver substeps, SAPIEN version | [`_base_task.py:215-224`](customized_robotwin/envs/_base_task.py#L215) | replay/version pinning |

### 4.2 Per-frame state trace (the new core)

Logged at `save_freq`, **1:1 aligned to video frames** (as `targeted_state` already is):

- **All actor poses** — `pos[T,A,3]`, `quat[T,A,4]` via `actor.get_pose()`.
- **All articulation state** — robot `qpos[T,Dq]`, `qvel[T,Dq]`; **plus articulated scene
  objects** (drawers, doors, microwaves) `qpos/qvel`. *(Today even the negative-data logger
  only stores root poses; articulated-object joint state must be added — see §8.)*
- **Gripper** — normalized vals + joint drive targets ([`robot.py:651-690`](customized_robotwin/envs/robot/robot.py)).
- **Camera link poses** `[T,C,7]` for wrist-mounted cameras (head cam is static).

### 4.3 Event logs (sparse)

- **Contacts** — `{frame, body0, body1, impulse, position}` on contact events
  (`robo_negative.contact_log` is exactly this).
- **Actions / control** — the commanded action vector per control step (currently only
  *implied* by joint deltas; log it explicitly so behavior-cloning targets are exact).
- **Planner / task events** — grasp-plan committed, replan, stage transitions (optional;
  useful for phase labels and for the targeted-desync triggers).

---

## 5. Storage: the tradeoff inverts

The headline finding: **the state trace is ~1–2% the size of today's pixel HDF5.** Full
kinematic state is cheap; pixels are what's expensive.

Rough per-frame budget (fp32, uncompressed): ~60 actors × 7 floats ≈ 1.7 KB + robot
qpos/qvel ≈ 0.2 KB ≈ **~2 KB/frame**. A ~400-frame episode ≈ **<1 MB** raw, **~150–300 KB**
compressed. Today's `episode{N}.hdf5` is **~20–50 MB** (JPEG frames × multiple cameras).

So the tradeoff you raised flips:

- **Adding** the state trace to current collection is **storage-negligible** (~+1–2%) and
  buys full replayability of every Tier-A/B semantic.
- The bottleneck only appears if you **pre-render many modalities** (depth + seg + extra
  views × cameras × frames) — which is exactly what offline derivation lets you **stop**
  doing. State-trace-plus-thin-pixels can be **smaller** than today while strictly more
  flexible.

This yields a **retention spectrum**, selectable per use case (not baked into the format):

| Policy | Keeps | Size vs today | Re-render needed? |
|---|---|---|---|
| **State-only** | manifest + trace + events | ~2% | yes, to view pixels |
| **State + keyframes** | + sparse RGB keyframes / 1 view | ~10–30% | for dense/other views |
| **State + full RGB** | + today's RGB | ~100–102% | no (future-proof, redundant) |

### Optimizations (your "compression + smart event-based logging")

1. **Static-actor sparsity (biggest win).** Clutter and furniture don't move. Log a static
   actor's pose **once** in the manifest; per-frame, store only the **moving set** (robot,
   grasped object, anything whose pose changed > ε). Most frames touch a handful of bodies.
2. **Temporal delta encoding.** Store pose deltas vs previous frame; near-zero for resting
   bodies → compresses to almost nothing.
3. **Quantization.** Positions to mm/fp16, quaternions to fp16 — within sim noise.
4. **Block compression.** zstd/gzip on the float arrays (`targeted_state` already gzips).
5. **Event-triggered contact logging.** Only emit contacts on change (already capped/sparse
   in `robo_negative`).
6. **Asset dedup by hash.** Don't copy meshes per episode; reference by content hash and
   pin once per dataset.
7. **Keep H.264 for any retained RGB.**

---

## 6. Proposed on-disk schema

Extend the existing HDF5 episode file (generalize `targeted_state` → `sim_state`); keep it
**backward compatible** — old readers ignore the new group.

```
episode{N}.hdf5
├── observation/ …            # unchanged (optional now — can be thinned/dropped)
├── joint_action/ …           # unchanged
├── endpose/ …                # unchanged
└── sim_state/                          # NEW — the replayable trace
    ├── @state_schema_version = 1
    ├── @manifest_json                  # §4.1 manifest (assets w/ hashes, scene init, sim params, seed)
    ├── actors/
    │   ├── names         [A] str
    │   ├── asset_ref     [A] str        # modelname/model_id → manifest
    │   ├── is_static     [A] bool       # static → pose only in manifest
    │   └── roles_json                   # {"target": "...", "destination": "...", ...}
    ├── frame_idx         [T] int32      # aligned to video
    ├── actor_pos         [T, A, 3] f32  # gzip + delta (moving set only)
    ├── actor_quat        [T, A, 4] f32
    ├── robot/
    │   ├── qpos          [T, Dq] f32
    │   ├── qvel          [T, Dq] f32
    │   └── gripper       [T, G]  f32     # vals + drive targets
    ├── articulated/                      # scene articulations (drawers/doors)
    │   ├── names         [K] str
    │   └── qpos          [T, K, *]
    ├── camera_pose       [T, C, 7] f32   # wrist cams
    └── events/
        ├── contacts                      # {frame, body0, body1, impulse, pos}
        ├── actions       [Tc, Da]        # explicit commanded actions
        └── markers                       # planner/stage events
```

The `@manifest_json` mirrors the structure `robo_negative` already stamps into
`targeted_labels/record_json`, so the two converge on one provenance story.

---

## 7. Determinism contract (important)

- **Render-from-state is deterministic and portable.** Loading a state snapshot and
  rendering is reproducible on any machine — this is why Tier-A semantics are exact and
  why the state trace is the source of truth.
- **Physics re-simulation is *not* guaranteed bit-identical** across hardware/driver/
  library versions (PhysX solver, contact ordering), and ray tracing has minor SPP-level
  variance ([`_base_task.py:215-218`](customized_robotwin/envs/_base_task.py#L215)). There
  is **no native `scene.pack()/unpack()`** in use here — state is composed from per-entity
  getters.

**Contract:**
1. The **logged state trace is canonical.** Never recompute a label by re-simulating when
   it could have been logged live (contacts/forces → event logs).
2. **Pin** asset content hashes + SAPIEN/renderer versions in the manifest; replay validates
   against them and refuses on drift.
3. Anything **render-derived** is regenerable forever; anything **physics-derived** must be
   captured at collection time.

---

## 8. Offline semantic layer (the payoff)

A library of **pure, versioned "projectors"**: `(manifest + state trace + events) → semantic`,
exactly mirroring `robo_negative.annotate()`. Each is unit-testable and reproducible.
Candidate projectors:

- **Re-render**: new camera poses/intrinsics → RGB/depth/normals/pointcloud.
- **Segmentation**: mesh/actor masks from the renderer.
- **Collision/contact labels**: from `events/contacts` + body roles.
- **Success / reward criteria**: recompute *any* success rule from object poses post-hoc
  (today success is hard-coded at collect time in `check_success()`).
- **Language re-annotation**: regenerate instructions from object roles + spatial relations
  (poses give "left of", "behind", "on top of" for free).
- **Relative-pose / keypoint / grasp-affordance features** for policy inputs.
- **Progress / safety curves** (`compute_progress`, `compute_safety` already do this).

New semantic needed in 6 months → **write a projector, run it over the dataset.** No
simulator, no recollection.

---

## 9. Implementation plan (phased, non-breaking)

- **Phase 0 — Recorder.** Promote `robo_negative`'s frame-state logger into a reusable
  `StateRecorder` in `Bench_base_task`, gated by a new `data_type.sim_state: true` flag
  (mirrors existing toggles). Add robot `qpos/qvel`, articulated-object qpos, camera poses,
  explicit actions. Writes the `sim_state` group. **Default off → zero disruption.**
- **Phase 1 — Manifest.** Capture the §4.1 manifest with asset content hashes + sim params.
- **Phase 2 — Replay harness + round-trip validation.** Offline: manifest → rebuild scene →
  `set_pose/set_qpos` per frame → render. **Validate sufficiency** by re-rendering collected
  frames and comparing to stored RGB (PSNR) and stored segmentation (IoU). This is the
  proof the trace is complete.
- **Phase 3 — Projector library.** Ship the §8 projectors as a small package next to
  `robo_negative`.
- **Phase 4 — Storage policy.** Turn on static-actor sparsity + delta + zstd; then choose a
  per-dataset pixel-retention policy (§5) — optionally drop redundant pre-rendered modalities.

---

## 10. Risks & open questions

- **Re-render fidelity** vs stored pixels — settle with the Phase-2 round-trip metric; if a
  gap exists, retain thin keyframes as anchors.
- **Articulated scene objects** — must log their `qpos`, not just root pose (gap in the
  current `targeted_state` logger).
- **Deformables / fluids / soft bodies** — state may not be fully pose-serializable; flag
  per-task whether the trace is sufficient or pixels must be retained.
- **Planner/CuRobo internal state** — *not* needed for replay (we log resulting poses +
  actions); only relevant if reproducing planner decisions, which the manifest+seed already
  determine.
- **Re-render compute at scale** — deriving dense modalities for a large dataset is a GPU
  batch job; budget it, or derive lazily on demand.
- **Asset drift** — content hashes + version pins guard this; replay must fail loudly.

---

## 11. Success criteria

1. **Round-trip:** episode → state trace → offline re-render reproduces the original
   frames within tolerance (PSNR / segmentation IoU).
2. **Novel semantic without recollection:** produce a camera view **and** a segmentation
   channel that were never collected, purely offline from the trace — and a re-derived
   success label that matches the original.
3. **Storage:** state trace ≤ a few % of current per-episode footprint; full pipeline
   storage-neutral-or-better under a chosen retention policy.
```
