# Data Generation Documentation — Paired Three-Source Collision Dataset

**Status:** spec, 2026-06-11.  Companion to `collision_avoidance_method.md` (the
consumer of this data: aux-head supervision §1.1, critic calibration §2, fine-tuning
buffer §5.3, Q̂_task targets §4.6).

---

## 1. What is collected

For **one task**, **10 seeds**, **three episodes per seed** — 30 episodes total — all
three episodes of a seed run on the **identical scene** (same `setup_demo(seed)`, same
object placements, verified by scene hash):

| # | Source label | What it is | Role in training |
|---|---|---|---|
| 1 | `planner_collision_aware` | Classical motion planner (CuRobo) with **clutter included in its collision world** — the collision-free expert | Positive examples: proof a feasible collision-free solution exists in this scene; clean clearance labels; corrected-label reference for Phase 1 |
| 2 | `pi05_base` | Current pi05 policy rollout (the existing base checkpoint: `pi05_aloha_full_base` / `roboreal_all_80tasks`, ckpt 20000) | The realistic distribution: where the policy actually goes, including its real contacts — this is the distribution Stage 2 must improve |
| 3 | `planner_collision_unaware` | Same planner, same task goals, but **clutter excluded from its collision world** (the existing "curobo planner skips clutter obstacles" mode) — plows through clutter on purpose | Contact-rich negatives with realistic task intent: dense supervision exactly in the margin shell and penetration regime, where the hinge has gradient and data is otherwise scarce |

Why three sources on the *same seed*: the triplet spans the clearance distribution
(safe / realistic / contact-rich) **in the same scene**, which gives paired
supervision no amount of single-source data provides — same observation geometry with
three different outcomes is ideal for the aux head (it must distinguish trajectories,
not scenes), for Q̂_task (success-to-go targets with both outcomes represented), and
for critic calibration (predicted-vs-realized clearance across the full range).

All episodes are kept **including failures** — a failed pi05 rollout is valid data
(hindsight-zero Q̂_task label, real contact distribution). The one exception:
a `planner_collision_aware` episode that itself makes forceful contact is a planning
artifact, not an expert — mark it `expert_valid: false` and resample the seed
(keep the episode on disk; never silently delete).

## 2. Recorded data — schema

```
collision_dataset/<task>/<env_type>/        # NOT rollout_data — own dataset root
├── curobo_collision_free/                  # source A: planner_collision_aware
│   ├── episode_seed<N>.hdf5                # per-step streams (below)
│   ├── episode_seed<N>.info.json           # written IMMEDIATELY after the episode:
│   │                                       #   seed/source/success/collision/frames,
│   │                                       #   collision_metrics, stream paths
│   ├── episode_seed<N>.meta.json           # written by the relabel pass
│   ├── metrics/seed<N>_contacts.jsonl      # per-step engine contacts (below)
│   ├── metrics/seed<N>_collisions.jsonl    # displacement-metric events (below)
│   └── videos/episode_seed<N>.mp4
├── pi05_rollout/                           # source B: pi05_base (same structure)
├── curobo_collision/                       # source C: planner_collision_unaware
├── scene/seed_<N>/                         # shared by the seed's three episodes
│   ├── scene.npz                           # samples (M,3) + normals + per-sample obj_id
│   ├── pointcloud.ply                      # same samples, downsampled, for viewing
│   ├── objects.json                        # per_scene_id → name, tags (target/
│   │                                       #   container/furniture/clutter/articulation),
│   │                                       #   pose, n_samples
│   └── scene_hash.txt                      # identity check across the 3 episodes
├── fk_basis.npz                            # once per dataset: serialized ArmFK +
│                                           #   sphere decomposition + validation error
├── index.jsonl                             # one line per saved episode
└── collect_summary.json                    # per-seed roll-up (legacy format)
```

### 2.1 `traj.hdf5` — per-step streams (every physics-control step t)

| Stream | Shape/type | Notes |
|---|---|---|
| `qpos` | (T, 14) | measured joint positions, both arms + grippers |
| `action` | (T, 14) | commanded action at t (planner waypoint or policy output) |
| `images/<cam>` | (T, H, W, 3) | the standard pi05 camera set (head + wrists); **required** — the aux head trains on RGB |
| `link_clearance/d` | (T, 14 links, 3) | per-link top-3 clearances, sphere-surface, clipped 0.3 m — §1.1 spec |
| `link_clearance/u` | (T, 14, 3, 3) | unit away-vectors, robot base frame, KD-tree exact |
| `link_clearance/obj` | (T, 14, 3) | per_scene_id of each of the 3 obstacles (distinct ids enforced) |
| `link_clearance/sphere_idx` | (T, 14, 3) int | index of the argmin sphere within the link — the exact application point for gradient use: ∂d/∂q = J_pᵀû needs the point Jacobian AT that sphere's center (link-origin approximation has lever-arm error); center reconstructible from qpos + FK + sphere YAML |
| `link_margin` | (T, 14) bool | d_top1 < margin (2.5 cm) — the calibration bit |
| `ee_pose` | (T, 2, 7) | both EE poses (convenience; reconstructible from qpos) |
| `object_poses/{ids,pose}` | (n,), (T, n, 7) | **per-step pose of every scene object** (p + wxyz quat, keyed by per_scene_id). Objects move mid-episode, uniquely per run — this track is what keeps labels correct after displacement (scene-versioning note below) |
| `articulation_qpos/art_<j>` | (T, dof_j) | articulation joint states per step (fridge door angle, ...) |
| `in_contact_window` | (T,) bool | within ±50 steps of a FORCE contact — the buffer prioritization bit, precomputed |

Labels are generated by the critic pipeline verbatim (`differentiable_proximity.py`:
FK + CuRobo spheres; distances against the scene samples via cKDTree — *not* the
voxel grid, so labels carry no voxel smoothing). **All per-step labels are evaluated
at that step's MEASURED `qpos` — the configuration the robot is actually in at that
instant.** They are never lookahead predictions and never derived from commanded
future actions; commanded-action clearances are a separate derived quantity (table
below) used only for the chunk-conditioned critic, not for aux-head supervision.
Per link: min over the link's spheres, top-3 over **distinct** `per_scene_id`; the
stored û is the away-vector of the link's argmin sphere. Exclusions (task target,
structural container, furniture) identical to the guidance exclusion rules — one
rulebook.

**Granularity note — links vs spheres.** "Link" means one of the 14 chain links, not
one of the ~130 collision spheres. The spheres are intermediate geometry: each link's
clearance is the min over its spheres. Sphere centers are **not stored** — they are a
deterministic function of `qpos` (FK link frames + fixed per-link sphere offsets from
the embodiment YAML), so they are exactly reconstructible offline from `qpos` +
`fk_basis.npz`. General rule of this schema: store generators (`qpos`, `action`,
scene), recompute derived quantities.

**Stored vs derived (recompute offline, never collect):**

| Derived quantity | Recomputed from |
|---|---|
| Sphere centers (130 × T) | `qpos` + `fk_basis.npz` + sphere YAML |
| Measured-pose clearances (the labels above) | `qpos` + `scene/` — the relabel pass |
| **Commanded-action clearances / chunk lookahead** | `action` + `scene/` (FK of the commanded chunk = what the policy *intended*; this is the regression target for the chunk-conditioned critic Q̂(s, a), and it differs from measured-pose clearance exactly where contact deflects execution — that divergence is itself a useful signal, and both sides come from stored streams) |
| SDF voxel grid | `scene/pointcloud.ply` (`sdf_grid.npz` is an optional cache) |

**Mid-episode scene motion (scene versioning).** `scene/seed_N/` stores only the
INITIAL state, shared by the triplet (identical by hash). The moment a run
displaces an object — uniquely per run, and precisely in the contact-rich
episodes — geometry diverges. The fix is generator-based, not new point clouds:
surface samples are rigid in the object frame, so each episode records the
per-step `object_poses` track (~50 KB), and the relabel pass splits the episode
into **scene versions** (new version when any object drifts > 3 cm or rotates
> 0.2 rad from the version's reference — the same rule as the guidance grid
rebuild), rigidly re-poses the moved objects' samples
(x_t = R_t R_0ᵀ (x_0 − p_0) + p_t), rebuilds the KD-tree per version, and labels
every step against its own version. `scene_versions` is reported per episode in
meta and the relabel summary. Known limitation: articulation link boxes
(negative ids) are NOT re-posed — joint states are recorded
(`articulation_qpos`) but their sampled boxes stay at the export configuration;
the structural container is excluded from labels by default anyway.

### 2.2 `contacts.jsonl` — the emphasized stream (one line per step WITH contact)

```json
{"t": 143, "pairs": [["fr_link6", "091_kettle#94", 0.486, [0.31, -0.24, 0.86], 5],
                     ["fr_link8", "091_kettle#94", 0.088, [0.30, -0.22, 0.84], 2]],
 "max_impulse": 0.486, "force": true,
 "pred_clear_at_t": -0.012, "worst_link": "fr_link6"}
```

Each pair carries `[link, object#id, impulse, contact_point_xyz, nearest_sphere_idx]`
— the engine's contact point and its attribution to the link's nearest sphere. This
is the primary record for **engine-gated corrections** (method doc §4.5): the gate is
the recorded contact itself; the push direction/application point come from the
attributed sphere ("perturb link3's sphere 7"), regardless of what the model distance
claimed at that step.

- `pairs`: every engine contact pair (link or held-target ↔ object#per_scene_id) with
  impulse norm. Gripper links included. Zero-impulse resting pairs included
  (`force: false`) — they are the boundary cases calibration needs.
- `pred_clear_at_t` / `worst_link`: the model clearance at that step, written inline so
  contact-vs-prediction calibration requires no join.
- Contact **windows** (start/end step of each contiguous contact run, per object, with
  peak impulse) are summarized in `meta.json` for O(1) sampling.

### 2.3 `collisions.jsonl` — displacement-metric events

One line per counted collision, current metric semantics (2026-06-10 final:
displacement > 1 cm or > 5.7° cumulative, **provenance-required** — only objects
touched by robot/held-target count; no object→object category):

```json
{"t": 144, "category": "target_to_static_object", "object": "025_chips-tub#98",
 "cumul_delta_m": 0.0025, "cumul_ang_deg": 5.75, "last_toucher": "target"}
```

(There is deliberately **no** chunk-lookahead stream in the collector: it is fully
derivable offline from `action` + `scene/` — see the stored-vs-derived table above.
Guidance is **OFF** for all collection; we record the policy, not the safety net.)

### 2.4 `meta.json` — episode header

```json
{
  "task": "pick_bottle_from_fridge", "seed": 300009,
  "source": "pi05_base",                       // the policy label, one of the 3
  "checkpoint": "roboreal_all_80tasks/20000",  // null for planner sources
  "planner_collision_world": "with_clutter",   // with_clutter | no_clutter | null
  "success": true, "episode_len": 215,
  "expert_valid": null,                        // planner_collision_aware only
  "contact_steps": 20, "force_steps": 12,
  "contact_windows": [{"object": "025_chips-tub#98", "t0": 138, "t1": 158,
                        "peak_impulse": 0.49}],
  "collision_count": 1,
  "min_clearance_m": -0.007,
  "code_rev": "<git sha>", "label_settings": {"margin": 0.025, "d_max": 0.3,
                                              "k": 3, "sphere_pad_gripper": 0.01}
}
```

## 3. Collection procedure

Per seed (loop over 10 seeds; identical order each time so partial runs resume):

```
1. setup_demo(seed) → settle → write scene/ (pointcloud, objects.json, hash).
2. Episode A — planner_collision_aware:
     CuRobo world = furniture + ALL clutter collision meshes (the entries already
     assembled in collision_list with is_obstacle=True); plan + execute the task;
     record all streams.  If force_steps > 0 → expert_valid=false (keep, flag).
3. Reset to the same seed (fresh setup_demo, verify scene_hash matches).
4. Episode B — pi05_base: standard policy server rollout (collect_rollout
     infrastructure), guidance OFF, expert-skip flag ON; record all streams.
5. Reset, verify hash.
6. Episode C — planner_collision_unaware: CuRobo world WITHOUT clutter (the
     current "skips clutter obstacles" behavior); same task goals; record.
7. Append seed summary line to <task>/index.jsonl (one line per episode:
     seed, source, success, contact_steps, force_steps, paths).
```

**Implementation status: IMPLEMENTED (2026-06-11).**  How to run:

```bash
# one task, 10 seeds → up to 30 HDF5 triplet episodes under
# collision_dataset/<task>/<env_type>/{curobo_collision_free,pi05_rollout,curobo_collision}/
# (the launcher runs the relabel pass automatically after each task):
TASKS="pick_bottle_from_fridge" TASK_CONFIG=bench_demo_kitchenl_d15 COLLECT_NUM=10 \
  bash policy/pi05/collect_rl_dataset.sh <train_config> <model_name> <ckpt> <gpu>

# relabel pass, re-runnable manually any time the label spec changes
# (per-link top-3 d/û/sphere_idx, contact attribution, windows,
# in_contact_window, per-episode meta — reads only stored generators):
python script/relabel_collision_dataset.py collision_dataset/<task>/<env_type> \
    --margin 0.025 --window 50
```

Where things live:
- Triplet loop (`collect_rollouts_paired`, `_run_curobo_leg`), scene export per
  seed, `index.jsonl`, source labels in HDF5 attrs (`source`, `seed`,
  `scene_dir`): `customized_robotwin/script/collect_rollout_proximity_client.py`
  (`COLLECT_MODE=paired`).
- Planner collision world: the collector legs call `update_world()` (clutter
  in — source A) / `update_world(exclude_obstacles=True)` (clutter out —
  source C) explicitly after setup; the bench task bases are untouched (their
  default — skip clutter in collision-metrics mode — is overridden per leg).
- Scene + FK serialization (`export_scene`, `serialize_fk_and_spheres`,
  `get_arm_dof_indices`): `customized_robotwin/script/differentiable_proximity.py`.
- Contact/collision jsonl streams (`start_metric_streams` /
  `stop_metric_streams`, written from `check_collisions` — zero-impulse pairs
  included, contact positions per point):
  `benchmark/bench_envs/_bench_base_task.py`.
- Offline relabeling (numpy FK from `fk_basis.npz`, validated against the torch
  implementation; KD-tree exact distances; per-link top-K distinct objects;
  nearest-sphere contact attribution; contact windows; `expert_valid`):
  `customized_robotwin/script/relabel_collision_dataset.py`.

Notes:
- Episode numbering is `seed_idx*3 + {0,1,2}` for sources A/B/C; HDF5
  `collector` attr is `curobo` / `pi05` / `curobo_unaware` (back-compat) and the
  canonical `source` attr carries the names above.
- **Strict same-seed triplets (default):** legs B & C run only on seeds where
  the collision-aware expert (leg A) succeeded — all three sources always share
  the identical seed/scene (verified by scene hash). Expert-blocked seeds are
  skipped entirely.
- **Opt-in `COLLECT_ON_EXPERT_FAIL=1`:** additionally collect pi05 +
  collision-unaware legs on expert-blocked seeds (still the same seed as the
  failed expert attempt). These are exactly the scenes where the unaware
  planner plows through clutter — the contact-rich data the contact logit and
  engine-gated corrections need, which strict gating anti-selects (observed:
  5/6 blocked seeds skipped, surviving seed contact-free). Flagged
  `expert_blocked: true`, not counted toward COLLECT_NUM; a guard stops after
  20× COLLECT_NUM seed attempts.
- `expert_valid` (force-free expert) is computed by the relabel pass, per §1.
- **Contact windows are force-gated** (`--force-thresh`, default 1e-3): PhysX
  reports zero-impulse proximity pairs on most steps near clutter, which would
  saturate `in_contact_window`; only real (forceful) contacts define contact
  regions, while resting pairs remain in the attributed stream as
  boundary/calibration data.
- Link count L is whatever the sphere YAML defines (8/arm for aloha: link1–6 +
  gripper links 7/8) — stored in `link_clearance/names`; do not hard-code 14.

### 3.1 Masking and re-including objects after collection

The point cloud saves **everything except `ground`/`wall`** — targets, containers,
table, furniture, clutter, articulation boxes — each sample tagged with its object's
id, each object carrying tags in `objects.json` (`target` / `container` /
`furniture` / `clutter` / `articulation`; ground/wall appear with `n_samples: 0`
and a `skipped` tag).  **Exclusion is a relabel-time decision, never a collection
filter:** masked objects' samples are dropped from the KD-tree before labeling
(invisible to proximity) but stay on disk untouched.  Changing what counts as a
collision obstacle later = re-running one script — no recollection:

```bash
# default: clutter + non-container articulations are obstacles
python script/relabel_collision_dataset.py collision_dataset/<task>/<env>

# also treat the table as an obstacle, but mask one specific clutter object
python script/relabel_collision_dataset.py collision_dataset/<task>/<env> \
    --include-names table --exclude-names 028_roll-paper

# make even the target an obstacle (e.g., pre-grasp-phase labels)
python script/relabel_collision_dataset.py collision_dataset/<task>/<env> \
    --exclude-tags furniture,skipped
```

- `--exclude-tags`   tag-level masking (default `target,container,furniture,skipped`)
- `--exclude-names`  mask additional objects by name
- `--include-names`  force-include an object despite excluded tags

The settings used are stored in every episode's `label_settings` (HDF5 attr and
meta.json), so datasets labeled under different exclusion rules can't be confused.
Known limitation: `ground` and `wall` have no samples at all (unbounded planes) —
re-including them would require an export change (workspace-bounded sampling); for
tabletop tasks the exported table surface is the relevant ground proxy.

Labeling can run **online** (during collection — labels are cheap: one KD-tree per
scene version, ~ms per step) or **offline** as a relabel pass over `qpos` + `scene/`
(everything is reconstructible; this is also how label-spec changes are applied later
without recollecting). Store the label settings used in `meta.json` either way.

## 4. Emphasis on contact timesteps — what that concretely means

1. **Nothing is subsampled away near contacts.** All streams are full-rate everywhere;
   "emphasis" is *indexing*, not extra fidelity: `in_contact_window` per step,
   `contact_windows` summaries in meta, and `index.jsonl` roll-ups make
   contact-region sampling O(1) for the §5.3 buffer (weights α, δ key off exactly
   these fields).
2. **Inline calibration fields** (`pred_clear_at_t` in every contact line) so the
   contact-vs-clearance calibration curve (method doc §6, open item 3) is a single
   scan over `contacts.jsonl` files.
3. **Zero-impulse contacts are recorded**, not filtered — the boundary between
   "resting touch" and "forceful" (impulse 1e-3) is a label, not a collection filter.
4. Source C exists *for* this emphasis: it manufactures dense in-shell and
   penetration samples with realistic kinematics, which neither the expert (always
   clear) nor the base policy (sparse contacts) provides in volume.

## 5. Expected volume and cost (one task, 10 seeds, 30 episodes)

| Item | Estimate |
|---|---|
| Steps | ~600 max/ep → ~10–18k steps total |
| Images | dominant cost; at 3 cams × 224² RGB ≈ raw ~25 GB, stored as the usual compressed HDF5/video → ~1–3 GB |
| Labels | (T,14,3) d + (T,14,3,3) û ≈ 2 MB/ep — negligible |
| Point clouds | ~1–4 MB/seed |
| Wall clock | planner eps ~1–2 min, policy eps ~2–4 min → ~1.5–2.5 h serial for 30 |

## 6. Validation checklist (run after collection, before training)

- [ ] Scene hash identical across the 3 episodes of every seed.
- [ ] FK basis validation error < 5 mm (hard-fail threshold, same as guidance).
- [ ] Label/engine consistency: at every `force: true` contact step, predicted top-1
      clearance of the contacting link ≤ +1.5 cm (the known under-read bound);
      violations listed, not just counted — each one is a label bug or a sphere
      coverage gap.
- [ ] Settle-creep audit: zero `collisions.jsonl` entries on never-touched objects
      (provenance metric guarantees this by construction — verify anyway).
- [ ] Source sanity: `planner_collision_aware` median force_steps = 0;
      `planner_collision_unaware` contact_steps ≫ `pi05_base` ≫ expert.
- [ ] Per-source success: expert ≈ 10/10 (resampled seeds), pi05_base = whatever it
      is (that's data), unaware planner success irrelevant (contacts are the point).
- [ ] Triplet completeness: 10 seeds × 3 sources present in `index.jsonl`.

## 7. Scaling note

This spec is per-task. The directory layout, index format, and validation checklist
are task-agnostic — scaling to more tasks is adding `<task>/` roots and re-running;
nothing in the schema changes. Seed count per task scales the same way (the §5.3
fine-tuning loop will eventually want 100–200 episodes per refresh — same pipeline,
larger seed list, sources B-only for refreshes since the planner episodes are
distribution-independent and collected once).
