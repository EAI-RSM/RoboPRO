# Proximity Tracking Feature Plan

## Goal

Record the closest surface distance from configurable robot parts to any object in the scene,
per step in HDF5, and summarized per episode in `scene_info.json`.

---

## Versions

### Version 1 (this implementation)
- **Source**: given robot part (e.g. end-effector, body link) — represented as a **point**
- **Target**: every scene object except the robot's own body and currently held objects
- **Output**: one minimum distance float per part per step

### Version 2 (future)
- **Source**: mesh of the object currently held in hand
- **Target**: every scene object except the robot's own body and the held object itself
- **Output**: one minimum distance float per held object per step
- Foundation already laid by `_held_actors` dict (see §Held-object tracking)

---

## Config (YAML)

```yaml
proximity_tracking:
  enabled: true
  robot_parts: ["left_ee", "right_ee"]
  aabb_threshold: 1.0      # metres; objects beyond this use AABB distance directly
  # Valid part names:
  #   "left_ee" / "right_ee"   — computed EE point via robot.get_left_ee_pose()
  #   "left_tcp" / "right_tcp" — gripper center TCP
  #   "left_<linkname>"        — any URDF link by name, e.g. "left_link5"
  #   "right_<linkname>"       — same for right arm
```

Follows the same opt-in pattern as `enable_collision_metrics`.

---

## HDF5 Output Structure

Anything added to `pkl_dic` in `get_obs()` auto-flows into HDF5 via the existing PKL→HDF5
pipeline. New data:

```
episode0.hdf5
├── joint_action/
├── endpose/
├── observation/
└── proximity/
    ├── left_ee/
    │   └── min_dist     # float32, shape [N_steps]
    └── right_ee/
        └── min_dist     # float32, shape [N_steps]
```

`scene_info.json` gains a `proximity_metrics` key per episode:
```json
{
  "episode_0": {
    "proximity_metrics": {
      "left_ee":  {"min_dist": 0.032},
      "right_ee": {"min_dist": 0.018}
    }
  }
}
```

---

## Object Set ("any object in the scene")

At each step, the target set is:

```
scene.get_all_actors() + all links of scene.get_all_articulations()
  MINUS robot.left_entity  (excluded by object identity, not name)
  MINUS robot.right_entity (same object as left_entity when dual_arm_embodied=True)
  MINUS any actor currently held (see §Held-object tracking)
```

Articulations contribute their **links** (not the articulation root itself), so each link's
AABB is queried independently and unioned to cover the full articulation extent.

---

## Held-Object Tracking

`_bench_base_task.py` already has `attach_object(actor, file_path, arms_tag)` and
`detach_object(arms_tag)` called by all tasks. Extend them to maintain:

```python
self._held_actors = {"left": None, "right": None}  # actor name or None per arm
```

```python
def attach_object(self, actor, file_path, arms_tag):
    ...existing code...
    if hasattr(self, '_held_actors'):
        self._held_actors[arms_tag] = actor.get_name()

def detach_object(self, arms_tag):
    ...existing code...
    if hasattr(self, '_held_actors'):
        self._held_actors[arms_tag] = None
```

Supports:
- Sequential holding (one arm picks up, places, picks up another)
- Simultaneous dual-arm holding (left holds object A, right holds object B)

Exclusion set computed each step:
```python
excluded = {name for name in self._held_actors.values() if name is not None}
```

**Why exclude held objects**: while holding the target, the robot part is in contact with it
so distance ≈ 0, which is noise. Exclusion is managed automatically via `attach_object` /
`detach_object` hooks — no task-level changes required.

---

## Distance Method: Broad Phase + Narrow Phase

### Why this design

SAPIEN 3.0.0b1 does not expose a native point-to-surface distance query:
- `scene.get_contacts()` — only reports active contacts (distance ≈ 0)
- `physx_system.raycast()` — distance along one specific ray, not closest surface distance
- `PxGeometryQuery::pointDistance` (PhysX internal) — statically compiled into
  `libsapien.so` with no exported symbols; unreachable via ctypes

`get_global_aabb_fast()` is the key native API: PhysX maintains AABBs in real time,
so querying it each step always reflects the current pose of any actor (static, dynamic,
or articulation link) — no staleness risk.

### Broad Phase (every object, every step)

For each object in the target set, query `get_global_aabb_fast()` from its PhysX component:

```python
physx_comp = next((c for c in entity.components if hasattr(c, 'get_global_aabb_fast')), None)
aabb = physx_comp.get_global_aabb_fast()   # np.ndarray (2,3): [[xmin,ymin,zmin],[xmax,ymax,zmax]]
clamped = np.clip(robot_part_pos, aabb[0], aabb[1])
aabb_dist = np.linalg.norm(robot_part_pos - clamped)  # O(1), always current
```

For articulation links: iterate `articulation.get_links()`, get AABB per link, take the
minimum AABB distance across all links.

Robot entity exclusion uses **object identity** (`art in {left_entity, right_entity}`),
not name comparison, to correctly handle the dual-arm-embodied case where
`left_entity is right_entity`.

### Narrow Phase (only objects within AABB threshold)

If `aabb_dist < aabb_threshold` **and** the object has a pre-loaded trimesh
(i.e. it appears in `collision_list`), compute a more accurate surface distance:

```python
# At init: scaled_mesh = base_mesh.copy(); scaled_mesh.apply_scale(actor.scale)
# Per step (no mesh copy — transform the query point instead):
T_inv = np.linalg.inv(actor.get_pose().to_transformation_matrix())
local_pt = T_inv[:3, :3] @ robot_part_pos + T_inv[:3, 3]
_, dists, _ = trimesh.proximity.closest_point(scaled_mesh, [local_pt])
surface_dist = float(dists[0])
```

Transforming the query point (not the mesh) preserves the BVH — no per-step O(n_vertices)
copy or re-transform.

### Decision table per object per step

| Object in `collision_list`? | AABB dist vs threshold | Reported distance |
|---|---|---|
| Yes (has trimesh) | > threshold | AABB distance |
| Yes (has trimesh) | ≤ threshold | trimesh surface distance |
| No (no trimesh) | any | AABB distance |

### Minimum distance

```python
min_dist = min(candidate_dist for all non-excluded objects)
```

AABB distance is an underestimate of true surface distance (AABB overestimates object
extent). For far objects this is acceptable. For close objects the trimesh narrows phase
gives the accurate value.

Sentinel: `-1.0` if no valid objects exist in the scene.

---

## Mesh Pre-loading at Init (`_init_proximity_tracking`)

Only objects in `collision_list` get pre-loaded trimesh meshes. Two path types:

| `collision_path` type | Handling |
|---|---|
| Single file (`.glb`, `.obj`) | `trimesh.load(path, force="mesh")` |
| Directory (multi-part convex decomp) | Load all `.obj` files, concatenate with `trimesh.util.concatenate` |

After loading: apply actor scale once → `scaled_mesh`. Pre-trigger BVH with a dummy
`trimesh.proximity.closest_point` call to avoid first-step latency spikes.

Objects NOT in `collision_list` (target objects, simple furniture, etc.) use only the
broad-phase AABB at runtime — no mesh pre-loading needed for them.

---

## SAPIEN Version Notes (3.0.0b1)

| API | Details |
|---|---|
| `comp.get_global_aabb_fast()` | `np.ndarray (2,3)` — real-time world-space AABB |
| `comp.compute_global_aabb_tight()` | tighter bound, slightly more expensive |
| `comp.get_collision_shapes()[i].get_half_size()` | box half-extents (NOT `.geometry.half_lengths`) |
| `actor.components` | list of components; filter with `hasattr(c, 'get_global_aabb_fast')` |
| Available on | `PhysxRigidStaticComponent`, `PhysxRigidDynamicComponent` (both confirmed) |
| `PxGeometryQuery` | **not accessible** — statically compiled, no exported symbols |

---

## Robot Part Resolution (`_get_robot_part_position`)

| Part name | Position source |
|---|---|
| `"left_ee"` | `robot.get_left_ee_pose()[:3]` |
| `"right_ee"` | `robot.get_right_ee_pose()[:3]` |
| `"left_tcp"` | `robot.get_left_tcp_pose()[:3]` |
| `"right_tcp"` | `robot.get_right_tcp_pose()[:3]` |
| `"left_<linkname>"` | `robot.left_entity.find_link_by_name(linkname).global_pose.p` |
| `"right_<linkname>"` | `robot.right_entity.find_link_by_name(linkname).global_pose.p` |

Returns `None` if link not found; that part emits `-1.0` sentinel for the entire step.
All configured parts are always present in the output dict (no missing keys).

---

## Files to Modify

### 1. `benchmark/bench_envs/_bench_base_task.py`

New methods on `Bench_base_task`:

**`_init_proximity_tracking(config)`**
- Initialize `_held_actors = {"left": None, "right": None}`
- Parse `enabled`, `robot_parts`, `aabb_threshold` from config
- For each entry in `collision_list`: load and scale trimesh mesh, pre-trigger BVH
  - Handle both single-file and directory `collision_path`
- Store as `_proximity_mesh_cache: dict[str, trimesh.Trimesh]` keyed by actor name
- Initialize `_proximity_episode_min`

**`_get_robot_part_position(part_name) → np.ndarray | None`**
- Resolves part name string to world-space position (see table above)

**`_get_actor_aabb_dist(entity, point) → float`**
- For a regular actor: get PhysX component, call `get_global_aabb_fast()`, return
  analytical point-to-AABB distance
- For an articulation: iterate `articulation.get_links()`, get AABB per link,
  return minimum across all links

**`_compute_proximity_step() → dict`**
Called each saved step from `get_obs()`:
1. Build exclusion set from `_held_actors`
2. Collect all entities: `scene.get_all_actors()` + `scene.get_all_articulations()`
   (excluding robot entities by identity)
3. For each entity not in exclusion set:
   - Broad phase: `aabb_dist = _get_actor_aabb_dist(entity, pos)`
   - Narrow phase: if `aabb_dist < aabb_threshold` and actor name in mesh cache →
     transform query point, call trimesh BVH → `surface_dist`
   - `candidate_dist = surface_dist if narrow phase ran else aabb_dist`
4. `min_dist = min(candidate_dist for all)`; update `_proximity_episode_min`
5. Return `{part_name: {"min_dist": float32}}` for all configured parts

Returns `{}` if `_proximity_enabled` is False.

**`get_proximity_metrics() → dict`**
Returns `_proximity_episode_min` with `inf` replaced by `-1.0`.

**Modified: `attach_object` / `detach_object`**
Add `_held_actors` update (guarded by `hasattr`) after existing code.

---

### 2. `customized_robotwin/envs/_base_task.py`

In `get_obs()`, after building `pkl_dic`:

```python
_prox = getattr(self, '_compute_proximity_step', None)
if _prox is not None:
    proximity = _prox()
    if proximity:
        pkl_dic["proximity"] = {
            part: {k: np.float32(v) for k, v in vals.items()}
            for part, vals in proximity.items()
        }
```

`getattr` guard keeps `Base_Task` free of any import dependency on `Bench_base_task`.

---

### 3–6. Four scene base tasks

After `get_cluttered_surfaces()` and `_build_collision_name_sets()` (so `collision_list`
is fully populated including distractors):

- `benchmark/bench_envs/kitchenl/_kitchen_base_large.py`
- `benchmark/bench_envs/kitchens/_kitchens_base_task.py`
- `benchmark/bench_envs/office/_office_base_task.py`
- `benchmark/bench_envs/study/_study_base_task.py`

```python
_proximity_config = kwags.get("proximity_tracking", {})
if _proximity_config.get("enabled", False):
    self._init_proximity_tracking(_proximity_config)
```

---

### 7. `customized_robotwin/script/collect_data.py`

After `play_once()` in the data collection loop, before writing `scene_info.json`:

```python
if hasattr(TASK_ENV, 'get_proximity_metrics') and getattr(TASK_ENV, '_proximity_enabled', False):
    info["proximity_metrics"] = TASK_ENV.get_proximity_metrics()
```

---

## Known Limitations

- **AABB accuracy**: AABB overestimates object extent → reported distance is a conservative
  underestimate of true clearance. Mitigated by trimesh narrow phase for close objects.
- **Articulation narrow phase**: fridge, cabinet links have no trimesh in `collision_list`;
  they use AABB distance only. Acceptable since they are large static furniture.
- **Non-uniform actor scale**: transforming the query point to mesh-local space gives exact
  Euclidean distance only under uniform scale. Non-uniform scale introduces minor error.
  Most objects in the scene use uniform scale.
- **Episode min tracks saved steps only**: `_proximity_episode_min` updates only when
  `_take_picture()` fires (at `save_freq` intervals), not every physics step. The true
  minimum over all physics steps may be slightly lower.
