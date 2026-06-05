# Proximity Data Collection

Proximity-aware policy rollout data collection for RoboPRO.
Saves every episode (success **and** failure) with per-step proximity distance,
direction vectors, depth, EE pose, and explicit action labels.

---

## Scripts

| File | Purpose |
|---|---|
| `policy/pi05/collect_rollout.sh` | Shell launcher |
| `script/collect_rollout_proximity_client.py` | Client-side collection logic |

---

## Usage

Run from `customized_robotwin/policy/pi05/`:

```bash
ROBOTWIN_BENCH_TASK=bench \
COLLECT_NUM=1 \
COLLECT_FIXED_SEED=1 \
COLLECT_BRANCH_NUM=1 \
bash collect_rollout.sh \
  <task_name> <task_config> \
  pi05_aloha_full_base roboreal_all_80tasks 20000 0 <server_gpu>:<client_gpu>
```

### Example — 5 tasks × 10 cluttered density levels (d6–d15)

```bash
ROBOTWIN_BENCH_TASK=bench COLLECT_NUM=1 COLLECT_FIXED_SEED=1 COLLECT_BRANCH_NUM=1 \
bash -c '
TASKS=(put_bottle_in_fridge pick_bottle_from_fridge put_bottle_in_basket
       put_sauce_can_in_cabinet pick_can_from_cabinet)
CONFIGS=(bench_demo_kitchenl_d6 bench_demo_kitchenl_d7 bench_demo_kitchenl_d8
         bench_demo_kitchenl_d9 bench_demo_kitchenl_d10 bench_demo_kitchenl_d11
         bench_demo_kitchenl_d12 bench_demo_kitchenl_d13 bench_demo_kitchenl_d14
         bench_demo_kitchenl_d15)
for task in "${TASKS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    bash collect_rollout.sh "$task" "$cfg" \
      pi05_aloha_full_base roboreal_all_80tasks 20000 0 0:1 || \
      echo "[WARN] $task/$cfg failed, continuing..."
  done
done'
```

### Key env vars

| Var | Default | Description |
|---|---|---|
| `COLLECT_NUM` | 100 | Episodes to collect per task/config |
| `COLLECT_FIXED_SEED` | unset | If set, skip expert check (faster) |
| `COLLECT_BRANCH_NUM` | 0 | CuRobo branches per failure; **must be ≥ 1** to populate the buffer |
| `COLLECT_START_SEED` | auto | Override starting seed |
| `ACTION_NOISE_VAR` | 0.001 | Gaussian noise variance on arm joints |

> **Note:** `COLLECT_BRANCH_NUM=0` (simple mode) does not save HDF5 files.
> Always use `COLLECT_BRANCH_NUM=1` or higher.

---

## Output location

All episodes from different density configs (d6–d15) merge into the same
`cluttered/` folder. Episode indices continue automatically across runs.

```
customized_robotwin/proximity_data/
  <task_name>/
    clean/
      data/
        episode_0.hdf5
        episode_1.hdf5
        ...
      videos/
        episode0.mp4
        ...
      collect_summary.json
      seed_state_bench_demo_kitchenl_clean.txt
    cluttered/
      data/
        episode_0.hdf5    ← from d6
        episode_1.hdf5    ← from d7
        episode_2.hdf5    ← from d8
        ...               ← all d6–d15 merged here
      videos/
        episode0.mp4
        ...
      collect_summary.json
      seed_state_bench_demo_kitchenl_d6.txt   ← one per density config
      seed_state_bench_demo_kitchenl_d7.txt
      ...
```

Each `seed_state_<config>.txt` persists the last seed for that density level,
so re-running any config automatically collects fresh seeds.

---

## HDF5 Schema

The schema **matches `collect_data.py` exactly** for all standard keys.
Three extra keys are added for rollout-specific information.

### Standard keys (identical to `collect_data.py`)

| Key | Type | Shape | Description |
|---|---|---|---|
| `observation/{cam}/rgb` | JPEG bytes | `(T,)` | Camera RGB frames |
| `observation/{cam}/depth` | PNG bytes | `(T,)` | Camera depth frames (uint16, mm) |
| `joint_action/left_arm` | float32 | `(T, 6)` | Left arm joint positions |
| `joint_action/left_gripper` | float32 | `(T,)` | Left gripper value |
| `joint_action/right_arm` | float32 | `(T, 6)` | Right arm joint positions |
| `joint_action/right_gripper` | float32 | `(T,)` | Right gripper value |
| `joint_action/vector` | float32 | `(T, 14)` | Full joint state vector |
| `endpose/left_endpose` | float32 | `(T, 7)` | Left EE pose [x,y,z,qw,qx,qy,qz] |
| `endpose/left_gripper` | float32 | `(T,)` | Left EE gripper value |
| `endpose/right_endpose` | float32 | `(T, 7)` | Right EE pose |
| `endpose/right_gripper` | float32 | `(T,)` | Right EE gripper value |
| `proximity/left_ee/min_dist` | float32 | `(T,)` | Min distance from left EE to nearest object |
| `proximity/left_ee/delta` | float32 | `(T, 3)` | Vector from left EE to nearest surface point (world frame) |
| `proximity/right_ee/min_dist` | float32 | `(T,)` | Min distance from right EE to nearest object |
| `proximity/right_ee/delta` | float32 | `(T, 3)` | Vector from right EE to nearest surface point |

Camera names: `head_camera`, `right_camera`, `left_camera`.

### Extra keys (not in `collect_data.py`)

| Key | Type | Shape | Description |
|---|---|---|---|
| `action/left_arm` | float32 | `(T, 6)` | Policy-commanded left arm joints (with noise) |
| `action/left_gripper` | float32 | `(T,)` | Policy-commanded left gripper |
| `action/right_arm` | float32 | `(T, 6)` | Policy-commanded right arm joints (with noise) |
| `action/right_gripper` | float32 | `(T,)` | Policy-commanded right gripper |
| `action/vector` | float32 | `(T, 14)` | Full policy command vector |
| `label` | int8 | scalar | `1` = task success, `0` = failure |

### HDF5 attributes

| Attr | Type | Description |
|---|---|---|
| `instruction` | bytes | Language instruction for this episode |
| `episode` | int | Episode index |
| `success` | bool | Whether the task was completed |
| `collision` | bool | Whether any collision occurred |

### Action alignment

`action[t]` is captured **before** `take_action` executes — it is the command the
policy issued at `observation[t]`. This gives the standard `(obs[t], action[t])`
pairing used in behavior cloning.

`joint_action[t]` is the **actual joint state** at timestep `t` (what the robot
measured), derived from `observation["joint_action"]["vector"]` via `encode_obs`.

---

## Side-by-side schema comparison

```
                          collect_data.py          collect_rollout_proximity
                          ───────────────          ─────────────────────────
observation/{cam}/rgb     ✅ JPEG (T,)             ✅ JPEG (T,)
observation/{cam}/depth   ✅ PNG  (T,)             ✅ PNG  (T,)
joint_action/left_arm     ✅ f32  (T,6)            ✅ f32  (T,6)
joint_action/left_gripper ✅ f32  (T,)             ✅ f32  (T,)
joint_action/right_arm    ✅ f32  (T,6)            ✅ f32  (T,6)
joint_action/right_gripper✅ f32  (T,)             ✅ f32  (T,)
joint_action/vector       ✅ f32  (T,14)           ✅ f32  (T,14)
endpose/left_endpose      ✅ f32  (T,7)            ✅ f32  (T,7)
endpose/left_gripper      ✅ f32  (T,)             ✅ f32  (T,)
endpose/right_endpose     ✅ f32  (T,7)            ✅ f32  (T,7)
endpose/right_gripper     ✅ f32  (T,)             ✅ f32  (T,)
proximity/*/min_dist      ✅ f32  (T,)             ✅ f32  (T,)
proximity/*/delta         ✅ f32  (T,3)            ✅ f32  (T,3)
action/vector (+splits)   ❌                       ✅ f32  (T,14)  [extra]
label                     ❌                       ✅ int8 scalar  [extra]
attrs (instruction, etc.) ❌                       ✅              [extra]
```

---

## Enabling proximity in standard collection (`collect_data.py`)

Proximity is controlled via `data_type.proximity` in the task config:

```yaml
data_type:
  rgb: true
  depth: true
  endpose: true
  qpos: true
  proximity: true       # ← enables proximity tracking (default: true)
```

When `proximity: true`, `collect_data.py` also writes `proximity/*/min_dist` and
`proximity/*/delta` to the standard HDF5. To disable, set `proximity: false`.

To configure which robot parts are tracked, add an optional section:

```yaml
proximity_tracking:
  robot_parts: [left_ee, right_ee]
  aabb_threshold: 1.0
```
