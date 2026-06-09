# Getting Started with RoboPRO

A plain-language guide for newcomers. If you've never worked with a robot simulator or a VLA policy before, start here.

---

## 1. What is this repo for?

RoboPRO is a **benchmark**: a fixed set of tests you run a robot-controller AI against, to get a number back that says how good it is.

The specific thing being tested is a **bimanual manipulation policy** — a neural network that controls a two-armed robot (the Aloha-Agilex) doing everyday tabletop tasks like "put the mouse on the mousepad" or "put the bowl in the sink." The policy takes camera images + a natural-language instruction as input, and outputs joint commands for the robot.

The thing that makes RoboPRO different from other benchmarks is that it doesn't just test the policy on **clean** scenes. It also systematically **perturbs** the scene to measure how robust the policy is when the world doesn't look exactly like the training data. Specifically:

| Perturbation axis | What changes |
|---|---|
| **Vision** | Lighting (4 levels), camera blur (5 types), per-frame pixel jitter |
| **Language** | Instruction wording (paraphrases, polite phrasing, indirect speech, reasoning chains, distractor sentences) |
| **Object** | Target texture swaps, out-of-distribution object shapes, unseen obstacles |
| **Density** | Number of unrelated clutter objects on the table — `d6` through `d15` (6 to 15 distractors) |

So instead of one success number per task, you get a **grid**: each cell tells you the policy's success rate on a specific (task, perturbation, density) combination. A robust policy holds up across the grid; a brittle one collapses as soon as the lights change or someone rephrases the instruction.

The benchmark has **80 tasks** spread across **4 scenes** (Study, Office, Kitchen-Large, Kitchen-Small) — full list in `TASKS.md`.

---

## 2. The mental model

Three things to keep distinct:

1. **The simulator** (SAPIEN) — a physics engine that pretends to be a real room with real objects and a real robot. Renders camera images, simulates gripper contacts, etc. *RoboPRO does not train policies; it just provides a place to run them.*
2. **The expert** (motion planner, CuRobo) — a hand-written algorithm that knows how to solve each task. It produces clean demonstration trajectories that look like a human teleoperator did the task. These are the **demonstrations** you'd use as training data.
3. **The policy** (your neural net) — what you actually want to evaluate. It sees the camera images and instruction, outputs actions, and is *graded* on whether the task succeeded.

The typical lifecycle:

```
                  ┌─────────────────────────────────────────────┐
                  │                                             │
[bench task] → [collect demos with expert] → [train your policy elsewhere] →
                                                                 │
                                                                 ▼
                  [eval policy in sim across the perturbation grid] → [success-rate table]
```

You probably don't train policies *in this repo* — training happens in a separate codebase (e.g. the openpi / pi05 stack, see `customized_robotwin/policy/pi05/`). RoboPRO is the **data factory** (step 2) and the **evaluator** (step 4).

---

## 3. Vocabulary cheat-sheet

| Term | What it means here |
|---|---|
| **Embodiment** | The physical robot model. RoboPRO uses Aloha-Agilex, a 2-arm setup with grippers. |
| **Scene** | The room/table layout (office, study, kitchen-large, kitchen-small). Includes furniture and shelving. |
| **Task** | A specific goal in a scene, like `put_mouse_on_pad`. ~80 of them total. |
| **Task config** | A YAML in `benchmark/bench_task_config/` that wraps a task with parameters (seed range, perturbation knobs, episode count). |
| **Episode** | One attempt at a task with a specific random seed. Lasts a few hundred sim steps. |
| **Rollout** | The act of running an episode end-to-end — either the expert (for demos) or the policy (for eval). |
| **Demonstration / demo** | A successful expert rollout, saved as an HDF5 of (images, robot state, actions) — used as training data. |
| **Policy** | The neural net you're evaluating. Different policies (pi05, ACT, RDT, DP3, …) plug into `customized_robotwin/policy/`. |
| **Success rate** | Of N seeds tried, how many succeeded. The main metric. |
| **Density `dN`** | N distractor objects added to the table. `d6`=6, `d15`=15. `clean`=0. |

---

## 4. Hands-on quickstart

Your environment is already set up — two parallel options:

```bash
# Option A: conda env
source ~/miniforge3/etc/profile.d/conda.sh && conda activate robopro

# Option B: uv venv
source /work/mohammed/RoboPRO/.venv/bin/activate
```

Then for every command below, you also need:

```bash
cd /work/mohammed/RoboPRO/customized_robotwin
source set_env.sh                  # exports BENCH_ROOT + ROBOTWIN_ROOT
export ROBOTWIN_BENCH_TASK=bench   # routes the loaders to benchmark/
```

GPUs 0–6 on this machine are busy. **Use GPU 7** (`CUDA_VISIBLE_DEVICES=7`).

### Quickstart 1 — see one task run

This runs the expert (motion planner) on one episode and saves a video. Use this to *visually verify* a task works on your machine.

```bash
CUDA_VISIBLE_DEVICES=7 python script/bench_script/visualize_task_scene.py \
    put_mouse_on_pad bench_demo_office_clean \
    --bench-subdir office --rollout --no-render --seed 0 --save_data
```

After ~30s you'll see `Success: True` and a file at `data/bench_data/video/episode_put_mouse_on_pad_0.mp4`. Open it — you'll see the robot pick up a mouse and place it on the pad.

Try changing the task and seed to explore:
- `put_mouse_on_pad` → `put_phone_on_holder` (different office task)
- `--seed 0` → `--seed 5` (different initial object placement)
- `bench_demo_office_clean` → `bench_demo_office_d10` (10 distractor objects on the table)

### Quickstart 2 — collect demonstrations for training

This runs the expert *many times* and saves every successful trajectory. Output is HDF5 files you'd feed into a training pipeline.

```bash
bash collect_data.sh put_mouse_on_pad bench_demo_office_clean 7
```

Episodes land in `data/put_mouse_on_pad/bench_demo_office_clean/`. The YAML controls how many episodes to collect (`episode_num`).

**What "the expert" actually is.** Each demo is generated by a motion planner that has *full state access* — it knows the exact pose of every object and joint. For the Aloha embodiment, that planner is **NVIDIA CuRobo**: a GPU-accelerated, gradient-based trajectory optimizer (`customized_robotwin/envs/robot/planner.py`, class `CuroboPlanner`). For each `pick → move → place` step the task code calls, CuRobo:

1. Receives the current scene's collision meshes (table, furniture, target objects) on the GPU.
2. Solves IK + collision-free trajopt for the target end-effector pose.
3. Hands the joint-space waypoints to **toppra** for time-optimal retiming under the robot's velocity/acceleration limits.
4. Sapien steps the simulation at 250 Hz and the camera streams are recorded into the HDF5.

The embodiment chooses its planner in `benchmark/assets/embodiments/<robot>/config.yml` (`planner: "curobo"` for Aloha; the `mplib_screw` / `mplib_RRT` alternatives are sample-based CPU planners from `mplib`, used as fallbacks for embodiments where CuRobo isn't tuned).

**The "clutter obstacles" cheat.** You'll see this line at startup: `<task> curobo planner skips clutter obstacles`. The distractor objects added by the density configs (`d6..d15`) are visible to the cameras but **invisible to the planner**. That's deliberate: it lets the expert produce clean demos even on cluttered scenes, so the resulting HDF5 contains the same kind of trajectory you'd want a policy to imitate. The policy, in turn, sees only RGB images during eval — that's where the clutter actually matters.

### Quickstart 3 — what's a config file actually look like?

Open one to demystify:

```bash
cat /work/mohammed/RoboPRO/benchmark/bench_task_config/bench_demo_office_clean.yml
```

You'll see knobs for episode count, camera type, embodiment, save format, and a `domain_randomization` block that controls perturbations. The `bench_demo_office_d12.yml` neighbour file is the same shape, but with `obstacle_density: 12`.

### Quickstart 4 — evaluating a trained policy

You need a checkpoint for this. Once you have one (e.g. a fine-tuned pi05 model):

```bash
bash policy/pi05/eval.sh \
    put_mouse_on_pad bench_demo_office_clean \
    my_train_config my_model_name 0 7
```

This runs your policy on a set of seeds and writes a `_result.txt` with pass/fail per seed under `eval_result/bench_eval_result/...`.

---

## 5. Where things live

```
RoboPRO/
├── README.md                 ← install + commands (terse)
├── GETTING_STARTED.md        ← this file
├── CLAUDE.md                 ← guidance for AI coding assistants
├── TASKS.md                  ← the 80-task list (Study/Office/KitchenL/KitchenS)
│
├── customized_robotwin/      ← the simulator fork (RoboTwin 2.0 + our tweaks)
│   ├── envs/                 ← upstream RoboTwin task envs + the robot model + CuRobo
│   ├── policy/               ← policy adapters (pi05, ACT, RDT, DP3, …)
│   ├── script/               ← runner scripts (collect_data.py, eval_policy.py)
│   └── set_env.sh            ← sourced before running anything
│
├── benchmark/                ← RoboPRO's additions on top of upstream RoboTwin
│   ├── bench_envs/           ← the 80 task definitions, grouped by scene
│   │   ├── office/           ← put_mouse_on_pad.py, put_phone_on_holder.py, …
│   │   ├── study/            ← put_pen_in_pencup.py, move_cup.py, …
│   │   ├── kitchenl/         ← put_bottle_in_basket.py, pick_can_from_cabinet.py, …
│   │   └── kitchens/         ← put_bowl_in_sink_ks.py, drop_apple_in_bin_ks.py, …
│   ├── bench_task_config/    ← YAML configs (clean + d6..d15 + perturbation variants)
│   └── assets/               ← downloaded 3D models, embodiment URDFs, textures (~15 GB)
│
└── scripts/                  ← install helpers + SLURM batch wrappers
```

A useful pattern: open any task file (`benchmark/bench_envs/office/put_mouse_on_pad.py`) to see the actual logic — they're short (~50 lines) and read top-to-bottom as a recipe (`pick_object → move_to_target → release`).

---

## 6. Common gotchas

- **Forgetting `ROBOTWIN_BENCH_TASK=bench`** → loaders won't find the bench tasks; you'll get cryptic `No Task` or `ModuleNotFoundError`. This is the most common new-user trip.
- **Wrong `--bench-subdir`** → the script will try and fail to import the task. Match the scene folder under `bench_envs/` (`office`, `study`, `kitchenl`, `kitchens`).
- **GPU OOM** → another user is on your GPU. Check with `nvidia-smi`, then pin to a free one with `CUDA_VISIBLE_DEVICES=<n>`.
- **`Success: False` on a clean config with seed 0** → almost always a planning or asset-path issue, not a task bug. The clean configs are tuned to be solvable.
- **MP4 saving fails** → the system `ffmpeg` binary isn't in PATH. Both envs have a workaround symlinked into `bin/`.

---

## 7. Where to go next

- **Reading the codebase**: start with a single task file (`benchmark/bench_envs/office/put_mouse_on_pad.py`) and trace what `play_once()` does — it'll make the rest of the framework click.
- **Understanding a perturbation axis**: read one perturbation YAML (`benchmark/bench_task_config/bench_demo_vision.yml`) and the `domain_randomization` handling in `customized_robotwin/envs/_base_task.py`.
- **Adding your own task**: copy a similar sibling task in the same scene folder, change the object names and target pose, add a step-limit entry to `_bench_eval_step_limit.yml`. Naming tip: don't reuse an existing upstream RoboTwin task name.
- **Running on a cluster**: `scripts/slurm/slurm_eval_bench.sh` is the SLURM wrapper.
- **For AI assistants working in this repo**: `CLAUDE.md` has the high-density architecture summary.
- **Project page / paper**: <https://anonymous.4open.science/w/RoboPRO-EDE0/index.html>
