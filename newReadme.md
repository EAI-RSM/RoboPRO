# RoboPRO — data-gen branch

**P**erturbation-**R**esilient **O**bstacle-awareness — a bimanual manipulation benchmark
(SAPIEN + CuRobo, Aloha-AgileX) extended here with a **grounding data-generation pipeline**:
every collected episode carries per-object masks, depth, role annotations, exact 3D boxes,
per-frame contact/collision labels, and full replayability. Dataset spec: **`DATA_GEN.md`** ·
Inspection/export tools: **`visualization/README.md`** · Benchmark/eval details: original `README.md`.

## Installation (verified flow)

System prereqs (one-time, host): `libvulkan1 mesa-vulkan-drivers vulkan-tools ffmpeg`
(apt), an NVIDIA driver with CUDA 12.x, and **both `gcc-11` and `g++-11`**
(`sudo apt install gcc-11 g++-11`) — the compiled deps need them.

### 1. Env + CUDA toolkit

```bash
micromamba create -n robopro python=3.10 -y          # (conda works the same)
micromamba activate robopro
export PYTHONNOUSERSITE=1                            # keep ~/.local out of the env

micromamba install -n robopro -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y
nvcc --version                                       # must say release 12.1 (matches torch cu121)
```

### 2. Python deps

CuRobo and pytorch3d compile CUDA code at install time → build **everything with
gcc-11 and `MAX_JOBS=8`** (newer gcc fails against CUDA 12.1; unbounded nvcc jobs
can freeze the machine).

```bash
cd customized_robotwin
pip install -r script/requirements.txt
pip install setuptools==69.5.1

CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11 MAX_JOBS=8 \
  pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# patches sapien/mplib + clones CuRobo into envs/curobo/ (run under the same caps):
CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11 MAX_JOBS=8 bash script/_install.sh

# pin CuRobo to v0.7.8 (the script leaves it on main) and re-assert pins:
cd envs/curobo && git checkout v0.7.8
CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11 MAX_JOBS=8 pip install -e . --no-build-isolation
cd ../.. && pip install warp-lang==1.12.0 setuptools==69.5.1

# verify:
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import curobo, pytorch3d, sapien, mplib; print('imports OK')"
```

### 3. Assets (~15 GB) + config generation

```bash
cd ..                                                # repo root
python scripts/install/download_assets.py --keep-zips   # resumable; delete *.zip after
ln -sfn ../benchmark/assets customized_robotwin/assets  # upstream-relative paths

ASSETS_PATH="$(pwd)/benchmark"
cd benchmark/assets/embodiments/aloha-agilex
for side in left right; do
  sed "s|\${ASSETS_PATH}|$ASSETS_PATH|g" curobo_${side}_tmp.yml > curobo_${side}.yml
done
cd - && python scripts/install/patch_aloha_curobo.py
```

### 4. CuRobo cache patch + verify

Apply the `clear_cache` patch to
`customized_robotwin/envs/curobo/src/curobo/geom/sdf/world_mesh.py`
(exact snippet in the original `README.md` §4). Then:

```bash
cd customized_robotwin
source set_env.sh && export ROBOTWIN_BENCH_TASK=bench
python script/bench_script/visualize_task_scene.py put_mouse_on_pad \
    bench_demo_office_clean --bench-subdir office --rollout --no-render --seed 0 --save_data
# expect: "Success: True" + an mp4 under data/bench_data/video/
```

## Usage

Every session, from `customized_robotwin/`:

```bash
micromamba activate robopro
source set_env.sh                  # exports BENCH_ROOT + ROBOTWIN_ROOT
export ROBOTWIN_BENCH_TASK=bench   # routes loaders to benchmark/{bench_task_config, bench_envs}
```

### Collect grounding data

```bash
bash collect_data.sh <task_name> <config> <gpu>      # single GPU
bash collect_data.sh <task_name> <config> 0,1        # multi-GPU (dynamic per-seed dispatch,
                                                     #   contiguous episodes in one run dir)
bash time_run.sh     <task_name> <config> <gpu>      # same, plus wall-time + sec/episode + projection
```

- Example config: **`benchmark/bench_task_config/datagen_template.yml`** — copy, rename, tune.
  Configs resolve by *name* from `benchmark/bench_task_config/`.
- Output: `data/<save_path>/<task>/<config>/` — episode HDF5s (RGB ×7 cams, depth, raw-id
  object masks, actions, camera matrices, per-frame object poses, per-frame
  contact/collision labels, exact 3D boxes), plus `scene_info.json` (mask-id → name,
  target/destination roles), per-episode scene geometry, instructions, videos, and
  provenance attrs in every file. **Full schema: `DATA_GEN.md` §5.**
- Collection is **resumable** (existing episodes are skipped; `seed.txt` continues).
- Task list: 80 tasks = 4 scenes × 20, under `benchmark/bench_envs/{office,study,kitchenl,kitchens}/`.

### Collision-aware vs collision-blind data (positive / negative pairs)

Two decoupled knobs (bottom of `datagen_template.yml`, semantics in `DATA_GEN.md` §5.2):

| variant | enable_collision_metrics | planner_exclude_obstacles | result |
|---|---|---|---|
| **A** aware | `true` | `false` | planner avoids clutter; rare-but-real collisions |
| **B** blind | `true` | `true` | planner ignores clutter; frequent collisions (contrast/negative data) |

Matched pairs: collect A, copy its `seed.txt` into B's run dir, set `use_seed: true` in B.

### Replay — add modalities without re-collecting

Every episode replays bit-exact from its recorded t=0 state + joint paths:

```bash
python script/replay_trajectory.py <task> <collection_config> --replay-config replay_rich
```

records whatever extra `data_type` flags `replay_rich.yml` enables into `data/replay_data/`.

### Inspect / export what you collected

```bash
python ../visualization/export.py        <run_dir> 0 --what panel --cam countertop_camera  # 6-row quick-look grid
python ../visualization/export.py        <run_dir> 0 --what all                            # + point clouds, 2D/3D boxes
python ../visualization/flag_timeline.py <run_dir> 0                                       # contact/collision timeline
python ../visualization/inspect_hdf5.py  <run_dir>/data/episode0.hdf5                      # HDF5 tree
```

Cameras, flags, and output formats: `visualization/README.md`.

### Evaluate a policy

Unchanged from stock RoboPRO — single-process `eval.sh`, dual-env `eval_double_env.sh`,
SLURM batch, and the perturbation config suite (`bench_demo_*`, `bench_vision_*`, …).
See the original `README.md` §Usage for the full argument tables.

## Repo map (what this branch adds)

| where | what |
|---|---|
| `DATA_GEN.md` | dataset contents, HDF5 schema, label semantics, replay — **read this first** |
| `benchmark/bench_task_config/datagen_template.yml` | commented example collection config |
| `customized_robotwin/script/{collect_parallel,collect_one_episode}.py` | multi-GPU collection |
| `customized_robotwin/script/{replay_trajectory,export_scene}.py` | replay + per-episode scene geometry |
| `visualization/` | inspect / panel / point-cloud / box / timeline tools (`visualization/README.md`) |
| `customized_robotwin/time_run.sh` | timing + full-dataset projection wrapper |
