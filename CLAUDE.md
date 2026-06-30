# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

RoboPRO is a bimanual manipulation robustness benchmark built on top of a forked RoboTwin 2.0 simulator. Two top-level packages share one repo:

- `customized_robotwin/` — the simulator fork (sapien + mplib + CuRobo). Contains the original RoboTwin task envs under `envs/`, the policy adapters under `policy/`, and the shared runner scripts under `script/` (`collect_data.py`, `eval_policy.py`, `policy_model_server.py`, `eval_policy_client.py`).
- `benchmark/` — RoboPRO additions: new task envs under `bench_envs/{office,study,kitchenl,kitchens}/`, perturbation YAMLs under `bench_task_config/`, instruction banks (`instruction_bank*.json`), and the bench-specific base class `bench_envs/_bench_base_task.py` that extends `envs/_base_task.py:Base_Task`.
- `scripts/` — install helpers (`install/download_assets.py`, `install/patch_aloha_curobo.py`), SLURM batch wrappers (`slurm/slurm_eval_bench.sh`), and upload tools.
- `docs/` — static project page; not runtime code.

## Negative-data collection (targeted desync)

Failure/success **twin-pair** data — the planner plans against a virtual scene while the
physical scene is edited by one controlled amount, so failures are causally labeled by
construction — the importable library is the **`robo_tools`** package (editable path dependency
in the root `.venv`); the planner hooks live in `benchmark/bench_envs/_bench_base_task.py` (all
guarded by `if getattr(self,"targeted",None) is not None` → zero behavior change when the runtime
is absent); and the runner/orchestrator/visualizer live under **`scripts/`**
(`run_targeted_episode.py`, `collect_targeted_data.py`, `visualize_negative_data.py`). Enrichment
(the per-frame `targeted_state` pose trace + clean annotation) is **default** behavior in
`customized_robotwin/script/collect_data.py`; `negative:` / `multimodal:` are per-task **toggles**
that emit sibling variations of each scene seed (parallel annotations: good action, bad action,
another good action), negatives carrying a causal-graph annotation. Unsupported (task, feature)
pairs are skipped. The replayable-state projectors that turn a trace into a web bundle live in
`robo_tools.replayable` + `scripts/replayable/`.

## Environment activation (read before running anything)

**Single env = the repo-root `.venv` (consolidated 2026-06).** Everything — the
simulator stack (sapien, mplib, CuRobo, pytorch3d, warp), the pi05/openpi stack (jax,
torch 2.6), and the `robo_negative` negative-data library — lives in **one uv venv at
the repo root: `.venv` (Python 3.11)**, fully described by the root
`pyproject.toml`/`uv.lock`. It is **self-contained, NOT a uv workspace**: `openpi` and
`openpi-client` are editable **path** dependencies and the compiled sim stack is
declared directly at the root, so `customized_robotwin/policy/pi05` keeps its own
(separate, unused) env config. Build/repair with **`uv sync` from the repo root** (set
`CUDA_HOME` for the compiled CuRobo/pytorch3d builds; then re-apply the sapien/mplib
in-place patches + the `.venv/bin/ffmpeg` symlink — see the install gotchas below).
Run with `.venv/bin/python ...` (or `uv run`) from the repo root; a Jupyter kernel
`robopro-root` points at it. The old conda env and the earlier `policy/pi05/.venv`
have been **removed** (`uv sync`/`uv run` at root are no-ops that keep the sim stack).

Every command must run with both env vars exported:

```bash
cd customized_robotwin
source set_env.sh                  # exports ROBOTWIN_ROOT and BENCH_ROOT
export ROBOTWIN_BENCH_TASK=bench   # critical: routes loaders to benchmark/
```

`ROBOTWIN_BENCH_TASK=bench` is a hard switch. Without it, `class_decorator()` in `script/collect_data.py` and `script/eval_policy.py` falls back to the upstream `envs.<task>` namespace and will not see anything under `benchmark/bench_envs/`. The same flag also switches config loading from `./task_config/<config>.yml` to `$BENCH_ROOT/bench_task_config/<config>.yml`.

The bench loader resolves task modules by walking a fixed list — `bench_envs.<task>`, then `bench_envs.{study,office,kitchenl,kitchens}.<task>` (see `script/collect_data.py:30`). Task module name must equal the class name inside the file.

## Common commands

All run from `customized_robotwin/`:

```bash
# Headless smoke test (also exercises the asset symlink + CuRobo patches)
python script/bench_script/visualize_task_scene.py \
    put_mouse_on_pad bench_demo_office_clean \
    --bench-subdir office --rollout --no-render --seed 0 --save_data

# Collect demonstrations
bash collect_data.sh <task_name> <task_config> <gpu_id>

# Eval — single-process (policy + sim in the ONE consolidated venv; preferred post-consolidation)
#   run with the venv python; set a low XLA mem fraction so jax + sapien/curobo share the GPU.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
  /path/to/repo/.venv/bin/python script/eval_policy.py --config policy/pi05/deploy_policy.yml \
    --overrides --task_name <task> --task_config <config> --train_config_name <train> \
    --model_name <model> --checkpoint_id <id> --ckpt_setting "<train>_<model>_<id>" \
    --policy_name pi05 --seed 0 --instruction_type seen --test_num 10

# Eval — dual-env over a socket (legacy; only needed if you deliberately split sim/policy GPUs)
bash policy/pi05/eval_double_env.sh <task> <task_config> <train_config> <model_name> <ckpt_id> <seed> <server_gpu>[:<client_gpu>]

# Direct eval invocation (same as above, generic interpreter)
python script/eval_policy.py --config policy/pi05/deploy_policy.yml \
    --overrides --task_name <task> --task_config <config> \
    --train_config_name <train> --model_name <model> --checkpoint_id <id> \
    --ckpt_setting "<train>_<model>_<id>" --policy_name pi05 \
    --seed 0 --instruction_type seen --test_num 10

# SLURM batch eval (set --chdir/--output in the header to your checkout)
sbatch scripts/slurm/slurm_eval_bench.sh <task> <config> <train> <model> <ckpt> <seed> <test_num>
```

There is no test suite, linter, or formatter configured. Validation is empirical: a successful rollout writes `Success: True` and an MP4 to `customized_robotwin/data/.../video/`.

## Install gotchas (already done in a working checkout)

These are easy to break and worth re-checking if rollouts blow up:

1. **Asset symlink** — upstream configs use `./assets/embodiments/...` but RoboPRO keeps assets under `benchmark/assets/`. Required:
   ```bash
   ln -sfn ../benchmark/assets customized_robotwin/assets
   ```
2. **Aloha CuRobo configs** — `benchmark/assets/embodiments/aloha-agilex/curobo_{left,right}_tmp.yml` are templates with `${ASSETS_PATH}` placeholders. Generate the real ones, then run `scripts/install/patch_aloha_curobo.py` to inject the `attached_object` link entries (CuRobo needs these to attach grasped objects).
3. **CuRobo `clear_cache` patch** — `customized_robotwin/envs/curobo/src/curobo/geom/sdf/world_mesh.py` requires a manual edit to `clear_cache` (see README "CuRobo cache patch"). Without it, repeated rollouts leak mesh state.
4. **sapien & mplib in-place patches** — small `sed` edits to `sapien/wrapper/urdf_loader.py` (utf-8 + `.srdf` ext) and `mplib/planner.py` (drop the over-eager `or collide` screw-plan abort). `uv sync` does NOT re-apply them, so after any sapien/mplib (re)install re-run the two `sed` edits **and** re-create the `.venv/bin/ffmpeg` symlink → `python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"` (eval shells out to a bare `ffmpeg`). (Legacy conda path: `script/_install.sh`.)
5. **`PYTHONNOUSERSITE=1`** — set it when running so sapien/torch don't resolve from `~/.local/lib` instead of the venv.
6. **CuRobo + pytorch3d compiled extensions** in the uv env are built against torch 2.6 / py3.11 by `uv sync` with `CUDA_HOME` set (`TORCH_CUDA_ARCH_LIST=9.0` for H100; `no-build-isolation-package` in the root pyproject). The aloha CuRobo configs (`benchmark/assets/embodiments/aloha-agilex/curobo_{left,right}.yml`) bake an absolute repo path — re-point `urdf_path`/`collision_spheres` if the repo moves.

## Task configs

Configs in `benchmark/bench_task_config/` follow a strict naming convention used by sweep scripts:

- `bench_demo_<scene>_clean.yml` — baseline (density 0)
- `bench_demo_<scene>_d<6..15>.yml` — increasing obstacle density (10 episodes each)
- `bench_demo_{language,vision,object,vision_lighting,vision_blur,vision_shake}.yml` — perturbation axes
- `bench_perturb_<scene>_<axis>_{clean,d6..d15}.yml` — combined scene × language perturbation × density grid
- `bench_object_ood_*` / `bench_vision_*` — OOD object and vision sweeps

`TASKS.md` is the canonical task list (80 tasks across study/office/kitchenl/kitchens, target 200 episodes each — 100 clean + 100 cluttered d6–d15).

## Adding a new task

1. Create the env under `benchmark/bench_envs/<scene>/<task>.py`. The class name must match the module name (`class_decorator` does `getattr(module, task_name)`).
2. Inherit from `Bench_base_task` (not the upstream `Base_Task` directly) — it sets up the collision-tracking constants (`FURNITURE_NAMES`, `GRIPPER_LINK_NAMES`, `COLLISION_FORCE_THRESHOLD_N`, static-pose thresholds) that the benchmark depends on.
3. Add a step-limit entry to `benchmark/bench_task_config/_bench_eval_step_limit.yml`.
4. Add a description template under `benchmark/bench_description/task_instructions/`.
5. Never reuse an existing upstream RoboTwin task name — collisions silently route to the wrong env via the loader fallback.

## Policy adapters

`policy/` mirrors upstream RoboTwin adapter layout (`ACT`, `DP`, `DP3`, `RDT`, `pi0`, `pi05`, etc.). Each adapter exposes `get_model(...)` and is imported by name through `eval_function_decorator()` in `script/eval_policy.py`. Historically pi05 needed its own uv venv (openpi+jax) separate from the conda sim env, so `eval_double_env.sh` ran `script/policy_model_server.py` in the venv and talked to `script/eval_policy_client.py` (conda) over a socket. **Post-consolidation that split is no longer required** — the repo-root `.venv` (uv workspace) holds both stacks, so `eval_policy.py` runs the policy and sim in **one process** (see the single-process eval command above). The dual-env script remains only as an option for deliberately splitting sim/policy across separate GPUs.

## Output locations

- Collected demos: `customized_robotwin/data/<task_name>/<task_config>/`
- Eval results: `customized_robotwin/eval_result/bench_eval_result/<task>/<policy>/<config>/<ckpt_setting>/<timestamp>/{_result.txt, *.mp4}`
- Asset binaries (objects, embodiments, textures, backgrounds) live in `benchmark/assets/` and are **not** git-tracked — fetched via `scripts/install/download_assets.py` from `Hoshipu/RoboPRO_assets` on HuggingFace.
