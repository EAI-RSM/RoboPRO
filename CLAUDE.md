# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

RoboPRO is a bimanual manipulation robustness benchmark built on top of a forked RoboTwin 2.0 simulator. Two top-level packages share one repo:

- `customized_robotwin/` — the simulator fork (sapien + mplib + CuRobo). Contains the original RoboTwin task envs under `envs/`, the policy adapters under `policy/`, and the shared runner scripts under `script/` (`collect_data.py`, `eval_policy.py`, `policy_model_server.py`, `eval_policy_client.py`).
- `benchmark/` — RoboPRO additions: new task envs under `bench_envs/{office,study,kitchenl,kitchens}/`, perturbation YAMLs under `bench_task_config/`, instruction banks (`instruction_bank*.json`), and the bench-specific base class `bench_envs/_bench_base_task.py` that extends `envs/_base_task.py:Base_Task`.
- `scripts/` — install helpers (`install/download_assets.py`, `install/patch_aloha_curobo.py`), SLURM batch wrappers (`slurm/slurm_eval_bench.sh`), and upload tools.
- `docs/` — static project page; not runtime code.

## Environment activation (read before running anything)

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

# Eval — single-env (policy + sim in one conda env)
bash policy/pi05/eval.sh <task> <task_config> <train_config> <model_name> <seed> <gpu_id>

# Eval — dual-env (recommended for pi05: openpi+jax in uv venv at policy/pi05/.venv/)
bash policy/pi05/eval_double_env.sh <task> <task_config> <train_config> <model_name> <ckpt_id> <seed> <server_gpu>[:<client_gpu>]

# Direct eval invocation (bypass wrapper)
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
4. **`script/_install.sh` patches sapien and mplib in-place** via `sed`, and installs CuRobo editable into `customized_robotwin/envs/curobo/`. Don't reinstall sapien/mplib without re-running this.
5. **`PYTHONNOUSERSITE=1`** on the conda env — otherwise sapien/torch may resolve from `~/.local/lib` instead of the env.

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

`policy/` mirrors upstream RoboTwin adapter layout (`ACT`, `DP`, `DP3`, `RDT`, `pi0`, `pi05`, etc.). Each adapter exposes `get_model(...)` and is imported by name through `eval_function_decorator()` in `script/eval_policy.py`. The pi05 adapter is special-cased because openpi+jax need their own uv-managed venv at `policy/pi05/.venv/` — that's why `eval_double_env.sh` exists, spawning `script/policy_model_server.py` in the pi05 venv and talking to `script/eval_policy_client.py` (running in the conda env) over a socket.

## Output locations

- Collected demos: `customized_robotwin/data/<task_name>/<task_config>/`
- Eval results: `customized_robotwin/eval_result/bench_eval_result/<task>/<policy>/<config>/<ckpt_setting>/<timestamp>/{_result.txt, *.mp4}`
- Asset binaries (objects, embodiments, textures, backgrounds) live in `benchmark/assets/` and are **not** git-tracked — fetched via `scripts/install/download_assets.py` from `Hoshipu/RoboPRO_assets` on HuggingFace.
