# RoboPRO

**P**erturbation-**R**esilient **O**bstacle-awareness — a bimanual manipulation benchmark for policy robustness evaluation.

**Project page:** https://anonymous.4open.science/w/RoboPRO-EDE0/index.html

RoboPRO extends the RoboTwin simulation framework with:
- **Realistic scenes** across office, study, kitchen (small & large) domains
- **Systematic perturbation suite** — Language, Vision, and Object axes for evaluating policy robustness
- **Aloha-Agilex** bimanual embodiment with CuRobo motion planning
- **Targeted failure generation** — labelled baseline↔perturbed *twin pairs* produced by deliberately desyncing the planner ([Targeted negative data](#targeted-negative-data-failure-generation))
- **Replayable-state capture** — reconstruct any episode's full 3D scene and re-derive its semantics offline, no simulator ([Replayable-state capture & demo](#replayable-state-capture--demo))

## Installation

System prereqs (one-time): `libvulkan1 mesa-vulkan-drivers vulkan-tools` (apt), `ffmpeg`, and an NVIDIA driver with CUDA 12.x.

```bash
git clone https://anonymous.4open.science/r/RoboPRO-EDE0
cd RoboPRO
```

### 1. Conda env

```bash
conda create -n robopro python=3.10 -y
# Keep the env isolated from ~/.local/lib site-packages (otherwise sapien/torch
# may resolve there instead of in robopro):
conda env config vars set -n robopro PYTHONNOUSERSITE=1
conda activate robopro
```

### 2. Python deps

```bash
cd customized_robotwin
pip install -r script/requirements.txt
pip install setuptools==69.5.1       # provides pkg_resources for sapien
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
bash script/_install.sh              # patches sapien urdf_loader + mplib planner
```

`script/_install.sh` also clones CuRobo v0.7.8 into `envs/curobo/` and pip-installs it editable, then re-pins `warp-lang==1.12.0` and `setuptools==69.5.1`. If you keep `scipy==1.10.1` from `requirements.txt`, `scikit-image` will print a version-conflict warning — harmless.

### 3. Assets (~15 GB)

```bash
cd ..                                # back to repo root
python scripts/install/download_assets.py
```

This fetches the HuggingFace bundle (`Hoshipu/RoboPRO_assets`) into `benchmark/assets/` (objects, embodiments, background_texture, backgrounds). The bundle already includes the large `aloha-agilex/.../meshes/box2_Link.dae` mesh — no separate fetch needed.

The shipped `task_config/_embodiment_config.yml` uses upstream-relative paths (`./assets/embodiments/...`). RoboPRO keeps assets under `benchmark/assets/`, so add a one-line symlink so the upstream paths resolve:

```bash
ln -sfn ../benchmark/assets customized_robotwin/assets
```

Generate the local-path curobo configs from the shipped templates, and patch them so CuRobo can attach grasped objects (the shipped configs lack the `attached_object` link entries):

```bash
ASSETS_PATH="$(pwd)/benchmark"
cd benchmark/assets/embodiments/aloha-agilex
for side in left right; do
  sed "s|\${ASSETS_PATH}|$ASSETS_PATH|g" curobo_${side}_tmp.yml > curobo_${side}.yml
done
cd -
python scripts/install/patch_aloha_curobo.py
```

### 4. CuRobo cache patch

In `customized_robotwin/envs/curobo/src/curobo/geom/sdf/world_mesh.py`, replace `clear_cache` with:

```python
def clear_cache(self):
    self._wp_mesh_cache = {}
    if self._mesh_tensor_list is not None:
        self._mesh_tensor_list[2][:] = 0
    if self._env_n_mesh is not None:
        self._env_n_mesh[:] = 0
    if self._env_mesh_names is not None:
        for i in range(self.n_envs):
            for j in range(len(self._env_mesh_names)):
                self._env_mesh_names[i][j] = None
    super().clear_cache()
```

### 5. Verify (headless rollout)

```bash
cd customized_robotwin
source set_env.sh
export ROBOTWIN_BENCH_TASK=bench
python script/bench_script/visualize_task_scene.py \
    put_mouse_on_pad bench_demo_office_clean \
    --bench-subdir office --rollout --no-render --seed 0 --save_data
```

Expected on success: a `Success: True` line and an MP4 at `customized_robotwin/data/bench_data/video/episode_put_mouse_on_pad_0.mp4` (~176 frames @ 320×240).

## Usage

All commands run from `customized_robotwin/` with the bench env exported:

```bash
cd customized_robotwin
source set_env.sh                  # exports BENCH_ROOT + ROBOTWIN_ROOT
export ROBOTWIN_BENCH_TASK=bench   # routes loaders to BENCH_ROOT/{bench_task_config, bench_envs}
```

### Collect demonstrations

```bash
bash collect_data.sh <task_name> <task_config> <gpu_id>
# Example:
bash collect_data.sh put_mouse_on_pad bench_demo_office_clean 0
```

Episodes land in `customized_robotwin/data/<task_name>/<task_config>/`.

**The pipeline (`script/collect_data.py`) is enriched by default and extensible by config
knobs** — the [`robo_tools`](robo_tools/) library does the work, and existing behavior is
unchanged when the knobs are off:

- **Enrichment (always on).** Every episode HDF5 additionally gets a per-frame `targeted_state`
  pose trace + a `targeted_labels` annotation — *additive* groups, nothing existing changes.
  This is the minimal state to re-derive scene semantics offline (see Replayable-state below).
- **Sibling variations (per-task toggles).** Add to the task config to produce, for each scene
  seed, **parallel annotations of one scene** — the clean episode (a *good* action) plus
  alternatives:

  ```yaml
  negative:   {enabled: true}    # bad actions  — causally-labelled failures (twins of the clean)
  multimodal: {enabled: true}    # other good actions — N distinct successful behaviors per scene
  ```

  Variations land under `<save_path>/variations/episode<i>/<variant>/`, linked by
  `variations/scene_index.jsonl`. Negatives carry a **causal-graph annotation** (HDF5 `causal`
  group: intervention → mechanism → mediators → outcome, plus the clean counterfactual) for
  diagnostics/assessment. **Tasks not yet audited for a feature are skipped automatically**
  (registries in `robo_tools`); roll out more by extending the registry after testing. Example:
  [`bench_demo_kitchens_variations.yml`](benchmark/bench_task_config/bench_demo_kitchens_variations.yml).

### Run inference (policy eval)

Eval rolls a trained checkpoint out against a `(task, config)` pair and writes a per-rollout success log. Two modes depending on whether your policy fits in the same Python env as the simulator.

**Args (shared by both modes):**

| Arg | Meaning |
|---|---|
| `task_name` | Bench env class, e.g. `put_mouse_on_pad` (file at `benchmark/bench_envs/<scene>/<task>.py`) |
| `task_config` | Perturbation YAML name, e.g. `bench_demo_office_clean` (in `benchmark/bench_task_config/`) |
| `train_config_name` | Training config used to fine-tune the checkpoint |
| `model_name` | Subdir name under `checkpoints/<train_config_name>/` |
| `checkpoint_id` | Step number, e.g. `30000` |
| `seed` | RNG seed for episode initialisation |
| `gpu_id` | CUDA device, or `<server_gpu>:<client_gpu>` for dual-env |

**Mode A — single-process** (policy + sim share one Python env, e.g. when openpi is conda-installable alongside SAPIEN):

```bash
bash policy/pi05/eval.sh <task_name> <task_config> <train_config_name> <model_name> <seed> <gpu_id>
# Example:
bash policy/pi05/eval.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 0 0
```

**Mode B — dual-env / dual-process** (recommended for pi05 since openpi+jax need an isolated uv venv at `policy/pi05/.venv/`):

```bash
bash policy/pi05/eval_double_env.sh <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <seed> <gpu_spec>
# Example (single GPU):
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 0 0
# Example (split: server on GPU 0, sim client on GPU 1):
bash policy/pi05/eval_double_env.sh put_mouse_on_pad bench_demo_office_clean my_office_train pi05_ckpt 30000 0 0:1
```

The script spawns a `policy_model_server.py` in the pi05 venv and an `eval_policy_client.py` in the RoboTwin conda env, communicating over a free socket port.

**Direct Python invocation** (bypassing the shell wrappers):

```bash
python script/eval_policy.py \
    --config policy/pi05/deploy_policy.yml \
    --overrides \
    --task_name put_mouse_on_pad \
    --task_config bench_demo_office_clean \
    --train_config_name my_office_train \
    --model_name pi05_ckpt \
    --checkpoint_id 30000 \
    --ckpt_setting "my_office_train_pi05_ckpt_30000" \
    --policy_name pi05 \
    --seed 0 \
    --instruction_type seen \
    --test_num 10
```

**Where results land:**

```
customized_robotwin/eval_result/bench_eval_result/<task_name>/<policy_name>/<task_config>/<ckpt_setting>/<timestamp>/
    _result.txt        # success count, per-seed pass/fail
    *.mp4              # rollout videos (if eval_video_save is enabled)
```

### Batch eval on SLURM

For sweeping many `(task, config)` pairs across nodes:

```bash
sbatch scripts/slurm/slurm_eval_bench.sh \
    <task_name> <task_config> <train_config_name> <model_name> <checkpoint_id> <seed> <test_num>
```

Set `--chdir` and `--output` in the sbatch header to your local checkout (see comments at the top of `scripts/slurm/slurm_eval_bench.sh`). Pin a specific Python with `export PI05_PYTHON=/path/to/miniconda3/envs/pi05/bin/python`.

For arrayed sweeps over a `(tasks × configs)` grid, see `customized_robotwin/robotwin_*.sbatch` for templates.

## Perturbation configs

Drop-in YAMLs under `benchmark/bench_task_config/`:

| Config | What it perturbs |
|---|---|
| `bench_demo_*_clean.yml` | Baseline (no perturbation) |
| `bench_demo_language.yml` | Per-episode instruction sampled from `instruction_bank.json` |
| `bench_demo_vision.yml` | Lighting (L1–L4) + blur (cycle 5 types) + per-frame pixel shake |
| `bench_demo_vision_lighting.yml` | Lighting only |
| `bench_demo_vision_blur.yml` | Blur only |
| `bench_demo_vision_shake.yml` | Pixel shake only |
| `bench_demo_object.yml` | Target texture swap + unseen obstacles + background_plus |

See the YAMLs in `benchmark/bench_task_config/` for parameter-level details, and `docs/videos/` (`vision*.mp4`, `language.mp4`, `object.mp4`) for sample outputs.

## Scenes and tasks

| Scene | Tasks |
|---|---|
| Office | `put_mouse_on_pad`, `put_phone_on_holder`, `put_book_on_book`, `put_book_in_fileholder`, `put_milktea_on_shelf`, `put_stapler_in_drawer`, `open_drawer`, `close_drawer`, ... |
| Study | `put_book_on_stand`, `put_pen_in_cup`, ... |
| Kitchen (Small) | `put_dish_in_rack`, `place_in_sink`, ... |
| Kitchen (Large) | `microwave_heat`, `fridge_store`, ... |

Full list in `benchmark/bench_envs/`.

## New tasks

1. Write the task env under `benchmark/bench_envs/<scene>/<task>.py`.
2. Add `_eval_step_lim.yml` entry under `benchmark/bench_task_config/`.
3. Add a description template under `benchmark/bench_description/task_instructions/`.

Naming tip: never reuse an existing RoboTwin task name. Start from an analogous sibling task (`kitchenl/`, `office/`, `study/`) — copying a proven recipe is faster than inventing from scratch.

## Targeted negative data (failure generation)

The perturbation suite above is for *evaluating* a trained policy. Separately, RoboPRO can
**generate labelled failure demonstrations** via [`robo_tools/`](robo_tools/),
a twin-pair desync library. For each scene it records a **baseline** (a clean scripted-expert
success) and one or more **perturbed twins** in which the expert is driven to fail in a
*specific, labelled* way — an object shifted right after its grasp is planned, an obstacle
hidden from the planner, the target moved, etc.

The mechanism is a *causal* desync, not random noise: `robo_tools.TargetedRuntime` attaches
to the env (`env.targeted`) and, through one hook in
[`benchmark/bench_envs/_bench_base_task.py`](benchmark/bench_envs/_bench_base_task.py), keeps
the CuRobo collision-world out of sync with reality (excluding actors, freezing pre-shift
poses, firing a shift after the grasp plan commits). Each episode logs a full per-frame
**state trace** (HDF5 `targeted_state` group: actor poses + contacts) and an offline-computed
**label record** (`outcome`, WHAT/WHY/quality, progress & safety curves) — all pure,
versioned, unit-tested functions in `robo_tools`.

```bash
# from the repo root, using the project's sim env (the scripts self-configure BENCH_ROOT etc.)

# minimal — defaults: task put_mouse_on_pad, ptypes {shift_object,shift_target,shift_obstacle},
# seeds 0:40, 3 baselines + up to 30 perturbed twins per group, written to ./targetted_dataset/
python scripts/collect_targeted_data.py --gpu 0

# customised
python scripts/collect_targeted_data.py \
    --tasks put_mouse_on_pad --ptypes shift_object shift_target \
    --seeds 0:10 --n-baseline 3 --target-perturbed 20 \
    --out-root targetted_dataset --gpu 0
```

Every episode runs in its own subprocess ([`scripts/run_targeted_episode.py`](scripts/run_targeted_episode.py))
so CuRobo/SAPIEN state never leaks between a baseline and its twin. Per-episode outputs:
`episode.json` (label record), `data/episode0.hdf5` (standard bench episode + the
`targeted_state` / `targeted_labels` groups), `video/episode0.mp4`.

Browse a collected set as an annotated HTML gallery (baseline vs. perturbed side-by-side,
computed offline — no simulator):

```bash
python scripts/visualize_negative_data.py --root targetted_dataset --out gallery.html
```

Library details in [`robo_tools/`](robo_tools/).

## Multimodal action generation

The same scene usually yields only one demo, so a scene-conditioned action distribution looks
*unimodal*. `robo_tools.MultimodalRuntime` re-parameterizes the planner to produce **several
distinct successful behaviors per scene** — it injects a different transport *waypoint* while
keeping the grasp and place *endpoints* fixed, so the task still succeeds but the arm takes a
visibly different path (`direct`, `high_arc`, `swing_left`, `swing_right`). It's the
`multimodal:` toggle in the collection pipeline above, or a standalone collector
([`scripts/collect_multimodal.py`](scripts/collect_multimodal.py)) with 3D/HTML visualizers
([`scripts/plot_modes.py`](scripts/plot_modes.py), [`scripts/build_multimodal_demo.py`](scripts/build_multimodal_demo.py)).
A mode the planner can't reach is honestly labelled `mode_unrealized`; un-audited tasks are skipped.

## Tooling layout

The data-collection tooling lives in two places: the [`robo_tools/`](robo_tools/) package (the
library — enrichment, the negative-data desync, multimodal generation, the variation planner,
the causal annotation, the task registries) and the repo-root [`scripts/`](scripts/) (the CLIs —
`run_targeted_episode.py`, `collect_targeted_data.py`, `run_mode_episode.py`,
`collect_multimodal.py`, `visualize_negative_data.py`, plotters). The collection entrypoint
`customized_robotwin/script/collect_data.py` consumes the library directly.

## Replayable-state capture & demo

The per-frame state trace above is enough to **reconstruct an episode's entire 3D scene
offline** — no simulator, no re-collection. [`scripts/replayable/`](scripts/replayable/README.md)
turns one collected episode into a self-contained, browser-based viewer: scrub the timeline,
**orbit to camera views that were never captured**, toggle real-mesh ↔ segmentation, and watch
object-level semantics — spatial-relation language, distance-to-goal, re-derived success —
recomputed live from the trace. The whole interactive scene is rebuilt from an ~80 KB trace
versus the ~40 MB of pixels collected for the same episode.

```bash
cd scripts/replayable
./build.sh                                # one collected episode -> web bundle (assets git-ignored)
python serve.py 8000                      # threaded static server -> http://localhost:8000
```

The reusable offline projectors (FK, scale recovery, depth decode, spatial-relation language,
success/grasp/lift) live in the importable library [`robo_tools.replayable`](robo_tools/); the
CLI/bundle (`export_scene.py`, `serve.py`, `web/`) lives under `scripts/replayable/`. The idea —
*capture sufficient state once, project every semantic offline* — and the demo internals + offline
validation are documented in [`scripts/replayable/README.md`](scripts/replayable/README.md).

## Documentation

| Doc | For |
|---|---|
| `README.md` (this file) | Install, eval, perturbation configs, and the two features above |
| [`CLAUDE.md`](CLAUDE.md) | Contributor / agent guidance — env activation, common commands, gotchas |
| [`robo_tools/`](robo_tools/) | Targeted negative-data generation (desync, capture, labels) |
| [`scripts/replayable/README.md`](scripts/replayable/README.md) | Replayable-state reconstruction + interactive HTML viewer |

## License

See `LICENSE`.
