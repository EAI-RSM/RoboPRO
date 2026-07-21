# bench_task_config — task/eval configs

Two kinds of YAML live here. **Do not delete or rename the `bench_*` files** — they are the
benchmark's evaluation matrix and exist so results stay comparable across runs and with the
paper.

## The `bench_*` eval configs

One file per **(axis, sub-type, density)**. The **scene is NOT in the config** — it comes
from the *task* you evaluate (the task name resolves to its scene, like `collect_data.sh`).
So one config is reused across many tasks/scenes.

| family | what it measures | scene in name? |
|---|---|---|
| `bench_demo_<scene>_<density>` | in-distribution baseline (also the benchmark's own data-collection set, `collect_data: true`) | yes |
| `bench_vision_<subtype>_<density>` | vision robustness (blur, …) | no |
| `bench_object_ood_<subtype>_<density>` | unseen targets / obstacles / appearance | no |
| `bench_perturb_<scene>_<subtype>_<density>` | hard / rephrased language (per-axis instruction banks) | yes |

`<density>` = `clean` (no clutter) or `d6`…`d15`. Eval **saves no dataset** — each run writes
only a success log to `eval_result/<task>/<policy>/<config>/<ckpt>/<timestamp>/_result.txt`.

## Running them — don't do it by hand, use the driver

You never run 300+ files individually. `customized_robotwin/eval_suite.sh` takes a policy +
a **task selector** and a **config selector** and runs the whole `{tasks} × {configs}` grid
against a single reused model server, then prints one table:

```bash
cd ../../customized_robotwin        # run from here, with the sim env active
source set_env.sh && export ROBOTWIN_BENCH_TASK=bench

# whole office scene, clean baseline (20 tasks x 1 config)
bash eval_suite.sh pi05 30000 office bench_demo_office_clean

# one task across all vision-blur densities (1 task x 11 configs)
bash eval_suite.sh pi05 30000 put_mouse_on_pad "bench_vision_blur_*"

# whole office scene under an OOD config at density 10
bash eval_suite.sh pi05 30000 office bench_object_ood_appearance_d10
```

- `task selector` = a scene (`office`/`study`/`kitchenl`/`kitchens` → all its tasks), a single
  task, or a comma-list.
- `config selector` = a glob (`"bench_vision_*"`), a single config, or a comma-list.
- Results table any time: `python script/eval_summary.py` (writes `eval_result/summary.csv`).

Match the scene for `bench_demo_*` / `bench_perturb_*` (their instruction banks are
scene-specific); `bench_vision_*` / `bench_object_ood_*` are scene-agnostic.

## `datagen_*` configs (different purpose)

`datagen_clean` / `datagen_d6`…`d15` / `datagen_template` are **our data-generation** configs
(they record RGB/depth/masks/3D boxes for training data via `collect_data.sh`). They are not
eval configs — don't mix them up.
