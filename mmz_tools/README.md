# mmz_tools — grounding data-gen tools

Standalone helpers (need only the `robopro` env: numpy/h5py/opencv); they touch no engine code.
For the full story of what the data-gen branch changes (and what's up for redesign), read
**`DATA_GEN_HANDOFF.md` at the repo root**.

Run everything from `customized_robotwin/` after:

```bash
micromamba activate robopro
source set_env.sh
export ROBOTWIN_BENCH_TASK=bench
```

## Tools

```bash
python ../mmz_tools/inspect_hdf5.py <run_dir>/data/episode0.hdf5      # tree + shapes + attrs
python ../mmz_tools/viz_episode.py  <run_dir> <ep> [--cam ...|all]    # panels/boxes/pcd/topdown
bash   ../mmz_tools/time_run.sh <task> <config> [gpu]                 # time one run + project dataset hours
```

Role colors everywhere: target=red, destination=green, obstacles=orange, robot=blue.

## Collecting data

The example config is `benchmark/bench_task_config/mmz_template.yml` (all knobs commented).
Copy → rename → tune, then:

```bash
bash collect_data.sh <task_name> <config_name> <gpu_id>
```

- Collection is resumable: existing `episodeN.hdf5` are skipped, `seed.txt` continues.
- Same scenes across two runs (stock `use_seed`): collect run A first, then
  `mkdir -p data/<save>/<task>/<cfgB> && cp data/<save>/<task>/<cfgA>/seed.txt data/<save>/<task>/<cfgB>/`
  and give config B `use_seed: true`.
- One collector per 16 GB GPU; bigger cards can run several (one task each, parallel panes).

## Timing anchor (measured 2026-07-07, 1× RTX 4080, d10 clutter)

- **~82 s / kept episode** all-in (includes seed-search rejects); seed replay without
  search runs ~37 s / episode.
- Episode size ≈ 37 MB (PNG-compressed masks). Scale estimates with `time_run.sh`.
