#!/bin/bash

set -euo pipefail

task_name=${1}
task_config=${2}
gpu_id=${3:-0}

# Benchmark configs live outside the stock RoboTwin task tree. Infer the task
# selector when the caller did not set it explicitly.
if [[ -z "${ROBOTWIN_BENCH_TASK:-}" && -f "../benchmark/bench_task_config/${task_config}.yml" ]]; then
    export ROBOTWIN_BENCH_TASK=bench
fi

# Respect an explicit interpreter, then an activated environment.  The repo's
# uv environment is the fallback for non-interactive shells (such as Codex),
# where the `python` alias may not exist even though dependencies are installed.
if [[ -n "${ROBOPRO_PYTHON:-}" ]]; then
    python_bin="${ROBOPRO_PYTHON}"
elif command -v python >/dev/null 2>&1; then
    python_bin="$(command -v python)"
elif [[ -x "$(dirname "$0")/../.venv/bin/python" ]]; then
    python_bin="$(dirname "$0")/../.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
else
    echo "No Python interpreter found. Activate the robopro environment or set ROBOPRO_PYTHON." >&2
    exit 127
fi

if [[ -x ./script/.update_path.sh ]]; then
    ./script/.update_path.sh > /dev/null 2>&1
fi

if [[ "${gpu_id}" == *,* ]]; then
    # Multi-GPU: dynamic per-seed dispatch across the listed GPUs (e.g. 0,1).
    # episode_num is read from the task config, so the amount collected matches
    # the single-GPU path. All episodes land in one run dir (episode0..N-1).
    PYTHONWARNINGS=ignore::UserWarning \
    "${python_bin}" -u script/collect_parallel.py "${task_name}" "${task_config}" --gpus "${gpu_id}"
else
    # Single-GPU: stock sequential collection.
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    PYTHONWARNINGS=ignore::UserWarning \
    "${python_bin}" -u script/collect_data.py "${task_name}" "${task_config}"
    rm -rf data/${task_name}/${task_config}/.cache
fi
