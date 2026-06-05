#!/bin/bash
# Rollout data collection for pi05 → rollout_data/ (proximity HDF5 schema).
# Replaces both collect_rollout.sh and collect_rollout_proximity.sh.
#
# Usage:
#   bash policy/pi05/collect_rollout.sh <task> <task_config> <train_config> <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]
# If client_gpu is omitted, both processes share server_gpu.
#
# ── How to set flags ────────────────────────────────────────────────────────
# Prefix the command with env vars, or export them before running:
#
#   COLLECT_NUM=200 bash policy/pi05/collect_rollout.sh task cfg train model 20000 0 0
#   COLLECT_BRANCH_NUM=1 COLLECT_NUM=500 bash policy/pi05/collect_rollout.sh ...
#   COLLECT_SELECTIVE_SAVE=0 bash policy/pi05/collect_rollout.sh ...   # save all episodes
#   COLLECT_SELECTIVE_SAVE=1 bash policy/pi05/collect_rollout.sh ...   # selective save
#
# ── Flags ───────────────────────────────────────────────────────────────────
#   COLLECT_NUM              — total episodes to save (default 100)
#   COLLECT_START_SEED       — starting seed (default auto-resumed from seed_state file)
#   COLLECT_BRANCH_NUM       — CuRobo branches per failure/collision; 0 = simple rollout (default 0)
#   COLLECT_EXPERT_CHECK     — if set to 1, run CuRobo expert feasibility check before each episode (slower)
#   COLLECT_SELECTIVE_SAVE   — 1: save primary only on fail/collision, branch only on success.
#                              0 (default): save every episode (success and failure).
#   COLLECT_MODE             — "" (default) / "stitched": use stitched pi05→CuRobo→pi05 rollout.

set -euo pipefail

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
checkpoint_id=${5:-20000}
seed=${6:-0}
gpu_spec=${7:-0}

if [[ "${gpu_spec}" == *":"* ]]; then
    server_gpu="${gpu_spec%%:*}"
    client_gpu="${gpu_spec##*:}"
else
    server_gpu="${gpu_spec}"
    client_gpu="${gpu_spec}"
fi
echo -e "\033[33mserver gpu: ${server_gpu}, client gpu: ${client_gpu}\033[0m"

# ── Env vars (with defaults) ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export BENCH_ROOT="${BENCH_ROOT:-$(cd "${ROBOTWIN_ROOT}/../benchmark" && pwd)}"
export ROBOTWIN_BENCH_TASK="${ROBOTWIN_BENCH_TASK:-}"


export CUDA_VISIBLE_DEVICES=${client_gpu}
export COLLECT_SELECTIVE_SAVE="${COLLECT_SELECTIVE_SAVE:-0}"
export COLLECT_EXPERT_CHECK="${COLLECT_EXPERT_CHECK:-0}"

cd ../..  # → customized_robotwin/

FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)
echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"

# --- Server (pi05 uv venv) ---
echo -e "\033[32m[server] Activating pi05 .venv\033[0m"
PI05_VENV="$(pwd)/policy/pi05/.venv"
(
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=${server_gpu}
    export PYTHONWARNINGS=ignore::UserWarning
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    exec "${PI05_VENV}/bin/python" script/policy_model_server.py \
        --port ${FREE_PORT} \
        --config policy/${policy_name}/deploy_policy.yml \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --checkpoint_id ${checkpoint_id} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name}
) &
SERVER_PID=$!
trap "echo -e '\033[31m[cleanup] Killing server PID=${SERVER_PID}\033[0m'; kill ${SERVER_PID} 2>/dev/null || true" EXIT

# --- Client (RoboPRO conda env, current process) ---
echo -e "\033[34m[client] Starting collect_rollout_proximity_client on port ${FREE_PORT}\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python script/collect_rollout_proximity_client.py \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name}

echo -e "\033[33m[main] Collection finished\033[0m"
