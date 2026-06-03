#!/bin/bash
# Proximity-aware rollout data collection for pi05.
# Saves every episode (success AND failure) with per-step proximity distance
# and direction vectors.  Output goes to proximity_data/ (separate from rollout_data/).
#
# Usage (from customized_robotwin/policy/pi05/):
#   bash collect_rollout_proximity.sh <task> <task_config> <train_config> <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]
#
# Env vars (override defaults):
#   COLLECT_NUM              — total episodes to save (default 100)
#   COLLECT_START_SEED       — starting seed override
#   COLLECT_BRANCH_NUM       — branches per collision seed; 0 = simple rollout (default 0)
#   COLLECT_BRANCH_LOOKBACK  — comma-separated lookback choices in take_action steps (default "5,10,15")
#   COLLECT_BRANCH_NOISE_STEPS — noised take_action calls per branch (default 1)
#   ACTION_NOISE_VAR         — noise variance for branch perturbation (default 0.005)
#   COLLECT_FIXED_SEED       — if set, skip expert check

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
