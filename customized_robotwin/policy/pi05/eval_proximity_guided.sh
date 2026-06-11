#!/bin/bash
# Proximity-guided policy evaluation using eval_policy_proximity_guidance.py.
# Starts the pi05 model server (pi05 .venv) and the eval client (RoboPRO conda)
# as separate processes, the same way collect_rollout.sh does.
#
# Usage:
#   bash policy/pi05/eval_proximity_guided.sh <task_name> <task_config> <train_config> <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]
#
# Example (server on GPU 1, client on GPU 0):
#   bash policy/pi05/eval_proximity_guided.sh move_book_onto_table bench_demo_study_clean pi05_aloha_full_base roboreal_all_80tasks 20000 0 1:0
#
# Env var overrides:
#   EVAL_TEST_NUM          — episodes to evaluate (default 100)
#   GUIDANCE_MODE          — "chunk" (CAPE-style, default) | "cbf" | "repulse"
#   SDF_MARGIN             — sdf mode: standoff from arm surface in metres
#                            (SDF_* defaults live in differentiable_proximity.py)
#   GUIDANCE_THRESHOLD     — legacy modes: activation distance in metres (default 0.05)
#   GUIDANCE_CHUNK_GAIN    — chunk mode: push per waypoint per pass (default 0.05)
#   GUIDANCE_CHUNK_ITERS   — chunk mode: refinement passes (default 2)
#   GUIDANCE_GAIN          — repulse mode: max joint correction per step (default 0.20)
#   GUIDANCE_CBF_ALPHA     — cbf mode: allowed approach speed factor (default 0.2)
#   GUIDANCE_CBF_MARGIN    — cbf mode: hard standoff distance in metres (default 0.005)
#   COMPARE                — set to 1 to run guided + unguided on the same seeds
#   EVAL_PROXIMITY_GUIDANCE — set to 0 to run as plain eval with no guidance

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export BENCH_ROOT="${BENCH_ROOT:-$(cd "${ROBOTWIN_ROOT}/../benchmark" && pwd)}"
export ROBOTWIN_BENCH_TASK="${ROBOTWIN_BENCH_TASK:-bench}"

export CUDA_VISIBLE_DEVICES=${client_gpu}

cd "${SCRIPT_DIR}/../.."  # → customized_robotwin/

FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)
echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"

# --- Server (pi05 uv venv) ---
echo -e "\033[32m[server] Starting pi05 model server on GPU ${server_gpu}\033[0m"
PI05_VENV="$(pwd)/policy/pi05/.venv"
(
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=${server_gpu}
    export PYTHONWARNINGS=ignore::UserWarning
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
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

# --- Client (RoboPRO conda env) ---
echo -e "\033[34m[client] Starting proximity-guided eval on port ${FREE_PORT}\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
ROBOTWIN_BENCH_TASK=${ROBOTWIN_BENCH_TASK} \
python script/eval_policy_proximity_guidance.py \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${model_name} \
    --guidance_threshold "${GUIDANCE_THRESHOLD:-0.05}" \
    --guidance_gain "${GUIDANCE_GAIN:-0.20}" \
    ${COMPARE:+--compare} \
    --overrides \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --seed ${seed} \
    --policy_name ${policy_name}

echo -e "\033[33m[main] Eval finished\033[0m"
