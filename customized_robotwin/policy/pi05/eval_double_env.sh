#!/bin/bash
# Dual-env eval for pi05:
#   - server: pi05/.venv (uv-managed, openpi+jax)  → server_gpu
#   - client: RoboTwin conda env (sapien sim)      → client_gpu
# Usage:
#   bash policy/pi05/eval_double_env.sh <task> <task_config> <train_config> <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]
# If client_gpu is omitted, both processes share server_gpu (legacy single-GPU mode).

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
export CUDA_VISIBLE_DEVICES=${client_gpu}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

if [[ -z "${CLIENT_PYTHON:-}" ]]; then
    CLIENT_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${CLIENT_PYTHON}" || ! -x "${CLIENT_PYTHON}" ]]; then
    echo "[error] simulator client Python is unavailable: ${CLIENT_PYTHON:-<unset>}" >&2
    echo "Set CLIENT_PYTHON to the RoboTwin environment's Python executable." >&2
    exit 1
fi
echo -e "\033[33mclient python: ${CLIENT_PYTHON}\033[0m"

# Cold checkpoint restore and JAX initialization can take several minutes.
export MODEL_SERVER_CONNECT_ATTEMPTS="${MODEL_SERVER_CONNECT_ATTEMPTS:-72}"
export MODEL_SERVER_RETRY_DELAY="${MODEL_SERVER_RETRY_DELAY:-5}"

# Find an available port
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
PI05_VENV="${PI05_VENV:-${SCRIPT_DIR}/.venv-jax083}"
if [[ "${PI05_VENV}" != /* ]]; then
    PI05_VENV="${ROBOTWIN_ROOT}/${PI05_VENV}"
fi
if [[ ! -x "${PI05_VENV}/bin/python" ]]; then
    echo "[error] pi05 virtual environment not found: ${PI05_VENV}/bin/python" >&2
    echo "Create it with: scripts/setup_pi05_runtime.sh --sync-gb10" >&2
    exit 1
fi
(
    # Don't `source venv/bin/activate`: conda is already active and `python`
    # would still resolve to the conda env's python (no jax). Use the venv's
    # python directly instead.
    cd "${ROBOTWIN_ROOT}"
    # Inline `VAR=value cmd` prefixes proved unreliable inside `( ... ) &`
    # subshells when slurm pre-sets CUDA_VISIBLE_DEVICES — JAX bound to the
    # parent's first visible GPU instead of ours. Use explicit exports.
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=${server_gpu}
    export PYTHONWARNINGS=ignore::UserWarning
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
    # GB10/CC 12.1 requires the CUDA 13/JAX 0.8 runtime selected above.
    # Keep checkpoint-native BF16 unless an explicit diagnostic overrides it.
    export PI05_COMPUTE_DTYPE="${PI05_COMPUTE_DTYPE:-bfloat16}"
    echo "[server] PI05_COMPUTE_DTYPE=${PI05_COMPUTE_DTYPE}"
    for cudnn_dir in "${PI05_VENV}"/lib/python*/site-packages/nvidia/cudnn/lib; do
        if [[ -d "${cudnn_dir}" ]]; then
            export LD_LIBRARY_PATH="${cudnn_dir}:${LD_LIBRARY_PATH:-}"
            echo "[server] cuDNN library path: ${cudnn_dir}"
            break
        fi
    done
    # `exec` replaces this subshell's bash process with python so that
    # ${SERVER_PID} points to python directly (otherwise `set -e` inhibits
    # bash's exec-optimization, leaving python reparented to init when we
    # `kill ${SERVER_PID}` on cleanup).
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

# --- Client (RoboTwin conda env, current process) ---
echo -e "\033[34m[client] Starting eval_policy_client on port ${FREE_PORT}\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
"${CLIENT_PYTHON}" script/eval_policy_client.py \
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
    --policy_name ${policy_name} \
    --test_num "${TEST_NUM:-100}" \
    --graph_input_condition "${GRAPH_INPUT_CONDITION:-visual_only}" \
    --graph_token_budget "${GRAPH_TOKEN_BUDGET:-120}" \
    --graph_default_camera "${GRAPH_DEFAULT_CAMERA:-countertop_camera}"

echo -e "\033[33m[main] Client finished\033[0m"
