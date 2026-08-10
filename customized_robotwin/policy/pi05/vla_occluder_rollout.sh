#!/bin/bash
# Start the pi05 JAX server and the no-expert office/occluder rollout client.
#
# Clean office smoke (from customized_robotwin/):
#   bash policy/pi05/vla_occluder_rollout.sh \
#       --scene office --num-seeds 1 --run-type office_smoke
#
# The original custom scene remains available with --scene occluder.
#
# Long stock-task run (1000 per density, 3000 total, interleaved):
#   bash policy/pi05/vla_occluder_rollout.sh \
#       --scene task --task-name put_cup_on_coaster --bench-subdir study \
#       --base-config bench_demo_study_clean --clutter-densities 6,10,15 \
#       --rollouts-per-density 1000 --max-steps 600 --report-every 10 \
#       --run-type association_d6_d10_d15
#
# Resume after interruption (the stored config restores every experiment flag):
#   bash policy/pi05/vla_occluder_rollout.sh --resume-dir /absolute/run/directory
#
# Optional environment overrides:
#   GPU_SPEC=0:0 PI0_STEP=50 bash policy/pi05/vla_occluder_rollout.sh ...

set -euo pipefail

policy_name=pi05
task_name=put_mouse_on_pad
task_config=bench_demo_office_clean
train_config_name=pi05_robopro_top_cam_jax
model_name=robopro
checkpoint_id=30000
seed=0
gpu_spec=${GPU_SPEC:-0:0}
pi0_step=${PI0_STEP:-50}

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
ROBOTWIN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_DIR}"
source set_env.sh
export ROBOTWIN_BENCH_TASK=bench

# Find an available port.
FREE_PORT=$(python3 - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)
echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"

# Server: pi05 uv environment with JAX/openpi.
echo -e "\033[32m[server] Activating pi05 .venv\033[0m"
PI05_VENV="${ROBOTWIN_DIR}/policy/pi05/.venv"
(
    cd "${ROBOTWIN_DIR}"
    unset CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=${server_gpu}
    export PYTHONWARNINGS=ignore::UserWarning
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    # 0.45 OOMs during checkpoint restore. Standalone inference peaked at
    # 8,454 MiB; 0.55 is the lowest tested fraction with enough headroom.
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.55
    exec "${PI05_VENV}/bin/python" script/policy_model_server.py \
        --port "${FREE_PORT}" \
        --config "policy/${policy_name}/deploy_policy.yml" \
        --overrides \
        --task_name "${task_name}" \
        --task_config "${task_config}" \
        --train_config_name "${train_config_name}" \
        --model_name "${model_name}" \
        --checkpoint_id "${checkpoint_id}" \
        --ckpt_setting "${model_name}" \
        --seed "${seed}" \
        --policy_name "${policy_name}"
) &
SERVER_PID=$!
trap "echo -e '\033[31m[cleanup] Killing server PID=${SERVER_PID}\033[0m'; kill ${SERVER_PID} 2>/dev/null || true" EXIT

# Client: current RoboTwin/SAPIEN Python environment.
echo -e "\033[34m[client] Starting vla_rollout on port ${FREE_PORT}\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python script/bench_script/vla_rollout.py \
    --port "${FREE_PORT}" \
    --pi0-step "${pi0_step}" \
    "$@"

echo -e "\033[33m[main] Client finished\033[0m"
