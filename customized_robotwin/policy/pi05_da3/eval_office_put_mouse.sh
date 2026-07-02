#!/bin/bash
# Eval pi05 (step 20000) on RoboPRO benchmark: office / put_mouse_on_pad.
#
# Checkpoint lives at /home/apexs/xuan/models/pi05/20000 and is symlinked into
# the loader's expected layout:
#   policy/pi05/checkpoints/<train_config>/<model_name>/<ckpt_id>/
#
# Dual-env: jax/openpi server (policy/pi05/.venv) + sapien client (robopro conda env),
# talking over a free socket. Server on GPU 0, sim client on GPU 1 by default.
#
# Usage:
#   bash policy/pi05/eval_office_put_mouse.sh [task_config] [seed] [gpu_spec]
# Examples:
#   bash policy/pi05/eval_office_put_mouse.sh                       # clean, seed 0, GPU 0:1
#   bash policy/pi05/eval_office_put_mouse.sh bench_demo_office_clean 0 0:1

set -euo pipefail

# ---- Fixed for this experiment ----
TASK_NAME=put_mouse_on_pad
TRAIN_CONFIG=pi05_aloha_full_base   # matches ckpt step 20000 (num_train_steps)
MODEL_NAME=roboreal_all_80tasks     # subdir under checkpoints/<train_config>/
CHECKPOINT_ID=20000

# ---- Overridable args ----
TASK_CONFIG=${1:-bench_demo_office_clean}
SEED=${2:-0}
GPU_SPEC=${3:-0:1}

# Resolve to the pi05 policy dir regardless of caller CWD, so the relative
# paths inside eval_double_env.sh (it does `cd ../..`) stay correct.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROBOTWIN_ROOT}"

# Bench env wiring (see README): route loaders to BENCH_ROOT/{bench_task_config,bench_envs}.
source set_env.sh
export ROBOTWIN_BENCH_TASK=bench

CKPT_LINK="policy/pi05/checkpoints/${TRAIN_CONFIG}/${MODEL_NAME}/${CHECKPOINT_ID}"
if [[ ! -e "${CKPT_LINK}/params" ]]; then
    echo -e "\033[31m[error] checkpoint not found at ${CKPT_LINK} (params/ missing)\033[0m" >&2
    echo "  expected the symlink -> /home/apexs/xuan/models/pi05/${CHECKPOINT_ID}" >&2
    exit 1
fi

echo -e "\033[36m=== pi05 eval: ${TASK_NAME} / ${TASK_CONFIG} @ step ${CHECKPOINT_ID} (seed ${SEED}, gpu ${GPU_SPEC}) ===\033[0m"

# eval_double_env.sh uses a relative `cd ../..` to reach ROBOTWIN_ROOT, so it
# must be launched from policy/pi05/ (its own directory), not from ROBOTWIN_ROOT.
cd "${SCRIPT_DIR}"
bash eval_double_env.sh \
    "${TASK_NAME}" \
    "${TASK_CONFIG}" \
    "${TRAIN_CONFIG}" \
    "${MODEL_NAME}" \
    "${CHECKPOINT_ID}" \
    "${SEED}" \
    "${GPU_SPEC}"
