#!/usr/bin/env bash

# Master/login-node entry point for a checkpoint inference smoke test: loads
# the pi0.5 checkpoint inside the GB10 Docker image and executes one synthetic
# 50-action inference. A load-only server test misses GPU compiler failures.
#
# Usage:
#   scripts/slurm/model_load_test.sh [train_config] [model_name] [ckpt_id] [load_timeout_s]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"
mkdir -p logs

TRAIN_CONFIG="${1:-pi05_robopro_top_cam_jax}"
MODEL_NAME="${2:-robopro_jax}"
CKPT_ID="${3:-30000}"
LOAD_TIMEOUT="${4:-240}"

PARTITION="${PARTITION:-gb10}"
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
MEMORY="${MEMORY:-32G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
# trt-gb10-1 currently cannot initialize NVIDIA containers because NVML is
# inaccessible to the container runtime. Keep it excluded by default for every
# submission path; callers can explicitly clear EXCLUDE_NODES if it is repaired.
EXCLUDE_NODES="${EXCLUDE_NODES-trt-gb10-1}"
IMAGE="${IMAGE:-robopro:gb10}"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
PI05_VENV="${PI05_VENV:-policy/pi05/.venv-jax083}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is not available on this node" >&2
  exit 1
fi
if [[ ! "$LOAD_TIMEOUT" =~ ^[0-9]+$ ]] || (( LOAD_TIMEOUT < 1 )); then
  echo "ERROR: load_timeout_s must be a positive integer: $LOAD_TIMEOUT" >&2
  exit 1
fi
CHECKPOINT_DIR="customized_robotwin/policy/pi05/checkpoints/${TRAIN_CONFIG}/${MODEL_NAME}/${CKPT_ID}"
if [[ -f "$CHECKPOINT_DIR/params/commit_success.txt" ]]; then
  :
elif [[ -s "$CHECKPOINT_DIR/_CHECKPOINT_METADATA" && -s "$CHECKPOINT_DIR/params/_METADATA" && -s "$CHECKPOINT_DIR/params/manifest.ocdbt" ]]; then
  :
else
  echo "ERROR: no complete checkpoint found under: $CHECKPOINT_DIR" >&2
  echo "Expected params/commit_success.txt or standard Orbax checkpoint metadata." >&2
  exit 1
fi

export IMAGE PROJECT_ROOT PI05_VENV
export PULL_IMAGE="${PULL_IMAGE:-0}"
export IMAGE_TAR="${IMAGE_TAR:-}"

node_args=()
if [[ -n "$EXCLUDE_NODES" ]]; then
  node_args+=(--exclude="$EXCLUDE_NODES")
fi

# Mirrors the server half of customized_robotwin/policy/pi05/eval_double_env.sh
# Mirrors the model-server runtime (venv selection, matching cuDNN path,
# and XLA memory flags), then executes one synthetic action inference.
read -r -d '' WORKER_CMD <<EOF || true
set -uo pipefail
cd customized_robotwin
PI05_VENV=${PI05_VENV@Q}
if [[ ! -x "\${PI05_VENV}/bin/python" ]]; then
  echo "[error] pi05 virtualenv not found: \${PI05_VENV}/bin/python" >&2
  exit 1
fi
export PYTHONWARNINGS=ignore::UserWarning
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PI05_COMPUTE_DTYPE="${PI05_COMPUTE_DTYPE:-bfloat16}"
echo "[inference-smoke] PI05_COMPUTE_DTYPE=\${PI05_COMPUTE_DTYPE}"
for cudnn_dir in "\${PI05_VENV}"/lib/python*/site-packages/nvidia/cudnn/lib; do
  if [[ -d "\${cudnn_dir}" ]]; then
    export LD_LIBRARY_PATH="\${cudnn_dir}:\${LD_LIBRARY_PATH:-}"
    echo "[inference-smoke] cuDNN library path: \${cudnn_dir}"
    break
  fi
done
echo "[inference-smoke] train_config=${TRAIN_CONFIG} model=${MODEL_NAME} ckpt=${CKPT_ID} timeout=${LOAD_TIMEOUT}s"
timeout ${LOAD_TIMEOUT} "\${PI05_VENV}/bin/python" ../scripts/pi05_inference_smoke.py \
  --train-config ${TRAIN_CONFIG} \
  --model-name ${MODEL_NAME} \
  --checkpoint-id ${CKPT_ID}
rc=\$?
if (( rc == 0 )); then
  echo "[inference-smoke] checkpoint produced a valid action chunk"
  exit 0
fi
echo "[inference-smoke] failed with code \$rc"
exit \$rc
EOF

JOB_ID=$(sbatch --parsable \
  --partition="$PARTITION" \
  --gres=gpu:1 \
  "${node_args[@]}" \
  --time="$TIME_LIMIT" \
  --mem="$MEMORY" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --job-name="model-load-test-${MODEL_NAME}" \
  --output="$ROOT_DIR/logs/%x_%j.out" \
  --export=ALL \
  "$ROOT_DIR/scripts/slurm/slurm_docker_gb10.sh" \
  bash -lc "$WORKER_CMD")

echo "submitted job $JOB_ID"
echo "log: $ROOT_DIR/logs/model-load-test-${MODEL_NAME}_${JOB_ID}.out"
echo "status: squeue -j $JOB_ID"
