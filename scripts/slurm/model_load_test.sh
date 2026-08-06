#!/usr/bin/env bash

# Master/login-node entry point for a checkpoint-load smoke test: allocates
# one GPU, starts policy_model_server.py (which loads the pi0/pi05 checkpoint
# via openpi) inside the GB10 Docker image, then lets `timeout` kill it once
# it has had time to load. No eval client is started and no episode runs —
# this only verifies the checkpoint loads and the server binds its socket.
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
TIME_LIMIT="${TIME_LIMIT:-00:10:00}"
MEMORY="${MEMORY:-32G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
# trt-gb10-1 currently cannot initialize NVIDIA containers because NVML is
# inaccessible to the container runtime. Keep it excluded by default for every
# submission path; callers can explicitly clear EXCLUDE_NODES if it is repaired.
EXCLUDE_NODES="${EXCLUDE_NODES-trt-gb10-1}"
IMAGE="${IMAGE:-robopro:gb10}"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"

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

export IMAGE PROJECT_ROOT
export PULL_IMAGE="${PULL_IMAGE:-0}"
export IMAGE_TAR="${IMAGE_TAR:-}"

node_args=()
if [[ -n "$EXCLUDE_NODES" ]]; then
  node_args+=(--exclude="$EXCLUDE_NODES")
fi

# Mirrors the server half of customized_robotwin/policy/pi05/eval_double_env.sh
# (venv selection, cudnn LD_LIBRARY_PATH shim, XLA memory flags) but never
# starts the sim-side client. `timeout` kills the server once it has had a
# chance to load; exit code 124 there means it was healthily waiting for a
# client and is the expected outcome, not a failure.
read -r -d '' WORKER_CMD <<EOF || true
set -uo pipefail
cd customized_robotwin
PI05_VENV=policy/pi05/.venv
if [[ ! -x "\${PI05_VENV}/bin/python" ]]; then
  echo "[error] pi05 virtualenv not found: \${PI05_VENV}/bin/python" >&2
  exit 1
fi
export PYTHONWARNINGS=ignore::UserWarning
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
for lib in "\${PI05_VENV}"/lib/python*/site-packages/nvidia/cudnn/lib; do
  if [[ -d "\$lib" ]]; then
    export LD_LIBRARY_PATH="\$lib:\${LD_LIBRARY_PATH:-}"
    break
  fi
done
echo "[model-load-test] train_config=${TRAIN_CONFIG} model=${MODEL_NAME} ckpt=${CKPT_ID} timeout=${LOAD_TIMEOUT}s"
timeout ${LOAD_TIMEOUT} "\${PI05_VENV}/bin/python" script/policy_model_server.py \\
  --port 0 \\
  --config policy/pi05/deploy_policy.yml \\
  --overrides \\
  --train_config_name ${TRAIN_CONFIG} \\
  --model_name ${MODEL_NAME} \\
  --checkpoint_id ${CKPT_ID} \\
  --ckpt_setting ${MODEL_NAME} \\
  --seed 0 \\
  --policy_name pi05
rc=\$?
if (( rc == 124 )); then
  echo "[model-load-test] server ran past the timeout without crashing (expected — it serves forever). Check the log above for 'loading model success!'."
  exit 0
fi
echo "[model-load-test] server exited early with code \$rc (unexpected)"
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
