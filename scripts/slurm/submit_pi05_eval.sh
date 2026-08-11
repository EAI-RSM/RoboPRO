#!/usr/bin/env bash

# Master/login-node entry point. This script only validates shared inputs and
# submits work; Docker, CUDA, rendering, and policy inference run on the
# allocated GB10 compute node.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"
mkdir -p logs

TASK_NAME="${1:-put_sauce_can_in_basket}"
TASK_CONFIG="${2:-bench_demo_kitchenl_clean}"
TRAIN_CONFIG="${3:-pi05_robopro_top_cam_jax}"
MODEL_NAME="${4:-robopro_jax}"
CKPT_ID="${5:-30000}"
SEED="${6:-0}"
TEST_NUM="${7:-1}"
GRAPH_INPUT_CONDITION="${8:-visual_only}"
EVAL_START_SEED="${9:-4}"

PARTITION="${PARTITION:-gb10}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
MEMORY="${MEMORY:-64G}"
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
if [[ ! -f "benchmark/bench_task_config/${TASK_CONFIG}.yml" ]]; then
  echo "ERROR: task config not found: benchmark/bench_task_config/${TASK_CONFIG}.yml" >&2
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
export PI05_ASSET_ID="${PI05_ASSET_ID:-roboreal_lerobot}"

node_args=()
if [[ -n "$EXCLUDE_NODES" ]]; then
  node_args+=(--exclude="$EXCLUDE_NODES")
fi

JOB_ID=$(sbatch --parsable \
  --partition="$PARTITION" \
  --gres=gpu:1 \
  "${node_args[@]}" \
  --time="$TIME_LIMIT" \
  --mem="$MEMORY" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --job-name="pi05-${GRAPH_INPUT_CONDITION}" \
  --output="$ROOT_DIR/logs/%x_%j.out" \
  --export=ALL \
  "$ROOT_DIR/scripts/slurm/slurm_docker_gb10.sh" \
  make eval-pi05-double \
  "TASK_NAME=$TASK_NAME" \
  "TASK_CONFIG=$TASK_CONFIG" \
  "TRAIN_CONFIG_NAME=$TRAIN_CONFIG" \
  "MODEL_NAME=$MODEL_NAME" \
  "CHECKPOINT_ID=$CKPT_ID" \
  "SEED=$SEED" \
  "EVAL_START_SEED=$EVAL_START_SEED" \
  "TEST_NUM=$TEST_NUM" \
  "GRAPH_INPUT_CONDITION=$GRAPH_INPUT_CONDITION" \
  GPU_SPEC=0)

echo "submitted job $JOB_ID"
echo "log: $ROOT_DIR/logs/pi05-${GRAPH_INPUT_CONDITION}_${JOB_ID}.out"
echo "status: squeue -j $JOB_ID"
