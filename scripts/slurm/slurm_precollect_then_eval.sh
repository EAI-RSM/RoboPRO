#!/usr/bin/env bash
#SBATCH --job-name=pi05-precollect-eval
#SBATCH --partition=gb10
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$ROOT_DIR/logs"

TASK_NAME="${1:-put_sauce_can_in_basket}"
TASK_CONFIG="${2:-relation_validation_d14}"
TRAIN_CONFIG="${3:-pi05_aloha_full_base}"
MODEL_NAME="${4:-pi05_base}"
CKPT_ID="${5:-0}"
SEED="${6:-0}"
TEST_NUM="${7:-20}"
GRAPH_INPUT_CONDITION="${8:-visual_only}"
EVAL_START_SEED="${9:-4}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export IMAGE="${IMAGE:-robopro:gb10}"
  export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
  export PULL_IMAGE="${PULL_IMAGE:-0}"
  export IMAGE_TAR="${IMAGE_TAR:-}"
  JOB_ID=$(sbatch --parsable \
    --chdir="$ROOT_DIR" \
    --output="$ROOT_DIR/logs/%x_%j.out" \
    --export=ALL \
    "$0" "$@")
  echo "submitted job $JOB_ID"
  echo "log: $ROOT_DIR/logs/pi05-precollect-eval_${JOB_ID}.out"
  exit 0
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"

exec "$ROOT_DIR/scripts/slurm/slurm_docker_gb10.sh" \
  bash -lc 'set -euo pipefail
    make precollect-seeds TASK_NAME="$1" TASK_CONFIG="$2"
    make eval-pi05-double \
      TASK_NAME="$1" \
      TASK_CONFIG="$2" \
      TRAIN_CONFIG_NAME="$3" \
      MODEL_NAME="$4" \
      CHECKPOINT_ID="$5" \
      SEED="$6" \
      TEST_NUM="$7" \
      GRAPH_INPUT_CONDITION="$8" \
      EVAL_START_SEED="$9" \
      GPU_SPEC=0' \
  robopro-precollect-eval \
  "$TASK_NAME" "$TASK_CONFIG" "$TRAIN_CONFIG" "$MODEL_NAME" "$CKPT_ID" \
  "$SEED" "$TEST_NUM" "$GRAPH_INPUT_CONDITION" "$EVAL_START_SEED"
