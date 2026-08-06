#!/usr/bin/env bash
#SBATCH --job-name=pi05-eval
#SBATCH --partition=gb10
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

# Backward-compatible entry point. Direct execution submits from the master
# node; sbatch execution delegates runtime work to the GB10 Docker image.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec "$ROOT_DIR/scripts/slurm/submit_pi05_eval.sh" "$@"
fi

TASK_NAME="${1:-put_sauce_can_in_basket}"
TASK_CONFIG="${2:-relation_validation_d14}"
TRAIN_CONFIG="${3:-pi05_aloha_full_base}"
MODEL_NAME="${4:-pi05_base}"
CKPT_ID="${5:-0}"
SEED="${6:-0}"
TEST_NUM="${7:-1}"
GRAPH_INPUT_CONDITION="${8:-visual_only}"
EVAL_START_SEED="${9:-4}"

export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"

exec "$ROOT_DIR/scripts/slurm/slurm_docker_gb10.sh" \
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
  GPU_SPEC=0
