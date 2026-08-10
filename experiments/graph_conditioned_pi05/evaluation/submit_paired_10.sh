#!/usr/bin/env bash

# Submit visual-only and retrieved-graph episodes as one bounded Slurm array.
# Each array task evaluates one seed. Run this on the Slurm master/login node.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT_DIR"

TASK_NAME="${1:-put_sauce_can_in_basket}"
START_INDEX="${2:-0}"
NUM_SEEDS="${3:-10}"
MAX_CONCURRENT="${4:-2}"
EVAL_CONDITION_MODE="${EVAL_CONDITION_MODE:-paired}"

TASK_CONFIG="${TASK_CONFIG:-bench_demo_kitchenl_clean}"
TRAIN_CONFIG="${TRAIN_CONFIG:-pi05_robopro_top_cam_jax}"
MODEL_NAME="${MODEL_NAME:-robopro_jax}"
CHECKPOINT_ID="${CHECKPOINT_ID:-30000}"
PARTITION="${PARTITION:-gb10}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
MEMORY="${MEMORY:-64G}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
EXCLUDE_NODES="${EXCLUDE_NODES-trt-gb10-1}"
IMAGE="${IMAGE:-robopro:gb10}"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
PI05_VENV="${PI05_VENV:-policy/pi05/.venv-jax083}"
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
  exit 1
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)
if [[ "$PROJECT_ROOT" != "$ROOT_DIR" ]]; then
  echo "ERROR: paired evaluation requires PROJECT_ROOT=$ROOT_DIR; got $PROJECT_ROOT" >&2
  exit 2
fi

for value_name in START_INDEX NUM_SEEDS MAX_CONCURRENT; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $value_name must be a non-negative integer: $value" >&2
    exit 2
  fi
done
if (( NUM_SEEDS < 1 || MAX_CONCURRENT < 1 )); then
  echo "ERROR: NUM_SEEDS and MAX_CONCURRENT must be positive" >&2
  exit 2
fi
if [[ "$EVAL_CONDITION_MODE" != "paired" && "$EVAL_CONDITION_MODE" != "graph_only" ]]; then
  echo "ERROR: EVAL_CONDITION_MODE must be paired or graph_only: $EVAL_CONDITION_MODE" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is unavailable; run this on the Slurm master/login node" >&2
  exit 1
fi
if [[ ! -f "benchmark/bench_task_config/${TASK_CONFIG}.yml" ]]; then
  echo "ERROR: task config not found: benchmark/bench_task_config/${TASK_CONFIG}.yml" >&2
  exit 1
fi
checkpoint_dir="customized_robotwin/policy/pi05/checkpoints/${TRAIN_CONFIG}/${MODEL_NAME}/${CHECKPOINT_ID}"
if [[ -f "$checkpoint_dir/params/commit_success.txt" ]]; then
  :
elif [[ -s "$checkpoint_dir/_CHECKPOINT_METADATA" && -s "$checkpoint_dir/params/_METADATA" && -s "$checkpoint_dir/params/manifest.ocdbt" ]]; then
  :
else
  echo "ERROR: no complete checkpoint found under: $checkpoint_dir" >&2
  echo "Expected params/commit_success.txt or standard Orbax checkpoint metadata." >&2
  exit 1
fi
if [[ ! -s "$checkpoint_dir/assets/roboreal_lerobot/norm_stats.json" ]]; then
  echo "ERROR: roboreal_lerobot normalization stats are missing under: $checkpoint_dir/assets" >&2
  exit 1
fi
if [[ "$PI05_VENV" == /* ]]; then
  echo "ERROR: PI05_VENV must be relative to customized_robotwin inside Docker: $PI05_VENV" >&2
  exit 2
fi
if [[ ! -x "$ROOT_DIR/customized_robotwin/$PI05_VENV/bin/python" ]]; then
  echo "ERROR: pi05 Python is unavailable: customized_robotwin/$PI05_VENV/bin/python" >&2
  exit 1
fi
if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "ERROR: simulator client Python is unavailable: $ROOT_DIR/.venv/bin/python" >&2
  exit 1
fi
runtime_files=(
  experiments/graph_conditioned_pi05/live_adapter.py
  experiments/graph_conditioned_pi05/graph_serializer.py
  customized_robotwin/policy/pi05_robopro_config.py
)
for runtime_file in "${runtime_files[@]}"; do
  if [[ ! -f "$ROOT_DIR/$runtime_file" ]]; then
    echo "ERROR: required runtime file is missing: $runtime_file" >&2
    exit 1
  fi
done

SEED_BANK="${EVAL_SEED_FILE:-$ROOT_DIR/benchmark/eval_seeds/$TASK_NAME/${TASK_CONFIG}.txt}"
if [[ ! -s "$SEED_BANK" ]]; then
  echo "ERROR: fixed eval seed file not found or empty: $SEED_BANK" >&2
  exit 1
fi
mapfile -t ALL_SEEDS < <(tr '[:space:]' '\n' < "$SEED_BANK" | sed '/^$/d')
for seed in "${ALL_SEEDS[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "ERROR: fixed eval seed file contains a non-integer: $seed" >&2
    exit 2
  fi
done
if (( START_INDEX + NUM_SEEDS > ${#ALL_SEEDS[@]} )); then
  echo "ERROR: requested seeds exceed the ${#ALL_SEEDS[@]} fixed seeds in $SEED_BANK" >&2
  exit 2
fi
SELECTED_SEEDS=("${ALL_SEEDS[@]:START_INDEX:NUM_SEEDS}")
first_seed=${SELECTED_SEEDS[0]}
last_seed=${SELECTED_SEEDS[$((NUM_SEEDS - 1))]}
batch_id=$(date -u +%Y-%m-%d-%H-%M-%S)-${TASK_NAME}-seeds${first_seed}-${last_seed}
if [[ "$EVAL_CONDITION_MODE" == "graph_only" ]]; then
  batch_id="${batch_id}-graph-only"
fi
BATCH_DIR="$ROOT_DIR/experiments/graph_conditioned_pi05/evaluation/runs/$batch_id"
if [[ "$EVAL_CONDITION_MODE" == "graph_only" ]]; then
  mkdir -p "$BATCH_DIR/visual_retrieved_graph"
else
  mkdir -p "$BATCH_DIR/visual_only" "$BATCH_DIR/visual_retrieved_graph"
fi
mkdir -p "$BATCH_DIR/tasks" "$ROOT_DIR/logs"

export TASK_NAME TASK_CONFIG TRAIN_CONFIG MODEL_NAME CHECKPOINT_ID
FIXED_SEEDS_FILE="$BATCH_DIR/fixed_seeds.txt"
printf '%s\n' "${SELECTED_SEEDS[@]}" > "$FIXED_SEEDS_FILE"
export START_INDEX NUM_SEEDS FIXED_SEEDS_FILE BATCH_DIR IMAGE PROJECT_ROOT PI05_VENV
export GRAPH_TOKEN_BUDGET="${GRAPH_TOKEN_BUDGET:-120}"
export GRAPH_DEFAULT_CAMERA="${GRAPH_DEFAULT_CAMERA:-countertop_camera}"
export IMAGE_TAR="${IMAGE_TAR:-$ROOT_DIR/docker/gb10/robopro-gb10.tar}"
export PULL_IMAGE="${PULL_IMAGE:-0}"
export PI05_ASSET_ID="${PI05_ASSET_ID:-roboreal_lerobot}"

if [[ "$EVAL_CONDITION_MODE" == "graph_only" ]]; then
  export EVAL_CONDITION_OVERRIDE=visual_retrieved_graph
  array_last=$((NUM_SEEDS - 1))
  job_name="pi05-graph-${TASK_NAME}"
  log_stem="pi05-graph"
else
  export EVAL_CONDITION_OVERRIDE=""
  array_last=$((2 * NUM_SEEDS - 1))
  job_name="pi05-paired-${TASK_NAME}"
  log_stem="pi05-paired"
fi

node_args=()
if [[ -n "$EXCLUDE_NODES" ]]; then
  node_args+=(--exclude="$EXCLUDE_NODES")
fi
job_id=$(sbatch --parsable \
  --partition="$PARTITION" \
  --gres=gpu:1 \
  "${node_args[@]}" \
  --array="0-${array_last}%${MAX_CONCURRENT}" \
  --time="$TIME_LIMIT" \
  --mem="$MEMORY" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --job-name="$job_name" \
  --output="$ROOT_DIR/logs/${log_stem}_%A_%a.out" \
  --export=ALL \
  "$ROOT_DIR/experiments/graph_conditioned_pi05/evaluation/slurm_paired_eval_task.sh")

printf '%s\n' "$job_id" > "$BATCH_DIR/job_id.txt"
{
  printf 'job_id=%s\n' "$job_id"
  printf 'task=%s\n' "$TASK_NAME"
  printf 'seed_bank=%s\n' "$SEED_BANK"
  printf 'seeds=%s\n' "${SELECTED_SEEDS[*]}"
  printf 'condition_mode=%s\n' "$EVAL_CONDITION_MODE"
  printf 'array=0-%s%%%s\n' "$array_last" "$MAX_CONCURRENT"
  if [[ "$EVAL_CONDITION_MODE" == "graph_only" ]]; then
    printf 'visual_retrieved_graph_indices=0-%s\n' "$array_last"
  else
    printf 'visual_only_indices=0-%s\n' "$((NUM_SEEDS - 1))"
    printf 'visual_retrieved_graph_indices=%s-%s\n' "$NUM_SEEDS" "$array_last"
  fi
  printf 'logs=%s/logs/%s_%s_<array_index>.out\n' "$ROOT_DIR" "$log_stem" "$job_id"
} > "$BATCH_DIR/README.txt"

echo "submitted $EVAL_CONDITION_MODE array job $job_id"
echo "batch directory: $BATCH_DIR"
echo "monitor: squeue -j $job_id"
echo "accounting: sacct -j $job_id --format=JobID,State,ExitCode,Elapsed,NodeList"
echo "logs: $ROOT_DIR/logs/${log_stem}_${job_id}_<array_index>.out"
