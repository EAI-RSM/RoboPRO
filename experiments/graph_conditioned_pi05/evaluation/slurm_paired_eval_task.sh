#!/usr/bin/env bash

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
: "${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is missing}"
: "${PROJECT_ROOT:?PROJECT_ROOT is missing}"
: "${BATCH_DIR:?BATCH_DIR is missing}"
: "${NUM_SEEDS:?NUM_SEEDS is missing}"
: "${FIXED_SEEDS_FILE:?FIXED_SEEDS_FILE is missing}"

index=$SLURM_ARRAY_TASK_ID
condition_override=${EVAL_CONDITION_OVERRIDE:-}
if [[ -n "$condition_override" ]]; then
  if [[ "$condition_override" != "visual_only" && "$condition_override" != "visual_retrieved_graph" ]]; then
    echo "ERROR: invalid EVAL_CONDITION_OVERRIDE: $condition_override" >&2
    exit 2
  fi
  condition=$condition_override
  seed_offset=$index
elif (( index < NUM_SEEDS )); then
  condition=visual_only
  seed_offset=$index
else
  condition=visual_retrieved_graph
  seed_offset=$((index - NUM_SEEDS))
fi
mapfile -t fixed_seeds < "$FIXED_SEEDS_FILE"
if (( seed_offset < 0 || seed_offset >= ${#fixed_seeds[@]} )); then
  echo "ERROR: seed offset $seed_offset is outside $FIXED_SEEDS_FILE" >&2
  exit 2
fi
eval_seed=${fixed_seeds[$seed_offset]}
if [[ ! "$eval_seed" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid fixed evaluation seed: $eval_seed" >&2
  exit 2
fi
seed_file="$BATCH_DIR/tasks/seed_${index}.txt"
printf '%s\n' "$eval_seed" > "$seed_file"

# Docker mounts PROJECT_ROOT at /workspace/RoboPRO. Translate the host path
# instead of passing an absolute path that does not exist inside the container.
case "$seed_file" in
  "$PROJECT_ROOT"/*)
    seed_file_container="/workspace/RoboPRO/${seed_file#"$PROJECT_ROOT"/}"
    ;;
  *)
    echo "ERROR: seed file is outside the Docker project mount: $seed_file" >&2
    exit 2
    ;;
esac
run_tag="${condition}-seed${eval_seed}-a${SLURM_ARRAY_JOB_ID}_${index}"
export EVAL_RUN_TAG="$run_tag"

echo "array job: $SLURM_ARRAY_JOB_ID"
echo "array index: $index"
echo "condition: $condition"
echo "evaluation seed: $eval_seed"
echo "run tag: $run_tag"

set +e
"$PROJECT_ROOT/scripts/slurm/slurm_docker_gb10.sh" \
  make eval-pi05-double \
  "TASK_NAME=$TASK_NAME" \
  "TASK_CONFIG=$TASK_CONFIG" \
  "TRAIN_CONFIG_NAME=$TRAIN_CONFIG" \
  "MODEL_NAME=$MODEL_NAME" \
  "CHECKPOINT_ID=$CHECKPOINT_ID" \
  SEED=0 \
  "EVAL_SEED_FILE=$seed_file_container" \
  TEST_NUM=1 \
  "GRAPH_INPUT_CONDITION=$condition" \
  "GRAPH_TOKEN_BUDGET=$GRAPH_TOKEN_BUDGET" \
  "GRAPH_DEFAULT_CAMERA=$GRAPH_DEFAULT_CAMERA" \
  GPU_SPEC=0
exit_code=$?
set -e

task_record="$BATCH_DIR/tasks/task_${index}.txt"
{
  printf 'condition=%s\n' "$condition"
  printf 'seed=%s\n' "$eval_seed"
  printf 'run_tag=%s\n' "$run_tag"
  printf 'seed_file_host=%s\n' "$seed_file"
  printf 'seed_file_container=%s\n' "$seed_file_container"
  printf 'exit_code=%s\n' "$exit_code"
} > "$task_record"

if (( exit_code != 0 )); then
  echo "ERROR: evaluation exited with code $exit_code" >&2
  exit "$exit_code"
fi

result_root="$PROJECT_ROOT/customized_robotwin/eval_result/$TASK_NAME/pi05/$TASK_CONFIG/$MODEL_NAME"
result_dir=$(find "$result_root" -mindepth 1 -maxdepth 1 -type d -name "*-${run_tag}" -print -quit)
if [[ -z "$result_dir" || ! -s "$result_dir/_episodes.jsonl" ]]; then
  echo "ERROR: non-empty result for run tag $run_tag was not found under $result_root" >&2
  exit 3
fi
if ! python3 - "$result_dir/_episodes.jsonl" "$eval_seed" "$condition" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_seed = int(sys.argv[2])
expected_condition = sys.argv[3]
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if len(records) != 1:
    raise SystemExit(f"expected exactly one episode record, found {len(records)} in {path}")
record = records[0]
if int(record.get("seed", -1)) != expected_seed:
    raise SystemExit(f"expected seed {expected_seed}, found {record.get('seed')} in {path}")
if record.get("graph_input_condition") != expected_condition:
    raise SystemExit(
        f"expected condition {expected_condition}, found "
        f"{record.get('graph_input_condition')} in {path}"
    )
PY
then
  echo "ERROR: result validation failed for $result_dir/_episodes.jsonl" >&2
  exit 3
fi

seed_dir="$BATCH_DIR/$condition/seed_$(printf '%03d' "$eval_seed")"
mkdir -p "$seed_dir"
cp "$result_dir/_episodes.jsonl" "$seed_dir/_episodes.jsonl"
printf '%s\n' "$result_dir" > "$seed_dir/result_path.txt"
printf 'result_dir=%s\n' "$result_dir" >> "$task_record"

echo "result directory: $result_dir"
echo "batch record: $seed_dir"
