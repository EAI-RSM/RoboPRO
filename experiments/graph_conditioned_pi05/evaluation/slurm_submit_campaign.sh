#!/usr/bin/env bash

# Run directly on the login node or as the afterok coordinator job.

set -euo pipefail

: "${MANIFEST:?MANIFEST is missing}"
: "${CAMPAIGN_DIR:?CAMPAIGN_DIR is missing}"
: "${ROOT_DIR:?ROOT_DIR is missing}"

cd "$ROOT_DIR"
condition_mode="${CAMPAIGN_CONDITION_MODE:-paired}"
if [[ "$condition_mode" != "paired" && "$condition_mode" != "graph_only" ]]; then
  echo "ERROR: invalid CAMPAIGN_CONDITION_MODE: $condition_mode" >&2
  exit 2
fi
job_table="$CAMPAIGN_DIR/evaluation_jobs.tsv"
printf 'task\tconfig\tlabel\tcondition_mode\tjob_id\n' > "$job_table"

while IFS=$'\t' read -r task config label; do
  [[ "$task" == "task" ]] && continue
  seed_file="$ROOT_DIR/benchmark/eval_seeds/$task/$config.txt"
  seed_count=0
  [[ -s "$seed_file" ]] && seed_count=$(wc -w < "$seed_file")
  if (( seed_count < 10 )); then
    echo "ERROR: $task/$config has only $seed_count/10 validated seeds" >&2
    exit 3
  fi

  output=$(TASK_CONFIG="$config" \
    EVAL_SEED_FILE="$seed_file" \
    EVAL_CONDITION_MODE="$condition_mode" \
    PARTITION="${PARTITION:-gb10}" \
    EXCLUDE_NODES="${EXCLUDE_NODES-trt-gb10-1}" \
    "$ROOT_DIR/experiments/graph_conditioned_pi05/evaluation/submit_paired_10.sh" \
    "$task" 0 10 "${MAX_CONCURRENT:-2}")
  printf '%s\n' "$output"
  job_id=$(sed -n 's/^submitted .* array job //p' <<< "$output" | head -n 1)
  if [[ ! "$job_id" =~ ^[0-9]+([_;].*)?$ ]]; then
    echo "ERROR: could not parse job ID for $task/$config" >&2
    exit 4
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$task" "$config" "$label" "$condition_mode" "$job_id" >> "$job_table"
done < "$MANIFEST"

echo "submitted all five $condition_mode evaluation arrays"
echo "job table: $job_table"
