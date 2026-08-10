#!/usr/bin/env bash

# Submit the frozen five-task, d10-clutter paired evaluation campaign.
#
# The launcher first expands each task-specific expert-validated seed bank to
# ten seeds when necessary. A dependency job then submits one paired 20-task
# array per experiment: 10 visual-only + the same 10 retrieved-graph seeds.
#
# Usage:
#   bash experiments/graph_conditioned_pi05/evaluation/submit_diverse_5x10.sh
#   MAX_CONCURRENT=2 PRECOLLECT_MAX_CONCURRENT=2 bash .../submit_diverse_5x10.sh

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT_DIR"

MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
PRECOLLECT_MAX_CONCURRENT="${PRECOLLECT_MAX_CONCURRENT:-2}"
PARTITION="${PARTITION:-gb10}"
EXCLUDE_NODES="${EXCLUDE_NODES-trt-gb10-1}"

for value_name in MAX_CONCURRENT PRECOLLECT_MAX_CONCURRENT; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is unavailable; run this on the Slurm login node" >&2
  exit 1
fi

# Four scene families and three materially different placement geometries:
# support surface, open container, plate, basket, and object stacking.
experiments=(
  $'put_mouse_on_pad\tbench_demo_office_d10\toffice-support'
  $'put_cup_in_box\tbench_demo_study_d10\tstudy-container'
  $'put_spoon_on_plate_ks\tbench_demo_kitchens_d10\tkitchens-support'
  $'put_sauce_can_in_basket\tbench_demo_kitchenl_d10\tkitchenl-container'
  $'put_book_on_book\tbench_demo_office_d10\toffice-stacking'
)

campaign_id=$(date -u +%Y-%m-%d-%H-%M-%S)-diverse-5x10-d10
CAMPAIGN_DIR="$ROOT_DIR/experiments/graph_conditioned_pi05/evaluation/runs/$campaign_id"
mkdir -p "$CAMPAIGN_DIR" "$ROOT_DIR/logs"
MANIFEST="$CAMPAIGN_DIR/experiments.tsv"
printf 'task\tconfig\tlabel\n' > "$MANIFEST"
printf '%s\n' "${experiments[@]}" >> "$MANIFEST"

needs_precollect=0
while IFS=$'\t' read -r task config label; do
  [[ "$task" == "task" ]] && continue
  config_file="$ROOT_DIR/benchmark/bench_task_config/$config.yml"
  seed_file="$ROOT_DIR/benchmark/eval_seeds/$task/$config.txt"
  [[ -f "$config_file" ]] || { echo "ERROR: missing $config_file" >&2; exit 1; }
  density=$(sed -n 's/^[[:space:]]*obstacle_density:[[:space:]]*//p' "$config_file" | head -n 1)
  [[ "$density" == "10" ]] || {
    echo "ERROR: $task/$config must use obstacle_density 10; got ${density:-missing}" >&2
    exit 2
  }
  seed_count=0
  [[ -s "$seed_file" ]] && seed_count=$(wc -w < "$seed_file")
  printf '%s: density=10, validated_seeds=%s\n' "$label" "$seed_count"
  (( seed_count >= 10 )) || needs_precollect=1
done < "$MANIFEST"

export CAMPAIGN_DIR MANIFEST ROOT_DIR PARTITION EXCLUDE_NODES MAX_CONCURRENT
if (( needs_precollect )); then
  node_args=()
  [[ -n "$EXCLUDE_NODES" ]] && node_args+=(--exclude="$EXCLUDE_NODES")
  precollect_job=$(sbatch --parsable \
    --partition="$PARTITION" \
    --gres=gpu:1 \
    "${node_args[@]}" \
    --array="0-4%${PRECOLLECT_MAX_CONCURRENT}" \
    --time="${PRECOLLECT_TIME_LIMIT:-03:00:00}" \
    --mem="${MEMORY:-64G}" \
    --cpus-per-task="${CPUS_PER_TASK:-8}" \
    --job-name=pi05-seeds-5x10 \
    --output="$ROOT_DIR/logs/pi05-seeds_%A_%a.out" \
    --export=ALL \
    "$ROOT_DIR/experiments/graph_conditioned_pi05/evaluation/slurm_precollect_task.sh")
  submit_job=$(sbatch --parsable \
    --partition="$PARTITION" \
    --dependency="afterok:$precollect_job" \
    --time=00:10:00 \
    --mem=1G \
    --cpus-per-task=1 \
    --job-name=pi05-submit-5x10 \
    --output="$ROOT_DIR/logs/pi05-submit_%j.out" \
    --export=ALL \
    "$ROOT_DIR/experiments/graph_conditioned_pi05/evaluation/slurm_submit_campaign.sh")
  printf '%s\n' "$precollect_job" > "$CAMPAIGN_DIR/precollect_job_id.txt"
  printf '%s\n' "$submit_job" > "$CAMPAIGN_DIR/coordinator_job_id.txt"
  echo "submitted seed-precollection array: $precollect_job"
  echo "submitted dependent evaluation coordinator: $submit_job"
  echo "the coordinator will submit five paired arrays after all seed banks reach 10"
else
  "$ROOT_DIR/experiments/graph_conditioned_pi05/evaluation/slurm_submit_campaign.sh"
fi

echo "campaign directory: $CAMPAIGN_DIR"
