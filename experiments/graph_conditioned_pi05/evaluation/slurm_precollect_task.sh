#!/usr/bin/env bash

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
: "${MANIFEST:?MANIFEST is missing}"
: "${ROOT_DIR:?ROOT_DIR is missing}"

line_number=$((SLURM_ARRAY_TASK_ID + 2))
IFS=$'\t' read -r task config label < <(sed -n "${line_number}p" "$MANIFEST")
if [[ -z "$task" || -z "$config" ]]; then
  echo "ERROR: no experiment at array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

echo "precollecting: $label ($task/$config)"
export PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
export IMAGE="${IMAGE:-robopro:gb10}"
export PULL_IMAGE="${PULL_IMAGE:-0}"
export IMAGE_TAR="${IMAGE_TAR:-$ROOT_DIR/docker/gb10/robopro-gb10.tar}"

"$ROOT_DIR/scripts/slurm/slurm_docker_gb10.sh" \
  make precollect-seeds \
  "TASK_NAME=$task" \
  "TASK_CONFIG=$config" \
  EVAL_SEED_TARGET=10

seed_file="$ROOT_DIR/benchmark/eval_seeds/$task/$config.txt"
seed_count=0
[[ -s "$seed_file" ]] && seed_count=$(wc -w < "$seed_file")
if (( seed_count < 10 )); then
  echo "ERROR: precollection produced only $seed_count/10 seeds: $seed_file" >&2
  exit 3
fi
echo "validated seed bank: $seed_file ($seed_count seeds)"
