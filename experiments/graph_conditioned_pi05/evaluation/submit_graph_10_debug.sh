#!/usr/bin/env bash

# Submit the graph-only 10-seed smoke comparison to Slurm's short-lived debug
# partition. The evaluator is an sbatch array, so the debug partition is passed
# directly to the shared launcher rather than nesting sbatch inside `srun --pty`.
#
# Usage:
#   bash experiments/graph_conditioned_pi05/evaluation/submit_graph_10_debug.sh
#   MAX_CONCURRENT=1 bash experiments/graph_conditioned_pi05/evaluation/submit_graph_10_debug.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EVAL_CONDITION_MODE=graph_only
export TASK_CONFIG=bench_demo_kitchenl_d10
export PARTITION=debug
export TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
# Slurm defines this partition cap as 24000M; 24G expands to 24576M and is
# rejected even though the partition diagnostic describes the limit as 24GB.
export MEMORY="${MEMORY:-24000M}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-5}"

TASK_NAME=put_sauce_can_in_basket
START_INDEX=0
NUM_SEEDS=10
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"

exec "$SCRIPT_DIR/submit_paired_10.sh" \
  "$TASK_NAME" "$START_INDEX" "$NUM_SEEDS" "$MAX_CONCURRENT"
