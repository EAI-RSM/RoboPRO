#!/usr/bin/env bash

# Submit a graph-only smoke comparison on the exact frozen sauce-can baseline.
# This intentionally matches the diverse 5x10 campaign task configuration and
# all ten task-specific d10 seeds; visual-only is not rerun.
#
# Usage:
#   bash experiments/graph_conditioned_pi05/evaluation/submit_graph_10.sh
#   MAX_CONCURRENT=2 bash experiments/graph_conditioned_pi05/evaluation/submit_graph_10.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EVAL_CONDITION_MODE=graph_only
export TASK_CONFIG=bench_demo_kitchenl_d10

TASK_NAME=put_sauce_can_in_basket
START_INDEX=0
NUM_SEEDS=10
MAX_CONCURRENT="${MAX_CONCURRENT:-2}"

exec "$SCRIPT_DIR/submit_paired_10.sh" \
  "$TASK_NAME" "$START_INDEX" "$NUM_SEEDS" "$MAX_CONCURRENT"
