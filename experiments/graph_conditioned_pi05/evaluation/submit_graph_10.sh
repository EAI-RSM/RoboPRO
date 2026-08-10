#!/usr/bin/env bash

# Submit only the retrieved-graph condition for the fixed evaluation seeds.
# Arguments match submit_paired_10.sh: TASK START_INDEX NUM_SEEDS MAX_CONCURRENT.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EVAL_CONDITION_MODE=graph_only
exec "$SCRIPT_DIR/submit_paired_10.sh" "$@"
