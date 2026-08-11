#!/usr/bin/env bash

# Submit only graph-conditioned evaluation for the frozen five-task d10 suite.
# Each task uses its ten expert-validated seeds; visual-only is not rerun.
#
# Usage:
#   bash experiments/graph_conditioned_pi05/evaluation/submit_diverse_graph_5x10.sh
#   MAX_CONCURRENT=2 bash .../submit_diverse_graph_5x10.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CAMPAIGN_CONDITION_MODE=graph_only
exec "$SCRIPT_DIR/submit_diverse_5x10.sh" "$@"
