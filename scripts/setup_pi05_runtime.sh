#!/usr/bin/env bash
# Restore the ignored upstream OpenPI runtime while preserving tracked RoboPRO adapters.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PI05_ROOT="$ROOT_DIR/customized_robotwin/policy/pi05"
UPSTREAM_SNAPSHOT="${ROBOPRO_PI05_SNAPSHOT:-eb48cf558ebb84536896eccf8f1daea95464ec66}"

if ! git -C "$ROOT_DIR" cat-file -e "${UPSTREAM_SNAPSHOT}^{commit}"; then
  echo "ERROR: pi05 source snapshot is unavailable: $UPSTREAM_SNAPSHOT" >&2
  exit 1
fi
mkdir -p "$PI05_ROOT"

# These files are RoboPRO-owned and tracked in the current checkout. Everything
# else is restored from the historical OpenPI snapshot.
git -C "$ROOT_DIR" archive "$UPSTREAM_SNAPSHOT" customized_robotwin/policy/pi05 |   tar -x -C "$ROOT_DIR"     --exclude='customized_robotwin/policy/pi05/__init__.py'     --exclude='customized_robotwin/policy/pi05/deploy_policy.py'     --exclude='customized_robotwin/policy/pi05/deploy_policy.yml'     --exclude='customized_robotwin/policy/pi05/pi_model.py'     --exclude='customized_robotwin/policy/pi05/eval.sh'     --exclude='customized_robotwin/policy/pi05/eval_double_env.sh'     --exclude='customized_robotwin/policy/pi05/collect_rollout.sh'

echo "restored upstream pi05 runtime from $UPSTREAM_SNAPSHOT"
case "${1:-}" in
  --sync)
    command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required for --sync" >&2; exit 1; }
    (cd "$PI05_ROOT" && uv sync)
    ;;
  --sync-gb10)
    command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required for --sync-gb10" >&2; exit 1; }
    GB10_VENV="$PI05_ROOT/.venv-jax083"
    uv venv --python 3.12 "$GB10_VENV"
    uv pip install --python "$GB10_VENV/bin/python" -e "$PI05_ROOT"
    uv pip install --python "$GB10_VENV/bin/python" \
      'jax[cuda13]==0.8.3' 'flax==0.12.6' \
      'orbax-checkpoint==0.11.40' 'numpy==2.5.1'
    ;;
esac

test -f "$PI05_ROOT/src/openpi/training/config.py"
test -f "$PI05_ROOT/pi_model.py"
echo "pi05 runtime ready at $PI05_ROOT"
