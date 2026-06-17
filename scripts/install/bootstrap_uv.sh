#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found on PATH." >&2
    exit 1
fi

echo "Creating/updating .venv with Python ${PYTHON_VERSION} ..."
uv venv --python "$PYTHON_VERSION"

echo "Syncing Python dependencies from pyproject.toml ..."
uv sync --locked

echo "Running RoboPRO post-install patches ..."
source .venv/bin/activate
cd customized_robotwin
SKIP_BASE_DEPS=1 bash script/_install.sh

cat <<'EOF'

uv bootstrap complete.

Next steps:
1. Download assets: python scripts/install/download_assets.py
2. Create the asset symlink and CuRobo config files described in README.md
3. Activate the environment when working interactively: source .venv/bin/activate
EOF
