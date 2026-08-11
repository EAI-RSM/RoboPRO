#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE="${IMAGE:-robopro:gb10}"
INSTALL_CUROBO="${INSTALL_CUROBO:-1}"

if [[ $(uname -m) != aarch64 ]]; then
  echo "ERROR: this image must be built natively on an aarch64 host" >&2
  exit 1
fi

docker build \
  --file "$ROOT_DIR/docker/gb10/Dockerfile" \
  --build-arg "INSTALL_CUROBO=$INSTALL_CUROBO" \
  --tag "$IMAGE" \
  "$ROOT_DIR"
