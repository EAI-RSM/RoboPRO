#!/usr/bin/env bash
set -euo pipefail

WHEELHOUSE="${1:-/wheelhouse}"
CMEEL_PREFIX="${CMEEL_PREFIX:-/usr/local/lib/python3.12/dist-packages/cmeel.prefix}"
FCL_PREFIX="${FCL_PREFIX:-/opt/fcl-cmeel}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/robopro-arm64-build}"

mkdir -p "$WHEELHOUSE" "$BUILD_ROOT"

curl -fsSL --retry 5 \
  https://github.com/flexible-collision-library/fcl/archive/refs/tags/0.7.0.tar.gz \
  -o "$BUILD_ROOT/fcl-0.7.0.tar.gz"
rm -rf "$BUILD_ROOT/fcl-0.7.0" "$BUILD_ROOT/fcl-build"
tar -xzf "$BUILD_ROOT/fcl-0.7.0.tar.gz" -C "$BUILD_ROOT"

cmake \
  -S "$BUILD_ROOT/fcl-0.7.0" \
  -B "$BUILD_ROOT/fcl-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$FCL_PREFIX" \
  -DCMAKE_PREFIX_PATH="$CMEEL_PREFIX" \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON
cmake --build "$BUILD_ROOT/fcl-build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
cmake --install "$BUILD_ROOT/fcl-build"

rm -rf "$BUILD_ROOT/mplib"
git clone --filter=blob:none --no-checkout \
  https://github.com/haosulab/MPlib.git "$BUILD_ROOT/mplib"
git config --global --add safe.directory "$BUILD_ROOT/mplib"

cd "$BUILD_ROOT/mplib"
git fetch --force --depth 1 origin refs/tags/v0.2.1:refs/tags/v0.2.1
git checkout --detach refs/tags/v0.2.1
git submodule update --init --recursive --depth 1
test "$(git describe --tags --exact-match HEAD)" = v0.2.1
export CMAKE_PREFIX_PATH="$FCL_PREFIX:$CMEEL_PREFIX"
export LD_LIBRARY_PATH="$FCL_PREFIX/lib:$CMEEL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LLVM_DIR=/usr/lib/llvm-18
export LIBCLANG_PATH=/usr/lib/llvm-18/lib/libclang.so
export CMAKE_ARGS=-DCMAKE_CXX_FLAGS=-Wno-error=maybe-uninitialized
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
PIP_CONSTRAINT= python -m build \
  --wheel \
  --no-isolation \
  --skip-dependency-check \
  --outdir "$WHEELHOUSE"

test -f "$WHEELHOUSE/mplib-0.2.1-cp312-cp312-linux_aarch64.whl"

pymp=$(find build -name 'pymp*.so' -type f -print -quit)
readelf -d "$pymp" | grep -q 'liboctomap.so.1.10'
if readelf -d "$pymp" | grep -q 'liboctomap.so.1.9'; then
  echo 'ERROR: mplib linked against mixed OctoMap ABIs' >&2
  exit 1
fi
