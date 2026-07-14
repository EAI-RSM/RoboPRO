# Building SAPIEN 3.0.0b1 from source on aarch64 (GB10 / DGX Spark)

PyPI ships no aarch64 wheel for `sapien==3.0.0b1`, so on ARM machines (e.g. NVIDIA GB10
/ DGX Spark) the `pip install -r script/requirements.txt` step fails to resolve sapien.
The fix is to build the wheel from source. The script below automates the whole build,
including the patches needed for aarch64 and for the GB10's sm_121 GPU (Blackwell,
PTX-compatible with sm_120).

## Prerequisites

- `gcc-11` / `g++-11` (`sudo apt install gcc-11 g++-11`) — newer GCC breaks the vendored deps
- CUDA 12.x toolkit (nvcc), `cmake`, `git`, `curl`, `unzip`
- The `robopro` conda env activated (the wheel is built against its Python)
- Two variables set before running:

```bash
THIRD_PARTY_DIR=/path/to/third_party    # where the SAPIEN source tree will live
CUDA_12=/usr/local/cuda-12.4            # your CUDA 12.x install root
```

## Build script

```bash
# ── SAPIEN 3.0.0b1 (build from source for aarch64) ───────────────────────────
echo "==> Building SAPIEN 3.0.0b1 from source (aarch64)..."
SAPIEN_SRC="${THIRD_PARTY_DIR}/sapien_build"
rm -rf "${SAPIEN_SRC}"
git clone --branch 3.0.0b1 --depth 1 https://github.com/haosulab/SAPIEN.git "${SAPIEN_SRC}"
cd "${SAPIEN_SRC}"
git submodule update --init --recursive

# Build SAPIEN in a subshell to isolate env var exports from later sections
(
  export CUDA_PATH="${CUDA_12}"
  export CUDACXX="${CUDA_12}/bin/nvcc"
  export PATH="${CUDA_12}/bin:${PATH}"
  export CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)"
  export CUDAARCHS="120"
  export CC=/usr/bin/gcc-11
  export CXX=/usr/bin/g++-11
  export CUDAHOSTCXX=/usr/bin/g++-11
  export CFLAGS="-w"
  export CXXFLAGS="-w"
  export CUDAFLAGS="-w -ccbin /usr/bin/g++-11"
  unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH
  SAPIEN_BUILD_DIR="sapien_build"

  # Patch hardcoded CUDA architectures
  sed -i 's/CUDA_ARCHITECTURES "60;61;70;75;80;86"/CUDA_ARCHITECTURES "120"/g' 3rd_party/simsense/CMakeLists.txt
  sed -i 's/CUDA_ARCHITECTURES "60;61;70;75;80;86"/CUDA_ARCHITECTURES "120"/g' CMakeLists.txt

  # Patch pybind11 refs (smart_holder branch was archived)
  for f in cmake/pybind11.cmake pinocchio/cmake/pybind11.cmake; do
    if [[ -f "$f" ]]; then
      sed -i 's/GIT_TAG smart_holder$/GIT_TAG archive\/smart_holder/' "$f"
    fi
  done

  # First build: fetch dependencies (may fail, that's OK)
  if [[ ! -d "${SAPIEN_BUILD_DIR}/_sapien_deps/ktx-src" ]]; then
    echo "==> Running cmake configure to fetch dependencies..."
    python setup.py bdist_wheel 2>&1 | tee "${SAPIEN_SRC}/sapien_build_log.txt" || true
  fi

  # Patch pybind11 again (SAPIEN's own copy, fetched during first build)
  for f in cmake/pybind11.cmake pinocchio/cmake/pybind11.cmake; do
    if [[ -f "$f" ]]; then
      sed -i 's/GIT_TAG smart_holder$/GIT_TAG archive\/smart_holder/' "$f"
    fi
  done

  # Patch ASTC encoder NEON intrinsics for GCC on aarch64
  NEON_FILE="${SAPIEN_BUILD_DIR}/_sapien_deps/ktx-src/lib/astc-encoder/Source/astcenc_vecmathlib_neon_4.h"
  if [[ -f "${NEON_FILE}" ]]; then
    echo "==> Patching ASTC NEON intrinsics for aarch64/GCC..."
    sed -i '/uint32_t lane/,/}/s/return vgetq_lane_s32(m, l);/return vgetq_lane_u32(m, l);/' "${NEON_FILE}"
    sed -i 's/int8x16_t table { t0\.m };/int8x16_t table = vreinterpretq_s8_s32(t0.m);/' "${NEON_FILE}"
    sed -i 's/int8x16x2_t table { t0\.m, t1\.m };/int8x16x2_t table { vreinterpretq_s8_s32(t0.m), vreinterpretq_s8_s32(t1.m) };/' "${NEON_FILE}"
    sed -i 's/int8x16x4_t table { t0\.m, t1\.m, t2\.m, t3\.m };/int8x16x4_t table { vreinterpretq_s8_s32(t0.m), vreinterpretq_s8_s32(t1.m), vreinterpretq_s8_s32(t2.m), vreinterpretq_s8_s32(t3.m) };/' "${NEON_FILE}"
    sed -i 's/\tint8x16_t idx_bytes = vreinterpretq_u8_s32(idx_masked);/\tuint8x16_t idx_bytes = vreinterpretq_u8_s32(idx_masked);/g' "${NEON_FILE}"
    sed -i 's/return vint4(vqtbl1q_s8(table, idx_bytes));/return vint4(vreinterpretq_s32_s8(vqtbl1q_s8(table, idx_bytes)));/' "${NEON_FILE}"
    sed -i 's/return vint4(vqtbl2q_s8(table, idx_bytes));/return vint4(vreinterpretq_s32_s8(vqtbl2q_s8(table, idx_bytes)));/' "${NEON_FILE}"
    sed -i 's/return vint4(vqtbl4q_s8(table, idx_bytes));/return vint4(vreinterpretq_s32_s8(vqtbl4q_s8(table, idx_bytes)));/' "${NEON_FILE}"
  fi

  # Patch OIDN to add sm_120 CUDA support (GB10 is sm_121, PTX-compatible with sm_120)
  OIDN_CUDA_CMAKE="${SAPIEN_BUILD_DIR}/_sapien_deps/oidn-src/devices/cuda/CMakeLists.txt"
  if [[ -f "${OIDN_CUDA_CMAKE}" ]] && ! grep -q SM120 "${OIDN_CUDA_CMAKE}"; then
    echo "==> Patching OIDN for sm_120 (GB10)..."
    sed -i '/^set(OIDN_NVCC_SM90_FLAGS/a\set(OIDN_NVCC_SM120_FLAGS "-gencode arch=compute_120,code=sm_120")' "${OIDN_CUDA_CMAKE}"
    sed -i 's/\(SM90_FLAGS}\)"/\1 ${OIDN_NVCC_SM120_FLAGS}"/' "${OIDN_CUDA_CMAKE}"
  fi

  # Patch OIDN maxSMArch to support sm_120+ (default 99 is too low for GB10)
  OIDN_CUDA_DEVICE_H="${SAPIEN_BUILD_DIR}/_sapien_deps/oidn-src/devices/cuda/cuda_device.h"
  if [[ -f "${OIDN_CUDA_DEVICE_H}" ]] && ! grep -q 'maxSMArch = 121' "${OIDN_CUDA_DEVICE_H}"; then
    echo "==> Patching OIDN maxSMArch 99 -> 121 for GB10..."
    sed -i 's/static constexpr int maxSMArch = 99;/static constexpr int maxSMArch = 121;/' "${OIDN_CUDA_DEVICE_H}"
  fi

  # Replace x86_64 PhysX5 precompiled libs with aarch64 version
  PHYSX_VERSION="105.1-physx-5.3.1.patch0"
  PHYSX_DIR="${SAPIEN_BUILD_DIR}/_sapien_deps/physx5-src"
  if [[ -d "${PHYSX_DIR}" ]]; then
    echo "==> Replacing PhysX5 libs with aarch64 build..."
    curl -sL "https://github.com/sapien-sim/physx-precompiled/releases/download/${PHYSX_VERSION}/linux-aarch64-release.zip" -o "${SAPIEN_SRC}/physx5-aarch64.zip"
    rm -rf "${SAPIEN_SRC}/physx5-aarch64"
    unzip -q "${SAPIEN_SRC}/physx5-aarch64.zip" -d "${SAPIEN_SRC}/physx5-aarch64"
    rm -rf "${PHYSX_DIR}/bin/linux.clang"
    mkdir -p "${PHYSX_DIR}/bin/linux.clang"
    ln -s "${SAPIEN_SRC}/physx5-aarch64/bin/linux.aarch64/release" "${PHYSX_DIR}/bin/linux.clang/release"
  fi

  # Clean OIDN build cache (must rebuild with new arch) and rebuild
  rm -rf "${SAPIEN_BUILD_DIR}/_sapien_build"
  rm -rf "${SAPIEN_BUILD_DIR}/_sapien_deps/oidn-build"
  rm -rf dist/
  python setup.py bdist_wheel > "${SAPIEN_SRC}/sapien_build_log.txt" 2>&1 || {
    echo "==> SAPIEN build failed. Errors:"
    grep -E "^.*error:" "${SAPIEN_SRC}/sapien_build_log.txt" | grep -iv "warning\|hmderrors\|PxError\|PxDefault\|codecvt_error" | head -20
    echo "    Full log: ${SAPIEN_SRC}/sapien_build_log.txt"
    exit 1
  }
  echo "==> SAPIEN build succeeded."
)
# Install the built wheel outside the subshell
pip install "${SAPIEN_SRC}/dist"/sapien-*.whl
```

The build runs twice by design: the first `python setup.py bdist_wheel` only exists to
make CMake fetch the vendored dependencies (ktx, OIDN, PhysX5, pybind11) into
`sapien_build/_sapien_deps/`, and is allowed to fail. The patches are then applied to
those fetched sources, and the second build produces the wheel. If the build fails, the
full log is at `${SAPIEN_SRC}/sapien_build_log.txt`.

After a successful `pip install`, verify with:

```bash
python -c "import sapien; print(sapien.__version__)"
```

## Troubleshooting

Each patch in the script exists because of a real build failure. If you hit one of
these errors, the corresponding patch did not apply (e.g. the file path changed in a
newer SAPIEN commit) — fix the patch rather than the symptom.

| Error in `sapien_build_log.txt` | Cause | Handled by |
|---|---|---|
| `fatal: Remote branch smart_holder not found` (pybind11 clone fails) | pybind11's `smart_holder` branch was archived to `archive/smart_holder` | pybind11 `sed` patches (applied twice: once to the repo copy, once to the copy CMake fetches during the first build) |
| `nvcc fatal : Unsupported gpu architecture 'compute_86'` or kernels silently built for the wrong GPU | CMakeLists hardcodes `CUDA_ARCHITECTURES "60;61;70;75;80;86"`, which excludes Blackwell | `CUDA_ARCHITECTURES` sed patches + `CUDAARCHS=120` export |
| `error: cannot convert 'int32x4_t' to 'int8x16_t'` (or similar NEON intrinsic type errors) in `astcenc_vecmathlib_neon_4.h` | ASTC encoder's NEON code relies on Clang-only implicit vector conversions; GCC enforces strict types | ASTC NEON `sed` patches |
| OIDN build fails with unsupported gencode, or at runtime `OIDN: unsupported CUDA device` | OIDN's CMake has no sm_120 gencode entry, and `maxSMArch = 99` rejects the GB10 (sm_121) at device init | OIDN CMakeLists + `cuda_device.h` patches |
| Link errors against PhysX (`skipping incompatible ... libPhysX*.a`, undefined `Px*` symbols) | SAPIEN downloads x86_64 PhysX5 precompiled libs | PhysX5 aarch64 lib replacement |
| Random header-not-found or wrong-header errors from vendored deps | A conda env leaking `CPLUS_INCLUDE_PATH` / `C_INCLUDE_PATH` into the build | `unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH` in the subshell |
| C++ errors deep inside vendored deps with GCC 12/13 | The 3.0.0b1 sources don't compile with newer GCC | `CC`/`CXX`/`CUDAHOSTCXX` pinned to gcc-11 |
| OIDN still built for the old arch after re-running the script | Stale CMake cache in `_sapien_build` / `oidn-build` | The `rm -rf` of both build dirs before the final build |

Other things worth knowing:

- **Re-running after a failure:** the script starts with `rm -rf "${SAPIEN_SRC}"`, i.e. a
  full re-clone and re-download. If you're iterating on a single patch, comment out the
  `rm -rf`/`git clone` lines and re-run from the subshell onward.
- **Different GPU arch:** `CUDAARCHS=120` and the sm_120 patches target GB10/Blackwell.
  For another GPU, set `CUDAARCHS` to your arch (e.g. `90` for GH200) and adjust the
  OIDN gencode/`maxSMArch` patches accordingly.
- **`pkg_resources` missing at import time:** sapien 3.0.0b1 imports `pkg_resources`;
  make sure `setuptools==69.5.1` is installed in the env (see main README, step 2).
