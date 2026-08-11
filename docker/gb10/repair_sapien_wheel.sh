#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <nightly-wheel> <stable-wheel> <output-dir>" >&2
  exit 2
fi

NIGHTLY_WHEEL=$(realpath "$1")
STABLE_WHEEL=$(realpath "$2")
OUTPUT_DIR=$(realpath -m "$3")
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$OUTPUT_DIR" "$WORK_DIR/nightly" "$WORK_DIR/stable"

python -m wheel unpack --dest "$WORK_DIR/nightly" "$NIGHTLY_WHEEL"
python -m wheel unpack --dest "$WORK_DIR/stable" "$STABLE_WHEEL"
nightly_root=$(find "$WORK_DIR/nightly" -mindepth 1 -maxdepth 1 -type d -print -quit)
stable_root=$(find "$WORK_DIR/stable" -mindepth 1 -maxdepth 1 -type d -print -quit)

rm -rf "$nightly_root/sapien/oidn_library"
cp -a "$stable_root/sapien/oidn_library" "$nightly_root/sapien/oidn_library"

while IFS= read -r -d '' library; do
  if readelf -d "$library" 2>/dev/null | grep -q 'librt-2-4cc790af.28.so'; then
    patchelf --replace-needed librt-2-4cc790af.28.so librt.so.1 "$library"
  fi
  if readelf -d "$library" 2>/dev/null | grep -q 'libstdc++-bf6214da.so.6.0.25'; then
    patchelf --replace-needed libstdc++-bf6214da.so.6.0.25 libstdc++.so.6 "$library"
  fi
  if readelf -d "$library" 2>/dev/null | grep -q 'libgcc_s-8-20210514-4d0aed74.so.1'; then
    patchelf --replace-needed libgcc_s-8-20210514-4d0aed74.so.1 libgcc_s.so.1 "$library"
  fi
done < <(find "$nightly_root/sapien.libs" "$nightly_root/sapien" -type f -name '*.so*' -print0)

rm -f \
  "$nightly_root/sapien.libs/librt-2-4cc790af.28.so" \
  "$nightly_root/sapien.libs/libstdc++-bf6214da.so.6.0.25" \
  "$nightly_root/sapien.libs/libgcc_s-8-20210514-4d0aed74.so.1"

python -m wheel pack --dest-dir "$OUTPUT_DIR" "$nightly_root"
repaired=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'sapien-*.whl' -type f -print -quit)
unzip -t "$repaired" >/dev/null
sha256sum "$repaired"
