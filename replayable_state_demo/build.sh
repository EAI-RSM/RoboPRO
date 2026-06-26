#!/usr/bin/env bash
# Regenerate the whole demo bundle (assets are git-ignored, so build before serving).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../.venv/bin/python}"
[ -x "$PY" ] || PY=python

echo "[1/3] exporting scene (state trace -> web bundle) ..."
"$PY" export_scene.py
echo "[2/3] slimming meshes/textures for the web ..."
"$PY" slim_assets.py
echo "[3/3] fetching three.js runtime ..."
"$PY" fetch_deps.py

cat <<EOF

done. serve it (threaded server -> fast frame streaming):
    "$PY" serve.py 8000
then open  http://localhost:8000   (layered viewer; 3D reconstruction at /scene3d.html)
EOF
