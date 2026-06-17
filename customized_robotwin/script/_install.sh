#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: python not found on PATH" >&2; exit 1; }
PIP_CMD=("$PYTHON_BIN" -m pip)

echo "Installing the necessary packages ..."
if [ "${SKIP_BASE_DEPS:-0}" != "1" ]; then
    "${PIP_CMD[@]}" install -r script/requirements.txt

    echo "Installing pytorch3d ..."
    # cd third_party/pytorch3d_simplified
    # pip install -e .
    # cd ../..
    "${PIP_CMD[@]}" install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
fi

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$("$PYTHON_BIN" -m pip show sapien | grep 'Location' | awk '{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
grep -q 'with open(urdf_file, "r") as f:' "$URDF_LOADER" || { echo "ERROR: sapien patch target not found in $URDF_LOADER" >&2; exit 1; }
grep -q 'srdf_file = urdf_file\[:-4\] + "srdf"' "$URDF_LOADER" || { echo "ERROR: sapien srdf patch target not found in $URDF_LOADER" >&2; exit 1; }
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER


echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$("$PYTHON_BIN" -m pip show mplib | grep 'Location' | awk '{print $2}')/mplib

# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
grep -q 'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:' "$PLANNER" || { echo "ERROR: mplib patch target not found in $PLANNER" >&2; exit 1; }
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

echo "Installing Curobo ..."
cd envs
CUROBO_TAG="v0.7.8"
if [ ! -d curobo ] || ! git -C curobo describe --tags --exact-match 2>/dev/null | grep -q "^${CUROBO_TAG}$"; then
    rm -rf curobo
    git clone --branch "${CUROBO_TAG}" --depth 1 https://github.com/NVlabs/curobo.git
fi
cd curobo
"${PIP_CMD[@]}" install -e . --no-build-isolation
"${PIP_CMD[@]}" install warp-lang==1.12.0 setuptools==69.5.1
cd ../..

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
echo "See INSTALLATION.md for more instructions."
