#!/usr/bin/env bash
# ============================================================================
# Planner comparison: baseline vs hamid, on the curated (occluder + clutter) scene
#
#   planners:  baseline = dev vanilla grasp->lift->place, no routing
#              hamid    = Hamid's single deployable planner (vendored, scene-paired)
#   scene:     curated  = milk-box occluder + BOTTLE target + standard table clutter
#
# Runs both cells sequentially, each in its own process so one cell crashing does
# NOT abort the other, tees every cell to its own log, saves rollout VIDEOS + HDF5
# + records.jsonl per cell, then prints a paired success-rate + Wilson-CI + McNemar
# summary.
#
# Shared curobo knobs are FROZEN identically across both cells, so the only
# variable is the planner. Seed acceptance is planner-independent, so both planners
# see the SAME seeds -> paired comparison.
#
# (Repurposed from the old 2x2 baseline-vs-reachability x scene grid; typical/mouse
#  scene is left plumbed but off by default. Override ALGO_A/ALGO_B/SCENES to revive.)
#
# Usage (all optional; sensible defaults):
#   NUM_SEEDS=50 CLUTTER_DENSITY=10 bash scripts/validation/run_2x2_planner_comparison.sh
#
# Key parameters (override via env):
#   NUM_SEEDS         seeds per cell            (default 50)
#   SEED_START        first seed                (default 0)
#   CLUTTER_DENSITY   obstacle density, typical (default 10)
#   OFFSET            occluder offset, curated  (default 0.2)
#   BASE_CONFIG       bench task config         (default bench_demo_office_clean)
#   OUT_ROOT          results dir               (default results/phase2_2x2/<timestamp>)
# Frozen curobo knobs (override only to redefine the whole experiment):
#   CUROBO_MAX_ATTEMPTS(24) CUROBO_TRAJOPT_SEEDS(16) CUROBO_BATCH_GRAPH_SEEDS(1)
#   CUROBO_ATTACH_SPHERE_RADIUS(0.001)  [finetune knobs left unset = curobo default]
# ============================================================================
set -uo pipefail

# ---- Parameters ----
NUM_SEEDS="${NUM_SEEDS:-50}"
SEED_START="${SEED_START:-0}"
CLUTTER_DENSITY="${CLUTTER_DENSITY:-10}"                 # clutter for the typical (mouse) scene
# Curated scene now ALSO spawns clutter (milk box + bottle + clutter). Defaults to the same
# density as the typical scene; set CURATED_CLUTTER_DENSITY=0 to go back to box-only.
CURATED_CLUTTER_DENSITY="${CURATED_CLUTTER_DENSITY:-$CLUTTER_DENSITY}"
OFFSET="${OFFSET:-0.2}"
BASE_CONFIG="${BASE_CONFIG:-bench_demo_office_clean}"
# DEBUG=1 -> verbose per-move planning trace (ROBOTWIN_LOG_MOVE) in every cell log. Off by
# default so big runs aren't flooded; turn on for small diagnostic runs.
DEBUG="${DEBUG:-0}"
# Which scenes to run. Default curated only (occluder + clutter); set SCENES="curated typical"
# to also run the mouse/clutter scene.
SCENES="${SCENES:-curated}"
# The two planners being compared (paired within each scene). Default baseline vs hamid.
ALGO_A="${ALGO_A:-baseline}"
ALGO_B="${ALGO_B:-hamid}"

# ---- Frozen shared curobo knobs (identical for ALL four cells) ----
export CUROBO_TRAJOPT_SEEDS="${CUROBO_TRAJOPT_SEEDS:-16}"
export CUROBO_MAX_ATTEMPTS="${CUROBO_MAX_ATTEMPTS:-24}"
export CUROBO_BATCH_GRAPH_SEEDS="${CUROBO_BATCH_GRAPH_SEEDS:-1}"
export CUROBO_ATTACH_SPHERE_RADIUS="${CUROBO_ATTACH_SPHERE_RADIUS:-0.001}"
# finetune knobs deliberately left UNSET -> CuRobo's own default (matches dev)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CUSTOMIZED="$REPO_ROOT/customized_robotwin"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
STAMP="$(date +%Y-%m-%d-%H-%M-%S)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/scripts/validation/results/phase2_2x2/$STAMP}"
LOG_DIR="$OUT_ROOT/logs"
CURATED_OUT="$OUT_ROOT/curated"   # script appends _rollout/<plan_algo>
TYPICAL_OUT="$OUT_ROOT/typical"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python interpreter not found at $PYTHON (set PYTHON=...)" >&2
  exit 1
fi

echo "============================================================"
echo " planner comparison: $ALGO_A vs $ALGO_B"
echo "   seeds        : $NUM_SEEDS (from $SEED_START)"
echo "   curated      : occluder ON, offset=$OFFSET, clutter=$CURATED_CLUTTER_DENSITY, bottle target"
echo "   typical      : occluder OFF, clutter_density=$CLUTTER_DENSITY, mouse target"
echo "   scenes       : $SCENES"
echo "   debug trace  : $([[ "$DEBUG" == "1" ]] && echo ON || echo off)"
echo "   base_config  : $BASE_CONFIG"
echo "   frozen knobs : MAX_ATTEMPTS=$CUROBO_MAX_ATTEMPTS TRAJOPT_SEEDS=$CUROBO_TRAJOPT_SEEDS"
echo "                  BATCH_GRAPH_SEEDS=$CUROBO_BATCH_GRAPH_SEEDS ATTACH_SPHERE=$CUROBO_ATTACH_SPHERE_RADIUS"
echo "   out_root     : $OUT_ROOT"
echo "============================================================"

cd "$CUSTOMIZED"
# shellcheck disable=SC1091
source set_env.sh
export ROBOTWIN_BENCH_TASK=bench
# Persistent per-move planning trace for post-hoc debugging (which subgoal/plan failed).
if [[ "$DEBUG" == "1" ]]; then
  export ROBOTWIN_LOG_MOVE=1
  echo "DEBUG=1 -> verbose per-move planning trace enabled in cell logs"
fi

# One cell = one full analyze_occluder_visibility.py run. A non-zero exit (often just
# the post-run plotting hiccuping AFTER records.jsonl is already written) is logged and
# we move on -- the summary reads whatever records.jsonl exists.
run_cell () {
  local name="$1"; shift
  local logf="$LOG_DIR/${name}.log"
  echo ""
  echo ">>> [$(date +%T)] START  $name"
  if "$PYTHON" script/bench_script/analyze_occluder_visibility.py "$@" >"$logf" 2>&1; then
    echo "<<< [$(date +%T)] DONE   $name   (log: $logf)"
  else
    local rc=$?
    echo "!!! [$(date +%T)] FAILED $name (exit $rc) -- see $logf (records.jsonl may still be complete)"
  fi
}

COMMON=(--base-config "$BASE_CONFIG" --seed-start "$SEED_START" --num-seeds "$NUM_SEEDS" --rollout)

scene_enabled () { [[ " $SCENES " == *" $1 "* ]]; }

# ---- Curated scene (bottle + milk box + clutter): occluder ALWAYS on ----
if scene_enabled curated; then
  run_cell "curated_$ALGO_A" "${COMMON[@]}" --out-dir "$CURATED_OUT" \
    --offsets "$OFFSET" --clutter-densities "$CURATED_CLUTTER_DENSITY" --no-occluder-prob 0 --plan-algo "$ALGO_A"
  run_cell "curated_$ALGO_B" "${COMMON[@]}" --out-dir "$CURATED_OUT" \
    --offsets "$OFFSET" --clutter-densities "$CURATED_CLUTTER_DENSITY" --no-occluder-prob 0 --plan-algo "$ALGO_B"
fi

# ---- Typical scene (mouse + clutter): occluder ALWAYS off, clutter on ----
if scene_enabled typical; then
  run_cell "typical_$ALGO_A" "${COMMON[@]}" --out-dir "$TYPICAL_OUT" \
    --offsets "$OFFSET" --clutter-densities "$CLUTTER_DENSITY" --no-occluder-prob 1 --plan-algo "$ALGO_A"
  run_cell "typical_$ALGO_B" "${COMMON[@]}" --out-dir "$TYPICAL_OUT" \
    --offsets "$OFFSET" --clutter-densities "$CLUTTER_DENSITY" --no-occluder-prob 1 --plan-algo "$ALGO_B"
fi

# ---- Paired summary ----
echo ""
echo ">>> [$(date +%T)] Summarizing ..."
"$PYTHON" "$SCRIPT_DIR/summarize_2x2_planner_comparison.py" --root "$OUT_ROOT" \
  --scenes $SCENES --algos "$ALGO_A" "$ALGO_B" \
  2>&1 | tee "$OUT_ROOT/summary.txt"

echo ""
echo "All done. Results under: $OUT_ROOT"
echo "  videos/hdf5 (bucketed): <scene>_rollout/<planner>/{success,fail}/"
echo "  summary               : $OUT_ROOT/summary.txt"
