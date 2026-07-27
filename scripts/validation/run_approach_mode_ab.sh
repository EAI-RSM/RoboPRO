#!/usr/bin/env bash
# ============================================================================
# Phase 4: APPROACH_MODE A/B/C  --  does the clearance-metric seed help?
#
# Three cells, same seed set, same scene, one variable each step:
#   off     stock around-box waypoint  (today's expert; scene-specific heuristic)
#   direct  waypoints OFF, pre_grasp planned straight from rest, NO seed
#   seed    waypoints OFF, pre_grasp planned straight from rest, WITH the seed
#
# The comparison that attributes the SEED is direct -> seed: those two differ by
# exactly one thing. off -> direct measures what the hand-tuned waypoint was worth,
# i.e. how far the honest generalization floor sits below today's number. Neither
# direct nor seed falls back to the waypoint, so the floor is not contaminated by
# the heuristic (see SEED_TRAJECTORY_PLAN.md sec. "Phase 3+").
#
# Pass-1 seed acceptance (visibility / stability / pad-blocked) does not touch the
# approach mode, so all three cells see the SAME scenes -> the summary pairs them.
#
# Fully unattended: each cell is its own process, so one crashing does not abort the
# others; every cell is tee'd to its own log; videos + HDF5 + records.jsonl per cell.
#
# Usage (all optional):
#   NUM_SEEDS=50 bash scripts/validation/run_approach_mode_ab.sh
#   MODES="direct seed" NUM_SEEDS=30 bash scripts/validation/run_approach_mode_ab.sh
#
# Parameters (override via env):
#   NUM_SEEDS        seeds per cell                  (default 50)
#   SEED_START       first seed                      (default 0)
#   MODES            which cells to run              (default "off direct seed")
#   OFFSET           occluder offset                 (default 0.2)
#   CLUTTER_DENSITY  table clutter                   (default 0)
#   BASE_CONFIG      bench task config               (default bench_demo_office_clean)
#   OUT_ROOT         results dir                     (default results/phase4_approach_mode/<stamp>)
#   SEED_VISUALS     1/0, save route figures in seed (default 1)
#   DEBUG            1 -> ROBOTWIN_LOG_MOVE trace    (default 0)
# Frozen curobo knobs -- identical in all cells, so the ONLY variable is the mode:
#   CUROBO_MAX_ATTEMPTS(24) CUROBO_TRAJOPT_SEEDS(16) CUROBO_BATCH_GRAPH_SEEDS(1)
#   CUROBO_ATTACH_SPHERE_RADIUS(0.001)
#
# NOTE: the seed cell pays a clearance-metric build per (scene, arm) -- tens of
# seconds -- so it is the slowest cell by construction. That cost is real and is
# reported as seconds/rollout and usable-samples/hour, not hidden.
# ============================================================================
set -uo pipefail

# ---- Parameters ----
NUM_SEEDS="${NUM_SEEDS:-50}"
SEED_START="${SEED_START:-0}"
MODES="${MODES:-off direct seed}"
OFFSET="${OFFSET:-0.2}"
CLUTTER_DENSITY="${CLUTTER_DENSITY:-0}"
BASE_CONFIG="${BASE_CONFIG:-bench_demo_office_clean}"
DEBUG="${DEBUG:-0}"

# ---- Frozen shared curobo knobs (identical for ALL cells) ----
export CUROBO_TRAJOPT_SEEDS="${CUROBO_TRAJOPT_SEEDS:-16}"
export CUROBO_MAX_ATTEMPTS="${CUROBO_MAX_ATTEMPTS:-24}"
export CUROBO_BATCH_GRAPH_SEEDS="${CUROBO_BATCH_GRAPH_SEEDS:-1}"
export CUROBO_ATTACH_SPHERE_RADIUS="${CUROBO_ATTACH_SPHERE_RADIUS:-0.001}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CUSTOMIZED="$REPO_ROOT/customized_robotwin"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/scripts/validation/results/phase4_approach_mode/$STAMP}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python interpreter not found at $PYTHON (set PYTHON=...)" >&2
  exit 1
fi

# Guard against the vendored-curobo seed patch having been reverted (envs/curobo is
# gitignored, so a reinstall silently drops it). Without it the seed is accepted and
# then IGNORED -- the 'seed' cell would quietly measure 'direct' and the A/B would
# report a true null for the wrong reason. Fail loudly instead.
if ! grep -q "seed_traj" "$CUSTOMIZED/envs/curobo/src/curobo/wrap/reacher/motion_gen.py" 2>/dev/null; then
  echo "ERROR: vendored curobo is missing the seed_traj patch -- the 'seed' cell would" >&2
  echo "       silently run unseeded. Re-apply it with:" >&2
  echo "  git -C $CUSTOMIZED/envs/curobo apply script/bench_script/curobo_seed_traj.patch" >&2
  exit 1
fi

echo "============================================================"
echo " Phase 4: APPROACH_MODE A/B/C  (does the clearance seed help?)"
echo "   modes        : $MODES"
echo "   seeds        : $NUM_SEEDS (from $SEED_START)"
echo "   scene        : occluder ON, offset=$OFFSET, clutter=$CLUTTER_DENSITY"
echo "   base_config  : $BASE_CONFIG"
echo "   frozen knobs : MAX_ATTEMPTS=$CUROBO_MAX_ATTEMPTS TRAJOPT_SEEDS=$CUROBO_TRAJOPT_SEEDS"
echo "                  BATCH_GRAPH_SEEDS=$CUROBO_BATCH_GRAPH_SEEDS ATTACH_SPHERE=$CUROBO_ATTACH_SPHERE_RADIUS"
echo "   out_root     : $OUT_ROOT"
echo "============================================================"

cd "$CUSTOMIZED"
# shellcheck disable=SC1091
source set_env.sh
export ROBOTWIN_BENCH_TASK=bench
export SEED_VISUALS="${SEED_VISUALS:-1}"
if [[ "$DEBUG" == "1" ]]; then
  export ROBOTWIN_LOG_MOVE=1
  echo "DEBUG=1 -> verbose per-move planning trace enabled in cell logs"
fi

# One cell = one full analyze_occluder_visibility.py run under one APPROACH_MODE.
# A non-zero exit (often just the post-run plotting hiccuping AFTER records.jsonl is
# written) is logged and we move on -- the summary reads whatever records.jsonl exists.
run_cell () {
  local mode="$1"
  local logf="$LOG_DIR/${mode}.log"
  echo ""
  echo ">>> [$(date +%T)] START  APPROACH_MODE=$mode"
  # -u: stdout is redirected to a file, so without it Python block-buffers and the cell
  # log stays EMPTY for hours on a 50-seed run. Unbuffered keeps `tail -f` live.
  if APPROACH_MODE="$mode" "$PYTHON" -u script/bench_script/analyze_occluder_visibility.py \
        --base-config "$BASE_CONFIG" --seed-start "$SEED_START" --num-seeds "$NUM_SEEDS" \
        --rollout --out-dir "$OUT_ROOT" --run-type "$mode" \
        --offsets "$OFFSET" --clutter-densities "$CLUTTER_DENSITY" --no-occluder-prob 0 \
        >"$logf" 2>&1; then
    echo "<<< [$(date +%T)] DONE   $mode   (log: $logf)"
  else
    local rc=$?
    echo "!!! [$(date +%T)] FAILED $mode (exit $rc) -- see $logf (records.jsonl may still be complete)"
  fi
}

for mode in $MODES; do
  case "$mode" in
    off|direct|seed) run_cell "$mode" ;;
    *) echo "!!! unknown mode '$mode' (expected off|direct|seed) -- skipping" ;;
  esac
done

# ---- Paired summary ----
echo ""
echo ">>> [$(date +%T)] Summarizing ..."
"$PYTHON" "$SCRIPT_DIR/summarize_approach_mode_ab.py" --root "$OUT_ROOT" \
  2>&1 | tee "$OUT_ROOT/summary.txt"

echo ""
echo "All done. Results under: $OUT_ROOT"
echo "  per cell   : $OUT_ROOT/<mode>/<timestamp>/records.jsonl"
echo "  videos     : $OUT_ROOT/<mode>/<timestamp>/{success,fail}/video/"
echo "  seed routes: $OUT_ROOT/seed/<timestamp>/seed_route_visuals/episode<N>_<arm>/"
echo "  summary    : $OUT_ROOT/summary.txt"
