#!/usr/bin/env bash
# ============================================================================
# Phase 4: APPROACH_MODE A/B  --  does the clearance-metric seed help?
#
# Two cells by default, same seed set, same scene, differing by exactly one thing:
#   direct  waypoints OFF, pre_grasp planned straight from rest, NO seed  (the floor)
#   seed    waypoints OFF, pre_grasp planned straight from rest, WITH the seed
#
# direct -> seed is the whole experiment: those two differ by only the seed, so any
# success delta is attributable to it. Neither falls back to the around-box waypoint,
# so a miss fails the candidate and the floor is not contaminated by the heuristic.
#
# The third mode, `off` (stock around-box waypoint), is deliberately NOT in the
# default set. It is a hardcoded one-occluder-in-front heuristic, so on the general
# scenes this benchmark is aiming at it is not a control -- it is a different task.
# Opt in with MODES="off direct seed" if you ever want today's expert as a reference
# number; the summary will then also report off -> direct.
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
# SCENES: the cells run on two scene types, the same NUM_SEEDS each, so the two are
# weighted equally by construction:
#   curated   the olive-oil occluder ring is always spawned (occluder offset OFFSET),
#             clutter at CURATED_CLUTTER_DENSITY (default 0). This is the scene the
#             clearance metric was built for.
#   standard  no occluder; STANDARD_CLUTTER_DENSITY random objects at random positions
#             and yaws from the office obstacle pool (default 8). Generalization test.
#
# The clearance metric now measures against the WHOLE scene obstacle set (clutter as well as
# the occluder ring) -- clearance_metric_3d --obstacles, default "all", which walks the same
# env.collision_list curobo's update_world uses, minus the target and pad. So the seed has
# something to route around on the standard scene too. Set SEED_OBSTACLES=occluders to restore
# the old curated-ring-only field (eps* is NOT comparable across the two).
#
# Parameters (override via env):
#   NUM_SEEDS                 seeds per cell, per scene   (default 50)
#   SEED_START                first seed                  (default 0)
#   MODES                     which modes to run          (default "direct seed")
#   SCENES                    which scenes to run         (default "curated standard")
#   OFFSET                    occluder offset, curated    (default 0.2)
#   CURATED_CLUTTER_DENSITY   clutter on curated          (default 0)
#   STANDARD_CLUTTER_DENSITY  clutter on standard         (default 8)
#   BASE_CONFIG               bench task config           (default bench_demo_office_clean)
#   OUT_ROOT                  results dir                 (default results/phase4_approach_mode/<stamp>)
#   SEED_VISUALS              1/0, save route figures     (default 1)
#   SEED_OBSTACLES            all | occluders             (default all)
#   DEBUG                     1 -> ROBOTWIN_LOG_MOVE      (default 0)
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
MODES="${MODES:-direct seed}"
SCENES="${SCENES:-curated standard}"
OFFSET="${OFFSET:-0.2}"
# Both scenes draw from the SAME seed range, so each scene gets NUM_SEEDS attempts and
# the two are equally weighted. Seed acceptance is scene-dependent (a scene is rejected
# for instability / a blocked pad), so the number of accepted seeds can still differ
# between scenes -- the summary reports each cell's own n and pairs within a scene.
CURATED_CLUTTER_DENSITY="${CURATED_CLUTTER_DENSITY:-0}"
STANDARD_CLUTTER_DENSITY="${STANDARD_CLUTTER_DENSITY:-8}"
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
echo " Phase 4: APPROACH_MODE A/B  (does the clearance seed help?)"
echo "   modes        : $MODES"
echo "   scenes       : $SCENES"
echo "   seeds        : $NUM_SEEDS per cell (from $SEED_START) -- equal across scenes"
echo "   curated      : occluder ON, offset=$OFFSET, clutter=$CURATED_CLUTTER_DENSITY"
echo "   standard     : occluder OFF, clutter=$STANDARD_CLUTTER_DENSITY (random objects/positions)"
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
export SEED_OBSTACLES="${SEED_OBSTACLES:-all}"
if [[ "$DEBUG" == "1" ]]; then
  export ROBOTWIN_LOG_MOVE=1
  echo "DEBUG=1 -> verbose per-move planning trace enabled in cell logs"
fi

# One cell = one full analyze_occluder_visibility.py run at one (scene, mode). Results
# land in <OUT_ROOT>/<scene>/<mode>/<timestamp>/, which is what the summary walks.
# A non-zero exit (often just the post-run plotting hiccuping AFTER records.jsonl is
# written) is logged and we move on -- the summary reads whatever records.jsonl exists.
run_cell () {
  local scene="$1" mode="$2"
  local logf="$LOG_DIR/${scene}_${mode}.log"
  local occ_prob clutter
  case "$scene" in
    curated)  occ_prob=0; clutter="$CURATED_CLUTTER_DENSITY" ;;   # occluder always spawned
    standard) occ_prob=1; clutter="$STANDARD_CLUTTER_DENSITY" ;;  # occluder never spawned
    *) echo "!!! unknown scene '$scene' -- skipping"; return ;;
  esac
  echo ""
  echo ">>> [$(date +%T)] START  scene=$scene APPROACH_MODE=$mode (clutter=$clutter)"
  # -u: stdout is redirected to a file, so without it Python block-buffers and the cell
  # log stays EMPTY for hours on a 50-seed run. Unbuffered keeps `tail -f` live.
  if APPROACH_MODE="$mode" "$PYTHON" -u script/bench_script/analyze_occluder_visibility.py \
        --base-config "$BASE_CONFIG" --seed-start "$SEED_START" --num-seeds "$NUM_SEEDS" \
        --rollout --out-dir "$OUT_ROOT/$scene" --run-type "$mode" \
        --offsets "$OFFSET" --clutter-densities "$clutter" --no-occluder-prob "$occ_prob" \
        >"$logf" 2>&1; then
    echo "<<< [$(date +%T)] DONE   $scene/$mode   (log: $logf)"
  else
    local rc=$?
    echo "!!! [$(date +%T)] FAILED $scene/$mode (exit $rc) -- see $logf (records.jsonl may still be complete)"
  fi
}

# Scene-major order: each scene's cells run back to back, so a run killed part-way still
# leaves at least one COMPLETE scene rather than two half-finished ones.
for scene in $SCENES; do
  for mode in $MODES; do
    case "$mode" in
      off|direct|seed) run_cell "$scene" "$mode" ;;
      *) echo "!!! unknown mode '$mode' (expected off|direct|seed) -- skipping" ;;
    esac
  done
done

# ---- Paired summary ----
echo ""
echo ">>> [$(date +%T)] Summarizing ..."
"$PYTHON" "$SCRIPT_DIR/summarize_approach_mode_ab.py" --root "$OUT_ROOT" \
  2>&1 | tee "$OUT_ROOT/summary.txt"

echo ""
echo "All done. Results under: $OUT_ROOT"
echo "  per cell   : $OUT_ROOT/<scene>/<mode>/<timestamp>/records.jsonl"
echo "  videos     : $OUT_ROOT/<scene>/<mode>/<timestamp>/{success,fail}/video/"
echo "  seed routes: $OUT_ROOT/<scene>/seed/<timestamp>/seed_route_visuals/episode<N>_<arm>/"
echo "  summary    : $OUT_ROOT/summary.txt"
