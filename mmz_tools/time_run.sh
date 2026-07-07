#!/usr/bin/env bash
# time_run.sh — time one collect_data.sh run and project full-dataset generation time.
#
# Usage (from customized_robotwin/):
#   bash ../mmz_tools/time_run.sh <task> <config> [gpu]
# Example (a positive config gives the best estimate — its wall time already
# includes the seed-search cost of rejected attempts):
#   bash ../mmz_tools/time_run.sh put_cup_on_coaster mmz_pos_d10 1
#
# Prints: wall time, episodes attempted vs kept, seconds per attempt / per kept,
# and a projection of the full 16K-positive + 8K-negative-attempt dataset on 1/4/20 GPUs.
set -u

TASK="${1:?usage: time_run.sh <task> <config> [gpu]}"
CONFIG="${2:?usage: time_run.sh <task> <config> [gpu]}"
GPU="${3:-0}"

# full-dataset targets (edit here if the spec changes)
POS_KEPT=16000      # positives: 80 tasks * (100 clean + 100 cluttered)
NEG_ATTEMPTS=8000   # negatives: 80 tasks * (10 * 10 densities) attempts

LOG="$(mktemp /tmp/mmz_time_XXXXXX.log)"
echo "▶ timing: bash collect_data.sh $TASK $CONFIG $GPU   (log: $LOG)"
echo "────────────────────────────────────────────────────────────────────"
start=$(date +%s)
bash collect_data.sh "$TASK" "$CONFIG" "$GPU" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
end=$(date +%s)
wall=$(( end - start ))

# parse the COLLECTION SUMMARY the collector prints
att=$(grep -aoiE 'episodes attempted[[:space:]]+[0-9]+' "$LOG" | tail -1 | grep -oE '[0-9]+$')
kept=$(grep -aoiE 'kept in dataset[[:space:]]+[0-9]+'   "$LOG" | tail -1 | grep -oE '[0-9]+$')
att=${att:-0}; kept=${kept:-0}

echo "════════════════════════════════════════════════════════════════════"
echo "TIMING RESULT · $TASK / $CONFIG  (exit $rc)"
echo "  wall clock            ${wall}s  (~$(awk "BEGIN{printf \"%.1f\", $wall/60}") min)"
echo "  episodes attempted    ${att}"
echo "  episodes kept         ${kept}"
if [ "$att" -gt 0 ]; then
  echo "  sec / attempted ep    $(awk "BEGIN{printf \"%.1f\", $wall/$att}")"
fi
if [ "$kept" -gt 0 ]; then
  echo "  sec / kept ep         $(awk "BEGIN{printf \"%.1f\", $wall/$kept}")"
fi

if [ "$att" -gt 0 ] && [ "$kept" -gt 0 ]; then
  echo "────────────────────────────────────────────────────────────────────"
  echo "PROJECTION → full dataset (${POS_KEPT} pos kept + ${NEG_ATTEMPTS} neg attempts)"
  echo "  assumptions: pos time uses THIS run's sec/kept (embeds seed-search"
  echo "  rejects if this was a positive config); neg time uses sec/attempt."
  awk -v wall="$wall" -v att="$att" -v kept="$kept" \
      -v posk="$POS_KEPT" -v nega="$NEG_ATTEMPTS" 'BEGIN{
    spk = wall/kept; spa = wall/att;
    total = posk*spk + nega*spa;          # seconds on 1 GPU
    printf "  %-8s %10s %10s %10s\n","GPUs","hours","days","(rounded)";
    split("1 4 20", g, " ");
    for(i=1;i<=3;i++){ n=g[i]; h=total/n/3600.0;
      printf "  %-8s %10.1f %10.2f\n", n, h, h/24.0; }
  }'
  echo "  NB: warmup (CuRobo init per launch) + high-density seed search inflate"
  echo "      real time; time a low- AND a high-density config to bracket it."
else
  echo "  (could not parse attempted/kept from the run — check $LOG)"
fi
