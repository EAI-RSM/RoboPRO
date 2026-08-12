#!/usr/bin/env bash
# Rollout-80 worker: qualify assigned seed blocks, then collect 4 fixed-seed
# rollouts per qualified scene, HF-layout output <OUT>/<task>/<density>/seed<S>/.
#
#   CONDA_BIN=<sim bin> OUT=<dir> SPEC="task:dens:start:want task:dens:start:want ..." \
#     bash tmp_collect80_worker.sh
#
# Seed safety: qualification walks a 100-wide per-density block (blocks are
# globally disjoint), collection pins each seed via COLLECT_FIXED_SEED=1 +
# COLLECT_START_SEED per single-episode invocation, and seed.txt is verified
# to contain exactly "S S S S" before the run dir is accepted.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="${CONDA_BIN:?}:$PATH"
source set_env.sh >/dev/null
export ROBOTWIN_BENCH_TASK=bench
export ROBOPRO_RT_DENOISER=${ROBOPRO_RT_DENOISER:-optix}
OUT=${OUT:?}; mkdir -p "$OUT"
SPEC=${SPEC:?}
SERVER_GPU=${SERVER_GPU:-0}; SIM_GPU=${SIM_GPU:-1}; XLA_FRAC=${XLA_FRAC:-0.55}
MAN="$OUT/manifest_$(hostname).json"

envof() { case "$1" in drop_apple_in_bin_ks) echo kitchens;; put_milktea_next_to_laptop) echo office;; *) echo unknown;; esac; }

echo "== [$(hostname)] PHASE 1: qualify =="
for entry in $SPEC; do
  IFS=: read -r T D S W <<< "$entry"
  CUDA_VISIBLE_DEVICES=$SIM_GPU PYTHONUNBUFFERED=1 python -u script/tmp_qualify_seeds.py \
    --task "$T" --density "$D" --start "$S" --want "$W" --out "$MAN" \
    >> "$OUT/qualify_$(hostname).log" 2>&1
  grep -E "qualify\] .*(qualified|WARNING)" "$OUT/qualify_$(hostname).log" | tail -2
done
echo "== manifest =="; cat "$MAN"

echo "== PHASE 2: server up =="
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1])")
(
  export CUDA_VISIBLE_DEVICES=$SERVER_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
         XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_FRAC PYTHONWARNINGS=ignore::UserWarning
  exec policy/pi05/.venv/bin/python script/policy_model_server.py \
    --port "$PORT" --config policy/pi05/deploy_policy.yml --overrides \
    --task_name put_milktea_next_to_laptop --task_config bench_demo_office_clean \
    --train_config_name pi05_robopro_cfm --model_name robopro_jax \
    --checkpoint_id 30000 --ckpt_setting robopro_jax --seed 0 --policy_name pi05
) > "$OUT/server_$(hostname).log" 2>&1 &
SP=$!; trap 'kill $SP 2>/dev/null' EXIT
for _ in $(seq 1 180); do
  kill -0 $SP 2>/dev/null || { echo "[server DIED]"; tail -20 "$OUT/server_$(hostname).log"; exit 1; }
  python3 -c "import socket;socket.create_connection(('127.0.0.1',$PORT),2)" 2>/dev/null && break
  sleep 5
done
echo "[server up :$PORT]"

collect_scene() {  # $1 task  $2 density  $3 seed
  local T=$1 D=$2 S=$3 CFG ENVN CAN DEST
  ENVN=$(envof "$T"); CFG="bench_demo_${ENVN}_${D}"
  CAN="./data/bench_data/$T/$CFG"
  DEST="$OUT/$T/$D/seed$S"
  [ -d "$DEST" ] && { echo "[skip] $DEST exists"; return 0; }
  rm -rf "$CAN"
  mkdir -p "$OUT/clientlogs"
  local CL="$OUT/clientlogs/${T}_${D}_seed${S}.log"
  COLLECT_FIXED_SEED=1 COLLECT_START_SEED=$S COLLECT_NUM=4 \
  CUDA_VISIBLE_DEVICES=$SIM_GPU PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
    python -u script/collect_rollout_client.py --port "$PORT" \
      --config policy/pi05/deploy_policy.yml --overrides \
      --task_name "$T" --task_config "$CFG" --policy_name pi05 \
      --ckpt_setting robopro_jax --instruction_type seen > "$CL" 2>&1
  tr '\r' '\n' < "$CL" | grep -E "episode .* saved|aborting|Traceback" | tail -2
  local SEEDS; SEEDS=$(cat "$CAN/seed.txt" 2>/dev/null | tr -s ' ')
  if [ "$(echo $SEEDS)" = "$S $S $S $S" ]; then
    mkdir -p "$(dirname "$DEST")" && mv "$CAN" "$DEST"
    echo "[scene OK] $T $D seed$S -> $DEST"
  else
    mkdir -p "$OUT/_quarantine" && mv "$CAN" "$OUT/_quarantine/${T}_${D}_seed${S}_$(date +%s)" 2>/dev/null
    echo "[scene BAD] $T $D seed$S: seed.txt='$SEEDS' (expected '$S x4') -> quarantined"
  fi
}

echo "== PHASE 3: collect =="
for entry in $SPEC; do
  IFS=: read -r T D S W <<< "$entry"
  SEEDS=$(python3 -c "import json;print(' '.join(map(str,json.load(open('$MAN'))['$T']['$D'])))")
  echo "== $T $D seeds: $SEEDS =="
  for s in $SEEDS; do collect_scene "$T" "$D" "$s"; done
done
echo "== [$(hostname)] ALL DONE $(date) =="
