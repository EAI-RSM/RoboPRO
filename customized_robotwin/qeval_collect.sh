#!/usr/bin/env bash
# Q-eval pair collection (fixed_replay protocol). One pi05 server (task-agnostic) + sim client per task.
#
#   CONDA_BIN=<sim env bin> bash tmp_qeval_collect.sh "<task1 task2 ...>" [outdir]
#
# Env: SERVER_GPU, SIM_GPU, XLA_FRAC, MAX_ATTEMPTS, MAX_SEEDS, BRANCH_STEPS,
#      DENSITIES, STEP_LIM, SEED_BASE
set -uo pipefail
cd "$(dirname "$0")"
CONDA_BIN=${CONDA_BIN:?set CONDA_BIN to the sim conda env bin dir}
export PATH="$CONDA_BIN:$PATH"
source set_env.sh >/dev/null
export ROBOTWIN_BENCH_TASK=bench
export ROBOPRO_RT_DENOISER=${ROBOPRO_RT_DENOISER:-optix}  # MUST match ws_amir + HF training data
export XDG_CACHE_HOME=/tmp/$(whoami)-qeval; export MPLCONFIGDIR=$XDG_CACHE_HOME/mpl
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

TASKS=${1:?pass a space-separated task list}
OUT=${2:-$(pwd)/qeval_pairs}
SERVER_GPU=${SERVER_GPU:-0}
SIM_GPU=${SIM_GPU:-1}
XLA_FRAC=${XLA_FRAC:-0.55}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-8}
MAX_SEEDS=${MAX_SEEDS:-4}
BRANCH_STEPS=${BRANCH_STEPS:-0,50,100}
DENSITIES=${DENSITIES:-clean,clean,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15}
STEP_LIM=${STEP_LIM:-300}
SEED_BASE=${SEED_BASE:-5000}
mkdir -p "$OUT"

echo "=================================================================="
echo " Q-EVAL PAIR COLLECTION"
echo " tasks    : $TASKS"
echo " out      : $OUT"
echo " densities: $DENSITIES"
echo " branches : $BRANCH_STEPS   attempts<=$MAX_ATTEMPTS  seeds<=$MAX_SEEDS"
echo " step_lim : $STEP_LIM   seed_base: $SEED_BASE (unseen; HF used 0,1,2,4,6,7)"
echo " protocol : fresh rebuild + policy-free prefix replay (NEVER restore)"
echo "=================================================================="

PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1])")
(
  export CUDA_VISIBLE_DEVICES=$SERVER_GPU
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_FRAC
  export PI05_NOISE_SEED=771000
  export PI05_NOISE_DIR="$OUT/_server_noise"
  export PYTHONWARNINGS=ignore::UserWarning
  exec policy/pi05/.venv/bin/python script/policy_model_server.py \
    --port "$PORT" --config policy/pi05/deploy_policy.yml --overrides \
    --task_name put_milktea_next_to_laptop --task_config bench_demo_office_clean \
    --train_config_name pi05_robopro_cfm --model_name robopro_jax \
    --checkpoint_id 30000 --ckpt_setting robopro_jax --seed 0 --policy_name pi05
) > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'echo "[cleanup] killing server $SERVER_PID"; kill $SERVER_PID 2>/dev/null' EXIT

up=0
for _ in $(seq 1 180); do
  kill -0 $SERVER_PID 2>/dev/null || { echo "[server DIED]"; tail -30 "$OUT/server.log"; exit 1; }
  python3 -c "import socket;socket.create_connection(('127.0.0.1',$PORT),2)" 2>/dev/null && { up=1; break; }
  sleep 5
done
[ $up -eq 1 ] || { echo "[server TIMEOUT]"; tail -30 "$OUT/server.log"; exit 1; }
echo "[server up on port $PORT, pid $SERVER_PID]"

for t in $TASKS; do
  echo ""
  echo "########## $t  $(date +%H:%M:%S) ##########"
  CUDA_VISIBLE_DEVICES=$SIM_GPU PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
    python -u script/collect_qeval_pairs.py \
      --port "$PORT" --task_name "$t" --outdir "$OUT" \
      --densities "$DENSITIES" --branch_steps "$BRANCH_STEPS" \
      --max_attempts "$MAX_ATTEMPTS" --max_seeds "$MAX_SEEDS" \
      --step_lim "$STEP_LIM" --seed_base "$SEED_BASE" \
    || echo "FAILED $t" | tee -a "$OUT/failures.log"
  kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER DIED mid-run" | tee -a "$OUT/failures.log"; break; }
done
echo ""
echo "[ALL TASKS DONE] $(date)"
