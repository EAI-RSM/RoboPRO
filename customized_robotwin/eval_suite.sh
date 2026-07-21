#!/bin/bash
# eval_suite.sh — run a policy eval across a {tasks} x {configs} grid, reusing ONE model
# server, and collate the per-run success logs into a single table.
#
#   bash eval_suite.sh <policy> <checkpoint_id> <task_selector> <config_selector> [client_gpu]
#
#   task_selector    a scene dir under benchmark/bench_envs/ (office|study|kitchenl|kitchens)
#                    -> all of its tasks; OR one task name; OR a comma-list of task names.
#   config_selector  a glob over benchmark/bench_task_config/ (e.g. "bench_vision_blur_*")
#                    OR one config name; OR a comma-list. ".yml" is added automatically.
#   client_gpu       GPU for the simulator client (default 1; GPU 0 is usually the display).
#
# The model SERVER runs on CPU by default (JAX_PLATFORMS=cpu) so it can share a small card
# with the simulator; set EVAL_SERVER_GPU=<id> to put the server on a GPU instead.
# EVAL_TEST_NUM caps successful episodes per (task,config) (default 100; use a small value
# for smoke tests) — it is read by the eval client from the environment.
#
# Scene is resolved from the task name (like collect_data.sh). Eval writes NO dataset — only
# a success log under eval_result/<task>/<policy>/<config>/<ckpt_setting>/<timestamp>/.
#
# Run from customized_robotwin/ with the policy's sim env (e.g. `robopro`) active.
#
# Examples:
#   bash eval_suite.sh pi05 30000 office bench_demo_office_clean            # 20 office tasks, clean
#   bash eval_suite.sh pi05 30000 put_mouse_on_pad "bench_vision_blur_*"    # 1 task, all blur densities
#   bash eval_suite.sh pi05 30000 office bench_object_ood_appearance_d10 1  # 20 office tasks, OOD @ d10
set -uo pipefail

POLICY=${1:?usage: eval_suite.sh <policy> <checkpoint_id> <task_selector> <config_selector> [client_gpu]}
CKPT=${2:?checkpoint_id required}
TASK_SEL=${3:?task_selector required}
CONF_SEL=${4:?config_selector required}
CLIENT_GPU=${5:-1}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# set_env.sh exports BENCH_ROOT / ROBOTWIN_ROOT; bench routing needs the task switch.
source set_env.sh >/dev/null
export ROBOTWIN_BENCH_TASK=bench

DEPLOY="policy/${POLICY}/deploy_policy.yml"
SERVER_PY="policy/${POLICY}/.venv/bin/python"
[ -f "$DEPLOY" ]    || { echo "no deploy config: $DEPLOY"; exit 1; }
[ -x "$SERVER_PY" ] || { echo "no server venv python: $SERVER_PY (run 'uv sync' in policy/${POLICY})"; exit 1; }

# model identity: read from the deploy yaml so nothing is hard-coded; ckpt id from the CLI
# (the yaml's checkpoint_id can be stale).
TRAIN_CONFIG=$(awk '/^train_config_name:/{print $2}' "$DEPLOY")
MODEL=$(awk '/^model_name:/{print $2}' "$DEPLOY")
[ -n "$TRAIN_CONFIG" ] && [ -n "$MODEL" ] || { echo "deploy yaml missing train_config_name/model_name"; exit 1; }

# ---- expand the task selector ------------------------------------------------
if [ -d "$BENCH_ROOT/bench_envs/$TASK_SEL" ]; then
    # a scene dir -> every task class in it (drop __init__ and any _-prefixed base class)
    TASKS=$(cd "$BENCH_ROOT/bench_envs/$TASK_SEL" && ls *.py 2>/dev/null | sed 's/\.py$//' \
            | grep -vE '^_')
else
    TASKS=$(echo "$TASK_SEL" | tr ',' ' ')
fi
[ -n "$TASKS" ] || { echo "no tasks match: $TASK_SEL"; exit 1; }

# ---- expand the config selector ----------------------------------------------
CONFIGS=$(cd "$BENCH_ROOT/bench_task_config" && ls ${CONF_SEL}.yml 2>/dev/null | sed 's/\.yml$//')
[ -n "$CONFIGS" ] || { echo "no configs match: $CONF_SEL (in bench_task_config/)"; exit 1; }

N_TASK=$(echo $TASKS | wc -w); N_CONF=$(echo $CONFIGS | wc -w)
FIRST_TASK=$(echo $TASKS   | awk '{print $1}')
FIRST_CONF=$(echo $CONFIGS | awk '{print $1}')
echo ">>> eval_suite: ${N_TASK} task(s) x ${N_CONF} config(s) = $((N_TASK*N_CONF)) run(s)  [policy=$POLICY ckpt=$CKPT]"

# ---- pick a free port --------------------------------------------------------
PORT=$(python - <<'PY'
import socket
s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()
PY
)

# ---- start ONE server (reused for the whole grid) ----------------------------
SLOG="$(mktemp)"
if [ -n "${EVAL_SERVER_GPU:-}" ]; then
    SRV_DEV="CUDA_VISIBLE_DEVICES=$EVAL_SERVER_GPU"; WHERE="gpu $EVAL_SERVER_GPU"
else
    SRV_DEV="JAX_PLATFORMS=cpu";                     WHERE="cpu"
fi
echo ">>> starting $POLICY model server on $WHERE (port $PORT) — loading checkpoint..."
# PYTHONUNBUFFERED so the server's "waiting for client" line reaches our readiness grep
# immediately (print() is block-buffered when stdout is a file, not a tty).
env $SRV_DEV PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$SERVER_PY" script/policy_model_server.py \
    --port "$PORT" --config "$DEPLOY" --overrides \
    --task_name "$FIRST_TASK" --task_config "$FIRST_CONF" \
    --train_config_name "$TRAIN_CONFIG" --model_name "$MODEL" \
    --checkpoint_id "$CKPT" --ckpt_setting "$MODEL" --seed 0 --policy_name "$POLICY" \
    >"$SLOG" 2>&1 &
SERVER_PID=$!
cleanup() { echo ">>> stopping server (pid $SERVER_PID)"; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; rm -f "$SLOG"; }
trap cleanup EXIT INT TERM

# wait until the server is listening (a cold 12 GB checkpoint load can take several minutes)
for _ in $(seq 1 300); do
    grep -q "waiting for client connections" "$SLOG" && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "!!! server exited during startup:"; tail -30 "$SLOG"; exit 1; }
    sleep 2
done
grep -q "waiting for client connections" "$SLOG" || { echo "!!! server not ready after timeout:"; tail -30 "$SLOG"; exit 1; }
echo ">>> server ready — starting sweep"

# ---- loop the grid (client on the GPU; one server serves them all) -----------
i=0
for TASK in $TASKS; do
  for CONF in $CONFIGS; do
    i=$((i+1))
    echo ""
    echo "===== [$i/$((N_TASK*N_CONF))]  $TASK  x  $CONF  ====="
    CUDA_VISIBLE_DEVICES="$CLIENT_GPU" PYTHONWARNINGS=ignore::UserWarning \
      python script/eval_policy_client.py \
      --port "$PORT" --config "$DEPLOY" --overrides \
      --task_name "$TASK" --task_config "$CONF" \
      --train_config_name "$TRAIN_CONFIG" --model_name "$MODEL" \
      --checkpoint_id "$CKPT" --ckpt_setting "$MODEL" --seed 0 --policy_name "$POLICY" \
      || echo "!!! [$TASK x $CONF] failed — continuing"
  done
done

echo ""
echo ">>> sweep done — summary:"
python script/eval_summary.py
