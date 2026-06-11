#!/bin/bash
# Paired evaluation: base checkpoint vs advantage-token checkpoint, task by
# task, on the EXACT same seeds (bench_demo_kitchenl_d15, plain eval).
#
# Per task:
#   1. base policy runs with the normal seed scan (expert check skips bad seeds)
#   2. the N seeds it evaluated are extracted from its episodes.jsonl
#   3. adv_token policy replays exactly that seed list (EVAL_SEED_LIST)
#
# Metrics per run (in _result.txt):
#   success rate, hard success rate (success w/ zero collisions),
#   collision rate (episodes with >=1 collision)
#
# Usage:
#   bash policy/pi05/eval_adv_batch.sh [server_gpu:client_gpu]   # default 1:0
#
# Env overrides:
#   EVAL_TEST_NUM — seeds per task (default 20)
#   EVAL_SEED     — seed offset (default 0 → st_seed 100000)
#   TASKS         — space-separated task list override

set -uo pipefail

gpu_spec=${1:-1:0}
export EVAL_TEST_NUM=${EVAL_TEST_NUM:-20}
EVAL_SEED=${EVAL_SEED:-0}
export EVAL_PROXIMITY_GUIDANCE=0   # plain eval, no guidance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="${SCRIPT_DIR}/../../eval_result"

task_config=bench_demo_kitchenl_d15
TASKS=${TASKS:-"put_bottle_in_fridge pick_bottle_from_fridge put_bottle_in_basket put_sauce_can_in_cabinet put_can_close_cabinet"}

# label : train_config : model_name : checkpoint_id
BASE_CKPT="base:pi05_aloha_full_base:roboreal_all_80tasks:20000"
NEW_CKPT="adv_token:pi05_aloha_advantage_token:pi05_adv_token_run2_norm:50000"

log_root="${EVAL_ROOT}/adv_batch_logs"
mkdir -p "${log_root}"

run_eval () {  # label train_config model_name ckpt_id task seed_list_or_empty
    local label=$1 train_config=$2 model_name=$3 ckpt_id=$4 task=$5 seeds=${6:-}
    local log="${log_root}/${label}_${task}.log"
    echo "=== [$(date +%H:%M:%S)] ${label} / ${task} (n=${EVAL_TEST_NUM}) ==="
    EVAL_SEED_LIST="${seeds}" bash "${SCRIPT_DIR}/eval_proximity_guided.sh" \
        "${task}" "${task_config}" \
        "${train_config}" "${model_name}" "${ckpt_id}" \
        "${EVAL_SEED}" "${gpu_spec}" \
        2>&1 | tee "${log}" | grep -E 'Success rate|replay|Done|Traceback|ModuleNotFound' || true
    echo "=== [$(date +%H:%M:%S)] ${label} / ${task} finished ==="
}

latest_result_dir () {  # model_name task
    ls -td "${EVAL_ROOT}/$2/pi05/${task_config}/$1/proximity_guided/"*/ 2>/dev/null | head -1
}

for task in ${TASKS}; do
    # ── 1. base policy: scan mode ────────────────────────────────────────
    IFS=':' read -r b_label b_cfg b_model b_id <<< "${BASE_CKPT}"
    run_eval "${b_label}" "${b_cfg}" "${b_model}" "${b_id}" "${task}" ""

    base_dir=$(latest_result_dir "${b_model}" "${task}")
    if [[ -z "${base_dir}" || ! -f "${base_dir}/episodes.jsonl" ]]; then
        echo "!!! no episodes.jsonl from base run on ${task} — skipping paired run"
        continue
    fi

    # ── 2. extract the exact seeds the base run evaluated ────────────────
    seeds=$(python3 -c "
import json, sys
print(','.join(str(json.loads(l)['seed']) for l in open('${base_dir}/episodes.jsonl')))
")
    echo "    base seeds for ${task}: ${seeds}"

    # ── 3. new policy: replay exact seeds ────────────────────────────────
    IFS=':' read -r n_label n_cfg n_model n_id <<< "${NEW_CKPT}"
    run_eval "${n_label}" "${n_cfg}" "${n_model}" "${n_id}" "${task}" "${seeds}"

    # ── 4. immediate side-by-side summary ────────────────────────────────
    new_dir=$(latest_result_dir "${n_model}" "${task}")
    echo ""
    echo "########## ${task} — paired comparison ##########"
    echo "--- base (${b_model}/${b_id}) ---"
    [[ -n "${base_dir}" ]] && cat "${base_dir}/_result.txt"
    echo "--- adv_token (${n_model}/${n_id}) ---"
    [[ -n "${new_dir}" ]] && cat "${new_dir}/_result.txt"
    echo "##################################################"
    echo ""
done

echo "=== Batch complete ==="
