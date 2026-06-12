#!/bin/bash
# RL dataset collection: same-seed triplets of (collision-free CuRobo expert,
# pi05 policy rollout, collision-unaware CuRobo) for critic training and
# iterative offline RL — see data_generation_documentation.md.
#
# For each task: 10 seeds × 3 sources = up to 30 HDF5 files (triplets, see
# data_generation_documentation.md):
#   A. CuRobo, clutter in collision world   (planner_collision_aware)
#   B. pi05 policy rollout                  (pi05_base)
#   C. CuRobo, clutter ignored              (planner_collision_unaware)
# A gates the seed (must produce a successful trajectory); B and C are saved
# regardless of outcome.  Per seed also exports scene/seed_N/ (pointcloud +
# objects + hash), per-episode contact/collision jsonl streams (metrics/),
# fk_basis.npz, and index.jsonl.  Run the offline relabel pass afterwards:
#   python script/relabel_collision_dataset.py rollout_data/<task>/<env>
#
# Usage:
#   bash policy/pi05/collect_rl_dataset.sh <train_config> <model_name> <checkpoint_id> <gpu_spec>
#
# Example:
#   bash policy/pi05/collect_rl_dataset.sh roboreal_all_80tasks pi0_base 20000 0
#   bash policy/pi05/collect_rl_dataset.sh roboreal_all_80tasks pi0_base 20000 0:1   # server:client gpu
#
# Env vars:
#   COLLECT_NUM          — seeds per task (default 10)
#   COLLECT_START_SEED   — starting seed (default auto-resume per task)
#   TASKS                — space-separated override list (default: all 20 study tasks)

set -euo pipefail

TRAIN_CONFIG=${1}
MODEL_NAME=${2}
CHECKPOINT_ID=${3:-20000}
GPU_SPEC=${4:-0}

SEEDS_PER_TASK=${COLLECT_NUM:-10}
TASK_CONFIG="${TASK_CONFIG:-bench_demo_study_d15}"

# All 20 study environment tasks
DEFAULT_TASKS=(
    empty_box
    move_book_onto_table
    move_cup_next_to_book
    move_cup_onto_table
    move_cup_put_pen_in_cup
    move_cup
    move_cups_into_box
    move_pen_to_box
    move_seal_cup_next_to_box
    move_seal_next_to_box
    move_seal_next_to_pencup
    move_seal_onto_book
    move_seal_onto_table
    put_cup_in_box
    put_cup_on_coaster
    put_cup_on_table
    put_glue_in_box
    put_pen_in_box
    put_pen_in_pencup
    put_seal_in_box
)

# Allow override via TASKS env var
if [[ -n "${TASKS:-}" ]]; then
    read -ra TASK_LIST <<< "${TASKS}"
else
    TASK_LIST=("${DEFAULT_TASKS[@]}")
fi

echo -e "\033[33m=== Study paired collection ===\033[0m"
echo -e "\033[33mTrain config : ${TRAIN_CONFIG}\033[0m"
echo -e "\033[33mModel        : ${MODEL_NAME}  ckpt=${CHECKPOINT_ID}\033[0m"
echo -e "\033[33mGPU spec     : ${GPU_SPEC}\033[0m"
echo -e "\033[33mSeeds/task   : ${SEEDS_PER_TASK}\033[0m"
echo -e "\033[33mTasks (${#TASK_LIST[@]}) : ${TASK_LIST[*]}\033[0m"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for TASK in "${TASK_LIST[@]}"; do
    echo -e "\033[36m──────────────────────────────────────────────\033[0m"
    echo -e "\033[36mTask: ${TASK}\033[0m"
    echo -e "\033[36m──────────────────────────────────────────────\033[0m"

    COLLECT_NUM=${SEEDS_PER_TASK} \
    COLLECT_MODE=paired \
    COLLECT_SELECTIVE_SAVE=0 \
    ROBOTWIN_BENCH_TASK=bench \
    bash "${SCRIPT_DIR}/collect_rollout.sh" \
        "${TASK}" \
        "${TASK_CONFIG}" \
        "${TRAIN_CONFIG}" \
        "${MODEL_NAME}" \
        "${CHECKPOINT_ID}" \
        0 \
        "${GPU_SPEC}" \
    || echo -e "\033[31m  [warn] Task ${TASK} failed — continuing with next task\033[0m"

    # Offline relabel pass (per-link clearance labels, contact attribution,
    # windows, per-episode meta).  Re-runnable any time with new settings.
    ENV_TYPE=$([[ "${TASK_CONFIG}" == *clean* ]] && echo clean || echo cluttered)
    DS_DIR="${SCRIPT_DIR}/../../collision_dataset/${TASK}/${ENV_TYPE}"
    if [[ -d "${DS_DIR}" ]]; then
        python "${SCRIPT_DIR}/../../script/relabel_collision_dataset.py" "${DS_DIR}" \
            || echo -e "\033[31m  [warn] relabel failed for ${TASK} — rerun manually\033[0m"
    fi

    echo ""
done

echo -e "\033[32m=== All tasks done ===\033[0m"
