cd /shared_work/hamidreza2/projects/RoboPRO
# trt-gb10-1 currently cannot start NVIDIA containers because its runtime
# lacks NVML permissions. Allow callers to override or clear this exclusion.
export EXCLUDE_NODES="${EXCLUDE_NODES-trt-gb10-1}"


while read -r task eval_seed; do
  TIME_LIMIT=01:30:00 scripts/slurm/submit_pi05_eval.sh \
    "$task" \
    relation_validation_d14 \
    pi05_aloha_full_base \
    pi05_base \
    0 \
    0 \
    1 \
    visual_only \
    "$eval_seed"
done <<'EOF'
set_up_table 0
put_rubikscube_in_drawer 1
chain_heat_hamburger_ks 2
put_sauce_can_in_cabinet 3
put_sauce_can_in_basket 4
EOF