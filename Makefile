SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:

ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
CUSTOMIZED_ROOT := $(ROOT_DIR)/customized_robotwin
PYTHON ?= $(ROOT_DIR)/.venv/bin/python
UV ?= uv

# Common benchmark settings
ROBOTWIN_BENCH_TASK ?= bench
TASK_NAME ?= put_mouse_on_pad
TASK_CONFIG ?= bench_demo_office_clean
BENCH_SUBDIR ?= office
SEED ?= 0
GPU_ID ?= 0
GPU_SPEC ?= 0
CUROBO_TRAJOPT_SEEDS ?= 16
CUROBO_MAX_ATTEMPTS ?= 24
CUROBO_BATCH_GRAPH_SEEDS ?= 1
# Left empty by default (not a hardcoded fallback) -- planner.py treats empty/unset as
# "use CuRobo's own MotionGenPlanConfig default" (finetune_attempts=5, dt_scale=0.85).
CUROBO_FINETUNE_ATTEMPTS ?=
CUROBO_FINETUNE_DT_SCALE ?=
CUROBO_ATTACH_SPHERE_RADIUS ?= 0.001
LOCAL_WAYPOINT_ATTEMPTS ?= 5
ATTACHED_TRAJECTORY_SLOWDOWN ?= 2
WAYPOINT_SHRINK_MIN_DISTANCE ?= 0.05
# Empty by default -> analyze_occluder_visibility.py's own default (.../phase2_occluder,
# with a _rollout suffix appended when ROLLOUT=1). Set to timestamp each validation run,
# e.g. OUT_DIR=../scripts/validation/results/$$(date +%Y-%m-%d-%H-%M-%S)
OUT_DIR ?=

# Visualization flags
RENDER_FREQ ?= 3
VIEWER_CAMERA ?= demo_camera
NO_RENDER ?= 1
ROLLOUT ?= 1
SAVE_DATA ?= 1
SAVE_PLAN_FAIL_DIR ?=
PLAN_FAIL_CAMERA ?= demo_camera
HDF5_FILE ?=
HDF5_CAMERA ?=
HDF5_FRAME ?= 0
HDF5_PREVIEW_PATH ?=
SHOW_TREE ?= 0
DUMP_JSON ?= 0
VIZ_HDF5_FILE ?=
VIZ_OUTPUT_VIDEO ?=
VIZ_WIDTH ?= 1280
VIZ_HEIGHT ?= 720
VIZ_FPS ?= 20
VIZ_TRAIL ?= 25
REL_HDF5_FILE ?=
REL_FRAME ?= 0
REL_OUTPUT_IMAGE ?=
REL_OUTPUT_DIR ?=
REL_SAMPLE_FRAMES ?= 0,74,124,182,191
REL_WIDTH ?= 1600
REL_HEIGHT ?= 1000
REL_SHOW_EDGE_LABELS ?= 1
REL_EXCLUDED_EDGES ?=
REL_INCLUDED_EDGES ?=
REL_ABSTRACT_LAYOUT ?= 0

# Graph-rich relation validation flags. REACHABLE_BY_INTERVAL is measured in
# exported frames; 1 evaluates every changed frame, 10 evaluates every tenth.
REACHABLE_BY_ENABLED ?= 1
REACHABLE_BY_INTERVAL ?= 10
REACHABLE_BY_MOVABLE_ONLY ?= 1
REACHABLE_BY_CACHE_UNCHANGED ?= 1
REACHABLE_BY_POSE_DECIMALS ?= 3
RELATION_OBSTACLE_DENSITY ?= 14
RELATION_EPISODE_NUM ?= 1
RELATION_SAVE_PATH ?= ./data/relation_validation_d14_actions_v2
ACTION_VALIDATION_MATRIX ?= $(ROOT_DIR)/benchmark/bench_task_config/action_validation_suite.yml
ACTION_VALIDATION_OUTPUT_ROOT ?= $(CUSTOMIZED_ROOT)/data/action_validation_suite_v1
ACTION_VALIDATION_MODE ?= all
ACTION_VALIDATION_TASKS ?=
ACTION_VALIDATION_START_SEED ?= 0
ACTION_VALIDATION_DRY_RUN ?= 0
ACTION_VALIDATION_OBSTACLE_DENSITY ?= 10

# Asset / install flags
PYTHON_VERSION ?= 3.10
ASSETS_DEST ?= $(ROOT_DIR)/benchmark/assets
KEEP_ZIPS ?= 0
ASSETS_DATASET_NAME ?= RoboPRO_assets
ASSETS_PATH ?= $(ROOT_DIR)/benchmark

# Eval / policy flags
POLICY_NAME ?= pi05
POLICY_CONFIG ?= policy/$(POLICY_NAME)/deploy_policy.yml
TRAIN_CONFIG_NAME ?= my_office_train
MODEL_NAME ?= pi05_ckpt
CHECKPOINT_ID ?= 30000
CKPT_SETTING ?= $(TRAIN_CONFIG_NAME)_$(MODEL_NAME)_$(CHECKPOINT_ID)
INSTRUCTION_TYPE ?= seen
TEST_NUM ?= 1
PORT ?= 5555

# pi05 rollout collection flags
COLLECT_NUM ?= 100
COLLECT_START_SEED ?=
COLLECT_BRANCH_NUM ?= 0
COLLECT_BRANCH_LOOKBACK ?= 5,10,15
COLLECT_BRANCH_NOISE_STEPS ?= 1
ACTION_NOISE_VAR ?= 0.005
COLLECT_FIXED_SEED ?= 0

# Occluder / reachability analysis flags (issue #35)
OFFSET ?= 0.2
OCC_DISTANCE_CM ?= 20,20
OCC_SEED_START ?= 0
OCC_NUM_SEEDS ?= 50
SAVE_IMAGES ?= 0
REACH_SEED ?= 1
REACH_Z ?= 0.90
REACH_ARMS ?= both
PICKUP_SEEDS ?= 1,2,3,4,5

define RUN_IN_CUSTOMIZED
	cd "$(CUSTOMIZED_ROOT)"
	source set_env.sh
	export ROBOTWIN_BENCH_TASK="$(ROBOTWIN_BENCH_TASK)"
	$(1)
endef

.PHONY: help check-prereqs bootstrap sync download-assets link-assets configure-curobo-assets \
	patch-curobo-config setup render-test verify-scene verify-rollout collect-data \
	relation-validation action-validation-suite check-action-validation-suite \
	precollect-seeds eval-direct eval-client policy-server eval-pi05-single eval-pi05-double \
	collect-rollout-pi05 diag-kitchen-curobo inspect-benchmark-hdf5 visualize-benchmark-rollout \
	visualize-relation-frame visualize-relation-samples occluder-visibility reachability-map \
	pickup-reachability analyze-occluder-rollout show-config

help:
	@printf '%s\n' \
	'RoboPRO Make targets' \
	'' \
	'Install / setup:' \
	'  make check-prereqs            Verify system tools (uv, nvcc, ffmpeg, etc.).' \
	'  make bootstrap                Create/sync .venv via uv and run post-install patches.' \
	'    Vars: PYTHON_VERSION=3.10' \
	'  make download-assets          Download benchmark bundles.' \
	'    Vars: ASSETS_DEST=benchmark/assets KEEP_ZIPS=0|1' \
	'  make link-assets              Wire benchmark/assets and customized_robotwin/assets to ASSETS_DEST.' \
	'  make configure-curobo-assets  Render curobo_{left,right}.yml from curobo_*_tmp.yml' \
	'    Vars: ASSETS_PATH=$(ASSETS_PATH)' \
	'  make patch-curobo-config      Run scripts/install/patch_aloha_curobo.py' \
	'  make setup                    Run link-assets + configure-curobo-assets + patch-curobo-config.' \
	'' \
	'Smoke tests:' \
	'  make render-test              Minimal Sapien renderer smoke test.' \
	'  make verify-scene             Load a benchmark task scene only.' \
	'  make verify-rollout           Headless rollout smoke test; saves video by default.' \
	'  make precollect-seeds         Generate eval seeds without saving demos.' \
	'  make diag-kitchen-curobo      Kitchen collision diagnostic script.' \
	'  make inspect-benchmark-hdf5   Inspect a benchmark HDF5 export and optional preview frame.' \
	'    Vars: HDF5_FILE=/abs/path/to/file.hdf5 HDF5_CAMERA=demo_camera HDF5_FRAME=0 HDF5_PREVIEW_PATH=/tmp/frame.png SHOW_TREE=1 DUMP_JSON=1' \
	'  make visualize-benchmark-rollout  Render a top-down verification MP4 from a benchmark HDF5 export.' \
	'    Vars: VIZ_HDF5_FILE=path/to/file.hdf5 VIZ_OUTPUT_VIDEO=/tmp/rollout.mp4 VIZ_WIDTH=1280 VIZ_HEIGHT=720 VIZ_FPS=20 VIZ_TRAIL=25' \
	'  make visualize-relation-frame Render one frame of canonical relation edges from a benchmark HDF5 export.' \
	'    Vars: REL_HDF5_FILE=path/to/file.hdf5 REL_FRAME=0 REL_OUTPUT_IMAGE= (default: <episode>/visualizations/scene_graph/) REL_WIDTH=1600 REL_HEIGHT=1000 REL_SHOW_EDGE_LABELS=1|0 REL_EXCLUDED_EDGES=[visible_to,near] REL_INCLUDED_EDGES=[in,held_by] REL_ABSTRACT_LAYOUT=1|0' \
	'  make visualize-relation-samples Render several graph-rich scene/action snapshots with node and edge legends.' \
	'    Vars: REL_HDF5_FILE=path/to/file.hdf5 REL_SAMPLE_FRAMES=0,74,124,182,191 REL_OUTPUT_DIR= (default: <episode>/visualizations/scene_graph/) REL_SHOW_EDGE_LABELS=1|0 REL_EXCLUDED_EDGES=[visible_to,near] REL_INCLUDED_EDGES=[in,held_by] REL_ABSTRACT_LAYOUT=1|0' \
	'' \
	'Occluder / reachability analysis (issue #35):' \
	'  make occluder-visibility      Occluder visibility sweep (+rollout with ROLLOUT=1).' \
	'    Vars: OCC_DISTANCE_CM=20,20 OCC_SEED_START=0 OCC_NUM_SEEDS=50 SAVE_IMAGES=0|1 ROLLOUT=0|1 CUROBO_TRAJOPT_SEEDS=16 CUROBO_MAX_ATTEMPTS=24' \
	'      CUROBO_FINETUNE_ATTEMPTS= CUROBO_FINETUNE_DT_SCALE= (empty = CuRobo default 5 / 0.85)' \
	'      CUROBO_ATTACH_SPHERE_RADIUS=0.001 LOCAL_WAYPOINT_ATTEMPTS=5 ATTACHED_TRAJECTORY_SLOWDOWN=2 WAYPOINT_SHRINK_MIN_DISTANCE=0.05 OUT_DIR= (timestamp a validation run, e.g. results/2026-07-07-10-10-12)' \
	'  make analyze-occluder-rollout Summarize saved rollout success/failure modes.' \
	'  make reachability-map         Collision-free gripper IK reachability map (one scene).' \
	'    Vars: REACH_SEED=1 OFFSET=0.2 REACH_ARMS=both|left|right REACH_Z=0.90' \
	'  make pickup-reachability      Per-seed post-pickup reachability maps (backward subgoals).' \
	'    Vars: PICKUP_SEEDS=1,2,3 OFFSET=0.2 REACH_Z=0.90' \
	'' \
	'Data collection:' \
	'  make collect-data             Run collect_data.sh for one task/config.' \
	'  make relation-validation      Collect graph-rich dense-scene validation data.' \
	'    Vars: TASK_NAME=put_sauce_can_in_basket GPU_ID=0 RELATION_EPISODE_NUM=1 RELATION_OBSTACLE_DENSITY=14 RELATION_SAVE_PATH=./data/relation_validation_d14_actions_v2' \
	'      REACHABLE_BY_ENABLED=1 REACHABLE_BY_INTERVAL=10 REACHABLE_BY_MOVABLE_ONLY=1' \
	'      REACHABLE_BY_CACHE_UNCHANGED=1 REACHABLE_BY_POSE_DECIMALS=3' \
	'  make action-validation-suite  Collect and check the schema-1.5 cross-task matrix.' \
	'    Vars: GPU_ID=0 ACTION_VALIDATION_MODE=collect|check|all ACTION_VALIDATION_TASKS=id1,id2' \
	'      ACTION_VALIDATION_OUTPUT_ROOT=customized_robotwin/data/action_validation_suite_v1' \
	'      ACTION_VALIDATION_START_SEED=0 ACTION_VALIDATION_DRY_RUN=0|1 REACHABLE_BY_INTERVAL=10 ACTION_VALIDATION_OBSTACLE_DENSITY=10' \
	'  make check-action-validation-suite  Check existing suite outputs without collecting.' \
	'  make collect-rollout-pi05     Dual-env pi05 rollout collection.' \
	'' \
	'Policy eval:' \
	'  make eval-direct              Direct eval via script/eval_policy.py.' \
	'  make policy-server            Start script/policy_model_server.py.' \
	'  make eval-client              Start script/eval_policy_client.py.' \
	'  make eval-pi05-single         Use policy/pi05/eval.sh (single-process).' \
	'  make eval-pi05-double         Use policy/pi05/eval_double_env.sh.' \
	'' \
	'Common vars:' \
	'  TASK_NAME=$(TASK_NAME) TASK_CONFIG=$(TASK_CONFIG) BENCH_SUBDIR=$(BENCH_SUBDIR) SEED=$(SEED)' \
	'  GPU_ID=$(GPU_ID) GPU_SPEC=$(GPU_SPEC) RENDER_FREQ=$(RENDER_FREQ) VIEWER_CAMERA=$(VIEWER_CAMERA)' \
	'  NO_RENDER=$(NO_RENDER) ROLLOUT=$(ROLLOUT) SAVE_DATA=$(SAVE_DATA)' \
	'  POLICY_NAME=$(POLICY_NAME) POLICY_CONFIG=$(POLICY_CONFIG)' \
	'  TRAIN_CONFIG_NAME=$(TRAIN_CONFIG_NAME) MODEL_NAME=$(MODEL_NAME) CHECKPOINT_ID=$(CHECKPOINT_ID)' \
	'  CKPT_SETTING=$(CKPT_SETTING) INSTRUCTION_TYPE=$(INSTRUCTION_TYPE) TEST_NUM=$(TEST_NUM) PORT=$(PORT)' \
	'' \
	'Examples:' \
	'  make bootstrap' \
	'  make verify-rollout TASK_NAME=put_mouse_on_pad TASK_CONFIG=bench_demo_office_clean BENCH_SUBDIR=office' \
	'  make eval-direct POLICY_NAME=pi05 TRAIN_CONFIG_NAME=my_train MODEL_NAME=pi05_ckpt CHECKPOINT_ID=30000'

show-config:
	@printf '%s\n' \
	"ROOT_DIR=$(ROOT_DIR)" \
	"PYTHON=$(PYTHON)" \
	"TASK_NAME=$(TASK_NAME)" \
	"TASK_CONFIG=$(TASK_CONFIG)" \
	"BENCH_SUBDIR=$(BENCH_SUBDIR)" \
	"SEED=$(SEED)" \
	"GPU_ID=$(GPU_ID)" \
	"GPU_SPEC=$(GPU_SPEC)" \
	"POLICY_NAME=$(POLICY_NAME)" \
	"POLICY_CONFIG=$(POLICY_CONFIG)"

check-prereqs:
	@missing=""
	@command -v uv     >/dev/null 2>&1 || missing="$$missing uv"
	@command -v git    >/dev/null 2>&1 || missing="$$missing git"
	@command -v nvcc   >/dev/null 2>&1 || missing="$$missing nvcc"
	@command -v ffmpeg >/dev/null 2>&1 || missing="$$missing ffmpeg"
	@if [ -n "$$missing" ]; then \
		printf 'Missing required tools:%s\n' "$$missing" >&2; \
		printf '  nvcc   → sudo apt install nvidia-cuda-toolkit\n' >&2; \
		printf '  ffmpeg → sudo apt install ffmpeg\n' >&2; \
		exit 1; \
	fi
	@printf 'All prerequisites found.\n'

bootstrap:
	PYTHON_VERSION="$(PYTHON_VERSION)" bash "$(ROOT_DIR)/scripts/install/bootstrap_uv.sh"

sync:
	cd "$(ROOT_DIR)"
	"$(UV)" sync

download-assets:
	cd "$(ROOT_DIR)"
	cmd=("$(PYTHON)" scripts/install/download_assets.py --dest "$(ASSETS_DEST)")
	if [[ "$(KEEP_ZIPS)" == "1" ]]; then
		cmd+=(--keep-zips)
	fi
	"$${cmd[@]}"

link-assets:
	cd "$(ROOT_DIR)"
	mkdir -p "$(ASSETS_DEST)"
	if [[ "$(ASSETS_DEST)" != "$(ROOT_DIR)/benchmark/assets" ]]; then
		if [[ -e "$(ROOT_DIR)/benchmark/assets" && ! -L "$(ROOT_DIR)/benchmark/assets" ]]; then
			printf 'benchmark/assets already exists as a real directory.\n' >&2
			printf 'Move or remove it first, then re-run make link-assets ASSETS_DEST=%s\n' "$(ASSETS_DEST)" >&2
			exit 1
		fi
		ln -sfn "$(ASSETS_DEST)" "$(ROOT_DIR)/benchmark/assets"
		printf 'linked benchmark/assets -> %s\n' "$(ASSETS_DEST)"
	fi
	ln -sfn ../benchmark/assets customized_robotwin/assets
	printf 'linked customized_robotwin/assets -> ../benchmark/assets\n'

configure-curobo-assets:
	cd "$(ROOT_DIR)/benchmark/assets/embodiments/aloha-agilex"
	ASSETS_PATH="$(ASSETS_PATH)" "$(PYTHON)" -c 'from pathlib import Path; import os; assets_path = os.environ["ASSETS_PATH"]; [Path(f"curobo_{side}.yml").write_text(Path(f"curobo_{side}_tmp.yml").read_text(encoding="utf-8").replace("$${ASSETS_PATH}", assets_path), encoding="utf-8") for side in ("left", "right")]'
	printf 'generated curobo_left.yml and curobo_right.yml with ASSETS_PATH=$(ASSETS_PATH)\n'

patch-curobo-config:
	cd "$(ROOT_DIR)"
	"$(PYTHON)" scripts/install/patch_aloha_curobo.py

setup: link-assets configure-curobo-assets patch-curobo-config

render-test:
	$(call RUN_IN_CUSTOMIZED,$(PYTHON) script/test_render.py)

verify-scene:
	$(call RUN_IN_CUSTOMIZED,\
		cmd='$(PYTHON) script/bench_script/visualize_task_scene.py "$(TASK_NAME)" "$(TASK_CONFIG)" --seed "$(SEED)" --render-freq "$(RENDER_FREQ)" --viewer-camera "$(VIEWER_CAMERA)"'; \
		if [[ -n "$(BENCH_SUBDIR)" ]]; then cmd+=" --bench-subdir $(BENCH_SUBDIR)"; fi; \
		if [[ "$(NO_RENDER)" == "1" ]]; then cmd+=" --no-render"; fi; \
		if [[ -n "$(SAVE_PLAN_FAIL_DIR)" ]]; then cmd+=" --save-plan-fail-dir $(SAVE_PLAN_FAIL_DIR) --plan-fail-camera $(PLAN_FAIL_CAMERA)"; fi; \
		eval "$$cmd")

verify-rollout:
	$(call RUN_IN_CUSTOMIZED,\
		cmd='$(PYTHON) script/bench_script/visualize_task_scene.py "$(TASK_NAME)" "$(TASK_CONFIG)" --seed "$(SEED)" --render-freq "$(RENDER_FREQ)" --viewer-camera "$(VIEWER_CAMERA)"'; \
		if [[ -n "$(BENCH_SUBDIR)" ]]; then cmd+=" --bench-subdir $(BENCH_SUBDIR)"; fi; \
		if [[ "$(ROLLOUT)" == "1" ]]; then cmd+=" --rollout"; fi; \
		if [[ "$(NO_RENDER)" == "1" ]]; then cmd+=" --no-render"; fi; \
		if [[ "$(SAVE_DATA)" == "1" ]]; then cmd+=" --save_data"; fi; \
		if [[ -n "$(SAVE_PLAN_FAIL_DIR)" ]]; then cmd+=" --save-plan-fail-dir $(SAVE_PLAN_FAIL_DIR) --plan-fail-camera $(PLAN_FAIL_CAMERA)"; fi; \
		eval "$$cmd")

collect-data:
	$(call RUN_IN_CUSTOMIZED,bash collect_data.sh "$(TASK_NAME)" "$(TASK_CONFIG)" "$(GPU_ID)")

relation-validation: TASK_NAME = put_sauce_can_in_basket
relation-validation: TASK_CONFIG = relation_validation_d14
relation-validation:
	$(call RUN_IN_CUSTOMIZED,\
		export ROBOPRO_REACHABLE_BY_ENABLED="$(REACHABLE_BY_ENABLED)"; \
		export ROBOPRO_REACHABLE_BY_FRAME_STRIDE="$(REACHABLE_BY_INTERVAL)"; \
		export ROBOPRO_REACHABLE_BY_MOVABLE_ONLY="$(REACHABLE_BY_MOVABLE_ONLY)"; \
		export ROBOPRO_REACHABLE_BY_CACHE_UNCHANGED="$(REACHABLE_BY_CACHE_UNCHANGED)"; \
		export ROBOPRO_REACHABLE_BY_POSE_DECIMALS="$(REACHABLE_BY_POSE_DECIMALS)"; \
		export ROBOPRO_RELATION_OBSTACLE_DENSITY="$(RELATION_OBSTACLE_DENSITY)"; \
		export ROBOPRO_RELATION_EPISODE_NUM="$(RELATION_EPISODE_NUM)"; \
		export ROBOPRO_RELATION_SAVE_PATH="$(RELATION_SAVE_PATH)"; \
		bash collect_data.sh "$(TASK_NAME)" "$(TASK_CONFIG)" "$(GPU_ID)")

action-validation-suite:
	$(call RUN_IN_CUSTOMIZED,\
		cmd='$(PYTHON) ../benchmark/bench_script/run_action_validation_suite.py "$(ACTION_VALIDATION_MODE)" --matrix "$(ACTION_VALIDATION_MATRIX)" --output-root "$(abspath $(ACTION_VALIDATION_OUTPUT_ROOT))" --gpu "$(GPU_ID)" --start-seed "$(ACTION_VALIDATION_START_SEED)" --reachability-interval "$(REACHABLE_BY_INTERVAL)" --obstacle-density "$(ACTION_VALIDATION_OBSTACLE_DENSITY)"'; \
		if [[ -n "$(ACTION_VALIDATION_TASKS)" ]]; then cmd+=" --tasks $(ACTION_VALIDATION_TASKS)"; fi; \
		if [[ "$(ACTION_VALIDATION_DRY_RUN)" == "1" ]]; then cmd+=" --dry-run"; fi; \
		eval "$$cmd")

check-action-validation-suite:
	$(call RUN_IN_CUSTOMIZED,\
		cmd='$(PYTHON) ../benchmark/bench_script/run_action_validation_suite.py check --matrix "$(ACTION_VALIDATION_MATRIX)" --output-root "$(abspath $(ACTION_VALIDATION_OUTPUT_ROOT))"'; \
		if [[ -n "$(ACTION_VALIDATION_TASKS)" ]]; then cmd+=" --tasks $(ACTION_VALIDATION_TASKS)"; fi; \
		eval "$$cmd")

precollect-seeds:
	$(call RUN_IN_CUSTOMIZED,$(PYTHON) script/precollect_eval_seeds.py "$(TASK_NAME)" "$(TASK_CONFIG)")

eval-direct:
	$(call RUN_IN_CUSTOMIZED,\
		$(PYTHON) script/eval_policy.py \
			--config "$(POLICY_CONFIG)" \
			--overrides \
			--task_name "$(TASK_NAME)" \
			--task_config "$(TASK_CONFIG)" \
			--train_config_name "$(TRAIN_CONFIG_NAME)" \
			--model_name "$(MODEL_NAME)" \
			--checkpoint_id "$(CHECKPOINT_ID)" \
			--ckpt_setting "$(CKPT_SETTING)" \
			--policy_name "$(POLICY_NAME)" \
			--seed "$(SEED)" \
			--instruction_type "$(INSTRUCTION_TYPE)" \
			--test_num "$(TEST_NUM)")

policy-server:
	$(call RUN_IN_CUSTOMIZED,\
		$(PYTHON) script/policy_model_server.py \
			--port "$(PORT)" \
			--config "$(POLICY_CONFIG)" \
			--overrides \
			--task_name "$(TASK_NAME)" \
			--task_config "$(TASK_CONFIG)" \
			--train_config_name "$(TRAIN_CONFIG_NAME)" \
			--model_name "$(MODEL_NAME)" \
			--checkpoint_id "$(CHECKPOINT_ID)" \
			--ckpt_setting "$(CKPT_SETTING)" \
			--policy_name "$(POLICY_NAME)" \
			--seed "$(SEED)")

eval-client:
	$(call RUN_IN_CUSTOMIZED,\
		$(PYTHON) script/eval_policy_client.py \
			--port "$(PORT)" \
			--config "$(POLICY_CONFIG)" \
			--overrides \
			--task_name "$(TASK_NAME)" \
			--task_config "$(TASK_CONFIG)" \
			--train_config_name "$(TRAIN_CONFIG_NAME)" \
			--model_name "$(MODEL_NAME)" \
			--checkpoint_id "$(CHECKPOINT_ID)" \
			--ckpt_setting "$(CKPT_SETTING)" \
			--policy_name "$(POLICY_NAME)" \
			--seed "$(SEED)" \
			--instruction_type "$(INSTRUCTION_TYPE)" \
			--test_num "$(TEST_NUM)")

eval-pi05-single:
	$(call RUN_IN_CUSTOMIZED,bash policy/pi05/eval.sh "$(TASK_NAME)" "$(TASK_CONFIG)" "$(TRAIN_CONFIG_NAME)" "$(MODEL_NAME)" "$(CHECKPOINT_ID)" "$(CKPT_SETTING)" "$(SEED)" "$(GPU_ID)")

eval-pi05-double:
	$(call RUN_IN_CUSTOMIZED,bash policy/pi05/eval_double_env.sh "$(TASK_NAME)" "$(TASK_CONFIG)" "$(TRAIN_CONFIG_NAME)" "$(MODEL_NAME)" "$(CHECKPOINT_ID)" "$(SEED)" "$(GPU_SPEC)")

collect-rollout-pi05:
	$(call RUN_IN_CUSTOMIZED,\
		export COLLECT_NUM="$(COLLECT_NUM)"; \
		if [[ -n "$(COLLECT_START_SEED)" ]]; then export COLLECT_START_SEED="$(COLLECT_START_SEED)"; fi; \
		export COLLECT_BRANCH_NUM="$(COLLECT_BRANCH_NUM)"; \
		export COLLECT_BRANCH_LOOKBACK="$(COLLECT_BRANCH_LOOKBACK)"; \
		export COLLECT_BRANCH_NOISE_STEPS="$(COLLECT_BRANCH_NOISE_STEPS)"; \
		export ACTION_NOISE_VAR="$(ACTION_NOISE_VAR)"; \
		if [[ "$(COLLECT_FIXED_SEED)" == "1" ]]; then export COLLECT_FIXED_SEED=1; fi; \
		bash policy/pi05/collect_rollout.sh "$(TASK_NAME)" "$(TASK_CONFIG)" "$(TRAIN_CONFIG_NAME)" "$(MODEL_NAME)" "$(CHECKPOINT_ID)" "$(SEED)" "$(GPU_SPEC)")

diag-kitchen-curobo:
	$(call RUN_IN_CUSTOMIZED,$(PYTHON) script/bench_script/diag_kitchen_curobo.py)

inspect-benchmark-hdf5:
	@if [[ -z "$(HDF5_FILE)" ]]; then \
		printf 'Set HDF5_FILE=/abs/path/to/episode.hdf5\n' >&2; \
		exit 1; \
	fi
	hdf5_file="$(HDF5_FILE)"; \
	if [[ "$$hdf5_file" != /* ]]; then hdf5_file="$(ROOT_DIR)/$$hdf5_file"; fi; \
	preview_path="$(HDF5_PREVIEW_PATH)"; \
	if [[ -n "$$preview_path" && "$$preview_path" != /* ]]; then preview_path="$(ROOT_DIR)/$$preview_path"; fi; \
	cmd='$(PYTHON) ../benchmark/bench_script/inspect_benchmark_hdf5.py --file "'"$$hdf5_file"'"'; \
	if [[ "$(SHOW_TREE)" == "1" ]]; then cmd+=" --show-tree"; fi; \
	if [[ "$(DUMP_JSON)" == "1" ]]; then cmd+=" --dump-json"; fi; \
	if [[ -n "$(HDF5_CAMERA)" ]]; then cmd+=" --camera $(HDF5_CAMERA)"; fi; \
	if [[ -n "$$preview_path" ]]; then cmd+=' --save-preview "'"$$preview_path"'"'; fi; \
	cmd+=" --frame $(HDF5_FRAME)"; \
	$(call RUN_IN_CUSTOMIZED,eval "$$cmd")

visualize-benchmark-rollout:
	@if [[ -z "$(VIZ_HDF5_FILE)" ]]; then \
		printf 'Set VIZ_HDF5_FILE=/abs/path/to/episode.hdf5\n' >&2; \
		exit 1; \
	fi
	hdf5_file="$(VIZ_HDF5_FILE)"; \
	if [[ "$$hdf5_file" != /* ]]; then hdf5_file="$(ROOT_DIR)/$$hdf5_file"; fi; \
	output_video="$(VIZ_OUTPUT_VIDEO)"; \
	if [[ -z "$$output_video" ]]; then \
		stem="$$(basename "$$hdf5_file" .hdf5)"; \
		output_video="$(ROOT_DIR)/tmp/$${stem}_benchmark_debug.mp4"; \
	fi; \
	if [[ "$$output_video" != /* ]]; then output_video="$(ROOT_DIR)/$$output_video"; fi; \
	cmd='$(PYTHON) ../benchmark/bench_script/visualize_benchmark_rollout.py --file "'"$$hdf5_file"'" --output "'"$$output_video"'"'; \
	cmd+=" --width $(VIZ_WIDTH) --height $(VIZ_HEIGHT) --fps $(VIZ_FPS) --trail $(VIZ_TRAIL)"; \
	$(call RUN_IN_CUSTOMIZED,eval "$$cmd")

visualize-relation-frame:
	@if [[ -z "$(REL_HDF5_FILE)" ]]; then \
		printf 'Set REL_HDF5_FILE=/abs/path/to/episode.hdf5\n' >&2; \
		exit 1; \
	fi
	hdf5_file="$(REL_HDF5_FILE)"; \
	if [[ "$$hdf5_file" != /* ]]; then hdf5_file="$(ROOT_DIR)/$$hdf5_file"; fi; \
	output_image="$(REL_OUTPUT_IMAGE)"; \
	if [[ -z "$$output_image" ]]; then \
		stem="$$(basename "$$hdf5_file" .hdf5)"; \
		episode_dir="$$(dirname "$$(dirname "$$hdf5_file")")"; \
		output_image="$$episode_dir/visualizations/scene_graph/$${stem}_frame_$$(printf '%04d' $(REL_FRAME)).png"; \
	fi; \
	if [[ "$$output_image" != /* ]]; then output_image="$(ROOT_DIR)/$$output_image"; fi; \
	cmd='$(PYTHON) ../benchmark/bench_script/visualize_relation_frame.py --file "'"$$hdf5_file"'" --frame "$(REL_FRAME)" --output "'"$$output_image"'"'; \
	cmd+=" --width $(REL_WIDTH) --height $(REL_HEIGHT) --show-edge-labels $(REL_SHOW_EDGE_LABELS) --excluded-edges '$(REL_EXCLUDED_EDGES)' --included-edges '$(REL_INCLUDED_EDGES)' --abstract-layout $(REL_ABSTRACT_LAYOUT)"; \
	$(call RUN_IN_CUSTOMIZED,eval "$$cmd")

visualize-relation-samples:
	@if [[ -z "$(REL_HDF5_FILE)" ]]; then \
		printf 'Set REL_HDF5_FILE=/abs/path/to/episode.hdf5\n' >&2; \
		exit 1; \
	fi
	hdf5_file="$(REL_HDF5_FILE)"; \
	if [[ "$$hdf5_file" != /* ]]; then hdf5_file="$(ROOT_DIR)/$$hdf5_file"; fi; \
	output_dir="$(REL_OUTPUT_DIR)"; \
	if [[ -z "$$output_dir" ]]; then output_dir="$$(dirname "$$(dirname "$$hdf5_file")")/visualizations/scene_graph"; fi; \
	if [[ "$$output_dir" != /* ]]; then output_dir="$(ROOT_DIR)/$$output_dir"; fi; \
	mkdir -p "$$output_dir"; \
	IFS=',' read -ra frames <<< "$(REL_SAMPLE_FRAMES)"; \
	for frame in "$${frames[@]}"; do \
		output_image="$$output_dir/$$(basename "$$hdf5_file" .hdf5)_frame_$$(printf '%04d' "$$frame").png"; \
		cmd='$(PYTHON) ../benchmark/bench_script/visualize_relation_frame.py --file "'"$$hdf5_file"'" --frame '"$$frame"' --output "'"$$output_image"'"'; \
		cmd+=" --width $(REL_WIDTH) --height $(REL_HEIGHT) --show-edge-labels $(REL_SHOW_EDGE_LABELS) --excluded-edges '$(REL_EXCLUDED_EDGES)' --included-edges '$(REL_INCLUDED_EDGES)' --abstract-layout $(REL_ABSTRACT_LAYOUT)"; \
		$(call RUN_IN_CUSTOMIZED,eval "$$cmd") || exit $$?; \
	done

occluder-visibility:
	$(call RUN_IN_CUSTOMIZED,\
		export CUROBO_TRAJOPT_SEEDS="$(CUROBO_TRAJOPT_SEEDS)"; \
		export CUROBO_MAX_ATTEMPTS="$(CUROBO_MAX_ATTEMPTS)"; \
		export CUROBO_BATCH_GRAPH_SEEDS="$(CUROBO_BATCH_GRAPH_SEEDS)"; \
		export CUROBO_FINETUNE_ATTEMPTS="$(CUROBO_FINETUNE_ATTEMPTS)"; \
		export CUROBO_FINETUNE_DT_SCALE="$(CUROBO_FINETUNE_DT_SCALE)"; \
		export CUROBO_ATTACH_SPHERE_RADIUS="$(CUROBO_ATTACH_SPHERE_RADIUS)"; \
		export LOCAL_WAYPOINT_ATTEMPTS="$(LOCAL_WAYPOINT_ATTEMPTS)"; \
		export ATTACHED_TRAJECTORY_SLOWDOWN="$(ATTACHED_TRAJECTORY_SLOWDOWN)"; \
		export WAYPOINT_SHRINK_MIN_DISTANCE="$(WAYPOINT_SHRINK_MIN_DISTANCE)"; \
		export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"; \
		cmd='$(PYTHON) script/bench_script/analyze_occluder_visibility.py --base-config "$(TASK_CONFIG)" --seed-start "$(OCC_SEED_START)" --num-seeds "$(OCC_NUM_SEEDS)" --occluding-object-distance "$(OCC_DISTANCE_CM)"'; \
		if [[ "$(ROLLOUT)" == "1" ]]; then cmd+=" --rollout"; fi; \
		if [[ "$(SAVE_IMAGES)" == "1" ]]; then cmd+=" --save-images"; fi; \
		if [[ -n "$(OUT_DIR)" ]]; then cmd+=" --out-dir \"$(OUT_DIR)\""; fi; \
		eval "$$cmd")

reachability-map:
	$(call RUN_IN_CUSTOMIZED,\
		$(PYTHON) script/bench_script/reachability_map.py --base-config "$(TASK_CONFIG)" \
			--seed "$(REACH_SEED)" --offset "$(OFFSET)" --arms "$(REACH_ARMS)" --z "$(REACH_Z)")

pickup-reachability:
	$(call RUN_IN_CUSTOMIZED,\
		$(PYTHON) script/bench_script/pickup_reachability_map.py --base-config "$(TASK_CONFIG)" \
			--seeds "$(PICKUP_SEEDS)" --offset "$(OFFSET)" --z "$(REACH_Z)")

analyze-occluder-rollout:
	"$(PYTHON)" scripts/validation/analyze_occluder_rollout_failures.py
