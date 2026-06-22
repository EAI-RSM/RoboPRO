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

# Visualization flags
RENDER_FREQ ?= 3
VIEWER_CAMERA ?= demo_camera
NO_RENDER ?= 1
ROLLOUT ?= 1
SAVE_DATA ?= 1
SAVE_PLAN_FAIL_DIR ?=
PLAN_FAIL_CAMERA ?= demo_camera

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

define RUN_IN_CUSTOMIZED
	cd "$(CUSTOMIZED_ROOT)"
	source set_env.sh
	export ROBOTWIN_BENCH_TASK="$(ROBOTWIN_BENCH_TASK)"
	$(1)
endef

.PHONY: help check-prereqs bootstrap sync download-assets link-assets configure-curobo-assets \
	patch-curobo-config setup render-test verify-scene verify-rollout collect-data \
	precollect-seeds eval-direct eval-client policy-server eval-pi05-single eval-pi05-double \
	collect-rollout-pi05 diag-kitchen-curobo show-config

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
	'' \
	'Data collection:' \
	'  make collect-data             Run collect_data.sh for one task/config.' \
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
