import os
import sys
from pathlib import Path

import numpy as np

repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from experiments.graph_conditioned_pi05.contract import InputCondition, RetrievalContract
from experiments.graph_conditioned_pi05.live_adapter import (
    keep_active_gripper_closed,
    live_task_state,
    prepare_instruction,
)

parent_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_directory)

_GRAPH_CONDITION = InputCondition.VISUAL_ONLY
_GRAPH_CONTRACT = RetrievalContract()


def configure(settings):
    """Configure graph conditioning for this client; visual-only is default."""
    global _GRAPH_CONDITION, _GRAPH_CONTRACT
    _GRAPH_CONDITION = InputCondition(
        settings.get("graph_input_condition", InputCondition.VISUAL_ONLY.value)
    )
    _GRAPH_CONTRACT = RetrievalContract(
        graph_token_budget=int(settings.get("graph_token_budget", 120)),
        default_camera=str(settings.get("graph_default_camera", "countertop_camera")),
    )


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["countertop_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    return input_rgb_arr, observation["joint_action"]["vector"]


def get_model(usr_args):
    from pi_model import PI0

    return PI0(
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"],
    )


def eval(TASK_ENV, model, observation):
    if _GRAPH_CONDITION is InputCondition.VISUAL_RETRIEVED_GRAPH:
        previous_phase = getattr(TASK_ENV, "_graph_prompt_phase", "grasp")
        prepared = prepare_instruction(
            TASK_ENV, model, observation, _GRAPH_CONDITION, _GRAPH_CONTRACT,
            previous_phase=previous_phase,
        )
        TASK_ENV._graph_prompt_phase = prepared.prompt_phase
        active_prompt = getattr(TASK_ENV, "_graph_active_prompt", None)
        if model.observation_window is None or prepared.instruction != active_prompt:
            model.set_language(prepared.instruction)
            TASK_ENV._graph_active_prompt = prepared.instruction
        stats = getattr(TASK_ENV, "_graph_conditioning_stats", None)
        if stats is None:
            stats = []
            TASK_ENV._graph_conditioning_stats = stats
        stats.append(
            {
                "retrieved": prepared.retrieved_fact_count,
                "selected": prepared.selected_fact_count,
                "dropped": prepared.dropped_fact_count,
                "graph_tokens": prepared.graph_token_count,
                "full_prompt_tokens_estimate": prepared.full_prompt_token_count_estimate,
                "destination_seed_available": prepared.destination_seed_available,
                "prompt_phase": prepared.prompt_phase,
                "prompt_updated": prepared.instruction != active_prompt,
            }
        )
    elif model.observation_window is None:
        model.set_language(TASK_ENV.get_instruction())

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        phase = getattr(TASK_ENV, "_graph_prompt_phase", "grasp")
        held_arm = getattr(TASK_ENV, "_graph_held_arm", None)
        executed_action = (
            keep_active_gripper_closed(action, held_arm)
            if _GRAPH_CONDITION is InputCondition.VISUAL_RETRIEVED_GRAPH
            and phase == "placement"
            else action
        )
        TASK_ENV.take_action(executed_action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
        if _GRAPH_CONDITION is not InputCondition.VISUAL_RETRIEVED_GRAPH:
            continue

        event = live_task_state(TASK_ENV, observation, _GRAPH_CONTRACT)
        if phase == "grasp" and event.target_held:
            TASK_ENV._graph_prompt_phase = "placement"
            TASK_ENV._graph_held_arm = event.held_arm
            TASK_ENV._graph_chunk_interrupts += 1
            break
        if phase == "placement":
            if event.target_held:
                TASK_ENV._graph_held_arm = event.held_arm
                TASK_ENV._graph_held_loss_count = 0
            else:
                TASK_ENV._graph_held_loss_count += 1
                if TASK_ENV._graph_held_loss_count >= 2:
                    TASK_ENV._graph_prompt_phase = "grasp"
                    TASK_ENV._graph_held_arm = None
                    TASK_ENV._graph_chunk_interrupts += 1
                    break
            if event.target_inside_destination or event.release_ready:
                TASK_ENV._graph_prompt_phase = "release"
                TASK_ENV._graph_chunk_interrupts += 1
                break


def reset_model(model):
    model.reset_obsrvationwindows()
