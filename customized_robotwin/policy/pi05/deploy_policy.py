import os
import sys
from pathlib import Path

import numpy as np

repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))

from experiments.graph_conditioned_pi05.contract import (
    GRAPH_TREATMENT_VERSION,
    InputCondition,
    RetrievalContract,
)
from experiments.graph_conditioned_pi05.live_adapter import (
    action_graph_state,
    build_live_graph_context,
    keep_active_gripper_closed,
    prepare_instruction,
)
from experiments.graph_conditioned_pi05.graph_replanning import (
    GraphControllerState,
    PromptPhase,
)
from experiments.graph_conditioned_pi05.action_diagnostics import graph_evidence
from experiments.graph_conditioned_pi05.simulator_evidence import (
    extract_simulator_evidence,
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


def _controller(task_env):
    controller = getattr(task_env, "_graph_controller", None)
    if controller is None:
        controller = GraphControllerState()
        task_env._graph_controller = controller
    return controller


def _prepare_graph_prompt(task_env, model, observation, controller):
    task_env._graph_treatment_version = GRAPH_TREATMENT_VERSION
    prepared = prepare_instruction(
        task_env, model, observation, _GRAPH_CONDITION, _GRAPH_CONTRACT,
        previous_phase=controller.phase.value,
    )
    active_prompt = getattr(task_env, "_graph_active_prompt", None)
    prompt_updated = prepared.instruction != active_prompt
    if model.observation_window is None or prompt_updated:
        model.set_language(prepared.instruction)
        task_env._graph_active_prompt = prepared.instruction
    task_env._graph_prompt_phase = controller.phase.value
    task_env._graph_held_arm = controller.held_arm
    task_env._graph_active_intent = (
        prepared.action_intent.as_dict() if prepared.action_intent else None
    )
    stats = getattr(task_env, "_graph_conditioning_stats", None)
    if stats is None:
        stats = []
        task_env._graph_conditioning_stats = stats
    stats.append(
        {
            "retrieved_nodes": prepared.retrieved_node_count,
            "selected_nodes": prepared.selected_node_count,
            "dropped_nodes": prepared.dropped_node_count,
            "retrieved": prepared.retrieved_fact_count,
            "selected": prepared.selected_fact_count,
            "dropped": prepared.dropped_fact_count,
            "graph_tokens": prepared.graph_token_count,
            "full_prompt_tokens_estimate": prepared.full_prompt_token_count_estimate,
            "destination_seed_available": prepared.destination_seed_available,
            "prompt_phase": controller.phase.value,
            "prompt_updated": prompt_updated,
            "prompt": prepared.instruction,
            "action_intent": task_env._graph_active_intent,
        }
    )


def _record_graph_observation(task_env, controller, state, remaining_actions):
    decision, record = controller.observe(state, remaining_actions)
    events = getattr(task_env, "_graph_delta_events", None)
    if events is None:
        events = []
        task_env._graph_delta_events = events
    if record["events"] or record["persistence"] or record["requires_replan"]:
        events.append(record)
    task_env._graph_prompt_phase = controller.phase.value
    task_env._graph_held_arm = controller.held_arm
    if decision.requires_replan:
        task_env._graph_chunk_interrupts += 1
    return decision.requires_replan


def _execute_action_chunk(task_env, model, observation, actions, controller):
    for index, action in enumerate(actions):
        executed_action = (
            keep_active_gripper_closed(action, controller.held_arm)
            if controller.phase is PromptPhase.PLACEMENT
            else action
        )
        task_env.take_action(executed_action)
        controller.actions_since_replan += 1
        observation = task_env.get_obs()
        context = build_live_graph_context(task_env, observation, _GRAPH_CONTRACT)
        evidence = extract_simulator_evidence(context)
        _record_action_trace(
            task_env, action, executed_action, observation, controller,
            context, evidence,
        )
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)
        state = action_graph_state(
            task_env, observation, _GRAPH_CONTRACT,
            context=context, evidence=evidence,
        )
        if _record_graph_observation(
            task_env, controller, state, len(actions) - index - 1
        ):
            break


def _record_action_trace(
    task_env, raw_action, executed_action, observation, controller=None,
    graph_context=None, simulator_evidence=None,
):
    recorder = getattr(task_env, "_action_trace_recorder", None)
    if recorder is None:
        return
    try:
        trace_evidence = graph_evidence(
            task_env, observation, context=graph_context,
            evidence=simulator_evidence,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        # Diagnostics must not make an otherwise valid rollout fail. Missing
        # graph support remains visible as absent/NaN fields in the trace.
        trace_evidence = {}
    recorder.record(
        frame=task_env.take_action_cnt,
        prompt=getattr(task_env, "_graph_active_prompt", None) or task_env.get_instruction(),
        phase=(controller.phase.value if controller is not None else "task"),
        raw_action=raw_action,
        executed_action=executed_action,
        observation=observation,
        evidence=trace_evidence,
        action_intent=getattr(task_env, "_graph_active_intent", None),
    )


def eval(TASK_ENV, model, observation):
    controller = None
    if _GRAPH_CONDITION is InputCondition.VISUAL_RETRIEVED_GRAPH:
        controller = _controller(TASK_ENV)
        if controller.frame == 0:
            _record_graph_observation(
                TASK_ENV,
                controller,
                action_graph_state(TASK_ENV, observation, _GRAPH_CONTRACT),
                0,
            )
        _prepare_graph_prompt(TASK_ENV, model, observation, controller)
    elif model.observation_window is None:
        model.set_language(TASK_ENV.get_instruction())
    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)
    actions = model.get_action()[:model.pi0_step]
    if controller is not None:
        _execute_action_chunk(TASK_ENV, model, observation, actions, controller)
        return
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        _record_action_trace(TASK_ENV, action, action, observation)
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()
