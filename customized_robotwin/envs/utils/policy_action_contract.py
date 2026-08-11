"""Provider-neutral high-level action/tool-call contract for benchmark export."""

from __future__ import annotations

from copy import deepcopy


CONTRACT_VERSION = "1.0.0"
TOOL_NAME = "execute_high_level_action"

_PROVIDER_ALIASES = {
    "expert": "rule_based",
    "expert_rule_based": "rule_based",
    "rule-based": "rule_based",
    "rule_based": "rule_based",
    "pi_0": "pi0",
    "pi0": "pi0",
    "pi_0.5": "pi05",
    "pi0.5": "pi05",
    "pi05": "pi05",
    "xvla": "xvla",
}

_PROVIDER_REGISTRY = {
    "rule_based": {
        "kind": "expert_planner",
        "module": None,
        "available": True,
        "action_representation": "high_level_action_nodes",
    },
    "pi0": {
        "kind": "policy_model",
        "module": "pi0",
        "available": True,
        "action_representation": "low_level_policy_commands",
    },
    "pi05": {
        "kind": "policy_model",
        "module": "pi05",
        "available": True,
        "action_representation": "low_level_policy_commands",
    },
    "xvla": {
        "kind": "policy_model",
        "module": "XVLA",
        "available": False,
        "action_representation": "adapter_required",
    },
}


def normalize_provider_name(name: str | None) -> str:
    value = "rule_based" if name in (None, "") else str(name).strip().lower()
    return _PROVIDER_ALIASES.get(value, value)


def resolve_provider(name: str | None, config_ref: str | None = None) -> dict:
    canonical = normalize_provider_name(name)
    base = deepcopy(_PROVIDER_REGISTRY.get(canonical, {
        "kind": "policy_model",
        "module": canonical,
        "available": False,
        "action_representation": "adapter_required",
    }))
    base.update({
        "name": canonical,
        "contract_version": CONTRACT_VERSION,
        "config_ref": None if config_ref in (None, "") else str(config_ref),
    })
    return base


def provider_registry() -> dict:
    return {name: resolve_provider(name) for name in _PROVIDER_REGISTRY}


def tool_schema() -> dict:
    return {
        "name": TOOL_NAME,
        "description": "Execute one grounded, temporally extended robot action.",
        "input_schema": {
            "type": "object",
            "required": ["action_type", "arm"],
            "properties": {
                "action_type": {"type": "string"},
                "arm": {"enum": ["left", "right", "none"]},
                "target_object_id": {"type": ["integer", "null"]},
                "destination_object_id": {"type": ["integer", "null"]},
                "effector_object_id": {"type": ["integer", "null"]},
                "parameters": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "required": ["status", "start_frame", "end_frame", "observed_effects"],
            "properties": {
                "status": {"enum": ["succeeded", "failed"]},
                "start_frame": {"type": "integer"},
                "end_frame": {"type": "integer"},
                "observed_effects": {"type": "array"},
            },
            "additionalProperties": False,
        },
    }


def action_node_to_tool_call(node: dict, provider: dict) -> dict:
    arguments = {
        "action_type": node.get("action_type"),
        "arm": node.get("arm", "none"),
        "target_object_id": node.get("target_object_id"),
        "destination_object_id": node.get("destination_object_id"),
        "effector_object_id": node.get("effector_object_id"),
        "parameters": deepcopy(node.get("parameters", {})),
    }
    result = {
        "status": node.get("status"),
        "start_frame": int(node.get("start_frame", 0)),
        "end_frame": int(node.get("end_frame", 0)),
        "observed_effects": deepcopy(node.get("observed_effects", [])),
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "decision": "ACT",
        "provider": provider["name"],
        "provenance": node.get("provenance"),
        "tool": {"name": TOOL_NAME, "arguments": arguments},
        "result": result,
    }
