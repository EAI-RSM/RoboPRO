import json
import tempfile
import unittest

import h5py
import numpy as np

from envs._base_task import Base_Task
from envs.utils.policy_action_contract import (
    CONTRACT_VERSION,
    action_node_to_tool_call,
    resolve_provider,
    tool_schema,
)


class PolicyActionContractTest(unittest.TestCase):
    def test_provider_aliases_and_availability(self):
        self.assertEqual(resolve_provider(None)["name"], "rule_based")
        self.assertEqual(resolve_provider("Pi_0")["name"], "pi0")
        self.assertEqual(resolve_provider("pi_0.5")["name"], "pi05")
        self.assertFalse(resolve_provider("XVLA")["available"])

    def test_action_node_maps_to_provider_neutral_tool_call(self):
        node = {
            "action_type": "place", "arm": "right", "target_object_id": 7,
            "destination_object_id": 12, "effector_object_id": -3,
            "parameters": {"speed": 0.2}, "status": "succeeded",
            "start_frame": 10, "end_frame": 20,
            "observed_effects": [{"relation": "in", "after": True}],
            "provenance": "expert_executed_action",
        }
        call = action_node_to_tool_call(node, resolve_provider("rule_based"))
        self.assertEqual(call["contract_version"], CONTRACT_VERSION)
        self.assertEqual(call["decision"], "ACT")
        self.assertEqual(call["provider"], "rule_based")
        self.assertEqual(call["tool"]["name"], "execute_high_level_action")
        self.assertEqual(call["tool"]["arguments"]["target_object_id"], 7)
        self.assertEqual(call["result"]["status"], "succeeded")

    def test_hdf5_export_embeds_contract_and_tool_calls(self):
        task = Base_Task.__new__(Base_Task)
        task._benchmark_action_provider = resolve_provider("rule_based")
        task._benchmark_episode_record = {
            "schema_name": "robopro_benchmark_support",
            "schema_version": "1.6.0",
            "episode_id": "contract_smoke_0",
            "scenario_metadata": {
                "task_name": "contract_smoke", "task_config": "unit",
                "seed": 0, "success": True, "config_snapshot": {},
            },
            "object_catalog": [],
            "action_nodes": [{
                "action_id": 0, "action_type": "verify_success",
                "execution_phase": "success_check", "arm": "none",
                "start_frame": 0, "end_frame": 0, "recorded_frame_count": 1,
                "status": "succeeded", "target_object_id": None,
                "destination_object_id": None, "effector_object_id": None,
                "parameters": {}, "preconditions": [], "postconditions": [],
                "observed_effects": [], "provenance": "task_success_check",
            }],
        }
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as output:
            with h5py.File(output.name, "w") as root:
                support = root.create_group("benchmark_support")
                state = support.create_group("object_state")
                state.create_dataset("pose_world", data=np.zeros((1, 0, 7), dtype=np.float32))
            task._write_benchmark_metadata_to_hdf5(output.name)
            with h5py.File(output.name, "r") as root:
                contract = root["benchmark_support/policy_action_contract"]
                self.assertEqual(contract["provider_name"][()].decode(), "rule_based")
                calls = root["benchmark_support/action_nodes/tool_calls_json"][()]
                call = json.loads(calls[0].decode())
                self.assertEqual(call["decision"], "ACT")
                self.assertEqual(call["tool"]["arguments"]["action_type"], "verify_success")

    def test_tool_schema_requires_grounded_action_identity(self):
        schema = tool_schema()
        self.assertEqual(schema["name"], "execute_high_level_action")
        self.assertEqual(schema["input_schema"]["required"], ["action_type", "arm"])


if __name__ == "__main__":
    unittest.main()
