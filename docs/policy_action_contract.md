# Policy-facing action contract

Schema `1.6.0` exports one provider-neutral representation of every high-level
action under `benchmark_support/action_nodes/tool_calls_json`. The default
provider is the repository's rule-based expert planner.

Each entry uses this envelope:

```json
{
  "contract_version": "1.0.0",
  "decision": "ACT",
  "provider": "rule_based",
  "provenance": "expert_executed_action",
  "tool": {
    "name": "execute_high_level_action",
    "arguments": {
      "action_type": "place",
      "arm": "right",
      "target_object_id": 7,
      "destination_object_id": 12,
      "effector_object_id": -3,
      "parameters": {}
    }
  },
  "result": {
    "status": "succeeded",
    "start_frame": 154,
    "end_frame": 169,
    "observed_effects": []
  }
}
```

The authoritative JSON input/output schema, resolved provider, and provider
registry are embedded below `benchmark_support/policy_action_contract`.

## Provider selection

For expert benchmark collection:

```bash
make action-validation-suite \
  POLICY_ACTION_PROVIDER=rule_based \
  POLICY_ACTION_PROVIDER_CONFIG= \
  GPU_ID=0
```

`rule_based` is the only valid provider for `collect_data.py`, because that
collector executes the expert scripts. Selecting a model there raises an error
instead of incorrectly relabeling expert actions.

For policy rollouts, use the existing `POLICY_NAME` selection. The rollout
collector stamps the actual model provider automatically:

```bash
make eval-direct POLICY_NAME=pi0
make eval-direct POLICY_NAME=pi05
```

Provider aliases normalize as follows:

| User value | Canonical provider | Current status | Exported representation |
|---|---|---|---|
| `rule_based`, `expert` | `rule_based` | available, default | high-level action nodes/tool calls |
| `Pi_0`, `pi_0`, `pi0` | `pi0` | adapter present | low-level policy commands |
| `Pi_0.5`, `pi_0.5`, `pi05` | `pi05` | adapter present | low-level policy commands |
| `XVLA`, `xvla` | `xvla` | adapter not installed | `adapter_required` |

Pi0/Pi0.5 rollout commands are not silently promoted to semantic high-level
actions. A future adapter must explicitly map model output to this contract and
provide grounded targets, intervals, status, and observed effects. This keeps
training labels honest and makes the downstream repository independent of the
model implementation.

The code registry is in
`customized_robotwin/envs/utils/policy_action_contract.py`; the human-readable
registry is `benchmark/bench_task_config/policy_action_providers.yml`.
