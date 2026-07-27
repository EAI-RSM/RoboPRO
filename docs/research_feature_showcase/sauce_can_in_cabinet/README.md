# Unified physics- and action-aware graph: pilot

This pilot pairs each exact exported camera frame with two graph views at the same HDF5 index: the authoritative `full_scene_graph` and the policy-facing `action_relevant_subgraph`.

| Frame | Evidence stage | Interpretation |
|---:|---|---|
| 0 | Initial state | The expert begins an approach; target, destination, camera, and end effector are grounded. |
| 72 | Active grasp | held_by becomes true: the target is physically associated with the right end effector. |
| 152 | Physical change | in(target, cabinet) and inverse contains(cabinet, target) become true. |
| 205 | Verified result | Containment persists after release while verify_success checks the resulting state. |

## Tool-calling transition

The graph grounds the target, destination, and acting arm. The exported provider-neutral call is:

```json
{
  "contract_version": "1.0.0",
  "decision": "ACT",
  "provenance": "expert_executed_action",
  "provider": "rule_based",
  "result": {
    "end_frame": 81,
    "observed_effects": [
      {
        "after": true,
        "before": false,
        "destination": -3,
        "relation": "held_by",
        "source": 92
      }
    ],
    "start_frame": 60,
    "status": "succeeded"
  },
  "tool": {
    "arguments": {
      "action_type": "grasp",
      "arm": "right",
      "destination_object_id": null,
      "effector_object_id": -3,
      "parameters": {
        "primitive": "gripper",
        "target_gripper_pos": 0.0
      },
      "target_object_id": 92
    },
    "name": "execute_high_level_action"
  }
}
```

The resulting graph retains `in(task_sauce_can, cabinet)` and inverse `contains(cabinet, task_sauce_can)` through success verification.

## Scope

- `visible_to` means at least one segmentation pixel, not full visibility.
- `reachable_by` is collision-aware IK, not full grasp or trajectory feasibility.
- `occludes` and `blocks` are not claimed as canonical relations.
