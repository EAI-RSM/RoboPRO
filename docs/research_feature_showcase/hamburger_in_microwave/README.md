# Hamburger-microwave physics-aware graph: pilot

This pilot pairs each exact exported camera frame with two graph views at the same HDF5 index: the authoritative `full_scene_graph` and the policy-facing `action_relevant_subgraph`.

| Frame | Evidence stage | Interpretation |
|---:|---|---|
| 122 | Contained Before Contact | The hamburger is inside the microwave while transport remains active. |
| 123 | Container Contact Established | Legacy non-support contact evidence appears between the hamburger and microwave. |
| 128 | Contact Re-established | The transient object-container contact appears again during transport. |
| 142 | Release Phase | The hamburger remains contained as the left gripper begins release. |

## Tool-calling transition

The graph grounds the target, destination, and acting arm. The exported provider-neutral call is:

```json
{
  "contract_version": "1.0.0",
  "decision": "ACT",
  "provenance": "expert_executed_action",
  "provider": "rule_based",
  "result": {
    "end_frame": 137,
    "observed_effects": [],
    "start_frame": 116,
    "status": "succeeded"
  },
  "tool": {
    "arguments": {
      "action_type": "transport",
      "arm": "left",
      "destination_object_id": null,
      "effector_object_id": -2,
      "parameters": {
        "primitive": "move",
        "target_pose": [
          -0.39495617666297156,
          0.12133662586186597,
          0.8443040251731873,
          0.696259081379631,
          -0.12276926161051974,
          0.12276926161051974,
          0.696259081379631
        ]
      },
      "target_object_id": 112
    },
    "name": "execute_high_level_action"
  }
}
```

The graph distinguishes persistent containment from transient object-container contact during transport.

## Scope

- `visible_to` means at least one segmentation pixel, not full visibility.
- `reachable_by` is collision-aware IK, not full grasp or trajectory feasibility.
- `occludes` and `blocks` are not claimed as canonical relations.
