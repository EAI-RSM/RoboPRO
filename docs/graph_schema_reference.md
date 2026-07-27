# RoboPRO Graph Schema Reference

This is the authoritative implementation-facing reference for the graph-rich
RoboPRO benchmark export. It describes schema version `1.6.0` as written by the
current exporter. Research ideas that are not yet exported are listed separately
under [Reserved and planned relations](#reserved-and-planned-relations).

## Graph model

Each episode is a heterogeneous temporal graph:

- persistent entity nodes describe scene objects, furniture, robot entities,
  and end effectors;
- action nodes describe temporally extended expert-planner actions;
- object state and relation tensors provide a graph snapshot at every saved frame;
- action-entity edges ground actions to agents, targets, and destinations; and
- action intervals and observed relation changes connect execution to graph state.

An edge is read from source to destination. For example:

```text
bottle --in--> basket
```

means “the bottle is in the basket.” Directed inverse pairs are stored
explicitly where noted. Symmetric relations are stored as symmetric matrices.

## HDF5 structure

Graph data is stored below `benchmark_support`:

```text
benchmark_support/
├── object_catalog/          persistent entity identities and attributes
├── object_state/            per-frame entity state
├── relation_state/          per-frame state-relation edges
├── action_nodes/            temporal actions plus tool_calls_json
├── policy_action_contract/  provider, registry, and tool schema
├── action_entity_edges/     action-to-entity participant edges
├── collision_metric_contact_events/  filtered auxiliary contact events
└── scenario_metadata/       task and collection context
```

The export identifies itself with:

```text
schema_name    = robopro_benchmark_support
schema_version = 1.6.0
```

## Node types

### Entity nodes

`object_catalog` is the persistent entity-node table. Every row has an integer
`object_id` and parallel fields:

| Field | Meaning |
|---|---|
| `names` | Simulator/export name. |
| `roles` | Task role such as `target`, `distractor`, `furniture`, or another role. |
| `entity_kinds` | Entity category used by the exporter. |
| `semantic_labels` | Human-readable semantic category. |
| `asset_refs` | Source asset reference when available. |
| `provenances` | Origin of the annotation. |
| `is_target` | Entity is a task target. |
| `is_distractor` | Entity is a distractor or clutter object. |
| `is_furniture` | Entity is furniture, a surface, or a receptacle. |
| `is_robot` | Entity belongs to the robot. |
| `is_articulated` | Entity has articulation. |
| `is_movable` | Entity is treated as movable. |
| `metadata_json` | Extensible entity metadata. |

Special robot identifiers are stable within the schema:

| ID | Name | Meaning |
|---:|---|---|
| `-1` | `robot` | Robot entity. |
| `-2` | `left_ee` | Left end effector. |
| `-3` | `right_ee` | Right end effector. |

Camera endpoints are named by `visible_to_camera_names`. They participate in
`visible_to` edges but are not currently rows in `object_catalog`.

Example entity node:

```json
{
  "object_id": 7,
  "name": "001_bottle",
  "role": "target",
  "semantic_label": "bottle",
  "is_movable": true,
  "provenance": "privileged"
}
```

### Per-frame entity state

`object_state` aligns entity state with saved benchmark frames. Important
datasets include `object_ids`, `pose_world`, and `is_present`. Relation tensor
axis `N` uses the `relation_state/object_ids` ordering; consumers must map by ID
rather than assume catalog row order.

### Action nodes

`action_nodes` contains actions actually issued by the rule-based expert. Each
node has:

- `action_ids`, `action_types`, and `execution_phases`;
- inclusive `start_frame` and `end_frame`, plus `recorded_frame_count`;
- `arms`, terminal `statuses`, and `provenance`;
  `expert_planner_attempt` denotes issued planning/search attempts and
  `expert_executed_action` denotes replayed execution;
- target, destination, and effector IDs with explicit `_valid` masks;
- `parameters_json`, `preconditions_json`, and `postconditions_json`;
- `observed_effects_json` and provider-neutral `tool_calls_json`; and
- `active[T,A]`, the per-frame action incidence matrix.

Example action node:

```json
{
  "action_id": 5,
  "action_type": "place",
  "execution_phase": "backward_placement",
  "start_frame": 154,
  "end_frame": 169,
  "arm": "right",
  "target_object_id": 7,
  "destination_object_id": 12,
  "effector_object_id": -3,
  "status": "succeeded"
}
```

Canonical action types include the object-manipulation sequence `approach`,
`grasp`, `lift`, `transport`, `place`, `release`, and `retreat`;
`verify_success`; and articulation-aware `approach_handle`, `grasp_handle`,
`open_articulation`, and `close_articulation`. Articulation actions target the
articulation entity. `parameters_json` identifies the interaction part and joint
index; handle parts are not yet independent catalog nodes.

Schema `1.6.0` also maps each action node to an `ACT` tool-call envelope. The
resolved provider, provider registry, and JSON tool schema are stored in
`policy_action_contract`; see [Policy-facing action contract](policy_action_contract.md).

Contact events, risk assessments, belief nodes, unissued candidate alternatives,
and rejected-plan lineage are research-roadmap concepts. Issued planner attempts
are retained as action nodes, including failed attempts, with zero
`recorded_frame_count` and no `active` frames. Filtered contact events are
exported as an auxiliary event table, not as graph nodes.

## Implemented state-relation edges

`T` is saved frames, `N` is catalogued relation entities, `E` is end effectors,
and `C` is cameras.

| Relation | Direction / shape | Meaning | Example |
|---|---|---|---|
| `on` | directed, `[T,N,N]` | Source rests on destination support. | `bottle --on--> table` |
| `supports` | directed, `[T,N,N]` | Source supports destination; exact inverse of `on`. | `table --supports--> bottle` |
| `in` | directed, `[T,N,N]` | Source is geometrically contained by destination. | `bottle --in--> basket` |
| `contains` | directed, `[T,N,N]` | Source contains destination; exact inverse of `in`. | `basket --contains--> bottle` |
| `held_by` | directed, `[T,N,E]` | Source is currently grasped by destination end effector. | `bottle --held_by--> right_ee` |
| `near` | symmetric, `[T,N,N]` | Entities are spatially close under the configured geometric threshold; contact is not required. | `bottle --near-- milk_box` |
| `reachable_by` | directed, `[T,N,E]` | Source position has a collision-aware, position-only IK solution for destination end effector. | `bottle --reachable_by--> right_ee` |
| `collides_with` | symmetric, `[T,N,N]` | Simulator contact passed benchmark collision filtering. | `gripper --collides_with-- milk_box` |
| `visible_to` | directed, `[T,N,C]` | Source contributes at least one actor-segmentation pixel to destination camera. | `bottle --visible_to--> countertop_camera` |
| `part_of` | directed, `[T,N,N]` | Source is a structural part of destination. | `drawer_handle --part_of--> drawer` |

### Important semantic distinctions

#### `near` versus contact

`near(A,B)` means geometrically close. It does not assert touching, collision,
reachability, or unobstructed access. `raw_contact` reports simulator contact;
`collides_with` reports contact that survives benchmark collision filtering.

#### `reachable_by` versus contact or graspability

`reachable_by(object,effector)` is prospective IK feasibility at the object's
origin. It does not mean the effector currently contacts the object and does
not guarantee grasp orientation, approach-path feasibility, gripper clearance,
or a complete task trajectory.

`reachable_by_valid[T,N,E]` distinguishes an evaluated negative result from an
unavailable or intentionally skipped query. `reachable_by_evaluated[T]` marks
frames on which a fresh IK batch was evaluated. A cached result is reused only
when the configured rounded scene signature is unchanged.

#### `visible_to` and occlusion

`visible_to(object,camera)` is true when at least one object pixel is present in
the saved actor segmentation. `visible_pixel_count[T,N,C]` retains the evidence,
and `visible_to_valid[T,N,C]` distinguishes evaluated zero visibility from
missing segmentation. An object can be partially occluded and still be
`visible_to` the camera. The current exporter does not identify which object is
the occluder; therefore it does not yet export `occludes` edges.

#### Containment

Containment is a privileged geometric label. The source object's 3-D center
must lie within the AABB envelope of an entity classified as a container.
`containment_valid[T,N,N]` and `contains_valid[T,N,N]` distinguish evaluated
negative pairs from pairs for which containment is not applicable.

## Action vocabulary

| Action type | Description | Example |
|---|---|---|
| `approach` | Move an end effector toward a target or pre-grasp pose. | `approach(right_ee, bottle)` |
| `grasp` | Close or engage the end effector to acquire the target. | `grasp(right_ee, bottle)` |
| `lift` | Raise a held object away from its initial support. | `lift(right_ee, bottle)` |
| `transport` | Move a held object toward another workspace pose. | `transport(bottle, basket)` |
| `place` | Move the target to its intended destination pose. | `place(bottle, basket)` |
| `release` | Disengage the end effector from the target. | `release(right_ee, bottle)` |
| `retreat` | Move the end effector away after manipulation. | `retreat(right_ee)` |
| `verify_success` | Evaluate the task success condition. | `verify_success(bottle_in_basket)` |
| `approach_handle` | Move toward an articulated fixture interaction point. | `approach_handle(left_ee, microwave)` |
| `grasp_handle` | Engage a handle or designated interaction part without asserting that the whole fixture is held. | `grasp_handle(left_ee, microwave)` |
| `open_articulation` | Actuate a joint toward its open state. | `open_articulation(left_ee, drawer)` |
| `close_articulation` | Actuate a joint toward its closed state. | `close_articulation(left_ee, microwave)` |

The six broader execution-phase labels are `setup`, `forward_grasp`,
`transition`, `backward_placement`, `final_descent`, and `success_check`.
Phases are temporal groupings, not action-node types.

## Action-entity edge types

Action participant edges are stored in `action_entity_edges` as parallel
`action_id`, `object_id`, and `roles` arrays.

| Role | Direction | Meaning | Example |
|---|---|---|---|
| `agent` | action to entity | End effector or robot entity executing the action. | `place_5 --agent--> right_ee` |
| `target` | action to entity | Object acted upon. | `place_5 --target--> bottle` |
| `destination` | action to entity | Intended receptacle, support, or destination entity. | `place_5 --destination--> basket` |

Together, a grounded action subgraph reads:

```text
place_5 --agent------> right_ee
place_5 --target-----> bottle
place_5 --destination> basket
bottle --in----------> basket
```

Equivalent compact JSON:

```json
{
  "action": {"id": 5, "type": "place"},
  "edges": [
    {"type": "agent", "destination": "right_ee"},
    {"type": "target", "destination": "bottle"},
    {"type": "destination", "destination": "basket"}
  ]
}
```

These participant roles differ from state relations. For example,
`target(action,bottle)` identifies what an action acts on, while
`held_by(bottle,right_ee)` describes world state.

## Observed action effects

`observed_effects_json` records valid relation-value changes between an
action's boundary frames. Current effect extraction covers:

- `reachable_by` for `approach` and `approach_handle`;
- `held_by` for `grasp` and `release`;
- `in` and `on` for `place` and `release`; and
- articulation joint-position deltas for `open_articulation` and
  `close_articulation`.

Example:

```json
{
  "relation": "in",
  "source": 7,
  "destination": 12,
  "before": false,
  "after": true
}
```

Preconditions and postconditions describe intended semantics. Consumers should
use relation tensors, validity masks, and `observed_effects_json` to determine
what actually changed.

## Auxiliary signals that are not canonical graph edges

| Dataset | Meaning |
|---|---|
| `relation_state/raw_contact` | Raw simulator contact adjacency. |
| `relation_state/grasped_by_code` | Compact grasp-state code used for reconstruction or debugging. |
| `collision_metric_contact_events` | Contact events retained by benchmark collision-metric filtering. |

These signals have different semantics and should not be treated as equivalent
to one another or silently promoted to canonical relations.

## Reserved and planned relations

The canonical vocabulary reserves these names, but schema `1.5.0` does not
populate relation tensors for them:

| Relation | Intended meaning | Example | Current status |
|---|---|---|---|
| `blocks` | Source obstructs access or motion to destination. | `milk_box --blocks--> bottle` | Planned; no exported tensor. |
| `occludes` | Source visually occludes destination from a specified viewpoint. | `milk_box --occludes--> bottle` | Planned; no occluder-attribution tensor. |
| `contact_risk_with` | Candidate action or entity has predicted unsafe contact risk with another entity. | `transport_4 --contact_risk_with--> glass` | Planned; no exported tensor. |

Do not train these as negative labels merely because their datasets are absent.
Absence means unimplemented or unknown, not false.

## Collection caveats

1. Executed planner attempts are retained, including failed attempts and attempts
   that produced no saved frame. Candidate alternatives that were never issued,
   rejected-plan lineage, and full failed-episode packaging remain future work.
2. Action intervals use saved benchmark frames, not every physics or controller
   substep; adjacent primitives may share boundary observations.
3. Tasks should supply explicit placement destinations in crowded scenes.
   Geometric fallback is permitted only when it has one uniquely nearest
   container/support candidate; ties raise an exception.
4. `move_by_displacement` uses an explicit action type when supplied. Otherwise
   it infers `lift`/`transport` only for a known held object and raises when the
   semantics are ambiguous.
5. Reachability collection may filter to movable targets and decimate frames.
   Cache reuse requires matching scene poses, articulation qpos, robot qpos,
   held-object state, and query objects. Validity datasets preserve the
   distinction between negative and unevaluated values.
6. Raw contacts require contact-point evidence. Contacts on articulation links
   map to the cataloged articulation root until links become first-class nodes.
7. Current expert tasks may use the left and right arms sequentially in one
   episode, and schema `1.5.0` can represent overlapping action intervals.
   However, the repository currently has no expert scenario or task config that
   issues simultaneous high-level actions to both arms. Simultaneous bimanual
   collection, synchronization semantics, shared-object participation, and
   cross-arm causal effects are a future extension; absence of overlapping arms
   in current data must not be interpreted as a negative capability label.

## Compatibility and integrity policy

Consumers must gate on both `schema_version` and the declared implemented
relation/action datasets; a matching version string alone is not permission to
assume an optional feature exists. Object tensor axes must be resolved through `relation_state/object_ids`, not
catalog row positions. Ambiguous acceptance-name lookups, duplicate IDs,
ambiguous grounding, malformed solver results, and unexpected episode counts
are data errors and must raise or fail validation. They must never be rewritten
as false labels. Expected-failure tasks require explicit failure-message
patterns; an unexpected pass is also a suite failure.

## Cross-task action validation suite

The Phase-2 matrix is defined in
`benchmark/bench_task_config/action_validation_suite.yml`. It covers a fixed
right-arm task, an articulated destination, a deterministically qualified
sequential multi-arm episode, and a recovery-continuation task with explicit
handle interaction, articulated targets, and observed joint-position effects. Run it with:

```bash
make action-validation-suite GPU_ID=0
make check-action-validation-suite
```

The suite writes `action_validation_report.json` below its output root. The
multi-arm task pins office arrangement `1`, which guarantees that both arms are
used sequentially. `simultaneous_dual_arm_expected: false` is an explicit
acceptance condition, not a claim that simultaneous dual-arm behavior is
unsupported by the robot or low-level controller.

## Source of truth

The schema constants and exporter implementation live in
`customized_robotwin/envs/_base_task.py`. The strict inspection utility is
`benchmark/bench_script/inspect_benchmark_hdf5.py`. If code and this document
diverge, treat that as a schema/documentation bug and update both in the same
change.
