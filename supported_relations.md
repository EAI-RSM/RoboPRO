# Supported Graph Relations

This repository currently defines the following relation types in the
RoboPRO-native graph schema. These are the canonical edge types supported by
the codebase for scene, belief, and safety reasoning.

1. `on`: the source object rests on a destination support surface or furniture
   element.
2. `in`: the source object is contained inside a destination container-like
   entity.
3. `supports`: the source support surface, furniture item, or room structure
   can physically support the destination object.
4. `contains`: the source container or furniture item can contain the
   destination object.
5. `held_by`: the source object is currently grasped by a destination end
   effector.
6. `near`: the source and destination entities are spatially close in
   task-relevant space.
7. `blocks`: the source entity obstructs access or motion to the destination
   entity.
8. `occludes`: the source entity visually occludes the destination from a
   relevant viewpoint.
9. `reachable_by`: the source entity is reachable by the destination end
   effector under the current scene assumptions.
10. `contact_risk_with`: the source and destination entities have a potential
    unsafe contact risk under a candidate motion or action.
11. `collides_with`: the source and destination entities are in a verified
    collision under execution, replay, or simulation.
12. `visible_to`: the source entity is visible to the destination camera.
13. `part_of`: the source entity is a structural part of the destination
    entity.

## Note

The list above reflects the full relation inventory defined by the repository's
current RoboPRO graph schema. Individual dataset exports may expose only a
subset of these relations directly; the remaining relations may be inferred,
retrieved, or populated by downstream tool calls.

## Normalization Rule

Some RoboPRO benchmark-support exports also include a raw tensor named
`support`. In this repository, `support` is not treated as a distinct graph
relation. It is normalized into the canonical support semantics:

1. `supports(source, destination)` is the canonical directed support relation.
2. `on(destination, source)` is the inverse convenience relation used when an
   object is resting on a support surface.
3. Raw dataset field names such as `support` are treated only as legacy
   storage aliases and should not appear in the exported graph ontology.

## Benchmark Export Note

Schema version `1.4.0` directly exports the following additional relations:

1. `in[T,N,N]` and its exact inverse `contains[T,N,N]`. Containment is a
   privileged geometric label: the source object's 3-D center must lie inside
   the AABB envelope of a catalog entity classified as a container. The
   `containment_valid` and `contains_valid` masks distinguish non-container
   pairs from evaluated negative relations.
2. `visible_to[T,N,C]`, indexed by `visible_to_camera_names[C]`, is derived
   from the actor segmentation image already captured at that timestep.
   `visible_pixel_count` preserves the evidence and `visible_to_valid`
   distinguishes an evaluated zero-pixel result from unavailable segmentation.
3. `reachable_by[T,N,E]`, indexed by `reachable_by_effector_names[E]`, is a
   collision-aware, position-only batched IK query at the entity origin.
   `reachable_by_valid` distinguishes an IK-negative result from a solver or
   entity for which the query was unavailable. It is a pose reachability label,
   not a guarantee that a task-specific grasp or full trajectory is feasible.

### Reachability collection policy

Reachability remains collision-aware when collection cost is reduced. The
collector can filter queries to movable and target objects, evaluate only every
configured number of changed-scene frames, and reuse a previous result only
when the rounded poses of all catalogued scene entities are unchanged. A
changed, skipped frame is exported with `reachable_by_valid=false`; it is never
silently treated as unreachable. `reachable_by_evaluated[T]` records whether a
fresh IK batch was run on each frame (as opposed to a valid cache reuse).

Configure this under `benchmark_relations.reachable_by`:

1. `enabled` enables the relation (default `true`).
2. `frame_stride` controls fresh evaluation cadence for changed scenes
   (default `1`, so ordinary exports preserve per-frame evaluation).
3. `movable_only` restricts queries to movable or task-target entities
   (default `true`).
4. `cache_unchanged` allows exact scene-signature reuse (default `true`).
5. `pose_round_decimals` controls pose-signature tolerance (default `3`).

For the dense validation run, these parameters are exposed as Make variables:

```bash
make relation-validation \
  GPU_ID=0 \
  REACHABLE_BY_INTERVAL=10 \
  REACHABLE_BY_MOVABLE_ONLY=1 \
  REACHABLE_BY_CACHE_UNCHANGED=1 \
  REACHABLE_BY_POSE_DECIMALS=3 \
  RELATION_OBSTACLE_DENSITY=14 \
  RELATION_EPISODE_NUM=1
```

Set `REACHABLE_BY_INTERVAL=1` for the highest temporal fidelity. Larger values
reduce fresh IK calls; validity masks continue to distinguish skipped/unknown
entries from evaluated negative reachability.

These optimizations do not turn `reachable_by` into a contact relation.
`raw_contact` describes contact that is occurring now; `reachable_by` is a
prospective collision-aware IK feasibility query. Full grasp orientation,
approach-path feasibility, gripper geometry, and task-specific constraints
remain future refinements.

Some benchmark-support exports also include auxiliary contact-derived signals
that are useful for debugging and reconstruction but are not part of the
canonical graph ontology:

1. `relation_state/raw_contact` is the raw simulator contact adjacency between
   exported objects.
2. `collision_metric_contact_events` is a filtered event log that records only
   benchmark-significant contacts that passed the collision-metric logic.
3. These artifacts have different semantics and should not be compared as if
   they explained each other one-to-one.
4. Neither `raw_contact` nor `collision_metric_contact_events` is a canonical
   graph relation type.
