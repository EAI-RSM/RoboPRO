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
12. `visible_to`: the source entity is visible to the destination robot or
    end-effector viewpoint abstraction.
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
