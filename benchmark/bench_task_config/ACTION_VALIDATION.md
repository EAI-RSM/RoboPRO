# Schema 1.6 action validation suite

The matrix in `action_validation_suite.yml` selects four expert tasks:

- `put_sauce_can_in_cabinet`: fixed right-arm grounding;
- `put_rubikscube_in_drawer`: articulated destination grounding;
- `set_up_table`: both arms in one episode, executed sequentially; and
- `chain_heat_hamburger_ks`: continuation/recovery behavior.

Collect and validate all tasks:

```bash
make action-validation-suite GPU_ID=0
```

Useful controls:

```bash
make action-validation-suite \
  ACTION_VALIDATION_MODE=collect \
  ACTION_VALIDATION_TASKS=fixed_right_arm,articulated_destination \
  ACTION_VALIDATION_OUTPUT_ROOT=customized_robotwin/data/action_validation_suite_v1 \
  ACTION_VALIDATION_START_SEED=0 \
  REACHABLE_BY_INTERVAL=10 \
  ACTION_VALIDATION_OBSTACLE_DENSITY=10 \
  GPU_ID=0

make check-action-validation-suite
```

Set `ACTION_VALIDATION_DRY_RUN=1` to print commands and resolved overrides
without launching the simulator.

Office arrangement `1` is forced for `set_up_table`, guaranteeing that the
episode uses both arms sequentially. No current expert task/config issues
simultaneous high-level actions to both arms. A future simultaneous-bimanual
scenario must define synchronization, overlapping action intervals, participant
edges for both effectors, shared-target semantics, and per-arm failure status.

The recovery/continuation task now requires `approach_handle`, `grasp_handle`,
and `close_articulation` nodes grounded to an articulated target. It also
requires an `interaction_part` parameter and a measured closing-direction joint
delta in `observed_effects_json`. Planner-attempt lineage and retention of
terminal failed episodes remain future extensions; missing output files remain
hard failures. Relative `ACTION_VALIDATION_OUTPUT_ROOT` values are normalized
against the repository root before collection.
