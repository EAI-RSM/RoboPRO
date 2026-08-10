# Graph-conditioned pi0.5 proof of concept

This isolated experiment compares two policy-input conditions while preserving the existing RoboTwin policy and evaluation pipeline:

- `visual_only`: the original instruction, images, and robot state.
- `visual_retrieved_graph`: the same inputs plus a deterministic, compact current-frame scene subgraph appended to the instruction.

## Frozen protocol

Both conditions must use the same episode/task split, seeds, pi0.5 checkpoint, images, state, instruction, action chunking, and evaluation settings. The only treatment difference is the graph text. Report the repository's existing success, hard-success, collision, and collision count/category metrics.

Retrieval is one-hop from target objects and robot effectors. It includes only true relations whose evidence-validity mask is true, uses `countertop_camera` for camera-conditioned facts, and excludes redundant inverse edges by default.

The current treatment uses event-driven replanning within the nominal 50-action horizon. During grasping, it preserves the original task instruction exactly. Its model-server call path is also identical to visual-only: the instruction is set once, unchanged grasp chunks skip tokenizer fitting and prompt rewrites, and only inference advances the deterministic policy RNG. After every executed action, valid `held_by` evidence is checked. A new grasp immediately discards the unused actions, switches to a compact placement cue such as `Move the held object forward and left into the basket right.`, and latches only the active arm's gripper closed during transport. Release is enabled by either valid containment or a conservative gravity-assisted readiness gate: held evidence must remain true, the target center must lie inside the destination's horizontal AABB footprint, and the target bottom must be no more than 10 cm above the destination top. This interrupts transport and replans with `Release the held object in the basket right.`; only this release phase permits an opening command. Full `in` containment remains the success signal. Two consecutive losses of held evidence outside the destination return to grasping. Metric values, aliases, node declarations, sizes, and general relation prose are omitted from live prompts, and a 5 cm dead band suppresses directional jitter. Tasks without exactly one target and one destination retain their base instruction.

The live model server still uses the checkpoint tokenizer and enforces the complete prompt limit. The dependency-aware graph packer remains available for offline analysis and controlled ablations, but general node and relation serialization is not part of this treatment.

Submit the paired 10-seed control/treatment array with `evaluation/submit_paired_10.sh`. When an unchanged visual-only baseline does not need to be rerun, submit only the same 10 graph-conditioned seeds with `evaluation/submit_graph_10.sh`. The graph-only wrapper creates ten array tasks indexed `0-9`, records `condition_mode=graph_only`, writes `pi05-graph_<job>_<index>.out` logs, and stores results only under `visual_retrieved_graph` in a timestamped `*-graph-only` batch directory.

For the diverse benchmark campaign, run `evaluation/submit_diverse_5x10.sh`. It covers d10 clutter in office (`put_mouse_on_pad`, `put_book_on_book`), study (`put_cup_in_box`), kitchens (`put_spoon_on_plate_ks`), and kitchen-large (`put_sauce_can_in_basket`). If any task has fewer than ten expert-validated seeds, a five-task precollection array expands its task-specific bank first; an `afterok` coordinator then submits five paired arrays, for 100 policy episodes total. Pairing is exact within each task. Seed banks are task-specific because a simulator seed that admits a valid expert plan for one task need not admit one for another.

Expert action nodes, action outcomes, and future annotations are prohibited policy inputs. Current pi0.5 rollouts do not create equivalent online high-level action nodes, so action history is disabled in both conditions. Exported action nodes may be used only for offline analysis or labels after rollout.

The present HDF5 episodes validate alignment and input construction; four successful episodes are not sufficient for a statistically meaningful policy comparison. Simulation-derived ground truth can also overstate real-world graph quality. Later evaluation must preserve validity/confidence metadata and test sensitivity to perception, depth, calibration, occlusion, and temporal noise.

## Phase 0-2 validation

From the repository root:

```bash
.venv/bin/python -m experiments.graph_conditioned_pi05.tests.test_graph_context
.venv/bin/python -m experiments.graph_conditioned_pi05.validate_alignment \
  customized_robotwin/data/action_validation_schema19_blocks_v1 \
  --frame-stride 10 \
  --graph-token-budget 120
```

The validator checks schema and frame alignment, repeats retrieval to detect nondeterminism, rejects action-node leakage, and reports retrieved/selected/dropped fact and token-count summaries.

## Phase 3 live pi0.5 adapter

The dual-environment evaluator now retrieves from each live observation immediately before an action-chunk inference. `visual_only` keeps the original instruction path. `visual_retrieved_graph` assigns deterministic episode-level aliases from the complete catalog (`T1...` targets, `D1...` destinations, `O1...` other objects, and `L`/`R` end effectors), declares each selected node once, then refers to those aliases in natural-language relations. For example: `Nodes: T1 = sauce can at (0.4, -0.2, 0.8), size (0.1, 0.1, 0.2). Relations: T1 is reachable by L.` The model server packs this text with the checkpoint's PaliGemma SentencePiece tokenizer. Relation dependencies are atomic: a fact can only be selected when every alias it references is declared, and shared declarations are charged once. The policy implementation, images, state, action chunking, and existing evaluation metrics are unchanged.

Node labels prefer an explicit catalog `semantic_label`. When the semantic label merely repeats the simulator name, known infrastructure prefixes (`task_`, `target_`, `object_`, `model_`) and trailing numeric instance IDs are removed, separators become spaces, and meaningful descriptors such as direction and color remain. Object identity never depends on this display label: stable aliases distinguish duplicate labels.

Run paired conditions with the same task, seed, checkpoint, and `TEST_NUM`:

```bash
make eval-pi05-double \
  TASK_NAME=put_sauce_can_in_basket \
  TASK_CONFIG=<task-config> \
  TRAIN_CONFIG_NAME=<train-config> \
  MODEL_NAME=<model-name> \
  CHECKPOINT_ID=<step> \
  SEED=0 TEST_NUM=10 \
  GRAPH_INPUT_CONDITION=visual_only

make eval-pi05-double \
  TASK_NAME=put_sauce_can_in_basket \
  TASK_CONFIG=<task-config> \
  TRAIN_CONFIG_NAME=<train-config> \
  MODEL_NAME=<model-name> \
  CHECKPOINT_ID=<step> \
  SEED=0 TEST_NUM=10 \
  GRAPH_INPUT_CONDITION=visual_retrieved_graph \
  GRAPH_TOKEN_BUDGET=120 \
  GRAPH_DEFAULT_CAMERA=countertop_camera
```

Each `_episodes.jsonl` record retains success/collision metrics and adds the condition plus graph inference counts, selected/dropped node and fact averages, tokenizer-measured graph tokens, and destination-seed availability.

Destination IDs are read only from `TASK_ENV.get_role_names()`; expert action nodes are never consulted. Singular and multi-destination tasks export concrete scene IDs. Tasks whose destination is not stored in a `des_obj*` attribute opt in with `benchmark_destination_attrs`; generic attributes such as `basket_right` are never inferred globally because the same object may be a source in another task. Older task metadata that only provides destination names is resolved against the object catalog only when the match is unambiguous; otherwise evaluation fails instead of guessing. Tasks with no destination role fall back to target-and-effector seeds and record `destination_seed_available: false`.

Graph-token counts use the exact checkpoint tokenizer. The logged full-prompt count is explicitly an estimate because pi0.5 discretizes the normalized state during its internal transform, while the server-side preflight receives raw state. The model's normal truncation guard remains authoritative. A future refinement can expose normalized-state token accounting without duplicating the full image preprocessing transform.

The current adapter intentionally supports the recommended dual-environment evaluator. A full rollout requires the local pi0.5 environment and checkpoint, which are not present in this checkout, so this checkpoint is validated through syntax, launch dry-runs, synthetic live snapshots, and the schema-1.9 episodes.
