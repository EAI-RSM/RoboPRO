# Graph-conditioned pi0.5 proof of concept

This isolated experiment compares two policy-input conditions while preserving the existing RoboTwin policy and evaluation pipeline:

- `visual_only`: the original instruction, images, and robot state.
- `visual_retrieved_graph`: the same inputs plus a deterministic, compact current-frame scene subgraph appended to the instruction.

## Frozen protocol

Both conditions must use the same episode/task split, seeds, pi0.5 checkpoint, images, state, instruction, action chunking, and evaluation settings. The only treatment difference is the graph text. Report the repository's existing success, hard-success, collision, and collision count/category metrics.

Retrieval is one-hop from target objects and robot effectors. It includes only true relations whose evidence-validity mask is true, uses `countertop_camera` for camera-conditioned facts, and excludes redundant inverse edges by default.

Every retrieved node (target/distractor objects, robot end effectors) is declared once with a world-frame 3D position and, for objects with collision geometry, an axis-aligned bounding-box size — both rounded to 1 decimal place. Without this, two same-name objects (e.g. two "bowl" distractors placed by the cluttered-table domain randomization) are only distinguishable by an opaque catalog ID the checkpoint never saw during fine-tuning, so the model has no way to ground which node in the text corresponds to which pixel region. Node and fact selection under the 120-token graph budget is a priority-weighted 0/1 knapsack, not a first-fit walk: ranks are spaced so a single higher-priority item always outvalues any combination of lower-priority ones, which is what makes "maximize information under a fixed priority order" well-defined rather than accidentally letting several cheap facts crowd out one expensive, important fact (or, within one priority tier, an iteration-order artifact silently packing fewer same-tier facts than the budget actually allows). The same packer (`graph_serializer.pack_items`) runs both offline (this validator, with an approximate tokenizer) and live (the model server, with the real checkpoint tokenizer), so there is one selection algorithm, not two independently maintained ones.

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

The dual-environment evaluator now retrieves from each live observation immediately before an action-chunk inference. `visual_only` keeps the original instruction path. `visual_retrieved_graph` renders ordered nodes and facts as natural-language sentences and sends them to the model server, which packs whole sentences with the checkpoint's PaliGemma SentencePiece tokenizer. For example, `reachable_by(obj_1,obj_2)` becomes `obj_1 is reachable by obj_2.`, and a node is declared as `bowl#23 is at (0.3, -0.1, 0.8) with bounding-box size (0.1, 0.1, 0.2).`. The resulting scene description (a `Nodes:` section followed by a `Scene graph:` section) is appended to the task instruction. The policy implementation, images, state, action chunking, and existing evaluation metrics are unchanged.

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

Destination IDs are read only from `TASK_ENV.get_role_names()`; expert action nodes are never consulted. Tasks that do not expose `destination_id` fall back to target-and-effector seeds and record `destination_seed_available: false`.

Graph-token counts use the exact checkpoint tokenizer. The logged full-prompt count is explicitly an estimate because pi0.5 discretizes the normalized state during its internal transform, while the server-side preflight receives raw state. The model's normal truncation guard remains authoritative. A future refinement can expose normalized-state token accounting without duplicating the full image preprocessing transform.

The current adapter intentionally supports the recommended dual-environment evaluator. A full rollout requires the local pi0.5 environment and checkpoint, which are not present in this checkout, so this checkpoint is validated through syntax, launch dry-runs, synthetic live snapshots, and the schema-1.9 episodes.
