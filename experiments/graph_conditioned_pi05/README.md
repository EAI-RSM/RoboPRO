# Graph-conditioned pi0.5 proof of concept

This experiment compares two rollout systems while preserving the fixed RoboTwin task and evaluation protocol:

- `visual_only`: the original instruction, images, and robot state.
- `visual_retrieved_graph`: a graph-aware hierarchical rollout that uses deterministic current-frame task state to issue compact phase instructions, protect a held object, and interrupt stale action chunks when task-relevant relations change.

## Frozen protocol

Both conditions must use the same task configuration, episode seeds, pi0.5 checkpoint, camera images, robot state, action representation, nominal action-chunk horizon, and evaluation criteria. The graph-aware treatment is intentionally a system-level intervention: it may derive compact instructions from current graph state, interrupt the unused remainder of a nominal chunk after a task-relevant graph event, replan at that event boundary, and clamp the active gripper closed while transporting an object known to be held. These mechanisms are part of the treatment, not controlled variables. Results therefore estimate the utility of the complete graph-aware planning system relative to the standard visual-only rollout; they do not isolate the causal effect of graph prose alone. Report success, hard-success, collision, and collision count/category metrics, together with prompt phases and chunk-interruption counts.

Retrieval is one-hop from target objects and robot effectors. It includes only true relations whose evidence-validity mask is true, uses `countertop_camera` for camera-conditioned facts, and excludes redundant inverse edges by default.

The current treatment uses event-driven replanning within the nominal 50-action horizon. During grasping, it preserves the original task instruction exactly. Its model-server call path is also identical to visual-only: unchanged grasp chunks skip tokenizer fitting and prompt rewrites, and only inference advances the deterministic policy RNG. After every executed action, the graph-aware controller still observes valid task predicates. A new grasp immediately discards the unused actions, switches to compact placement guidance, and latches only the active arm's gripper closed during transport. Release readiness is distinct from strict goal completion: a conservative gravity-assisted gate may start release, while the actual `in` or `on` relation remains the goal signal. Tasks without exactly one target and one destination retain their base instruction.

The live model server uses the checkpoint tokenizer and enforces the complete prompt limit for post-grasp phase guidance. Retrieved node declarations and general relation prose are deliberately excluded from this baseline treatment.

Submit the paired 10-seed control/treatment array with `evaluation/submit_paired_10.sh`. When an unchanged visual-only baseline does not need to be rerun, submit only the same 10 graph-conditioned seeds with `evaluation/submit_graph_10.sh`. The graph-only wrapper creates ten array tasks indexed `0-9`, records `condition_mode=graph_only`, writes `pi05-graph_<job>_<index>.out` logs, and stores results only under `visual_retrieved_graph` in a timestamped `*-graph-only` batch directory.

For the diverse benchmark campaign, run `evaluation/submit_diverse_5x10.sh`. It covers d10 clutter in office (`put_mouse_on_pad`, `put_book_on_book`), study (`put_cup_in_box`), kitchens (`put_spoon_on_plate_ks`), and kitchen-large (`put_sauce_can_in_basket`). If any task has fewer than ten expert-validated seeds, a five-task precollection array expands its task-specific bank first; an `afterok` coordinator then submits five paired arrays, for 100 policy episodes total. Pairing is exact within each task. Seed banks are task-specific because a simulator seed that admits a valid expert plan for one task need not admit one for another.

After the paired visual-only baseline has been established, use `evaluation/submit_diverse_graph_5x10.sh` for graph-only iterations. It uses the same five tasks and validated seeds but submits five 10-task graph arrays (50 episodes total), stores `condition_mode=graph_only` in the campaign job table, and does not rerun visual-only.

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

The dual-environment evaluator reads graph state from every live observation. `visual_only` executes the standard pi0.5 action chunks with the original task instruction. `visual_retrieved_graph` uses valid simulator evidence as an oracle task-state interface for a `grasp → placement → release` controller. Grasp uses structured `align`, `move_down`, `move_up`, `move_closer`, and `close` intents. The exported end-effector poses are the simulator's gripper-center TCP poses. Within the 12 cm alignment vicinity, signed TCP height error selects move-down or move-up until the TCP enters a ±2 cm band around the target AABB's upper surface; a height-aligned TCP outside close range selects move-closer. Non-close transitions require two-frame persistence. Close becomes eligible for height-aligned geometry within the 10 cm ceiling. Preferred geometry at most 8 cm or validated contact enters immediately; other 8--10 cm entries require two-frame persistence. Close is a recoverable attempt: two invalid frames return to the geometry-selected correction unless three consecutive frames of simultaneous `held_by` and held-arm contact advance the controller to placement. An effective substage or phase transition may discard unused actions in the current nominal chunk and request a new chunk with a compact instruction. A plan terminates as stalled only when both TCPs remain within a 2 mm motion radius for 50 consecutive actions; ordinary 50-action chunk completion does not terminate a moving episode. Effective replans reset the motion window, and recovery is deferred. During placement, the active gripper channel is latched closed; only release permits the model to open it. Placement and release behavior is unchanged by the pre-grasp subdivision. Prompt selection, executed chunk length, replanning frequency, early termination, and gripper protection may differ between conditions by design. Images, robot state, checkpoint, action representation, nominal chunk horizon, seeds, simulator configuration, and evaluation metrics remain controlled.

The `graph_delta_top_tcp_motion_stall_contact_persistence_v3` treatment selects an arm only when exactly one validated straight target corridor is clear and every alternative corridor is blocked. Outside the 12 cm vicinity it says `Align the <arm> gripper with <target>.` Inside that vicinity, a TCP more than 2 cm above the target AABB's upper surface says `Move the <arm> gripper down to align it with <target>.`, a TCP more than 2 cm below that surface says `Move the <arm> gripper up to align it with <target>.`, and a height-aligned TCP outside close range says `Move the <arm> gripper closer to <target> for grasping.` Missing target AABBs use the target pose height with the same ±2 cm tolerance. A close attempt requires valid height within the 10 cm ceiling. Preferred geometry at most 8 cm or validated contact enters immediately; other 8--10 cm entries require two-frame persistence. Close says `Close the <arm> gripper to grasp <target>.` If readiness is lost for two frames before acquisition, the attempt returns to the current geometry-selected correction. Candidate arms are evaluated independently so invalid-height contact cannot mask a geometrically ready arm. A collision warning is added only when the nearest validated blocker's AABB is within 20 cm of the blocked end-effector. Action chunks are truncated to the remaining episode-step budget. A 50-action rolling window sets `termination_reason=graph_motion_stall:<phase>:<substage>` only when neither TCP leaves a 2 mm radius; effective replans reset the window. Placement and release behavior is otherwise unchanged. Recovery, approach-arm latching, and gripper-frame lateral alignment are deliberately deferred to later isolated treatments.

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
