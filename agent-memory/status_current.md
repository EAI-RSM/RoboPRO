---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Everything else in `agent-memory/`
holds durable reasoning and gotchas. Last rewritten 2026-08-10.

## Branch and worktree

Branch: `codex/bench-script-refactor`.

- Research checkpoint: `417fbb2` (`Checkpoint validated research work before cleanup`).
- Corrected execution plan: `5af3033` (`Add verified codebase cleanup plan`).
- Phase 3 is `05532eb` (`Clean generated repository artifacts`).
- The cleanup plan and this status note are now dirty only to record the user's campaign-gate
  waiver before Phase 1 starts. Generated
  `graphify-out/` trees and the two visibility-validation PNGs remain on disk but are ignored.

## Cleanup execution

Source of truth:
`customized_robotwin/script/bench_script/plans/CLEANUP_PLAN.md`.

Completed:

- The pre-existing research snapshot was CPU-validated and committed separately.
- The corrected cleanup plan was committed separately.
- Phase 3 repo hygiene is complete: both generated `graphify-out/` trees are ignored; the repo-root
  `d` dump, empty `tools/` directory, and seven explicitly named stale `.pyc` files were removed;
  two generated visibility PNGs remain on disk but are no longer tracked; stale ignore entries were
  removed while both collision-test ignores were preserved.

Current authorization and pending work:

- Metric post-processing stopped at **620/3000**, with `processing_complete: false`, under
  `scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`.
- On 2026-08-10 the user explicitly waived that completion gate. Preserve the 620 existing metric
  records on disk, but preserving in-place resumption is no longer required; a future run may
  restart post-processing from episode 1.
- Run Phase 1 (`benchmark/bench_envs`) first, then Phases 2, 2.5, 2.6, and 4 in plan order. Do not
  interleave `bench_envs` and `bench_script` cleanup.
- Section 5 remains findings only. Delete none of those candidates without a new explicit user
  decision for the specific file.

## Verification state

Passed immediately before the research checkpoint:

- `git diff --check` and `compileall` for `benchmark/bench_envs` and `bench_script`.
- Live manifest/current-tree scene-code equality.
- Twelve CPU regression scripts: lib/task API, ring, obstacle, geometric metric, metric buckets,
  task metric, geometric-vs-gated, route visualization, reach-envelope, metric correlation,
  VLA reporting, and office smoke.
- Sixteen CPU-safe CLI `--help`/import checks and the real frozen-bucket validator.
- Corrected-plan Markdown fence, trailing-whitespace, and embedded-shell syntax checks.

GPU gap:

- `diag_kitchen_curobo.py --help` imports cuRobo before argparse and requires CUDA. The attempted
  check failed because no GPU was available, not because of a cleanup edit. The plan now places
  this command with the GPU-required Phase 1 gates.
- No Phase 1 real-scene before/after gate has run because this environment has no CUDA GPU. Use
  the static dispatch regression and CPU suite here, and leave the GPU gate explicitly unverified.

## Active research state

- The 3000-rollout source campaign itself is complete: 1000 episodes each at d6/d10/d15, with 554
  hard successes and all source videos present. Only integrated metric post-processing is partial.
- The independent route audit is complete at 50/50 selected episodes and 200 figures in
  `metric_route_visuals_v4`; its source hashes match the committed research snapshot.
- The task metric has a measured scientific-validity problem (endpoint pinning, wrist/contact
  offset, and grasp-candidate sensitivity). No remedy was authorized. Read
  `agent-memory/tool_task_metric_validity.md` before changing metric code, bucket definitions, or
  interpreting the association result.
- The bench-script refactor's historical GPU A/B verification remains separate from this cleanup;
  see `agent-memory/domain_bench_script_layout.md` and `agent-memory/domain_expert_baseline.md`.

## Next action

Preserve the existing partial metric artifacts and execute Phase 1 now. Continue through Phases 2,
2.5, 2.6, and 4 only after the preceding phase passes its available verification gates.
