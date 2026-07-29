# Agent memory

Durable working notes for AI agents on this repo (Claude Code and Codex both read this folder).
Human-readable, but written for agents: only facts that are **not derivable from the code**, plus
the reasoning behind decisions that look arbitrary from the diff.

**Filing rules — read before adding anything.**
- Put a new fact in the existing file that covers it. Only start a new file for a genuinely new
  subject. This folder was once 47 files because every chat appended one; that is the failure mode.
- `status_current.md` is the ONLY file that holds volatile state (uncommitted / unverified / next
  steps). **Rewrite it, never append**, and keep status out of every other file.
- Prefer grep-able symbol names over `file:line` — the 2026-07-29 refactor invalidated nearly every
  line number recorded here.
- Record what was surprising, not what the code already says. Delete what turns out to be wrong.

**Working with the user**
- [Role boundary](feedback_role_boundary.md) — CRITICAL: user is the scientist, the agent is the engineer
- [Scientific rigor](feedback_scientific_rigor.md) — faithful controls, confounds, scene-specific mechanisms OFF
- [Minimal changes](feedback_minimal_changes.md) — smallest diff; no frameworks
- [Script conventions](feedback_script_conventions.md) — timestamped folder + timings.json + legible results
- [Math background](user_math_background.md) — strong stats/algebra, weak 3D geometry

**Repo invariants**
- [Env and git](repo_env_and_git.md) — required exports, remote URL, PR base branch
- [SAPIEN gotchas](repo_sapien_gotchas.md) — 3.x API breaks, viewer unusable in VSCode
- [Task assets](repo_task_assets.md) — task_objects.yml traps, OOD IDs, expert-first eval gate

**How the system works**
- [bench_script layout](domain_bench_script_layout.md) — post-refactor lib/ + task/ packages, dependency rule
- [curobo](domain_curobo.md) — why subgoals, batched-IK recipe, knobs, what each failure means
- [Scene](domain_scene.md) — target/occluder geometry, ring layout, collision-registration traps
- [Visibility](domain_visibility.md) — 5 buckets, denominator timing, the one safe render speedup

**Tools**
- [Clearance metric](tool_clearance_metric.md) — eps*, the 2.5D pipeline, known inaccuracies
- [Seed from clearance](tool_seed_from_clearance.md) — APPROACH_MODE/PLACEMENT_MODE, seed format, A/B design
- [Reach envelope](tool_reach_envelope.md) — producer/consumer/validator, calibration numbers
- [VLA / pi05 port](tool_vla_pi05_port.md) — wired but blocked on the expert gate

**State**
- [Current status](status_current.md) — branch, uncommitted, unverified, next steps
- [Archive: planner comparison](archive_planner_comparison.md) — dead multi-planner line + its lessons
- [Artifacts](reference_artifacts.md) — published Artifact URLs and their caveats
