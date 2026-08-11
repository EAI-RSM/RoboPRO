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
- [Role boundary](feedback_role_boundary.md) — CRITICAL: user is the scientist, agent is the engineer — but engineering must SERVE the science; classify validity/power/scope/quality; ask when unsure
- [Scientific rigor](feedback_scientific_rigor.md) — faithful controls, scene-specific mechanisms OFF (test: can you construct a scene that breaks it?), **verify the treatment actually fired**, don't harden a claim ahead of its measurement
- [Minimal changes](feedback_minimal_changes.md) — smallest diff; no frameworks; over-engineering is the WORST failure mode; "turn X off" never licenses building a replacement
- [Script conventions](feedback_script_conventions.md) — timestamped folder + timings.json + legible results
- [Math background](user_math_background.md) — strong stats/algebra, weak 3D geometry

**Repo invariants**
- [Env and git](repo_env_and_git.md) — required exports, remote URL, PR base branch
- [SAPIEN gotchas](repo_sapien_gotchas.md) — 3.x API breaks, viewer unusable in VSCode
- [Task assets](repo_task_assets.md) — task_objects.yml traps, OOD IDs, expert-first eval gate

**How the system works**
- [Expert baseline](domain_expert_baseline.md) — **READ BEFORE ANY CLAIM ABOUT "THE BASELINE"**: `direct` is a heavily-engineered expert minus two things; Hamid's July 0%→48% trajectory; what is stripped; the `off` cell that has never been run
- [bench_script layout](domain_bench_script_layout.md) — post-refactor lib/ + task/ packages, dependency rule
- [curobo](domain_curobo.md) — why subgoals, batched-IK recipe, knobs, what each failure means
- [Scene](domain_scene.md) — target/occluder geometry, ring layout, collision-registration traps
- [Seed conventions](domain_seed_conventions.md) — **READ BEFORE ANY FINETUNE/EVAL SEED CHOICE**: three disjoint bands (train 0.., eval 40000+, upstream 100000*(1+s)), the expert-feasibility gate, and the fact that `eval_seeds/` is frozen but wired into NOTHING
- [Visibility](domain_visibility.md) — 5 buckets, denominator timing, the one safe render speedup

**Tools**
- [Clearance metric](tool_clearance_metric.md) — eps*, the 2.5D pipeline, known inaccuracies
- [Geometric metric](tool_geometric_metric.md) — **LIVE, not retired** (a 2026-08-10 sweep was over-broad and was reversed); CPU-only envelope relaxation, target mask, Stage 3 measured at rho=1.0 but gate-unpassed on n=6, **study furniture IS in the obstacle set**, eps* is endpoint clearance
- [Route visualizer](tool_route_visualizer.md) — the "shifted figures" false alarm (refuted twice, don't redo), the 0.12 m tool-offset gap, config-hash forces a fresh `--out-dir` on every edit
- [Task-metric validity](tool_task_metric_validity.md) — **MEASURED 2026-08-07**: eps is endpoint-pinned 800/800, the 12 cm wrist offset makes it uncorrelated (rho=0.078) with real grasp tightness, 72/100 buckets flip on a legal grasp choice; plus the CPU offline-rebuild trick
- [Seed from clearance](tool_seed_from_clearance.md) — APPROACH_MODE/PLACEMENT_MODE, seed format, A/B design
- [Reach envelope](tool_reach_envelope.md) — producer/consumer/validator, calibration numbers
- [VLA / pi05 port](tool_vla_pi05_port.md) — runs end to end no-expert; ONE GPU is enough, MEM_FRACTION is a ceiling, planner-free mode moves no arms silently

**State**
- [Current status](status_current.md) — branch, uncommitted, unverified, next steps, and the live rehaul decisions. VOLATILE ONLY — durable findings must not live here (the pre_grasp/carry_transit headroom finding was once advertised here and is correctly held by [feedback_scientific_rigor](feedback_scientific_rigor.md) and [domain_expert_baseline](domain_expert_baseline.md))
- [Archive: planner comparison](archive_planner_comparison.md) — dead multi-planner line + its lessons
- [Artifacts](reference_artifacts.md) — published Artifact URLs and their caveats
