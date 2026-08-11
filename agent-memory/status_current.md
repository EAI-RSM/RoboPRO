---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold volatile state.** Last rewritten 2026-08-11.

## Branch and worktree

Branch: `peng-training-branch` (the previous entry naming `codex/bench-script-refactor` was stale;
that branch is 1 commit behind this one and the same line of work). Tip `e3a09ce`.

- **S1 IS DONE (2026-08-11). All local-only work is backed up off-machine.** Before it,
  `git rev-list --all --not --remotes` was **71**; it is now **0**, verified against a fresh bare
  clone rather than local tracking state. Five archive refs on `origin`:
  `backup/peng-training-branch` (`0d7df5c`), `backup/occluder-testbench` (`91b2d26`),
  `backup/pre-merge-35-occluder` (`35801cb`),
  `backup/reachability-motion-validated-placement` (`585d307`),
  `backup/visibility-constraint-experiments` (`41ee00f`), plus tag `pre-rehaul-2026-08-11` on
  `e3a09ce`. The `backup/` prefix marks these as archives, not review requests — no PR was opened.
- Scope note: pushing `peng-training-branch` alone would have saved only 64 of the 71 commits. Four
  other branches held unique work, including `585d307`, the vanilla `--plan-algo` baseline that
  [[domain_expert_baseline]] calls "recoverable from that commit."
- Working tree clean. Tip is `0d7df5c` (rehaul plan + memory corrections); `e3a09ce` is its parent.
- Against `origin/dev` (`64840ce`): **82 ahead, 60 behind**, merge base `aabeff4` (2026-06-25).
  See [[repo_env_and_git]] for the full divergence numbers and the stale-local-`dev` trap.

## The rehaul plan (2026-08-11)

`plans/REHAUL_PLAN.md` scopes a fresh `peng-dev-new` off **`origin/dev`**, with the research layer
ported and consolidated rather than rewritten. Parts 0–3 (safety net, branch, ~150 lines of shared
carry-over, the scripted 84-file cleanup) are ~2 days; Part 4 (research layer) is the bulk.

Decisions taken this session, so they are not relitigated:

- Goals: cheap future merges, clean architecture, dead-weight removal. Budget 1–2 weeks. Data
  continuity **partial** — results stay readable, not re-runnable. Fork policy: track dev weekly.
- **Task-metric line stays LIVE** — the correlation question is the research programme. Reach/
  clearance tools port. `task/` is ported, instrumented, then pruned by measurement.
- **`swept_volume_3d.py` is NOT ported** (412 L; last run 2026-07-15, the one real staleness
  outlier). Its 285 MB of results stay on disk.
- **Nothing under `scripts/` is dropped.** An earlier draft proposed deleting `scripts/upload/`,
  `scripts/slurm/` and the five June OOD scripts (~1,142 L); all three **exist on `origin/dev`
  unmodified by us**, so deleting them would be a shared-file edit — the exact merge tax the rehaul
  exists to remove.
- `phase4_approach_mode` `off` cell: **deferred, optional** (Part 6b). See [[domain_expert_baseline]]
  for the pooling deadline that makes deferral non-free.
- Entry-point collapse (14 → ~4 subcommands) is **flagged but NOT in scope** — awaiting an explicit
  yes, because it renames every command and breaks Makefile targets and `run_approach_mode_ab.sh`.

## Preserved research state

- Source rollout campaign complete at 3000 episodes with all source videos.
- Metric post-processing partial at **620/3000**, under
  `results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`.
  Not resumable (stored code hash matches no commit), and moot — any fingertip-clearance remedy
  recomputes all of them.
- Route audit complete at 50 episodes / 200 figures. No result directory deleted or rewritten.
- `results/` totals ~13 GB, untracked and unregenerable. Rollouts are non-deterministic by design.

## Verification state

Unchanged from the 2026-08-10 prune gates: all active/archive Python parses; CPU-safe top-level
commands pass `--help`; CPU-safe checks pass as `python -m checks.<module>`; the lib/task→top-level
dependency count is zero. `checks.smoke_test_seed_2a` and `diag_kitchen_curobo.py` remain **unrun
(CUDA-only)** — do not claim those gates passed.

**Nothing has executed in this repo since 2026-07-31.** All of August has been prune/refactor work.

## Next action

Not started — the rehaul is scoped, not begun.

`plans/REHAUL_PLAN.md` now opens with a **Work breakdown** of twelve executable sections, S1–S12,
each self-contained enough to prompt an agent with "write a technical plan for S6" or "execute S8"
without reading the rest of the document. **Numbering is stable; do not renumber.** Each section
carries in-scope / out-of-scope / a done-when gate / verification commands, and the ones that must
not change behaviour are marked NO-BEHAVIOUR-CHANGE.

Order: ~~S1~~ **done**, then **S2** (pytest harness — run it on `peng-training-branch` *before* the
move, so the port has a passing baseline to diff against), then S3/S4 (branch, port, make it run).
S5 is parallelisable. S10 needs a user GPU run mid-section — schedule it early.

**Next up: S2.** Note one wrinkle recorded in its section — `pyproject.toml` is SHARED and
byte-identical to `origin/dev`, so adding pytest to it is best sent as a one-line PR to dev rather
than carried as a branch diff.

One open question: whether to add §4e, the entry-point collapse (14 → ~4 subcommands). It is
deliberately unsectioned because it renames every command and breaks Makefile targets,
`run_approach_mode_ab.sh:220`, and command strings recorded across `agent-memory/` and `plans/`.
The `checks/` fixture consolidation is resolved — it is folded into S2.
