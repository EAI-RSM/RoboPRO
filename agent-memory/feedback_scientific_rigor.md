---
name: feedback_scientific_rigor
description: "User holds experiments to a science bar: faithful controls, scene-specific mechanisms OFF, CHECK THE TREATMENT ACTUALLY FIRED, never harden a conclusion ahead of its measurement, isolate parts even when unpublished"
metadata:
  type: feedback
---

"We are doing science here, so we gotta be precise with our variables." The user pushes on whether
a control is really the thing it claims to be (e.g. "is baseline actually faithful to dev, beyond
the max_tries parameter?").

**Why:** a headline claim ("new planner beats old") is only defensible if the control is the real
old thing and the only varied quantity is the one under test.

**How to apply:**
- Distinguish PARAMETERS (max_attempts 10→24) from real BUG FIXES / algorithm changes, and answer
  at the right layer — the curobo primitive (`plan_path`) is not the task-level expert
  (`play_once`). Claude's "only max_attempts" answer was too narrow and the user caught it.
- **Any mechanism that reads scene-specific state must be OFF in a generality experiment.**
  (Standing rule from 2026-07-28: the backward carry leg was silently running a hand-authored
  chain that read occluder 0's position and a hand-entered footprint constant, in *all* modes.
  It never broke the pairing — it broke the generality claim.) Keep only scene-agnostic plumbing.
- **The user's own sharper wording of that test (2026-07-28), which is the one to use:** a
  mechanism may stay if *you cannot construct a scene that breaks it*. Their example — "the
  NUMBER of obstacles is scene agnostic, but the LOCATION of a specific obstacle is not, since it
  depends on which obstacle is chosen." This is better than "doesn't read the scene" because it
  **permits reading the scene as long as you quantify over all of it**. That is exactly why the
  clearance seed is legitimate (it walks every obstacle in `env.collision_list`) while
  `_box_side_x` is not (it indexes `occluders[0]`). Use *universal vs privileged*, not *reads vs
  doesn't read*.
- **Classify before proposing** — validity / power / scope / quality; only validity blocks a run.
  See [[feedback_role_boundary]]. A shared constant applied to BOTH arms is never a confound; it
  can still cost power or narrow the claim, but say which.
- **Watch for arm-only asymmetries that are not the treatment.** The principle stands: an
  uncontrolled arm-only difference with zero upside should be removed, not argued to be inert.
  **But the 2026-07-28 example originally recorded here was wrong** and is kept only as a warning:
  it claimed the carry seed "never fired" so the seeded arm paid ~190 s/episode for nothing. Measured
  2026-07-30, the carry leg builds ~100% of the time (7/9 lifetime). Acting on the note would have
  meant `CARRY_SEED=0`, suppressing the treatment on the ONLY leg with measurable headroom. Check
  whether the "inert" mechanism is actually inert before calling it an asymmetry.
- **ISOLATION IS NOT ENOUGH — measure that the treatment was actually DELIVERED.** The single most
  expensive lesson of 2026-07-30. `direct` vs `seed` had textbook isolation (one variable, killed at
  a single source, cannot leak) and still measured **nothing for a full day**: `a301e2e` deleted
  `_pick_side_grasp_id`, `build_seed` raised `AttributeError`, `_get_approach_seed` caught it *by
  design* ("never let seeding break the expert"), and the cell produced clean-looking numbers for a
  treatment that was never administered. A perfectly isolated comparison of nothing.
  **Why:** defensive exception handling is correct here — seeding must not break the expert — which
  makes the silent no-op the DEFAULT failure mode, not an edge case. A silent no-op and a true null
  are indistinguishable in the outcome measure.
  **How to apply:** every A/B needs a separate "did the treatment fire" channel, and it must be read
  BEFORE the outcome. `rollout_seed_stats` (per-episode `built` / `reason` / `leg`) is that channel
  here and is what caught this — the summarizer's `[SEED FIRING]` block prints it and even warns
  `!! the seed NEVER fired -- seed-vs-direct below measures nothing`. Read that block first, always.
  `test_lib_env_api.py` now guards the specific regression class (duck-typed `env.<method>()` calls
  from `lib/` are invisible to a by-name dead-code scan of `task/`).
- **Do not let a conclusion harden faster than the measurement behind it.** Three claims in
  `status_current.md` were plausible, recorded as settled, acted on, and WRONG — all on 2026-07-30,
  all refuted by data already sitting in `records.jsonl`:
  *"the carry seed never builds"* (from n=4; it built 2/4, later 7/9);
  *"tau is not the lever, use warm_seeds/ik_seeds"* (from a few failing scenes; the measured
  distribution was 3 of 6 failures fixable by `SEED_GATE_TAU` alone);
  *"attached-object IK leaves more holes"* (pure inference; both legs failed at the same rate).
  **Why:** each one then shaped decisions — the first nearly justified `CARRY_SEED=0`, which would
  have suppressed the best-firing leg. The failure mode is generalizing a shared *cause* into a
  universal *rate*, or a mechanism deduced from reading code into an established fact.
  **How to apply:** tabulate the recorded distribution before writing any "always/never/the cause
  is" sentence; mark deductions as deductions in the note itself; and when the user says the
  observed behaviour contradicts a note, believe the observation and go re-tabulate — in all three
  cases the user's read was right and the note was the outlier.
- Surface confounds proactively. Object vs algorithm vs scene-geometry are entangled in this repo;
  don't overclaim a finding's scope.
- **Isolate the parts even when the work will not be published** (user, 2026-07-30). The July
  entanglement — Hamid's bug fixes landing on top of the user's waypoint scaffolding, never
  separately measured — cost ~3 weeks of not knowing which half produced the gain, and is *still*
  unresolved: there are **zero `off`-mode records** in the entire repo. One ~18-minute cell in July
  would have answered it. The cost of skipping isolation is not reviewer scepticism, it is your own
  time spent reasoning about a system you cannot decompose.
- Make changes EASY TO REVERT — a single named flag/constant, not scattered edits.
  Aligns with [[feedback_minimal_changes]].
- External-facing visuals: NO jargon (no curobo/clutter/occluder/seeds/reachability-as-a-concept),
  simpler is better, and be honest *in the figure* — e.g. show flawed "spinning" successes as a
  distinct hatched segment rather than hiding them. See [[feedback_script_conventions]].
