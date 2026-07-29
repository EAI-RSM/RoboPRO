---
name: feedback_scientific_rigor
description: "User holds experiments to a science bar: faithful controls, proactive confounds, scene-specific mechanisms OFF, easy-to-revert changes, jargon-free honest visuals"
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
- Surface confounds proactively. Object vs algorithm vs scene-geometry are entangled in this repo;
  don't overclaim a finding's scope.
- Make changes EASY TO REVERT — a single named flag/constant, not scattered edits.
  Aligns with [[feedback_minimal_changes]].
- External-facing visuals: NO jargon (no curobo/clutter/occluder/seeds/reachability-as-a-concept),
  simpler is better, and be honest *in the figure* — e.g. show flawed "spinning" successes as a
  distinct hatched segment rather than hiding them. See [[feedback_script_conventions]].
