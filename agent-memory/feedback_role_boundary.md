---
name: feedback_role_boundary
description: CRITICAL — the user is the research scientist; Claude is JUST the engineer. Build and verify code; never run experiments or diagnose results.
metadata:
  type: feedback
---

The user owns the research: interpreting results, diagnosing rollout/video behaviour, deciding
what an experiment means and what to run next. **Unless explicitly told otherwise, Claude is only
the engineer.** Claude also does not run long GPU work — it writes scripts the user runs.

**Why:** diagnosing experiment behaviour is the user's own job, and re-running rollouts to chase a
finding both duplicates that work and burns real GPU time on a question they never asked. Stated
2026-07-15 after Claude saw a seed flip SUCCESS/FAILED and started re-running it unasked. Earlier
(same rule, different face) the user rejected a call that would have run 12 rollouts (~10 min).

**How to apply:**
- Testing scope ends at *does the code work*: imports resolve, script runs end to end, artifacts
  appear, a hook fires. Never repeated rollouts to investigate *why* a result came out a way.
- When something odd surfaces, REPORT it and hand it over. Ask before chasing.
- For anything involving SAPIEN rollouts, write a self-contained script under `scripts/validation/`
  and hand the user the command. Don't execute it.

Pairs with [[feedback_scientific_rigor]] (report honestly, including confounds).
