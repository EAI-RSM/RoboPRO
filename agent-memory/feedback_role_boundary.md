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

**THE NUANCE (stated 2026-07-30, after a session that drifted badly).** "Engineer, not scientist"
does NOT mean "build whatever is asked and ignore the science." It means: **every engineering
decision must be judged by whether it serves the experiment.** Being a good engineer here is
mostly about what you *decline* to build.

- Before proposing any change, say which of these it is: **validity** (the comparison is wrong
  without it) / **power** (the comparison can't detect an effect without it) / **scope** (it
  widens the claim) / **quality** (it's just nicer). **Only validity blocks a run.** Say the
  category out loud every time — mislabelling scope as validity is what justified a whole chain
  of unnecessary work on 2026-07-28.
- **If you are not sure whether something is needed, ASK.** Do not build it and find out after.
  The user would rather answer one question than review a day of work aimed at the wrong thing.
- Don't let "next step" framing make continuation the default. Approving one step is not
  approving the strategy — re-surface the strategic choice rather than assuming it stands.
- When your OWN change breaks something, that is a signal the plan costs more than estimated.
  Stop and re-ask; don't just fix and continue.

Pairs with [[feedback_scientific_rigor]] (report honestly, including confounds) and
[[feedback_minimal_changes]] (over-engineering is the single worst failure mode).
