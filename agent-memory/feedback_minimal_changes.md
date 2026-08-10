---
name: feedback_minimal_changes
description: Prefer the smallest change; the user pushes back hard on over-engineering and abstraction
metadata:
  type: feedback
---

The user repeatedly pushes back when Claude over-engineers. On the subgoal work Claude twice began
building a framework (a `_subgoal_line`/`_run_subgoal_line` traversal system) when the real change
was editing a 3-line `(name, pose)` list — "you may be overthinking this. this shouldnt require
much changes... what is going on?" Reinforced later: "this is literally just plotting numbers
defined. what are you doing", "please dont overthink this".

**Why:** the user designed these scripts to be hand-edited (they run a fast edit-and-rerun loop and
change numbers by hand between turns). They value a small obvious diff over a clean abstraction,
and they read the changes closely.

**How to apply:**
- "Just add/tweak a subgoal / parameter" = minimal edit to the existing structure. No new methods,
  indirection, or refactors unless asked.
- A conceptual description ("backward is the reverse of forward") is *reasoning*, not a request to
  build a mirroring engine — usually the edit is reordering a list.
- Prefer ONE named tunable constant at the top of the file (e.g. `FORWARD_SUBGOAL_Z`) over
  scattered literals, with sign/per-arm logic automatic so a single positive number just works.
- Asked to plot values the code already defines: wire it up and stop. Don't restructure signatures.
- If a bigger refactor really is warranted, propose it in one line and let them opt in.
- Keep explanations short.

**"Over-engineering is the worst thing you can do" (stated 2026-07-30).** Stronger than a style
preference — it is the failure mode that costs the most, because it burns the user's time on
review-and-rerun cycles for work that was never on the path to the result.

The canonical example, 2026-07-28: the user said *"unless the mechanism is scene agnostic, I want
it turned off."* That is a **removal** request, and the removal was one small diff. Claude then
treated it as "turn it off AND keep the expert working" and built a whole replacement — a carry
seed, a held-object sphere module, a gate-tau sweep — none of it asked for. Letting the stripped
leg simply fail *was* the valid experiment; Claude had even called it "the floor" in its own plan
and then went and propped the floor up. The user's summary: *"this is more about removal, not new
features, so i dont see the big challenges in this."*

**How to apply:**
- "Turn X off" means delete/disable X. It does NOT license building a replacement for X. If the
  removal leaves something broken, REPORT that and ask — the breakage may be the intended result.
- Exhaust the data already on disk before writing code or asking for a GPU run. On 2026-07-28 the
  single most useful finding of the session came from a cross-tab of records that already existed
  and could have been run in the first five minutes.
- State a **stopping condition** with any multi-step plan ("done when X"). Without one there is
  always one more defect to fix.
- Count each smoke/rerun cycle as ~15 min of the user's time plus a context switch. Optimise for
  time-to-answer, not correctness-per-change.
