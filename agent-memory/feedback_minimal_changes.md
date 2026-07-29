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
