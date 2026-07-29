---
name: reference_artifacts
description: "Published Artifact URLs from this project, with the caveats attached to each"
metadata:
  type: reference
---

Republish the same local file path to keep a URL, or pass the URL as `url` from a new chat to
update in place.

- **Planner comparison bar chart** (2026-07-10, jargon-free, external audience) —
  https://claude.ai/code/artifact/6a31317d-6db1-4f87-96b8-8229a685f35a
  (favicon 🦾, file `scratchpad/planner_comparison.html`). Shows baseline 4/50 = 8% (only 2/50
  clean — 2 involved an unintended SPINNING motion, drawn as a hatched sub-segment) vs
  reachability 17/50 = 34%, curated task, clutter density 8.
  **CAVEAT: those numbers were given verbally, NOT from a verified frozen-knob same-seeds run**,
  though the chart asserts "same 50 setups per method". Also "baseline" is not the dev planner.
  See [[archive_planner_comparison]].
- **Hamid pipeline, detailed diagram** (2026-07-14) —
  https://claude.ai/code/artifact/0eb890d5-8d00-4362-921d-23cb30b60f12
  (file `scratchpad/hamid_pipeline.html`). Vertical flowchart of `play_once`, stages labelled with
  real methods, 3 lanes (plan/execute/verify), grasp-retry loop. Each stage has a dashed **count
  chip** wired to the `rollout_stage` buckets — fill them from a run's stage table to turn it into
  an annotated diagnostic. Theme-aware.
- **Hamid pipeline, simplified export figure** (2026-07-14) —
  https://claude.ai/code/artifact/e040a7be-f5f9-4113-97e3-49cf0cabf25e
  (file `scratchpad/hamid_figure.html`). 5 plain-language steps, no code detail, locked to a single
  light theme so it screenshots identically. Framed as a card = the screenshot target. For slides.
