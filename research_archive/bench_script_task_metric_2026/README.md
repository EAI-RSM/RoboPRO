# Retired bench-script task metric (2026)

This directory preserves the core source and bucket specification retired from
`customized_robotwin/script/bench_script/` after commit `3dc788a`.

The metric was retired because its measured construct validity was inadequate for the intended
rollout-difficulty analysis: all 800 audited legs were endpoint-pinned, and the fixed wrist
waypoint's clearance had Spearman correlation 0.078 with clearance where the fingers close. The
source remains useful for inspecting or recovering the research logic, but it is cold archival
material rather than a supported command or package.

Imports and source-hash paths are intentionally not maintained. There is no `__init__.py`, and the
files are not expected to run against the current active tree without deliberate restoration work.

The 620 completed records from the partial 3000-episode post-process remain unchanged at:

`scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`

The former tests and execution plans were not copied here. They remain recoverable from Git
history at and before `3dc788a`.
