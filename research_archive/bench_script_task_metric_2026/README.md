# Retired: geometric-vs-gated eps* validation (2026)

This directory holds the **one-time methodology validation** retired from
`customized_robotwin/script/bench_script/` after commit `3dc788a`:

| File | What it was |
|---|---|
| `compare_geometric_vs_gated.py` | Stage-3 A/B: envelope-only geometric eps* vs the gated IK metric on identical scenes |
| `validate_task_geometric_ranking.py` | calibration harness for that A/B (`summarize_task_calibration`) |
| `validate_bucket_spec.py` | reproduces the frozen `bucket_spec.json` pilot counts 17/35/30/18 over seeds 1000-1099 |

Its spec is `plans/GEOMETRIC_EPS_VALIDATION_PLAN.md`, recoverable from Git at or before `3dc788a`,
as is `test_compare_geometric_vs_gated.py`.

Imports and source-hash paths are not maintained here. There is no `__init__.py`; these files are
cold archival material, not runnable against the current tree without deliberate restoration.

## What was NOT retired — the reusable metric pipeline is back in the active tree

An earlier revision of this archive also swept up the metric-generation and analysis pipeline. That
was broader than intended and has been reversed. Live again under `bench_script/`:
`task_metric.py`, `analyze_metric_distribution.py`, `analyze_metric_correlation.py`,
`visualize_task_metric_routes.py`, `bucket_spec.json`, `lib/geometric_metric.py`,
`lib/metric_buckets.py`, plus their checks in `bench_script/checks/` and
`plans/TASK_METRIC_CORRELATION_PLAN.md`.

## Read this before building on the restored pipeline

The predictor has a **measured construct-validity problem**, documented in
`agent-memory/tool_task_metric_validity.md`: all 800 audited legs were endpoint-pinned, and the
fixed wrist waypoint's clearance had Spearman 0.078 with clearance where the fingers actually close.
The cheap remedy — evaluate clearance at the contact point rather than the 12 cm wrist offset — is
testable offline on CPU against the existing 3000 rollouts. Do that before extending the pipeline to
new tasks, not after.

Also task-specific and needing generalization for a custom task: `task_metric.py`'s `SUPPORTED_TASK`
pin, `lib/task_roles.py`'s cup/coaster role resolution, `lib/waypoints.py`'s four-leg chain, and
`bucket_spec.json`'s boundaries (frozen to the `put_cup_on_coaster` d10 pilot).

## The 620 partial records

`scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`
is unchanged, at 620/3000 with `processing_complete: false`.

**They are not resumable, and were not made unresumable by this archive.** Resuming validates
`task_metric.py::_metric_code_version()` against the stored
`33061fcbecadf56c8a52283b9b274458ba5b20474d7107a1fc84f6e6494b91f3`. That hash matches no commit in
the repo — the campaign was run from an uncommitted working tree (see `status_current.md`), so the
exact source state no longer exists. It is also moot: those records were computed with the predictor
measured above, so any remedy recomputes all of them.
