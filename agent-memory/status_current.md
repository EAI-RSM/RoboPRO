---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold status.** Every other file holds durable knowledge.
When something here becomes permanent, move it out; when it becomes false, delete it.
**Rewrite this file, do not append to it.** Last rewritten 2026-08-07.

**Branch.** `codex/bench-script-refactor`, HEAD `351cf79`, forked from `peng-research-branch` @
`83e392f`. No PR open. Multi-planner testbench branches are dead — see
[[archive_planner_comparison]].
**Uncommitted:** the `PLACEMENT_MODE` work, `_get_carry_seed`, `carry_object_spheres.py` (new),
`sweep_gate_tau`, summarizer per-leg split; staged clean-office VLA smoke mode and opt-in
countertop evaluation-video selection; task-metric Stage C/C2 engineering described below; and the
route-visualizer work immediately below.

## Route visualizer — where it stands (2026-08-07)

Full background in [[tool_route_visualizer]]. **Uncommitted**, CPU-tested, and rendered on real
scenes: `lib/plotting.py` (+`_true_aspect_3d`, `+_draw_ground_plane`; existing functions untouched),
`metric_viz.py::_metric_path3d` (7 opt-in kwargs, all defaulting to the historical behaviour so
`clearance_metric_3d.py` / `seed_from_clearance.py` are unchanged),
`visualize_task_metric_routes.py` (table/coaster sourcing, all-legs loop, `is_minimum`,
`leg{i}_` filename ordering), `test_task_metric_route_visualization.py` (+1 test, fixture now has a
binding and a non-binding figure).

`test_task_metric_route_visualization.py`, `test_geometric_metric.py`,
`test_compare_geometric_vs_gated.py`, `test_task_metric.py`, `test_metric_buckets.py`, and
`test_lib_env_api.py` all pass. Episodes 0/2/19/33 were rendered end to end with the final code
(scene identity + regenerated-metric validation passed on each).

**NOT DONE: the 50-episode render has never been run to completion under the final code.** ~28 min.
Every previous attempt is stale because each code edit changes `config_sha256` — see
[[tool_route_visualizer]]. Existing output dirs under
`scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/`:

| dir | episodes | figures | code version | state |
|---|---|---|---|---|
| `metric_route_visuals` | 50 | 65 | `367a13a6…` | original, pre-framing-fix |
| `metric_route_visuals_v2` | 32 | 42 | `0421456c…` | partial, framing-only, superseded |

Check for a `_v3` before picking a name (`ls -d …/metric_route_visuals_v*`); run with
`--out-dir …/metric_route_visuals_v4` or the first unused name. Both older dirs are superseded and
can be deleted once a good run exists — **user's call, not the agent's.**

Deliberately NOT changed: `TASK_METRIC_ROUTE_VISUALIZATION_PLAN.md` §5/§6 still describe
minimum-legs-only and are now wrong; the user scoped the work to code only.

## BLOCKING (2026-08-07): the association study's predictor has a measured validity problem

Read [[tool_task_metric_validity]] before touching `task_metric.py`, `lib/waypoints.py`,
`bucket_spec.json`, or any post-processing of the 3000-rollout campaign. Audited on the real d10/d15
pilots and reproduced bit-for-bit offline (max abs diff 0.000e+00 on all 100 scenes):

- `eps_geom` is endpoint-pinned in **800/800** legs — it is `min(clearance at the two endpoints)`,
  not a corridor width.
- The 0.12 m tool offset puts the `grasp` waypoint 18 cm above a table carrying 6–8 cm clutter, so
  **spearman(real grasp tightness, `eps_geom_min`) = 0.078**. This is why `videos_by_clearance` looks
  wrong to the eye — the user raised exactly this and was right.
- `very_low` is 100% home-pose-pinned; `high_clearance` is grasp-pinned. The bucket label means a
  different physical quantity at each end of the scale.
- 72/100 scenes change bucket if you pick a different grasp candidate the expert itself would
  accept.

Nothing was changed in response — no code edits, no rerun, no `bucket_spec.json` revision. **The
user has not yet chosen a remedy.** The three defects have different fixes and different costs;
evaluating clearance at the contact point instead of the wrist is the cheap one and is testable
offline on CPU without a GPU or a scene rebuild (method in the note). Do not silently re-cut buckets
or rerun the campaign without an explicit decision.

This bears on Codex's `visualize_task_metric_routes.py` audit described below: with the bottleneck
endpoint-pinned in 800/800 legs, those renders will show the eps* sphere sitting on a waypoint —
usually the arm's home pose or a point floating ~18 cm above the table — rather than in a tight
gap. That is the expected output of the current metric, not a rendering bug.

**Geometric eps validation.** `GEOMETRIC_EPS_VALIDATION_PLAN.md` Stage 1 engineering is complete:
commit `372df8e`. `validate_reach_envelope.py --mode false-keep` solves only envelope-kept cells, writes
`false_keep.json`, one top-down/side/z-profile figure, `timings.json`, and
`false_keep_mask.npz` for Stage 3 route cross-referencing. The default false-prune path is
preserved. The focused CPU test, import/help check, compilation, `test_lib_env_api.py`,
`test_ring_config.py`, and `test_obstacle_set.py` pass. **Stage 1 GPU gate passed on both arms.**
Fresh current-grid `(16,71,121)` occupancy artifacts used 1.5M feasible samples/arm. Right prunes
31.70% (43,578 cells), left 27.08% (37,226); both false-prune validations found exactly 0 reachable
pruned cells. False-keeps at the fixed grasp orientation are substantial but close across arms:
right 39,051 / 93,878 kept = 41.60% (28.41% of grid), left 38,299 / 100,230 = 38.21%
(27.86% of grid). Per-z rates rise from ~39% to 54% right and ~33% to 48% left. This does not fail
Stage 1—the plan has no false-keep threshold—but makes the Stage 3 route/rank gate decisive.
**Stage 2 is complete in commit `173e1f2`.** New `lib/geometric_metric.py::geometric_eps` builds
one CPU-only envelope/obstacle/target-label volume per call and reuses it for every requested leg;
`LegResult` has the planned eight fields. Target collision blocks route nodes without entering the
obstacle EDT. `torch` imports in `lib/ik_grid.py` and `lib/labeling.py` are now function-local so
the canonical pure helpers can be reused without loading the GPU stack; their gated algorithms are
unchanged. Focused target-mask, two-leg volume-reuse, and torch/curobo-blocked subprocess tests
pass, as do compilation, all three existing CPU regressions, the reach-validator test, and the
three legacy CLI import/help checks. **Stage 3 engineering is complete in commit `ccd3bc5`.**
`compare_geometric_vs_gated.py` measures the same grasp→pad world coordinates in each live scene,
writes gated/geometric values, Spearman, route overlays, and Stage-1 false-keep overlap. Its
focused CPU artifact/control-flow test, real-mask discovery/grid checks for both arms, import/help,
compilation, dependency rule, existing CPU regressions, and legacy help checks pass. Commit
`351cf79` changes the default to the validated current-grid `_reach_cache_geometric_stage1` and
preflights it before scene setup; the `(16,71,121)` right artifact loads with 31.7% pruned / 1.5M
feasible samples.

The first corrected one-seed smoke exposed two Stage-3 defects. First, it reported
`eps_gated=0.070 > eps_geom=0.050`, but the two calls had independently snapped starts
(`(0.15,0.16,0.81)` vs `(0.17,0.15,0.81)`) and mismatched target policy, so it did not test a graph
against its relaxation. Second, 42/44 geometric route voxels lay at `zmin=0.78`; post-eps BFS picked
the shortest member of the eps-optimal component instead of a high-clearance representative.
`GEOMETRIC_EPS_VALIDATION_PLAN.md` now has §3a/§3b amendments: compare no-target and target-masked
graph pairs on exact common voxel endpoints with hard grid/EDT/FREE-subset and
`eps_geom>=eps_gated` assertions, then use a clearance-preferred geometric reporting route without
changing eps*. **Both amendments are now implemented, uncommitted.** The comparison fails
immediately if an aligned precondition or the relaxation invariant breaks; records both aligned
rank series plus native values; and writes old-BFS versus clearance-preferred height/length/profile
diagnostics. The synthetic strict-relaxation case (gated 1.0, geometric 5.0), FREE-subset and EDT
rejection cases, floor-vs-climb reconstruction, CPU-only import, target-mask, shared-volume,
artifact/report, reach-validator, seed selftests, lib/task API, ring, obstacle, compilation, and
legacy help checks pass. **Stage-3 recovery/diversity work is implemented, uncommitted.** Expected
gated early exits become auditable `not_alignable` records; true grid/EDT/FREE/invariant failures
remain fatal; `--resume-dir` is available; and the shared deterministic ring sampler supports
per-occluder radius ranges, count menus, and random formation rotation while stamping the actual
configuration into every record. Focused comparison/resume/ring tests, geometric-metric tests,
compilation, lib/task API, obstacle, and CLI import/help checks pass.

**Current diverse Stage-3 result:**
`scripts/validation/results/geometric_vs_gated/20260731-110834/` requested 10 right-arm scenes with
offsets 0.15–0.25, counts 2–5, and random ring rotation. Six aligned; four were excluded because
the table-edge filter dropped a sampled occluder. Both target-masked and no-target aligned series
have Spearman 1.0, with exact `eps_geom == eps_gated` on all six and no invariant violations. In
all six, eps* equals the smaller endpoint EDT, so the bottleneck is endpoint-pinned and many
distinct paths can maximize the same minimum clearance. Geometric and gated representative paths
do differ; one geometric route used 25/59 false-keep voxels, so **route fidelity is not established
and must not be claimed.**

**User scope decision (2026-07-31): the target claim is scalar eps* approximation for association
with actual VLA outcomes, not geometric-route fidelity.** The geometric representative path is
diagnostic only. CuRobo seeding continues to use the proper gated graph, so differing non-unique
representative paths do not by themselves fail this scalar claim. The user explicitly waived the
former predeclared `n>=8` gate as arbitrary and accepts the six aligned scenes as sufficient for
moving forward. `TASK_METRIC_CORRELATION_PLAN.md` may proceed, but a small aligned calibration on
the selected stock task's relevant leg/orientation families remains required before the main study;
the occluder scenes do not establish rank fidelity for new task geometry.

**Task-metric association protocol locked by user (2026-07-31):** association, not prediction;
one task, `put_cup_on_coaster` (training membership confirmed by the RoboPRO team via the user);
episode-level
`hard_success = task_success and not collision_metrics.is_collision`, with HSR equal to its sample
mean; `eps_geom_min` across all canonical legs as the primary predictor; no grasped-object effect on
any metric calculation; no visibility measurement or covariate (visibility remains a separate
artifact until the user raises it again); and final `N` chosen by the user. Analysis reports overall
HSR with a two-sided 95% Wilson interval and the primary Spearman association with a two-sided 95%
episode-bootstrap CI. **New user gate:** implement metric generation and a metric-only distribution
visualization first, without loading rollout/HSR data; then stop so the user can choose and freeze
the bucket count, boundaries, closure, tie handling, and `+inf` assignment in `bucket_spec.json`.
Only after that choice should outcome rollout/join and bucket-specific correlation analysis be
implemented.

**Task-metric Stage C/C2 engineering is implemented, uncommitted.** New `lib/task_roles.py`
explicitly resolves the cup/coaster roles and preflights every included obstacle mesh;
`lib/waypoints.py` builds the planning-free four-leg nominal chain actually present in
`put_cup_on_coaster` (pre-grasp, grasp, carry-to-pre-place, place; the task's lift is commented
out); and `lib/scene_provenance.py` provides stable intended scene IDs plus canonical exact scene
fingerprints. `task_metric.py` builds metric-only Study scenes, hard-asserts zero table-height
randomization and active clutter settings, calls `geometric_eps` once for all legs, records finite
versus `+inf` explicitly, and includes no outcome fields. `analyze_metric_distribution.py` rejects
any outcome/rollout/collision/success/HSR-bearing record and writes the three planned figures,
descriptive distribution summary, source-record hashes, and timings without making buckets.
Synthetic finite/tied/mixed-`+inf`/all-`+inf` checks pass; the figures were inspected and are
legible. Compilation; both new `--help` checks; `test_task_metric.py`; geometric, obstacle, ring,
and lib/task CPU checks; existing VLA/clearance/seed help checks; dependency scan; and diff check
pass under `/home/haccerkat/miniconda3/envs/RoboTwin/bin/python`. The user's first live two-seed
SAPIEN run is `scripts/validation/results/task_metric/20260731-153418`: both scenes realized 10/10
clutter objects; all four legs merged; left and right task adapters were both exercised;
`eps_geom_min` was finite (0.0671 and 0.1241 m) and was the pre-grasp value in both scenes. The run
also exposed an artifact gap: no visual scene record. `task_metric.py` now saves initialized-scene
PNGs from `countertop_camera`, `demo_camera`, and `demo_camera_2` by default, renders those three
static cameras once per seed, records their relative paths, and offers `--no-scene-images` as an
opt-out. **The real image gate passed** in the exact repeat
`scripts/validation/results/task_metric/20260731-154006`: all six images clearly show the full
Study setup, cup/coaster, ten clutter objects, furniture, and initialized arms; both scene IDs,
fingerprints, and metric values exactly match the no-image run, so capture did not perturb the
measurement. The real-record Stage C2 integration smoke also passed and wrote all required files
to `scripts/validation/results/task_metric_distribution/20260731-154205`; its n=2 summaries are
plumbing evidence only. **The 100-scene metric-only d10 pilot is now complete** at
`scripts/validation/results/task_metric/20260731-154726`, using reserved seeds 1000–1099. All 100
records are `ok` with unique scene IDs/fingerprints, all four legs merged, realized clutter 10/10,
and all 300 initialized-scene images present; arm assignment was 75 left / 25 right. Stage C2 output
is `scripts/validation/results/task_metric_distribution/20260731-160620` and proves no outcome data
was loaded. `eps_geom_min` is finite in all 100 scenes, ranges 0.0500–0.2015 m, has 78 unique values
(39 observations in 17 tie groups), and has descriptive quartiles 0.08544 / 0.11874 / 0.14009 m.
The minimum leg is pre-grasp in 56 scenes, place in 14, and an exact grasp/carry tie in 30. **The
user has now frozen four `rho_geom_min = eps_geom_min / 0.03 m` clearance buckets:** boundaries
2.5/4.0/5.0 (equivalent to 0.075/0.120/0.150 m), lower-closed/upper-open, boundary values assigned
upward, identical values never split, and `+inf` in `high_clearance`; pilot counts are 17/35/30/18.
Use clearance labels rather than outcome-difficulty labels. **Stage C2 closure engineering is now
complete, uncommitted:** `bucket_spec.json` binds the approved rules to exact records/config/summary
hashes; `lib/metric_buckets.py` is the standard-library-only loader/assignment implementation;
`validate_bucket_spec.py` reproduces the real pilot counts 17/35/30/18 over seeds 1000–1099; and
`test_metric_buckets.py` covers upward boundary assignment, ties, `+inf`, malformed boundaries,
inconsistent metre/rho definitions, bad provenance counts, and malformed metric records. Focused
tests, compilation, the real-source validator, task-metric/lib API/geometric/obstacle/ring
regressions, and relevant CLI help checks pass. No outcome data was loaded. This pilot pool stays
separate from final rollout `N` and is excluded from HSR inference.

**Additional metric-only d15 check (2026-07-31):** the user's 100-seed run is
`scripts/validation/results/task_metric/20260731-163538`; all 100 records are valid and contain no
outcome fields. Configured density is 15; realized clutter counts are 15 in 98 scenes, 14 in one,
and 13 in one. Distribution artifacts are
`scripts/validation/results/task_metric_distribution/20260731-170100`. All `eps_geom_min` values
are finite, range 0.0500–0.2121 m, and have descriptive quartiles 0.07263 / 0.10124 / 0.12237 m.
The user explicitly chose to keep the already-frozen rho buckets unchanged for both d10 and d15;
do not revise `bucket_spec.json`.

**The training-membership gate is cleared:** on 2026-07-31 the user reported direct confirmation
from the RoboPRO team that
`put_cup_on_coaster` was included in the checkpoint's `roboreal_lerobot` training data. **The
narrow Stage-A rollout/outcome plumbing is now implemented, uncommitted.** `vla_rollout.py` adds an
additive stock-task mode with task/subdir/config/density/checkpoint provenance, a required pinned
step limit, deterministic instruction-bank lookup, strict collision-metric validation,
`task_success`, and episode-level `hard_success`; `summary.json` reports pilot HSR and whether the
outcome is non-degenerate. The first clean attempt,
`scripts/validation/results/task_metric_vla_pilot/clean_viability/20260731-170002`, is **invalid and
must never be used as HSR evidence**: seeds 2001 and 2005--2008 were labelled successful after one
action and wrote nearly empty videos. Root cause was the stock task's signed XY comparison in
`put_cup_on_coaster.check_success`, which accepted large offsets in one direction. The predicate is
now corrected to absolute XY error while preserving the 2 cm per-axis tolerance and both-open-
grippers requirement; do not add a rollout-length heuristic. The same attempt exposed that Study
dropped the driver's explicit `countertop_camera` selection and fell back to `demo_camera`;
`Study_base_task` now retains that config. **The corrected clean pilot passed** at
`scripts/validation/results/task_metric_vla_pilot/clean_viability/20260731-171503`: HSR was 3/10;
successes took 129--130 steps, failures took the full 600, all ten videos were playable from
`countertop_camera`, and visual inspection supported the labels. No one-step false success
remained.

**The 3000-rollout association campaign is complete** at
`scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037`: exactly
1000 committed episodes at each of d6/d10/d15, 554 hard successes (HSR 0.1847, Wilson 95%
[0.1712, 0.1990]), 3000 countertop videos, and zero missing videos. The first implementation only
regenerated outcome reports every 10 rollouts and grouped videos by hard outcome; it incorrectly
left all clearance-dependent work as an unrelated manual offline command.

**Integrated crash-safe post-processing is now implemented, uncommitted (2026-08-07).** Rerunning
`task_metric.py --rollout-run <run>` uses `<run>/metric_postprocess/`, commits one atomic fsynced
`episodes/episodeNNNNNN.json` per metric, repairs JSONL from those authoritative files, validates
an immutable config binding the rollout/manifest/bucket spec/metric code, and skips committed
metrics. Every 10 commits it regenerates the three metric-distribution plots, provisional joined
tables/association plots, and idempotent `videos_by_clearance/<bucket>/<hard_outcome>` symlinks.
Partial reports require an exact fingerprint-valid subset and defer bootstrap inference; 3000/3000
switches to the strict one-to-one join, 10,000-resample final bootstrap, and a zero-missing-video
gate. `vla_rollout.py --postprocess-metrics` now starts this phase automatically after policy/model
teardown; resuming a fully collected rollout skips model startup and goes directly to metrics.
Focused crash-repair, immutable-config, partial/final join, repeated-link, integrated n=10 report,
rollout durability, task-metric, office-smoke, CLI, compilation, and diff checks pass.

**The n=10 live post-processing gate passed after one plotting fix.** The first every-10 report
hit Matplotlib's `yerr must not contain negative values`: the 0/3 bucket's Wilson lower endpoint
was `5.55e-17` from floating-point roundoff, slightly above the observed rate of zero. Atomic
metric records were unaffected. `wilson_interval` now clamps bounds to `[0,1]` and guarantees they
contain the observed rate; the integrated regression deliberately includes an all-failure bucket.
The real run currently has 10 committed metric episodes, all five plots, joined tables and
summaries, and 10 valid `videos_by_clearance` links; `report_state.json` is provisional with target
3000. The first resume then exposed a JSON-normalization bug: `gate_tau_sweep` was a tuple in the
live Python config but a list after the immutable JSON round trip, so raw dictionary equality
rejected two configurations with the same valid hash. `SeedMetricConfig` now uses a JSON-native
list; the serialized values, stored config hash, and stored metric-code hash are unchanged. The
actual saved config now passes the resume validator exactly. No rollout needs rerunning.

**Reporting hierarchy update (2026-08-07):** the user made d6, d10, and d15 separate primary
analyses in that order; mixed-density results are secondary only. Every refresh now writes each
density's three metric-distribution plots and summary plus joined JSONL/CSV, association summary,
HSR-by-clearance plot, and metric-by-outcome plot under `by_density/d6`, `by_density/d10`, and
`by_density/d15`. Top-level association plots and `secondary_pooled` in
`association_summary.json` are explicitly labeled pooled secondary results. The stopped real run
has since resumed and is actively advancing toward 3000/3000; inspect its live `report_state.json`
instead of relying on a count in memory. Do not edit the files bound into `_metric_code_version`
until that run completes. Deleting only `<rollout>/metric_postprocess` is scientifically safe but
would discard the completed metric computations, so resume in place.

**Per-rollout 3D epsilon audit engineering is complete, uncommitted (2026-08-07).**
`visualize_task_metric_routes.py` is a separate crash-safe replay/backfill: it validates source
config/manifest/rollout/metric hashes, regenerates and verifies each committed metric, reuses
`metric_viz._metric_path3d` for every tied `eps_geom_min` leg, atomically commits PNGs plus
source-bound episode records, and builds idempotent density/bucket/outcome video+PNG symlink indexes.
The renderer's existing defaults and filenames remain compatible; the new caller labels the route
as geometric and explicitly not the executed VLA trajectory. The user narrowed this to a small
visual audit rather than a 3000-scene backfill: no selector now means `--first 50`, and completion
is reported against that selected prefix while retaining the 3000-rollout source provenance.
`--episode` and `--stratified-per-cell` remain optional selectors, with stratification requiring
all d6/d10/d15 bucket cells. The focused route-visualization test,
actual-artifact schema/source preflight, CLI/help and incomplete-full refusal, compilation, diff
check, and metric-bucket/correlation/task-metric/geometric/obstacle/lib-env regressions pass. No
real SAPIEN episode or 50-scene audit was run by the agent; those remain user-owned.

**VLA rollout validation.** `VLA_ROLLOUT_PLAN.md` Stages 1–4 complete (`88e4fd5`, `af94c1e`,
`98422a8`, `92a5023`), plus `e4a2624` and `21a64d7` for the two planner-free bugs. **The office
control scene passes end to end** — pi05 loads, infers, and the user confirmed visible arm motion in
the countertop MP4. Mechanism/gotchas: [[tool_vla_pi05_port]].

**Next/user-owned: the occluder-scene smoke.** One seed, 50 steps,
`--scene occluder --run-type occluder_smoke`. Pass = the video shows the bottle/olive-oil scene AND
arm motion, inference completes, one complete JSONL record + playable MP4. **Task success is NOT
required** and should not be read as a result — the checkpoint is finetuned on RoboPRO *real* data,
so near-floor success on an OOD sim scene is expected.

Not yet done, deliberately: no clearance/eps* join, no multi-seed run, no figures.

---

## THE HEADLINE FINDING (2026-07-30) — the experiment was measuring the wrong leg

25-seed direct-only census, curated, **`OFFSET=0.15-0.25 OCCLUDER_COUNTS=2,3`**
(`results/phase4_approach_mode/20260728-151618/`). **64% success (16/25)**, mean 59 s/episode.
Failures: `placement:pre_place_descent` 5 (20%), `grasp` 3, `grasp_candidate_selection` 1.
**Zero failures at `pre_grasp`. Zero at `carry_transit`.**

Per-stage planning effort is where the real answer is:

| stage | n | median attempts | mean | max | direct-attempt failures |
|---|---|---|---|---|---|
| `pre_grasp` | 21 | **1** | 1.8 | 8 | **0** |
| `grasp` | 21 | 1 | 1.0 | 2 | 0 |
| `lift` | 21 | 1 | 1.0 | 1 | 0 |
| **`carry_transit`** | 21 | **8** | **12.7** | 24 | **8 of 21** |

**`pre_grasp` — the ONLY leg the seed currently fires on — converges first try, every time. There
is no headroom there.** That fully explains the 22%-vs-20% null of the first big run: the metric
was not failing to help, it was measured on a leg that was never in trouble.

All the difficulty is at `carry_transit` (median 8 curobo attempts; 8/21 exhaust the direct
attempt and survive only because the shrinking-waypoint fallback rescues them) — the leg the seed
does NOT currently reach.

**Two consequences, both decisive for how the experiment is run:**
1. **Success rate is the wrong outcome measure.** `carry_transit` never kills an episode (the
   fallback always saves it), so seeding it cannot move success rate — only *effort*. Use
   `carry_transit` attempts (median/mean, already in `records.jsonl` and already reported by the
   summarizer), with success rate quoted alongside unconditionally. A continuous measure at n=25
   has far more power here than a binomial at n=50.
2. **The carry seed is not optional after all.** It was correctly *not* required for validity, but
   the carry leg is the only place with anything to measure, so without it there is no signal to
   find. Right aim, badly sequenced — this table should have been produced BEFORE any of the
   Phase C machinery. See [[feedback_minimal_changes]].

## Scene settings matter more than any code change

Cross-tab of the old 50-seed curated run (n=100, both cells) by minimum ring radius:

| min radius | n | success | `grasp` failures |
|---|---|---|---|
| 0.10 – 0.14 | 72 | 21% | **58%** |
| 0.15 – 0.25 | 28 | 46% | 21% |

By occluder count: 2 → 33%, 3 → 38%, **4 → 16%**.

At a 0.10 m ring radius the target (7.7 cm across) and occluders (~8 cm) leave ~2 cm for the
gripper — the **grasp is infeasible**, so those episodes die upstream of anything the seed touches
and contribute pure noise to both arms. Tightening to `OFFSET=0.15-0.25 OCCLUDER_COUNTS=2,3` took
success 22% → 64%, which is also near the p≈0.5 sweet spot for paired tests.

**This is a legitimate design choice and must be DISCLOSED**: the selection criterion is "the
target is graspable", which is orthogonal to the seed and favours neither arm. It would become
cherry-picking only if tuned on the seed's success rate.

## FIRST SIGNAL ON THE RIGHT MEASURE (2026-07-30, `20260730-121853`)

Paired `carry_transit` attempts, same scene seed, both cells, carry seed genuinely delivered
(`ct_seeded=True`) on all three:

| scene seed | direct | seed | Δ |
|---|---|---|---|
| 0 | 3 | 2 | −1 |
| 1 | 1 | 1 | 0 |
| 2 | 10 | 8 | −2 |

Median 3 → 2, mean 4.7 → 3.7. Two improvements, one tie, **zero worse**. n=3, sign test p=0.5 —
statistically nothing, but it is the predicted direction on the predicted measure on the only leg
with headroom, which had never previously lined up. This is the measure to power up, not success rate.

**Success rate at n=5 is pure noise — proven, not asserted.** `20260730-113818` gave direct 40% vs
seed 20%; `20260730-121853` gave direct 20% vs seed 40%. Same code, opposite sign. Do not read it.

**`occluders=4, offset=0.2` (fixed ring, random rotation only) is the USER'S DELIBERATE CHOICE, not
a bug.** Do not "fix" it. Rationale: the comparison is paired by scene seed, so randomizing the ring
buys no validity, while a fixed geometry tightens the spread of the `carry_transit` attempt
distribution and so gives more power per pair on the paired test. An agent flagged it twice as a
plumbing failure; it is not. The only real cost is yield — at 4 occluders more episodes die at
`grasp` before reaching `carry_transit` (2 of 5 in `20260730-121853`), so raise `NUM_SEEDS` to
compensate rather than loosening the scene.

**Cost now that both legs build:** seeded 448.3 s/rollout vs direct 69.3 s (6.5x); approach build
152.9 s + carry build 253.7 s. One `OutOfMemoryError` reappeared at this load — the leak fix is
still unproven at scale.

## Seed firing rate — CORRECTED 2026-07-30. "The carry seed never builds" was WRONG.

An earlier version of this file claimed the carry seed never builds. **It does.** Tabulating every
`rollout_seed_stats` row in `results/phase4_approach_mode/` (1101 rows; 80 real build attempts, the
rest `cached`) gives:

| leg | real attempts | built | failed |
|---|---|---|---|
| `carry` | 4 | **2** | 2 (both `gate disconnects`) |
| `approach` (post-split, 2026-07-28 runs) | 8 | 4 | 4 (both `gate disconnects`) |

**Both legs fail at the same rate and for the same reason** — so the gate cut is NOT carry-specific,
and "attached-object IK leaves more holes" (plausible, was asserted in conversation) has **no
support** in this data. The user's own observation — routes visible forwards AND backwards most of
the time — is the correct read; the note was the outlier.

**Provenance of the error, worth remembering:** the sweep result was real, the deduction from it was
plausible, and then n=4 got written up as "never." Failure mode = generalizing a shared *cause* into
a universal *rate*. Check the recorded rate before claiming one; `rollout_seed_stats` has it.

**MEASURED 2026-07-30** (`results/phase4_approach_mode/20260730-121853`, 5 seeds/cell, post-fix):

| leg | fresh builds | built | rate |
|---|---|---|---|
| `approach` | 8 | 2 | 25% |
| `carry` | 3 | 3 | **100%** |

Lifetime carry: 7 built / 9 attempts. `CARRY_SEED=0` would suppress the best-firing leg — do not set it.

### `SEED_GATE_TAU` IS the lever for most failures — CORRECTED 2026-07-30

The claim below ("levers are warm_seeds/ik_seeds, NOT SEED_GATE_TAU") is **false for the majority
of scenes**. The 6 approach-leg no-route reasons in the 20260730-121853 run:

| reason | n |
|---|---|
| `SWEEP: tau=0.5 WOULD connect` | 2 |
| `SWEEP: tau=1.0 WOULD connect` | 1 |
| no route even ungated (geometry) | 1 |
| `no tau up to 2.0` (the NaN case) | 1 |
| `exception:OutOfMemoryError` | 1 |

**3 of 6 are fixed by raising `SEED_GATE_TAU` alone.** Only ONE is the NaN case that motivated the
whole relaxed-seed plan. Cheapest available action by far: rerun with `SEED_GATE_TAU=1.0` — one env
var, no code, no rebuild, plausibly 2/8 → 5/8 approach firing. Do this BEFORE any relaxed-seed work.

Same failure mode as the "never builds" error: a real sweep verdict on a few scenes generalized into
a universal claim. The per-scene verdict is in `seed_stats.reason`; read the distribution, not one case.

### What the tau sweep does and does not establish

`sweep_gate_tau` (2026-07-28, `seed_from_clearance.py`) re-runs the widest path at a looser tau
ladder over volumes already in memory — no IK, no rebuild. On the scenes where it fired: **"no tau
up to 2.0 connects it."** At a large enough tau the gate admits every edge the ungated pass does, so
the *threshold* is not the blocker **on those scenes**. It only fires when `merged_u and not
merged_g`, i.e. on already-failing scenes — it says nothing about the typical case.

**NaN holes are a DEDUCTION, not a measurement.** `_wrap_linf` (`lib/continuity.py`) returns NaN
where a voxel has no converged branch (`warm_start_branches` leaves those cells NaN), and
`NaN <= tau` is False at EVERY tau — permanently un-gateable while still counting FREE for the
ungated pass. That mechanism fits the sweep result, but nobody has counted NaN voxels or confirmed
the severed edges touch them. **Verify before building on it.** If true, levers are `warm_seeds` /
`ik_seeds`, not `SEED_GATE_TAU` and not `SEED_RES`. The sweep did prevent a res-halving that would
have ~4x'd a 190 s build for no reason.

**The design mismatch the user identified:** the builder applies *metric-grade rigor to a hint*.
The joint gate exists so eps\* means something; a seed is only a trajopt initialization and does
not need to be provably followable. The contained fix (NOT yet built): take the **ungated** route
(which exists in exactly the failing cases), read the raw per-voxel IK configs already computed and
**discarded** at `seed_from_clearance.py:196` (`qfield`), and enforce continuity along that single
1-D route instead of over the whole 3-D volume.

## How to run the clean experiment RIGHT NOW (no revert needed)

Every flag added defaults to the OLD behaviour — `_placement_mode()` returns `scripted` unless set,
`CARRY_SEED` follows the approach mode. Only the A/B driver opts into the new path.

```bash
# NOTE: CARRY_SEED=0 removed from this recipe on 2026-07-30 -- see below. Leave CARRY_SEED
# UNSET so it follows APPROACH_MODE, and read the carry firing rate off rollout_seed_stats.
PLACEMENT_MODE=scripted \
OFFSET=0.15-0.25 OCCLUDER_COUNTS=2,3 \
NUM_SEEDS=50 bash scripts/validation/run_approach_mode_ab.sh
```

With those two flags the only differences from pre-session code are a stage label
(`pose` → `carry_transit`) and richer `seed_stats.reason` diagnostics. Neither affects behaviour.

**`CARRY_SEED=0` — the argument for it COLLAPSED on 2026-07-30, do not set it blindly.** The case
was "the carry seed never fires, so the seeded arm pays ~190 s/episode for nothing: an uncontrolled
arm-only asymmetry with zero upside." That rested entirely on the never-builds claim, which is
false — it built 2 of 4 recorded attempts (see the corrected section above). If it fires at any real
rate, `CARRY_SEED=0` **suppresses the treatment on the only leg with measurable headroom**
(`carry_transit`), which is the opposite of what the experiment needs.

What survives: the asymmetry itself is real (only the seeded arm does that ~190 s build, so RNG
draws and GPU memory pressure differ between arms) — but it is the *treatment*, not a confound, the
moment the seed actually fires. **Measure the carry firing rate first**, then decide.

Path choice: `PLACEMENT_MODE=scripted` = the original experiment, narrower claim, ~50 episodes of
prior mileage. `PLACEMENT_MODE=direct` = broader generality claim, new path, less mileage.

## Scene-agnostic strip — DONE, and the audit is in [[domain_expert_baseline]]

All three scene-specific mechanisms are off in `direct`; the full three-tier reachability audit,
the surviving cosmetic residue, and the optional ~10-line cleanup live in [[domain_expert_baseline]].
GPU-exercised incidentally on 2026-07-30 (the grasp-filter widening ran without incident).

## Verified on GPU

Attached-sphere transfer works (0 `no_attached_object`; carry route built once, 13 voxels,
eps=0.085 m, object extent 0.157 m). Memory across episodes 5.32 → 7.23 → 6.76 GiB allocated, the
**same pattern on three independent runs**, episode 1 below episode 0's peak → returned between
episodes. **No leak signature, but unproven over 50 episodes.** Costs: approach build ~128 s +
carry build ~190 s (the carry build costs the same whether or not it produces a route) = 82% of a
385 s seeded episode; direct episode 49–70 s.

## Bugs found and fixed this session

- **`mid_pose` regression (self-inflicted by Phase A).** `mid_pose` is an INTERMEDIATE — a blend
  toward `place_pre_pose`. With the scripted chain it started above the pad so the blend landed at
  the pad; with the chain OFF it starts at the LIFT pose and stops half way →
  `landing_search_preconditions_not_met(xy_distance=0.206 m)`, with `carry_transit` the episode's
  LAST plan. Now one hop straight to `place_pre_pose`, seeded end-to-end; scripted unchanged.
- **Empty-gripper detection.** The robot yml's `attached_object` placeholders are radius **+0.001**
  — the same value as `CUROBO_ATTACH_SPHERE_RADIUS` — so a radius-only "is anything attached" check
  passes an EMPTY gripper and would have labelled the whole carry grid as if nothing were held.
  Detection is now **centre**-based (a real attach spreads spheres ~0.12 m from the link origin;
  placeholders sit exactly on it). Caught by a stub test, not by a GPU run.
- **`rollout_plan_effort` is NOT a stage verdict.** `_record_plan_effort` fires only on the DIRECT
  attempt; the shrinking-waypoint fallback records nothing. A stage can read `Fail, attempts=24`
  and the episode still succeed. Read `rollout_success` / `rollout_failure_stage` for outcome;
  treat `plan_effort` as "how hard was the first attempt".

## NOT GPU-verified

The bench_script refactor itself (`ea31499`..`abb917a`) — REFACTOR_PLAN §7.6 requires one A/B cell
showing the firing rate unmoved from `83e392f`; the `83e392f` leak fix over a long run; the Phase C
carry seed at scale (it never builds); the true-mesh `--occ-shape` path end to end; the LEFT arm
reach envelope.

## Next steps, in order (as of 2026-07-31)

1. ~~Run the milk-box verification.~~ **DONE 2026-07-30: 8/15 = 53.3% vs the old 0/16, p=0.0008**
   (`20260730-160723`). Headline result; see [[domain_expert_baseline]]. Seeding is now
   **experimental**, not the headline — do not force a positive seed result.
2. **Add the `off` cell** — `MODES="off direct seed"`. The single most load-bearing missing
   measurement in the repo; see [[domain_expert_baseline]]. ~18 extra minutes.
3. **`SEED_GATE_TAU=1.0`** on the next seeded run — the sweep named the exact threshold in 3 of 6
   approach-leg failures. One env var, no rebuild. Note it loosens the joint-continuity gate, so
   eps\* is not comparable across the change (identical in both cells, so the A/B is unaffected).
4. **Then size the real run** on paired `carry_transit` attempts, success rate quoted alongside but
   not leaned on. At 4 occluders roughly 60% of seeds yield a usable pair.
5. Still open from before: P3 cache split (one grid build serves any endpoint pair — the grid
   depends on arm/scene/orientation/attached but NOT endpoints) against the 82% overhead; strip the
   `ROBOPRO_SEED_DUMP` / `ROBOPRO_SEED_ROUNDTRIP` debug blocks from vendored `motion_gen.py`.

## Uncommitted as of 2026-07-31 (nothing from 2026-07-30/31 is committed)

`task/grasp_mixin.py` (restored `_pick_side_grasp_id`; `horiz` filter dropped),
`lib/planning_tuning.py` (`GRASP_CANDIDATE_LIMIT` 4→8), `lib/scene_constants.py` (occluder-asset
table), `analyze_occluder_visibility.py` (`--occluder-asset`, record stamp, banner),
`metric_viz.py` + `seed_from_clearance.py` + `task/seeding_mixin.py` (seed-route visuals: weld
segments + per-leg labels), and NEW `test_lib_env_api.py`. Plus Codex's `lib/scene_build.py` and
`vla_rollout.py`. All CPU-tested; only the grasp change and the visuals have touched a GPU.

**Archived A/B results live in `~/.local/share/Trash/files/phase4_approach_mode/`** (7 runs incl.
the 25-seed census `20260728-151618`) — still readable, not yet permanently deleted. The 2026-07-30
runs are in `scripts/validation/results/phase4_approach_mode/`.

**USER DIRECTIVE 2026-07-28: placement has ALWAYS been a problem area — do NOT go deep fixing it.**
Only make fixes that directly impede moving to the next step. `place_actor` / landing-search /
object-ejection failures are to be reported and left alone.
