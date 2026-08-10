---
name: domain_expert_baseline
description: "What the `direct` baseline actually contains, where its performance came from (Hamid's July commits), what has been stripped for scene-agnosticism, and the measured numbers behind all of it"
metadata:
  type: project
---

**`direct` is NOT a naive baseline.** It is a heavily engineered expert with exactly two things
removed. Every claim about "the baseline" has to be read against that. Established by git
archaeology 2026-07-30 — expensive to re-derive, so it is written down here.

## Provenance: where the performance came from

Hamid recorded validation numbers in his own commit messages, all on occluder-present seeds. This
is the only surviving record of the trajectory and it is not reproducible today:

| state | success |
|---|---|
| before `2df658b` (≤ 2026-07-07) | **0/16 = 0%** |
| `2df658b` (07-08) trajectory-replay divergence + placement smoothness | 5/16 = 31% |
| `385d969` (07-09) fallback truncation + local-waypoint search | 6/16 = 37.5% |
| `a84a6cf` (07-13) grasp simplification + attached dynamics/release | 43.5% → 46.4% → 42.0% → **47.8%** (four 69-seed sweeps) |
| `5bbe2d2` (07-11) descent slicing + local landing search | "meaningful jump", no single figure |

**The user's remembered "0% baseline" is real but predates their assumed 2026-07-10 boundary by two
days.** The single largest jump (0% → 31%) is `2df658b` on 07-08. Under a July-10 cut, that jump
falls *inside* what they were calling the baseline. Current runs at ~45–50% sit exactly where Hamid
left it — nothing anomalous to explain.

**The waypoint scaffolding is the USER's, not Hamid's.** `SIDE_WAYPOINT_GAP` and `beside_box` both
first appear in `b3c2e66` (2026-07-03, HaccerKat). Hamid layered his fixes ON TOP of it, which is
precisely why the two cannot be separated by disabling waypoints — a belief the user held the other
way round until this was checked.

## THE VERIFICATION: 0/16 → 8/15 on the SAME scene, with NO waypoints (2026-07-30)

`results/phase4_approach_mode/20260730-160723/curated/direct/` — the rebuilt original scene
(`occluder_asset=milk_box`, `occ=[1]`, `off=[0.2]`, `angle0=[0.0]`), run in `direct`, i.e. the
around-box waypoint OFF and the scripted placement chain OFF.

| | success |
|---|---|
| old baseline, Hamid, ≤2026-07-07, waypoints and everything else present | **0/16 = 0%** |
| `direct`, 2026-07-30, **no waypoint generation at all** | **8/15 = 53.3%** |

Wilson 95% CI [30.1%, 75.2%]. Fisher exact two-sided **p = 0.0008**. Remaining failures:
`grasp` 4, `placement:pre_place_descent` 2, `place_actor` 1.

**This is the strongest result in the project.** The scene that defeated the old expert entirely is
now cleared by a solver with *less* scene-specific machinery than that expert had. It is direct
evidence that the July fix stack — not the waypoint heuristics — is what produces the performance.
53.3% also sits at the TOP of the olive-oil range (27–53% across 2–5 occluders), consistent with one
occluder being easier than four despite the carton being ~2x the bottle's footprint.

**User's own reading (2026-07-30):** "what I didn't realize before was how good the baseline was
even without waypoints." Seeding is consequently demoted to **experimental** — it is no longer the
headline, and there is no need to force a positive seed result.

**Limits, state them with the number:** cannot attribute across individual July fixes (the whole
stack moved together); the reconstruction is faithful in obstacle, distance and placement but the
target-spawn ranges, stability gates and clutter handling have all changed since July, so this is
not a controlled A/B against the old code; n=15 vs n=16; and the comparison baseline is Hamid's
commit-message figure, not a re-run.

## What survives in `direct` (all live, none gated)

`APPROACH_MODE=direct` removes ONLY the around-box side waypoint. `PLACEMENT_MODE=direct` removes
ONLY the scripted backward subgoal chain. Proof that the gate is that narrow: **`_placement_mode()`
appears exactly ONCE in the whole placement path** — `placement_mixin.py`, inside
`_backward_subgoal_poses`. Everything below runs in every cell:

| fix | commit | now in |
|---|---|---|
| cached trajectory replay | 2df658b | `grasp_mixin`, `planning_mixin` |
| local-waypoint shrinking retry | 385d969 | `planning_mixin` |
| fallback ladders, `PLACEMENT_SEARCH_RETRIES` | 385d969 | `planning_tuning`, `placement_mixin` |
| `relax_orientation` | 385d969 | `planning_mixin`, `placement_mixin` |
| deterministic descent slicing | 5bbe2d2 | `placement_mixin`, `state_checks_mixin` |
| local landing search | 5bbe2d2 | `placement_mixin`, `occluder_task` |
| grasp-retry reset | a84a6cf | `occluder_task` |
| attached-object slowdown | a84a6cf | `planning_mixin` |
| contact-release tolerance | a84a6cf | `placement_mixin` |

## Scene-agnostic strip: COMPLETE behaviourally (audited 2026-07-30)

No scene-specific value influences any decision on the direct path. Three tiers:

1. **Unreachable** — `_around_box_waypoint` (`grasp_waypoint=None`); `_backward_subgoal_poses` body
   (early `return []`); **all of `_select_attached_placement_plan`**, skipped by the `if/else` at
   `occluder_task.py` (~line 358) — that kills every occluder-derived corridor and hand-tuned ladder;
   `_rank_side_grasp_ids`'s `horiz>0.35` filter and limit-4 (deleted 2026-07-30).
2. **Evaluated then discarded** — `_box_side_x` is still CALLED at three sites in `grasp_mixin.py`
   and reads occluder 0's pose + `OCC_HALF_FOOTPRINT`. All three feed `x_side=` into
   `_backward_subgoal_poses`, which returns `[]` before using it; Python evaluates the argument
   anyway. Zero behavioural effect, but a reviewer greps `occluder` and sees it.
3. **No-op branches** — `grasp_mixin.py` still branches on `occluder_present` for `orients` /
   `y_offsets`, but `WAYPOINT_ORIENTATIONS=("grasp_aligned",)` and `WAYPOINT_Y_OFFSETS=(0.0,)` make
   both arms identical.

Not scene-specific in the relevant sense: `PAD_XY`/`fixed_pad_xy` (task definition — where the goal
is, used in `load_actors`), `GRASP_LIFT_HEIGHT` (bare constant).

**Optional ~10-line cleanup, NOT done, user's call:** let `_backward_subgoal_poses` compute
`x_side` itself and drop the two dead ternaries, so the code proves scene-agnosticism instead of
requiring the tier-2/3 analysis above to defend it.

## Measured generality of `direct`

| scene | occluders | success | n |
|---|---|---|---|
| curated | 2 | 34.5% | 29 |
| curated | 3 | 47.2% | 36 |
| curated | 4 | 27.1% | 48 |
| curated | 5 | 53.3% | 15 |
| standard | 0 | **22.2%** | 54 |

Runs 0–5 occluders without special-casing, never collapses. Against dev's vanilla at 0/16 that is
the difference between working and not working. **Caveats to state if this is ever claimed:**
standard (no occluder) is the WORST cell — likely clutter density 8 making it a different hard, but
unexplained; and the whole table is ONE task family (this bottle, this pad, this ring), so
"generalized" is demonstrated across scene *configuration* only, never across tasks or objects.

## The waypoint question is STILL UNMEASURED

**Zero `off`-mode records existed as of 2026-07-30** (182 `direct`, 81 `seed`, 0 `off`). Superseded
2026-07-31: `results/phase4_approach_mode/20260731-143654/` has an `off` cell with
`placement_mode="scripted"` (n=2, 1 success) — so `off:scripted` DOES run. Still far too small to
compare. The only
comparison available is across code versions, matched on scene config (4 occluders, radius 0.2):

- waypoints ON (pre-flag runs, Jul 17–27): **6/9 = 67%**, zero grasp failures
- waypoints OFF (`direct`, Jul 30): **11/25 = 44%**, grasp is the largest failure bucket

Fisher exact two-sided **p = 0.438** — point estimate favours the waypoints, well inside noise,
confounded by code version (pre-refactor, pre-grasp-ranking-change) and by three of the nine being
single-episode debug runs. **"The waypoints did not do much" is UNSUPPORTED; the weak evidence
leans the other way.** Settle it with `MODES="off direct seed"` — the `off` cell runs at direct's
speed (~70 s/rollout), roughly 18 extra minutes. `off → direct` is the generality claim;
`direct → seed` is the seed experiment.

Caveat on `off` as a control: the around-box waypoint is a one-occluder-in-front heuristic, so on a
rotated multi-occluder ring it is arguably solving a different problem. Weak scientific control,
but it is exactly the historical configuration, so it does answer "what did we have before".

## PATH QUALITY is a second criterion, and the ablation was never scored on it (2026-07-31)

User-raised: in `direct` on the milk-box scene the expert **grasps the bottle and spins it over**
during the carry. The expert's product is DEMONSTRATIONS (see [[tool_vla_pi05_port]]), so a clean
path is an objective in its own right — success rate is not the only axis. Everything in
`domain_expert_baseline` above, and the `p=0.438` waypoint comparison, scores ONLY success.

**The stripped chain was, in large part, orientation staging** — never described that way because
it was only ever measured on success:
- scripted carry = 6–7 SHORT hops, each planned from the live qpos, all at one `quat` and all with
  `relax_orientation=True` EXCEPT `PLACEMENT_STRICT_ORIENTATION_STAGES=("placement:center_over_pad",)`.
  Short position-only hops give trajopt no reason to rotate; the rotation is confined to the final
  `center_over_pad → place_pre_pose` blend.
- `direct` carry = **ONE** `carry_transit` plan, post-lift → `place_pre_pose`, FULL-pose goal, big
  translation and the whole orientation change in one trajopt problem. Nothing prefers rotate-late,
  so the rotation smears over the entire carry. That is the spin.
- `place_actor(constrain="free")` still applies `z_align_matrix` (`transforms.get_place_pose`) — it
  demands the bottle be stood UPRIGHT. Grasp it tilted and the goal requires a large wrist rotation.
- `_select_attached_placement_plan` also searched `quat_options=(grasp_quat, top_down)`, i.e. it
  CHOSE a carry wrist orientation. Skipped entirely in `direct` (`occluder_task.py` ~line 357).
- Suspect worth checking first: `_rank_side_grasp_ids`'s `horiz>0.35` hard filter was DELETED
  2026-07-30 (scene-agnosticism) and now only breaks ties, so tilted grasps are admissible where
  they previously were not. Same date the spin-showing runs start. Not confirmed causal.
- The single big jump is partly an ARTIFACT OF THE SEED EXPERIMENT: `occluder_task.py`'s own comment
  records that a two-hop split was rejected because "the seed's whole claim is that it makes the big
  jump tractable, so the big jump is what has to be attempted".

**No path-quality metric exists anywhere in the pipeline** — `records.jsonl` has success,
fail_reason and plan effort only. Any claim here needs one built first (integrated EE rotation ÷
endpoint geodesic; path length ÷ straight-line; held-object tilt over time). Until then
"waypoints prevent the spin" is exactly as unmeasured as "waypoints don't help" — and the `off`
cell that would settle both still has ZERO records.

## Strategic reframing the user landed on (2026-07-30)

Possible headline contribution is **"a scene-agnostic expert matching a hand-tuned one"**, not
"the clearance seed helps". **Strongly reinforced by the 0/16 → 8/15 milk-box verification above,
and the user has since demoted seeding to "experimental".** The expert story is a large already-measured effect against a documented
0%; the seed story is a null on a leg with no headroom, diluted by a 25–45% firing rate. Under the
reframe the seed becomes an honest negative result inside a positive paper. **This depends on the
`off` cell existing.** Also: `585d307` (2026-07-09) once added a true vanilla `--plan-algo` baseline
(dev's grasp→lift→place); it is GONE from the tree, recoverable from that commit if a three-cell
vanilla → direct → seed design is wanted.
