---
name: tool_task_metric_validity
description: "MEASURED validity problems in the put_cup_on_coaster task-metric chain: endpoint-pinning, the 12 cm wrist offset, and grasp-candidate arbitrariness — plus the offline rebuild trick that proves them without a GPU"
metadata:
  type: project
---

Findings from a 2026-08-07 audit of the `lib/waypoints.py` canonical chain feeding
[[tool_geometric_metric]]'s `geometric_eps` in the `put_cup_on_coaster` association study.
**All four numbers below are measurements on real recorded runs, not deductions.** Provenance:
`scripts/validation/results/task_metric/20260731-154726` (d10, seeds 1000–1099) and
`20260731-163538` (d15), 100 scenes each. The same metric code produced the 3000-rollout campaign's
metrics, so these apply there too.

**The chain is deliberately planner-free, and that part is right.** `canonical_waypoints` does NOT
call `choose_best_pose`; it hard-codes contact point 0 with the fixed contact→gripper remap. A
planner-selected grasp would make the predictor endogenous to the difficulty it measures. Do not
"fix" this by reintroducing the planner. The constants are lifted from `Base_Task.get_grasp_pose`
(remap matrix, 0.12 m tool offset) and `put_cup_on_coaster.play_once` (`pre_grasp_dist=0.1`,
`pre_dis=0.07`, `dis=0.005`). Nothing about the VLA enters — rollouts save only final state, **no EE
traces**, so what the policy actually does is unmeasured and currently unmeasurable from the runs.

**1. eps is endpoint-pinned — 800/800 legs, both pilots, zero mid-path bottlenecks.** The
widest-path machinery is contributing nothing in this scene family: `eps_geom` reduces to
`min(clearance at start, clearance at goal)`. Cause is structural — the EDT is unbounded above the
tallest obstacle, so a route can always climb over, leaving only the endpoints (which sit where the
arm must actually be) able to bind. Any reasoning that treats eps* as a corridor width is wrong here.

**2. The bucket label means a different physical thing at each end of the scale.** Cross-tab of what
set `eps_geom_min` (d10 / d15):

| bucket | n | binding endpoint |
|---|---|---|
| very_low | 17 / 27 | **HOME 17/17 and 27/27** |
| low | 35 / 41 | HOME 24, place 8, grasp 3 |
| moderate | 30 / 26 | grasp 13, HOME 13, place 4 |
| high_clearance | 18 / 6 | **grasp 14/18 and 6/6** |

`very_low` means "clutter is parked near the arm's rest pose"; `high_clearance` means "the airspace
above the cup is empty". The home EE pose is fixed to 1–3 mm across all 100 scenes, so in 56% (d10)
/ 65% (d15) of scenes the primary predictor involves neither the cup, the grasp, nor the path.

**3. The 12 cm tool offset is the biggest single defect.** `grasp` is the WRIST waypoint, not the
fingers: contact point + 0.12 m back along the approach axis lands at z≈0.920 — 12 cm above the cup
rim and **18 cm above a table carrying 6–8 cm clutter**. It floats in open air above the obstacles.
Exact continuous mesh distance (no voxel bias) at the rim vs at the wrist: wrist reads higher in
**92/100** scenes, median inflation **+0.023 m (18%)**. Consequence:

> **spearman(clearance where the fingers close, `eps_geom_min`) = 0.078**

The metric is essentially uncorrelated with how boxed-in the cup actually is. Worked example:
seed 1024 has the 3rd-tightest grasp of 100 (0.0728 m) and is bucketed `high_clearance`; seed 1015
is looser (0.0782 m) and is bucketed `very_low`. **This is the mechanism behind the user's
observation that the `videos_by_clearance` buckets look wrong.** It is present at a fixed approach
direction — it is not the approach-direction problem.

**4. The grasp candidate is arbitrary, and the arbitrariness is the size of the whole signal.** The
expert's own `choose_grasp_pose` + `Robot.create_target_pose_list` enumerate ~40 poses: 4 annotated
cup contact points × 10 rotations θ∈[0,1] rad (`rotate_lim: [0,1]` in the aloha-agilex config,
`ROTATE_NUM=10`), then pick by planner feasibility. The metric always takes (contact 0, θ=0).
Sweeping 12 legal members over the 100 d10 scenes: **72/100 scenes change bucket**; Spearman vs
nominal falls to **0.341** at θ=1.0 (37% bucket agreement); median within-scene spread is
**0.048 m = 45% of the nominal value**, against a whole-pilot IQR of 0.055 m. Even contact 0 → 2
alone (same vertical axis, two rim points 5.5 cm apart) moves **25%** of scenes. Median eps falls
monotonically 0.1187 → 0.0758 as θ grows, so **θ=0 is the most optimistic member of the family**.
Nuance: median |diff| is 0.000 for small perturbations because home-pinned scenes are untouched —
the disagreement is concentrated in the ~44% where a grasp-dependent waypoint binds.

**All four cup contact points share one approach axis, and it is exactly vertical.** Measured
`pre_grasp − grasp = (0, 0, 0.100)` with std 7e-5 across all 200 scenes. The cup only ever gets a
top-down grasp, so the candidate family varies *tilt and which rim point*, never a fundamentally
different approach side. Do not assume the metric could have picked a horizontal approach.

**The offline rebuild trick — use this before ever booking GPU time.** `records.jsonl`'s
`scene_fingerprint_source.scene.metric_obstacles` carries every obstacle's collision-mesh path,
world pose, and scale, and `target`/`destination`/`destination_pose` carry the rest. That is
sufficient to rebuild the exact `geometric_eps` input **on CPU with no SAPIEN and no scene build**:
stub actors exposing `get_pose()`/`scale`, assemble `collision_list`, reuse the real
`SeedMetricConfig` from `config.json` and the recorded `reach_cache_dir`. Verified to reproduce all
100 recorded `eps_geom_min` values with **max abs diff 0.000e+00**. Cost ~40 s/scene, single core.
Any counterfactual about waypoints, buckets, or obstacle sets is answerable this way.

**Not established, do not claim.** Which grasp candidate the expert *realizes* in practice (only the
legal family was swept; `choose_grasp_pose` prefers top-down when `dis_top_down < 0.15`, so small θ
is probably commoner). What the VLA does at all. Whether any of this changes the sign of the
association — the audit is about construct validity and ranking, not about feasibility, and **no
pilot scene has fingertip clearance below the 0.03 m gripper half-width**, so nothing here says the
grasps were impossible.

See [[feedback_scientific_rigor]] on verifying the treatment actually fired: the same failure shape
appears here — an elaborate widest-path pipeline whose output, when measured, was set entirely by
endpoint clearance at a point floating above the obstacles.
