---
name: tool_reach_envelope
description: "reach_envelope.py producer / consumer / validator split for IK pruning — the strict-bound argument, the two masks, and the aloha-agilex calibration"
metadata:
  type: project
---

Precomputed per-arm reach envelope used to prune IK solves in [[tool_clearance_metric]].
Committed `290873b`. **Three-file split the user INSISTED on** ("precompute once, never again;
clearance_metric are ACTUAL RUNS") — do NOT re-tangle compute/viz back into the runs file:

- **`reach_envelope.py` = PRODUCER**, run ONCE per (robot, grid). Writes
  `results/clearance_metric_3d/_reach_cache/reach_envelope_<arm>.npz` + an eliminated-vs-kept
  image (top-down grey = eliminated for any pose, green = kept). Occluders OFF; the seed only
  builds the robot, since the envelope is scene-independent. Caches R_max in `reach_radius_<arm>.json`.
- **`clearance_metric_3d.py` = CONSUMER.** `--reach-envelope` (opt-in) + `--reach-mode` +
  `--reach-cache-dir`. `load_reach_envelope()` returns the prune MASK; `label_volume(prune_mask=)`
  solves IK only on un-masked cells, masked → BEYOND. No FK/IK/viz for the envelope in this file.
- **`validate_reach_envelope.py` = GATE.** Loads the exact mask a run would apply, solves IK on
  ONLY the PRUNED cells (`prune_mask=~mask`), asserts ALL come back BEYOND. On FAIL,
  `diagnose_violations` prints EDT depth-into-pruned, a suggested `--occ-mc-safety`, and top-down
  + side x–z figures.

**Why pruning is sound (strict bound):** a grid cell is a gripper POSITION and orientation is a
separate IK input, so union-over-orientations FK ⊇ metric-reachable(grasp_q). Pruning that union's
complement can never drop a FREE cell. Both masks are pose-independent.

- **Sphere (Tier 1, grid-INDEPENDENT).** Ball centred at the arm's KINEMATIC ROOT (shoulder):
  `C = robot_origion_pose.p − R_base @ frame_bias`, radius = R_max + gripper_offset + margin.
  **frame_bias SHIFTS THE CENTRE, never the radius** — the original 0%-prune bug centred at the
  floor origin and ADDED ‖frame_bias‖=0.87 to the radius, making the ball 2× too big. Tell for a
  wrong centre: a *reachable* grasp point printing farther than R_max. Inherently loose
  (circumscribing ball) — pruned only 14%.
- **Occupancy (Tier 2, grid-DEPENDENT, default and preferred).** `build_occupancy` samples FEASIBLE
  self-collision-free configs via `ik.sample_configs` on a NO-WORLD solver, FK → world endlink
  `X = C + M @ p_ee` (M = R_base @ R_rot^T; R_rot = per-arm yaw −0.02 L / −0.01 R, the exact
  inverse of `_world_gripper_to_curobo`), voxelises onto the metric grid, dilates by
  gripper_offset + mc_safety via anisotropic `distance_transform_edt`, prune = ~reachable.
  Ceiling = the BEYOND fraction from a full sweep, ≈67% for right/seed1; realised prune lands below
  that (orientation-agnostic + dilated).

**Calibration (aloha-agilex; empirical, not derivable):** uniform Halton joint sampling
under-covers the fully-extended reach edge (thin config pre-image) while the metric's IK optimises
to it — a **boundary rind**. Measured worst gap on right/seed1 = **0.075 m**, so total dilation
D ≥ 0.14 + 0.075 → **`--occ-mc-safety` default bumped 0.02 → 0.11** (D = 0.12 + 0.11 = 0.23 m).
Only the SUM gripper_offset + mc_safety matters; the split is cosmetic (the gripper↔endlink
position offset is actually zero — see [[domain_curobo]]). **RIGHT ARM: PASS, 36.6% of IK skipped,
0 reachable cells pruned.** Re-validate after any embodiment or grid change. If prune is ever too
low, raise `--occ-samples` (1.5M → 6–10M) to tighten the rind rather than shrinking dilation.

**Workflow gotchas:**
- After changing ANY `--occ-*` / `--reach-*` / grid arg you MUST re-run the PRODUCER; the validator
  and consumer only READ the .npz. The user hit this — a stale mask re-reported an identical 1846
  violations; the tell is the diagnostic printing the OLD "mc 0.020". Order:
  `reach_envelope.py --arms <arm>` → `validate_reach_envelope.py --seed 1 --arm <arm>` →
  `clearance_metric_3d.py --reach-envelope`.
- Occupancy is grid-specific, so all three scripts must agree on the grid. Since `abb917a` they do
  so by construction: the envelope tools take their defaults from `lib/metric_config.py::
  SeedMetricConfig` (res 0.01, zres 0.03, x[−0.6,0.6] y[−0.35,0.35] z[0.78,**1.23**]) rather than
  each declaring their own. A mismatch still errors with a copy-paste regen command; the fallback is
  `--reach-mode sphere` (grid-independent).
- OOM: `ik.sample_configs` defaults to `rejection_ratio=50`, so a big batch collision-checks
  batch × 50 configs at once (14 GiB at batch 200k). Cap with `--occ-batch 32000` ×
  `--occ-rejection 8` ≈ 0.36 GiB. Lower `--occ-batch` first.
- Warm-start is NOT needed for the envelope — that is only for the gated 2.5D metric.
