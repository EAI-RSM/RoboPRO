---
name: tool_clearance_metric
description: "clearance_metric_3d.py (2.5D): what eps* means, the design decisions behind it, and the known inaccuracies"
metadata:
  type: project
---

A scene-difficulty metric — "how boxed-in is the target." `clearance_metric_3d.py` in
`customized_robotwin/script/bench_script/` is the live 2.5D pipeline (added `592af56` alongside
`customized_robotwin/script/bench_script/plans/CLEARANCE_2D_TO_3D.md`, the 2D→2.5D explainer).

**Post-refactor (2026-07-29) most of the compute lives in `lib/`, not in this file** — see
[[domain_bench_script_layout]]. The 2D predecessor `clearance_metric.py` was **DELETED** (10 of its
14 shared functions were byte-identical to the 3D file and nothing imported it), which also cleared
the dead 2D scaffolding this note used to flag. Diagnostics and figures split into
`lib/metric_diagnostics.py` / `lib/metric_viz.py`; all knobs now come from
`lib/metric_config.py::SeedMetricConfig`.

**What eps\* is.** Widest-path (Kruskal max-min) bottleneck clearance between the grasp cell and
the PAD cell. Pad-as-second-endpoint was the user's call: it converges to the boxed-in-target
clearance BECAUSE the dataset has no "mazy middle" (open space target→pad). Gripper half-width `r`
is compared to eps* at READ time only (`--gripper-r`), never baked in. Difficulty is normalized as
**rho = eps*/gripper_r**, bucketed at rho ≥ 3 / 2 / 1 (easy/medium/hard/very-hard) — rho=1 is
sphere-feasibility, the only non-arbitrary edge; rho=3 is the gripper CARRYING the bottle.

**Non-derivable design decisions (the WHY):**
- **Clearance is measured to OCCLUDERS ONLY, not the IK-OBSTACLE set.** IK-OBSTACLE lumps
  occluders + table + furniture + target, so its EDT would measure distance to the table. Routing
  nodes stay IK-`FREE` (body-aware); only the clearance VALUE is geometric → embodiment-free, and
  it never measures to the reach boundary.
- **2.5D = z-STACK + 3D EDT/DSU** (the route may CLIMB OVER an occluder), which makes the
  joint-space edge gate MANDATORY. It is not "gate vs height" — the gate is a component the height
  version forces on. In pure 2D the gate could stay off *because* of the no-mazy-middle assumption.
- **Framing caveat the user settled:** a 3D widest path can route through a vertical detour the
  committed grasp never uses, shifting eps* from "difficulty of THIS grasp" toward "difficulty of
  the EASIEST route". A scene tight in-plane but open overhead reads EASY. `eps* = inf` means a
  route was found over the bottle top (unbounded clearance) = the ideal climb-over.
- **Manipulability / reach-edge channel CUT** — curobo does not expose a Jacobian
  (`cuda_robot_model.py`: `log_error("Outputting jacobian is not supported")`). The BEYOND-REACH
  label is still kept in the npz, so a distance-to-reach-edge channel stays a 1-liner later.

**The warm-start finding (Phase 0, user-run on GPU).** The arm is 6-DOF → the IK solution set is
DISCRETE. curobo's raw single best-cost solution BRANCH-HOPS between adjacent cells at higher z
(z=0.9/1.0 shows scattered speckle in the roughness map) while staying smooth at grasp height.
**That speckle is bookkeeping NOISE, not genuine config-space seams** — cost-tie-breaking among
~100 internal seeds returns different branches cell-to-cell. Proven by warm-start: a multi-branch
solve (`return_seeds=K`) plus BFS continuity propagation eliminated it, so a consistent smooth
branch exists. Real seams survive (propagation only picks among returned candidates), so this is
not over-smoothing — and the WARM maps show coherent yellow STREAKS (codim-1 curves), the intended
outcome. **Design consequence: the gate MUST read WARM-STARTED q, never raw best-cost q**, or it
spuriously cuts good vertical edges. 3D (26-conn) propagation collapses the vertical-jump
histogram to ~0 vs per-slice 2D.

**Pipeline (3D file):** `build_grid` (x/y/z) → `label_volume` (two IK passes → FREE / OBSTACLE /
BEYOND-REACH, `prune_mask` from [[tool_reach_envelope]]) → `warm_start_branches_3d` (26-conn BFS
continuity over the FREE volume) → `occluder_footprints_3d` + `occluder_mask_3d` → anisotropic 3D
`distance_transform_edt`, UNBOUNDED above the bottle top (the pass-over region) →
`widest_path_eps_3d` (26-conn Kruskal max-min with the joint gate on `q_warm_3d`) →
`phase4_metric` writes `metric_*.json` + route. eps* is computed BOTH ungated (reach+clear) and
GATED; "ungated merges but gated disconnects" = branch seams block the continuous climb-over.
Runs save `stack_data.npz` (label, qfield, edt, q_warm_2d/3d, xs/ys/zs) so results are
re-analysable offline.

**Envelope-only relaxation.** `lib/geometric_metric.py` is additive and does not change this
pipeline. It replaces the per-scene IK node set with the precomputed reach envelope, drops the
joint gate, and stamps the target mesh into labels only. It is called geometric eps*
(`eps_geom`), never plain eps*, and remains gated on the empirical rank test described in
[[tool_geometric_metric]].

**curobo IK API used** (verified in-repo `curobo/wrap/reacher/ik_solver.py`):
`IKSolver.solve_batch(goal, seed_config=(n,batch,dof), return_seeds=K, num_seeds=None)`. Default
`num_seeds`=100; extra `seed_config` seeds are topped up with random ones, so injecting a
continuity seed costs no reachability. `result.solution` is (batch, K, dof); dof = 6, no gripper.

**Key flags.** Since the refactor the argparse defaults are `None` and the real values come from
`SeedMetricConfig`, overlaid by `from_args` and then `SEED_<FIELD>` env vars — so every flag below
also has an env twin. `--zmin/--zmax/--zres` (grid; **zmax is now 1.23 everywhere**, unified in
`abb917a` — it was 1.4 in this tool's argparse while the seed builder used 1.23, which meant the
standalone tool and the in-rollout builder were measuring different volumes), `--gate-tau` (0.35,
from the phase-0/2 histogram gap; active only with `--warm-start`), `--ik-seeds` (30, ~3× faster
than 100; use ~60 for a quality pass), `--free-only` (skips the collision-OFF sweep ≈ half of `label_volume`;
OFF is viz-only so the metric is unaffected — every run reports `projected_free_only_seconds`),
`--occ-shape {mesh,extruded}` (**default mesh**: re-cuts the posed collision mesh at every z via
`trimesh.section(...).discrete`; `extruded` reproduces pre-2026-07-24 numbers — **this changes
eps\***, see [[domain_scene]] on the taper), `--num-occluders` / `--offset` (ring radius) /
`--occluder-angle0`, `--reach-envelope` / `--reach-mode {occupancy,sphere}`.

**Visuals** (all four requested by the user, from `phase4_metric`): `_viz_side_elevation` (profile
through grasp→pad; the clearest "does it climb over" view), `_viz_clearance_profile` (1D clearance
vs arc-length, min = eps*), `_viz_topdown` (route coloured by height), `_viz_ceiling` (max FREE z
per x,y), plus a wireframe **eps\* sphere** at the bottleneck and `_equal_aspect_3d` (without a
1:1:1 box aspect the sphere renders as an ellipsoid and "just touches" is unreadable). They collapse
3D into 2D/1D deliberately — see [[user_math_background]]. Scene anchors on every spatial figure:
target spawn (blue star) vs grasp seed (that pose snapped to a FREE voxel), and current gripper
pose (orange triangle).

**Known inaccuracies / open items:**
- **The EDT is voxel-centre-to-voxel-centre**, so eps* sits up to half a voxel (5 mm in x,y;
  15 mm in z) ABOVE the true point-to-surface distance — optimistic by construction, worst in z.
  Exact fix = continuous `trimesh.proximity.closest_point` per FREE voxel (no grid bias, but
  O(100k) mesh queries/run). Mitigation already in place: the metric json reports the EXACT
  mesh-surface distance at the bottleneck, so the bias is visible per run.
- `--gate-tau 0.35` is tentative; calibrate from the WARM histogram gap on a real run.
- `--ik-seeds 30` needs a FREE-count check vs 100 before fast runs are trusted.
- **Held-object collision is still only a documented hook here** (it IS built on the seeding side —
  see [[tool_seed_from_clearance]]).
- `reachability_view.py`'s `OCC_HEIGHT = 0.2542` is the MILK-BOX extent; the occluder is olive-oil
  id 3 at **0.30542**, so the wireframe in `swept_volume_3d.py` draws the obstacle ~17% too short
  and two labels still read "milk box". Display-only — no metric reads it — but unfixed.
  (By contrast `OCC_HALF_FOOTPRINT = 0.04` IS correct for olive-oil, 0.0795 × 0.0796 → half 0.0398.
  Only the "round, yaw-invariant" half of that comment is wrong — see [[domain_scene]].)
