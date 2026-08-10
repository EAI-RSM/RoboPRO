# Plan 1 of 2 — envelope-only geometric eps\*, and whether it agrees with gated eps\*

Executable plan for Codex. Written 2026-07-30 against branch `codex/bench-script-refactor`,
HEAD `21a64d7`.

**Follow-on:** `TASK_METRIC_CORRELATION_PLAN.md` (port to stock RoboPRO tasks + the pi05 correlation
study). That plan is **gated on stage 3 of this one**. Do not start it until this plan reports.

---

## Goal

Build an **envelope-only geometric eps\*** — the existing widest-path clearance metric with the
per-scene IK volume sweep and the joint-continuity gate removed, reachability supplied instead by
the precomputed reach envelope — and then **measure whether it ranks scenes the same way the gated
metric does**.

Rank agreement is the whole question. If it holds, the IK sweep comes out: the metric becomes
CPU-only, drops from ~150 s to seconds per scene, stops competing with a VLA for the 16 GB card,
and porting it across tasks becomes affordable. If it does not hold, the follow-on plan shrinks
sharply and the cost argument has to be re-made.

**Read first:** `agent-memory/tool_clearance_metric.md`, `agent-memory/tool_reach_envelope.md`,
`agent-memory/domain_bench_script_layout.md`, `agent-memory/status_current.md`.

---

## 0. Ground rules

**Do not move any already-collected number.** The gated path (`compute_route_configs`,
`label_volume`, `_build_warm_field`, `widest_path_eps_3d` with `qvol`) must remain byte-identical in
behaviour — the Phase 4 A/B results depend on it. The geometric path is **additive**: a new module
plus new callers, never an edit to the existing one.

**Seeding is out of scope.** The seed builder needs per-voxel joint configs and the geometric path
does not produce them. Do not attempt a route-only IK pass; do not touch
`seed_from_clearance.py::build_seed` / `resample_route_to_seed` / `route_qs_to_seed_tensor`.

**Do not tighten or rebuild the reach envelope.** The current union-over-orientations artifact
ships, is validated on the right arm, and costs nothing to use. Making it orientation-conditioned
would fix the largest looseness term, but it is a producer refactor for a problem stage 1 has not
yet shown exists. Explicitly deferred — if stage 1 or 3 comes back bad, **report and stop**, do not
start refactoring `reach_envelope.py`.

**Dependency rule holds.** `lib/` imports nothing from a CLI script. Shared code goes in `lib/`.

**Script conventions apply unasked** (`agent-memory/feedback_script_conventions.md`): timestamped
run folder `<out-dir>/<YYYYmmdd-HHMMSS>/`, `timings.json`, results under
`scripts/validation/results/<topic>/`, few panels, large fonts, on-figure annotations.

**Environment.** Run from `customized_robotwin/` with `source set_env.sh` and
`export ROBOTWIN_BENCH_TASK=bench`. Single GPU (RTX 4080, 16 GB); `GPU_SPEC` must be `0:0`.

**GPU steps are user-owned.** Codex writes the code, the CPU tests, and the exact command line; the
user runs anything needing SAPIEN/curobo and reports back. Marked below.

**One stage per commit.** Stage 3 is the gate — do not build past it if it fails.

---

## 1. Stage 1 (GPU, user-run) — measure the envelope's false-keep rate

Everything here rests on the reach envelope being a *usable* reachability answer, not merely a safe
pre-filter. The current numbers:

- Grid is 121 x 71 x 16 = **137,456 voxels** (`SeedMetricConfig`: res 0.01, x[-0.6,0.6],
  y[-0.35,0.35], z[0.78,1.23] at zres 0.03).
- The occupancy envelope prunes **36.6%** (right arm; validated, 0 reachable cells falsely pruned).
- A full IK sweep at a fixed `grasp_q` labels **≈67%** `BEYOND`.

That ~30-point gap is **~42,000 voxels** the geometric metric will treat as reachable and the real
arm cannot use at the working orientation. It comes from two sources, both scene-independent: the
envelope is a union over *all* orientations, and it is dilated by
`gripper_offset 0.12 + mc_safety 0.11 = 0.23 m`. Against eps\* values around 0.085 m, the
reachability boundary is fuzzy at roughly **3x the scale of the quantity being measured**.

**The measurement is an inversion of a validator that already exists.**
`validate_reach_envelope.py:176` calls `label_volume(..., prune_mask=~prune_mask)` — it solves IK on
**only the pruned** cells and asserts every one comes back `BEYOND`. That is the false-**prune**
rate, currently 0 on the right arm, and it is what makes pruning sound. The complement is the same
call with the mask not inverted:

- solve IK on the **kept** cells: `prune_mask=prune_mask`
- count how many come back `BEYOND` → the **false-keep** rate

Add `--mode {false-prune,false-keep}` to `validate_reach_envelope.py`, defaulting to `false-prune`
so existing behaviour and its recorded numbers are untouched.

Write into the run folder, as `false_keep.json` plus one figure:

- false-keep count and fraction, both of kept cells and of the whole grid
- **spatial distribution** — reuse `diagnose_violations`' top-down and side x–z views. This is the
  number that actually matters: false-keeps out at the reach edge are harmless if routes never go
  there; false-keeps in the working volume over the table are not.
- a z-profile: false-keep fraction per slice

Run the **right** arm first (the validated one), then left. The left-arm envelope has never been
GPU-verified at all (`status_current.md`), so its false-*prune* rate must be confirmed as 0 before
its false-keep number means anything.

**Pass condition — deliberately soft.** No threshold makes this pass or fail on its own; the number
is an input to stage 3, which is the real gate. What *would* kill the plan here is a
false-**prune** rate above 0 on either arm, because then the envelope is not a valid outer bound at
all and every eps\* built on it is unsound in an unknown direction.

Cost: one full IK sweep per arm, once, ever. Not per scene.

---

## 2. Stage 2 (CPU) — `lib/geometric_metric.py`

New module. Mirrors `seed_from_clearance.py::compute_route_configs` with the IK stages removed.
Everything it needs already exists in `lib/`.

```
geometric_eps(env, arm, legs, cfg, reach_cache_dir, reach_mode="occupancy")
    -> list[LegResult]
```

`legs` is an ordered list of `(start_xyz, goal_xyz)` world gripper positions. **One volume is built
per call and reused across every leg** — the label and clearance fields do not depend on the
endpoints, only the widest-path query does. This is what makes a multi-leg metric affordable, and
it is the piece the gated path never had.

Pipeline:

1. `build_grid(cfg)` — existing, unchanged.
2. `load_reach_envelope(cache_dir, arm, xs, ys, zs, XX, YY, mode)` → `prune_mask`. Existing. This is
   the **only** reachability input; no IK solver is constructed anywhere in this module.
3. `foots = occluder_footprints_3d(env, obstacles=cfg.obstacles)` with `cfg.obstacles="all"` —
   existing, and it already picks up procedural table clutter, because `clutter_surface_split`
   registers every clutter object into `env.collision_list` with its real collision mesh path.
4. `occ_mask = occluder_mask_3d(foots, XX, YY, zs, shape=cfg.occ_shape)` — existing.
5. `edt = occluder_clearance_3d(...)` — existing, unchanged. **The clearance values are identical to
   the gated metric's.** Only the node set changes.
6. `label = np.where(prune_mask | occ_mask | target_mask, BEYOND, FREE)`.
7. Per leg: `nearest_free_voxel` on each endpoint, then
   `widest_path_eps_3d(label, edt, None, seed_a, seed_b, cfg.gate_tau)`. Passing `qvol=None` is
   already a supported path — it is the ungated call the gated metric makes on every run — so
   **no change to `lib/widest_path.py`**. Then `reconstruct_widest_path_3d(..., qvol=None, ...)`
   for the route.

### `target_mask` — the one genuinely new piece, and it is not optional

`scene_obstacle_entries` deliberately excludes `target_obj` and `des_obj` from the obstacle set, so
they contribute nothing to the EDT. That is correct: eps\* measures clearance to *obstacles*. But
with IK on, curobo's world **does** contain the target, so cells inside it came back `OBSTACLE`.
Drop IK and the target becomes passable — a route can cut straight through the object being grasped,
right at the pinned grasp endpoint where the bottleneck usually lives.

Stamp the target's own posed collision mesh into the **label** (reuse `occluder_mask_3d` on a
one-entry footprint list), and **not** into the EDT. Add a focused test asserting a voxel at the
target centre is non-`FREE` while the EDT at that voxel is unchanged from the no-`target_mask` case.

### Known and accepted gaps — document in the module docstring, do not fix

- **Full-arm collision is gone.** The envelope says the endlink can be at a point; nothing says the
  forearm clears *this* scene's clutter. Irreducibly scene-dependent — the one thing in the IK sweep
  that cannot be precomputed.
- **Structural geometry is handled inconsistently — see §2b.** An earlier draft said "furniture and
  walls are passable, acceptable because we are scoped to flat-tabletop tasks." That is wrong in
  both directions and is corrected below.
- **Under-table routing is impossible by construction** — `cfg.zmin = 0.78` is above the ~0.74 table
  surface and `cfg.zmax = 1.23` caps the climb. Stated so nobody re-derives it.
- **`eps_geom >= eps_gated` is a hard invariant only for an aligned graph pair. See §3a.**
  Both sides must use the same grid/EDT, the exact same snapped voxel pair, matched mask policy,
  geometric FREE ⊇ gated FREE, and geometric edges ⊇ gated edges. The current native calls violate
  the endpoint and target-policy conditions. Stage 3 therefore constructs aligned pairs and asserts
  the theorem there; independently snapped native values are not used for this check.

`LegResult` carries `eps_star`, `merged`, `bottleneck_xyz`, `route_world`, `start_xyz`, `goal_xyz`,
`n_free`, `reason`. Reuse `save_route_visuals` for figures rather than writing new plotting.

CPU-only. No `torch`, no curobo import, no `_build_ik_solver`. A test must assert this — see §4.

---

## 2b. STRUCTURAL GEOMETRY (added 2026-07-30) — the arm can hit the bookshelf

Verified against the code; this corrects an earlier "furniture is passable" note that was wrong in
both directions. There are **three different situations**, not one gap.

**(i) The office wall-shelf is already IN the obstacle set — and should not be.**
`_office_base_task.py` appends `121_wall-shelf` to `collision_list` as a plain mesh with **no
`"link"` key**, and it is neither `target_obj` nor `des_obj`, so `scene_obstacle_entries(env, "all")`
picks it up. Its pose is `[furn_x, 0.28, table_height + 0.27 = 1.01]` with
`shelf_area = [0.62, 0.26]` → y ≈ [0.15, 0.41], z ≈ 1.01, which **overlaps the metric grid**
(y[-0.35, 0.35], z[0.78, 1.23]). Consequences: eps\* partly measures distance to a wall shelf rather
than to clutter, and `occluder_mask_3d` will `mesh.section()` a **34 MB** glb at all 16 z slices —
not free on the CPU path.

**(ii) The cabinet and the table are genuine blind spots.** The cabinet is registered with
`"link": "link_0"` etc., and `scene_obstacle_entries` drops `"link"` entries — but `update_world`
forwards them to curobo. The table is in `cuboid_collision_list`, which `update_world` also forwards
to curobo. So **gated sees cabinet + table; geometric sees neither.** This is a second, independent
contributor to the §3a subset violation, separate from `target_mask`. The table is saved only by
`zmin = 0.78 > 0.74`.

**(iii) `SPAWN_BACK_FURNITURE = True` is the office default, and `task/occluder_task.py:35` sets it
`False`.** **Every eps\* number collected to date was measured in a scene with no shelf and no
cabinet.** None of this has ever been exercised. Treat any claim about furniture handling as
unverified until a run with `SPAWN_BACK_FURNITURE=True` exists.

Study is cleaner: no back furniture; `014_bookcase` and `042_wooden_box` sit on the table, both have
collision meshes, and `add_collision` registers them into `collision_list` (`incl_collision` defaults
`True`), so they are correctly counted. The study wall is at y ∈ [0.4, 1.6] — outside the grid's
y-max of 0.35 by 5 cm. Study registers nothing in `cuboid_collision_list`, so its table is absent
from curobo's world too; both metrics are equally blind there, which is at least consistent.

### The fix: a label-only `structure_mask`, generalising `target_mask`

> **Structural geometry — table, walls, shelf, cabinet links — belongs in the LABEL (blocks routes)
> but NOT in the EDT (does not set clearance values).**

This is the same split §3a specifies for the target, and it unifies three patches into one rule. The
rationale is already in `lib/obstacles.py`: a tabletop in the obstacle set "would swamp the EDT and
drive eps\* to zero everywhere" — the identical argument applies to a 1.7 m wall shelf.

Implementation, in `_build_geometric_volume`:

```
label = np.where(prune_mask | occ_mask | target_mask | structure_mask, BEYOND, FREE)
edt   = occluder_clearance_3d(foots_dynamic, ...)   # structure NOT in foots_dynamic
```

`structure_mask` covers, in this order of value:
1. the `cuboid_collision_list` entries (the table) — rasterise the box directly, no mesh load
2. `"link"`-tagged `collision_list` entries, posed **per link** (the reason they were dropped is that
   the rigid whole-mesh transform misplaces them; per-link posing is the fix, and articulated
   furniture is static within an episode so it only needs posing once)
3. the wall-shelf — **moved out of the EDT into the label**

Item 3 changes eps\* on any office scene with back furniture. That is a correction, not a
regression, but it must be reported as a delta, exactly like `target_mask_delta`.

**Scope control.** Items 1 and 3 are small and should be done. **Item 2 (per-link posing) is
deferred** unless a tier-2 office task needs it — it is the only piece that is real work, and
plan 2's tier 1 is Study, which has no articulated furniture. Do not build it speculatively.

---

## 3. Stage 3 (THE GATE) — does the relaxation preserve rank?

The geometric metric is a **relaxation**, not the same construct at lower precision. It may make a
scene look easier, never harder, and **the inflation is scene-dependent**: a scene with a far-field
bypass inflates more than one whose bottleneck is pinned at the target. A per-scene one-sided bound
does **not** imply the ordering survives — and ordering is the only property the follow-on study
consumes. This has to be measured, not argued.

Write `script/bench_script/compare_geometric_vs_gated.py`. On ~10 occluder scenes:

- compute gated eps\* (existing path) and geometric eps\* (stage 2) on the **same seeds**, same
  arm, same `cfg`, and the **same snapped voxel pair** (§3a). Passing the same world-space
  grasp→pad coordinates is not enough: independently snapping them creates different graph
  problems.
- report **Spearman rank correlation**, per-scene inflation `eps_geom - eps_gated`, and a scatter
  with the identity line
- report both the no-target isolation series and the target-masked production series (§3a). The
  isolation series diagnoses the relaxation; the production series is the quantity plan 2 would
  actually consume. Passing only the isolation series is not enough.
- overlay the clearance-preferred geometric representative route (§3b) beside the existing gated
  route on `_viz_side_elevation` and `_viz_topdown` for every scene — the direct visual answer to
  "is the geometric route wandering out through unreachable space"
- flag any scene where gated says `INACCESSIBLE` but geometric merges
- cross-reference stage 1: do the geometric routes pass through voxels stage 1 marked false-keep?
  If they do not, the 42k-voxel gap is irrelevant in practice and that is the cleanest possible
  result. Report the fraction of route voxels that are false-keeps.

Reuse an existing gated artifact only if it contains the raw label, warm field, EDT, grid and
snapped endpoints needed to reconstruct the aligned pair. A scalar eps\* from
`results/phase4_approach_mode/` is insufficient and must not be mixed with a newly snapped
geometric value.

**Pass requires all three:**

1. The aligned relaxation invariant holds on every comparable scene:
   `eps_geom >= eps_gated` (not necessarily strict; equality is valid).
2. Spearman is >= 0.8 for both the no-target isolation series and the target-masked production
   series.
3. No visible far-field or floor-slice artifact remains in the clearance-preferred route overlays.

**On pass** — the relaxation is sound for a rank-based study. Record the coefficient; it is the
justification cited for having dropped the IK sweep. Proceed to
`TASK_METRIC_CORRELATION_PLAN.md`.

**Failure handling depends on which gate fails:**

- precondition or `eps_geom >= eps_gated` failure → engineering bug; stop before rank reporting;
- rank failure → take the middle path: keep `label_volume` but pass `qvol=None`, dropping only the
  gate; or
- route-overlay failure → inspect §3b diagnostics. If the clearance-preferred route still uses
  false-keeps or an implausible far-field bypass, the envelope relaxation is not usable for the
  study.

Do **not** start tightening the envelope automatically. The middle path can group legs by
orientation so the follow-on builds ~2 volumes per scene instead of one per leg — ~5 min/scene
instead of ~12, tolerable for one task at n=30 but not for three. Report the cost change and let
the user decide the scope before continuing.

---

## 3a. AMENDMENT (2026-07-30) — restore the relaxation invariant before judging rank

**Status: stages 1–3 are built (`372df8e`, `173e1f2`, `ccd3bc5`, `351cf79`). This section is a
correction to apply to existing code, not new scope.**

**Implementation status (2026-07-30): complete, GPU smoke pending.** The runner now builds both
aligned pairs, fails immediately on grid/EDT/FREE-subset/invariant violations, and retains native
independently snapped values only as diagnostics. Focused synthetic and artifact/report tests pass.

**What the first stage-3 scene showed:** `eps_gated = 0.070 > eps_geom = 0.050`. That does not
disprove the relaxation theorem because the runner did not actually compare a graph to its
relaxation. It passed the same world-space coordinates, but:

- gated snapped the start to `(0.15, 0.16, 0.81)`;
- geometric snapped it independently to `(0.17, 0.15, 0.81)`; and
- geometric applied `target_mask` while gated did not.

The exact guarantee is **`eps_geom >= eps_gated`**, not strict `>`. Equality is expected whenever
the gated optimum already exists in the relaxed graph. The guarantee requires all of:

1. the same grid and EDT values;
2. the exact same snapped start and goal voxels;
3. geometric FREE ⊇ gated FREE; and
4. geometric edges ⊇ gated edges (true when the geometric side removes the joint gate).

Stage 3 must assert these preconditions and then treat any remaining reversal as a code bug. Native
values produced from independently snapped endpoints are useful operational diagnostics, but they
are not an invariant test and must not drive the gate.

**(b) fails by construction, because of `target_mask`.** `task/occluder_task.py` registers only the
occluders into `collision_list` (the append at the ring-build loop); `self.target_obj` is created and
**never added**. `update_world` builds curobo's world from `collision_list`, so **curobo's world at
t=0 contains no target**. The gated sweep therefore labels the voxels the target occupies as `FREE` —
the gated route may pass straight through the object it is about to grasp. `_target_mask` blocks
exactly those voxels in the geometric build.

This is not a marginal region. The EDT measures distance to *occluders*, and the target sits at the
**centre of the ring** — the highest-clearance region in the scene (≈ `offset − occ_half` ≈ 0.16 m at
offset 0.2). `target_mask` deletes the single best corridor, at the pinned grasp endpoint, forcing a
detour through lower-clearance space. Expect a large gap, not a small one, and expect it in the
observed direction.

**(a) fails too, independently.** Both sides call `nearest_free_voxel(..., cfg.seed_snap)` against
their own FREE set. From the Kruskal loop in `lib/widest_path.py`, voxels are added in descending
clearance, so **eps\* <= min(edt[seed_a], edt[seed_b])**. Different FREE sets snap the seeds to
different voxels, up to `seed_snap = 0.10 m` apart — larger than a typical eps\* of ~0.085 m. This
can move eps\* in either direction on its own.

Stage 1 rules out the other candidate: the envelope is safe (false-prune 0), merely loose
(~38–42% false-keep). Over-pruning is not the cause.

### The fix — compare two properly aligned graph pairs

Build the geometric grid, envelope, obstacle mask, target mask and EDT once. Reuse the gated
`label`, `q_warm_3d` and EDT already returned by `compute_route_configs`; retain its native snapped
endpoints only as diagnostics. Snap each aligned pair once as specified below. Do not repeat IK.

**Pair A — no-target isolation.** This answers only whether removing per-scene IK nodes and the
joint gate is a relaxation.

1. Gated label: the existing raw gated label (the target is absent from curobo's world).
2. Geometric label: envelope + obstacle mask, with `mask_target=False`.
3. Snap the world endpoints **once against the gated FREE set**.
4. Pass that exact voxel pair to both widest-path calls.

Record `eps_gated_notarget_common` and `eps_geom_notarget_common`. Assert equal grid/EDT, assert
`gated_FREE & ~geom_FREE` is empty, and assert
`eps_geom_notarget_common >= eps_gated_notarget_common`.

**Pair B — target-masked production.** This validates the physically meaningful geometric metric
that plan 2 would consume.

1. Apply the same `target_mask` to both the gated and geometric labels. Do not add the target to
   `collision_list`; this is a local metric-label correction, not a curobo-world mutation.
2. Snap the world endpoints **once against the target-masked gated FREE set**.
3. Pass that exact voxel pair to both widest-path calls.

Record `eps_gated_target_common` and `eps_geom_target_common`. Assert the same preconditions and
the same one-sided inequality. This is the primary production rank series.

Keep the existing independently snapped `eps_gated_native` and `eps_geom_native` only as a
separate operational diagnostic. Also record:

- both common voxel pairs and their world coordinates;
- each endpoint's EDT value;
- `gated_FREE & ~geom_FREE`, attributed to `prune_mask`, `occ_mask`, or `target_mask`;
- `target_mask_delta` on both sides, evaluated at Pair B's common voxel pair so only the mask
  changes; and
- exact grid equality plus maximum absolute EDT difference.

This is deliberately a comparison-runner correction. Do not change the seed builder or the
already-collected gated pipeline.

### What the target_mask delta is worth on its own

It quantifies how much eps\* the **existing gated metric** has been gaining by routing through a
bottle curobo cannot see. If it is large, that is a finding about the metric already in use, not
only about the relaxation — and it belongs in `tool_clearance_metric.md` under known inaccuracies
regardless of how stage 3 resolves.

**Do not "fix" it by registering the target into `collision_list`.** That would change curobo's world
for every consumer, move already-collected gated numbers, and violate §0. Measure it; do not repair
it here.

### Re-run

Stage 3's verdict does not stand on n=1. Re-run the full ~10 scenes after this amendment. A reversal
is a relaxation violation only in one of the **common-endpoint, matched-mask** pairs after equal
grid/EDT and the FREE-set subset relation have been verified. If that happens, fail immediately and
report which precondition or widest-path assertion broke; do not proceed to Spearman.

---

## 3b. AMENDMENT (2026-07-30) — geometric route reconstruction hugs `zmin`

**Implementation status (2026-07-30): complete, GPU smoke pending.** The clearance-preferred
reconstruction and old-vs-new route diagnostics are wired into the geometric reporting path. The
synthetic floor-vs-climb test passes with byte-identical eps\*.

The first usable overlay exposed a second, separate bug in the reported route:

- 42 of 44 geometric route voxels were at `z = 0.78`;
- only the two endpoint transitions left that slice; and
- the gated route climbed to `z = 1.11`.

This does **not** explain `eps_geom < eps_gated`; §3a does. `widest_path_eps_3d` computes the
maximum bottleneck first. `reconstruct_widest_path_3d` then runs an unweighted BFS through
`FREE & (edt >= eps_star)`, so it returns the fewest-voxel member of the eps\*-optimal component.
When endpoint clearance caps eps\*, both a climb and the table-surface route can be
bottleneck-optimal, and BFS chooses the shorter floor route.

### Contained fix — change the representative route, not the objective

Add one geometric/reporting-only clearance-preferred reconstruction. Keep the allowed component
exactly `FREE & (edt >= eps_star)` and run Dijkstra over the same 26-neighbour edges:

- physical edge length uses `cfg.res` in x/y and `cfg.zres` in z;
- edge cost is physical length divided by the local effective clearance, defined as the minimum
  clearance of the edge's two endpoint voxels;
- map `inf` clearance to the largest finite clearance in the allowed component; if every allowed
  value is `inf`, fall back to physical shortest path.

This has no weighting knob. It prefers spending distance in high-clearance space, so a sustained
low-clearance floor route loses to a short climb into the open region. It cannot lower the route
below eps\* because it never leaves the already-established eps\*-optimal component.

Use this route for geometric Stage-3 overlays, false-keep overlap, and route diagnostics. Leave
`compute_route_configs`, the gated route used by the seed, `widest_path_eps_3d`, the EDT, masks,
and the grid unchanged.

For both the old BFS route and the clearance-preferred route, record `z_min`, `z_max`, fraction of
voxels on the `zmin` slice, anisotropic physical length, and clearance along the route. The fix is
accepted when the synthetic test below climbs and the real overlay no longer hugs the floor; it
does not get to change either aligned eps\* value.

---

## 4. Verification

No test runner is configured. The bar, in order:

1. `python script/bench_script/compare_geometric_vs_gated.py --help` — full import chain, no GPU.
2. **`lib/geometric_metric.py` imports no GPU stack.** A CPU test that imports it in a subprocess
   with `torch` blocked (or asserts `torch` absent from `sys.modules` after a fresh import) and
   fails loudly otherwise. This is the property the entire cost argument rests on — assert it, do
   not assume it.
3. `python script/bench_script/test_lib_env_api.py` — the AST check that every `env.<method>()` call
   in `lib/` resolves in `task/`. New `lib/` code must not break it. This check exists because a
   by-name refactor scan deleted a duck-typed method and voided a week of runs.
4. `test_ring_config.py`, `test_obstacle_set.py` — must pass unchanged.
5. `clearance_metric_3d.py --help`, `seed_from_clearance.py --help`,
   `analyze_occluder_visibility.py --help` — proves nothing additive broke the existing callers.
6. **Byte-identical check on the gated path.** Run `compute_route_configs` on one fixed scene before
   and after the stage-2 commit; diff eps\* and the route. Must be identical.
7. New focused tests: `target_mask` (§2), and the `LegResult` shape for a two-leg call sharing one
   volume.
8. **Relaxation theorem test (§3a).** On a small synthetic volume, use one common voxel pair and
   construct gated FREE/edges as strict subsets of geometric FREE/edges. Assert
   `eps_geom >= eps_gated` for both matched-mask variants. A second case with independently snapped
   endpoints must demonstrate why native values are not an invariant test.
9. **Stage-3 precondition test (§3a).** The comparison must fail before reporting rank if grids or
   EDTs differ, if either common endpoint is not FREE on both sides, or if
   `gated_FREE & ~geom_FREE` is non-empty.
10. **Floor-vs-climb reconstruction test (§3b).** Build a synthetic eps\*-optimal component where
    the old BFS hugs `zmin` and a slightly longer corridor climbs through higher clearance. Assert
    the clearance-preferred route climbs, stays entirely in `edt >= eps_star`, connects the exact
    same endpoints, and leaves the widest-path eps\* byte-identical.

---

## 5. Deliverables

| # | Artifact | Stage |
|---|---|---|
| 1 | `--mode false-keep` on `validate_reach_envelope.py` + `false_keep.json` + spatial figure, both arms | 1 |
| 2 | `lib/geometric_metric.py` (+ `target_mask`, CPU-only import test) | 2 |
| 3 | `compare_geometric_vs_gated.py` + Spearman + per-scene inflation + route overlays on ~10 scenes | 3 |
| 4 | Two matched-mask, common-voxel comparison pairs; native independently snapped values retained only as diagnostics | 3a |
| 5 | Hard precondition/invariant checks: identical grid/EDT, `gated_FREE ⊆ geom_FREE`, common endpoints, and `eps_geom >= eps_gated` | 3a |
| 6 | Subset attribution plus endpoint voxel/world coordinates, endpoint EDTs, and `target_mask_delta` | 3a |
| 7 | Clearance-preferred geometric reconstruction plus route height/length/clearance diagnostics | 3b |
| 8 | A one-line verdict: does the IK sweep come out, yes or no, with both aligned Spearman coefficients | 3 / 3a |
| 9 | The measured `target_mask_delta`, written into `tool_clearance_metric.md` known inaccuracies — it is a fact about the EXISTING gated metric | 3a |
| 10 | `agent-memory/tool_geometric_metric.md` (new) + edits to `tool_clearance_metric.md` and `tool_reach_envelope.md`; `status_current.md` rewritten | — |

---

## 6. What NOT to do

- **Do not tighten or rebuild the reach envelope** (§0). On a bad stage 1, report and stop.
- **Do not touch the seed builder** (§0).
- **Do not edit the gated path.** Additive only; verification item 6 enforces it.
- **Do not add a plotting framework.** Reuse `lib/plotting.py` and `save_route_visuals`.
- **Do not rename eps\*.** In every artifact call the new quantity `eps_geom` / "geometric eps\*",
  never plain `eps*`. It is a different construct — clearance along a gripper path through
  plausibly-reachable space, not along a route this arm can follow — and the existing notes define
  `eps*` as the latter. Conflating them in a filename or a json key will cost someone a day.
- **Do not tune `gate_tau`, `ik_seeds`, `warm_seeds` or `res` to make stage 3 pass.** Stage 3 is a
  measurement of the relaxation, not an optimisation target. Changing gated-side knobs to improve
  agreement invalidates it.
- **Do not register the target into `collision_list` to make the two sides agree** (§3a). It would
  change curobo's world for every consumer and move already-collected gated numbers. Measure the
  discrepancy; apply the target mask locally to both metric labels for the production comparison.
- **Do not test `eps_geom >= eps_gated` on independently snapped endpoints** (§3a). It is a hard
  invariant only after grid, EDT, masks and the exact voxel pair are aligned. Once those
  preconditions pass, a reversal is a bug and Stage 3 stops.
- **Do not fix the floor route by adding the table to the EDT, raising `zmin`, changing grid
  resolution/bounds, or tuning a route-weight constant** (§3b). Those change the scientific
  quantity or introduce a knob. The contained fix selects a better representative from the same
  eps\*-optimal component.
- **Do not use the clearance-preferred route in `compute_route_configs` or any seed tensor** (§3b).
  It is geometric/reporting-only until separately validated for joint continuity.
- **Do not draw a stage-3 verdict from one scene.** The gate is ~10 scenes on the
  two aligned series.

---

## 7. Known risks

| Risk | Signal | Response |
|---|---|---|
| Envelope false-keep is large *and* in the working volume | Stage 1 figure shows false-keeps over the table, not just at the reach edge | Expect stage 3 to fail; take its fallback |
| Envelope is not a valid outer bound on the left arm | Stage 1 false-prune > 0 for left | Right-arm-only, or regenerate the left artifact. Never proceed on an unsound bound |
| Ranks do not survive | Stage 3 Spearman < 0.8 | Middle path: keep `label_volume`, drop only the gate |
| Too few comparable scenes | Fewer than ~10 seeds have usable gated numbers | Re-run gated on fresh seeds; n < 8 makes a Spearman meaningless — say so rather than reporting one |
| `eps_geom` saturates at `inf` on the comparison scenes | Many scenes read "over the top" on both sides | `inf` ties destroy rank resolution. Report the tied fraction, and rank on the non-inf subset separately |
| Gated side is expensive to regenerate | Stage 3 turns into a long GPU run | Reuse only artifacts containing raw label/q/EDT/grid data; scalar historical eps\* values cannot support the aligned comparison |
| **`eps_gated > eps_geom` on native values** | Observed on the first stage-3 scene | Not an invariant test: endpoints and target policy differ. Recompute the two aligned pairs in §3a |
| **`eps_gated > eps_geom` on an aligned pair** | Common voxels/masks used and all preconditions pass | Widest-path/runner bug. Stop before rank reporting and reduce to the synthetic theorem test |
| FREE subset fails | `gated_FREE & ~geom_FREE` is non-empty in either matched-mask pair | Report mask attribution and stop. Do not call the result a relaxation |
| The two sides silently use different clearance fields | Grid mismatch or nonzero EDT difference | Stop before widest-path comparison; the theorem requires the same node weights |
| Production ranks fail while no-target ranks pass | Target-masked Spearman < 0.8 but isolation Spearman >= 0.8 | Removing IK preserves rank, but the metric plan 2 would consume does not. Do not proceed without a user decision |
| Geometric representative hugs the floor | Large `zmin` fraction despite an available high-clearance corridor | Use §3b's clearance-preferred reconstruction; verify eps\* and endpoints are unchanged |

---

## 8. What this unblocks

On pass, the metric becomes CPU-only and seconds per scene. That is what makes
`TASK_METRIC_CORRELATION_PLAN.md` affordable: a per-leg metric over stock RoboPRO tasks, computed on
CPU while the GPU runs pi05 rollouts, instead of ~150 s of curobo per leg fighting an 8.5 GB VLA
server for a 16 GB card.

It also removes the whole class of failures that has cost the most time on this branch — the
torch/jax co-residency CUDA illegal-memory-access, the checkpoint-restore OOM, the 169k-identical-
error spin. Those are not metric bugs; they are consequences of running curobo next to a VLA. This
deletes the situation rather than working around it.
