---
name: status_current
description: "THE volatile file — where the work stands, what is uncommitted, what is unverified, what is next. Rewrite it; never append."
metadata:
  type: project
---

**This is the only memory allowed to hold status.** Every other file holds durable knowledge.
When something here becomes permanent, move it out; when it becomes false, delete it.
**Rewrite this file, do not append to it.** Last rewritten 2026-07-29.

**Branch.** Working branch is `codex/bench-script-refactor`, HEAD `abb917a` ("Use unified metric
grid defaults in envelope tools"), forked from `peng-research-branch` @ `83e392f`. No PR open. The
multi-planner testbench branches are dead — see [[archive_planner_comparison]].

**The refactor just landed** (`ea31499`..`abb917a`, 8 commits, Codex, following
`bench_script/REFACTOR_PLAN.md`). Structural only — no behaviour changes, deliberately, so the A/B
data already collected stays comparable. Layout and the new dependency rule:
[[domain_bench_script_layout]]. **The refactor has NOT been GPU-verified**: `--help` import checks
and the CPU unit tests are the bar that was met; `smoke_test_seed_2a.py` (GPU) and the one-A/B-cell
check in REFACTOR_PLAN §7.6 — "a pure refactor must not move the seed firing rate from `83e392f`" —
have not been run. **Do that before trusting any new A/B numbers.**

Resolved by the refactor: the `zmax` split (the standalone tool used 1.4 while the seed builder
used 1.23 — they were measuring different volumes) is now unified at **1.23** in `SeedMetricConfig`.

Still open from `REFACTOR_PLAN.md` §6, all found by reading not running: `reachability_view.py`'s
`OCC_HEIGHT` is the milk-box extent, not olive-oil (display-only, see [[tool_clearance_metric]]);
the seed cache key collapses when the scene signature throws; broad exception swallowing.

**The live experiment: Phase 4 A/B (`direct` vs `seed`).** First big run
(`results/phase4_approach_mode/20260727-171544/`, launched 07-27, killed 07-28 after ~16 h on the
last of 4 cells) is **INCONCLUSIVE — keep it only as a defect-finding run.** Cells: curated/direct
50/50, curated/seed 50/50, standard/direct 40/50, standard/seed 9/50. Headline numbers
uninformative (curated 22% vs 20%, p=1.0). Dominant failure stage in BOTH cells was `grasp`
(~25/50), not pre_grasp — **the approach mode was not the binding constraint.**

Defects that run exposed:
1. The waypoint was still executed in direct/seed (fixed `2e07f33`, before the run). Any
   direct/seed rollout from BEFORE `2e07f33` is invalid A/B data.
2. **GPU memory leak — the real blocker** (fix `83e392f`, **NOT yet verified on GPU**).
   `_get_approach_seed` built a fresh IKSolver per (arm, scene) and never released it; 43/53
   curated no-route reasons were `OutOfMemoryError` and the machine ran out of VRAM entirely.
   **Open: unproven this was the ONLY leak** — `setup_demo` rebuilds planners per episode and
   predates this work. Watch `[seed-mem]` next run: flat `reserved` = that was the whole story;
   still climbing = a second leak.
3. **Seed firing rate was 7%**, which alone makes direct→seed meaningless (the summary's
   firing-rate gate caught it). Causes in order: `OutOfMemoryError` 43, "goal(grasp) seed
   unsnappable within 0.1 m" 7, joint-gate seam at tau=0.35 cutting a connected route 3. Only the
   first is addressed by `83e392f`; the other two set the ceiling and are next to attack.
4. The backward carry leg ran a scene-specific hand-authored chain in ALL modes — fixed by
   `PLACEMENT_MODE`. The general rule it produced now lives in [[feedback_scientific_rigor]].

**Measured costs (real, not extrapolated).** Seed build 133 s curated / 466 s standard at res=0.02,
zres=0.03, zmax=1.23. Approach build 128 s + carry build 187 s = 315 s of a 385 s seeded episode
(82% overhead) vs a 49 s direct episode. standard/seed ran 366 s/rollout vs 67 s direct — why 9
episodes took 12 h. **The old ~38 s figure was a COARSE-grid (0.03/0.06) smoke test and badly
under-estimates the real cost.** Disk: HDF5 ≈200 MB/episode (rgb+depth, 3 cams), videos ≈250 KB →
~40 GB per 200-episode run; videos saved per episode into `{success,fail}/video/`.

**Verified on GPU (pre-refactor):** attached-sphere transfer works (0 `no_attached_object`, carry
route built); memory across 2 episodes went 5.32 → 7.23 → 6.75 GiB allocated, i.e. returned between
episodes — **unproven over 50.**

**NOT yet GPU-verified:** the refactor itself; the `83e392f` leak fix; `PLACEMENT_MODE` Phase A+B
(expect a much lower absolute rate — `place_actor` must absorb the whole lift→pad transit, the jump
the chain existed to break up, so budget for low power at n=50/cell); the Phase C carry seed at
scale; the true-mesh `--occ-shape` path and the full `clearance_metric_3d` pipeline end to end; the
LEFT arm reach envelope (`reach_envelope.py --arms left` then `validate_reach_envelope.py --seed 1
--arm left`; 0.11 mc-safety should hold but confirm).

**Open at n=1 — do NOT overread:** seeded `carry_transit` succeeded in 1 attempt, then the unseeded
`carry_to_pad` failed 24/24; the unseeded direct cell planned both hops but physics EJECTED the
object (pos_drift 1.76 m).

**Next steps:** (a) GPU-verify the refactor is behaviour-neutral (one A/B cell, firing rate
unchanged vs `83e392f`); (b) re-run and watch `[seed-mem]` for flat reserved; (c) expect firing rate
to be the gate; (d) the dominant failure is `grasp`, so even a working seed may not move the
headline number until that is understood; (e) P3 cache split — one grid build can serve any endpoint
pair, since the grid depends on arm/scene/orientation/attached but NOT endpoints — still open and
worth it against the 82% overhead; (f) strip the temporary `ROBOPRO_SEED_DUMP` /
`ROBOPRO_SEED_ROUNDTRIP` debug blocks still present in the vendored `motion_gen.py`; (g) deferred 3c
(collapse the gap×z_lift×orient×y_offset sweep) only AFTER seeding reliably fires.

**USER DIRECTIVE 2026-07-28: placement has ALWAYS been a problem area — do NOT go deep fixing it.**
Only make fixes that directly impede moving to the next step. `place_actor` / landing-search /
object-ejection failures are to be reported and left alone.
