# Codebase cleanup: reduce lines and moving parts, no functionality lost

> **For the executing agent.** Written by Claude Code on 2026-08-10, reviewed and corrected by
> Codex on 2026-08-10, and approved in direction by the user. Counts describe the dirty tree at
> review time; re-run the scans in *Verification* before trusting them.
>
> - **All paths are relative to the repo root** (`/home/haccerkat/Documents/Research/Experimental/RoboPRO`).
> - **Read `agent-memory/MEMORY.md` and `agent-memory/status_current.md` first**, per `AGENTS.md`.
>   This plan leans on `feedback_minimal_changes.md` (over-engineering is the worst failure mode),
>   `domain_bench_script_layout.md` (the `lib/`-must-not-import-CLI rule), and
>   `feedback_script_conventions.md`. It does not restate them.
> - **After code changes, run `graphify update .`** (AGENTS.md).
> - **Update `agent-memory/status_current.md`** when phases land — rewrite, never append.
>
> **Decisions the user already made on 2026-08-10 — do not re-litigate:**
> 1. **Superseding decision:** the incomplete 620/3000 metric post-process no longer blocks
>    cleanup. Preserve its data on disk, but in-place resumption is not required; a future
>    post-process may restart from episode 1. C1 below remains as provenance, not a gate.
> 2. `bench_envs` is cleaned first; `bench_script` waits for the research work to be committed (C2).
> 3. In §1.3, delete only the inert methods; **keep** the unused `set_*`/`is_*` articulation API.
>
> **Where to start:** the research checkpoint, corrected plan, and Phase 3 are committed. Begin
> Phase 1, then continue through Phases 2, 2.5, 2.6, and 4 in order. Do not delete or rewrite the
> existing 620 metric records; source changes may intentionally make that partial run
> non-resumable under its original immutable configuration.
>
> **Commit granularity:** one commit per numbered sub-phase (1.1, 1.2, …), each independently
> revertable. Before every commit, verify the staged path allowlist and `git diff --cached --check`.
> Existing research changes must land in their own earlier commit; never stage them accidentally
> with cleanup.

## Context

The repo has accumulated ~18 months of research iteration. The 2026-07-29 Codex refactor already
fixed the worst structural problem (`lib/` no longer imports from CLI scripts), but it was
deliberately *structural only* and left the accumulated cruft in place: dead imports, copy-pasted
boilerplate, one-off validation scripts from closed issues, and vestigial env-var fallbacks.

Measured, not estimated:

| Bucket | Removable | Verified by |
|---|---|---|
| `bench_envs` dead imports | **371 names / 77 files** | AST scan (rerun in Verification) |
| `bench_envs` `setup_demo` boilerplate | **~180 lines** | 58 plain copies, 22 variants |
| `bench_envs` inert methods | **223 lines** | per-symbol repo-wide grep |
| `bench_envs` commented-out `info` blocks | **~108 lines** | 18 files |
| `bench_script` dead imports | **143 names / 20 files** | AST scan excluding `__future__` imports |
| `bench_script` duplicate helpers | **~150 lines** | byte-diffed |
| Orphan files (see §5) | **~1,900 lines** | zero-reference grep |
| **Scheduled Phases 1–2** | **~1,140 Python lines** | excludes optional §5 deletions |

**Intended outcome:** a smaller, more legible tree with the same behaviour.

Two kinds of change, and nothing else:

1. **Removal** (Phases 1–2, 3) — the table above.
2. **Relocation** (Phases 2.5–2.6) — 5 library modules stranded at the CLI level move into `lib/`,
   and 7 plan docs move into `plans/`. Imports, `__file__` anchors, usage text, and references are
   updated only where the move requires it.

It introduces **no new modules, no frameworks, no abstractions, and no new dependencies**
(`agent-memory/feedback_minimal_changes.md`). Where a tidier-looking reorganization would have
required new indirection, it is listed under *Explicitly NOT doing* with the reason.

---

## The two hard constraints

### C1 — Historical campaign freeze, explicitly waived on 2026-08-10

`scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess`
was at **620/3000** committed episodes when cleanup began, with `processing_complete: false`.
Resumption
validates `task_metric.py::_metric_code_version()`, which hashes:

```
task_metric.py
lib/geometric_metric.py
lib/metric_buckets.py
lib/task_roles.py
lib/waypoints.py
```

Editing any byte changes the immutable post-process configuration and prevents an in-place resume.
The existing 620 committed metric records remain on disk, but the code would need to be restored
byte-for-byte before the remaining 2380 can be computed in the same run. The user explicitly
accepted losing that in-place resume path so cleanup can proceed. **Preserve the partial artifacts;
do not delete or rewrite them.**

There is a second binding that the original plan missed. `task_scene_code_version()` hashes:

```
benchmark/bench_envs/study/put_cup_on_coaster.py
benchmark/bench_envs/study/_study_base_task.py
benchmark/bench_envs/_bench_base_task.py
benchmark/bench_envs/utils/scene_gen_utils.py
benchmark/bench_task_config/task_objects.yml
benchmark/bench_task_config/<base_config>.yml
```

Every manifest row stores that version. `task_metric.py` recomputes it, then currently checks it
only after scene setup and geometric computation. Editing any of these files both blocks resume
and wastes one expensive metric before the mismatch is reported. The current tree matches the
manifest (`d8e0abb…` for `bench_demo_study_clean`); preserve it.

The exact five-file hash is not the entire behavioral dependency closure. That was why the original
execution deferred all `bench_script` cleanup. The 2026-08-10 waiver supersedes that deferral; the
cleanup must still be behavior-preserving for current code, but need not preserve the partial run's
recorded source hashes.

The independent visualization audit is no longer pending. The final `metric_route_visuals_v4`
run is complete at 50/50 episodes and 200 figures, and its three-file code hash matches the current
tree. Preserve the source in the research commit; after that commit, visualization cleanup is not
blocked by the audit.

### C2 — Sequencing: `bench_envs` first.

At review time the working tree held **2,876 added and 452 deleted lines across 35 tracked files,
plus 33 untracked status entries**. Four tracked `bench_envs` files were dirty and
`benchmark/bench_envs/eval_video.py` was untracked. Treat every pre-existing change as user-owned.

Sequence: the research checkpoint, this plan, and Phase 3 are committed. **Run Phase 1
(`bench_envs`) before any Phase 2 work.** Do not interleave `bench_envs` and `bench_script`
cleanup. The campaign waiver allows the full Phase 1 scope, including the formerly hash-bound
files.

---

## Phase 1 — `benchmark/bench_envs/` (~880 lines, self-contained)

### 1.1 Dead imports — 371 names across 77 files

The AST scan finds no local loads of these imported names. That is evidence of repo-internal
non-use, not proof that every import is redundant: ten scanned files do not import
`envs.utils.*`, an import may be a re-export, and an import can have side effects. Inspect each
`__init__.py` and any public utility module before removal. Biggest offenders: `glob` (66 files),
`math` (64), `deepcopy` (61), `sapien` (50), `euler2quat` (18),
`get_collison_with_objs` (16).

Worst single files, all import headers copy-pasted from a common ancestor:
- [_study_base_task.py](benchmark/bench_envs/study/_study_base_task.py) — 23 dead names
- [_office_base_task.py](benchmark/bench_envs/office/_office_base_task.py) — 19
- [_kitchens_base_task.py](benchmark/bench_envs/kitchens/_kitchens_base_task.py) — 18
- [_kitchen_base_large.py](benchmark/bench_envs/kitchenl/_kitchen_base_large.py) — 15
- [_bench_base_task.py](benchmark/bench_envs/_bench_base_task.py) — 14, including `torch`, `gym`,
  `imageio`. Their removal tidies this module's namespace; it does not avoid loading those packages,
  because `Base_Task` imports them independently.

Drive it from the corrected AST scan in Verification, not by hand. The scan must never classify
`from __future__ import ...` as removable. Run this sub-phase only after C1/C2 clear; by then the
research changes must already be committed and the cleanup commit may cover all 77 files.

### 1.2 `setup_demo` boilerplate — 80 leaf definitions collapse to 23

**Confirmed:** neither `Bench_base_task` nor `Base_Task` (`customized_robotwin/envs/_base_task.py`)
defines `setup_demo`. All 80 leaf tasks define it. **58 have the same two-statement body**:

```python
def setup_demo(self, is_test=False, **kwargs):
    kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
    super()._init_task_env_(**kwargs)
```

The value is *not* redundant — `customized_robotwin/envs/robot/robot.py:42` defaults to
`{"mesh": 100, "obb": 1}`.

Add once to `Bench_base_task` in [_bench_base_task.py](benchmark/bench_envs/_bench_base_task.py),
next to the existing `_init_task_env_` stub at line 58:

```python
def setup_demo(self, is_test=False, **kwargs):
    kwargs["collision_cache"] = {"mesh": 100, "obb": 3}
    self._init_task_env_(**kwargs)
```

`self._init_task_env_` dispatches to whichever of the 4 scene bases the leaf inherits — the same
target `super()._init_task_env_` resolves to today. Use assignment, not `setdefault`: every current
leaf unconditionally overwrites a caller-provided `collision_cache`, so `setdefault` would be a
behavior change.

Then:
- **58 files:** delete the whole `setup_demo` override.
- **22 variant files** (19 `kitchenl/`, plus
  `study/{put_cup_in_box,move_book_onto_table,move_seal_onto_book}.py`):
  keep the override, delete only the `kwargs["collision_cache"] = ...` line, and change the tail
  from `super()._init_task_env_(**kwargs)` to `super().setup_demo(**kwargs)`.

Callers are unaffected — `TASK_ENV.setup_demo(...)` in `collect_data.py`, `eval_policy.py`,
`eval_policy_client.py`, `precollect_eval_seeds.py`, `collect_rollout_client.py`, and 8
bench_script tools all keep resolving.

### 1.3 Inert methods — 223 lines (user's scope decision: inert only, keep the `set_*`/`is_*` API)

Each verified zero-reference repo-wide by individual grep:

| File | Symbol | Lines |
|---|---|---|
| [create_actor_custom.py:68](benchmark/bench_envs/utils/create_actor_custom.py#L68) | `create_multiple_obj_actor` | 66 |
| [_kitchen_base_large.py:733](benchmark/bench_envs/kitchenl/_kitchen_base_large.py#L733) | `_entity_aabb` | 54 |
| [scene_gen_utils.py:459](benchmark/bench_envs/utils/scene_gen_utils.py#L459) | `get_random_valid_placement` | 35 |
| [scene_gen_utils.py:635](benchmark/bench_envs/utils/scene_gen_utils.py#L635) | `get_obj_new_pose` | 22 |
| [_kitchen_base_large.py:1156-1168](benchmark/bench_envs/kitchenl/_kitchen_base_large.py#L1156-L1168) | drawer quartet: `_init_drawer_states`, `set_drawer_closed`, `set_drawer_open`, `is_drawer_open` | 12 |
| [_kitchen_base_large.py:1226](benchmark/bench_envs/kitchenl/_kitchen_base_large.py#L1226) | `_sample_model_id` | 5 |
| [pick_boxdrink_from_basket.py:48,106](benchmark/bench_envs/kitchenl/pick_boxdrink_from_basket.py#L48) | `_world_point_in_entity_local`, `_ee_pose_above_place_target` | 13 |
| [put_sauce_can_in_cabinet.py:120](benchmark/bench_envs/kitchenl/put_sauce_can_in_cabinet.py#L120), [put_milk_box_in_fridge.py:145](benchmark/bench_envs/kitchenl/put_milk_box_in_fridge.py#L145) | `_cabinet_inside_target_pose`, `_fridge_inside_target_pose` | 16 |

The drawer quartet is never called in this repository, including from `__init__`. Note that an
in-tree comment says it was retained for older-code compatibility, and
`create_multiple_obj_actor` is wildcard-exported from `bench_envs.utils`; the deletion claim is
therefore **repo-internal zero-reference**, not external API preservation. The user explicitly
approved deleting these inert methods. The two `scene_gen_utils.py` removals remain blocked by C1
until the campaign completes.

**Keeping** (user's explicit call — do not delete these): `set_fridge_open_random_angle_between`, `set_fridge_closed`,
`set_cabinet_closed`, `is_cabinet_open`, `is_fridge_open`, `is_fridge_fully_open`,
`is_object_in_sink`, `is_object_on_dishrack` — an unused but coherent articulation affordance
surface.

### 1.4 Commented-out `info["info"]` blocks — 18 files, ~108 lines

18 `office/` files carry a commented-out 6-line `self.info["info"]` block, e.g.
[put_mouse_on_pad.py:138-143](benchmark/bench_envs/office/put_mouse_on_pad.py#L138-L143).

**The live version in 17 `study/`/`kitchenl/` files must stay** — the return value is consumed by
`collect_data.py:268` and `eval_policy.py:245`. Delete only the commented-out ones.

### 1.5 The `include_collison` typo — rename only

- [_study_base_task.py:144](benchmark/bench_envs/study/_study_base_task.py#L144): `kwags.get("include_collison", True)` (missing `i`)
- 3 study tasks feed the typo'd key: `put_cup_in_box.py:20`, `move_book_onto_table.py:20`, `move_seal_onto_book.py:30`

All 4 sites are internally consistent. Update the three task writers to `include_collision`, but
keep a compatibility read in the Study base so external callers using the old typo do not change:

```python
self.incl_collision = kwags.get(
    "include_collision", kwags.get("include_collison", True)
)
```

The correctly spelled key wins if both are supplied. Note `_study_base_task.py` defaults to `True` while
`_kitchens_base_task.py:124` / `_kitchen_base_large.py:301` default to `False` — **flag this to the
user, do not "fix" it.** Changing a default is a behaviour change and is the scientist's call.

`_study_base_task.py` is scene-hash-bound, so this waits for C1 and lands in the §1.5 commit.

### 1.6 `Bench_base_task._init_task_env_` stub (line 58-59)

All four scene bases override it and none calls `super()`, so the `pass` is unreachable. **Leave
it.** It shadows `Base_Task._init_task_env_` (`customized_robotwin/envs/_base_task.py:53`), and
deleting it re-exposes the parent — a real behaviour change for a 2-line gain. Not worth it.

---

## Phase 2 — `bench_script` (~300 lines; after Phase 1)

The user waived C1 and accepts that a future post-process may restart from episode 1. Preserve the
existing partial artifacts, but do not delay this phase for the old source hashes.

### 2.1 Dead imports — 143 names across 20 files at review time

Worst: [clearance_metric_3d.py](customized_robotwin/script/bench_script/clearance_metric_3d.py).
The count is expected to drift when the research work lands, so rerun the corrected scan after C2.
For every public-looking module, check accidental re-exports before deleting imports. Never remove
`from __future__ import annotations`; the original scan incorrectly reported `annotations` as dead.

The 6 `task/` mixins each begin with the same copy-pasted header lifted from `occluder_task.py`
(`"""Methods extracted mechanically from analyze_occluder_visibility.py."""`), carrying ~57 dead
names. Two are not cosmetic: **`import seed_from_clearance as sfc` (536 lines) is dead in 5 of 6
mixins**, and `import torch` is dead in 4 of 6 — both paid at import time.

Removing `sfc` from those 5 plus its dead import in `analyze_occluder_visibility.py` leaves four
live importers. It also shrinks the upward `task/` → CLI-script dependency from 6 modules to 1
(`seeding_mixin.py`, which genuinely uses it).

### 2.2 Duplicate helpers — fold into the lib module that already owns the concern

No new modules. Three moves only:

- **`_sha256`** — 5 byte-identical 6-line copies: `analyze_metric_distribution.py:43`,
  `analyze_metric_correlation.py:39`, `visualize_task_metric_routes.py:61`,
  `validate_bucket_spec.py:17`, and `task_metric.py:182`. Move to
  [lib/run_io.py](customized_robotwin/script/bench_script/lib/run_io.py), which is already the IO
  module and already imported by all of them. The visualization audit has completed and C1 has
  cleared before this phase, so all five copies may be handled together.
- **`_save_figure`** — `analyze_metric_correlation.py:352` and
  `lib/vla_reporting.py:274` (`_save_figure_atomic`) are identical 6-liners;
  `analyze_metric_distribution.py:195` is the strict superset (adds `mkdir(parents=True)` and a
  `bbox_inches` kwarg). Promote the superset to `lib/plotting.py`. Pass `bbox_inches` explicitly
  at each call site — the current defaults differ (`None` vs `"tight"`), and silently unifying
  them would change existing figures.
- **`reachability_view._iso_mesh`/`_ceiling`** — verbatim copies of the geometry core of
  `reachability_map._plot_isosurface:304-318` and a strict subset of
  `reachability_map._column_bounds:202-211`. **The same file pair already does this correctly for
  the other figure**: `reachability_map.py:243` does `import reachability_view as rv` and delegates
  `ceiling_heatmap` with a "single source" comment. Extend the existing precedent to these two.
  ~25 lines.

### 2.3 `diag_kitchen_curobo.py` — dedup only what is equivalent

Import `get_embodiment_config` from `lib.scene_build` and drop the unused `numpy` import. Keep the
local `build_cfg(seed)`: it deliberately sets `need_plan=True` and `save_data=False`, while the
shared builder's expert mode is true/true and measurement mode is false/false. Replacing it would
be a behavior change for a diagnostic and is not worth a new shared mode.

### 2.4 One internal vestige — `lib/run_io.effective_out_dir`

- **`lib/run_io.effective_out_dir`** (L17-24) — 8 lines of docstring wrapping
  `return Path(args.out_dir)`, two call sites. Its own docstring says it exists to shadow
  `analyze_natural_visibility.effective_out_dir:206`, which *does* have real behaviour (a
  `_rollout` suffix). Two same-named functions in one directory doing different things. Inline and
  delete.

Keep `SEED_FROM_CLEARANCE`, `POST_GRASP_ESCAPE_ATTEMPTS`, and the single-choice `--obstacles all`
flag. Repo grep cannot observe environment variables set in a user's shell, and deleting a CLI
option breaks command compatibility even when it has only one valid value. Their small line saving
does not justify claiming “no functionality lost.”

---

## Phase 2.5 — Layout: finish what `lib/` and `task/` started

### What must NOT move, and why

Every top-level CLI script depends on being invoked with `bench_script/` as `sys.path[0]`:

```python
from setup_paths import setup_paths   # flat import, same dir
setup_paths()
from lib.geometric_metric import ...  # package relative to same dir
```

The Makefile invokes them by path (`$(PYTHON) script/bench_script/visualize_task_scene.py`,
5 targets at lines 222/230/310/324/332), as do 4 `scripts/validation/*.sh`,
`policy/pi05/vla_occluder_rollout.sh:92`, `README.md:113`, and `docs/install.html:132`.

Moving a CLI script one directory down makes `sys.path[0]` the *new* subdir, breaking
`import setup_paths` and `from lib.*` in the same stroke. The fix would be a
`sys.path.insert(...)` above the imports of every moved file — precisely the `# noqa: E402`
pattern the 2026-07-29 refactor was undertaken to eliminate
(`agent-memory/domain_bench_script_layout.md`).

**So: the CLI scripts and the 12 `test_*.py` stay flat.** (Tests would otherwise be the obvious
`tests/` candidate, but there is no pytest — `.venv/bin/pytest` is absent and `pyproject.toml`
declares none — so they'd need the same path hack for no gain.)

### What does move: the 5 library modules stranded at top level

These are **imported, never invoked by path** — verified zero references in Makefile, `scripts/`,
`docs/`, or `README.md`; the only hits are stale stack traces in `results/*/logs/`. Moving them
makes the layout state the rule that already exists: **`lib/` = imported, top level = invoked.**
It is not a pure import edit: two modules derive paths from `__file__` and must be corrected.

| module | importers to update | note |
|---|---|---|
| `metric_diagnostics.py` | 1 — `clearance_metric_3d.py` | no `__main__`; trivial |
| `reachability_view.py` | 2 — `reachability_map.py`, `swept_volume_3d.py` | keeps `main()` → `python -m lib.reachability_view`; after the move, change the repository-root anchor from `parents[3]` to `parents[4]`. Do **after** §2.2 merges `_iso_mesh`/`_ceiling`. |
| `carry_object_spheres.py` | 1 — `task/seeding_mixin.py` | keeps argparse `main()` → `python -m lib.carry_object_spheres`; change both customized-root anchors from `parents[2]` to `parents[3]`. |
| `metric_viz.py` | 6 | **defer to Phase 4** — see blocker below |
| `seed_from_clearance.py` | 10, → 4 after §2.1 | **defer to Phase 4** — `seed_from_clearance.py:40` does `import metric_viz as cm`, so it cannot move before `metric_viz` does |

Convert each moved module to lib's relative-import convention (`from .ik_grid import build_grid`),
matching `lib/geometric_metric.py:30-40`. Use `git mv` so history follows. Update their embedded
usage text to the `python -m lib.<module>` form. The verification gate must run the carry selftest
and assert the reachability default path; `--help` alone does not exercise either anchor.

**The payoff.** `task/seeding_mixin.py` is the last module importing upward into the CLI layer
(`sfc` and `cos`). Once §2.1 drops the dead `sfc` imports and these moves land, the AST dependency
check in Verification returns no top-level-script imports from `lib/` and at most the live `sfc`
import from `task/` until Phase 4.

Do not use a hard-coded top-level file count; the research commit adds entry points. The structural
gate is semantic: every imported-only module is under `lib/`, except the intentional bootstrap
`setup_paths.py` and the two Phase-4 modules until they move.

**Blocker for the two Phase-4 moves.** `visualize_task_metric_routes.py:390-394`:

```python
def _visualization_code_version():
    root = Path(__file__).resolve().parent
    return hash_files([Path(__file__), root / "metric_viz.py", root / "lib" / "plotting.py"])
```

Moving `metric_viz.py` makes that raise `FileNotFoundError` until line 393 is updated. The v4 audit
is complete and remains valid as an artifact of its recorded source hash; the move intentionally
creates a new visualization code version and must happen in one atomic commit with the path update.

*Note on `hash_files` (`lib/scene_provenance.py:45`): it sorts resolved full paths, then digests
`path.name` + contents. Directory bytes are not hashed directly, but a move can change sort order
and therefore the aggregate hash even when basename and contents are identical. Treat every move
of a hashed file as a version change.*

---

## Phase 2.6 — Plan docs into `plans/`

`bench_script/plans/` already exists — this document lives in it as `CLEANUP_PLAN.md`. Move the
other 7 `.md` files out of `bench_script/` to join it, via `git mv`:

```
CLEARANCE_2D_TO_3D.md          REFACTOR_PLAN.md *
GEOMETRIC_EPS_VALIDATION_PLAN.md   SEED_TRAJECTORY_PLAN.md
TASK_METRIC_CORRELATION_PLAN.md    VLA_ROLLOUT_PLAN.md
TASK_METRIC_ROUTE_VISUALIZATION_PLAN.md *
```

**Not into `docs/`** — that is the published GitHub Pages site (`index.html`, `app.js`,
`manifest.json`, 102 mp4s). Internal agent plan docs do not belong in a published site.

`*` = stale content, but move rather than delete: `REFACTOR_PLAN.md` describes a completed
refactor, and `TASK_METRIC_ROUTE_VISUALIZATION_PLAN.md` §5/§6 are known-wrong
(`status_current.md:49`). Both are still cited from agent-memory.

**Referrers to update — each needs only a path-string edit:**

- Code docstrings: `compare_geometric_vs_gated.py` (→ `GEOMETRIC_EPS_VALIDATION_PLAN.md`),
  `seed_from_clearance.py` (→ `SEED_TRAJECTORY_PLAN.md`)
- `curobo_seed_traj.patch`
- Vendored `customized_robotwin/envs/curobo/src/curobo/wrap/reacher/motion_gen.py` — a comment.
  We already patch this file (`ROBOPRO_SEED_DUMP` blocks), so updating it is consistent.
- Cross-references *inside* the moved docs (`TASK_METRIC_CORRELATION_PLAN` ↔
  `GEOMETRIC_EPS_VALIDATION_PLAN`, `REFACTOR_PLAN` → `CLEARANCE_2D_TO_3D`). These stay same-dir, so
  bare filenames keep resolving — verify none uses a `bench_script/`-prefixed path.
- **`agent-memory/` — 7 files, and this is the one that matters.** `domain_bench_script_layout.md`,
  `status_current.md`, `tool_seed_from_clearance.md`, `domain_visibility.md`,
  `tool_route_visualizer.md`, `tool_reach_envelope.md`, `tool_clearance_metric.md`. Per
  `CLAUDE.md` this folder is shared with Codex; a stale path there sends the other agent hunting
  for a missing file. Update in the **same commit** as the move.

`agent-memory/domain_bench_script_layout.md` also needs its layout description updated for both
Phase 2.5 and 2.6 — it is the note future agents read first.

---

## Phase 3 — repo hygiene (5 minutes, zero code risk)

- Add `graphify-out/` to `.gitignore` — 96 MB untracked at root plus a second nested copy at
  `customized_robotwin/script/graphify-out/` (1.9 MB), both showing as `??` in every `git status`.
  `AGENTS.md` already says "dirty graphify-out/ files are expected", implying this was intended.
- Delete the **repo-root file** `$WORKSPACE_ROOT/d` — a 63 KB stray ANSI-coloured `git diff` dump
  from Jul 28. Do not use `/d`, which is an unrelated absolute path. The repo file is untracked and
  un-ignored, so it is permanent `git status` noise. It also pollutes name greps for
  `analyze_occluder_visibility`, `seed_from_clearance`, etc.
- Delete the empty `tools/` directory with `rmdir`, not a recursive removal.
- Run `git rm --cached` on the two explicit tracked PNG paths under
  `customized_robotwin/visibility_validation/`, then ignore that directory. The PNGs must remain on
  disk and become ignored.
- Remove exactly the 7 named orphan `.pyc` files listed in the Phase 3 gate. Do not use a wildcard
  such as `clearance_metric*.pyc`; live `clearance_metric_3d` bytecode also exists.
- Drop the stale ignore entries for `benchmark_data/*` plus its `.gitkeep` exception,
  `presentation_material/`, `docs/references/`, and `usecase_roadmap.tex`. Keep both
  `collision_test/` ignores: tracked collision-metric tooling documents one as generated output,
  and output ignores need not correspond to a directory that exists before a run.

---

## Phase 4 — formerly frozen cleanup, enabled by the campaign waiver

The metric receipt may remain incomplete. Preserve it on disk, and verify the completed route-audit
receipt before changing visualization code:

- Dead imports in the five frozen files.
- Revisit `visualize_task_metric_routes.py` / `metric_viz.py` / `lib/plotting.py`. The required
  50-episode v4 audit is already complete; verify its `processing_complete: true` receipt before
  changing the code.
- **The two deferred Phase-2.5 moves**, in this order: `metric_viz.py` → `lib/` (update
  `visualize_task_metric_routes.py:393` in the same edit), then `seed_from_clearance.py` → `lib/`.
  Update direct commands and doc references to module invocation, including
  `VLA_ROLLOUT_PLAN.md`, `SEED_TRAJECTORY_PLAN.md`, and the verification commands. Finish by
  re-running the AST dependency check — it must return 0 hits in `lib/` *and* `task/`.

---

## §5 — Orphan files: the user decides, the agent does not

**Nothing in §5 may be deleted without an explicit instruction from the user for that specific
file.** This section is a findings list, not a work list. Present it and wait.

### Zero references anywhere — safest candidates

| Path | Lines | Note |
|---|---|---|
| `customized_robotwin/script/bench_script/view_object.py` | 258 | Zero references repo-wide. Last touched 2026-06-26. |
| `customized_robotwin/script/bench_script/validate_visibility_measurement.py` | 218 | Appears only in a `REFACTOR_PLAN.md` list. |
| `customized_robotwin/script/bench_script/analyze_natural_visibility.py` | 253 | Docs-only refs. **Deleting it makes `lib/visibility.py` single-consumer.** |
| `scripts/upload/upload_benchmark_data.py` | 108 | Zero refs; the `benchmark_data/` dir it targets does not exist. |
| `scripts/upload/upload_instructions_to_hf.py` | 136 | Zero refs. |
| `scripts/slurm/slurm_precollect_then_eval.sh` | ~60 | Zero refs (its sibling `slurm_eval_bench.sh` *is* in the README — keep that one). |
| `docs/pr_assets/*.png` (3 files) | — | One-off PR screenshots, tracked in git, referenced by no HTML/MD. |

### One-off scripts from closed issues — archive rather than delete?

`scripts/validation/{baseline_test_ood.sh, integration_test_ood.sh, validate_ood_targets.sh,
smoke_test_compositional.sh}` (~430 lines total). Written for issues #19/#21/#22; only
cross-reference each other. `validate_ood_objects.py` in the same dir **is** live
(`agent-memory/repo_task_assets.md`) — keep it.

### The name-collision cluster

`benchmark/bench_script/` — 3 unwired files (`run_collision_metrics_all.py`,
`test_collision_metrics.py`, `generate_comparison_videos.py`, ~890 lines). No Makefile target, doc,
or script invokes any of them, and `generate_comparison_videos.py`'s docstring still says
`conda activate RoboTwin`, contradicting the current uv workflow. It also collides by name with the
actively-developed `customized_robotwin/script/bench_script/` — a standing source of confusion for
both agents and greps. **Recommended: rename the directory even if the contents are kept.**

### Keep despite looking orphaned — do not delete

- `reach_envelope.py` — has no importers, but produces `reach_envelope_{arm}.npz`, consumed by
  `lib/labeling.load_reach_envelope`. A name grep understates it.
- `swept_volume_3d.py`, `smoke_test_seed_2a.py` — manual GPU tools; `swept_volume_3d` is named in
  live observer-hook comments in `task/occluder_task.py:196` and `_bench_base_task.py:1527`.
- `scripts/validation/hdf5_to_video.py` — unreferenced but a genuinely useful standalone utility.
- `customized_robotwin/data/process_stuck.py` — the single allowlisted file in an ignored tree.

### Stale in-tree design docs — moved by Phase 2.6, but two could be deleted instead

All 7 go to `plans/` under Phase 2.6. Two are stale enough that the user may prefer deletion — ask:

- `REFACTOR_PLAN.md` (388 L) — describes a refactor completed 2026-07-29; referenced only from
  agent-memory, and `domain_bench_script_layout.md` already records everything durable in it.
- `TASK_METRIC_ROUTE_VISUALIZATION_PLAN.md` — §5/§6 are known-wrong (`status_current.md:49`).

The other five stay load-bearing: `GEOMETRIC_EPS_VALIDATION_PLAN.md` is cited from
`compare_geometric_vs_gated.py`'s docstring, and `SEED_TRAJECTORY_PLAN.md` from
`seed_from_clearance.py`, `curobo_seed_traj.patch`, and the vendored `motion_gen.py`.

### Disk, not git

`scripts/validation/results/` is **13 GB** (`phase4_approach_mode` 6.9 G, `occluder_visibility`
3.5 G, `task_metric_vla_full` 1.2 G). Correctly gitignored, but it is the dominant disk consumer.
The v4 route audit is complete. Older `metric_route_visuals*` directories are superseded, but
deleting result directories remains explicitly the user's call, not the agent's.

---

## Explicitly NOT doing

Listed so they don't get proposed later as "obvious wins" —
`agent-memory/feedback_minimal_changes.md` treats over-engineering as the worst failure mode.

- **No generated argparse.** An `add_metric_args(parser)` derived from
  `fields(SeedMetricConfig)` would remove ~40 lines of duplicated flag blocks, but it replaces
  hand-editable flag declarations with metaprogramming. The user hand-edits these scripts in a fast
  edit-and-rerun loop (`agent-memory/feedback_minimal_changes.md`). Not worth it. *(Worth knowing:
  the hand-maintained mirror is already out of
  sync — `gate_tau_sweep` is a `SeedMetricConfig` field with no CLI flag anywhere. That is a
  one-line fix, not a reason to build a generator.)*
- **No `open_env` context manager.** Would collapse 24 lines of duplicated `finally: env.close_env()`
  teardown across 3 files, but 2 of the 3 are frozen by C1.
- **No `iter_jsonl` helper.** Five JSONL readers share an 8-line skeleton, but their validators all
  differ and two same-named `_read_records` have genuinely different semantics. Merging risks a
  silent semantic change for ~40 lines.
- **No merging of same-named-but-different functions.** `summarize_records`
  (`analyze_metric_distribution.py:149` vs `compare_geometric_vs_gated.py:162`) and `_scene_record`
  (`compare_geometric_vs_gated.py:736` vs `validate_task_geometric_ranking.py:167`) are *not*
  duplicates. Rename at most.
- **No behaviour changes.** Not the `include_collision` default divergence (§1.5), not the
  `Base_Task._init_task_env_` shadowing (§1.6), not `bbox_inches` unification (§2.2).
- **No removal of shell/CLI compatibility surfaces merely because repo grep finds no caller.**
  Keep `SEED_FROM_CLEARANCE`, `POST_GRASP_ESCAPE_ATTEMPTS`, and `--obstacles all`; environment
  variables set interactively and external command lines are outside repo grep's visibility.
- **No moving the CLI scripts into role subdirectories** (`cli/`, `viz/`, `validation/`). It reads
  tidier and costs real fragility: it breaks `import setup_paths` and `from lib.*` in every moved
  file, breaks 5 Makefile targets, 4 shell scripts, `README.md` and `docs/install.html`, and the
  repair is the `# noqa: E402` path-hack pattern the last refactor removed. §2.5 gets the same
  readability by moving the *imported* modules into `lib/` instead, which costs only import edits.
- **No pytest migration for the 12 `test_*.py`.** A `tests/` folder is the natural home, but
  pytest is not installed (`.venv/bin/pytest` absent, none in `pyproject.toml`), so this means
  either a new dependency or the same path hack. They stay flat and stay `python test_x.py`.
- **No `plans/README.md` index.** The folder is 7 files with self-describing names.

**Stopping conditions** — each phase is done at its line, not when the tree looks perfect:

- **Phase 1** — AST scan reports 0 dead imports in `bench_envs`, `setup_demo` definitions total 23
  (22 leaf variants plus `Bench_base_task`),
  and the Phase 1 gate passes.
- **Phase 2** — AST scan reports 0 dead imports in `bench_script` (excluding deliberate re-exports),
  and the CPU tests plus `--help` checks pass.
- **Phase 2.5** — imported-only modules have moved under `lib/`, apart from the intentional
  `setup_paths.py` bootstrap and the two Phase-4 modules; the AST dependency check returns 0 for
  `lib/` and at most 1 for `task/`.
- **Phase 2.6** — all 7 expected docs exist under `plans/`, all 7 old paths are absent, and every
  known referrer points at the new location.
- **Phase 3** — the explicit hygiene receipt below passes and the staged path list contains only
  `.gitignore`, `agent-memory/status_current.md`, and the two PNG index deletions.
- **Phase 4** — both report receipts are complete, all five frozen files are cleaned, both deferred
  modules have moved, their selftests/help checks pass, and the AST dependency check returns 0 for
  both `lib/` and `task/`.

Do not keep hunting for one more defect past these lines.

---

## Verification

No pytest is configured. Every block below sets its own working directory and fails nonzero; do
not replace failures with `|| echo`. Start each shell with:

```bash
export WORKSPACE_ROOT=/home/haccerkat/Documents/Research/Experimental/RoboPRO
source "$WORKSPACE_ROOT/customized_robotwin/set_env.sh"
export ROBOTWIN_BENCH_TASK=bench
export ROBOPRO_PY=/home/haccerkat/miniconda3/envs/RoboTwin/bin/python
test -x "$ROBOPRO_PY"
export PYTHONPATH="$BENCH_ROOT:$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORKSPACE_ROOT"
set -euo pipefail
```

### Prerequisite gates for Phases 1, 2, 2.5, 2.6, and 4

The metric completion requirement was waived. Before the first source edit, verify that its existing
partial data is still present and that the independent visualization audit is complete:

```bash
"$ROBOPRO_PY" - <<'PY'
import json
from pathlib import Path

run = Path("scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037")
metric = json.loads((run / "metric_postprocess/report_state.json").read_text())
visual = json.loads((run / "metric_route_visuals_v4/report_state.json").read_text())
assert metric["processing_complete"] is False, metric
assert metric["target_metrics"] == 3000, metric
assert 0 < metric["metrics_in_report"] < metric["target_metrics"], metric
assert visual["processing_complete"] is True, visual
assert visual["visualized_episodes"] == visual["target_episodes"] == 50, visual
assert visual["figure_count"] == 200, visual
episodes = sorted((run / "metric_postprocess/episodes").glob("episode*.json"))
assert len(episodes) == metric["metrics_in_report"], (len(episodes), metric)
print("partial metric data preserved; visualization prerequisite passed")
PY

# Run before making the first cleanup edit. CLEANUP_PLAN.md itself may be dirty while reviewed;
# every research source under these roots must otherwise already be committed.
dirty="$({ git status --porcelain -- benchmark/bench_envs customized_robotwin/script/bench_script \
  | grep -v 'customized_robotwin/script/bench_script/plans/CLEANUP_PLAN.md' || true; })"
test -z "$dirty" || { printf '%s\n' "$dirty"; exit 1; }
```

Before the first source edit, retain one last read-only proof that the pre-cleanup scene version
matches the manifest. It is expected to change after cleanup and is not a post-cleanup gate:

```bash
PYTHONPATH="$WORKSPACE_ROOT/customized_robotwin/script/bench_script" "$ROBOPRO_PY" - <<'PY'
import json
from pathlib import Path
from lib.scene_provenance import task_scene_code_version

manifest = Path(
    "scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/"
    "20260731-182037/metric_scene_manifest.jsonl"
)
row = json.loads(manifest.open(encoding="utf-8").readline())
current = task_scene_code_version(row["base_config"])
assert current == row["expected_scene_code_version"], (current, row["expected_scene_code_version"])
print("scene code version matches", current)
PY
```

**Dead-import scan (drives §1.1 and §2.1, and is the Phase 1 stopping condition):**

```bash
cd "$WORKSPACE_ROOT"
# Run once with benchmark/bench_envs after Phase 1, and once with
# customized_robotwin/script/bench_script after Phase 2.
"$ROBOPRO_PY" - benchmark/bench_envs <<'PY'
import ast, pathlib, sys
found = []
for root in sys.argv[1:]:
    for p in sorted(pathlib.Path(root).rglob("*.py")):
        if "__pycache__" in p.parts or p.name == "__init__.py":
            continue  # package initializers intentionally re-export names
        tree = ast.parse(src := p.read_text(encoding="utf-8"))
        imported = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names: imported[a.asname or a.name.split('.')[0]] = n.lineno
            elif isinstance(n, ast.ImportFrom) and n.module != "__future__":
                for a in n.names:
                    if a.name != '*': imported[a.asname or a.name] = n.lineno
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        dead = [k for k in imported if k not in used and f'"{k}"' not in src and f"'{k}'" not in src]
        if dead:
            found.append((p, dead))
for p, dead in found:
    print(f"{p}: {sorted(dead)}")
raise SystemExit(bool(found))
PY
```

This is a removal candidate scan, not proof by itself. During implementation, inspect each hit;
after each dead-import sub-phase it becomes a zero-output/nonzero-safety gate. If an intentional
non-`__init__` re-export exists, document and allowlist that exact `(file, name)` pair rather than
weakening the scanner globally.

**Phase 1 static and setup-dispatch gates:**

```bash
cd "$WORKSPACE_ROOT"
"$ROBOPRO_PY" -m compileall -q benchmark/bench_envs

# Structural inventory: after §1.2 there are 22 variant leaf methods plus the base method.
"$ROBOPRO_PY" - <<'PY'
import ast
from pathlib import Path
root = Path("benchmark/bench_envs")
defs = []
for path in root.rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defs.extend((path, n.lineno) for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == "setup_demo")
assert len(defs) == 23, defs
print("setup_demo definitions:", len(defs))
PY

# GPU/import environment required: import and construct exactly 20 leaf tasks in each scene family.
"$ROBOPRO_PY" - <<'PY'
import importlib
from pathlib import Path
families = ("office", "study", "kitchens", "kitchenl")
mods = []
for family in families:
    paths = sorted(p for p in Path("benchmark/bench_envs", family).glob("*.py")
                   if not p.name.startswith("_"))
    assert len(paths) == 20, (family, len(paths), paths)
    mods.extend((family, p.stem) for p in paths)
for family, stem in mods:
    module = importlib.import_module(f"bench_envs.{family}.{stem}")
    cls = getattr(module, stem)
    cls()
print("imported and constructed", len(mods), "leaf tasks")
PY

# The inherited base must preserve the old unconditional collision-cache assignment.
"$ROBOPRO_PY" - <<'PY'
from bench_envs._bench_base_task import Bench_base_task

class Probe(Bench_base_task):
    def _init_task_env_(self, **kwargs):
        self.received = kwargs

obj = object.__new__(Probe)
obj.setup_demo(collision_cache={"mesh": 1, "obb": 1}, marker=7)
assert obj.received["collision_cache"] == {"mesh": 100, "obb": 3}
assert obj.received["marker"] == 7
print("setup_demo dispatch and collision-cache overwrite passed")
PY
```

The import/construct check requires the real RoboTwin GPU environment because importing the robot
stack initializes cuRobo/CUDA. `compileall` is the CPU-safe fallback, not a substitute for this
gate.

**Phase 1 real-scene gate — run once immediately before §1.2 and once after Phase 1:**

```bash
cd "$ROBOTWIN_ROOT/script/bench_script"
"$ROBOPRO_PY" - <<'PY'
from setup_paths import setup_paths
setup_paths()
from lib.scene_build import build_cfg, get_env_class

cfg = build_cfg("put_mouse_on_pad", "bench_demo_clean", 0, {}, mode="measure")
env = get_env_class("put_mouse_on_pad", bench_subdir="office")()
try:
    env.setup_demo(**cfg)
    assert env.robot.collision_cache == {"mesh": 100, "obb": 3}
    env._update_render()
    env.cameras.update_picture(camera_names=["demo_camera"])
    rgb = env.cameras.get_rgb(camera_names=["demo_camera"])["demo_camera"]["rgb"]
    print("collision_cache:", env.robot.collision_cache)
    print("demo_camera shape:", rgb.shape, "pixel_sum:", float(rgb.sum()))
finally:
    env.close_env()
PY
```

Record the shape/pixel sum before and after and inspect a saved camera image if desired, but the
load-bearing assertion is the actual `env.robot.collision_cache` value. The old
`visualize_task_scene.py --task-name ...` command was invalid and that script does not emit the
claimed fingerprint.

**Phase 2 gate — the CPU tests and the import chain:**

```bash
cd "$ROBOTWIN_ROOT/script/bench_script"
for t in test_lib_env_api test_ring_config test_obstacle_set test_geometric_metric \
         test_metric_buckets test_task_metric test_compare_geometric_vs_gated \
         test_task_metric_route_visualization test_validate_reach_envelope \
         test_metric_correlation test_vla_reporting; do
  echo "== $t"
  "$ROBOPRO_PY" "$t.py"
done
# --help exercises each full import chain without a GPU
for s in analyze_occluder_visibility clearance_metric_3d seed_from_clearance vla_rollout \
         compare_geometric_vs_gated reach_envelope validate_reach_envelope reachability_map \
         visualize_task_scene; do
  "$ROBOPRO_PY" "$s.py" --help >/dev/null
done
```

`diag_kitchen_curobo.py --help` imports the task/robot stack before argparse and initializes
cuRobo/CUDA, so it is **not** part of the CPU-safe loop. Run it once in the GPU environment with
the Phase 1 import/construct and real-scene gates:

```bash
cd "$ROBOTWIN_ROOT/script/bench_script"
"$ROBOPRO_PY" diag_kitchen_curobo.py --help >/dev/null
```

**Phase 2 invariant — the dependency rule from `domain_bench_script_layout.md`:**

```bash
cd "$WORKSPACE_ROOT"
"$ROBOPRO_PY" - <<'PY'
import ast
from pathlib import Path

root = Path("customized_robotwin/script/bench_script")
top = {p.stem for p in root.glob("*.py")}
hits = []
for area in ("lib", "task"):
    for path in sorted((root / area).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in top:
                    hits.append((area, str(path), node.lineno, name))

for hit in hits:
    print(hit)
lib_hits = [h for h in hits if h[0] == "lib"]
task_hits = [h for h in hits if h[0] == "task"]
assert not lib_hits, lib_hits
# After Phase 2.5, only seeding_mixin -> seed_from_clearance remains.
assert {(Path(p).name, name) for _, p, _, name in task_hits} <= {
    ("seeding_mixin.py", "seed_from_clearance")
}, task_hits
PY
```

**Phase 2.5 gate — the moved modules still resolve, from the real invocation directory:**

```bash
cd "$ROBOTWIN_ROOT"
# every importer of a moved module still starts up
for s in clearance_metric_3d reachability_map swept_volume_3d; do
  "$ROBOPRO_PY" "script/bench_script/$s.py" --help >/dev/null
done
# the moved modules keep their runnable entry points
cd "$ROBOTWIN_ROOT/script/bench_script"
"$ROBOPRO_PY" -m lib.reachability_view --help >/dev/null
"$ROBOPRO_PY" -m lib.carry_object_spheres --selftest
"$ROBOPRO_PY" - <<'PY'
from pathlib import Path
from lib.reachability_view import RESULTS_DIR
expected = Path("/home/haccerkat/Documents/Research/Experimental/RoboPRO/scripts/validation/results/reachability")
assert RESULTS_DIR.resolve() == expected.resolve(), (RESULTS_DIR, expected)
print("reachability results path:", RESULTS_DIR)
PY
# nothing still imports them by their old flat name
cd "$WORKSPACE_ROOT"
if rg -n '^\s*(import|from)\s+(metric_diagnostics|reachability_view|carry_object_spheres)\b' \
    customized_robotwin/script/bench_script --glob '*.py' --glob '!**/__pycache__/**'; then
  exit 1
fi
```

**Phase 4 moved-module and final dependency gate:**

```bash
cd "$ROBOTWIN_ROOT/script/bench_script"
"$ROBOPRO_PY" -m lib.seed_from_clearance --selftest
"$ROBOPRO_PY" visualize_task_metric_routes.py --help >/dev/null
"$ROBOPRO_PY" clearance_metric_3d.py --help >/dev/null
"$ROBOPRO_PY" compare_geometric_vs_gated.py --help >/dev/null

cd "$WORKSPACE_ROOT"
"$ROBOPRO_PY" - <<'PY'
import ast
from pathlib import Path
root = Path("customized_robotwin/script/bench_script")
top = {p.stem for p in root.glob("*.py")}
hits = []
for area in ("lib", "task"):
    for path in sorted((root / area).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                names = []
            hits.extend((str(path), node.lineno, name) for name in names if name in top)
assert not hits, hits
print("lib/task -> top-level script dependencies: 0")
PY
```

**Phase 2.6 gate — no dangling doc references:**

```bash
cd "$WORKSPACE_ROOT"
docs=(
  CLEARANCE_2D_TO_3D.md REFACTOR_PLAN.md GEOMETRIC_EPS_VALIDATION_PLAN.md
  SEED_TRAJECTORY_PLAN.md TASK_METRIC_CORRELATION_PLAN.md VLA_ROLLOUT_PLAN.md
  TASK_METRIC_ROUTE_VISUALIZATION_PLAN.md
)
for name in "${docs[@]}"; do
  test -f "customized_robotwin/script/bench_script/plans/$name"
  test ! -e "customized_robotwin/script/bench_script/$name"
done
if rg -n -P 'bench_script/(?!plans/)[A-Z_]+\.md' . \
    --glob '*.py' --glob '*.sh' --glob '*.md' --glob '*.patch' \
    --glob '!graphify-out/**' --glob '!**/results/**'; then
  exit 1
fi
# Print every remaining name reference for human review; bare same-directory links inside plans are valid.
for name in "${docs[@]}"; do
  rg -n --fixed-strings "$name" customized_robotwin/script/bench_script agent-memory \
    --glob '!**/results/**' || true
done
```

**Phase 3 receipt:**

```bash
cd "$WORKSPACE_ROOT"
test ! -e "$WORKSPACE_ROOT/d"
test ! -e "$WORKSPACE_ROOT/tools"
stale_pyc=(
  customized_robotwin/script/bench_script/__pycache__/clearance_metric.cpython-310.pyc
  customized_robotwin/script/bench_script/__pycache__/experiment_free_target.cpython-310.pyc
  customized_robotwin/script/bench_script/__pycache__/gripper_path_3d.cpython-310.pyc
  customized_robotwin/script/bench_script/__pycache__/hamid_occluder_expert.cpython-310.pyc
  customized_robotwin/script/bench_script/__pycache__/pickup_reachability_map.cpython-310.pyc
  customized_robotwin/script/bench_script/__pycache__/subgoal_reachability_map.cpython-310.pyc
  scripts/validation/__pycache__/summarize_2x2_planner_comparison.cpython-310.pyc
)
for path in "${stale_pyc[@]}"; do test ! -e "$path"; done
git check-ignore -q graphify-out/probe
git check-ignore -q customized_robotwin/script/graphify-out/probe
pngs=(
  customized_robotwin/visibility_validation/put_mouse_on_pad_seed0_occluder_h0.06.png
  customized_robotwin/visibility_validation/put_mouse_on_pad_seed0_overlay.png
)
for path in "${pngs[@]}"; do
  test -f "$path"
  test -z "$(git ls-files -- "$path")"
  git check-ignore -q "$path"
done
expected=$(printf '%s\n' .gitignore agent-memory/status_current.md "${pngs[@]}" | sort)
actual=$(git diff --cached --name-only | sort)
test "$actual" = "$expected" || { printf 'expected:\n%s\nactual:\n%s\n' "$expected" "$actual"; exit 1; }
```

**Line-count receipt:**

```bash
cd "$WORKSPACE_ROOT"
find benchmark/bench_envs customized_robotwin/script/bench_script -name '*.py' \
  -not -path '*__pycache__*' -print0 | xargs -0 wc -l | tail -1
```
Baseline taken 2026-08-10, dirty tree: **37,906 total** (`bench_envs` 17,560). Record before and
after. The scheduled Phase 1–2 reduction is approximately **1,140 Python lines**, not 2,600;
relocations do not change the total, Phase 3 mostly removes non-Python artifacts, and §5 is excluded.

**Staged-commit gate — run before every cleanup commit:**

```bash
cd "$WORKSPACE_ROOT"
git diff --cached --check
git diff --cached --name-status
# Compare the printed paths with the numbered sub-phase's explicit scope. Abort on any research file
# or unrelated user change; do not rely on `git diff --name-only`, which misses staged/untracked state.
```

**After each phase lands, per `AGENTS.md`:**

```bash
graphify update .        # AST-only, no API cost; keeps graphify-out/ current
```

Then rewrite (do not append to) `agent-memory/status_current.md` with what landed, what is still
uncommitted, and which phase is next. Stage that memory change only with the phase it describes.
If a phase produced a durable, non-obvious lesson, file it in the existing topical note that covers
it — `domain_bench_script_layout.md` for anything in Phases 2.5/2.6 — rather than starting a new
file (`agent-memory/MEMORY.md` filing rules).
