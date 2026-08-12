# S5 — bench_envs mechanical cleanup, as a scripted pass

Detailed technical plan for `REHAUL_PLAN.md` § S5. On approval, copy this file to
`customized_robotwin/script/bench_script/plans/S5_BENCH_ENVS_CLEANUP_PLAN.md` and link it from the
S5 section, matching how S2/S3/S4 are recorded.

**Working branch: `peng-dev-new`** (per the REHAUL_PLAN branch table). Depends on S3 only;
independent of S6–S12. No GPU run required.

**Status: completed 2026-08-12.** The implementation used current `origin/dev@600089d`; the two
commits added after the plan's pinned `64840ce` baseline did not touch `benchmark/bench_envs`.

## Execution record

- Tool and permanent import gate: `7b86360`. The isolated subprocess gate prevents its fake
  `envs.robot` module from contaminating other pytest modules.
- Local transforms, in required order: T1 `1cc381f` (58 canonical overrides), T4 `aa58e03`
  (13 dead defs), T2 `12b55fc` (347 import-statement edits in 77 files), T3 `ba83c6a` (18 comment
  blocks), and T5 `d7a8609` (one reader plus three writers). Shared-tree total: **80 files,
  +34/−880**.
- Full local gate: cleanup check reported `no changes`; manifest subtraction compared 92 modules;
  all 160 leaf `check_success`/`play_once` functions matched current dev by AST; imports were
  **95/0**; pytest was **51 passed, 1 skipped**; all 20 top-level commands passed `--help`; the old
  typo spelling is absent from Python code.
- Upstream branch `cleanup/bench-envs-mechanical` starts from `600089d`. Its first commit is the
  required base hook (`266091e`), followed by T1–T5 (`e728bd5`, `ac1a118`, `5075952`, `3f3a988`,
  `c8bf785`). Its dev tree imports **94/0** because the research-only `eval_video.py` is not on dev.
  PR: [EAI-RSM/RoboPRO#72](https://github.com/EAI-RSM/RoboPRO/pull/72), targeting `dev`.

No rollout was run, as planned. The AST and import proofs do not establish SAPIEN runtime
equivalence; that limitation remains exactly as stated below.

---

## Context

`benchmark/bench_envs/` holds 95 task modules that `origin/dev` owns and actively maintains — dev
made 22 commits touching this directory in the six weeks since the fork, including six weeks of
`check_success` false-positive repairs. The pre-rehaul branch carried a systematic cleanup of these
files (net +265 / −1,000 across 88 files). S3 deliberately did **not** port those files: taking ours
would have discarded dev's repairs.

S5 re-derives the cleanup as a **scripted pass on top of dev's current versions**, so dev's work is
preserved by construction rather than by careful merging. It serves rehaul goal #3 (dead weight
removal); it delivers nothing functional to the research layer, so it must not consume risk budget.

Per the fork policy (REHAUL_PLAN Part 7), the result also goes upstream as a PR — that is what makes
the divergence on 88 shared files temporary rather than a permanent merge tax.

### Five corrections to the S5 summary, all measured

The one-paragraph S5 description in REHAUL_PLAN.md is imprecise in ways that change the work:

1. **`origin/dev` has no `setup_demo` at all.** Not in `benchmark/bench_envs/_bench_base_task.py`,
   not in `customized_robotwin/envs/_base_task.py`. The base hook is ours, added by S3 (`f5271af`,
   4 lines at `_bench_base_task.py:95`). Locally the deletions are safe. **Upstream they are not:
   the PR must carry the hook as its first hunk, or all 58 tasks break with `AttributeError`.**
2. **"Delete the override in 80 files" is really 58 + 22.** An AST classification finds 58 overrides
   that are *exactly* the canonical two-statement boilerplate, and 22 that carry real content
   (model ids, spawn rotations, scales, `set_fridge_open()`, `_ensure_cabinet_open()`). The original
   branch rewrote those 22 to `super().setup_demo(**kwargs)`, saving one line each.
   **Decided: the 22 are out of scope** — a semantic edit on shared files for ~22 lines.
3. **"Four dead per-task helpers" is 13 defs across 6 files**, including six drawer/model-id methods
   on the shared base `kitchenl/_kitchen_base_large.py`. All 13 verified to have zero references
   repo-wide on `peng-dev-new`, but the count and the base-class exposure were not recorded.
4. **The `include_collison` typo is consistent on dev, not broken.** Dev's
   `study/_study_base_task.py:147` *reads* the typo spelling and its three call sites *write* it, so
   it works. Our carried "shim" reads both. The fix is therefore a pure rename that must be
   **atomic across the reader and all three writers** — miss one and that task silently flips to the
   `True` default. Two of the three pass `False`.
5. **Whitespace normalisation is not a category.** `git diff -w` shows only 42 of the original 1,000
   deleted lines were whitespace — incidental, not a reformat. Dropped, to keep the shared-file diff
   free of pure noise.

Also confirmed safe: all 80 task classes use single inheritance from a `*_base_task`; no leaf task
and no `bench_script/task/` mixin defines `setup_demo` or `_init_task_env_`. So deleting an override
routes `setup_demo` to `Bench_base_task.setup_demo`, whose `self._init_task_env_(**kwargs)` resolves
by MRO to the same method the deleted `super()._init_task_env_(**kwargs)` reached. Equivalent.

---

## Scope

Five transforms. Each is independently gated, independently committed, and independently
cherry-pickable.

| # | Transform | Files | ~Lines | Proof it is safe |
|---|---|---|---|---|
| T1 | Delete pure-boilerplate `setup_demo` overrides | 58 | −230 | AST-exact body match; MRO equivalence above |
| T2 | Prune unused imports | 77 | −300 | AST unused **and** raw-text absent (double gate) |
| T3 | Delete commented-out `self.info["info"]` blocks | 18 | −130 | comments are absent from the AST; `play_once` AST provably unchanged |
| T4 | Delete 13 dead defs | 6 | −200 | zero repo-wide references, re-verified at run time |
| T5 | `include_collison` → `include_collision` | 4 | ±4 | atomic reader+writer rename, same default |

Expect roughly **−850 lines net** across ~85 files.

**Out of scope.**
- The 22 content-bearing `setup_demo` overrides (decision above).
- Trailing-whitespace normalisation (correction 5).
- `study/put_cup_on_coaster.py`'s `abs` bug — REHAUL_PLAN already excludes it; dev's fix is
  equivalent. T1–T4 may still touch that file's *header*; only `check_success` is off limits.
- **Any edit inside a `check_success` or `play_once` body.** T3 removes comment lines that sit at the
  tail of `play_once`, which is why the equivalence check below is AST-based and not textual.
- Anything under `customized_robotwin/` — S5 does not touch the research layer.

---

## The tool

One new file, exclusive to this branch, zero merge tax:

`customized_robotwin/script/bench_script/tools/bench_envs_cleanup.py` (plus `tools/__init__.py`)

Stdlib `ast` only — no linter is installed in `.venv` and S5 must not add a dependency (that would
touch shared `pyproject.toml`/`uv.lock`). It is deliberately a *tool*, not a 15th root entry point;
REHAUL_PLAN §4e already flags entry-point count as the thing making this tree feel heavy.

```
python -m tools.bench_envs_cleanup --check                 # report only; exit 1 if anything would change
python -m tools.bench_envs_cleanup --apply --only T1       # apply one transform
python -m tools.bench_envs_cleanup --verify <git-ref>      # AST-equivalence proof vs a ref
```

Transform rules, stated precisely so the implementer does not have to infer them:

- **T1** — delete a `FunctionDef` named `setup_demo` only when `ast.unparse(node.args)` is exactly
  `self, is_test=False, **kwargs` **and** the unparsed body is exactly
  `["kwargs['collision_cache'] = {'mesh': 100, 'obb': 3}", "super()._init_task_env_(**kwargs)"]`.
  Never touch `_bench_base_task.py`. Any other shape is skipped and reported, not guessed at.
- **T2** — for each module-level `Import`/`ImportFrom` alias, drop the alias when the bound name
  appears in **no** `ast.Name` node anywhere in the module **and** a `\bNAME\b` regex over the file
  with all import statements masked finds zero hits. The second gate exists because identifiers can
  hide in string annotations, which the AST scan would miss. Never remove `from X import *`
  (87 of 95 files have one). If a statement loses every alias, delete the line; otherwise rewrite it
  preserving the surviving names in their original order.
- **T3** — delete a maximal contiguous run of comment-only lines that contains
  `# self.info["info"]` and terminates at `# return self.info`, together with an immediately
  preceding `# Record information about ...` line. All 18 instances are in `office/` and all
  terminate at that sentinel.
- **T4** — delete these 13 defs, refusing if a repo-wide scan finds any reference at run time:
  `_kitchen_base_large.py`: `_entity_aabb`, `_init_drawer_states`, `set_drawer_open`,
  `set_drawer_closed`, `is_drawer_open`, `_sample_model_id`;
  `pick_boxdrink_from_basket.py`: `_world_point_in_entity_local`, `_ee_pose_above_place_target`;
  `put_milk_box_in_fridge.py`: `_fridge_inside_target_pose`;
  `put_sauce_can_in_cabinet.py`: `_cabinet_inside_target_pose`;
  `utils/create_actor_custom.py`: `create_multiple_obj_actor`;
  `utils/scene_gen_utils.py`: `get_random_valid_placement`, `get_obj_new_pose`.
- **T5** — not part of the AST tool. A 4-line hand edit: `study/_study_base_task.py:147-149` reader
  and the three writers in `move_book_onto_table.py`, `move_seal_onto_book.py`, `put_cup_in_box.py`.
  After the rename the two-key shim is dead; collapse it to a single
  `kwags.get("include_collision", True)`, which brings that line back to dev's shape modulo spelling.

Order matters: run T4 before T2, so imports orphaned by a deleted helper are then caught by T2.

---

## Execution

**Step 0.** Copy this file to `plans/S5_BENCH_ENVS_CLEANUP_PLAN.md`; link it from REHAUL_PLAN's S5
section the way S2/S3/S4 are linked.

**Step 1 — tool + permanent gate (exclusive files).** Write
`tools/bench_envs_cleanup.py` and `checks/test_bench_envs_import.py`. The latter imports all 95
`bench_envs` modules and asserts zero failures; `conftest.py`'s `setup_paths()` already puts
`benchmark/` on `sys.path`, so no new bootstrap is needed. Measured today: **95 imported, 0 failed**,
CPU-only. This is the gate that would have caught the `a301e2e` class of bug.
Run `--check` and record the manifest (which file, which node, which line) as the baseline artifact.
**→ COMMIT** *"Add bench_envs cleanup tool and import gate"* — exclusive files only, so it is
separable from every shared hunk that follows.

**Step 2 — T1, 58 boilerplate overrides.** `--apply --only T1`, then `--verify HEAD~1`.
**→ COMMIT** *"Drop boilerplate setup_demo overrides in bench_envs"*

**Step 3 — T4, 13 dead defs.** Before T2, per the ordering note.
**→ COMMIT** *"Remove unreferenced bench_envs helpers"*

**Step 4 — T2, unused imports.** The largest and most mechanical hunk.
**→ COMMIT** *"Prune unused imports in bench_envs"*

**Step 5 — T3, commented `self.info` blocks.**
**→ COMMIT** *"Remove commented-out info blocks in office tasks"*

**Step 6 — T5, the typo rename.** All four edits in one commit; verify no `include_collison`
remains repo-wide.
**→ COMMIT** *"Rename include_collison to include_collision"*

**Step 7 — upstream PR.** Cut `cleanup/bench-envs-mechanical` from `origin/dev@64840ce`. First
commit is the 4-line `setup_demo` hook from `_bench_base_task.py` (correction 1 — without it the
next commit breaks 58 tasks). Then replay steps 2–6 by running the same tool on that tree.
Determine the T4 dead-symbol list on `peng-dev-new`, not on the dev branch: `peng-dev-new`'s tree is
a superset, so it finds strictly more references and is the conservative direction. Open the PR
against `dev`; do not merge it into `peng-dev-new`. **→ COMMIT + push.**

Once dev merges, `peng-dev-new`'s divergence on these 88 files collapses to zero at the next weekly
merge — that is the whole point of doing it this way.

---

## Test plan

### 1. Universal regression gate

From `customized_robotwin/script/bench_script/`, using the `.venv` interpreter explicitly:

```bash
../../../.venv/bin/python -m pytest -q
```

S4's baseline is **50 passed, 1 skipped**. Step 1 adds `test_bench_envs_import`, so from Step 1
onward the expected result is **51 passed, 1 skipped** (`test_seed_2a_smoke` remains the only
GPU-marked skip). Any lower pass count means S5 broke something. Per the triage rule, check a
failure against `pre-rehaul-2026-08-11` before debugging it.

### 2. Section-specific commands

```bash
cd customized_robotwin && source set_env.sh && export ROBOTWIN_BENCH_TASK=bench && cd script/bench_script

# every bench_envs module still imports, CPU only
../../../.venv/bin/python -m checks.test_bench_envs_import      # expect: 95 imported, 0 failed

# the tool is idempotent: a second pass finds nothing left to do
../../../.venv/bin/python -m tools.bench_envs_cleanup --check   # expect: exit 0, "no changes"

# every top-level research command still imports through the changed base classes
for f in ./*.py; do ../../../.venv/bin/python "$f" --help >/dev/null || echo "FAIL $f"; done

# the typo is gone, and nothing else answers to the old spelling
grep -rn "include_collison" ../../../benchmark ../../../customized_robotwin   # expect: empty

# the diff is subtraction, not reformatting
git diff --stat -w 64840ce HEAD -- ../../../benchmark/bench_envs   # ~85 files, ≈ −850, few insertions
```

### 3. Equivalence check (this section is NO-BEHAVIOUR-CHANGE)

Two checks. The second is the one that matters.

**a. Manifest-subtraction proof.** `--verify <ref>` parses each file at `<ref>` and at HEAD, applies
the recorded removal manifest to the *before* tree, and asserts
`ast.dump(before_minus_manifest) == ast.dump(after)`. Passing means the only differences are the
ones the tool declared it would make. Any unintended edit — a mangled import rewrite, a comment
deletion that swallowed a statement — fails here.

**b. Dev's six weeks of repairs are byte-identical.** For every `check_success` and `play_once`
FunctionDef in all 95 modules, assert `ast.dump()` equals `origin/dev@64840ce`'s. This directly
enforces the REHAUL_PLAN constraint and is stronger than a textual guard, since T3 legitimately
deletes comment lines from inside `play_once` bodies — comments are absent from the AST, so this
check confirms the *code* is untouched while permitting the comment removal.

Expected: both pass with zero differences. Neither needs SAPIEN or a GPU; both run in seconds.

### 4. What S5 cannot verify — stated plainly

- **No rollout is run.** Nothing here proves any task still *behaves* identically at runtime.
  The AST proof covers the code; it does not cover SAPIEN. `feedback_scientific_rigor.md`'s
  2026-07-30 lesson applies: a green suite is not delivery confirmation.
- **The import gate only exercises module scope.** A name referenced solely inside a method body
  that no test calls is caught by T2's AST+text double gate, not by any executed code path.
- **String-dispatch references are invisible.** T4's dead-symbol scan is textual; a call through
  `getattr(env, "set_drawer_open")` or a name assembled at run time would not be found. All 13
  symbols are private-ish helpers on kitchenl tasks that the research layer never touches, so the
  exposure is low — but it is real and unmeasured.
- **T5 is unverifiable here.** The rename is behaviour-neutral by inspection (same key, same
  default, atomic), but the three affected study tasks are outside the research line and no CPU
  assertion exercises `if self.incl_collision:`. If that is not acceptable, drop T5 from S5 and
  file it upstream as an issue instead — it is one commit, cleanly separable.
- **Upstream acceptance is not in this section's control.** Until dev merges the PR, the 88-file
  divergence exists on `peng-dev-new` and every weekly merge pays it.

---

## Done when

All five hold, and then S5 stops — no further defects in `bench_envs` are in scope:

1. `--check` reports no remaining changes (idempotent).
2. `pytest -q` reports **51 passed, 1 skipped**, and `test_bench_envs_import` shows 95/0.
3. Both equivalence checks pass: manifest-subtraction clean, and every `check_success` / `play_once`
   AST byte-identical to `origin/dev@64840ce`.
4. `--help` passes on every top-level `bench_script/` command.
5. The upstream PR is open against `dev` with the `setup_demo` hook as its first commit.
