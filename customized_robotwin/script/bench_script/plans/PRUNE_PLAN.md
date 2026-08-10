# Second pass: prune `bench_script` to its live surface

**Primary scope:** `customized_robotwin/script/bench_script/` only. This pass removes retired
research code from the actively developed folder; it does not refactor unrelated policy clients,
plotting scripts, `benchmark/bench_envs/`, vendored policies, or stock RoboTwin machinery.

Written by Claude Code and corrected after Codex review on 2026-08-10. It supersedes the first
`CLEANUP_PLAN.md` hygiene pass, already executed through `d7d001a`. Paths are repo-root-relative.
Read `agent-memory/MEMORY.md` and `agent-memory/status_current.md` before execution, preserve
unrelated worktree changes, and use one commit per numbered section.

## Outcome and honest accounting

Baseline: `bench_script/` contains 20,152 Python lines and 34 top-level Python files.

This pass has three distinct effects:

| Effect | Scope |
|---|---:|
| Delete four confirmed orphan entry points | 745 lines |
| Cold-archive retired task-metric implementation outside `bench_script/` | 4,812 code/spec lines, of which 4,760 are Python |
| Delete the retired slice's six tests and three stale plans | 3,248 lines |
| Detach the retired post-process hook and remove newly dead metric-only symbols | measure after execution |
| Relocate seven surviving checks under `checks/` | 1,251 lines, plus focused live-API coverage |

The cold archive remains tracked, so its 4,812 lines are not a repository-wide deletion. They are
removed from the active `bench_script` tree, top-level navigation, and live dependency surface.
The six obsolete tests and three stale plans are deleted from the current tree; Git history remains
their recovery mechanism.

Expected structural result:

- `bench_script/` top level: **34 Python files -> 10**.
- All top-level survivors are live commands except `setup_paths.py`, the intentional bootstrap.
- Active `bench_script` Python should land near 13,000 lines including `checks/`; record the actual
  count instead of treating an estimate as a gate. The exact total depends on the focused coverage
  transferred into `checks/`.
- Active code excluding `checks/` should be about 11,700 Python lines.

Do not claim that the repository shrank by the archive size. Report separately: deleted, cold
archived, relocated, and the measured active-tree before/after totals.

## Hard constraints

### Preserve all research data

The integrated metric post-process remains at 620/3000 records under:

`scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037/metric_postprocess/`

All 620 episode JSONs stay on disk. Cold-archiving `task_metric.py` retires compatibility with
in-place resumption, but it does **not** discard those records. Describe them as preserved data from
a retired predictor. The predictor has a measured validity problem (endpoint-pinned in 800/800
legs and Spearman 0.078 against grasp tightness), but the records retain provenance and diagnostic
value. Delete or rewrite nothing under `scripts/validation/results/`.

The source rollout campaign and completed 50-episode/200-figure route audit also remain untouched.

### The archive is cold source, not a supported package

Place the retired implementation at:

`research_archive/bench_script_task_metric_2026/`

This is outside `customized_robotwin/script/bench_script/`. Preserve the original relative shape so
the relationships are readable:

```text
research_archive/bench_script_task_metric_2026/
├── README.md
├── task_metric.py
├── compare_geometric_vs_gated.py
├── visualize_task_metric_routes.py
├── analyze_metric_correlation.py
├── validate_task_geometric_ranking.py
├── analyze_metric_distribution.py
├── validate_bucket_spec.py
├── bucket_spec.json
└── lib/
    ├── geometric_metric.py
    └── metric_buckets.py
```

Do not add `__init__.py`, repair imports, change source hashes, or promise that archived commands
run against the current tree. The purpose is to retain the core logic for inspection or later
recovery. `README.md` should record:

- the pre-prune commit;
- why the slice was retired;
- that imports are intentionally not maintained;
- the location of the preserved 620-record partial run;
- that Git history contains the former tests and plans.

## §1 - Delete four confirmed orphan entry points (745 lines)

Delete:

| File | Lines | Reason |
|---|---:|---|
| `customized_robotwin/script/bench_script/view_object.py` | 257 | No live caller or project entry point |
| `customized_robotwin/script/bench_script/analyze_natural_visibility.py` | 243 | No live caller; `analyze_occluder_visibility.py` remains the live `lib.visibility` consumer |
| `customized_robotwin/script/bench_script/validate_visibility_measurement.py` | 218 | One-off validator with no live caller |
| `customized_robotwin/script/bench_script/analyze_vla_rollouts.py` | 27 | Thin wrapper over reporting already called inline by `vla_rollout.py` |

Before deletion, re-run repo-wide references excluding `results/`, `graphify-out/`, and the files'
own usage docstrings. Historical references inside completed plans are allowed to remain historical;
live documentation and memory must not claim the commands still exist.

Update `agent-memory/repo_env_and_git.md`, which currently names
`analyze_natural_visibility.py` as a self-bootstrapping exception. Do not delete
`lib/visibility.py`: `analyze_occluder_visibility.py` imports it and remains a live Makefile entry.

Stopping gate for §1: all four files are absent; `analyze_occluder_visibility.py --help` and the
visibility import chain still start in the CPU environment; no unrelated path is staged.

## §2 - Retire the task-metric/geometric slice

### 2.1 Cold-archive only the implementation and spec (4,812 lines)

Use `git mv` to preserve history while moving these files to the cold archive layout above:

- `task_metric.py` (738)
- `compare_geometric_vs_gated.py` (1,238)
- `visualize_task_metric_routes.py` (879)
- `analyze_metric_correlation.py` (625)
- `validate_task_geometric_ranking.py` (406)
- `analyze_metric_distribution.py` (397)
- `validate_bucket_spec.py` (89)
- `bucket_spec.json` (52)
- `lib/geometric_metric.py` (237)
- `lib/metric_buckets.py` (151)

Do not archive the tests or old execution plans. They are not core logic.

### 2.2 Delete the retired tests (1,620 lines)

Delete:

- `test_metric_correlation.py`
- `test_task_metric_route_visualization.py`
- `test_compare_geometric_vs_gated.py`
- `test_task_metric.py`
- `test_geometric_metric.py`
- `test_metric_buckets.py`

Their recovery mechanism is Git history. In the same §2 commit, before deleting
`test_task_metric.py`, transfer focused coverage of the two APIs still used by `vla_rollout.py`
into the surviving top-level `test_vla_office_smoke.py`, following the coverage requirements in
§3. Do not transfer metric distribution, scene-image, fingerprint, or `canonical_legs` assertions.

### 2.3 Delete the retired plans (1,628 lines)

Delete:

- `plans/TASK_METRIC_CORRELATION_PLAN.md`
- `plans/GEOMETRIC_EPS_VALIDATION_PLAN.md`
- `plans/TASK_METRIC_ROUTE_VISUALIZATION_PLAN.md`

The durable scientific finding remains in `agent-memory/tool_task_metric_validity.md`; the deleted
plans are historical execution scaffolding, not current design documentation.

### 2.4 Detach the retired metric hook from live VLA rollout

In `vla_rollout.py`, remove:

- `_run_metric_postprocess`;
- the default-off `--postprocess-metrics` option;
- its task-only argument guard;
- both call sites.

In `test_vla_reporting.py`, remove `test_integrated_postprocess_command` and its `main()` call.
The pi05 launcher never passes this flag, so the rollout path and record schema remain unchanged.

### 2.5 Keep the shared live provenance/waypoint modules

Do not move:

- `lib/scene_provenance.py`
- `lib/task_roles.py`
- `lib/waypoints.py`

`vla_rollout.py` uses all three on its live `--scene task` path. In particular,
`task_scene_identity()` still calls `scene_id`, `scene_fingerprint`, and `actor_snapshot`; those
symbols are **not dead** and must remain.

After the metric slice moves, the only explicitly expected dead waypoint surface is
`CanonicalLeg`, `canonical_legs`, and `LEG_KINDS`. Remove those only after an exact live-source
reference check confirms that nothing outside the cold archive uses them. Do not use a generic
substring-count scan to authorize any additional deletion.

### 2.6 Repair every live reference

At minimum, review and update:

- `agent-memory/status_current.md`
- `agent-memory/domain_visibility.md`
- `agent-memory/domain_bench_script_layout.md`
- `agent-memory/tool_clearance_metric.md`
- `agent-memory/tool_geometric_metric.md`
- `agent-memory/tool_reach_envelope.md`
- `agent-memory/tool_route_visualizer.md`
- `agent-memory/tool_task_metric_validity.md`
- `agent-memory/repo_env_and_git.md`

Run a repo-wide search for every moved filename and deleted plan name. Historical statements may
point to the cold archive or explicitly say the source was retired; live instructions must not
point at missing paths.

Stopping gate for §2: none of the retired modules is imported or subprocess-invoked by active
`bench_script` code; the cold archive has the exact ten source/spec files plus README; all six tests
and three plans are absent from the active tree; all 620 partial records still exist.

## §3 - Move the seven surviving checks into `checks/`

Create `checks/__init__.py` and move:

- `test_vla_office_smoke.py`
- `test_obstacle_set.py`
- `test_vla_reporting.py`
- `smoke_test_seed_2a.py`
- `test_ring_config.py`
- `test_lib_env_api.py`
- `test_validate_reach_envelope.py`

Run CPU checks as `python -m checks.<module>` from `bench_script/`; pytest is not required.

Required path repairs:

- In `checks/test_lib_env_api.py`, set the bench root to
  `Path(__file__).resolve().parent.parent` so `lib/` and `task/` resolve from the active folder.
- In `checks/test_vla_office_smoke.py`, resolve and insert that same bench root rather than the
  `checks/` directory.
- Review every remaining `Path(__file__)` anchor after the move.

Required focused coverage, transferred in §2 before `test_task_metric.py` is deleted:

- Confirm the surviving check contains a small CPU case covering `resolve_task_roles()` and deterministic
  `canonical_waypoints()` for `put_cup_on_coaster`, because `vla_rollout.py` still uses both.
- Assert the returned acting arm used by `vla_rollout`, but do not import or exercise
  `canonical_legs`; that metric-only API should disappear in §2.
- Reuse only the minimum fake actor/pose/robot surface needed from the old
  `test_task_metric.py`. Do not retain metric reporting or artifact tests.

Update `agent-memory/domain_bench_script_layout.md` with the new module commands and verification
bar.

Stopping gate for §3: all six CPU-safe checks pass from `checks/`; the GPU-only seed smoke remains
present under `checks/` and its unrun CUDA requirement is recorded; no `test_*.py` or
`smoke_test_*.py` remains at top level.

## Explicitly out of scope

Do not include these in this bench-focused pass:

- `scripts/validation/`, `scripts/upload/`, or `scripts/slurm/` orphan cleanup;
- `customized_robotwin/script/{eval_policy_client,collect_rollout_client}.py` ModelClient rewrites;
- figure-helper refactors in `summarize_approach_mode_ab.py`;
- `benchmark/bench_envs/` rewrites;
- vendored policy code or stock RoboTwin machinery;
- `bench_task_config/` deduplication;
- `eval_policy.py` / `eval_policy_client.py` consolidation.

These do not reduce the active `bench_script` surface and introduce independent behavior or
verification risks. Track them in a separate proposal if they remain desirable.

Open user decision, not part of this plan: `benchmark/bench_script/` contains three unwired files
and 893 lines. Do not delete or rename that directory without a separate explicit instruction.

## Verification

Every shell starts from the repository root and fails on the first error:

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

### Pre-edit preservation receipt

```bash
"$ROBOPRO_PY" - <<'PY'
import json
from pathlib import Path

run = Path("scripts/validation/results/task_metric_vla_full/association_d6_d10_d15/20260731-182037")
metric = json.loads((run / "metric_postprocess/report_state.json").read_text())
visual = json.loads((run / "metric_route_visuals_v4/report_state.json").read_text())
episodes = sorted((run / "metric_postprocess/episodes").glob("episode*.json"))
figures = sorted((run / "metric_route_visuals_v4/figures").glob("*.png"))
assert metric["processing_complete"] is False
assert metric["metrics_in_report"] == len(episodes) == 620
assert metric["target_metrics"] == 3000
assert visual["processing_complete"] is True
assert visual["visualized_episodes"] == visual["target_episodes"] == 50
assert visual["figure_count"] == len(figures) == 200
print("preserved metric records:", len(episodes))
print("completed route figures:", len(figures))
PY
```

### Compilation and active dependency gates

```bash
"$ROBOPRO_PY" -m compileall -q \
  customized_robotwin/script/bench_script \
  research_archive/bench_script_task_metric_2026

# No active import or subprocess path may name the retired modules.
if rg -n '(^\s*(from|import)\s+(task_metric|analyze_metric_|compare_geometric_|visualize_task_metric_|validate_task_)\b|task_metric\.py)' \
  customized_robotwin/script/bench_script --glob '*.py' --glob '!checks/**'; then
  exit 1
fi

# Preserve the lib/task -> top-level dependency rule.
"$ROBOPRO_PY" - <<'PY'
import ast
from pathlib import Path

root = Path("customized_robotwin/script/bench_script")
top = {path.stem for path in root.glob("*.py")}
hits = []
for area in ("lib", "task"):
    for path in sorted((root / area).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                names = []
            hits.extend((path, node.lineno, name) for name in names if name in top)
assert not hits, hits
print("lib/task -> top-level script dependencies: 0")
PY
```

### CPU-safe live entry points

```bash
cd "$ROBOTWIN_ROOT/script/bench_script"
for script in visualize_task_scene analyze_occluder_visibility reachability_map \
              vla_rollout clearance_metric_3d reach_envelope \
              validate_reach_envelope swept_volume_3d; do
  "$ROBOPRO_PY" "$script.py" --help >/dev/null
done
```

`diag_kitchen_curobo.py --help` initializes cuRobo/CUDA and is not a CPU gate. Preserve it as a
top-level entry point, compile it, and record the missing GPU check if CUDA remains unavailable.

### Surviving checks

```bash
cd "$ROBOTWIN_ROOT/script/bench_script"
for check in test_lib_env_api test_ring_config test_obstacle_set test_vla_reporting \
             test_vla_office_smoke test_validate_reach_envelope; do
  "$ROBOPRO_PY" -m "checks.$check"
done
```

`checks.smoke_test_seed_2a` remains the GPU-only seed-pipeline smoke. Do not claim it passed unless
it runs in a CUDA environment.

### Exact survivor and line-count receipts

```bash
cd "$WORKSPACE_ROOT"
expected=$(printf '%s\n' \
  analyze_occluder_visibility.py clearance_metric_3d.py diag_kitchen_curobo.py \
  reach_envelope.py reachability_map.py setup_paths.py swept_volume_3d.py \
  validate_reach_envelope.py visualize_task_scene.py vla_rollout.py | sort)
actual=$(find customized_robotwin/script/bench_script -maxdepth 1 -type f -name '*.py' \
  -printf '%f\n' | sort)
test "$actual" = "$expected"

find customized_robotwin/script/bench_script -name '*.py' \
  -not -path '*__pycache__*' -print0 | xargs -0 wc -l | tail -1
find customized_robotwin/script/bench_script -name '*.py' \
  -not -path '*__pycache__*' -not -path '*/checks/*' -print0 | xargs -0 wc -l | tail -1
find research_archive/bench_script_task_metric_2026 -type f -print0 | xargs -0 wc -l | tail -1
```

Re-run the preservation receipt after all edits. Also run `git diff --check` and inspect
`git diff --cached --name-status` before every commit; unrelated changes such as concurrent memory
work must remain unstaged.

After the final code commit, run `graphify update .`, query the retired-module and active-dependency
relationships once more, and rewrite `agent-memory/status_current.md` with:

- the cold archive location and pre-prune commit;
- the exact active-tree and deleted line counts;
- the preserved 620-record partial run and retired in-place resume path;
- the checks that passed;
- any remaining CUDA-only verification gap.

**Final stopping condition:** §§1-3 are committed independently, every available gate above passes,
the exact top-level survivor list matches, all research data is preserved, and no unrelated cleanup
is added.
