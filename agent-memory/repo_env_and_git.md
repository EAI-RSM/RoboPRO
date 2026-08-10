---
name: repo_env_and_git
description: "Required env exports before any benchmark script, plus git remote and PR-base conventions"
metadata:
  type: project
---

**Env setup.** All benchmark commands run from `customized_robotwin/`:

```bash
cd customized_robotwin
source set_env.sh                  # sets BENCH_ROOT + ROBOTWIN_ROOT
export ROBOTWIN_BENCH_TASK=bench   # routes loaders to benchmark paths
```

Without `ROBOTWIN_BENCH_TASK=bench` the system silently uses upstream RoboTwin configs instead of
benchmark configs. Scripts in `scripts/validation/` expect invocation from `customized_robotwin/`
with paths like `../scripts/validation/foo.sh`.

*Exception:* `analyze_occluder_visibility.py` self-bootstraps — `setup_paths()` derives
ROBOTWIN_ROOT/BENCH_ROOT from the file location if unset, and the script uses `setdefault` for
ROBOTWIN_BENCH_TASK=bench. Every other script still needs the exports.

**Git remote:** `git@github.com:EAI-RSM/RoboPRO.git`. The local clone sometimes shows
`HaccerKat/EAI-RSM/RoboPRO`, which is invalid (three path segments) and fails push with "not a
valid repository name". Fix with `git remote set-url origin git@github.com:EAI-RSM/RoboPRO.git`.

**PRs target `8-research-branch`, not `dev`** — `dev` is only the GitHub default. Use
`gh pr create --base 8-research-branch`. The team uses stacked PRs: each sub-issue gets its own
branch, PR'd into its *dependency's* branch rather than straight into `8-research-branch`; merge
bottom-up and GitHub auto-retargets the rest. GitHub issue and PR numbers share ONE sequence
(creating a PR consumes the next number) — this surprised the user once.

Note (2026-07-29): active work sits on local `peng-research-branch` with no open PR.
