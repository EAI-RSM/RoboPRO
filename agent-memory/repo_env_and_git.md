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

**THE STALE-REF TRAP (cost a whole analysis, 2026-08-11).** Local `dev` is not `origin/dev`. Local
`dev` sat two months behind at `4b57b67` (2026-06-02) while `origin/dev` was at `64840ce`
(2026-08-06). A comparison against local `dev` reported the research branch as **0 commits behind**
and concluded there was nothing to merge — the exact opposite of the truth. **Always
`git fetch` and compare against `origin/dev`**, and state which ref you used.

Measured divergence of the research line from `origin/dev` (2026-08-11):

- merge base `aabeff4`, 2026-06-25; **82 ahead / 60 behind**
- 208 files changed on our side (+26,969), 460 on dev's (+13,996); 37 files touched by both
- `git merge-tree` gives **9 real conflicts**, 2 of them add/add
- Of our 208 changed files, **106 are shared with dev (+2,135/−1,116 — the entire merge tax) and
  102 are exclusive (+24,834 — zero merge cost, nobody else touches them)**
- Most of the shared work is **already upstream**: `8-research-branch` merged as PR #57 and the
  visibility work re-landed as PR #56, so genuine carry-over is **~150 lines**. Several of our
  copies are now *behind* dev — `envs/camera/camera.py` in particular is a regression (dev moved
  segmentation to raw `uint16` ids; ours still has the `ImageColor` palette).

**The structural rule that follows:** merge cost is driven by how many *shared* files research code
edits, not by how much code it has. Keep research work in exclusive files; send shared-file changes
to dev as PRs.

Note (2026-08-11): active work sits on local `peng-training-branch`, **no remote tracking branch and
64 of its 82 commits exist only on this machine**. Tag `pre-rehaul-2026-08-11` marks tip `e3a09ce`.
