# RoboPRO

## Agent memory

Durable working notes live in [agent-memory/](agent-memory/). The index is imported below, so it is
in context from the first turn; open the individual files it points to when they are relevant — do
not read all of them up front.

@agent-memory/MEMORY.md

- `agent-memory/status_current.md` is the only file holding volatile state (branch, uncommitted
  work, what is unverified, what is next). Read it before acting on anything time-sensitive, and
  **rewrite** it rather than appending when it changes.
- Everything else is durable knowledge. It records what is *not* derivable from the code — the
  reasoning behind decisions that look arbitrary in the diff, and gotchas that cost real debugging
  time.
- The filing rules at the top of `MEMORY.md` apply when adding anything.

This folder is shared with Codex (see [AGENTS.md](AGENTS.md)); both agents read and write it, so
treat a note you did not write as possibly newer than your own context.

## Environment

Benchmark scripts run from `customized_robotwin/` with `source set_env.sh` and
`export ROBOTWIN_BENCH_TASK=bench`. Details and exceptions: `agent-memory/repo_env_and_git.md`.
