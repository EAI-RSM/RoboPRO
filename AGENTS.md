## Agent memory

Durable working notes live in `agent-memory/`. **Read `agent-memory/MEMORY.md` at the start of a
session**, then open the individual files it points to when relevant — do not read all of them up
front.

- `agent-memory/status_current.md` is the only file holding volatile state (branch, uncommitted
  work, what is unverified, what is next). Read it before acting on anything time-sensitive, and
  **rewrite** it rather than appending when it changes.
- Everything else is durable knowledge: what is *not* derivable from the code — the reasoning
  behind decisions that look arbitrary in the diff, and gotchas that cost real debugging time.
- The filing rules at the top of `MEMORY.md` apply when adding anything.

Claude Code reads this same folder via `CLAUDE.md`, so treat a note you did not write as possibly
newer than your own context.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
