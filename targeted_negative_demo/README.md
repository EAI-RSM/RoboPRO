# Targeted Negative Data — demo

A management-facing showcase of `robo_negative`'s **causally-labelled failure data**: take a
task the scripted expert solves cleanly, then desync the planner's belief from the true scene by
one controlled amount. Because we know exactly what we changed, the failure is labelled by
construction. The same generic perturbation code runs on every task — nothing is hand-tuned.

Each task gets a clean **baseline** plus the **4 perturbations**, all in the same clean scene:

| ptype | what it does | typical outcome |
|---|---|---|
| `shift_object`   | move the target a few cm **after** its grasp is planned | grasp / plan failure |
| `shift_target`   | move the destination before placement | placement miss |
| `shift_obstacle` | a corridor obstacle the planner believes is elsewhere (mislocated) | collision / disruption |
| `hide_obstacle`  | a corridor obstacle hidden from the planner (undetected) | collision / disruption |

The obstacle perturbations **inject their own** corridor obstacle (a heavy milk-box placed at the
clearance-maximising point of the grasp→place corridor), so they need no per-task obstacle code.
Whether they produce a *visible* failure depends on the task's arm trajectory — low/flat transports
collide, high-lift/container transports clear the obstacle. The UI shows the **true** outcome of
every cell (failure / absorbed), never forcing a failure.

## Run it

```bash
# 1. collect (5 episodes/task, isolated subprocesses, parallel over GPUs) — writes
#    ../enriched_negative_demo/runs/<task>/{baseline,shift_*,hide_obstacle}/
python collect_negative_demo.py --tasks-file tasks.json --gpus 0,1,2,3,4,5,6,7 \
    --out-root ../enriched_negative_demo
#    tasks.json = [["put_book_on_book","office"], ["drop_apple_in_bin_ks","kitchens"], ...]
#    (tasks must be in robo_negative.SUPPORTED_TASKS)

# 2. build the web bundle (copies videos + writes web/manifest.json)
python build_demo.py

# 3. serve (threaded; manifest is no-cache so a rebuild shows on reload)
python serve.py 8001     # -> http://localhost:8001
```

Collected data (`../enriched_negative_demo/`) and the generated bundle (`web/videos/`,
`web/manifest.json`) are git-ignored — regenerate with the steps above.

## Layout

```
collect_negative_demo.py   orchestrator: per task, baseline + 4 perturbations via run_targeted_episode.py
build_demo.py              run tree -> web/videos + web/manifest.json
serve.py                   threaded static server
web/index.html             the showcase (fetches manifest.json; videos + causal labels)
```

The perturbation machinery lives in [`robo_negative`](../robo_negative/) (the injector
`TargetedRuntime.inject_corridor_obstacle`, `believe_displaced`, the task registry) and the
episode runner [`run_targeted_episode.py`](../run_targeted_episode.py).
