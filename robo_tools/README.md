# robo_tools

RoboPRO data-collection tooling — the library behind the enriched, extensible collection
pipeline (`customized_robotwin/script/collect_data.py`) and the repo-root `scripts/` CLIs.

| module | what |
|---|---|
| `core` | per-frame state **enrichment** (`TargetedRuntime`, the `targeted_state` logger) + the targeted **negative-data** desync (sampler, planner-belief override, `derive_outcome`/labels), HDF5 writers, `SUPPORTED_TASKS` |
| `multimodal` | `MultimodalRuntime` + `MODES` — parameterize the planner for N distinct successful behaviors per scene; `MULTIMODAL_TASKS` registry |
| `pipeline` | `plan_variations` (sibling variations per scene, registry-gated → skips unaudited tasks), `clean_record`, `build_causal_annotation` (intervention → mechanism → mediators → outcome) |

```python
import robo_tools as rt
rt.TargetedRuntime().attach(env)          # enrichment + (optional) perturbation hooks
rt.plan_variations(task, args, seed)      # what to collect for a scene (clean's siblings)
rt.build_causal_annotation(record)        # causal graph for a negative episode
```

Run the unit tests: `python -m robo_tools`. The single env-side hook the negative perturbations
need lives in `benchmark/bench_envs/_bench_base_task.py` (guarded by `env.targeted`; no-op when
absent). Registries are intentionally conservative — add a task after auditing it.
