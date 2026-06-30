"""robo_tools — RoboPRO data-collection tooling.

Consolidates the tools used to build the data-collection improvements:
  * core       — per-frame state enrichment + targeted negative-data desync
                 (TargetedRuntime, sampler, labels, HDF5 writers, SUPPORTED_TASKS)
  * multimodal — parameterize the planner for N distinct behaviors per scene
                 (MultimodalRuntime, MODES, MULTIMODAL_TASKS)
  * pipeline   — plan sibling variations + build the causal-graph annotation

`import robo_tools as rt` exposes the full library (drop-in for the old
`robo_negative`). The CLIs live in the repo-root `scripts/`.
"""
from .core import *  # noqa: F401,F403  (runtime, sampler, labels, io, registries, tests)
from . import core  # noqa: F401
from .multimodal import (  # noqa: F401
    MODES, MODE_ORDER, MultimodalRuntime, MULTIMODAL_TASKS, multimodal_entry)
from .pipeline import (  # noqa: F401
    plan_variations, build_causal_annotation, variations_enabled, supported_summary, clean_record)


def main() -> None:
    """`robo-tools` console entry — run the pure unit tests."""
    fns = [v for k, v in sorted(vars(core).items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
