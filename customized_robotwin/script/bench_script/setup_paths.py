#  Add paths to the system path
import sys
from pathlib import Path
import os


def setup_paths():
    # ROBOTWIN_ROOT / BENCH_ROOT are normally exported by sourcing set_env.sh.
    # If they are absent (e.g. set_env.sh wasn't sourced in this shell), derive
    # them from this file's location so the bench scripts still run. Explicit env
    # vars always win -- we only fill in the gaps.
    #   this file:       <robotwin_root>/script/bench_script/setup_paths.py
    #   robotwin_root  = parents[2]
    #   bench_root     = <robotwin_root>/../benchmark   (matches set_env.sh)
    here = Path(__file__).resolve()
    if "ROBOTWIN_ROOT" not in os.environ:
        os.environ["ROBOTWIN_ROOT"] = str(here.parents[2])
        print(f"[setup_paths] ROBOTWIN_ROOT not set; derived {os.environ['ROBOTWIN_ROOT']}")
    if "BENCH_ROOT" not in os.environ:
        os.environ["BENCH_ROOT"] = str(Path(os.environ["ROBOTWIN_ROOT"]).parent / "benchmark")
        print(f"[setup_paths] BENCH_ROOT not set; derived {os.environ['BENCH_ROOT']}")

    robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
    bench_root = Path(os.environ["BENCH_ROOT"])
    for p in [robotwin_root, bench_root]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        # if f"{robotwin_root}/script" not in sys.path:
        #     sys.path.insert(0, f"{robotwin_root}/script")
