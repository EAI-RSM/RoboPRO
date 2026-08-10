#!/usr/bin/env python3
"""Regenerate descriptive VLA rollout tables and figures from a run directory."""

import argparse
from pathlib import Path

from setup_paths import setup_paths

setup_paths()

from lib.vla_reporting import write_rollout_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="VLA run directory containing episode records")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    summary = write_rollout_reports(run_dir)
    print(
        f"[report] {summary['n_episodes']} episodes, "
        f"HSR={summary['hard_success_rate']} -> {run_dir}"
    )


if __name__ == "__main__":
    main()
