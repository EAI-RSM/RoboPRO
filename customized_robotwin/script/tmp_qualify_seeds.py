#!/usr/bin/env python
"""Qualify scene seeds with the collector's own expert-solvability check.

Replicates collect_rollout_client's gate exactly: setup_demo(need_plan=True)
-> play_once() -> plan_success and check_success(). Walks a candidate block
until `want` seeds pass; the block layout guarantees densities can never
overlap no matter how many candidates get rejected.

    python script/tmp_qualify_seeds.py --task drop_apple_in_bin_ks \
        --density d6 --start 80100 --want 10 --out manifest.json
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
from script.bench_script.setup_paths import setup_paths  # noqa: E402

setup_paths()
sys.path.insert(0, "script")
from replay_utils import build_args, make_task, task_environment  # noqa: E402

try:
    from envs.utils.error import UnStableError  # matches the collector's except
except Exception:                               # pragma: no cover
    class UnStableError(Exception):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--density", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--want", type=int, required=True)
    ap.add_argument("--limit", type=int, default=100, help="block width")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out_p = Path(a.out)
    if out_p.exists():
        m = json.loads(out_p.read_text())
        have = m.get(a.task, {}).get(a.density, [])
        if len(have) >= a.want:
            print(f"[qualify] {a.task} {a.density}: already qualified ({len(have)}) -- skip", flush=True)
            return
    cfg = f"bench_demo_{task_environment(a.task)}_{a.density}"
    args = build_args(a.task, cfg)
    task = make_task(a.task)
    passed, tried = [], 0
    for seed in range(a.start, a.start + a.limit):
        if len(passed) >= a.want:
            break
        tried += 1
        try:
            chk = dict(args)
            chk["need_plan"] = True
            chk["save_data"] = False
            task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **chk)
            task.play_once()
            ok = bool(task.plan_success and task.check_success())
            task.close_env()
        except UnStableError:
            task.close_env()
            ok = False
        except Exception:
            traceback.print_exc()
            try:
                task.close_env()
            except Exception:
                pass
            ok = False
        print(f"[qualify] {a.task} {a.density} seed {seed}: {'PASS' if ok else 'fail'}"
              f"  ({len(passed) + int(ok)}/{a.want})", flush=True)
        if ok:
            passed.append(seed)

    out = Path(a.out)
    m = json.loads(out.read_text()) if out.exists() else {}
    m.setdefault(a.task, {})[a.density] = passed
    out.write_text(json.dumps(m, indent=2, sort_keys=True))
    print(f"[qualify] {a.task} {a.density}: {len(passed)}/{a.want} qualified "
          f"from {tried} candidates -> {out}", flush=True)
    if len(passed) < a.want:
        print(f"[qualify] WARNING: block exhausted before reaching {a.want}", flush=True)


if __name__ == "__main__":
    main()
