#!/usr/bin/env python3
"""Collate policy-eval success logs into one table.

Eval writes one folder per run:
    eval_result/<task>/<policy>/<task_config>/<ckpt_setting>/<timestamp>/_result.txt
(sometimes under an extra eval_result/bench_eval_result/... prefix). Each _result.txt ends
with the run's success rate(s). This walks the tree, keeps the LATEST run per
(task, policy, config, ckpt), prints a table grouped by config, and writes summary.csv.

    python script/eval_summary.py [--root eval_result] [--filter SUBSTR] [--csv PATH]

Pure stdlib — runs under any Python. Read-only except for the CSV it writes.
"""
import argparse
import csv
import glob
import os
import re

FLOAT = re.compile(r"[-+]?\d*\.?\d+")


def read_success(path):
    """Success rate from a _result.txt = the max float on any of its lines (the file ends
    with suc/test_num; taking the max is robust to the header lines and to top-k lists)."""
    best = None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.lstrip().startswith(("Timestamp", "Instruction")):
                    continue
                for tok in FLOAT.findall(line):
                    v = float(tok)
                    if 0.0 <= v <= 1.0 and (best is None or v > best):
                        best = v
    except OSError:
        return None
    return best


def parse_run(result_path, root):
    """(task, policy, config, ckpt, timestamp) from a _result.txt path, or None.
    Layout counted from the end: .../<task>/<policy>/<config>/<ckpt>/<timestamp>/_result.txt"""
    rel = os.path.relpath(result_path, root)
    parts = rel.split(os.sep)
    if len(parts) < 6:
        return None
    task, policy, config, ckpt, ts = parts[-6], parts[-5], parts[-4], parts[-3], parts[-2]
    return task, policy, config, ckpt, ts


def collect(root, filt):
    """Latest run per (task, policy, config, ckpt) -> dict keyed by that tuple."""
    runs = {}
    for rp in glob.glob(os.path.join(root, "**", "_result.txt"), recursive=True):
        parsed = parse_run(rp, root)
        if parsed is None:
            continue
        task, policy, config, ckpt, ts = parsed
        if filt and filt not in task and filt not in config:
            continue
        sr = read_success(rp)
        if sr is None:
            continue
        key = (task, policy, config, ckpt)
        if key not in runs or ts > runs[key][0]:
            runs[key] = (ts, sr)
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="eval_result", help="eval output root (default eval_result)")
    ap.add_argument("--filter", default="", help="only rows whose task or config contains this")
    ap.add_argument("--csv", default=None, help="CSV path (default <root>/summary.csv)")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        print(f"no eval output at '{a.root}' — run eval_suite.sh first")
        return
    runs = collect(a.root, a.filter)
    if not runs:
        print(f"no _result.txt found under '{a.root}'" + (f" matching '{a.filter}'" if a.filter else ""))
        return

    rows = [(t, p, c, k, ts, sr) for (t, p, c, k), (ts, sr) in runs.items()]
    rows.sort(key=lambda r: (r[2], r[0]))  # by config, then task

    tw = max(len(r[0]) for r in rows) + 2
    cw = max(len(r[2]) for r in rows) + 2
    print(f"\n{'TASK':<{tw}}{'CONFIG':<{cw}}{'SR':>7}")
    print("-" * (tw + cw + 7))
    last_cfg, bucket = None, []
    def flush():
        if bucket:
            avg = sum(bucket) / len(bucket)
            print(f"{'  AVG ('+str(len(bucket))+' tasks)':<{tw+cw}}{avg:>7.2f}")
            print("-" * (tw + cw + 7))
    for t, p, c, k, ts, sr in rows:
        if c != last_cfg:
            flush(); bucket = []; last_cfg = c
        print(f"{t:<{tw}}{c:<{cw}}{sr:>7.2f}")
        bucket.append(sr)
    flush()

    csv_path = a.csv or os.path.join(a.root, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "policy", "config", "ckpt", "timestamp", "success_rate"])
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {csv_path}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
