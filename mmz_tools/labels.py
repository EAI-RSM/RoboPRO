#!/usr/bin/env python3
"""Show the outcome label of every episode, grouped by task, pos next to neg.

Usage:
    python labels.py [root]          # root defaults to data/mmz_samples
"""
import glob
import json
import os
import sys
from collections import Counter

ABBR = {"success": "S", "success_with_accident": "SA", "crashed_and_failed": "CF",
        "failed_no_accident": "FN", "failed": "F", "error": "E", "?": "?"}

root = sys.argv[1] if len(sys.argv) > 1 else "data/mmz_samples"
rows, totals = {}, Counter()
for f in sorted(glob.glob(os.path.join(root, "*", "*", "scene_info.json"))):
    parts = f.split(os.sep)
    task, cfg = parts[-3], parts[-2]
    db = json.load(open(f))
    eps = sorted((k for k in db if k.startswith("episode_")),
                 key=lambda k: int(k.split("_")[1]))
    run_dir = os.path.dirname(f)
    labels, kept = [], []
    for k in eps:
        labels.append(db[k].get("outcome", {}).get("label", "?"))
        idx = k.split("_")[1]
        kept.append(os.path.exists(os.path.join(run_dir, "data", f"episode{idx}.hdf5")))
    rows.setdefault(task, {})[cfg] = (labels, kept)
    totals.update(labels)

kept_total = 0
for task in sorted(rows):
    print(f"\n\033[1m{task}\033[0m")
    for cfg in sorted(rows[task]):
        labs, kept = rows[task][cfg]
        kept_total += sum(kept)
        seq = " ".join(f"{i}:{ABBR.get(l, l)}{'' if k else '×'}"
                       for i, (l, k) in enumerate(zip(labs, kept)))
        summ = " ".join(f"{ABBR.get(k, k)}={v}" for k, v in sorted(Counter(labs).items()))
        print(f"  {cfg:20s} [{seq}]   ({summ}, kept {sum(kept)}/{len(kept)})")

print("\n\033[1mTOTAL\033[0m  " + "  ".join(f"{ABBR.get(k, k)}={v}" for k, v in sorted(totals.items()))
      + f"  ·  kept {kept_total}/{sum(totals.values())}")
print("legend: S=success  SA=success_with_accident  CF=crashed_and_failed  "
      "FN=failed_no_accident  E=error  ×=filtered out (no hdf5)")
