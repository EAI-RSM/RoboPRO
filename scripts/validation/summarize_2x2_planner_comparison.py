#!/usr/bin/env python3
"""Paired summary for the planner comparison (run_2x2_planner_comparison.sh).

Reads each cell's records.jsonl:
    <root>/<scene>_rollout/<algo>/records.jsonl
(default scene = curated, default algos = baseline vs hamid)

and reports, per cell, the rollout success rate with a Wilson 95% CI, then a
WITHIN-SCENE paired comparison of the two planners (McNemar exact test on the seeds
both cells share -- valid because seed acceptance is planner-independent, so both
planners saw the same scenes).

stdlib only (math/json/argparse); no numpy/scipy needed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_SCENES = ["curated"]
DEFAULT_ALGOS = ["baseline", "hamid"]


def load_records(cell_dir: Path) -> dict[int, bool]:
    """seed -> rollout_success for one cell. Empty dict if the cell is missing.
    With one (offset, density) combo per cell there is one rollout per seed; if a
    seed somehow appears twice, the last record wins. Value is
    {"success": bool, "seconds": float|None} (seconds absent in pre-timing runs)."""
    path = cell_dir / "records.jsonl"
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "rollout_success" not in r:
            continue
        secs = r.get("rollout_seconds")
        out[int(r["seed"])] = {"success": bool(r["rollout_success"]),
                               "seconds": float(secs) if secs is not None else None,
                               "stage": r.get("rollout_stage")}
    return out


# Coarse failure/outcome taxonomy, in pipeline order. Both planners are normalized into
# this: baseline already emits these names; hamid emits granular checkpoint names that we
# bucket by substring here.
STAGE_ORDER = ["setup", "forward/grasp", "transition", "backward-placement",
               "final descent", "success check", "success", "exception", "other", "unknown"]


def coarse_stage(s: str | None) -> str:
    """Map a raw rollout_stage (coarse baseline label OR granular hamid checkpoint) to one
    of STAGE_ORDER."""
    if not s:
        return "unknown"
    if s == "success":
        return "success"
    if s == "success_check":
        return "success check"
    if s in STAGE_ORDER:
        return s                      # baseline already emits coarse names
    if s.startswith("exception"):
        return "exception"
    r = s.lower()
    if "descent" in r or "place_actor" in r:
        return "final descent"
    if "placement" in r or "place" in r:
        return "backward-placement"
    if "attach" in r or "enable_table" in r or "lift" in r:
        return "transition"
    if "grasp" in r or "waypoint" in r:
        return "forward/grasp"
    return "other"


def stage_tally_table(algos: list[str], res_by_algo: dict[str, dict]) -> None:
    """Print a stage x planner count table for one scene."""
    counts = {al: {st: 0 for st in STAGE_ORDER} for al in algos}
    for al in algos:
        for v in res_by_algo[al].values():
            counts[al][coarse_stage(v.get("stage"))] += 1
    # only show rows any planner hit
    rows = [st for st in STAGE_ORDER if any(counts[al][st] for al in algos)]
    if not rows:
        print("  stage tally: (no rollout_stage in records -- older run?)")
        return
    w = max(len("stage"), max(len(st) for st in rows))
    header = "  " + "stage".ljust(w) + "".join(f"  {al:>12}" for al in algos)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for st in rows:
        line = "  " + st.ljust(w) + "".join(f"  {counts[al][st]:>12d}" for al in algos)
        print(line)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion (robust at small n and
    at rates near 0/1, unlike the normal approximation)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on the discordant pairs.
    b = only-A-succeeds, c = only-B-succeeds. Under H0 each discordant pair is a
    fair coin, so p = 2 * P(X <= min(b,c)), X ~ Bin(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def cell_stats(res: dict[int, dict]) -> dict | None:
    """Success rate + timing throughput for one cell. None if the cell is empty.
    - avg_seconds: mean wall-clock per rollout (over successes AND failures -- fails
      cost time too), or None if this run predates timing.
    - rollouts_per_hour  : raw attempts you can run per hour = 3600 / avg_seconds
    - samples_per_hour   : USABLE (successful) samples per hour = rollouts/hr * rate
      This is the headline: it folds success rate and speed into one throughput number."""
    n = len(res)
    if n == 0:
        return None
    k = sum(1 for v in res.values() if v["success"])
    rate = k / n
    lo, hi = wilson_ci(k, n)
    secs = [v["seconds"] for v in res.values() if v["seconds"] is not None]
    avg_seconds = (sum(secs) / len(secs)) if secs else None
    rollouts_per_hour = (3600.0 / avg_seconds) if avg_seconds else None
    samples_per_hour = (rollouts_per_hour * rate) if rollouts_per_hour is not None else None
    return {"n": n, "k": k, "rate": rate, "ci": (lo, hi), "avg_seconds": avg_seconds,
            "rollouts_per_hour": rollouts_per_hour, "samples_per_hour": samples_per_hour}


def summarize_cell(algo: str, st: dict | None) -> None:
    if st is None:
        print(f"  {algo:<13} MISSING (no records.jsonl -- cell did not run or produced nothing)")
        return
    lo, hi = st["ci"]
    print(f"  {algo:<13} {st['k']:>3}/{st['n']:<3} = {st['rate']:6.1%}  95% CI [{lo:5.1%}, {hi:5.1%}]", end="")
    if st["avg_seconds"] is not None:
        print(f"   |  {st['avg_seconds']:6.1f}s/rollout"
              f"   {st['rollouts_per_hour']:6.1f} rollouts/hr"
              f"   {st['samples_per_hour']:6.1f} usable samples/hr")
    else:
        print("   |  (no timing in these records)")


def paired_scene(a_res: dict[int, dict], b_res: dict[int, dict],
                 a_name: str, b_name: str) -> None:
    shared = sorted(set(a_res) & set(b_res))
    if not shared:
        print(f"  paired test: SKIPPED (no shared seeds between the two planners)")
        return
    only_a = set(a_res) - set(b_res)
    only_b = set(b_res) - set(a_res)
    if only_a or only_b:
        print(f"  note: seed sets differ ({a_name}-only={len(only_a)}, "
              f"{b_name}-only={len(only_b)}); pairing on the {len(shared)} shared seeds")

    asucc = {s: a_res[s]["success"] for s in shared}
    bsucc = {s: b_res[s]["success"] for s in shared}
    both = sum(1 for s in shared if asucc[s] and bsucc[s])
    neither = sum(1 for s in shared if not asucc[s] and not bsucc[s])
    b = sum(1 for s in shared if asucc[s] and not bsucc[s])       # only A solved
    c = sum(1 for s in shared if bsucc[s] and not asucc[s])       # only B solved
    p = mcnemar_exact_p(b, c)

    a_rate = sum(asucc.values()) / len(shared)
    b_rate = sum(bsucc.values()) / len(shared)
    delta = b_rate - a_rate

    print(f"  paired on {len(shared)} seeds:")
    print(f"     both solved       : {both}")
    print(f"     both failed       : {neither}")
    print(f"     only {a_name:<12}: {b}")
    print(f"     only {b_name:<12}: {c}")
    print(f"     {b_name} - {a_name} = {delta:+.1%} "
          f"({b_rate:.1%} vs {a_rate:.1%})")
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"     McNemar exact two-sided p = {p:.4f}  ({verdict} at alpha=0.05)")


def make_figure(stats: dict, out_path: Path, scenes: list[str], algos: list[str]) -> None:
    """Two-panel bar chart: (left) avg time per rollout, (right) usable samples/hour,
    the two planners within each scene. Skips cells with no timing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [(sc, al) for sc in scenes for al in algos
             if stats.get((sc, al)) and stats[(sc, al)]["avg_seconds"] is not None]
    if not cells:
        print("no timing data -> skipping figure")
        return

    labels = [f"{sc}\n{al}" for sc, al in cells]
    times = [stats[c]["avg_seconds"] for c in cells]
    thru = [stats[c]["samples_per_hour"] for c in cells]
    colors = ["#4C72B0" if al == algos[0] else "#DD8452" for _, al in cells]
    x = range(len(cells))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(x, times, color=colors, edgecolor="white")
    ax1.set_title("Average time per rollout")
    ax1.set_ylabel("seconds / rollout")
    for xi, t in zip(x, times):
        ax1.text(xi, t, f"{t:.1f}s", ha="center", va="bottom", fontsize=10)

    ax2.bar(x, thru, color=colors, edgecolor="white")
    ax2.set_title("Usable data throughput (success rate x speed)")
    ax2.set_ylabel("successful samples / hour")
    for xi, v in zip(x, thru):
        ax2.text(xi, v, f"{v:.0f}/hr", ha="center", va="bottom", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("2x2 planner comparison -- throughput", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nsaved figure -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="results root passed to run_2x2_planner_comparison.sh")
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES,
                    help=f"scenes to summarize (default: {' '.join(DEFAULT_SCENES)})")
    ap.add_argument("--algos", nargs=2, default=DEFAULT_ALGOS, metavar=("ALGO_A", "ALGO_B"),
                    help=f"the two planners to pair (default: {' '.join(DEFAULT_ALGOS)})")
    args = ap.parse_args()

    scenes, algos = args.scenes, args.algos
    a_name, b_name = algos

    data: dict[tuple[str, str], dict[int, dict]] = {}
    stats: dict[tuple[str, str], dict | None] = {}
    for scene in scenes:
        for algo in algos:
            cell = args.root / f"{scene}_rollout" / algo
            data[(scene, algo)] = load_records(cell)
            stats[(scene, algo)] = cell_stats(data[(scene, algo)])

    print("=" * 64)
    print(f"PLANNER COMPARISON SUMMARY  ({a_name} vs {b_name})")
    print(f"root: {args.root}")
    print("=" * 64)

    for scene in scenes:
        title = ("CURATED  (milk-box occluder + bottle + clutter)" if scene == "curated"
                 else "TYPICAL  (random clutter + mouse)")
        print(f"\n[{scene.upper()}] {title}")
        for algo in algos:
            summarize_cell(algo, stats[(scene, algo)])
        paired_scene(data[(scene, a_name)], data[(scene, b_name)], a_name, b_name)
        print("  stage breakdown (rollouts per outcome/failure stage):")
        stage_tally_table(algos, {al: data[(scene, al)] for al in algos})

    make_figure(stats, args.root / "timing_throughput.png", scenes, algos)

    print("\n" + "-" * 64)
    print(f"Note: {a_name} vs {b_name} is paired WITHIN each scene (same seeds).")
    if len(scenes) > 1:
        print("Across scenes is descriptive only (different object + scene).")


if __name__ == "__main__":
    main()
