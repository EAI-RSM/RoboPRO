#!/usr/bin/env python3
"""Paired summary for the 2x2 planner comparison (run_2x2_planner_comparison.sh).

Reads the four cells' records.jsonl:
    <root>/curated_rollout/{baseline,reachability}/records.jsonl
    <root>/typical_rollout/{baseline,reachability}/records.jsonl

and reports, per cell, the rollout success rate with a Wilson 95% CI, then a
WITHIN-SCENE paired baseline-vs-reachability comparison (McNemar exact test on the
seeds both cells share -- valid because seed acceptance is planner-independent, so
both planners saw the same scenes).

stdlib only (math/json/argparse); no numpy/scipy needed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCENES = ["curated", "typical"]
ALGOS = ["baseline", "reachability"]


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
                               "seconds": float(secs) if secs is not None else None}
    return out


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
    b = only-baseline-succeeds, c = only-reachability-succeeds. Under H0 each
    discordant pair is a fair coin, so p = 2 * P(X <= min(b,c)), X ~ Bin(b+c, 0.5)."""
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


def paired_scene(base: dict[int, dict], reach: dict[int, dict]) -> None:
    shared = sorted(set(base) & set(reach))
    if not shared:
        print(f"  paired test: SKIPPED (no shared seeds between the two planners)")
        return
    only_b = set(base) - set(reach)
    only_r = set(reach) - set(base)
    if only_b or only_r:
        print(f"  note: seed sets differ (baseline-only={len(only_b)}, "
              f"reachability-only={len(only_r)}); pairing on the {len(shared)} shared seeds")

    bs = {s: base[s]["success"] for s in shared}
    rs = {s: reach[s]["success"] for s in shared}
    both = sum(1 for s in shared if bs[s] and rs[s])
    neither = sum(1 for s in shared if not bs[s] and not rs[s])
    b = sum(1 for s in shared if bs[s] and not rs[s])       # only baseline solved
    c = sum(1 for s in shared if rs[s] and not bs[s])       # only reachability solved
    p = mcnemar_exact_p(b, c)

    base_rate = sum(bs.values()) / len(shared)
    reach_rate = sum(rs.values()) / len(shared)
    delta = reach_rate - base_rate

    print(f"  paired on {len(shared)} seeds:")
    print(f"     both solved      : {both}")
    print(f"     both failed      : {neither}")
    print(f"     only baseline    : {b}")
    print(f"     only reachability: {c}")
    print(f"     reachability - baseline = {delta:+.1%} "
          f"({reach_rate:.1%} vs {base_rate:.1%})")
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"     McNemar exact two-sided p = {p:.4f}  ({verdict} at alpha=0.05)")


def make_figure(stats: dict, out_path: Path) -> None:
    """Two-panel bar chart: (left) avg time per rollout, (right) usable samples/hour,
    baseline vs reachability within each scene. Skips cells with no timing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [(sc, al) for sc in SCENES for al in ALGOS
             if stats.get((sc, al)) and stats[(sc, al)]["avg_seconds"] is not None]
    if not cells:
        print("no timing data -> skipping figure")
        return

    labels = [f"{sc}\n{al}" for sc, al in cells]
    times = [stats[c]["avg_seconds"] for c in cells]
    thru = [stats[c]["samples_per_hour"] for c in cells]
    colors = ["#4C72B0" if al == "baseline" else "#DD8452" for _, al in cells]
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
    args = ap.parse_args()

    data: dict[tuple[str, str], dict[int, dict]] = {}
    stats: dict[tuple[str, str], dict | None] = {}
    for scene in SCENES:
        for algo in ALGOS:
            cell = args.root / f"{scene}_rollout" / algo
            data[(scene, algo)] = load_records(cell)
            stats[(scene, algo)] = cell_stats(data[(scene, algo)])

    print("=" * 64)
    print("2x2 PLANNER COMPARISON SUMMARY")
    print(f"root: {args.root}")
    print("=" * 64)

    for scene in SCENES:
        title = ("CURATED  (milk-box occluder + bottle)" if scene == "curated"
                 else "TYPICAL  (random clutter + mouse)")
        print(f"\n[{scene.upper()}] {title}")
        for algo in ALGOS:
            summarize_cell(algo, stats[(scene, algo)])
        paired_scene(data[(scene, "baseline")], data[(scene, "reachability")])

    make_figure(stats, args.root / "timing_throughput.png")

    # Cross-scene reminder: we do NOT pair across scenes (different objects/scenes),
    # so those are descriptive only.
    print("\n" + "-" * 64)
    print("Note: baseline vs reachability is paired WITHIN each scene (same seeds).")
    print("Curated vs typical is descriptive only (different object + scene).")
    print("For cell TYPICAL/reachability, eyeball a few videos under")
    print("  <root>/typical_rollout/reachability/{success,fail}/  --  the placement")
    print("path is box-tuned, so confirm failures are real clutter collisions, not a")
    print("gratuitous high box-detour.")


if __name__ == "__main__":
    main()
