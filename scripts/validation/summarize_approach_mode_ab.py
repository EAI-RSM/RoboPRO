#!/usr/bin/env python3
"""Paired summary for the Phase 4 APPROACH_MODE A/B (run_approach_mode_ab.sh).

Reads each cell's records.jsonl:
    <root>/<mode>/<timestamp>/records.jsonl        mode in {off, direct, seed}

and reports:
  1. per-cell rollout success rate with a Wilson 95% CI, plus wall-clock throughput;
  2. SEED FIRING RATE in the seed cell -- how often build_seed actually produced a
     route. This is not a nicety: a miss falls back to an unseeded plan silently, so
     without it a null result is unreadable (did the seed not help, or did it never
     fire?). A firing rate near zero invalidates the seed-vs-direct comparison rather
     than answering it;
  3. paired McNemar comparisons on the seeds the cells share --
       direct -> seed  attributes the SEED (those two differ by only the seed),
       off -> direct   measures what the hand-tuned waypoint was worth;
  4. curobo attempt counts for the pre_grasp plan (the stage the mode varies): a good
     seed should converge in fewer attempts even when the success rate does not move.

stdlib only for the stats; matplotlib is imported lazily and only for the figure.

Self-test (no rollout data needed):  python summarize_approach_mode_ab.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path

MODES = ["direct", "seed", "off"]
MODE_BLURB = {
    "direct": "waypoints OFF, direct pre_grasp, NO seed (the floor)",
    "seed": "waypoints OFF, direct pre_grasp, WITH clearance seed (the method)",
    "off": "stock around-box waypoint (opt-in reference; one-occluder heuristic)",
}
# The experiment, and an optional reference comparison. `off` is not run by default:
# the around-box waypoint is hardcoded for one-occluder-in-front, so on general scenes
# it is a different task rather than a control. Pairs whose cells are absent are
# skipped silently.
PAIRS = [
    ("direct", "seed", "THE EXPERIMENT -- these differ by only the seed"),
    ("off", "direct", "optional reference: what the around-box waypoint was worth"),
]


# --------------------------------------------------------------------------- io

def find_cell(root: Path, mode: str) -> Path | None:
    """Newest <root>/<mode>/<timestamp>/ that actually has a records.jsonl.

    The driver stamps each cell with its own timestamp, so the mode folder holds one
    run per invocation; picking the newest lets a re-run of a single cell be folded
    into an existing root without hand-editing paths."""
    base = root / mode
    if not base.is_dir():
        return None
    runs = sorted((d for d in base.iterdir() if d.is_dir() and (d / "records.jsonl").is_file()),
                  key=lambda d: d.name)
    return runs[-1] if runs else None


def load_records(cell_dir: Path | None) -> dict[int, dict]:
    """seed -> per-rollout fields. Empty dict when the cell is missing.

    One (offset, density) combo per cell means one rollout per seed; if a seed somehow
    appears twice the last record wins."""
    if cell_dir is None:
        return {}
    path = cell_dir / "records.jsonl"
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue          # a run killed mid-write leaves a torn final line
        if "rollout_success" not in r:
            continue          # pass-1-only record (no rollout was run for it)
        secs = r.get("rollout_seconds")
        out[int(r["seed"])] = {
            "success": bool(r["rollout_success"]),
            "seconds": float(secs) if secs is not None else None,
            "mode": r.get("approach_mode"),
            "plan_effort": r.get("rollout_plan_effort") or [],
            "seed_stats": r.get("rollout_seed_stats") or [],
            "failure_stage": r.get("rollout_failure_stage"),
        }
    return out


# ------------------------------------------------------------------------ stats

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion (behaves at small n and at
    rates near 0/1, unlike the normal approximation)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on the discordant pairs.
    b = only-A-succeeds, c = only-B-succeeds. Under H0 each discordant pair is a fair
    coin, so p = 2 * P(X <= min(b, c)), X ~ Bin(b + c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def pre_grasp_effort(rec: dict, field: str = "attempts") -> int | None:
    """curobo's effort on this rollout's FIRST pre_grasp plan -- the stage
    APPROACH_MODE actually varies. Later pre_grasp entries come from the candidate
    sweep retrying other contact points/gaps, which is a different question, so taking
    the first keeps the number comparable across modes. None when unrecorded.

    field='attempts'         -> _plan_attempts loop count (1-based; 1 = solved first try)
    field='trajopt_attempts' -> trajopt retries summed over those attempts

    CAVEAT for the 'off' cell: there pre_grasp is planned from the waypoint's qpos (a
    short, easy hop), while direct/seed plan it from rest. So this number is comparable
    between direct and seed -- the pair that matters -- but NOT against off."""
    for e in rec.get("plan_effort") or []:
        if e.get("stage") == "pre_grasp":
            return int(e.get(field) or 0)
    return None


def pre_grasp_attempts(rec: dict) -> int | None:
    """Back-compat alias for the `attempts` field (see pre_grasp_effort)."""
    return pre_grasp_effort(rec, "attempts")


def cell_stats(res: dict[int, dict]) -> dict | None:
    """Success rate + throughput + attempt counts for one cell. None if empty.

    samples_per_hour folds success rate and speed into the one number that matters for
    data collection: usable (successful) rollouts per hour."""
    n = len(res)
    if n == 0:
        return None
    k = sum(1 for v in res.values() if v["success"])
    rate = k / n
    lo, hi = wilson_ci(k, n)
    secs = [v["seconds"] for v in res.values() if v["seconds"] is not None]
    avg_seconds = (sum(secs) / len(secs)) if secs else None
    rollouts_per_hour = (3600.0 / avg_seconds) if avg_seconds else None
    att = [a for a in (pre_grasp_effort(v, "attempts") for v in res.values()) if a is not None]
    tro = [a for a in (pre_grasp_effort(v, "trajopt_attempts") for v in res.values())
           if a is not None]
    att_ok = [a for s, a in ((v["success"], pre_grasp_effort(v, "attempts")) for v in res.values())
              if s and a is not None]
    modes = {v["mode"] for v in res.values() if v["mode"]}
    return {
        "n": n, "k": k, "rate": rate, "ci": (lo, hi),
        "avg_seconds": avg_seconds,
        "rollouts_per_hour": rollouts_per_hour,
        "samples_per_hour": (rollouts_per_hour * rate) if rollouts_per_hour else None,
        "attempts_median": (statistics.median(att) if att else None),
        "attempts_mean": (sum(att) / len(att) if att else None),
        "attempts_median_success": (statistics.median(att_ok) if att_ok else None),
        "attempts_n": len(att),
        "trajopt_median": (statistics.median(tro) if tro else None),
        "trajopt_mean": (sum(tro) / len(tro) if tro else None),
        "modes_seen": modes,
    }


def seed_firing(res: dict[int, dict]) -> dict | None:
    """Seed build outcomes across the cell. Counts FRESH builds only (cache hits are
    tagged reason='cached' and would double-count the same route). Returns None when
    nothing was recorded -- i.e. this cell never tried to seed."""
    fresh = [s for v in res.values() for s in v.get("seed_stats") or []
             if s.get("reason") != "cached"]
    if not fresh:
        return None
    built = [s for s in fresh if s.get("built")]
    secs = [s["seconds"] for s in fresh if s.get("seconds") is not None]
    eps = [s["eps_gated"] for s in built if s.get("eps_gated") is not None]
    reasons: dict[str, int] = {}
    for s in fresh:
        if not s.get("built"):
            reasons[str(s.get("reason"))] = reasons.get(str(s.get("reason")), 0) + 1
    # A build is per (scene, arm), so "episodes with >=1 seed" is the number that says
    # whether the METHOD was actually in play for a given rollout.
    eps_with_seed = sum(1 for v in res.values()
                        if any(s.get("built") for s in v.get("seed_stats") or []))
    return {
        "builds": len(fresh), "built": len(built),
        "rate": len(built) / len(fresh),
        "episodes_with_seed": eps_with_seed, "episodes": len(res),
        "avg_seconds": (sum(secs) / len(secs)) if secs else None,
        "total_seconds": sum(secs) if secs else None,
        "median_eps": (statistics.median(eps) if eps else None),
        "fail_reasons": reasons,
    }


# ---------------------------------------------------------------------- printing

def print_cell(mode: str, st: dict | None) -> None:
    if st is None:
        print(f"  {mode:<7} MISSING (no records.jsonl -- cell did not run or produced nothing)")
        return
    lo, hi = st["ci"]
    print(f"  {mode:<7} {st['k']:>3}/{st['n']:<3} = {st['rate']:6.1%}  95% CI [{lo:5.1%}, {hi:5.1%}]", end="")
    if st["avg_seconds"] is not None:
        print(f"  |  {st['avg_seconds']:6.1f}s/rollout"
              f"  {st['samples_per_hour']:6.1f} usable samples/hr", end="")
    if st["attempts_median"] is not None:
        print(f"  |  pre_grasp attempts: median {st['attempts_median']:.0f}"
              f" mean {st['attempts_mean']:.1f} (n={st['attempts_n']})", end="")
    if st["trajopt_median"] is not None:
        print(f"  trajopt: median {st['trajopt_median']:.0f}"
              f" mean {st['trajopt_mean']:.1f}", end="")
    print()
    # A cell whose records disagree with its folder name means the driver was bypassed
    # or APPROACH_MODE was not exported -- the comparison below would be meaningless.
    seen = st["modes_seen"]
    if seen and seen != {mode}:
        print(f"          !! WARNING: records say approach_mode={sorted(seen)} but this is "
              f"the '{mode}' cell -- the cells may not differ as intended")


def print_firing(fr: dict | None) -> None:
    print("\n[SEED FIRING]  did the method actually engage?")
    if fr is None:
        print("  no seed builds recorded (the 'seed' cell did not run, or never reached a build)")
        return
    print(f"  routes built     : {fr['built']}/{fr['builds']} fresh builds = {fr['rate']:.1%}")
    print(f"  episodes seeded  : {fr['episodes_with_seed']}/{fr['episodes']}")
    if fr["avg_seconds"] is not None:
        print(f"  build cost       : {fr['avg_seconds']:.1f}s avg, {fr['total_seconds']:.0f}s total")
    if fr["median_eps"] is not None:
        print(f"  median eps_gated : {fr['median_eps']:.3f} m")
    if fr["fail_reasons"]:
        detail = ", ".join(f"{r}={n}" for r, n in sorted(fr["fail_reasons"].items(),
                                                         key=lambda kv: -kv[1]))
        print(f"  no-route reasons : {detail}")
    if fr["episodes_with_seed"] == 0:
        print("  !! the seed NEVER fired -- seed-vs-direct below measures nothing. Fix the")
        print("     build before reading the comparison (check a cell log for '[seed] ...').")
    elif fr["rate"] < 0.5:
        print("  !! fired on under half of builds -- seed-vs-direct is DILUTED toward null;")
        print("     the per-episode effect is larger than the cell-level delta suggests.")


def print_pair(a: str, b: str, why: str,
               ra: dict[int, dict], rb: dict[int, dict]) -> None:
    print(f"\n[PAIRED] {a} -> {b}   ({why})")
    shared = sorted(set(ra) & set(rb))
    if not shared:
        print("  SKIPPED (the two cells share no seeds)")
        return
    only_a, only_b = set(ra) - set(rb), set(rb) - set(ra)
    if only_a or only_b:
        print(f"  note: seed sets differ ({a}-only={len(only_a)}, {b}-only={len(only_b)}); "
              f"pairing on the {len(shared)} shared seeds")

    sa = {s: ra[s]["success"] for s in shared}
    sb = {s: rb[s]["success"] for s in shared}
    both = sum(1 for s in shared if sa[s] and sb[s])
    neither = sum(1 for s in shared if not sa[s] and not sb[s])
    only_a_win = sum(1 for s in shared if sa[s] and not sb[s])
    only_b_win = sum(1 for s in shared if sb[s] and not sa[s])
    p = mcnemar_exact_p(only_a_win, only_b_win)
    rate_a = sum(sa.values()) / len(shared)
    rate_b = sum(sb.values()) / len(shared)

    print(f"  paired on {len(shared)} seeds:  both {both} | neither {neither} | "
          f"only {a} {only_a_win} | only {b} {only_b_win}")
    print(f"  {b} - {a} = {rate_b - rate_a:+.1%}   ({rate_b:.1%} vs {rate_a:.1%})")
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"  McNemar exact two-sided p = {p:.4f}  ({verdict} at alpha=0.05)")
    if only_a_win + only_b_win == 0:
        print("  (no discordant pairs -- the two cells succeeded and failed on exactly the")
        print("   same seeds, so this run carries no evidence either way)")

    # Effort on the shared seeds: the seed can pay off as faster convergence even when
    # the success rate is flat, so report it whether or not McNemar fires.
    for field, label in (("attempts", "pre_grasp attempts"),
                         ("trajopt_attempts", "trajopt attempts")):
        aa = [x for x in (pre_grasp_effort(ra[s], field) for s in shared) if x is not None]
        bb = [x for x in (pre_grasp_effort(rb[s], field) for s in shared) if x is not None]
        if aa and bb:
            print(f"  {label}: {a} median {statistics.median(aa):.0f} -> "
                  f"{b} median {statistics.median(bb):.0f}  "
                  f"(mean {sum(aa)/len(aa):.1f} -> {sum(bb)/len(bb):.1f})")
    if "off" in (a, b):
        print("  NOTE: off plans pre_grasp from the WAYPOINT's qpos, direct/seed from rest --")
        print("        the effort numbers above are not comparable for this pair (rates are).")


def print_failure_stages(mode: str, res: dict[int, dict]) -> None:
    """Where the failures died. In direct/seed mode a pile-up at 'pre_grasp' is the
    expected shape (no waypoint fallback); a pile-up elsewhere means the mode change
    is not what is costing the rollouts."""
    stages: dict[str, int] = {}
    for v in res.values():
        if not v["success"]:
            stages[str(v.get("failure_stage"))] = stages.get(str(v.get("failure_stage")), 0) + 1
    if not stages:
        return
    detail = ", ".join(f"{s}={n}" for s, n in sorted(stages.items(), key=lambda kv: -kv[1]))
    print(f"  {mode:<7} failure stages: {detail}")


# ----------------------------------------------------------------------- figure

def make_figure(stats: dict[str, dict | None], out_path: Path) -> None:
    """Three panels: success rate with Wilson CI, usable samples/hour, median
    pre_grasp attempts -- one bar per mode. Skips panels that have no data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [m for m in MODES if stats.get(m)]
    if not cells:
        print("\nno cell data -> skipping figure")
        return
    colors = {"off": "#8C8C8C", "direct": "#4C72B0", "seed": "#DD8452"}
    x = range(len(cells))
    cc = [colors[m] for m in cells]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes

    rates = [stats[m]["rate"] for m in cells]
    # Wilson CI as asymmetric error bars -- the interval is not centred on the rate.
    err_lo = [max(0.0, stats[m]["rate"] - stats[m]["ci"][0]) for m in cells]
    err_hi = [max(0.0, stats[m]["ci"][1] - stats[m]["rate"]) for m in cells]
    ax1.bar(x, rates, color=cc, edgecolor="white")
    ax1.errorbar(list(x), rates, yerr=[err_lo, err_hi], fmt="none", ecolor="#333", capsize=5)
    ax1.set_title("Rollout success rate (95% Wilson CI)")
    ax1.set_ylabel("success rate")
    ax1.set_ylim(0, 1)
    for xi, m in zip(x, cells):
        ax1.text(xi, stats[m]["rate"], f" {stats[m]['k']}/{stats[m]['n']}",
                 ha="center", va="bottom", fontsize=10)

    thru = [stats[m]["samples_per_hour"] for m in cells]
    if any(t is not None for t in thru):
        vals = [t or 0.0 for t in thru]
        ax2.bar(x, vals, color=cc, edgecolor="white")
        for xi, v in zip(x, vals):
            ax2.text(xi, v, f"{v:.0f}/hr", ha="center", va="bottom", fontsize=10)
    ax2.set_title("Usable data throughput\n(success rate x speed)")
    ax2.set_ylabel("successful rollouts / hour")

    att = [stats[m]["attempts_median"] for m in cells]
    if any(a is not None for a in att):
        vals = [a or 0.0 for a in att]
        ax3.bar(x, vals, color=cc, edgecolor="white")
        for xi, v in zip(x, vals):
            ax3.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    ax3.set_title("curobo attempts on the pre_grasp plan\n(median; lower is better)")
    ax3.set_ylabel("attempts")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(cells, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Phase 4: grasp-approach mode  --  off (waypoint) vs direct vs seeded", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nsaved figure -> {out_path}")


# ------------------------------------------------------------------------- main

def summarize(root: Path, figure: bool = True) -> None:
    cells = {m: find_cell(root, m) for m in MODES}
    data = {m: load_records(cells[m]) for m in MODES}
    stats = {m: cell_stats(data[m]) for m in MODES}

    print("=" * 74)
    print("PHASE 4  --  APPROACH_MODE A/B SUMMARY")
    print(f"root: {root}")
    print("=" * 74)
    for m in MODES:
        if cells[m]:
            print(f"  {m:<7} <- {cells[m].relative_to(root)}   ({MODE_BLURB[m]})")

    print("\n[CELLS]  rollout success (Wilson 95% CI)")
    for m in MODES:
        # A mode with no folder at all simply was not requested (off is opt-in) -- say
        # nothing. A mode WITH a folder but no usable records is a real failure worth
        # flagging, and print_cell says so.
        if cells[m] is None and not (root / m).is_dir():
            continue
        print_cell(m, stats[m])

    print("\n[FAILURES]")
    for m in MODES:
        if data[m]:
            print_failure_stages(m, data[m])

    print_firing(seed_firing(data["seed"]))

    for a, b, why in PAIRS:
        # Silently skip a pair whose cells were not both run -- `off` is opt-in, and a
        # SKIPPED banner on every default run is noise, not information.
        if data[a] and data[b]:
            print_pair(a, b, why, data[a], data[b])

    if figure:
        make_figure(stats, root / "approach_mode_ab.png")

    print("\n" + "-" * 74)
    print("Reading this: 'direct -> seed' IS the experiment -- both cells plan pre_grasp")
    print("straight from rest with no waypoint fallback, so they differ by only the seed")
    print("and the delta is attributable to it. Check SEED FIRING first: a seed that")
    print("rarely builds cannot show an effect, and a null would mean nothing.")
    if data["off"]:
        print("The 'off' cell is a reference number only -- the around-box waypoint is a")
        print("one-occluder-in-front heuristic, so it is a different task, not a control.")


def _selftest() -> None:
    """End-to-end check on synthetic records: exercises the loader, the stats, both
    paired comparisons and the figure, with a known answer. No sim/GPU needed."""
    # Cook a case with a deliberate signal: seed wins 8 discordant pairs, loses 0.
    def rec(seed, ok, mode, attempts, seeded_route=None, secs=100.0):
        r = {"seed": seed, "rollout_success": ok, "approach_mode": mode,
             "rollout_seconds": secs, "rollout_failure_stage": None if ok else "pre_grasp",
             "rollout_plan_effort": [{"stage": "pre_grasp", "arm": "left", "status": "Success",
                                      "attempts": attempts, "trajopt_attempts": 2 * attempts,
                                      "seeded": mode == "seed"}]}
        if seeded_route is not None:
            r["rollout_seed_stats"] = [{"arm": "left", "built": seeded_route, "reason":
                                        None if seeded_route else "no_gated_path",
                                        "seconds": 40.0, "route_voxels": 16,
                                        "eps_gated": 0.17 if seeded_route else None}]
        return r

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        plan = {  # mode -> list of (seed, success, attempts, built)
            "off":    [(i, i < 12, 3, None) for i in range(20)],
            "direct": [(i, i < 4, 9, None) for i in range(20)],
            "seed":   [(i, i < 12, 2, True) for i in range(20)],
        }
        for mode, rows in plan.items():
            d = root / mode / "20260727-120000"
            d.mkdir(parents=True)
            with (d / "records.jsonl").open("w") as fh:
                for s, ok, att, built in rows:
                    fh.write(json.dumps(rec(s, ok, mode, att, built)) + "\n")

        cells = {m: find_cell(root, m) for m in MODES}
        assert all(cells[m] is not None for m in MODES), "find_cell missed a cell"
        data = {m: load_records(cells[m]) for m in MODES}
        assert all(len(data[m]) == 20 for m in MODES), "loader dropped records"

        st = cell_stats(data["seed"])
        assert st["k"] == 12 and st["n"] == 20, st
        assert abs(st["rate"] - 0.6) < 1e-9, st
        lo, hi = st["ci"]
        assert lo < 0.6 < hi, (lo, hi)
        assert st["attempts_median"] == 2, st
        assert st["trajopt_median"] == 4, st          # trajopt_attempts = 2 * attempts
        assert abs(st["samples_per_hour"] - 36.0 * 0.6) < 1e-6, st
        assert pre_grasp_effort(data["direct"][0], "trajopt_attempts") == 18, "field select"

        fr = seed_firing(data["seed"])
        assert fr["built"] == 20 and abs(fr["rate"] - 1.0) < 1e-9, fr
        assert fr["episodes_with_seed"] == 20, fr
        assert seed_firing(data["direct"]) is None, "direct cell should record no builds"

        # direct(4/20) vs seed(12/20): 8 discordant, all favouring seed -> p = 2*0.5^8
        b = sum(1 for s in range(20) if data["direct"][s]["success"] and not data["seed"][s]["success"])
        c = sum(1 for s in range(20) if data["seed"][s]["success"] and not data["direct"][s]["success"])
        assert (b, c) == (0, 8), (b, c)
        p = mcnemar_exact_p(b, c)
        assert abs(p - 2 * 0.5 ** 8) < 1e-12, p
        assert p < 0.05, p
        # and the degenerate cases the real data will hit
        assert mcnemar_exact_p(0, 0) == 1.0
        assert abs(mcnemar_exact_p(3, 3) - 1.0) < 1e-12
        assert wilson_ci(0, 0) == (0.0, 0.0)
        assert pre_grasp_attempts({"plan_effort": []}) is None

        # torn final line (a run killed mid-write) must not crash the loader
        torn = root / "off" / "20260727-120000" / "records.jsonl"
        torn.write_text(torn.read_text() + '{"seed": 99, "rollout_suc')
        assert len(load_records(cells["off"])) == 20, "torn line was not skipped"

        summarize(root, figure=True)
        assert (root / "approach_mode_ab.png").is_file(), "figure not written"

        # The DEFAULT run is direct+seed only (off is opt-in): a missing off cell must
        # not crash, must drop the off->direct pair silently, and must still plot.
        import shutil
        shutil.rmtree(root / "off")
        (root / "approach_mode_ab.png").unlink()
        assert find_cell(root, "off") is None, "find_cell should report the removed cell"
        summarize(root, figure=True)
        assert (root / "approach_mode_ab.png").is_file(), "figure not written without off"

    print("\n[selftest] ALL PASS (loader/Wilson/McNemar/firing-rate/attempts/figure,"
          " with and without the opt-in off cell)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    help="results root passed to run_approach_mode_ab.sh")
    ap.add_argument("--no-figure", action="store_true", help="skip the PNG")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic-data self-test and exit")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if args.root is None:
        ap.error("--root is required (or pass --selftest)")
    if not args.root.is_dir():
        print(f"ERROR: no such results root: {args.root}", file=sys.stderr)
        sys.exit(1)
    summarize(args.root, figure=not args.no_figure)


if __name__ == "__main__":
    main()
