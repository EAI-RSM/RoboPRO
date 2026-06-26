"""
Phase 1 (#30): natural visibility distribution / rejection-sampling baseline.

Sweeps seeds x obstacle_density for an office task and measures the t=0
countertop `visible_fraction` of the target (no rollout), reusing the Phase 0
primitive (#28). Produces:

  - a JSONL of per-(seed, density) records,
  - per-density histograms of visible_fraction with the bucket boundaries marked,
  - a per-bucket readout (natural mass / acceptance rate) classifying each bucket
    as selectable-by-rejection vs rare/empty -> needs-occluder.

DENOMINATOR CHOICE (documented):
  visible_fraction = visible_target_px(scene at density D) / full_target_px(clean scene),
  both on the countertop camera, for the SAME seed. The denominator is captured
  from a clean build (`cluttered_table=False`, density 0). This is valid because
  the target pose is set in load_actors() *before* any clutter is generated and
  np.random is seeded first, so the target sits at the same pose for every density
  of a given seed (verified by asserting pose equality). The clean count still
  includes any static-furniture / gripper occlusion present at t=0, so the
  fraction isolates the marginal effect of table clutter.

USAGE (run from the benchmark folder):
    cd benchmark
    source set_env.sh
    export ROBOTWIN_BENCH_TASK=bench
    python script/bench_script/analyze_natural_visibility.py \
        put_mouse_on_pad --bench-subdir office \
        --base-config bench_demo_office_d8 \
        --seed-start 0 --num-seeds 50 \
        --densities 8,12,15 --bins 20 \
        --out-dir ./visibility_phase1

    # Re-plot from an existing sweep without rebuilding scenes:
    python script/bench_script/analyze_natural_visibility.py --plot-only \
        --out-dir ./visibility_phase1
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

from setup_paths import setup_paths
setup_paths()

bench_root = Path(os.environ["BENCH_ROOT"])
robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
os.chdir(robotwin_root)

import yaml
from envs import CONFIGS_PATH
from visualize_task_scene import get_env_class, get_embodiment_config

CAMERA = "countertop_camera"
TARGET_ATTRS = ("target_obj", "target", "mouse")

# Bucket taxonomy (must match DEFAULT_VISIBILITY_BUCKETS in envs/_base_task.py):
#   not_visible        frac == 0
#   heavily_occluded   0    < frac < 0.25
#   mostly_occluded    0.25 <= frac < 0.5
#   partially_occluded 0.5  <= frac < 0.9
#   fully_visible      0.9  <= frac
BUCKET_ORDER = ["not_visible", "heavily_occluded", "mostly_occluded",
                "partially_occluded", "fully_visible"]
BUCKET_BOUNDARIES = [0.25, 0.5, 0.9]  # interior guide lines (plus 0 for not_visible)
BUCKET_COLORS = {
    "not_visible": "#C44E52",        # red
    "heavily_occluded": "#DD8452",   # orange
    "mostly_occluded": "#CCB974",    # yellow
    "partially_occluded": "#55A868", # green
    "fully_visible": "#4C72B0",      # blue
}


def _resolve_target(env):
    for attr in TARGET_ATTRS:
        if hasattr(env, attr):
            return getattr(env, attr)
    raise SystemExit("Could not find a target actor attribute on the env")


def build_cfg(task_name, base_config, seed, dr_overrides):
    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        config_path = bench_root / "bench_task_config" / f"{base_config}.yml"
    else:
        config_path = Path(f"./task_config/{base_config}.yml")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = 0
    cfg["now_ep_num"] = 0
    cfg["seed"] = int(seed)
    cfg["need_plan"] = False          # no planning/rollout for a t=0 measurement
    cfg["save_data"] = False

    cfg.setdefault("domain_randomization", {})
    cfg["domain_randomization"].update(dr_overrides)

    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(name):
        robot_file = _embodiment_types[name]["file_path"]
        if robot_file is None:
            raise SystemExit("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])
    return cfg


# Denominator: clean (no table clutter), density 0.
DR_CLEAN = {"cluttered_table": False, "obstacle_density": 0, "clean_background_rate": 0}


def dr_dense(density, tall_only=False, handcrafted=False):
    dr = {"cluttered_table": True, "obstacle_density": int(density), "clean_background_rate": 0}
    if tall_only:
        dr["obstacle_tall_only"] = True
    if handcrafted:
        dr["obstacle_handcrafted"] = True
    return dr


def save_overlay(env, mask, out_path, header):
    """Save [countertop RGB | same view with the target's visible pixels in red],
    with a one-line header. Must be called while the render buffers are current
    (i.e. after measure_target_visibility, before close_env)."""
    import imageio
    import cv2
    rgb = env.cameras.get_rgb()[CAMERA]["rgb"]
    overlay = rgb.copy()
    overlay[mask] = (0.5 * np.array([255, 0, 0]) + 0.5 * overlay[mask]).astype(np.uint8)
    side = np.ascontiguousarray(np.concatenate([rgb, overlay], axis=1))
    cv2.putText(side, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 0), 2, cv2.LINE_AA)
    imageio.imwrite(out_path, side)


def run_sweep(args):
    env_class = get_env_class(args.task_name, bench_subdir=args.bench_subdir)
    env = env_class()  # reuse one instance across builds (collect_data pattern)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    densities = [int(d) for d in args.densities.split(",")]

    print(f"task={args.task_name} base={args.base_config} camera={CAMERA}")
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  densities={densities}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        for si, seed in enumerate(seeds):
            # --- denominator: clean build (skip seed if the build is unstable) ---
            try:
                env.setup_demo(**build_cfg(args.task_name, args.base_config, seed, DR_CLEAN))
                target = _resolve_target(env)
                clean_pose = np.array(target.actor.get_pose().p)
                res_clean = env.measure_target_visibility(target, camera_name=CAMERA)
                full_px = res_clean["visible_pixel_count"]
                if args.save_images:
                    save_overlay(env, res_clean["mask"], img_dir / f"seed{seed:04d}_clean.png",
                                 f"seed {seed} CLEAN (denominator)  full_px={full_px}")
            except Exception as e:
                print(f"[seed {seed}] clean build failed ({type(e).__name__}: {e}); skipping seed")
                safe_close()
                continue
            safe_close()

            if full_px <= 0:
                print(f"[seed {seed}] WARNING: full_target_px=0 on clean scene; skipping seed.")
                continue

            for density in densities:
                try:
                    env.setup_demo(**build_cfg(args.task_name, args.base_config, seed,
                                               dr_dense(density, args.tall_only, args.handcrafted)))
                    target = _resolve_target(env)
                    # guard: target pose must match the clean build for the denominator to be valid
                    pose_ok = bool(np.allclose(np.array(target.actor.get_pose().p), clean_pose, atol=1e-4))
                    res = env.measure_target_visibility(
                        target, camera_name=CAMERA, denominator=full_px,
                    )  # default DEFAULT_VISIBILITY_BUCKETS taxonomy
                    if args.save_images:
                        tag = "hand" if args.handcrafted else "tall" if args.tall_only else "mix"
                        save_overlay(
                            env, res["mask"],
                            img_dir / f"seed{seed:04d}_d{density:02d}_{tag}.png",
                            f"seed {seed} d={density} {tag}  vis_px={res['visible_pixel_count']} "
                            f"full={full_px} frac={res['visible_fraction']:.3f} "
                            f"{res['bucket']} pose_match={pose_ok}",
                        )
                except Exception as e:
                    print(f"[seed {seed} d{density}] build failed ({type(e).__name__}: {e}); skipping")
                    safe_close()
                    continue
                safe_close()

                rec = {
                    "seed": int(seed),
                    "density": int(density),
                    "full_px": int(full_px),
                    "visible_px": int(res["visible_pixel_count"]),
                    "visible_fraction": float(res["visible_fraction"]),
                    "bucket": res["bucket"],
                    "in_fov": bool(res["in_fov"]),
                    "pose_match": pose_ok,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            print(f"[{si+1}/{len(seeds)}] seed {seed}: full_px={full_px} done")

    env.close_env()
    print(f"\nsweep complete -> {jsonl_path}")
    return jsonl_path


def load_records(out_dir):
    jsonl_path = Path(out_dir) / "records.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"No records at {jsonl_path}; run the sweep first.")
    return [json.loads(l) for l in open(jsonl_path) if l.strip()]


def analyze(out_dir, bins, selectable_threshold):
    import csv
    import math as _m
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = load_records(out_dir)
    # Keep ALL records. Pose drift (clutter nudging the target during settling) can
    # push visible_fraction above 1 against the clean-build denominator; we clamp to
    # [0, 1] rather than dropping, so genuinely occluded/hidden scenes (the whole
    # point) are still counted. A clamped frac>1 just means "fully visible".
    n_drift = sum(1 for r in recs if not r.get("pose_match", True))
    n_over = sum(1 for r in recs if r["visible_fraction"] > 1.0)
    if n_drift or n_over:
        print(f"NOTE: {n_drift}/{len(recs)} records had target pose drift; "
              f"{n_over} had frac>1 -> clamped to 1.0 (kept, not dropped).")

    densities = sorted({r["density"] for r in recs})
    buckets = BUCKET_ORDER
    edges = np.linspace(0.0, 1.0, bins + 1)   # equal-width bins over [0, 1]

    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})

    def clamped(r):
        return min(max(r["visible_fraction"], 0.0), 1.0)

    # Re-derive the bucket from the clamped visible_fraction (not the stored label),
    # so the current taxonomy applies even when re-plotting older runs.
    def classify(frac):
        if frac <= 0.0:
            return "not_visible"
        for name, hi in zip(buckets[1:], BUCKET_BOUNDARIES + [float("inf")]):
            if frac < hi:
                return name
        return buckets[-1]

    def fracs_for(d):
        return [clamped(r) for r in recs if r["density"] == d]

    def bucket_counts(d):
        sub = [r for r in recs if r["density"] == d]
        n = len(sub)
        counts = {b: 0 for b in buckets}
        for r in sub:
            counts[classify(clamped(r))] += 1
        return n, counts

    # ---- numeric equal-bin histogram table (stdout + CSV) ----
    print(f"\n================ Numeric histogram ({bins} equal bins over [0,1]) ================")
    csv_path = Path(out_dir) / "bin_counts.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["density", "bin_lo", "bin_hi", "count"])
        for d in densities:
            counts, _ = np.histogram(fracs_for(d), bins=edges)
            n = int(counts.sum())
            print(f"\nobstacle_density = {d}  (n={n})")
            for i, c in enumerate(counts):
                lo, hi = edges[i], edges[i + 1]
                bar = "#" * int(c)
                print(f"  [{lo:4.2f}, {hi:4.2f}) {c:>4}  {bar}")
                writer.writerow([d, f"{lo:.4f}", f"{hi:.4f}", int(c)])
    print(f"\nsaved {csv_path}")

    # ---- histograms: equal bins, max 3 panels per row, annotated with bucket % ----
    ncols = min(3, len(densities))
    nrows = _m.ceil(len(densities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 4.8 * nrows), squeeze=False)
    flat = axes.flatten()
    for ax, d in zip(flat, densities):
        n, counts = bucket_counts(d)
        ax.hist(fracs_for(d), bins=edges, color="#4C72B0", edgecolor="white")
        for x in BUCKET_BOUNDARIES:
            ax.axvline(x, color="0.4", ls="--", lw=1.5)
        ax.set_title(f"obstacle_density = {d}   (n={n})")
        ax.set_xlabel("visible_fraction")
        ax.set_ylabel("seed count")
        ax.set_xlim(-0.03, 1.05)
        txt = "\n".join(f"{b}: {(counts[b] / n if n else 0):.0%}" for b in buckets)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    for ax in flat[len(densities):]:
        ax.axis("off")
    fig.suptitle("Natural visibility distribution on countertop camera\n"
                 "dashed guides at bucket boundaries 0.25 / 0.5 / 0.9", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    hist_path = Path(out_dir) / "histograms.png"
    fig.savefig(hist_path, dpi=130)
    plt.close(fig)
    print(f"saved {hist_path}")

    # ---- dedicated pooled histogram of visible_fraction (all densities), bar counts labeled ----
    all_fr = [r["visible_fraction"] for r in recs]
    counts, _ = np.histogram(all_fr, bins=edges)
    fig3, ax3 = plt.subplots(figsize=(11, 5.5))
    centers = (edges[:-1] + edges[1:]) / 2
    ax3.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95,
            color="#4C72B0", edgecolor="white")
    for x in BUCKET_BOUNDARIES:
        ax3.axvline(x, color="0.4", ls="--", lw=1.5)
    for c, cx in zip(counts, centers):
        if c > 0:
            ax3.text(cx, c, str(int(c)), ha="center", va="bottom", fontsize=10)
    ax3.set_xlabel("visible_fraction")
    ax3.set_ylabel("count (all seeds × densities)")
    ax3.set_xlim(-0.03, 1.05)
    ax3.set_xticks(np.round(edges, 2))
    ax3.tick_params(axis="x", labelrotation=90, labelsize=9)
    dens_txt = ",".join(str(d) for d in densities)
    ax3.set_title(f"Pooled visible_fraction histogram — {bins} equal bins  "
                  f"(densities {dens_txt}, n={len(all_fr)})\n"
                  "dashed guides at bucket boundaries 0.25 / 0.5 / 0.9")
    fig3.tight_layout()
    pooled_path = Path(out_dir) / "fraction_histogram.png"
    fig3.savefig(pooled_path, dpi=130)
    plt.close(fig3)
    print(f"saved {pooled_path}")

    # ---- summary: stacked bucket proportions vs density (headline figure) ----
    fig2, ax2 = plt.subplots(figsize=(max(8, 1.6 * len(densities) + 4), 5.8))
    x = np.arange(len(densities))
    bottom = np.zeros(len(densities))
    for b in buckets:
        vals = np.array([bucket_counts(d)[1][b] / (bucket_counts(d)[0] or 1) for d in densities])
        ax2.bar(x, vals, bottom=bottom, label=b, color=BUCKET_COLORS[b], edgecolor="white")
        for xi, (v, bo) in enumerate(zip(vals, bottom)):
            if v > 0.03:
                ax2.text(xi, bo + v / 2, f"{v:.0%}", ha="center", va="center",
                         color="white", fontsize=12, fontweight="bold")
        bottom += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in densities])
    ax2.set_xlabel("obstacle_density")
    ax2.set_ylabel("fraction of seeds")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Bucket proportions vs clutter density", fontsize=15)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3, frameon=False)
    fig2.tight_layout()
    summary_fig = Path(out_dir) / "bucket_proportions.png"
    fig2.savefig(summary_fig, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {summary_fig}")

    # ---- per-bucket readout (selectable vs needs-occluder) ----
    summary = {}
    print("\n================ Per-bucket natural mass (acceptance rate) ================")
    header = f"{'density':>8} | " + " | ".join(f"{b:>18}" for b in buckets) + " |  needs-occluder"
    print(header)
    print("-" * len(header))
    for d in densities:
        n, counts = bucket_counts(d)
        row = {"n": n, "buckets": {}}
        cells, needs = [], []
        for b in buckets:
            c = counts[b]
            frac = c / n if n else 0.0
            row["buckets"][b] = {"count": c, "fraction": frac,
                                 "status": ("selectable" if frac >= selectable_threshold
                                            else "rare" if frac > 0 else "empty")}
            cells.append(f"{c:>4} ({frac:>5.1%})")
            # any non-fully_visible bucket the natural distribution can't reach needs the occluder
            if b != "fully_visible" and frac < selectable_threshold:
                needs.append(b)
        summary[str(d)] = row
        print(f"{d:>8} | " + " | ".join(f"{c:>18}" for c in cells) + f" |  {', '.join(needs) or '-'}")

    print(f"\nselectable threshold = {selectable_threshold:.0%} of seeds; "
          f"non-fully_visible buckets below it need the occluder.")
    summary_path = Path(out_dir) / "bucket_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"buckets": buckets, "bins": bins,
                   "selectable_threshold": selectable_threshold,
                   "per_density": summary}, f, indent=2)
    print(f"saved {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_name", nargs="?", default="put_mouse_on_pad")
    ap.add_argument("--bench-subdir", default="office")
    ap.add_argument("--base-config", default="bench_demo_office_d8")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--densities", default="30")
    ap.add_argument("--bins", type=int, default=20, help="equal-width histogram bins over [0,1]")
    ap.add_argument("--tall-only", action="store_true",
                    help="place table clutter from the tall obstacle pool only (diagnostic)")
    ap.add_argument("--handcrafted", action="store_true",
                    help="restrict table clutter to the handcrafted tallest/widest set (diagnostic)")
    ap.add_argument("--save-images", action="store_true",
                    help="save per-run countertop overlays (RGB | visible mouse pixels in red) to <out-dir>/images/")
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="./visibility_phase1")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip the sweep; just (re)build figures from records.jsonl.")
    args = ap.parse_args()

    if not args.plot_only:
        run_sweep(args)
    analyze(args.out_dir, args.bins, args.selectable_threshold)


if __name__ == "__main__":
    main()
