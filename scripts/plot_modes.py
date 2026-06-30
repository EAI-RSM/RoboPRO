#!/usr/bin/env python
"""The PoC proof: overlay the per-mode end-effector trajectories for each scene.

For every task it reads the collected episodes (one per mode), pulls the active
arm's EE path from the standard `endpose` group, and plots them on a shared
top-down (x-y) + side (x-z) view — color-coded by mode. Four clearly different
paths from ONE scene = multimodal behavior. Also overlays the target object's
path (from targeted_state) to show every mode reaches the same destination.

  python plot_modes.py --root ../multimodal_data/runs --out ../multimodal_data
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE_COLORS = {"direct": "#444444", "high_arc": "#1f77b4",
               "swing_left": "#2ca02c", "swing_right": "#d62728"}


def active_ee_path(f):
    """Return the [T,3] EE position of whichever arm actually moved (largest path
    length), from the standard endpose group."""
    best, best_len = None, -1.0
    for arm in ("left_endpose", "right_endpose"):
        key = f"endpose/{arm}"
        if key not in f:
            continue
        ep = f[key][:]                       # [T, 6 or 7] (xyz + orientation)
        xyz = np.asarray(ep)[:, :3]
        plen = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
        if plen > best_len:
            best, best_len = xyz, plen
    return best


def target_path(f):
    """Target object [T,3] path from the targeted_state trace, if present."""
    if "targeted_state" not in f:
        return None
    g = f["targeted_state"]
    names = [n.decode() if isinstance(n, bytes) else n for n in g["actor_names"][:]]
    ent = json.loads(g.attrs["entities_json"]) if "entities_json" in g.attrs else {}
    tgt = ent.get("target")
    if tgt not in names:
        return None
    return g["actor_pos"][:, names.index(tgt), :]


def plot_task(task, task_dir, out_path):
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle(f"{task} — one scene, {len(MODE_COLORS)} planner modes", fontsize=14, y=0.98)
    plotted = 0
    for mode, color in MODE_COLORS.items():
        hp = task_dir / mode / "data" / "episode0.hdf5"
        if not hp.exists():
            continue
        with h5py.File(hp, "r") as f:
            ee = active_ee_path(f)
            tp = target_path(f)
            mm = f["multimodal"].attrs.get("outcome", "?") if "multimodal" in f else "?"
        if ee is None:
            continue
        lbl = f"{mode} ({mm})"
        ax_top.plot(ee[:, 0], ee[:, 1], "-", color=color, lw=2.0, label=lbl, alpha=0.9)
        ax_top.plot(ee[0, 0], ee[0, 1], "o", color=color, ms=6)
        ax_side.plot(ee[:, 0], ee[:, 2], "-", color=color, lw=2.0, alpha=0.9)
        if tp is not None:
            ax_top.plot(tp[:, 0], tp[:, 1], ":", color=color, lw=1.2, alpha=0.7)
        plotted += 1

    ax_top.set_title("top-down (x-y) — EE path (—), object path (···)")
    ax_top.set_xlabel("x (m)"); ax_top.set_ylabel("y (m)"); ax_top.axis("equal"); ax_top.grid(alpha=.3)
    ax_top.legend(fontsize=9, loc="best")
    ax_side.set_title("side (x-z) — transport height")
    ax_side.set_xlabel("x (m)"); ax_side.set_ylabel("z (m)"); ax_side.grid(alpha=.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}  ({plotted} modes)")
    return plotted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="multimodal_data/runs")
    ap.add_argument("--out", required=True, help="output dir for the PNGs")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tasks = sorted([p.name for p in root.iterdir() if p.is_dir()]) if root.exists() else []
    print(f"plotting {len(tasks)} task(s) from {root}")
    for task in tasks:
        plot_task(task, root / task, out / f"modes_{task}.png")


if __name__ == "__main__":
    main()
