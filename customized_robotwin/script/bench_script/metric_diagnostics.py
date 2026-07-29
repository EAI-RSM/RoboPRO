"""Joint-continuity and clearance diagnostics for the metric pipeline."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lib.continuity import _NEIGH8, _wrap_linf
from lib.labeling import FREE
from lib.obstacles import occluder_slice_polys
from lib.plotting import _scene_anchor_markers

GATE_TAU_CANDIDATES = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def _jump_field(free, qfield):
    """For a given per-cell config field, the 8-connected FREE-FREE joint-jump statistics the gate
    would see. Returns (cellmax (ny,nx) NaN off-FREE = max jump to any FREE neighbour, edges 1D =
    each undirected FREE-FREE edge jump once). Cells with non-finite q (e.g. no warm branch) skip."""
    ny, nx = free.shape
    cellmax = np.full(free.shape, np.nan)
    edges = []
    for iy in range(ny):
        for ix in range(nx):
            if not free[iy, ix] or not np.all(np.isfinite(qfield[iy, ix])):
                continue
            qa = qfield[iy, ix]
            best = 0.0
            for dy, dx in _NEIGH8:
                jy, jx = iy + dy, ix + dx
                if (0 <= jy < ny and 0 <= jx < nx and free[jy, jx]
                        and np.all(np.isfinite(qfield[jy, jx]))):
                    d = _wrap_linf(qa, qfield[jy, jx])
                    best = max(best, d)
                    if (jy, jx) > (iy, ix):    # count each undirected edge once
                        edges.append(d)
            cellmax[iy, ix] = best
    return cellmax, np.asarray(edges)


def _print_jump_stats(tag, edges):
    if edges.size == 0:
        print(f"[phase0] {tag}: no FREE-FREE edges"); return
    pct = np.percentile(edges, [50, 90, 95, 99])
    passfrac = "  ".join(f"{t:.2f}:{float((edges <= t).mean()):.3f}" for t in GATE_TAU_CANDIDATES)
    print(f"[phase0] {tag}: edges={edges.size}  jump(rad) median={pct[0]:.3f} p90={pct[1]:.3f} "
          f"p95={pct[2]:.3f} p99={pct[3]:.3f} max={edges.max():.3f}")
    print(f"[phase0] {tag}: pass-frac (jump<=tau)  {passfrac}")


def _draw_pair(fig, axmap, axhist, extent, cellmax, edges, title):
    im = axmap.imshow(cellmax, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=axmap, fraction=0.046, pad=0.04, label="max joint jump to a FREE neighbour (rad)")
    axmap.set_title(f"{title}: per-cell roughness\nbright = branch switch"); axmap.set_xlabel("x (m)")
    axmap.set_ylabel("y (m)")
    if edges.size:
        axhist.hist(edges, bins=60, color="#3949ab"); axhist.set_yscale("log")
        for t in GATE_TAU_CANDIDATES:
            axhist.axvline(t, color="#d84315", ls="--", lw=1)
            axhist.text(t, axhist.get_ylim()[1], f"{t:.2f}", rotation=90, va="top", ha="right",
                        fontsize=7, color="#d84315")
    axhist.set_title(f"{title}: edge jump dist\n(dashed = candidate --gate-tau)")
    axhist.set_xlabel("joint jump (rad)"); axhist.set_ylabel("edge count (log)")


def phase0_gate_diagnostic(out_dir, args, XX, YY, label, qfield, q_warm=None, z_slice=0.0):
    """PHASE 0: is the joint-space edge gate trustworthy on this slice? (single-slice diagnostic)

    For every 8-connected pair of FREE cells, measure the joint jump = max over joints of the
    wrapped |dq| (radians) -- the exact quantity the gate would threshold. A gate is meaningful iff
    that jump distribution is a smooth LOW bulk (adjacent cells share an IK branch) with a clean gap
    to a few HIGH-jump seam edges (real branch switches); the gap is where --gate-tau belongs. A
    broad smear with no gap instead means curobo is branch-hopping on seed noise -> a gate would cut
    good edges.

    RAW field = curobo's single best-cost solution per cell (what --warm-start off measures). If a
    warm field is supplied (--warm-start), it is the continuity-propagated branch assignment; the
    two are drawn/printed side by side. If the warm field's tail COLLAPSES relative to raw, the raw
    smear was branch-hopping noise (fixable, gate becomes usable); if the tail SURVIVES, the seams
    are genuine config-space marginality that a warm-start can't remove. Read-only otherwise."""
    free = label == FREE
    nfree = int(free.sum())
    cellmax_r, edges_r = _jump_field(free, qfield)
    print(f"[phase0] FREE cells={nfree}")
    _print_jump_stats("RAW ", edges_r)

    have_warm = q_warm is not None
    if have_warm:
        cellmax_w, edges_w = _jump_field(free, q_warm)
        missing = int((free & ~np.all(np.isfinite(q_warm), axis=-1)).sum())
        _print_jump_stats("WARM", edges_w)
        print(f"[phase0] WARM: {missing} FREE cells had no converged candidate branch (left blank)")
        if edges_r.size and edges_w.size:
            print(f"[phase0] tail comparison p99  raw={np.percentile(edges_r,99):.3f} -> "
                  f"warm={np.percentile(edges_w,99):.3f}  (collapse => noise, survives => real seams)")
    else:
        print("[phase0] read: smooth bulk + clean gap => gate meaningful; broad smear, no gap => "
              "branch-hopping, re-run with --warm-start to test if it collapses")

    stem = f"gate_diagnostic_seed{args.seed}_z{z_slice:.2f}_{args.arm}" + ("_warm" if have_warm else "")
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    if have_warm:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        _draw_pair(fig, axes[0][0], axes[0][1], extent, cellmax_r, edges_r, "RAW (best-cost)")
        _draw_pair(fig, axes[1][0], axes[1][1], extent, cellmax_w, edges_w, "WARM (continuity)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        _draw_pair(fig, axes[0], axes[1], extent, cellmax_r, edges_r, "RAW (best-cost)")
    fig.suptitle(f"Phase 0 gate diagnostic  |  seed {args.seed}, z={z_slice:.2f}, arm={args.arm}"
                 + ("  |  raw vs warm-start" if have_warm else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / f"{stem}.png", dpi=130); plt.close(fig)
    print(f"[phase0] wrote {stem}.png")


def phase1_stack_report(out_dir, args, xs, ys, zs, label, qfield, q_warm_vol=None):
    """PHASE 1 (stack): per-slice reachability + config-roughness across the z-stack. Prints a
    compact per-height table (FREE count + raw/warm joint-jump median/p99) and writes a montage of
    per-slice roughness maps on a shared 0..1 rad colour scale, so heights are directly comparable.
    No 3D gate yet -- this only verifies the stack builds and shows how FREE-space and IK smoothness
    evolve with height (e.g. does the arm still reach up near the bottle's top for a climb-over?)."""
    nz = len(zs)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    cm_raw, cm_warm = [], []
    header = "[stack]    z    FREE    raw med/p99" + ("     warm med/p99" if q_warm_vol is not None else "")
    print(header)
    for iz, z in enumerate(zs):
        free = label[iz] == FREE
        cmr, er = _jump_field(free, qfield[iz]); cm_raw.append(cmr)
        line = f"[stack] {z:5.2f}  {int(free.sum()):5d}    "
        line += (f"{np.median(er):.3f}/{np.percentile(er, 99):.3f}" if er.size else "  -  ")
        if q_warm_vol is not None:
            cmw, ew = _jump_field(free, q_warm_vol[iz]); cm_warm.append(cmw)
            line += "      " + (f"{np.median(ew):.3f}/{np.percentile(ew, 99):.3f}" if ew.size else "  -  ")
        print(line)

    def _montage(cmlist, tag):
        ncol = min(5, nz)
        nrow = int(np.ceil(nz / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
        im = None
        for k in range(nrow * ncol):
            ax = axes[k // ncol][k % ncol]
            if k >= nz:
                ax.axis("off"); continue
            im = ax.imshow(cmlist[k], origin="lower", extent=extent, cmap="viridis",
                           vmin=0.0, vmax=1.0, aspect="equal")
            ax.set_title(f"z={zs[k]:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        if im is not None:
            fig.colorbar(im, ax=list(axes.ravel()), fraction=0.02, pad=0.02,
                         label="max joint jump to neighbour (rad, capped at 1.0)")
        fig.suptitle(f"Phase 1 stack roughness ({tag})  |  seed {args.seed}, arm {args.arm}", fontsize=12)
        stem = f"stack_roughness_{tag}_seed{args.seed}_{args.arm}"
        fig.savefig(out_dir / f"{stem}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f"[stack] wrote {stem}.png")

    _montage(cm_raw, "raw")
    if q_warm_vol is not None:
        _montage(cm_warm, "warm")


def _vertical_edges_by_z(free_vol, q_vol):
    """List (len nz-1) of arrays: FREE-FREE VERTICAL-edge joint jumps for each z->z+1 transition
    (each voxel vs the voxel directly above it). This is the continuity the per-slice 2D propagation
    never enforces and the 3D propagation does -- the edges a climb-over route rides on."""
    nz = free_vol.shape[0]
    out = []
    for iz in range(nz - 1):
        both = free_vol[iz] & free_vol[iz + 1]
        e = []
        for iy, ix in zip(*np.nonzero(both)):
            qa, qb = q_vol[iz, iy, ix], q_vol[iz + 1, iy, ix]
            if np.all(np.isfinite(qa)) and np.all(np.isfinite(qb)):
                e.append(_wrap_linf(qa, qb))
        out.append(np.asarray(e))
    return out


def _edge_stats(e):
    """Compact machine-readable summary of an edge-jump array."""
    if e.size == 0:
        return {"edges": 0}
    return {"edges": int(e.size), "median": round(float(np.median(e)), 4),
            "p90": round(float(np.percentile(e, 90)), 4), "p99": round(float(np.percentile(e, 99)), 4),
            "max": round(float(e.max()), 4),
            "pass_at": {f"{t:.2f}": round(float((e <= t).mean()), 4) for t in GATE_TAU_CANDIDATES}}


def phase2_vertical_report(out_dir, args, zs, free_vol, q_warm_2d, q_warm_3d):
    """PHASE 2 check: does the 3D propagation actually buy vertical continuity? Compares the joint jump
    across vertical (between-slice) FREE-FREE edges for the per-slice-2D-propagated field vs the
    3D-propagated field. The 2D field never saw vertical neighbours, so its columns are branch-
    inconsistent -> a fat vertical-jump tail; the 3D field should collapse that tail (leaving only real
    seams). Prints stats, writes an overlaid histogram AND a phase2_vertical_*.json (overall + per-z
    transition) so the result is machine-readable, not just a picture."""
    by2 = _vertical_edges_by_z(free_vol, q_warm_2d)
    by3 = _vertical_edges_by_z(free_vol, q_warm_3d)
    e2 = np.concatenate(by2) if by2 else np.array([])
    e3 = np.concatenate(by3) if by3 else np.array([])
    for tag, e in (("2D-per-slice", e2), ("3D-propagated", e3)):
        if e.size:
            pct = np.percentile(e, [50, 90, 99])
            print(f"[phase2] vertical jump {tag}: edges={e.size} median={pct[0]:.3f} p90={pct[1]:.3f} "
                  f"p99={pct[2]:.3f} max={e.max():.3f}  pass@0.35={float((e <= 0.35).mean()):.3f}")

    stem = f"phase2_vertical_seed{args.seed}_{args.arm}"
    data = {"config": {"seed": args.seed, "arm": args.arm, "zmin": args.zmin, "zmax": args.zmax,
                       "zres": args.zres, "gate_tau_candidates": list(GATE_TAU_CANDIDATES)},
            "vertical_2d_per_slice": _edge_stats(e2),
            "vertical_3d_propagated": _edge_stats(e3),
            "per_z_transition": [{"z_lo": round(float(zs[i]), 3), "z_hi": round(float(zs[i + 1]), 3),
                                  "stats_2d": _edge_stats(by2[i]), "stats_3d": _edge_stats(by3[i])}
                                 for i in range(len(by2))]}
    (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2))

    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0, np.pi, 60)
    if e2.size:
        ax.hist(e2, bins=bins, alpha=0.5, color="#d84315", label="2D per-slice (no vertical continuity)")
    if e3.size:
        ax.hist(e3, bins=bins, alpha=0.5, color="#3949ab", label="3D propagated (vertical continuity)")
    ax.set_yscale("log")
    for t in GATE_TAU_CANDIDATES:
        ax.axvline(t, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("vertical-edge joint jump (rad)"); ax.set_ylabel("edge count (log)")
    ax.set_title(f"Phase 2: vertical continuity 2D vs 3D propagation  |  seed {args.seed}, arm {args.arm}")
    ax.legend()
    fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[phase2] wrote {stem}.png + {stem}.json")


def phase3_clearance_report(out_dir, args, xs, ys, zs, label, edt, foots, tgt_p=None, ee_xyz=None):
    """PHASE 3 sanity: per-slice montage of the 3D occluder-clearance field (0 inside a footprint,
    growing outward; opening up above the bottle top where there is no occluder), footprint outline
    overlaid where it exists. Prints per-z clearance-over-FREE stats so the height profile is legible."""
    nz = len(zs)
    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    disp = np.where(np.isfinite(edt), edt, np.nan)
    vmax = float(np.nanpercentile(disp, 99)) if np.isfinite(disp).any() else 1.0
    print("[phase3]    z    FREE   clearance over FREE (min/med/max, m)")
    for iz, z in enumerate(zs):
        free = label[iz] == FREE
        fe = edt[iz][free]
        fe = fe[np.isfinite(fe)]
        s = (f"{fe.min():.3f}/{np.median(fe):.3f}/{fe.max():.3f}" if fe.size else "  -  ")
        print(f"[phase3] {z:5.2f}  {int(free.sum()):5d}   {s}")
    ncol = min(5, nz)
    nrow = int(np.ceil(nz / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), squeeze=False)
    im = None
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k >= nz:
            ax.axis("off"); continue
        im = ax.imshow(disp[k], origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=vmax,
                       aspect="equal")
        if foots:
            from matplotlib.patches import Polygon as MplPolygon
            for f in foots:
                # outline the occluder as it exists AT THIS HEIGHT: the true cross-section under
                # --occ-shape mesh (so the neck visibly narrows), else the constant footprint
                loops = occluder_slice_polys(f, float(zs[k])) if args.occ_shape == "mesh" else []
                if loops:
                    for l in loops:
                        ax.add_patch(MplPolygon(l, closed=True, fill=False, edgecolor="red", lw=1.2))
                elif f["poly"] is not None and f["zlo"] - 1e-9 <= zs[k] <= f["zhi"] + 1e-9:
                    ax.add_patch(MplPolygon(f["poly"], closed=True, fill=False, edgecolor="red", lw=1.2))
        _scene_anchor_markers(ax, tgt_p, ee_xyz, args.arm)
        ax.set_title(f"z={zs[k]:.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=list(axes.ravel()), fraction=0.02, pad=0.02, label="clearance to occluder (m)")
    fig.suptitle(f"Phase 3 occluder clearance  |  seed {args.seed}, arm {args.arm}", fontsize=12)
    stem = f"stack_clearance_seed{args.seed}_{args.arm}"
    fig.savefig(out_dir / f"{stem}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[phase3] wrote {stem}.png")
