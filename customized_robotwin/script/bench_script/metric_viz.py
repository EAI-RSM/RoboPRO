"""Reports and visualisations for the clearance metric pipeline."""

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from lib.labeling import BEYOND, FREE, LABEL_NAMES, OBSTACLE
from lib.obstacles import occluder_slice_polys, surface_distance_to_occluders
from lib.plotting import (
    _draw_eps_sphere, _draw_ground_plane, _draw_occluder_solids_3d, _equal_aspect_3d,
    _line_axis, _scene_anchor_markers, _true_aspect_3d,
)
from lib.scene_constants import OCC_HALF_FOOTPRINT

LABEL_COLORS = {BEYOND: "#9e9e9e", OBSTACLE: "#d84315", FREE: "#2e7d32"}


def feasibility(eps_star, merged, r):
    """Derive an embodiment feasibility verdict at READ time. eps* itself stays robot-free;
    the gripper half-width r is only compared here, never baked into the metric."""
    if not merged:
        return "INACCESSIBLE", None
    if np.isinf(eps_star):
        return "unbounded (no obstacles)", None
    margin = eps_star - r
    return ("fits" if margin >= 0 else "INFEASIBLE"), margin


def _overlays(ax, polys, occ_ps, tgt_p, pad_xy, seed_t_xy, seed_p_xy, bott_xy, path_xy):
    if path_xy is not None and len(path_xy) > 1:      # widest-path route, drawn under the markers
        px, py = zip(*path_xy)
        ax.plot(px, py, "-", color="yellow", lw=2.2, alpha=0.9, label="widest path")
    if polys is not None and any(p is not None for p in polys):
        from matplotlib.patches import Polygon as MplPolygon
        first = True
        for p in polys:                        # true posed mesh footprint per occluder
            if p is None:
                continue
            ax.add_patch(MplPolygon(p, closed=True, fill=False, edgecolor="red", lw=2,
                                    label=("occluder" if first else None)))
            first = False
    else:                                      # fallback: circle of radius OCC_HALF_FOOTPRINT
        for i, op in enumerate(occ_ps):
            ax.add_patch(plt.Circle((op[0], op[1]), OCC_HALF_FOOTPRINT, fill=False,
                                    edgecolor="red", lw=2, label=("occluder" if i == 0 else None)))
    ax.plot(tgt_p[0], tgt_p[1], "b*", ms=16, label="target")
    ax.plot(pad_xy[0], pad_xy[1], "ms", ms=11, label="pad")
    if seed_t_xy is not None:
        ax.plot(*seed_t_xy, "o", mfc="none", mec="cyan", mew=2, ms=13, label="grasp seed")
    if seed_p_xy is not None:
        ax.plot(*seed_p_xy, "o", mfc="none", mec="magenta", mew=2, ms=13, label="pad seed")
    if bott_xy is not None:
        ax.plot(*bott_xy, "kX", ms=8, mew=1.5, label="bottleneck (eps*)")


def report(args, out_dir, XX, YY, label, edt, box_p, occ_ps, polys, tgt_p, pad_xy,
           seed_t_xy, seed_p_xy, eps_star, bott_xy, merged, boxed_dist, counts, path_xy):
    """Write summary (json + txt), cache the raster (npz -- keeps the BEYOND label for a later
    reach-edge channel), and save the two-panel figure (labels + clearance heatmap)."""
    verdict, margin = feasibility(eps_star, merged, args.gripper_r)
    tag = "topdown" if args.topdown else "sidegrasp"
    occ = "occ" if box_p is not None else "noocc"
    stem = f"clearance_seed{args.seed}_off{args.offset}_z{args.z:.2f}_{args.arm}_{tag}_{occ}"

    boxed_in = bool(boxed_dist is not None and boxed_dist <= args.boxed_in_radius)
    summary = {
        "eps_star_m": (None if np.isinf(eps_star) else round(float(eps_star), 4)),
        "eps_star_unbounded": bool(np.isinf(eps_star)),
        "merged": bool(merged),
        "bottleneck_xy": ([round(bott_xy[0], 4), round(bott_xy[1], 4)] if bott_xy else None),
        "boxed_in": boxed_in,
        "boxed_in_dist_m": (None if boxed_dist is None else round(float(boxed_dist), 4)),
        "gripper_r_m": args.gripper_r,
        "feasibility": verdict,
        "margin_m": (None if margin is None else round(float(margin), 4)),
        "label_counts": counts,
        "config": {"seed": args.seed, "offset": args.offset, "z": args.z, "arm": args.arm,
                   "res": args.res, "topdown": args.topdown, "occluder": box_p is not None},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))

    np.savez_compressed(out_dir / f"{stem}.npz",
                        label=label, edt=edt, XX=XX, YY=YY,
                        box_p=(box_p if box_p is not None else np.array([])),
                        occ_ps=(np.array(occ_ps) if occ_ps else np.array([])),
                        path_xy=(np.array(path_xy) if path_xy else np.array([])),
                        tgt_p=tgt_p, pad_xy=pad_xy, res=args.res)

    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 7))

    cmap = ListedColormap([LABEL_COLORS[BEYOND], LABEL_COLORS[OBSTACLE], LABEL_COLORS[FREE]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    axL.imshow(label, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect="equal")
    _overlays(axL, polys, occ_ps, tgt_p, pad_xy, seed_t_xy, seed_p_xy, bott_xy, path_xy)
    # eps*-radius circle at the bottleneck: it should just touch the nearest occluder
    # (eps* IS the distance from the bottleneck to that occluder), so a larger eps* = a
    # larger circle that barely kisses a bottle.
    if merged and bott_xy is not None and np.isfinite(eps_star) and eps_star > 0:
        axL.add_patch(plt.Circle(bott_xy, eps_star, fill=False, edgecolor="#00e5ff",
                                  ls="--", lw=1.8, label=f"eps* radius ({eps_star:.3f} m)"))
    from matplotlib.patches import Patch
    proxies = [Patch(color=LABEL_COLORS[c], label=LABEL_NAMES[c]) for c in (FREE, OBSTACLE, BEYOND)]
    h, _ = axL.get_legend_handles_labels()
    axL.legend(handles=proxies + h, loc="upper right", fontsize=8)
    axL.set_title("Three-way labels"); axL.set_xlabel("x (m)"); axL.set_ylabel("y (m)")

    disp = np.where(np.isfinite(edt), edt, np.nan)
    im = axR.imshow(disp, origin="lower", extent=extent, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="clearance to nearest OCCLUDER (m)")
    _overlays(axR, polys, occ_ps, tgt_p, pad_xy, seed_t_xy, seed_p_xy, bott_xy, path_xy)
    axR.legend(loc="upper right", fontsize=8)
    axR.set_title("Occluder clearance (m)"); axR.set_xlabel("x (m)"); axR.set_ylabel("y (m)")

    eps_txt = "inf" if np.isinf(eps_star) else (f"{eps_star:.3f} m" if merged else "0 (inaccessible)")
    fig.suptitle(f"Clearance metric  |  seed {args.seed}, offset {args.offset}, z={args.z:.2f}, "
                 f"arm={args.arm}, occluder {'ON' if box_p is not None else 'OFF'}\n"
                 f"eps* = {eps_txt}   gripper r = {args.gripper_r:.3f} m   ->  {verdict}"
                 + ("" if margin is None else f"  (margin {margin:+.3f} m)"),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / f"{stem}.png", dpi=130); plt.close(fig)
    print(f"[report] wrote {stem}.png / .json / .npz  ->  feasibility: {verdict}")


def _metric_path3d(out_dir, args, foots, occ_ps, g_xyz, p_xyz, bott_xyz, route_w, eps_star, merged,
                   tgt_p=None, ee_xyz=None, start_label="grasp seed", goal_label="pad seed",
                   route_label="gated widest path", metric_label="eps* (gated)",
                   title_context=None, stem=None, frame_xy=None, ground_z=None, view=None,
                   tool_link_xyz=None, dest_xyz=None, target_label="target bottle (spawn)",
                   geom_label="occluder"):
    """3D view of the gated climb-over route through the stack: occluder solids, the route, the scene
    anchors (target bottle spawn + the gripper's current pose), and the eps* sphere sitting on the
    bottleneck.

    start_label/goal_label name the two widest-path ends. They default to the metric's own
    grasp->pad pair so this tool's output is unchanged; the seed builder overrides them because
    its ends are different (gripper->grasp on the approach leg, post-lift->pad on the carry leg)
    and two figures both captioned "grasp/pad" are unreadable side by side.

    Every argument from frame_xy on is opt-in framing/annotation for a tabletop scene, and each
    defaults to the historical behaviour so the ring-benchmark callers are byte-identical:

      frame_xy/ground_z  fix the frame to the support surface instead of the data's own bounding
                         cube (see _true_aspect_3d) and draw that surface, so the scene stops
                         floating in empty space. Both are needed; either alone is ignored.
      view               explicit (elev, azim); the matplotlib default is a poor angle for a wide
                         shallow tabletop.
      tool_link_xyz      the END-EFFECTOR-LINK waypoint that pairs with tgt_p. The canonical
                         waypoints carry a fixed tool offset, so the leg endpoint sits well above
                         the target and the path looks like it misses; the dotted connector says
                         that gap is the tool offset, not an error.
      dest_xyz           where the object is going (a second scene anchor, like tgt_p).
      target_label       tgt_p's legend entry -- the target is not a bottle in every task.
      geom_label         what the obstacle set is called in the title."""
    framed = frame_xy is not None and ground_z is not None
    fig = plt.figure(figsize=(10, 7.5) if framed else (9.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    if framed:                              # lowest surface first: keeps the painter's order right
        _draw_ground_plane(ax, frame_xy[0], frame_xy[1], float(ground_z))
    _draw_occluder_solids_3d(ax, foots, args.occ_shape)
    if route_w and len(route_w) > 1:
        rx, ry, rz = zip(*route_w)
        ax.plot(rx, ry, rz, "-", color="gold", lw=2.5, label=route_label)
    ax.scatter(*g_xyz, c="cyan", marker="o", s=80, label=start_label)
    ax.scatter(*p_xyz, c="magenta", marker="s", s=70, label=goal_label)
    if tgt_p is not None:
        ax.scatter(tgt_p[0], tgt_p[1], tgt_p[2], c="blue", marker="*", s=200,
                   label=target_label)
        if tool_link_xyz is not None:
            ax.plot([tgt_p[0], tool_link_xyz[0]], [tgt_p[1], tool_link_xyz[1]],
                    [tgt_p[2], tool_link_xyz[2]], ":", color="0.35", lw=1.2,
                    label="tool offset (EE link -> target)")
    if dest_xyz is not None:
        ax.scatter(dest_xyz[0], dest_xyz[1], dest_xyz[2], c="limegreen", marker="P", s=110,
                   edgecolors="k", linewidths=0.5, label="destination (spawn)")
    if ee_xyz is not None:
        ax.scatter(ee_xyz[0], ee_xyz[1], ee_xyz[2], c="darkorange", marker="^", s=110,
                   edgecolors="k", linewidths=0.5, label=f"gripper now ({args.arm})")
    if bott_xyz is not None:
        ax.scatter(*bott_xyz, c="black", marker="X", s=80, label="bottleneck eps*")

    # eps* sphere: radius = the bottleneck's clearance, so it should just touch the nearest occluder
    exact = None
    drew_sphere = merged and bott_xyz is not None and np.isfinite(eps_star) and eps_star > 0
    if drew_sphere:
        _draw_eps_sphere(ax, bott_xyz, float(eps_star))
        exact = surface_distance_to_occluders(foots, bott_xyz)

    corners = [np.asarray(g_xyz), np.asarray(p_xyz), tgt_p, ee_xyz]
    for f in (foots or []):
        if f["poly"] is not None:
            corners += [np.array([f["poly"][:, 0].min(), f["poly"][:, 1].min(), f["zlo"]]),
                        np.array([f["poly"][:, 0].max(), f["poly"][:, 1].max(), f["zhi"]])]
    if route_w:
        corners += [np.asarray(route_w).min(axis=0), np.asarray(route_w).max(axis=0)]
    if drew_sphere:
        corners += [np.asarray(bott_xyz) - eps_star, np.asarray(bott_xyz) + eps_star]
    if dest_xyz is not None:
        corners.append(np.asarray(dest_xyz))
    # Framed mode pins x/y to the support surface so every figure shares one frame, and only lets
    # the z ceiling follow the data; otherwise fall back to the data's own bounding cube.
    if framed:
        zs = [float(np.asarray(c)[2]) for c in corners if c is not None]
        lo = (frame_xy[0][0], frame_xy[1][0], float(ground_z) - 0.01)
        hi = (frame_xy[0][1], frame_xy[1][1], max(max(zs) + 0.02, float(ground_z) + 0.20))
        framed = _true_aspect_3d(ax, lo, hi)
    if not framed:
        _equal_aspect_3d(ax, corners)
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])

    eps_txt = "inf" if (merged and np.isinf(eps_star)) else (f"{eps_star:.3f} m" if merged else "INACCESSIBLE")
    sub = ""
    if exact is not None:
        sub = (f"\ntrue mesh-surface distance at the bottleneck = {exact:.3f} m  "
               f"(EDT - true = {float(eps_star) - exact:+.3f} m, grid bias)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    context = "" if title_context is None else f"\n{title_context}"
    ax.set_title(f"2.5D metric route  |  seed {args.seed}, arm {args.arm}, "
                 f"{geom_label} geom={args.occ_shape}"
                 f"\n{metric_label} = {eps_txt}{sub}{context}", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    stem = stem or f"metric_path3d_seed{args.seed}_{args.arm}"
    # A wide, flat framed box pushes the z-axis label outside matplotlib's tight bbox for 3D axes.
    # Extra padding keeps it in frame; 0.1 is the matplotlib default, so unframed output is unchanged.
    fig.savefig(out_dir / f"{stem}.png", dpi=120, bbox_inches="tight",
                pad_inches=0.3 if framed else 0.1)
    plt.close(fig)
    return exact


def _viz_side_elevation(out_dir, args, XX, YY, zs, label, foots, g_xyz, p_xyz, route_w, eps_star,
                        merged, tgt_p=None, ee_xyz=None, start_label="grasp", goal_label="pad"):
    """SIDE ELEVATION: the 3D scene projected onto the vertical plane through grasp->pad. Background =
    labels sampled along that line at every height; the bottle is a red box (footprint projected onto
    the line x its z-range); the route arcs up and over. The most legible 'is it climbing over?' view."""
    xs, ys = XX[0], YY[:, 0]
    p0, u, L = _line_axis(g_xyz[:2], p_xyz[:2])
    ns = max(2, int(round(L / args.res)) + 1)
    svals = np.linspace(0, L, ns)
    img = np.full((len(zs), ns), BEYOND, dtype=np.int8)
    for j, s in enumerate(svals):
        px, py = p0 + s * u
        ix = int(np.argmin(np.abs(xs - px)))
        iy = int(np.argmin(np.abs(ys - py)))
        img[:, j] = label[:, iy, ix]
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = ListedColormap([LABEL_COLORS[BEYOND], LABEL_COLORS[OBSTACLE], LABEL_COLORS[FREE]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(img, origin="lower", extent=[0, L, zs.min(), zs.max()], cmap=cmap, norm=norm, aspect="auto")
    if foots:
        for f in foots:
            if f["poly"] is None:
                continue
            # mesh mode: per-height silhouette from the real cross-sections (the bottle necks in);
            # extruded mode: the constant-width box the metric actually used
            sil = []
            if args.occ_shape == "mesh" and f.get("mesh") is not None:
                # sampled on its OWN fine z-grid (not the coarse stack) so the neck reads as a curve
                for z in np.linspace(f["zlo"] + 1e-4, f["zhi"] - 1e-4, 60):
                    ss = [((loop - p0) @ u) for loop in occluder_slice_polys(f, float(z))]
                    if ss:
                        allss = np.concatenate(ss)
                        sil.append((float(allss.min()), float(allss.max()), float(z)))
            if sil:
                lo, hi, zv = zip(*sil)
                ax.plot(lo, zv, "-", color="red", lw=2)
                ax.plot(hi, zv, "-", color="red", lw=2)
                ax.plot([lo[0], hi[0]], [zv[0]] * 2, "-", color="red", lw=2)
                ax.plot([lo[-1], hi[-1]], [zv[-1]] * 2, "-", color="red", lw=2)
            else:
                proj = (f["poly"] - p0) @ u                   # footprint projected onto the line axis
                ax.add_patch(plt.Rectangle((float(proj.min()), f["zlo"]), float(proj.max() - proj.min()),
                                           f["zhi"] - f["zlo"], fill=False, edgecolor="red", lw=2))
    if route_w and len(route_w) > 1:
        rs = [float((np.asarray(r[:2]) - p0) @ u) for r in route_w]
        ax.plot(rs, [r[2] for r in route_w], "-", color="gold", lw=2.5, label="route")
    ax.plot(0, g_xyz[2], "o", color="cyan", ms=11, mec="k", label=start_label)
    ax.plot(L, p_xyz[2], "s", color="magenta", ms=10, mec="k", label=goal_label)
    # scene anchors, projected onto the same start->goal axis
    if tgt_p is not None:
        ax.plot(float((np.asarray(tgt_p[:2]) - p0) @ u), float(tgt_p[2]), "*", color="blue", ms=17,
                mec="k", mew=0.6, label="target bottle (spawn)")
    if ee_xyz is not None:
        ax.plot(float((np.asarray(ee_xyz[:2]) - p0) @ u), float(ee_xyz[2]), "^", color="darkorange",
                ms=12, mec="k", mew=0.6, label=f"gripper now ({args.arm})")
    from matplotlib.patches import Patch
    proxies = [Patch(color=LABEL_COLORS[c], label=LABEL_NAMES[c]) for c in (FREE, OBSTACLE, BEYOND)]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=proxies + h, loc="upper right", fontsize=8)
    eps_txt = "inf" if (merged and np.isinf(eps_star)) else (f"{eps_star:.3f}m" if merged else "INACCESSIBLE")
    _pair = f"{start_label}->{goal_label}"
    ax.set_xlabel(f"arc distance {_pair} (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"Side elevation (profile through {_pair})  |  seed {args.seed}, arm {args.arm}"
                 f"   eps*={eps_txt}")
    stem = f"metric_side_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def _viz_clearance_profile(out_dir, args, route_w, route_clear, eps_star, merged):
    """CLEARANCE-ALONG-ROUTE: line chart of occluder clearance vs arc length from grasp to pad; the
    minimum is eps*, the gripper half-width is a dashed feasibility line, and route height rides a
    secondary axis. A stats-friendly view that pinpoints the tightest squeeze."""
    if not route_w or len(route_w) < 2:
        return
    P = np.asarray(route_w, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    clr = np.asarray(route_clear, float)
    finite = np.isfinite(clr)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s, np.where(finite, clr, np.nan), "-", color="#3949ab", lw=2, label="occluder clearance")
    ax.axhline(args.gripper_r, color="#d84315", ls="--", lw=1.5, label=f"gripper r={args.gripper_r:.3f}m")
    if merged and np.isfinite(eps_star) and finite.any():
        j = int(np.nanargmin(np.where(finite, clr, np.inf)))
        ax.plot(s[j], clr[j], "kX", ms=13, label=f"eps*={eps_star:.3f}m")
    elif merged and np.isinf(eps_star):
        ax.text(0.5, 0.9, "eps* = inf (route clears over the bottle top)", transform=ax.transAxes,
                ha="center", color="#2e7d32", fontsize=11)
    ax.set_xlabel("arc length along route, grasp->pad (m)")
    ax.set_ylabel("clearance to occluder (m)", color="#3949ab")
    ax2 = ax.twinx(); ax2.plot(s, P[:, 2], "-", color="gray", lw=1.2, alpha=0.7)
    ax2.set_ylabel("route height z (m)", color="gray")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Clearance & height along the route  |  seed {args.seed}, arm {args.arm}")
    stem = f"metric_profile_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def _viz_topdown(out_dir, args, XX, YY, label, foots, g_xyz, p_xyz, route_w, tgt_p=None, ee_xyz=None,
                 start_label="grasp", goal_label="pad"):
    """TOP-DOWN plan: reachable-at-some-height footprint (green) + occluder outline + the route's (x,y)
    coloured by height z. Shows the sideways detour and how high it climbs, in one plan view."""
    any_free = (label == FREE).any(axis=0)
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(np.where(any_free, 1.0, np.nan), origin="lower", extent=extent, cmap="Greens",
              vmin=0, vmax=1.5, aspect="equal")
    if foots:
        from matplotlib.patches import Polygon as MplPolygon
        for f in foots:
            if f["poly"] is not None:
                ax.add_patch(MplPolygon(f["poly"], closed=True, fill=False, edgecolor="red", lw=2))
    if route_w and len(route_w) > 1:
        P = np.asarray(route_w, float)
        sc = ax.scatter(P[:, 0], P[:, 1], c=P[:, 2], cmap="plasma", s=16)
        fig.colorbar(sc, ax=ax, label="route height z (m)")
    ax.plot(g_xyz[0], g_xyz[1], "o", color="cyan", ms=12, mec="k", label=start_label)
    ax.plot(p_xyz[0], p_xyz[1], "s", color="magenta", ms=10, mec="k", label=goal_label)
    _scene_anchor_markers(ax, tgt_p, ee_xyz, args.arm)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Top-down: route coloured by height  |  seed {args.seed}, arm {args.arm}\n"
                 f"green = reachable at some height")
    stem = f"metric_topdown_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def _viz_ceiling(out_dir, args, XX, YY, zs, label, foots, tgt_p=None, ee_xyz=None):
    """REACHABILITY CEILING: highest FREE z per (x,y) -- the 'lid' of the reachable envelope. Cells
    brighter than the bottle-top height can be cleared over; the red footprint shows where the bottle
    sits. Answers 'where can the arm actually get above the occluder?' at a glance."""
    free = label == FREE
    ceil = np.full(free.shape[1:], np.nan)
    for iz in range(len(zs)):                                 # ascending -> keeps the highest FREE z
        ceil[free[iz]] = zs[iz]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(ceil, origin="lower", extent=[XX.min(), XX.max(), YY.min(), YY.max()],
                   cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, label="highest reachable z (m)")
    zhi = None
    if foots:
        from matplotlib.patches import Polygon as MplPolygon
        for f in foots:
            if f["poly"] is not None:
                ax.add_patch(MplPolygon(f["poly"], closed=True, fill=False, edgecolor="red", lw=2))
        zhis = [f["zhi"] for f in foots if f["poly"] is not None]
        zhi = max(zhis) if zhis else None
    _scene_anchor_markers(ax, tgt_p, ee_xyz, args.arm)
    if tgt_p is not None or ee_xyz is not None:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Reachability ceiling (max FREE z per x,y)  |  seed {args.seed}, arm {args.arm}"
                 + (f"\nbottle top z={zhi:.2f}m -- brighter cells can clear over it" if zhi else ""))
    stem = f"metric_ceiling_seed{args.seed}_{args.arm}"
    fig.tight_layout(); fig.savefig(out_dir / f"{stem}.png", dpi=120); plt.close(fig)
    print(f"[viz] wrote {stem}.png")


def phase4_visuals(out_dir, args, XX, YY, zs, label, edt, foots, g_xyz, p_xyz, route, route_w,
                   eps_star, merged, tgt_p=None, ee_xyz=None):
    """The four 3D-legible views: side elevation, clearance-along-route, top-down-by-height, ceiling.
    All the spatial ones also carry the scene anchors (target bottle spawn + current gripper pose);
    the clearance-vs-arc-length chart has no xy plane, so it gets neither."""
    route_clear = [float(edt[iz, iy, ix]) for (iz, iy, ix) in route] if route else []
    _viz_side_elevation(out_dir, args, XX, YY, zs, label, foots, g_xyz, p_xyz, route_w, eps_star,
                        merged, tgt_p=tgt_p, ee_xyz=ee_xyz)
    _viz_clearance_profile(out_dir, args, route_w, route_clear, eps_star, merged)
    _viz_topdown(out_dir, args, XX, YY, label, foots, g_xyz, p_xyz, route_w, tgt_p=tgt_p, ee_xyz=ee_xyz)
    _viz_ceiling(out_dir, args, XX, YY, zs, label, foots, tgt_p=tgt_p, ee_xyz=ee_xyz)
