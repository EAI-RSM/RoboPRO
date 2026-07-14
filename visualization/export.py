#!/usr/bin/env python3
"""Export derived data from a collected episode — on demand, after collection.

Roles (target / bin / obstacle) come from the offline MASKING RESOLVER
(masking_resolve.py): the manipulated object per stage is the target, where it
settles is the bin (object / fixture link / region), everything else visible is
an obstacle. Roles are PER-STAGE, so a chain flips them correctly. Most outputs
are COMPUTED from what the HDF5 already stores (rgb + depth + camera matrices +
masks + actor_bbox/link_bbox); nothing extra is needed at export time.

Always writes:
  masking.json        the stage timeline: per stage target id + bin (object id /
                      link id / region params) + per-frame stage index (training label)
  masking_panel.png   stage-aware review panel: one row per stage (grasp+settle),
                      target=red, bin=green (object mask or projected region)

Optional --what outputs (roles are the per-frame masking roles):
  pcd                 dense labeled point cloud pcd_<cam>_f<k>.ply (role-colored)
  bbox2d              per-frame 2D boxes from the masks -> bbox2d.json
  bbox3d              per-frame VISIBLE-SURFACE 3D boxes from masked depth -> bbox3d.json
  bbox3d_exact        per-frame EXACT full-extent boxes from physx (actor_bbox) -> json
  panel               overview grid: 6 rows, cols = RGB | depth | seg | roles | 2D boxes

Usage (run_dir = the (task, config) output dir):
    python export.py <run_dir> <episode>                          # default set below
    python export.py <run_dir> 0 --what pcd,panel --cam head_camera --frames 0,mid,last
    python export.py <run_dir> 0 --what all --cam all --frames every:10
"""
import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_episode import (ROLE_COLORS, ROBOT_COLOR, SCENE_NAMES,
                         decode_frames, load_seg, load_sidecar, unproject,
                         write_ply, colorize_ids, role_overlay, depth_vis,
                         boxes2d_image, box_edge_points)
from masking_resolve import (resolve_episode, compute_visible_ids, frame_role_sets,
                             table_ids_of, region_bin_mask, backproject_full,
                             table_obstacle_rects, TABLE_OBJ_PAD, build_obbs,
                             _obb_to_aabb, render_masking_panel, BIN_COL)


def _footprints(actor_bbox):
    """{id, aabb_min, aabb_max} for table padding, from either bbox schema (OBB or legacy)."""
    if actor_bbox is None:
        return None
    if "aabb_min" in actor_bbox:
        return {k: actor_bbox[k] for k in ("id", "aabb_min", "aabb_max")}
    amin, amax = _obb_to_aabb(actor_bbox["obb_center"], actor_bbox["obb_half"], actor_bbox["obb_quat"])
    return {"id": actor_bbox["id"], "aabb_min": amin, "aabb_max": amax}

WHAT_ALL = ("pcd", "bbox2d", "bbox3d", "bbox3d_exact", "panel")
WHAT_DEFAULT = "pcd,bbox2d,bbox3d,bbox3d_exact,panel"


def parse_frames(spec, T):
    """'all' | 'every:N' | comma list of ints and/or keywords first/mid/last."""
    if spec == "all":
        return list(range(T))
    if spec.startswith("every:"):
        return list(range(0, T, max(1, int(spec.split(":", 1)[1]))))
    keyword = {"first": 0, "mid": T // 2, "last": T - 1}
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        k = keyword[tok] if tok in keyword else int(tok)
        if not 0 <= k < T:
            sys.exit(f"!! frame {k} out of range (T={T})")
        out.append(k)
    return sorted(set(out))


def panel_frames(T, n=6):
    if T <= n:
        return list(range(T))
    return sorted(set(np.linspace(0, T - 1, n).round().astype(int).tolist()))


def role_of_id(i, role_ids, robot_ids):
    for role, ids in role_ids.items():
        if i in ids:
            return role
    if i in robot_ids:
        return "robot"
    return "other"


def boxes2d(seg, id_map, role_ids, robot_ids, min_px=15):
    out = []
    for i in np.unique(seg):
        i = int(i)
        if i == 0 or i not in id_map or id_map[i].split("/")[-1] in SCENE_NAMES:
            continue
        m = seg == i
        if m.sum() < min_px:
            continue
        ys, xs = np.nonzero(m)
        out.append({"id": i, "name": id_map[i],
                    "role": role_of_id(i, role_ids, robot_ids),
                    "box_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "pixels": int(m.sum())})
    return out


def boxes3d_visible(pts, seg_vals, id_map, role_ids, robot_ids, min_pts=30):
    out = []
    for i in np.unique(seg_vals):
        i = int(i)
        if i == 0 or i not in id_map or id_map[i].split("/")[-1] in SCENE_NAMES:
            continue
        sel = seg_vals == i
        if sel.sum() < min_pts:
            continue
        p = pts[sel]
        mn, mx = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
        out.append({"id": i, "name": id_map[i],
                    "role": role_of_id(i, role_ids, robot_ids),
                    "min": mn.round(4).tolist(), "max": mx.round(4).tolist(),
                    "center": ((mn + mx) / 2).round(4).tolist(),
                    "size": (mx - mn).round(4).tolist(),
                    "points": int(sel.sum())})
    return out


# 12 edges of the 8-corner box, corner order = (sx,sy,sz) with sx outer, sz inner
_OBB_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
              (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def obb_edge_points(corners, per_edge=24):
    """Wireframe points along the 12 edges of an 8-corner oriented box."""
    corners = np.asarray(corners, float)
    t = np.linspace(0, 1, per_edge)[:, None]
    return np.concatenate([corners[a] * (1 - t) + corners[b] * t for a, b in _OBB_EDGES], 0)


def boxes3d_exact(actor_bbox, k, id_map, role_ids, robot_ids):
    """Per-object exact 3D boxes at frame k. PRIMARY is the oriented box (obb); a derived
    axis-aligned min/max is included for convenience. Schema-agnostic via build_obbs."""
    out = []
    for e in build_obbs(actor_bbox, k, id_map, drop_scene=True):
        rec = {"id": e["id"], "name": e["name"],
               "role": role_of_id(e["id"], role_ids, robot_ids),
               "obb": e["obb"]}
        if e["obb"]:
            c = np.asarray(e["obb"]["corners"], float)
            rec["min"], rec["max"] = c.min(0).round(4).tolist(), c.max(0).round(4).tolist()
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("episode", type=int)
    ap.add_argument("--what", default=WHAT_DEFAULT,
                    help=f"comma list of {','.join(WHAT_ALL)} or 'all' (default: {WHAT_DEFAULT})")
    ap.add_argument("--cam", default="countertop_camera",
                    help="camera name, comma-separated list, or 'all'")
    ap.add_argument("--frames", default="first,mid,last",
                    help="'all', 'every:N', or comma list of indices / first / mid / last")
    ap.add_argument("--stride", type=int, default=1,
                    help="pixel stride for point clouds (1 = every depth pixel)")
    ap.add_argument("--out", default=None,
                    help="output dir (default <run_dir>/export/episode<idx>)")
    args = ap.parse_args()

    what = list(WHAT_ALL) if args.what == "all" else [w.strip() for w in args.what.split(",")]
    bad = [w for w in what if w not in WHAT_ALL]
    if bad:
        sys.exit(f"!! unknown --what {bad}; choose from {WHAT_ALL}")

    ep = args.episode
    h5_path = os.path.join(args.run_dir, "data", f"episode{ep}.hdf5")
    out_dir = args.out or os.path.join(args.run_dir, "export", f"episode{ep}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- masking: resolve roles (target/bin per stage) + save the label ----------
    ep_info, id_map, roles, robot_ids = load_sidecar(args.run_dir, ep)
    masking = resolve_episode(args.run_dir, ep, verbose=False)
    with open(os.path.join(out_dir, "masking.json"), "w") as f:
        json.dump({f"episode_{ep}": masking}, f, indent=2)
    visible = compute_visible_ids(h5_path)
    table_ids = table_ids_of(id_map)

    def rk(k):
        return frame_role_sets(masking, k, id_map, visible, robot_ids)

    # ---- episode data ------------------------------------------------------------
    with h5py.File(h5_path, "r") as f:
        available = list(f["observation"].keys())
        cams = available if args.cam == "all" else [c.strip() for c in args.cam.split(",")]
        missing = [c for c in cams if c not in available]
        if missing:
            sys.exit(f"!! camera(s) {missing} not in this file. Available: {available}")
        data = {}
        for cam in cams:
            g = f["observation"][cam]
            d = {"seg": load_seg(g["actor_segmentation"])}
            if "rgb" in g:
                d["rgb"] = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            for im in decode_frames(g["rgb"], cv2.IMREAD_COLOR)]
            if "depth" in g:
                d["depth"] = decode_frames(g["depth"], cv2.IMREAD_UNCHANGED)
                d["K"], d["E"] = np.asarray(g["intrinsic_cv"]), np.asarray(g["extrinsic_cv"])
            data[cam] = d
        actor_bbox = None
        if "actor_bbox" in f:                            # read whatever schema is present
            gb = f["actor_bbox"]                         # (OBB: obb_center/half/quat, or legacy)
            actor_bbox = {kk: np.asarray(gb[kk]) for kk in gb.keys()}
    footprints = _footprints(actor_bbox)                 # {id,aabb_min,aabb_max} for table padding

    T = len(data[cams[0]]["seg"])
    frames = parse_frames(args.frames, T)

    print(f"export · {args.run_dir} ep{ep} · {masking['task_type']} "
          f"{len(masking['stages'])} stage(s) · what={what} cams={cams} frames={frames} (T={T})")

    # ---- masking review panel (stage-aware; always) ------------------------------
    render_masking_panel(args.run_dir, ep, cam=cams[0], masking=masking,
                         out_path=os.path.join(out_dir, "masking_panel.png"))

    need_depth = any(w in what for w in ("pcd", "bbox3d"))
    for cam in cams:
        if need_depth and "depth" not in data[cam]:
            sys.exit(f"!! {cam} has no depth in this file — pcd/bbox3d need depth")

    # ---- exact (physics) boxes ---------------------------------------------------
    exact = None
    if "bbox3d_exact" in what:
        if actor_bbox is None:
            print("!! bbox3d_exact requested but no actor_bbox group — skipping")
        else:
            exact = {f"f{k}": boxes3d_exact(actor_bbox, k, id_map, rk(k), robot_ids)
                     for k in frames}
            with open(os.path.join(out_dir, "bbox3d_exact.json"), "w") as f:
                json.dump(exact, f, indent=2)

    # ---- per-camera pcd / boxes --------------------------------------------------
    results2d, results3d = {}, {}
    for cam in cams:
        d = data[cam]
        boxes2d_cam, boxes3d_cam = {}, {}
        for k in frames:
            seg = d["seg"][k]
            rkk = rk(k)
            if "bbox2d" in what:
                boxes2d_cam[f"f{k}"] = boxes2d(seg, id_map, rkk, robot_ids)
            if "pcd" in what or "bbox3d" in what:
                K = d["K"][k] if d["K"].ndim == 3 else d["K"]
                E = d["E"][k] if d["E"].ndim == 3 else d["E"]
                pts, (vv, uu) = unproject(d["depth"][k], K, E, stride=args.stride)
                seg_s = seg[vv, uu]
                if "bbox3d" in what:
                    boxes3d_cam[f"f{k}"] = boxes3d_visible(pts, seg_s, id_map, rkk, robot_ids)
                if "pcd" in what:
                    rgb_s = (d["rgb"][k][vv, uu].astype(np.uint8) if "rgb" in d
                             else np.full((len(pts), 3), 180, np.uint8))
                    cols = rgb_s.copy()
                    if robot_ids:
                        cols[np.isin(seg_s, list(robot_ids))] = ROBOT_COLOR
                    for role, ids in rkk.items():
                        if ids and role in ROLE_COLORS:
                            cols[np.isin(seg_s, list(ids))] = ROLE_COLORS[role]
                    if exact is not None:
                        for b in exact[f"f{k}"]:
                            if b["role"] in ROLE_COLORS and b.get("obb"):
                                edge = obb_edge_points(b["obb"]["corners"])
                                pts = np.concatenate([pts, edge])
                                cols = np.concatenate(
                                    [cols, np.tile(ROLE_COLORS[b["role"]], (len(edge), 1))])
                    write_ply(os.path.join(out_dir, f"pcd_{cam}_f{k}.ply"), pts, cols)
        if boxes2d_cam:
            results2d[cam] = boxes2d_cam
        if boxes3d_cam:
            results3d[cam] = boxes3d_cam

    # ---- overview panel: 6 rows, cols=RGB|depth|seg|roles|2D boxes ----------------
    if "panel" in what:
        pf = panel_frames(T)
        for cam in cams:
            d = data[cam]
            if "rgb" not in d:
                print(f"panel: {cam} has no rgb — skipping")
                continue
            rows = []
            for k in pf:
                seg, rkk = d["seg"][k], rk(k)
                header = d["rgb"][k].copy()
                cv2.putText(header, f"{cam} f{k}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                over = role_overlay(d["rgb"][k], seg, rkk, robot_ids, set())
                # paint region bins (no seg id) green using this frame's stage
                st = masking["stages"][masking["frame_stage"][k]] if (
                    0 <= masking["frame_stage"][k] < len(masking["stages"])) else None
                if st and st["bin"]["bin_type"].startswith("region") and "depth" in d:
                    K = d["K"][k] if d["K"].ndim == 3 else d["K"]
                    E = d["E"][k] if d["E"].ndim == 3 else d["E"]
                    world, valid = backproject_full(d["depth"][k], K, E)
                    rects = (table_obstacle_rects(footprints, k, id_map, robot_ids,
                                                  int(st["target_id"]),
                                                  st["bin"].get("obj_pad", TABLE_OBJ_PAD),
                                                  st["bin"].get("surface_z"))
                             if st["bin"]["bin_type"] == "region_table" else None)
                    bm = region_bin_mask(st["bin"], seg, table_ids, world, valid, rects)
                    over[bm] = (0.4 * over[bm] + 0.6 * BIN_COL).astype(np.uint8)
                cells = [header]
                if "depth" in d:
                    cells.append(depth_vis(d["depth"][k]))
                cells += [colorize_ids(seg), over,
                          boxes2d_image(d["rgb"][k], seg, rkk, id_map)]
                rows.append(np.hstack(cells))
            name = "panel.png" if len(cams) == 1 else f"panel_{cam}.png"
            cv2.imwrite(os.path.join(out_dir, name),
                        cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))

    if "bbox2d" in what:
        with open(os.path.join(out_dir, "bbox2d.json"), "w") as f:
            json.dump(results2d, f, indent=2)
    if "bbox3d" in what:
        with open(os.path.join(out_dir, "bbox3d.json"), "w") as f:
            json.dump(results3d, f, indent=2)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "episode": ep, "cameras": cams, "frames": [int(k) for k in frames],
            "what": what, "pcd_stride": args.stride,
            "actor_bbox_available": actor_bbox is not None,
            "task": masking["task"], "task_type": masking["task_type"],
            "roles_source": "masking_resolve (per-stage target/bin)",
            "stages": [{"stage": s["stage"], "target_id": s["target_id"],
                        "target_name": s["target_name"], "bin": s["bin"]}
                       for s in masking["stages"]],
            "actor_id_map": {str(k): v for k, v in sorted(id_map.items())},
            "units": {"xyz": "meters, world frame", "bbox2d": "pixels xyxy",
                      "bbox3d": "meters, world frame, VISIBLE-surface AABB",
                      "bbox3d_exact": "meters, world frame; 'obb' = PRIMARY oriented box "
                                      "(center/half_size/quat/corners), 'min/max' = legacy "
                                      "axis-aligned AABB; obb null on pre-OBB data"},
        }, f, indent=2)

    print(f"wrote {out_dir}/  (masking.json + masking_panel.png + {what})")


if __name__ == "__main__":
    main()
