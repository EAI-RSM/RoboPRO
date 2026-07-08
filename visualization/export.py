#!/usr/bin/env python3
"""Export derived data from a collected episode — on demand, after collection.

Most outputs are COMPUTED from what the HDF5 already stores (rgb + depth +
camera matrices + masks + sidecar); nothing needs to be enabled at collection
time. The one exception is bbox3d_exact, which reads the physics-engine boxes
that must have been recorded during collection (data_type.actor_bbox: true).

  --what pcd          dense labeled point clouds:
                        pcd_<cam>_f<k>.ply   (role-colored, open in MeshLab)
                        pcd_<cam>_f<k>.npz   (xyz float32, rgb uint8, seg_id uint16)
  --what bbox2d       per-frame 2D boxes from the masks -> bbox2d.json
  --what bbox3d       per-frame VISIBLE-SURFACE 3D boxes from masked depth (what the
                      cameras can see) -> bbox3d.json
  --what bbox3d_exact per-frame EXACT full-extent 3D boxes from the physics engine
                      (includes occluded parts; needs data_type.actor_bbox at
                      collection) -> bbox3d_exact.json  [+ wireframes into the .ply]
  --what masks        masks_<cam>_f<k>.png (raw uint16 ids) + *_color.png preview
  --what overlay      overlay_<cam>_f<k>.png (role-colored rgb)
  --what panel        panel[_<cam>].png — quick-look grid, rows=frames,
                      cols = RGB | depth | seg | role overlay | 2D boxes

Always writes meta.json (id->name map, role->ids, settings) next to the outputs.

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
from viz_episode import (ROLE_COLORS, ROBOT_COLOR, OTHER_COLOR, SCENE_NAMES,
                         decode_frames, load_seg, load_sidecar, compute_role_ids,
                         merge_env_ids, unproject, write_ply, colorize_ids,
                         role_overlay, depth_vis, boxes2d_image, box_edge_points)

WHAT_ALL = ("pcd", "bbox2d", "bbox3d", "bbox3d_exact", "masks", "overlay", "panel")
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


def role_of_id(i, role_ids, robot_ids):
    for role, ids in role_ids.items():
        if i in ids:
            return role
    if i in robot_ids:
        return "robot"
    return "other"


def boxes2d(seg, id_map, role_ids, robot_ids, min_px=15):
    """All named objects visible in this mask -> [{id,name,role,box}] (pixel xyxy)."""
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
    """Axis-aligned world-frame box of each object's VISIBLE 3D points (meters)."""
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


def boxes3d_exact(actor_bbox, k, id_map, role_ids, robot_ids):
    """EXACT full-extent world-AABBs recorded by physx for frame k (meters).
    Includes fully/partly occluded objects; robot links are not recorded."""
    ids = actor_bbox["id"][k]
    mns, mxs, poses = actor_bbox["aabb_min"][k], actor_bbox["aabb_max"][k], actor_bbox["pose"][k]
    out = []
    for j in range(len(ids)):
        i = int(ids[j])
        if i not in id_map or id_map[i].split("/")[-1] in SCENE_NAMES:
            continue
        mn, mx = np.asarray(mns[j], float), np.asarray(mxs[j], float)
        out.append({"id": i, "name": id_map.get(i, ""),
                    "role": role_of_id(i, role_ids, robot_ids),
                    "min": mn.round(4).tolist(), "max": mx.round(4).tolist(),
                    "center": ((mn + mx) / 2).round(4).tolist(),
                    "size": (mx - mn).round(4).tolist(),
                    "pose": np.asarray(poses[j], float).round(4).tolist()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("episode", type=int)
    ap.add_argument("--what", default=WHAT_DEFAULT,
                    help=f"comma list of {','.join(WHAT_ALL)} or 'all' (default: {WHAT_DEFAULT})")
    ap.add_argument("--cam", default="head_camera",
                    help="camera name, comma-separated list, or 'all'")
    ap.add_argument("--frames", default="first,mid,last",
                    help="'all', 'every:N', or comma list of indices / first / mid / last")
    ap.add_argument("--stride", type=int, default=1,
                    help="pixel stride for point clouds (1 = every depth pixel)")
    ap.add_argument("--split-env", action="store_true",
                    help="keep environment objects a separate class instead of "
                         "merging them into obstacle")
    ap.add_argument("--out", default=None, help="output dir "
                    "(default <run_dir>/export/episode<idx>)")
    args = ap.parse_args()

    what = list(WHAT_ALL) if args.what == "all" else [w.strip() for w in args.what.split(",")]
    bad = [w for w in what if w not in WHAT_ALL]
    if bad:
        sys.exit(f"!! unknown --what {bad}; choose from {WHAT_ALL}")

    ep = args.episode
    h5_path = os.path.join(args.run_dir, "data", f"episode{ep}.hdf5")
    out_dir = args.out or os.path.join(args.run_dir, "export", f"episode{ep}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- sidecar: names + roles ----------------------------------------------
    ep_info, id_map, roles, robot_ids = load_sidecar(args.run_dir, ep)
    role_ids, clutter_names = compute_role_ids(ep_info, id_map, roles, robot_ids)

    # ---- episode data ---------------------------------------------------------
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
        # physics ground-truth boxes (present only if collected with actor_bbox)
        actor_bbox = None
        if "actor_bbox" in f:
            gb = f["actor_bbox"]
            actor_bbox = {kk: np.asarray(gb[kk]) for kk in ("id", "pose", "aabb_min", "aabb_max")}

    T = len(data[cams[0]]["seg"])
    frames = parse_frames(args.frames, T)

    # env objects folded into obstacle (default) using ids seen in chosen frames
    seen_ids = set()
    for cam in cams:
        for k in frames:
            seen_ids |= set(np.unique(data[cam]["seg"][k]).tolist())
    seen_ids.discard(0)
    env_ids, other_ids = merge_env_ids(role_ids, id_map, robot_ids, seen_ids,
                                       split_env=args.split_env)

    print(f"export · {args.run_dir} ep{ep} · what={what} cams={cams} "
          f"frames={frames} (T={T})")

    need_depth = any(w in what for w in ("pcd", "bbox3d"))
    for cam in cams:
        if need_depth and "depth" not in data[cam]:
            sys.exit(f"!! {cam} has no depth in this file — pcd/bbox3d need depth")

    # ---- exact (physics) boxes: world-frame, camera-independent --------------
    exact = None
    if "bbox3d_exact" in what:
        if actor_bbox is None:
            print("!! bbox3d_exact requested but this HDF5 has no actor_bbox group "
                  "(collect with data_type.actor_bbox: true) — skipping exact boxes")
        else:
            exact = {f"f{k}": boxes3d_exact(actor_bbox, k, id_map, role_ids, robot_ids)
                     for k in frames}
            with open(os.path.join(out_dir, "bbox3d_exact.json"), "w") as f:
                json.dump(exact, f, indent=2)

    # ---- per-camera outputs ---------------------------------------------------
    results2d, results3d = {}, {}
    for cam in cams:
        d = data[cam]
        boxes2d_cam, boxes3d_cam = {}, {}
        for k in frames:
            seg = d["seg"][k]

            if "bbox2d" in what:
                boxes2d_cam[f"f{k}"] = boxes2d(seg, id_map, role_ids, robot_ids)

            if "masks" in what:
                cv2.imwrite(os.path.join(out_dir, f"masks_{cam}_f{k}.png"),
                            seg.astype(np.uint16))
                cv2.imwrite(os.path.join(out_dir, f"masks_{cam}_f{k}_color.png"),
                            cv2.cvtColor(colorize_ids(seg), cv2.COLOR_RGB2BGR))

            if "overlay" in what and "rgb" in d:
                ov = role_overlay(d["rgb"][k], seg, role_ids, robot_ids, other_ids)
                cv2.imwrite(os.path.join(out_dir, f"overlay_{cam}_f{k}.png"),
                            cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

            if "pcd" in what or "bbox3d" in what:
                K = d["K"][k] if d["K"].ndim == 3 else d["K"]
                E = d["E"][k] if d["E"].ndim == 3 else d["E"]
                pts, (vv, uu) = unproject(d["depth"][k], K, E, stride=args.stride)
                seg_s = seg[vv, uu]

                if "bbox3d" in what:
                    boxes3d_cam[f"f{k}"] = boxes3d_visible(pts, seg_s, id_map,
                                                           role_ids, robot_ids)

                if "pcd" in what:
                    rgb_s = (d["rgb"][k][vv, uu].astype(np.uint8) if "rgb" in d
                             else np.full((len(pts), 3), 180, np.uint8))
                    np.savez_compressed(
                        os.path.join(out_dir, f"pcd_{cam}_f{k}.npz"),
                        xyz=pts.astype(np.float32), rgb=rgb_s,
                        seg_id=seg_s.astype(np.uint16))
                    cols = rgb_s.copy()
                    if robot_ids:
                        cols[np.isin(seg_s, list(robot_ids))] = ROBOT_COLOR
                    if other_ids:
                        cols[np.isin(seg_s, list(other_ids))] = OTHER_COLOR
                    for role, ids in role_ids.items():
                        if ids:
                            cols[np.isin(seg_s, list(ids))] = ROLE_COLORS[role]
                    # overlay EXACT box wireframes so occluded full-extent is visible
                    if exact is not None:
                        for b in exact[f"f{k}"]:
                            if b["role"] in ROLE_COLORS:
                                edge = box_edge_points(b["min"], b["max"])
                                pts = np.concatenate([pts, edge])
                                cols = np.concatenate(
                                    [cols, np.tile(ROLE_COLORS[b["role"]], (len(edge), 1))])
                    write_ply(os.path.join(out_dir, f"pcd_{cam}_f{k}.ply"), pts, cols)
        if boxes2d_cam:
            results2d[cam] = boxes2d_cam
        if boxes3d_cam:
            results3d[cam] = boxes3d_cam

    # ---- panel (quick-look grid): rows=frames, cols=RGB|depth|seg|overlay|2D --
    if "panel" in what:
        for cam in cams:
            d = data[cam]
            if "rgb" not in d:
                print(f"panel: {cam} has no rgb — skipping")
                continue
            rows = []
            for k in frames:
                seg = d["seg"][k]
                header = d["rgb"][k].copy()
                cv2.putText(header, f"{cam} f{k}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cells = [header]
                if "depth" in d:
                    cells.append(depth_vis(d["depth"][k]))
                cells += [colorize_ids(seg),
                          role_overlay(d["rgb"][k], seg, role_ids, robot_ids, other_ids),
                          boxes2d_image(d["rgb"][k], seg, role_ids, id_map, other_ids)]
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
            "roles_from_task_code": roles,
            "clutter_object_types": clutter_names,
            "role_ids": {r: sorted(int(i) for i in ids) for r, ids in role_ids.items()},
            "robot_ids": sorted(int(i) for i in robot_ids),
            "environment_ids": sorted(int(i) for i in env_ids),
            "environment_merged_into_obstacle": not args.split_env,
            "actor_id_map": {str(k): v for k, v in sorted(id_map.items())},
            "units": {"xyz": "meters, world frame", "bbox2d": "pixels xyxy",
                      "bbox3d": "meters, world frame, VISIBLE-surface AABB",
                      "bbox3d_exact": "meters, world frame, EXACT full-extent AABB (physx)"},
        }, f, indent=2)

    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
