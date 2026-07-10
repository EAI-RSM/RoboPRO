#!/usr/bin/env python3
"""Visualize one collected episode: masks, roles, depth, point clouds, boxes.

Reads the episode HDF5 + scene_info.json (actor_id_map / role_names sidecar)
and writes, under <run_dir>/viz/episode<idx>/:

  panel.png            frames as rows: RGB | depth | actor-seg | role overlay | 2D boxes
  legend.json          role -> ids, id -> name map, coverage report, 3D boxes
  pcd_stored_f<k>.ply  the stored point cloud at frame k (if recorded)
  pcd_depth_f<k>.ply   dense cloud rebuilt from depth, role-colored + 3D box wireframes
  topdown_f<k>.png     top-down (x,y) view of the rebuilt cloud + 3D box outlines

Colors: target=red, destination=green, obstacles=orange, robot=blue, other=gray.

Usage (run_dir = the (task, config) output dir):
    python viz_episode.py data/dataset/put_mouse_on_pad/datagen_template 0
    python viz_episode.py <run_dir> <episode_idx> [--cam head_camera] [--frames 6]
"""
import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np

ROLE_COLORS = {  # RGB
    "target": np.array([230, 40, 40], np.uint8),
    "destination": np.array([40, 200, 60], np.uint8),
    "obstacle": np.array([255, 160, 20], np.uint8),
}
ROBOT_COLOR = np.array([80, 130, 230], np.uint8)
OTHER_COLOR = np.array([130, 220, 220], np.uint8)  # known object with NO role assigned
SCENE_NAMES = {"ground", "wall", "table", "floor"}


# ---------------------------------------------------------------- loading --

def decode_frames(ds, flags):
    return [cv2.imdecode(np.frombuffer(bytes(b), np.uint8), flags) for b in ds]


def load_seg(ds):
    """Segmentation dataset -> (T,H,W) uint16. Handles PNG-encoded and raw."""
    if ds.dtype.kind == "S":
        return np.stack(decode_frames(ds, cv2.IMREAD_UNCHANGED)).astype(np.uint16)
    arr = np.asarray(ds)
    if arr.ndim == 4:
        sys.exit("!! segmentation is COLORIZED (T,H,W,3) legacy data — "
                 "re-collect with the patched camera.py (raw uint16 ids).")
    return arr.astype(np.uint16)


# ------------------------------------------------------------ id matching --

def clutter_object_types(raw):
    """cluttered_table_info -> object_type names (ignores index digits)."""
    names = []
    if isinstance(raw, list):
        for e in raw:
            if isinstance(e, dict) and "object_type" in e:
                names.append(str(e["object_type"]))
            elif isinstance(e, str) and not e.isdigit():
                names.append(e)
    return list(dict.fromkeys(names))


def names_to_ids(names, id_map, substring=False):
    """Exact name match (all instances); optional substring fallback for
    non-digit names of length >= 3."""
    ids = set()
    for want in names:
        if not want or not isinstance(want, str) or want.isdigit():
            continue
        exact = [i for i, n in id_map.items() if n == want]
        if exact:
            ids.update(exact)
        elif substring and len(want) >= 3:
            ids.update(i for i, n in id_map.items() if want in n)
    return ids


def resolve_role(roles, key, id_map):
    """Prefer the exact per_scene_id the task recorded (disambiguates
    same-name objects); fall back to name-matching for older data."""
    rid = roles.get(f"{key}_id")
    if rid is not None and int(rid) in id_map:
        return {int(rid)}, "id"
    name = roles.get(key)
    return (names_to_ids([name], id_map, substring=True) if name else set()), "name"


def load_sidecar(run_dir, ep):
    """scene_info.json for one episode -> (ep_info, id_map, roles, robot_ids)."""
    with open(os.path.join(run_dir, "scene_info.json")) as f:
        ep_info = json.load(f).get(f"episode_{ep}", {})
    id_map = {}
    for k, v in ep_info.get("actor_id_map", {}).items():
        id_map[int(k)] = ("robot" + v) if v.startswith("/") else v  # old data: robot art unnamed
    roles = ep_info.get("role_names", {})
    robot_ids = {i for i, n in id_map.items() if n.startswith("robot/")}
    return ep_info, id_map, roles, robot_ids


def compute_role_ids(ep_info, id_map, roles, robot_ids, verbose=True):
    """target/destination/obstacle(-spawned-clutter) id sets for one episode."""
    clutter_names = clutter_object_types(ep_info.get("cluttered_table_info"))
    tgt_ids, tgt_how = resolve_role(roles, "target", id_map)
    dst_ids, dst_how = resolve_role(roles, "destination", id_map)
    role_ids = {
        "target": tgt_ids,
        "destination": dst_ids,
        "obstacle": names_to_ids(clutter_names, id_map, substring=False),
    }
    if verbose:
        print(f"target matched by {tgt_how}, destination by {dst_how} "
              "(id = exact instance, name = fallback)")
    role_ids["target"] -= robot_ids
    role_ids["destination"] -= robot_ids
    if not role_ids["target"]:
        # tasks without a target_obj attr (e.g. kitchenl) still declare names via
        # _get_target_object_names(); fall back to those
        fallback = list(roles.get("target_object_names", []) or [])
        role_ids["target"] = (names_to_ids(fallback, id_map, substring=True)
                              - role_ids["destination"] - robot_ids)
        if role_ids["target"] and verbose:
            print(f"note: target ids taken from target_object_names fallback: {fallback}")
    if not role_ids["destination"] and verbose:
        print("note: no destination OBJECT resolved — this task's goal may be a pose/region "
              "(e.g. 'next to X'), not an object. That's the semantics question for the manager.")
    role_ids["obstacle"] -= role_ids["target"] | role_ids["destination"] | robot_ids
    return role_ids, clutter_names


def merge_env_ids(role_ids, id_map, robot_ids, seen_ids, split_env=False, verbose=True):
    """Environment objects = known, visible, no category (furniture, appliances,
    task extras). DEFAULT: an obstacle is an obstacle -> merged into the obstacle
    class; split_env keeps them separate. Returns (env_ids, other_ids); mutates
    role_ids['obstacle'] when merging."""
    assigned = role_ids["target"] | role_ids["destination"] | role_ids["obstacle"] | robot_ids
    env_ids = {i for i in seen_ids
               if i in id_map and i not in assigned
               and id_map[i].split("/")[-1] not in SCENE_NAMES
               and not id_map[i].startswith("robot/")}
    if split_env:
        other_ids = env_ids
    else:
        role_ids["obstacle"] |= env_ids
        other_ids = set()
    if env_ids and verbose:
        how = "separate (cyan)" if split_env else "merged into obstacle (orange)"
        print(f"environment objects, {how}: "
              + ", ".join(sorted({id_map[i] for i in env_ids})))
    return env_ids, other_ids


# ------------------------------------------------------------- rendering --

def id_color(i):
    if i == 0:
        return np.array([25, 25, 25], np.uint8)
    rng = np.random.RandomState((int(i) * 9973 + 17) % (2**31))
    return rng.randint(60, 256, 3).astype(np.uint8)


def colorize_ids(seg):
    out = np.zeros((*seg.shape, 3), np.uint8)
    for i in np.unique(seg):
        out[seg == i] = id_color(i)
    return out


def depth_vis(depth_mm):
    d = depth_mm.astype(np.float32)
    valid = d > 0
    out = np.zeros((*d.shape, 3), np.uint8)
    if valid.any():
        lo, hi = np.percentile(d[valid], [1, 99])
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        out = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
        out[~valid] = 0
    return out


def role_overlay(rgb, seg, role_ids, robot_ids, other_ids):
    out = (rgb.astype(np.float32) * 0.5).astype(np.uint8)
    if other_ids:
        m = np.isin(seg, list(other_ids))
        out[m] = (0.6 * out[m] + 0.4 * OTHER_COLOR).astype(np.uint8)
    if robot_ids:
        m = np.isin(seg, list(robot_ids))
        out[m] = (0.55 * out[m] + 0.45 * ROBOT_COLOR).astype(np.uint8)
    for role, ids in role_ids.items():
        if ids:
            m = np.isin(seg, list(ids))
            out[m] = (0.3 * out[m] + 0.7 * ROLE_COLORS[role]).astype(np.uint8)
    return out


def _draw_box2d(img, seg, i, col, id_map, min_px):
    m = seg == i
    if m.sum() < min_px:
        return
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
    label = id_map.get(int(i), str(i)).split("/")[-1][:12]
    cv2.putText(img, label, (x0, max(y0 - 3, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1)


def boxes2d_image(rgb, seg, role_ids, id_map, other_ids=()):
    img = rgb.copy()
    for i in other_ids:  # drawn first so role boxes sit on top
        _draw_box2d(img, seg, i, tuple(int(c) for c in OTHER_COLOR), id_map, min_px=40)
    for role, ids in role_ids.items():
        for i in ids:
            _draw_box2d(img, seg, i, tuple(int(c) for c in ROLE_COLORS[role]), id_map, min_px=15)
    return img


# ------------------------------------------------------------- 3D pieces --

def unproject(depth_mm, K, E, stride=2):
    """depth (H,W) mm; K (3,3); E (3,4) world->cam (CV). World pts + pixel idx."""
    H, W = depth_mm.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_mm[::stride, ::stride].astype(np.float32) / 1000.0
    ok = (z > 0) & (z < 4.0)
    u, v, z = u[ok], v[ok], z[ok]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pcam = np.stack([x, y, z], 1)
    R, t = E[:, :3], E[:, 3]
    return (pcam - t) @ R, (v, u)


def boxes3d(pts, seg_vals, role_ids, id_map):
    """Per role object: axis-aligned box of its (visible) 3D points, in meters."""
    out = []
    for role, ids in role_ids.items():
        for i in ids:
            sel = seg_vals == i
            if sel.sum() < 30:
                continue
            p = pts[sel]
            mn, mx = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
            out.append({"id": int(i), "name": id_map.get(int(i), ""), "role": role,
                        "min": mn.round(4).tolist(), "max": mx.round(4).tolist(),
                        "center": ((mn + mx) / 2).round(4).tolist(),
                        "size": (mx - mn).round(4).tolist()})
    return out


def box_edge_points(mn, mx, per_edge=24):
    mn, mx = np.asarray(mn), np.asarray(mx)
    c = np.array([[x, y, z] for x in (mn[0], mx[0])
                  for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return np.concatenate([np.linspace(c[a], c[b], per_edge) for a, b in edges])


def write_ply(path, pts, cols):
    # vectorized write (no per-row Python loop): faster on 80k+ points and avoids
    # a flaky interpreter crash this env throws on manual zip-iteration.
    pts = np.asarray(pts, np.float32).reshape(-1, 3)
    cols = np.asarray(cols).reshape(-1, 3)
    n = len(pts)
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    body = np.column_stack([pts, cols.astype(np.int64)])
    with open(path, "w") as f:
        f.write(header)
        if n:
            np.savetxt(f, body, fmt="%.4f %.4f %.4f %d %d %d")


def topdown(pts, cols, boxes, path, size=800):
    if len(pts) == 0:
        return
    xy = pts[:, :2]
    lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-3)

    def to_px(p):
        return ((np.asarray(p) - lo) / span * (size - 1)).clip(0, size - 1).astype(int)

    img = np.full((size, size, 3), 15, np.uint8)
    pix = to_px(xy)
    order = np.argsort(cols.astype(int).sum(1))  # colorful (role) points on top
    ys, xs = size - 1 - pix[:, 1], pix[:, 0]
    for dy in (0, 1):
        for dx in (0, 1):
            img[(ys[order] + dy).clip(0, size - 1), (xs[order] + dx).clip(0, size - 1)] = cols[order]
    for b in boxes:
        col = tuple(int(c) for c in ROLE_COLORS[b["role"]])
        p0, p1 = to_px(b["min"][:2]), to_px(b["max"][:2])
        cv2.rectangle(img, (p0[0], size - 1 - p1[1]), (p1[0], size - 1 - p0[1]), col, 1)
        cv2.putText(img, b["name"].split("/")[-1][:14], (p0[0], max(size - 1 - p1[1] - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("episode", type=int)
    ap.add_argument("--cam", default="head_camera",
                    help="camera name, comma-separated list, or 'all'")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--split-env", action="store_true",
                    help="show environment objects as separate cyan instead of merging into obstacle")
    args = ap.parse_args()

    ep = args.episode
    h5_path = os.path.join(args.run_dir, "data", f"episode{ep}.hdf5")
    out_dir = os.path.join(args.run_dir, "viz", f"episode{ep}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "═" * 74)
    print(f"VIZ · {args.run_dir} · episode {ep}")
    print("═" * 74)

    # ---- sidecar: id map + roles -------------------------------------------
    ep_info, id_map, roles, robot_ids = load_sidecar(args.run_dir, ep)
    role_ids, clutter_names = compute_role_ids(ep_info, id_map, roles, robot_ids)

    print(f"actor_id_map: {len(id_map)} entries ({len(robot_ids)} robot links)")
    print(f"role_names from task code: {roles}")
    outcome = ep_info.get("outcome")
    if outcome:
        print(f"OUTCOME: {outcome.get('label')} (code={outcome.get('label_code')}, "
              f"success={outcome.get('success')}, collision={outcome.get('collision')}, "
              f"planner_blind={outcome.get('planner_blind_to_obstacles')})")
    print(f"clutter object types: {clutter_names}")
    print("role ids: " + ", ".join(f"{r}={sorted(ids)}" for r, ids in role_ids.items()))
    if not id_map:
        print("!! actor_id_map is EMPTY — per_scene_id lookup failed; paste this output back.")

    # ---- episode data -------------------------------------------------------
    with h5py.File(h5_path, "r") as f:
        available = list(f["observation"].keys())
        cams = available if args.cam == "all" else [c.strip() for c in args.cam.split(",")]
        missing = [c for c in cams if c not in available]
        if missing:
            sys.exit(f"!! camera(s) {missing} not in this file. Available: {available}")
        data = {}
        for cam in cams:
            g = f["observation"][cam]
            data[cam] = dict(
                rgb=[cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                     for im in decode_frames(g["rgb"], cv2.IMREAD_COLOR)],
                depth=decode_frames(g["depth"], cv2.IMREAD_UNCHANGED),
                seg=load_seg(g["actor_segmentation"]),
                K=np.asarray(g["intrinsic_cv"]),
                E=np.asarray(g["extrinsic_cv"]),
            )
        stored_pcd = np.asarray(f["pointcloud"]) if "pointcloud" in f else None

    print(f"cameras in file: {available}")
    primary = cams[0]
    rgb, depth, seg = data[primary]["rgb"], data[primary]["depth"], data[primary]["seg"]
    K_all, E_all = data[primary]["K"], data[primary]["E"]

    T = len(rgb)
    frames = sorted(set(np.linspace(0, T - 1, args.frames).astype(int)))
    print(f"T={T} frames, visualizing {frames}, panels for {cams} (3D/topdown from {primary})")

    # ---- coverage check -----------------------------------------------------
    seen_ids = set()
    for k in frames:
        seen_ids |= set(np.unique(seg[k]).tolist())
    seen_ids.discard(0)
    unmatched = sorted(seen_ids - set(id_map))
    if unmatched:
        print(f"!! seg ids with NO name in actor_id_map: {unmatched}")

    spawned_obstacle_ids = set(role_ids["obstacle"])
    env_ids, other_ids = merge_env_ids(role_ids, id_map, robot_ids, seen_ids,
                                       split_env=args.split_env)

    # ---- panels (one per camera) ---------------------------------------------
    for cam in cams:
        d = data[cam]
        rows = []
        for k in frames:
            r = d["rgb"][k].copy()
            cv2.putText(r, f"{cam} f{k}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            rows.append(np.hstack([
                r, depth_vis(d["depth"][k]), colorize_ids(d["seg"][k]),
                role_overlay(d["rgb"][k], d["seg"][k], role_ids, robot_ids, other_ids),
                boxes2d_image(d["rgb"][k], d["seg"][k], role_ids, id_map, other_ids),
            ]))
        name = "panel.png" if len(cams) == 1 else f"panel_{cam}.png"
        cv2.imwrite(os.path.join(out_dir, name),
                    cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))

    # ---- point clouds + 3D boxes (first + middle sampled frame) -------------
    all_boxes = {}
    for k in {frames[0], frames[len(frames) // 2]}:
        if (stored_pcd is not None and stored_pcd.ndim == 3
                and stored_pcd.shape[1] > 0 and len(stored_pcd) > k):
            pc = stored_pcd[k]
            write_ply(os.path.join(out_dir, f"pcd_stored_f{k}.ply"),
                      pc[:, :3], (pc[:, 3:6] * 255).clip(0, 255).astype(np.uint8))
        K = K_all[k] if K_all.ndim == 3 else K_all
        E = E_all[k] if E_all.ndim == 3 else E_all
        pts, (vv, uu) = unproject(depth[k], K, E)
        seg_s = seg[k][vv, uu]
        cols = rgb[k][vv, uu].astype(np.uint8)
        if robot_ids:
            cols[np.isin(seg_s, list(robot_ids))] = ROBOT_COLOR
        for role, ids in role_ids.items():
            if ids:
                cols[np.isin(seg_s, list(ids))] = ROLE_COLORS[role]
        boxes = boxes3d(pts, seg_s, role_ids, id_map)
        all_boxes[f"f{k}"] = boxes
        for b in boxes:  # wireframes into the ply
            edge = box_edge_points(b["min"], b["max"])
            pts = np.concatenate([pts, edge])
            cols = np.concatenate([cols, np.tile(ROLE_COLORS[b["role"]], (len(edge), 1))])
        write_ply(os.path.join(out_dir, f"pcd_depth_f{k}.ply"), pts, cols)
        topdown(pts, cols, boxes, os.path.join(out_dir, f"topdown_f{k}.png"))
        for b in boxes:
            if b["role"] in ("target", "destination"):
                print(f"  3D box [{b['role']:11s}] {b['name']:20s} "
                      f"center={b['center']} size={b['size']}")

    # ---- legend / report ----------------------------------------------------
    with open(os.path.join(out_dir, "legend.json"), "w") as f:
        json.dump({
            "cameras": cams, "frames": [int(x) for x in frames],
            "roles_from_task_code": roles,
            "clutter_object_types": clutter_names,
            "role_ids": {r: sorted(int(i) for i in ids) for r, ids in role_ids.items()},
            "robot_ids": sorted(int(i) for i in robot_ids),
            "obstacle_ids_spawned_clutter": sorted(int(i) for i in spawned_obstacle_ids),
            "obstacle_ids_environment": sorted(int(i) for i in env_ids),
            "environment_object_names": sorted({id_map[i] for i in env_ids}),
            "actor_id_map": {str(k): v for k, v in sorted(id_map.items())},
            "seg_ids_unmatched": [int(i) for i in unmatched],
            "boxes_3d": all_boxes,
            "cluttered_table_info_raw": ep_info.get("cluttered_table_info"),
        }, f, indent=2)

    instr_path = os.path.join(args.run_dir, "instructions", f"episode{ep}.json")
    if os.path.exists(instr_path):
        with open(instr_path) as f:
            instr = json.load(f)
        print(f"instruction (seen[0]): {instr.get('seen', [''])[0]}")

    print(f"\nWrote {out_dir}/: panel.png, legend.json, pcd_*.ply, topdown_*.png")


if __name__ == "__main__":
    main()
