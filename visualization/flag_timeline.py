#!/usr/bin/env python3
"""Print per-timestep contact/collision ranges of collected episodes, with the
matching video timestamps for visual verification.

- CONTACT windows show ALL contacts by default (min-impulse 0, resting/grazing
  included), each with its peak impulse. Pass --min-impulse X to hide light
  touches (0.01 = light brush; 0.04 = the collector's 10 N furniture gate).
- COLLISION windows additionally show the hit object's movement: its episode-
  start pose -> pose at the window end (from /object_poses), with the position
  delta in cm and rotation delta in degrees.

The collection video (video/episodeN.mp4) has one frame per dataset frame, so
dataset frame i appears at video time i/fps (fps read from the mp4, default 30).

Usage:
    python flag_timeline.py <run_dir> [ep_idx] [--min-impulse 0]

<run_dir> is the config-level output dir, e.g.
    data/<save_path>/<task_name>/<task_config>
(contains data/episodeN.hdf5, video/episodeN.mp4, scene_info.json).
"""
import argparse
import glob
import json
import os
import re

import cv2
import h5py
import numpy as np

ROBOT_HINTS = ("fl_link", "fr_link", "robot", "wheel_link", "base_link",
               "left_camera", "right_camera", "head_camera")


def _ranges(mask):
    """Contiguous index ranges where mask is nonzero: [(start, end), ...] inclusive."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    breaks = np.nonzero(np.diff(idx) > 1)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def _fmt_t(seconds):
    m, s = divmod(seconds, 60.0)
    return f"{int(m)}:{s:05.2f}"


def _fmt_pose(p):
    return "[" + ", ".join(f"{v:+.3f}" for v in p[:3]) + "]"


def _video_fps(video_path, default=30.0):
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return fps
    return default


def _read_pairs(f, key):
    if key not in f:
        return None
    out = []
    for row in f[key]:
        raw = bytes(row).rstrip(b"\0")
        try:
            out.append(json.loads(raw.decode("utf-8")) if raw else [])
        except Exception:
            out.append([])
    return out


def _load_episode_meta(run_dir, ep_idx):
    """(object_pose_ids, id->name map, task-role names) from scene_info.json.
    Role names = targets + destinations + intended-contact bodies: objects the
    TASK deliberately moves/touches, so their pose change is not collision
    evidence and should not be traced under a collision window."""
    p = os.path.join(run_dir, "scene_info.json")
    if not os.path.exists(p):
        return None, {}, set()
    try:
        db = json.load(open(p, encoding="utf-8"))
        e = db.get(f"episode_{ep_idx}", {})
        ids = e.get("object_pose_ids")
        amap = {int(k): v for k, v in (e.get("actor_id_map") or {}).items()}
        roles = e.get("role_names") or {}
        # schema mixes shapes: name lists (target_object_names, ...), scalar
        # names ("target": "001_bottle"), and int ids ("target_id": 97) — take
        # strings and lists-of-strings only (set.update(int) raises; update(str)
        # would splat characters)
        role_names = set()
        for v in roles.values():
            if isinstance(v, str):
                role_names.add(v)
            elif isinstance(v, (list, tuple)):
                role_names.update(x for x in v if isinstance(x, str))
        return ids, amap, role_names
    except Exception:
        return None, {}, set()


def _object_movement(pair, a, b, obj_poses, pose_ids, amap, role_names=frozenset()):
    """For a collision pair "x|y", pick the object side whose motion is the
    collision evidence — non-robot AND non-task-role (a target/destination's
    motion is intended, e.g. the mouse being organized; showing it under a
    collision window is misleading). Falls back to a role object, annotated,
    only when the pair has no clutter side. Duplicate names -> the column that
    moved most in the window. Returns (name, pose_frame0, pose_window_end,
    dpos_m, dang_deg, intended: bool)."""
    if obj_poses is None or pose_ids is None:
        return None
    # pair sides are "name#per_scene_id" (current datasets) or plain names
    # (legacy: office_v2/sweep_80). The id, when present, resolves the exact
    # /object_poses column — no guessing between same-named instances.
    sides = []
    for tok in pair.split("|"):
        name, _, sid = tok.partition("#")
        if not any(h in name for h in ROBOT_HINTS):
            sides.append((tok, name, int(sid) if sid.isdigit() else None))
    if not sides:
        return None
    clutter = [s for s in sides if s[1] not in role_names]
    obj_name, name, obj_sid = (clutter or sides)[0]
    intended = not clutter
    if obj_sid is not None:
        cols = [j for j, oid in enumerate(pose_ids) if int(oid) == obj_sid]
    else:
        cols = [j for j, oid in enumerate(pose_ids) if amap.get(int(oid)) == name]
    if not cols:
        return None
    end = min(b, obj_poses.shape[0] - 1)

    def _disp_from_t0(j, t):
        dp = float(np.linalg.norm(obj_poses[t, j, :3] - obj_poses[0, j, :3]))
        qd = abs(float(np.dot(obj_poses[t, j, 3:7], obj_poses[0, j, 3:7])))
        return dp, 2 * np.arccos(min(1.0, qd))

    def _peak(j):  # max displacement-from-t0 anywhere inside the window
        return max((_disp_from_t0(j, t) for t in range(a, end + 1)),
                   key=lambda v: v[0] + 0.1 * v[1])

    if len(cols) > 1:
        # duplicates: pick the instance whose IN-WINDOW PEAK displacement from
        # t0 is largest, rotation included (0.1 rad weighted ~ 1 cm, matching
        # the collision thresholds). End-vs-start position alone mistraces
        # rock-and-recover hits: the whacked twin can end near its start pose
        # while a resting twin wins on millimeters of drift.
        cols = [max(cols, key=lambda j: (lambda v: v[0] + 0.1 * v[1])(_peak(j)))]
    j = cols[0]
    p0, pe = obj_poses[0, j], obj_poses[end, j]
    dpos = float(np.linalg.norm(pe[:3] - p0[:3]))
    qdot = abs(float(np.dot(pe[3:7], p0[3:7])))
    dang = float(np.degrees(2 * np.arccos(min(1.0, qdot))))
    pk_dpos, pk_dang = _peak(j)
    return obj_name, p0, pe, dpos, dang, intended, pk_dpos, float(np.degrees(pk_dang))


def show_episode(run_dir, ep_idx, min_impulse):
    h5_path = os.path.join(run_dir, "data", f"episode{ep_idx}.hdf5")
    video_path = os.path.join(run_dir, "video", f"episode{ep_idx}.mp4")
    fps = _video_fps(video_path)

    with h5py.File(h5_path, "r") as f:
        contact = np.asarray(f["contact"]) if "contact" in f else None
        collision = np.asarray(f["collision"]) if "collision" in f else None
        contact_pairs = _read_pairs(f, "contact_pairs")
        collision_pairs = _read_pairs(f, "collision_pairs")
        c_imp = np.asarray(f["contact_impulse"]) if "contact_impulse" in f else None
        obj_poses = np.asarray(f["object_poses"]) if "object_poses" in f else None
        attrs = dict(f.attrs)
        T = f["joint_action/vector"].shape[0] if "joint_action/vector" in f else None

    pose_ids, amap, role_names = _load_episode_meta(run_dir, ep_idx)
    dt = float(attrs.get("physics_timestep", 1.0 / 250.0))  # J (N*s) -> F ~ J/dt

    regime = {1: "BLIND planner (collision-unaware)",
              0: "AWARE planner (collision-aware)",
              -1: "legacy coupling"}.get(int(attrs.get("planner_exclude_obstacles", -1)), "?")
    print(f"\n=== episode{ep_idx}  ({os.path.basename(run_dir)}) ===")
    print(f"  frames: {T}  |  video: {os.path.relpath(video_path)} @ {fps:g} fps "
          f"({_fmt_t((T or 0) / fps)} long)")
    print(f"  regime: {regime}  |  metrics on: {attrs.get('enable_collision_metrics')}")

    # ── contact: impulse-filtered ────────────────────────────────────────────
    if contact is None:
        print("  contact   MISSING")
    else:
        raw_n = int(np.count_nonzero(contact))
        if c_imp is not None:
            mask = (contact > 0) & (c_imp >= min_impulse)
            if min_impulse > 0:
                print(f"  contact   {int(mask.sum())}/{contact.size} frames with impulse >= "
                      f"{min_impulse:g} N*s  (raw contact: {raw_n} frames — lighter touches hidden)")
            else:
                print(f"  contact   {raw_n}/{contact.size} frames (all contacts incl. resting; "
                      f"filter with --min-impulse)")
        else:
            mask = contact > 0
            print(f"  contact   {raw_n}/{contact.size} frames "
                  f"(no impulse data — showing all)")
        for a, b in _ranges(mask):
            peak = float(c_imp[a:b + 1].max()) if c_imp is not None else float("nan")
            # .4g keeps tiny-but-real impulses visible (3.2e-04, not "0.000")
            print(f"      frames {a:4d}-{b:<4d}  ->  video {_fmt_t(a / fps)} - "
                  f"{_fmt_t((b + 1) / fps)}   (peak impulse {peak:.4g} N*s ~ {peak/dt:.1f} N)")
            if contact_pairs is not None:
                for pr in sorted({p for i in range(a, b + 1) for p in contact_pairs[i]}):
                    print(f"          · {pr.replace('|', '  <->  ')}")
        if not _ranges(mask):
            print("      (none)")

    # ── collision: with object movement ──────────────────────────────────────
    if collision is None:
        print("  collision MISSING")
    else:
        rngs = _ranges(collision)
        pct = 100.0 * int(collision.sum()) / max(1, collision.size)
        print(f"  collision {int(np.count_nonzero(collision))}/{collision.size} frames ({pct:.1f}%)")
        for a, b in rngs:
            print(f"      frames {a:4d}-{b:<4d}  ->  video {_fmt_t(a / fps)} - {_fmt_t((b + 1) / fps)}")
            if collision_pairs is None:
                continue
            window_pairs = sorted({p for i in range(a, b + 1) for p in collision_pairs[i]})
            shown_objs = set()
            for pr in window_pairs:
                print(f"          · {pr.replace('|', '  <->  ')}")
                mv = _object_movement(pr, a, b, obj_poses, pose_ids, amap, role_names)
                if mv is not None and mv[0] not in shown_objs:
                    shown_objs.add(mv[0])
                    name, p0, pe, dpos, dang, intended, pk_dpos, pk_dang = mv
                    note = "   (task target/destination - motion intended)" if intended else ""
                    # rock-and-recover: end-of-window pose understates the hit;
                    # show the in-window peak when it meaningfully exceeds it
                    peak = (f"   (peak in window: {pk_dpos*100:.1f} cm, {pk_dang:.1f} deg)"
                            if (pk_dpos > dpos + 0.002 or pk_dang > dang + 1.0) else "")
                    print(f"            {name}: pose@t0 {_fmt_pose(p0)} -> pose@t{min(b, (T or 1)-1)} "
                          f"{_fmt_pose(pe)}   moved {dpos*100:.1f} cm, {dang:.1f} deg{peak}{note}")
        if not rngs:
            print("      (none)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("ep_idx", nargs="?", type=int, default=None)
    ap.add_argument("--min-impulse", type=float, default=0.0,
                    help="hide contact frames whose max impulse (N*s) is below this "
                         "(default 0 = show ALL contacts; 0.01 light brush, 0.04 = 10 N gate)")
    args = ap.parse_args()
    run_dir = args.run_dir.rstrip("/")

    if args.ep_idx is not None:
        eps = [args.ep_idx]
    else:
        eps = sorted(
            int(m.group(1)) for p in glob.glob(os.path.join(run_dir, "data", "episode*.hdf5"))
            if (m := re.search(r"episode(\d+)\.hdf5$", p))
        )
        if not eps:
            print(f"No episodes found under {run_dir}/data/")
            raise SystemExit(1)

    for ep in eps:
        show_episode(run_dir, ep, args.min_impulse)
    print()


if __name__ == "__main__":
    main()
