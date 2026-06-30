"""Replayable-state projection library — pure offline projectors over a logged
state trace (no simulator needed).

The scene-export CLI (`scripts/replayable/export_scene.py`) turns one collected
episode into a self-contained web bundle: every semantic shown in the viewer
(novel views, segmentation colours, spatial-relation language, success,
grasp/lift, occlusion) is a pure function of the trace + a reconstructed scene
manifest. Those projectors live here; the CLI owns the path-bound parts (asset
resolution, robot-FK mesh loading, and the output bundle layout).

Heavier deps (cv2, trimesh) live beyond robo_tools' core (numpy/h5py) — they are
only imported when this submodule is imported, which only the export CLI does, so
`import robo_tools` itself stays light.
"""
import os
import re

import cv2
import numpy as np
import trimesh

# Scene actors treated as static environment (not tracked / coloured as objects).
ENV_ACTORS = {"ground", "table", "wall"}


# ----------------------------------------------------------------------------- math
def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat(p, q_wxyz):
    T = np.eye(4)
    T[:3, 3] = p
    T[:3, :3] = quat_wxyz_to_R(q_wxyz)
    return T


def wxyz_to_xyzw(q):
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def pretty_label(name, role):
    if role == "destination":
        return "pad"
    return re.sub(r"^\d+_", "", name).replace("-", " ").replace("_", " ")


# ----------------------------------------------------------------------------- geometry
def sample_points(mesh, n, scale=(1, 1, 1)):
    """Surface point cloud in the object (node-baked) frame, scaled. Flat [n*3]."""
    pts = trimesh.sample.sample_surface(mesh, n)[0] * np.asarray(scale, float)
    return np.round(pts.astype(np.float32), 4).reshape(-1).tolist()


def export_depth_stream(g, T, out_dir):
    """Decode the PNG/uint16-mm depth, normalize to 8-bit over the stream's range, write
    one grayscale PNG/frame. Returns (near_m, far_m) so the viewer can recover metric depth
    for occlusion tests and back-projection."""
    frames = [cv2.imdecode(np.frombuffer(bytes(g["depth"][t]), np.uint8), cv2.IMREAD_UNCHANGED)
              for t in range(T)]
    arr = np.stack(frames).astype(np.float32)          # [T,H,W] mm
    valid = arr[arr > 0]
    if valid.size:
        near = float(np.percentile(valid, 1))
        far = float(min(np.percentile(valid, 99), near + 2500.0))   # cap span at 2.5 m for precision
    else:
        near, far = 300.0, 1500.0
    rng = max(far - near, 1.0)
    os.makedirs(out_dir, exist_ok=True)
    for t in range(T):
        d = frames[t].astype(np.float32)
        g8 = np.clip((d - near) / rng * 255.0, 0, 255).astype(np.uint8)
        g8[frames[t] == 0] = 255                       # invalid -> far
        cv2.imwrite(os.path.join(out_dir, f"f{t:04d}.png"), g8)
    return round(near / 1000.0, 4), round(far / 1000.0, 4)


def recover_scale(verts, R_obj0, p0, table_top):
    """The manifest gap: per-object scale was not logged. Recover a uniform scale
    from the trace -- for a table-resting object, the scale that drops its lowest
    mesh vertex onto the table surface at frame 0. Wall-mounted / floating objects
    fall back to normalizing the largest extent to a plausible size.
    `verts` are already in the SAPIEN actor frame (glb node transforms baked in by
    trimesh force='mesh'), so no extra rotation is applied."""
    world_z = (R_obj0 @ verts.T).T[:, 2]          # vertex height once posed at frame 0
    min_z = float(world_z.min())
    gap = float(p0[2] - table_top)                # origin height above table
    if 0.005 < gap < 0.20 and min_z < -0.02:      # plausibly resting on the table
        s = (table_top - p0[2]) / min_z
        if 0.02 < s < 2.0:
            return [s, s, s], "rest-on-table"
    ext = verts.max(0) - verts.min(0)             # fallback: normalize size
    s = 0.35 / float(ext.max())
    return [s, s, s], "extent-norm"


def export_mesh_glb(mesh, out_path):
    scene = mesh if isinstance(mesh, trimesh.Scene) else trimesh.Scene(mesh)
    with open(out_path, "wb") as f:
        f.write(trimesh.exchange.gltf.export_glb(scene))


# ----------------------------------------------------------------------------- robot FK
def fk_cfg(la, ra, lg, rg, t):
    d = {}
    for i in range(6):
        d[f"fl_joint{i+1}"] = float(la[t, i])
        d[f"fr_joint{i+1}"] = float(ra[t, i])
    # gripper: map normalized value -> finger joint via config gripper_scale [-0.01,0.045]
    glo, ghi = -0.01, 0.045
    lv = glo + (ghi - glo) * float(np.clip(lg[t], 0, 1))
    rv = glo + (ghi - glo) * float(np.clip(rg[t], 0, 1))
    for j in ("fl_joint7", "fl_joint8"):
        d[j] = lv
    for j in ("fr_joint7", "fr_joint8"):
        d[j] = rv
    return d


def skeleton_bones(moving):
    """Chain bones for the two arms (link1..6) + the two gripper fingers (link6->7,8)."""
    idx = {n: i for i, n in enumerate(moving)}
    bones = []
    for arm in ("fl", "fr"):
        for k in range(1, 6):
            a, b = f"{arm}_link{k}", f"{arm}_link{k+1}"
            if a in idx and b in idx:
                bones.append([idx[a], idx[b]])
        for fng in (7, 8):
            a, b = f"{arm}_link6", f"{arm}_link{fng}"
            if a in idx and b in idx:
                bones.append([idx[a], idx[b]])
    return bones


# ----------------------------------------------------------------------------- semantics
def derive_signals(names, pos, ep_json, table_top_z):
    """Pure offline projectors over the pose trace: distances, spatial relations,
    grasp/lift, success -- none of which were stored per frame."""
    ent = ep_json.get("progress_entities", {})
    tgt, dst = ent.get("target"), ent.get("destination")
    ti = names.index(tgt) if tgt in names else 0
    di = names.index(dst) if dst in names else 1
    T = pos.shape[0]
    out = {"target": tgt, "destination": dst}
    tp, dp = pos[:, ti, :], pos[:, di, :]
    out["dist_target_dest_cm"] = np.round(np.linalg.norm(tp - dp, axis=1) * 100, 2).tolist()
    out["dist_xy_cm"] = np.round(np.linalg.norm(tp[:, :2] - dp[:, :2], axis=1) * 100, 2).tolist()
    out["target_height_cm"] = np.round((tp[:, 2] - table_top_z) * 100, 2).tolist()
    # spatial relation of target wrt destination (world: +x right, +y back/away, +z up)
    def _clean(nm, fallback):
        return re.sub(r"^\d+_", "", nm).replace("-", " ").replace("_", " ") if nm else fallback
    tlab, dlab = _clean(tgt, "object"), _clean(dst, "goal")
    rels = []
    for t in range(T):
        d = tp[t] - dp[t]
        ax = "right of" if d[0] > 0.03 else "left of" if d[0] < -0.03 else "aligned with"
        ay = "behind" if d[1] > 0.03 else "in front of" if d[1] < -0.03 else ""
        az = "above" if d[2] > 0.03 else ""
        parts = [p for p in (az, ax, ay) if p]
        rels.append(f"{tlab} is {' and '.join(parts)} the {dlab}" if parts
                    else f"{tlab} is on the {dlab}")
    out["relation"] = rels
    # lifted / placed / grasp heuristics
    h = np.array(out["target_height_cm"])
    dxy = np.array(out["dist_xy_cm"])
    out["lifted"] = (h > 3.0).tolist()
    out["placed"] = ((dxy < 4.0) & (h < 2.0) & (np.arange(T) > T // 2)).tolist()
    # --- curves for the timeline + label panel (pure functions of the trace) ---
    out["target_speed_cm"] = np.round(
        np.concatenate([[0.0], np.linalg.norm(np.diff(tp, axis=0), axis=1) * 100]), 2).tolist()
    d0 = max(out["dist_xy_cm"][0], 1e-3)
    out["progress"] = [round(max(0.0, min(1.0, 1 - dxy_ / d0)), 3) for dxy_ in out["dist_xy_cm"]]
    others = [j for j, nm in enumerate(names) if nm not in ENV_ACTORS and j != ti]
    clr = (np.min([np.linalg.norm(pos[:, j, :] - tp, axis=1) for j in others], axis=0) * 100
           if others else np.full(T, 100.0))
    out["clearance_cm"] = np.round(clr, 2).tolist()
    out["safety"] = [round(min(1.0, c / 10.0), 3) for c in clr]
    # discrete events for the timeline strip
    events = []
    f0 = next((t for t in range(T) if out["lifted"][t]), None)
    if f0 is not None:
        events.append({"frame": f0, "kind": "lift", "label": "lift"})
    fp = next((t for t in range(T) if out["placed"][t]), None)
    if fp is not None:
        events.append({"frame": fp, "kind": "place", "label": "placed"})
    out["events"] = events
    sig = ep_json.get("signals", {})
    out["episode_success"] = bool(sig.get("success_flag", False))
    out["place_error_cm"] = round(float(sig.get("place_error_cm", 0)), 3)
    out["object_moved_cm"] = round(float(sig.get("object_moved_cm", 0)), 2)
    out["collision"] = bool(sig.get("collision_metrics", {}).get("is_collision", False))
    out["outcome"] = ep_json.get("outcome")
    out["perturbation_type"] = ep_json.get("perturbation_type")
    return out
