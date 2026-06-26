#!/usr/bin/env python
"""
Offline "projector": turn ONE collected episode into a self-contained web bundle that
reconstructs the full 3D scene from nothing but the logged *state trace* + a reconstructed
scene manifest -- exactly the workflow proposed in REPLAYABLE_STATE_PROPOSAL.md.

Inputs  : episode{N}.hdf5 (targeted_state pose trace + joint_action + endpose + cameras)
          episode.json    (label / derived-signal sidecar)
          asset meshes     (object .glb, robot URDF .dae/.stl)
Outputs : replayable_state_demo/web/data/scene.json  (manifest + per-frame transforms +
                                                       derived semantics + camera params)
          replayable_state_demo/web/assets/*.glb      (object + robot link meshes)
          replayable_state_demo/web/data/rgb/<cam>/*.jpg (collected RGB for round-trip panel)

Nothing here re-runs the simulator. Every semantic (novel views, segmentation colours,
spatial-relation language, success, contact/grasp) is a pure function of the trace.
"""
import io
import json
import os
import re
import shutil

import cv2
import h5py
import numpy as np
import trimesh
import yourdfpy

# ----------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBJ_ROOT = os.path.join(ROOT, "customized_robotwin/assets/objects")
EMB = os.path.join(ROOT, "customized_robotwin/assets/embodiments/aloha-agilex")
URDF = os.path.join(EMB, "urdf/arx5_description_isaac.urdf")

WEB = os.path.join(ROOT, "replayable_state_demo/web")
ASSETS = os.path.join(WEB, "assets")
DATA = os.path.join(WEB, "data")
SCENES_DIR = os.path.join(DATA, "scenes")

# Robot base pose in world (from embodiment config.yml: robot_pose)
ROBOT_BASE_P = np.array([0.0, -0.65, 0.0])
ROBOT_BASE_Q = np.array([0.707, 0.0, 0.0, 0.707])  # wxyz, 90deg about Z
# Cameras to export RGB streams for (the layered viewer lets you switch between them)
RGB_CAMS = ["head_camera", "front_camera", "demo_camera", "left_camera"]
# Cameras to also export depth for (powers occlusion + depth point cloud + depth view)
DEPTH_CAMS = ["head_camera", "demo_camera"]

# Collected episodes to publish as switchable scenes (id, path under negative_data_demo, twin_of)
SCENES = [
    ("clean_baseline",     "clean/seed00000/baseline",     None),
    ("clean_shift_object", "clean/seed00000/shift_object", "clean_baseline"),
    ("clean_shift_target", "clean/seed00000/shift_target", "clean_baseline"),
    ("d6_baseline",        "d6/seed00000/baseline",        None),
    ("d6_shift_obstacle",  "d6/seed00000/shift_obstacle",  "d6_baseline"),
]

ENV_ACTORS = {"ground", "table", "wall"}
PALETTE = [
    [0.90, 0.30, 0.30], [0.30, 0.65, 0.95], [0.45, 0.80, 0.45], [0.95, 0.75, 0.30],
    [0.70, 0.45, 0.85], [0.40, 0.80, 0.80], [0.95, 0.55, 0.75], [0.60, 0.60, 0.60],
]


# ----------------------------------------------------------------------------- helpers
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


def resolve_object_glb(name):
    """Find a visual .glb for an object dir, preferring visual/base0.glb."""
    d = os.path.join(OBJ_ROOT, name)
    if not os.path.isdir(d):
        return None
    for cand in [os.path.join(d, "visual", "base0.glb"), os.path.join(d, "base.glb")]:
        if os.path.isfile(cand):
            return cand
    hits = []
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.endswith(".glb") and "collision" not in root:
                hits.append(os.path.join(root, fn))
    return sorted(hits)[0] if hits else None


def model_data_scale(name):
    """Collection-time scale, IF it was recorded in model_data (often it wasn't)."""
    for mdf in ["model_data0.json", "model_data1.json", "model_data.json"]:
        p = os.path.join(OBJ_ROOT, name, mdf)
        if os.path.isfile(p):
            d = json.load(open(p))
            s = d.get("scale")
            if isinstance(s, (list, tuple)):
                return [float(v) for v in s]
            if isinstance(s, (int, float)):
                return [float(s)] * 3
    return None


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
def load_robot_meshes():
    """Load each link's visual mesh in link-local frame (merged per link)."""
    rob = yourdfpy.URDF.load(URDF, load_meshes=False, build_scene_graph=True)
    urdf_dir = os.path.dirname(URDF)
    link_mesh = {}
    for lname, link in rob.link_map.items():
        parts = []
        for vis in link.visuals:
            g = vis.geometry
            if g.mesh is None:
                continue
            mp = os.path.join(urdf_dir, g.mesh.filename)
            if not os.path.isfile(mp):
                continue
            try:
                m = trimesh.load(mp, force="mesh")
            except Exception:
                continue
            sc = g.mesh.scale
            if sc is not None:
                m.apply_scale(sc)
            origin = vis.origin
            if origin is not None:
                origin = np.asarray(origin, float)
                if origin.shape == (4, 4):
                    m.apply_transform(origin)  # link -> visual
            parts.append(m)
        if parts:
            link_mesh[lname] = trimesh.util.concatenate(parts)
    return rob, link_mesh


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


def export_robot(f, n_frames, write_glbs=True):
    la = f["joint_action/left_arm"][:]
    ra = f["joint_action/right_arm"][:]
    lg = f["joint_action/left_gripper"][:]
    rg = f["joint_action/right_gripper"][:]
    rob, link_mesh = load_robot_meshes()
    Twb = mat(ROBOT_BASE_P, ROBOT_BASE_Q)

    moving = [f"fl_link{i}" for i in range(1, 9)] + [f"fr_link{i}" for i in range(1, 9)]
    moving = [m for m in moving if m in link_mesh]
    static = [l for l in link_mesh if l not in moving]

    rob.update_cfg(fk_cfg(la, ra, lg, rg, 0))
    static_parts = []
    for l in static:
        m = link_mesh[l].copy()
        m.apply_transform(Twb @ rob.get_transform(l, frame_from="base_link"))
        static_parts.append(m)
    if write_glbs:
        if static_parts:
            export_mesh_glb(trimesh.util.concatenate(static_parts),
                            os.path.join(ASSETS, "robot_static.glb"))
        for l in moving:
            export_mesh_glb(link_mesh[l].copy(), os.path.join(ASSETS, f"robot_{l}.glb"))

    link_pose = np.zeros((n_frames, len(moving), 7), np.float32)
    for t in range(n_frames):
        rob.update_cfg(fk_cfg(la, ra, lg, rg, t))
        for j, l in enumerate(moving):
            Tw = Twb @ rob.get_transform(l, frame_from="base_link")
            q = trimesh.transformations.quaternion_from_matrix(Tw)  # wxyz
            link_pose[t, j, :3] = Tw[:3, 3]
            link_pose[t, j, 3:] = wxyz_to_xyzw(q)
    return {
        "static_glb": "assets/robot_static.glb" if static_parts else None,
        "moving_links": [{"name": l, "glb": f"assets/robot_{l}.glb"} for l in moving],
        "link_pose": np.round(link_pose, 5).reshape(n_frames, -1).tolist(),
        "n_links": len(moving),
        "bones": skeleton_bones(moving),
    }


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
    rels = []
    for t in range(T):
        d = tp[t] - dp[t]
        ax = "right of" if d[0] > 0.03 else "left of" if d[0] < -0.03 else "aligned with"
        ay = "behind" if d[1] > 0.03 else "in front of" if d[1] < -0.03 else ""
        az = "above" if d[2] > 0.03 else ""
        parts = [p for p in (az, ax, ay) if p]
        rels.append(f"mouse is {' and '.join(parts)} the pad" if parts else "mouse is on the pad")
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


# ----------------------------------------------------------------------------- per-scene
def export_one(scene_id, ep_relpath, twin_of, write_glbs=False):
    ep_dir = os.path.join(ROOT, "negative_data_demo", ep_relpath)
    out_dir = os.path.join(SCENES_DIR, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    f = h5py.File(os.path.join(ep_dir, "data/episode0.hdf5"), "r")
    ep_json = json.load(open(os.path.join(ep_dir, "episode.json")))

    names = [n.decode() if isinstance(n, bytes) else n for n in f["targeted_state/actor_names"][:]]
    pos = f["targeted_state/actor_pos"][:].astype(np.float32)     # [T,A,3]
    quat = f["targeted_state/actor_quat"][:].astype(np.float32)   # [T,A,4] wxyz
    T, A = pos.shape[:2]
    print(f"[{scene_id}] {ep_json['task_name']}  frames={T} actors={A}  outcome={ep_json.get('outcome')}")

    table_top_z = float(np.median(pos[:, names.index("table"), 2])) if "table" in names else 0.74

    # ----- object manifest + meshes
    ent = ep_json.get("progress_entities", {})
    objects = []
    for i, name in enumerate(names):
        role = ("target" if name == ent.get("target")
                else "destination" if name == ent.get("destination")
                else "env" if name in ENV_ACTORS else "distractor")
        entry = {"name": name, "label": pretty_label(name, role), "role": role,
                 "id": i, "actor_index": i, "color": PALETTE[i % len(PALETTE)],
                 "is_static": name in ENV_ACTORS, "tracked": role != "env"}
        if name in ENV_ACTORS:
            entry["primitive"] = name
            entry["scale"] = [1, 1, 1]
        elif role == "destination":
            # the placement pad: a thin slab in the actor frame
            entry["primitive"] = "pad"
            entry["scale"] = [1, 1, 1]
            half = [0.115, 0.085, 0.006]
            entry["bbox"] = {"center": [0, 0, 0], "half": half}
            entry["points"] = sample_points(
                trimesh.creation.box(extents=[2 * h for h in half]), 200)
        else:
            glb = resolve_object_glb(name)
            if glb:
                if write_glbs:                                    # glbs only feed the 3D scene viewer
                    shutil.copy(glb, os.path.join(ASSETS, f"{name}.glb"))
                entry["glb"] = f"assets/{name}.glb"
                mesh = trimesh.load(glb, force="mesh")            # node-baked actor frame
                verts = mesh.vertices
                ms = model_data_scale(name)
                if ms is not None:
                    entry["scale"], entry["scale_src"] = ms, "model_data"
                else:
                    R0 = quat_wxyz_to_R(quat[0, i])
                    entry["scale"], entry["scale_src"] = recover_scale(
                        verts, R0, pos[0, i], table_top_z)
                s = np.asarray(entry["scale"], float)
                lo, hi = verts.min(0) * s, verts.max(0) * s    # object-frame AABB (scaled)
                entry["bbox"] = {"center": [round(x, 4) for x in (lo + hi) / 2],
                                 "half": [round(x, 4) for x in (hi - lo) / 2]}
                entry["points"] = sample_points(mesh, 600 if role == "target" else 450, s)
            else:
                entry["primitive"] = "box"
                entry["scale"] = [1, 1, 1]
        objects.append(entry)
        info = (f"glb scale={[round(s,3) for s in entry['scale']]}({entry.get('scale_src')})"
                if "glb" in entry else entry.get("primitive"))
        print(f"  actor {name:18s} role={role:11s} {info}")

    # ----- per-frame object transforms (xyzw quats)
    obj_pos = np.round(pos, 5).reshape(T, -1).tolist()
    obj_quat = np.zeros((T, A, 4), np.float32)
    for t in range(T):
        for a in range(A):
            obj_quat[t, a] = wxyz_to_xyzw(quat[t, a])
    obj_quat = np.round(obj_quat, 5).reshape(T, -1).tolist()

    # ----- robot via FK
    try:
        robot = export_robot(f, T, write_glbs=write_glbs)
    except Exception as e:
        print("    robot FK failed, falling back to endpose gizmos:", e)
        lep, rep = f["endpose/left_endpose"][:], f["endpose/right_endpose"][:]
        robot = {"endpose_left": np.round(lep, 5).tolist(),
                 "endpose_right": np.round(rep, 5).tolist(), "moving_links": [], "static_glb": None}

    # ----- collected cameras (params + RGB streams + depth)
    cameras = []
    pixel_bytes = 0
    for cam in f["observation"]:
        if cam == "pointcloud" or not isinstance(f["observation"][cam], h5py.Group):
            continue
        g = f["observation"][cam]
        if "rgb" not in g:
            continue
        pixel_bytes += sum(g["rgb"][k].nbytes for k in range(T))
        c2w = g["cam2world_gl"][:] if "cam2world_gl" in g else None
        K = g["intrinsic_cv"][0].tolist()
        ext = g["extrinsic_cv"][:]  # [T,3,4]
        rgb_path = depth_path = depth_range = None
        if cam in RGB_CAMS:
            rgb_dir = os.path.join(out_dir, "rgb", cam)
            os.makedirs(rgb_dir, exist_ok=True)
            for t in range(T):
                img = cv2.imdecode(np.frombuffer(bytes(g["rgb"][t]), np.uint8), cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(rgb_dir, f"f{t:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 72])
            rgb_path = f"data/scenes/{scene_id}/rgb/{cam}"
            if cam in DEPTH_CAMS and "depth" in g:
                depth_range = list(export_depth_stream(g, T, os.path.join(out_dir, "depth", cam)))
                depth_path = f"data/scenes/{scene_id}/depth/{cam}"
        cameras.append({
            "name": cam, "K": K,
            "extrinsic_cv": np.round(ext, 5).reshape(T, -1).tolist(),
            "cam2world_gl": np.round(c2w, 5).reshape(T, -1).tolist() if c2w is not None else None,
            "rgb": rgb_path, "depth": depth_path, "depth_range": depth_range,
            "width": int(K[0][2] * 2), "height": int(K[1][2] * 2),
        })
    print(f"    cameras: {[c['name'] for c in cameras if c['rgb']]}  depth: {[c['name'] for c in cameras if c['depth']]}")

    # ----- grippers (end-effector grasp frames + openness) for overlays & scene graph
    def grip(ep, gv):
        out = np.zeros((T, 7), np.float32)
        out[:, :3] = ep[:, :3]
        for t in range(T):
            out[t, 3:] = wxyz_to_xyzw(ep[t, 3:7])
        return {"pose": np.round(out, 5).reshape(T, -1).tolist(),
                "open": np.round(np.asarray(gv, float), 4).tolist()}
    grippers = {
        "left": grip(f["endpose/left_endpose"][:], f["joint_action/left_gripper"][:]),
        "right": grip(f["endpose/right_endpose"][:], f["joint_action/right_gripper"][:]),
    }

    # ----- derived semantics
    signals = derive_signals(names, pos, ep_json, table_top_z)

    # ----- storage accounting (the headline tradeoff)
    state_bytes = (pos.nbytes + quat.nbytes
                   + f["joint_action/vector"].nbytes
                   + sum(f[f"endpose/{k}"].nbytes for k in f["endpose"]))
    storage = {
        "state_trace_kb": round(state_bytes / 1024, 1),
        "pixels_mb": round(pixel_bytes / 1e6, 2),
        "ratio_pct": round(100 * state_bytes / max(pixel_bytes, 1), 2),
        "n_cameras": len(cameras),
    }
    print(f"  storage: state={storage['state_trace_kb']}KB  pixels={storage['pixels_mb']}MB "
          f"({storage['ratio_pct']}%)")

    scene = {
        "meta": {
            "scene_id": scene_id,
            "twin_of": twin_of,
            "task_name": ep_json["task_name"],
            "task_config": ep_json["task_config"],
            "instruction": "Put the mouse onto the pad.",
            "outcome": ep_json.get("outcome"),
            "perturbation_type": ep_json.get("perturbation_type"),
            "seed": ep_json.get("scene_seed"),
            "n_frames": T,
            "fps": 15,
            "world_up": [0, 0, 1],
            "entities": ent,
            "table_top_z": round(table_top_z, 4),
            "robot_base": {"p": ROBOT_BASE_P.tolist(), "q_xyzw": wxyz_to_xyzw(ROBOT_BASE_Q)},
        },
        "storage": storage,
        "objects": objects,
        "frames": {"object_pos": obj_pos, "object_quat": obj_quat},
        "robot": robot,
        "grippers": grippers,
        "cameras": cameras,
        "signals": signals,
    }
    out = os.path.join(out_dir, "scene.json")
    with open(out, "w") as fp:
        json.dump(scene, fp, separators=(",", ":"))
    print(f"    wrote {os.path.relpath(out, WEB)}  ({os.path.getsize(out)/1024:.0f} KB)")
    return {
        "id": scene_id, "twin_of": twin_of, "task": ep_json["task_name"],
        "config": ep_json["task_config"], "outcome": ep_json.get("outcome"),
        "perturbation_type": ep_json.get("perturbation_type"), "n_frames": T,
        "scene": f"data/scenes/{scene_id}/scene.json",
    }


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(SCENES_DIR, exist_ok=True)
    index = []
    for i, (sid, rel, twin) in enumerate(SCENES):
        try:
            index.append(export_one(sid, rel, twin, write_glbs=(i == 0)))
        except Exception as e:
            print(f"  SKIP {sid}: {e}")
    with open(os.path.join(DATA, "scenes.json"), "w") as fp:
        json.dump({"scenes": index, "default": index[0]["id"] if index else None}, fp, indent=1)
    print(f"wrote scenes.json with {len(index)} scenes: {[s['id'] for s in index]}")


if __name__ == "__main__":
    main()
