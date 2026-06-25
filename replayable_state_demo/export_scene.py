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
import shutil

import cv2
import h5py
import numpy as np
import trimesh
import yourdfpy

# ----------------------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EP_DIR = os.path.join(ROOT, "negative_data_demo/clean/seed00000/baseline")
EP_HDF5 = os.path.join(EP_DIR, "data/episode0.hdf5")
EP_JSON = os.path.join(EP_DIR, "episode.json")
OBJ_ROOT = os.path.join(ROOT, "customized_robotwin/assets/objects")
EMB = os.path.join(ROOT, "customized_robotwin/assets/embodiments/aloha-agilex")
URDF = os.path.join(EMB, "urdf/arx5_description_isaac.urdf")

WEB = os.path.join(ROOT, "replayable_state_demo/web")
ASSETS = os.path.join(WEB, "assets")
DATA = os.path.join(WEB, "data")

# Robot base pose in world (from embodiment config.yml: robot_pose)
ROBOT_BASE_P = np.array([0.0, -0.65, 0.0])
ROBOT_BASE_Q = np.array([0.707, 0.0, 0.0, 0.707])  # wxyz, 90deg about Z
# Cameras to export RGB for (policy view + a nice external view)
RGB_CAMS = ["head_camera", "demo_camera"]
# glTF (Y-up) -> SAPIEN (Z-up) visual correction applied to object meshes
GLTF_TO_Z = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

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


def recover_scale(verts, R_corr, R_obj0, p0, table_top):
    """The manifest gap: per-object scale was not logged. Recover a uniform scale
    from the trace -- for a table-resting object, the scale that drops its lowest
    mesh vertex onto the table surface at frame 0. Wall-mounted / floating objects
    fall back to normalizing the largest extent to a plausible size."""
    vc = (R_corr[:3, :3] @ verts.T).T            # correction-aligned, unscaled
    world_z_dir = (R_obj0 @ vc.T).T[:, 2]         # local z component once posed
    min_z = float(world_z_dir.min())
    gap = float(p0[2] - table_top)                # origin height above table
    if 0.005 < gap < 0.20 and min_z < -0.02:      # plausibly resting on the table
        s = (table_top - p0[2]) / min_z
        if 0.02 < s < 2.0:
            return [s, s, s], "rest-on-table"
    ext = vc.max(0) - vc.min(0)                    # fallback: normalize size
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


def export_robot(f, n_frames):
    la = f["joint_action/left_arm"][:]
    ra = f["joint_action/right_arm"][:]
    lg = f["joint_action/left_gripper"][:]
    rg = f["joint_action/right_gripper"][:]
    rob, link_mesh = load_robot_meshes()
    Twb = mat(ROBOT_BASE_P, ROBOT_BASE_Q)

    moving = [f"fl_link{i}" for i in range(1, 9)] + [f"fr_link{i}" for i in range(1, 9)]
    moving = [m for m in moving if m in link_mesh]
    static = [l for l in link_mesh if l not in moving]

    # bake static links (base, wheels, body) into one world-space glb at cfg=0
    rob.update_cfg(fk_cfg(la, ra, lg, rg, 0))
    static_parts = []
    for l in static:
        Tbl = rob.get_transform(l, frame_from="base_link")
        m = link_mesh[l].copy()
        m.apply_transform(Twb @ Tbl)
        static_parts.append(m)
    if static_parts:
        export_mesh_glb(trimesh.util.concatenate(static_parts),
                        os.path.join(ASSETS, "robot_static.glb"))

    # export moving link meshes (link-local) once
    for l in moving:
        export_mesh_glb(link_mesh[l].copy(), os.path.join(ASSETS, f"robot_{l}.glb"))

    # per-frame world transform for each moving link
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
    sig = ep_json.get("signals", {})
    out["episode_success"] = bool(sig.get("success_flag", False))
    out["place_error_cm"] = round(float(sig.get("place_error_cm", 0)), 3)
    out["object_moved_cm"] = round(float(sig.get("object_moved_cm", 0)), 2)
    out["collision"] = bool(sig.get("collision_metrics", {}).get("is_collision", False))
    return out


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(ASSETS, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    f = h5py.File(EP_HDF5, "r")
    ep_json = json.load(open(EP_JSON))

    names = [n.decode() if isinstance(n, bytes) else n for n in f["targeted_state/actor_names"][:]]
    pos = f["targeted_state/actor_pos"][:].astype(np.float32)     # [T,A,3]
    quat = f["targeted_state/actor_quat"][:].astype(np.float32)   # [T,A,4] wxyz
    T, A = pos.shape[:2]
    print(f"episode: {ep_json['task_name']}  frames={T} actors={A}")

    table_top_z = float(np.median(pos[:, names.index("table"), 2])) if "table" in names else 0.74

    # ----- object manifest + meshes
    ent = ep_json.get("progress_entities", {})
    objects = []
    for i, name in enumerate(names):
        role = ("target" if name == ent.get("target")
                else "destination" if name == ent.get("destination")
                else "env" if name in ENV_ACTORS else "distractor")
        entry = {"name": name, "role": role, "color": PALETTE[i % len(PALETTE)],
                 "is_static": name in ENV_ACTORS, "actor_index": i}
        if name in ENV_ACTORS or name == ent.get("destination"):
            # rendered as primitives by the viewer (no reliable mesh authored as an actor)
            entry["primitive"] = name
            entry["scale"] = [1, 1, 1]
        else:
            glb = resolve_object_glb(name)
            if glb:
                shutil.copy(glb, os.path.join(ASSETS, f"{name}.glb"))
                entry["glb"] = f"assets/{name}.glb"
                entry["mesh_correction"] = wxyz_to_xyzw(
                    trimesh.transformations.quaternion_from_matrix(GLTF_TO_Z))
                ms = model_data_scale(name)
                if ms is not None:
                    entry["scale"], entry["scale_src"] = ms, "model_data"
                else:
                    verts = trimesh.load(glb, force="mesh").vertices
                    R0 = quat_wxyz_to_R(quat[0, i])
                    entry["scale"], entry["scale_src"] = recover_scale(
                        verts, GLTF_TO_Z, R0, pos[0, i], table_top_z)
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
    print("  building robot FK ...")
    try:
        robot = export_robot(f, T)
        print(f"    robot: {robot['n_links']} moving links + static base")
    except Exception as e:
        print("    robot FK failed, falling back to endpose gizmos:", e)
        lep, rep = f["endpose/left_endpose"][:], f["endpose/right_endpose"][:]
        robot = {"endpose_left": np.round(lep, 5).tolist(),
                 "endpose_right": np.round(rep, 5).tolist(), "moving_links": [], "static_glb": None}

    # ----- collected cameras (params + RGB) for round-trip / frusta
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
        rgb_dir = None
        if cam in RGB_CAMS:
            rgb_dir = os.path.join(DATA, "rgb", cam)
            os.makedirs(rgb_dir, exist_ok=True)
            for t in range(T):
                arr = np.frombuffer(bytes(g["rgb"][t]), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(rgb_dir, f"f{t:04d}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        cameras.append({
            "name": cam, "K": K,
            "extrinsic_cv": np.round(ext, 5).reshape(T, -1).tolist(),
            "cam2world_gl": np.round(c2w, 5).reshape(T, -1).tolist() if c2w is not None else None,
            "rgb": f"data/rgb/{cam}" if rgb_dir else None,
            "width": int(K[0][2] * 2), "height": int(K[1][2] * 2),
        })
    print(f"  cameras: {[c['name'] for c in cameras]}")

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

    tgt_label = "mouse"
    scene = {
        "meta": {
            "task_name": ep_json["task_name"],
            "task_config": ep_json["task_config"],
            "instruction": f"Put the {tgt_label} onto the pad.",
            "outcome": ep_json.get("outcome"),
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
        "cameras": cameras,
        "signals": signals,
    }
    out = os.path.join(DATA, "scene.json")
    with open(out, "w") as fp:
        json.dump(scene, fp, separators=(",", ":"))
    print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
