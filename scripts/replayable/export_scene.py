#!/usr/bin/env python
"""
Offline "projector" CLI: turn ONE collected episode into a self-contained web bundle
that reconstructs the full 3D scene from nothing but the logged *state trace* + a
reconstructed scene manifest.

Inputs  : episode{N}.hdf5 (targeted_state pose trace + joint_action + endpose + cameras)
          episode.json    (label / derived-signal sidecar)
          asset meshes     (object .glb, robot URDF .dae/.stl)
Outputs : web/data/scenes/<id>/scene.json  (manifest + per-frame transforms +
                                             derived semantics + camera params)
          web/assets/*.glb                  (object + robot link meshes)
          web/data/scenes/<id>/rgb|depth/   (collected streams for the round-trip panel)

Nothing here re-runs the simulator. The pure offline projectors (FK, scale recovery,
depth decode, spatial-relation language, success/grasp/lift) live in the importable
library `robo_tools.replayable`; this script owns the path-bound parts: asset
resolution, robot-FK mesh loading, the scene list, and the output bundle layout.
"""
import json
import os
import shutil

import cv2
import h5py
import numpy as np
import trimesh
import yourdfpy

from robo_tools.replayable import (
    ENV_ACTORS, derive_signals, export_depth_stream, export_mesh_glb, fk_cfg,
    mat, pretty_label, quat_wxyz_to_R, recover_scale, sample_points,
    skeleton_bones, wxyz_to_xyzw,
)

# ----------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))          # scripts/replayable
ROOT = os.path.dirname(os.path.dirname(HERE))              # repo root
OBJ_ROOT = os.path.join(ROOT, "customized_robotwin/assets/objects")
EMB = os.path.join(ROOT, "customized_robotwin/assets/embodiments/aloha-agilex")
URDF = os.path.join(EMB, "urdf/arx5_description_isaac.urdf")

WEB = os.path.join(HERE, "web")                            # bundle next to this CLI
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

# Collected episodes to publish as switchable scenes (id, repo-relative path, twin_of).

# Variety set: 20 different bench tasks (fixed seed 0) across all four scenes.
# 17 clean_success + 3 longest clean_failure (the scripted expert misses on the
# harder articulated tasks — kept as honest failure scenarios). Scene id == task.
VARIETY_TASKS = [
    # office (5)
    "close_drawer", "put_book_in_fileholder", "put_phone_on_holder",
    "put_stapler_on_book", "put_milktea_on_shelf",
    # study (6)
    "move_cup", "put_cup_on_coaster", "move_book_onto_table",
    "put_seal_in_box", "empty_box", "put_pen_in_pencup",
    # kitchenl (3)
    "pick_can_from_basket", "put_sauce_can_in_basket", "move_milk_close_fridge",
    # kitchens (6)
    "drop_apple_in_bin_ks", "close_microwave_ks", "put_spoon_in_sink_ks",
    "put_bread_on_board_ks", "move_hamburger_onto_plate_ks", "place_bowl_in_dishrack_ks",
]
_DEMO = "../dummy_data/consolidated_demo/runs"   # fresh enriched episodes (runner format)
SCENES = [
    ("bread_clean",        f"{_DEMO}/put_bread_on_board_ks/baseline",     None),
    ("bread_shift_object", f"{_DEMO}/put_bread_on_board_ks/shift_object", "bread_clean"),
    ("bread_shift_target", f"{_DEMO}/put_bread_on_board_ks/shift_target", "bread_clean"),
    ("bread_high_arc",     f"{_DEMO}/put_bread_on_board_ks/high_arc",     "bread_clean"),
    ("stapler_clean",        f"{_DEMO}/put_stapler_on_book/baseline",     None),
    ("stapler_shift_object", f"{_DEMO}/put_stapler_on_book/shift_object", "stapler_clean"),
    ("stapler_shift_target", f"{_DEMO}/put_stapler_on_book/shift_target", "stapler_clean"),
]
# (kept for reference) standard variety set: [(t, f"enriched_variety_data/runs/{t}", None) for t in VARIETY_TASKS]

PALETTE = [
    [0.90, 0.30, 0.30], [0.30, 0.65, 0.95], [0.45, 0.80, 0.45], [0.95, 0.75, 0.30],
    [0.70, 0.45, 0.85], [0.40, 0.80, 0.80], [0.95, 0.55, 0.75], [0.60, 0.60, 0.60],
]


# ----------------------------------------------------------------------------- asset resolution
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


# ----------------------------------------------------------------------------- per-scene
def export_one(scene_id, ep_relpath, twin_of, write_glbs=False):
    ep_dir = os.path.join(ROOT, ep_relpath)
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
            "instruction": ep_json.get("instruction") or ep_json["task_name"].replace("_", " "),
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
