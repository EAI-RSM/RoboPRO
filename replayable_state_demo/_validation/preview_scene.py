"""Offline preview of the *viewer's* composed scene from its default camera.
Mirrors viewer.js transform composition exactly (object mesh_correction+scale, robot
static world mesh + FK link poses, env primitives) so we can eyeball the initial view
without a browser. Renders sampled mesh vertices as colored points.
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(ROOT, "replayable_state_demo/web")
SCENE = json.load(open(os.path.join(WEB, "data/scene.json")))
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 60
W, H = 960, 540

# viewer default camera
EYE = np.array([1.1, -1.25, 1.45])
TGT = np.array([0.0, -0.05, SCENE["meta"]["table_top_z"] + 0.04])
UP = np.array([0.0, 0.0, 1.0])
FOV = 45.0


def R_xyzw(q):
    x, y, z, w = q
    n = np.linalg.norm([x, y, z, w]) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


f = TGT - EYE; f = f / np.linalg.norm(f)
r = np.cross(f, UP); r = r / np.linalg.norm(r)
u = np.cross(r, f)
Rw2c = np.stack([r, -u, f])                       # cam x=right, y=down, z=forward
fy = (H / 2) / np.tan(np.radians(FOV) / 2); fx = fy
img = np.full((H, W, 3), 245, np.uint8)
zbuf = np.full((H, W), 1e9, np.float32)


def draw(verts, color, n=2500):
    if len(verts) > n:
        verts = verts[np.random.choice(len(verts), n, replace=False)]
    Pc = (Rw2c @ (verts - EYE).T).T
    Pc = Pc[Pc[:, 2] > 1e-3]
    u_ = fx * Pc[:, 0] / Pc[:, 2] + W / 2
    v_ = fy * Pc[:, 1] / Pc[:, 2] + H / 2
    for x, y, z in zip(u_, v_, Pc[:, 2]):
        xi, yi = int(x), int(y)
        if 0 <= xi < W and 0 <= yi < H and z < zbuf[yi, xi]:
            zbuf[yi, xi] = z
            cv2.circle(img, (xi, yi), 2, color, -1)


def glb_verts(path):
    return trimesh.load(os.path.join(WEB, path), force="mesh").vertices


A = len(SCENE["objects"])
op = np.array(SCENE["frames"]["object_pos"][FRAME]).reshape(A, 3)
oq = np.array(SCENE["frames"]["object_quat"][FRAME]).reshape(A, 4)

# env primitives (match viewer)
ztop = SCENE["meta"]["table_top_z"]
def box_verts(cx, cy, cz, sx, sy, sz):
    b = trimesh.creation.box(extents=[sx, sy, sz]); b.apply_translation([cx, cy, cz]); return b.vertices
draw(box_verts(0.1, -0.05, ztop - 0.02, 1.5, 0.95, 0.04), (205, 205, 200))   # tabletop
draw(box_verts(0.1, -0.05, (ztop - 0.05) / 2, 1.2, 0.7, ztop - 0.05), (190, 190, 180))  # pedestal

# objects
for i, o in enumerate(SCENE["objects"]):
    col = tuple(int(255 * c) for c in reversed(o["color"]))
    if "glb" in o:
        v = glb_verts(o["glb"])             # force='mesh' already bakes the glb's own node xform
        s = np.array(o["scale"])
        world = (R_xyzw(oq[i]) @ (v * s).T).T + op[i]   # no added correction (matches three.js)
        draw(world, col)
    elif o.get("primitive") == "box" and o["role"] == "destination":
        v = box_verts(0, 0, 0, 0.23, 0.17, 0.01)
        world = (R_xyzw(oq[i]) @ v.T).T + op[i]
        draw(world, (60, 60, 70))

# robot
rob = SCENE["robot"]
if rob.get("static_glb"):
    draw(glb_verts(rob["static_glb"]), (170, 175, 185), n=6000)
if rob.get("link_pose"):
    lp = np.array(rob["link_pose"][FRAME]).reshape(rob["n_links"], 7)
    for j, lk in enumerate(rob["moving_links"]):
        v = glb_verts(lk["glb"])
        world = (R_xyzw(lp[j, 3:]) @ v.T).T + lp[j, :3]
        draw(world, (120, 140, 160))

out = os.path.join(os.path.dirname(__file__), f"preview_f{FRAME}.png")
cv2.imwrite(out, img)
print("wrote", out)
