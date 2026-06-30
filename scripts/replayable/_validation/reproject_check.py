"""Overlay the *reconstructed* scene (object glbs posed by the state trace + robot links
posed by FK) onto the real collected RGB, by projecting mesh vertices into the camera.
If silhouettes land on the real objects, mesh orientation/scale/pose/FK are all correct.
"""
import json
import os
import sys

import cv2
import h5py
import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
SCENE = json.load(open(os.path.join(WEB, "data/scene.json")))
# point EP_HDF5 at any collected episode0.hdf5
EP = os.environ.get("EP_HDF5", "")
CAM = sys.argv[1] if len(sys.argv) > 1 else "head_camera"
FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 0


def R_xyzw(q):
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_glb_vertices(path, n=400):
    m = trimesh.load(os.path.join(WEB, path), force="mesh")
    v = m.vertices
    if len(v) > n:
        v = v[np.random.choice(len(v), n, replace=False)]
    return v


def main():
    cam = next(c for c in SCENE["cameras"] if c["name"] == CAM)
    K = np.array(cam["K"])
    ext = np.array(cam["extrinsic_cv"][FRAME]).reshape(3, 4)
    R, t = ext[:, :3], ext[:, 3]

    f = h5py.File(EP, "r")
    g = f[f"observation/{CAM}"]
    img = cv2.imdecode(np.frombuffer(bytes(g["rgb"][FRAME]), np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    A = len(SCENE["objects"])
    opos = np.array(SCENE["frames"]["object_pos"][FRAME]).reshape(A, 3)
    oquat = np.array(SCENE["frames"]["object_quat"][FRAME]).reshape(A, 4)

    def draw(verts_world, color):
        Pc = (R @ verts_world.T).T + t
        good = Pc[:, 2] > 1e-3
        Pc = Pc[good]
        uv = (K @ Pc.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        for u, v in uv:
            if 0 <= u < W and 0 <= v < H:
                img[int(v), int(u)] = color

    # objects with glb meshes
    for i, obj in enumerate(SCENE["objects"]):
        if "glb" not in obj:
            continue
        verts = load_glb_vertices(obj["glb"])   # force='mesh' bakes the glb's own node xform
        s = np.array(obj["scale"])
        world = (R_xyzw(oquat[i]) @ (verts * s).T).T + opos[i]   # no added correction
        col = [int(255 * c) for c in reversed(obj["color"])]  # BGR
        draw(world, col)

    # robot moving links
    rob = SCENE["robot"]
    if rob.get("moving_links"):
        lp = np.array(rob["link_pose"][FRAME]).reshape(rob["n_links"], 7)
        for j, link in enumerate(rob["moving_links"]):
            verts = load_glb_vertices(link["glb"], n=200)
            world = (R_xyzw(lp[j, 3:]) @ verts.T).T + lp[j, :3]
            draw(world, [60, 220, 255])  # yellow
        if rob.get("static_glb"):
            verts = load_glb_vertices(rob["static_glb"], n=1500)
            draw(verts, [200, 200, 200])

    out = os.path.join(os.path.dirname(__file__), f"reproject_{CAM}_f{FRAME}.png")
    up = cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out, up)
    print("wrote", out)


if __name__ == "__main__":
    main()
