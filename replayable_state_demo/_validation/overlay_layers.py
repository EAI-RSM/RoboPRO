"""Validate the layered-viewer overlay math (3D bbox + trajectory + gripper + ID),
drawn onto the real collected RGB exactly as the JS layer will. If boxes hug the
objects and trajectories track them, the projection is right."""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEB = os.path.join(ROOT, "replayable_state_demo/web")
S = json.load(open(os.path.join(WEB, "data/scene.json")))
CAM = sys.argv[1] if len(sys.argv) > 1 else "demo_camera"
FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 80

cam = next(c for c in S["cameras"] if c["name"] == CAM)
K = np.array(cam["K"])
ext = np.array(cam["extrinsic_cv"][FRAME]).reshape(3, 4)
R, t = ext[:, :3], ext[:, 3]
img = cv2.imread(os.path.join(WEB, "data/rgb", CAM, f"f{FRAME:04d}.jpg"))
H, W = img.shape[:2]
A = len(S["objects"])
OP, OQ = S["frames"]["object_pos"], S["frames"]["object_quat"]


def Rq(q):
    x, y, z, w = q
    n = np.linalg.norm(q) or 1
    x, y, z, w = np.array(q) / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def proj(P):  # world -> pixel (or None if behind)
    Pc = R @ P + t
    if Pc[2] <= 1e-4:
        return None
    uv = K @ Pc
    return np.array([uv[0] / uv[2], uv[1] / uv[2]])


def opos(f, i):
    return np.array(OP[f][i * 3:i * 3 + 3])


def oquat(f, i):
    return OQ[f][i * 4:i * 4 + 4]


EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

for o in S["objects"]:
    if not o.get("tracked") or "bbox" not in o:
        continue
    i = o["id"]
    col = tuple(int(255 * c) for c in reversed(o["color"]))  # BGR
    p, Rm = opos(FRAME, i), Rq(oquat(FRAME, i))
    c0, hf = np.array(o["bbox"]["center"]), np.array(o["bbox"]["half"])
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(proj(p + Rm @ (c0 + np.array([sx, sy, sz]) * hf)))
    for a, b in EDGES:
        if corners[a] is not None and corners[b] is not None:
            cv2.line(img, tuple(corners[a].astype(int)), tuple(corners[b].astype(int)), col, 1, cv2.LINE_AA)
    # trajectory up to this frame
    pts = [proj(opos(f, i)) for f in range(0, FRAME + 1, 2)]
    pts = [q for q in pts if q is not None]
    for a in range(1, len(pts)):
        cv2.line(img, tuple(pts[a - 1].astype(int)), tuple(pts[a].astype(int)), col, 1, cv2.LINE_AA)
    cc = proj(p)
    if cc is not None:
        cv2.putText(img, f"{o['id']}:{o['label']}", tuple((cc + [4, -4]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

# grippers
for side, gcol in (("left", (80, 200, 255)), ("right", (255, 200, 80))):
    gp = np.array(S["grippers"][side]["pose"][FRAME][:3])
    q = proj(gp)
    if q is not None:
        cv2.drawMarker(img, tuple(q.astype(int)), gcol, cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(img, f"{side[0].upper()}-grip", tuple((q + [6, 0]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, gcol, 1, cv2.LINE_AA)

out = os.path.join(os.path.dirname(__file__), f"overlay_layers_{CAM}_f{FRAME}.png")
cv2.imwrite(out, cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST) if W < 400 else img)
print("wrote", out)
