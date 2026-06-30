"""Sanity check: project logged 3D actor poses into a collected camera and overlay
on the decoded RGB. If markers land on the right objects, our world + camera +
quaternion conventions are correct -- which is the precondition for the 3D viewer.

This is itself a tiny demonstration of the proposal: the per-frame 2D keypoints are
*derived offline* from the state trace + camera params, not stored.
"""
import io
import json
import os
import sys

import cv2
import h5py
import numpy as np

# episode HDF5 to reproject; point EP_HDF5 at any collected episode0.hdf5
EP = os.environ.get("EP_HDF5", "")
CAM = sys.argv[1] if len(sys.argv) > 1 else "head_camera"
FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"overlay_{CAM}_f{FRAME}.png")


def decode_jpeg(buf):
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR


def main():
    f = h5py.File(EP, "r")
    names = [n.decode() if isinstance(n, bytes) else n for n in f["targeted_state/actor_names"][:]]
    pos = f["targeted_state/actor_pos"][FRAME]   # [A,3]
    quat = f["targeted_state/actor_quat"][FRAME]  # [A,4]

    obs = f[f"observation/{CAM}"]
    K = obs["intrinsic_cv"][FRAME]      # [3,3]
    ext = obs["extrinsic_cv"][FRAME]    # [3,4] world->cam (OpenCV)
    rgb = decode_jpeg(obs["rgb"][FRAME])
    H, W = rgb.shape[:2]

    R = ext[:, :3]
    t = ext[:, 3]

    print(f"cam={CAM} frame={FRAME} img={W}x{H}")
    print("K=\n", np.round(K, 1))
    img = rgb.copy()
    for i, name in enumerate(names):
        Pw = pos[i]
        Pc = R @ Pw + t            # camera frame (OpenCV: z forward)
        if Pc[2] <= 1e-4:
            on = "behind"
            continue
        uv = K @ Pc
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        inside = (0 <= u < W) and (0 <= v < H)
        print(f"  {name:18s} world={np.round(Pw,3)}  px=({u:7.1f},{v:7.1f}) {'IN ' if inside else 'out'} depth={Pc[2]:.3f}")
        if inside:
            cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)
            cv2.circle(img, (int(u), int(v)), 7, (255, 255, 255), 1)
            cv2.putText(img, name, (int(u) + 8, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(OUT, img)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
