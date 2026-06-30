#!/usr/bin/env python
"""Shrink the exported glb bundle for fast web loading:
  - object meshes: downscale oversized textures (keep geometry + UVs intact),
  - robot meshes (no textures): quadric-decimate the heavy geometry.
Run after export_scene.py. Idempotent on already-small files.
"""
import glob
import os

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web/assets")


def decimate(geom, target_faces):
    if len(geom.faces) <= target_faces:
        return geom
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(geom.vertices)),
        o3d.utility.Vector3iVector(np.asarray(geom.faces)))
    m = m.simplify_quadric_decimation(int(target_faces))
    out = trimesh.Trimesh(vertices=np.asarray(m.vertices),
                          faces=np.asarray(m.triangles), process=False)
    out.remove_unreferenced_vertices()      # decimation leaves orphan verts behind
    out.merge_vertices()
    return out


def shrink_textures(geom, maxsz):
    mat = getattr(geom.visual, "material", None)
    if mat is None:
        return
    for attr in ("baseColorTexture", "metallicRoughnessTexture", "normalTexture",
                 "occlusionTexture", "emissiveTexture", "image"):
        img = getattr(mat, attr, None)
        if isinstance(img, Image.Image) and max(img.size) > maxsz:
            r = maxsz / max(img.size)
            new = img.resize((max(1, int(img.size[0] * r)), max(1, int(img.size[1] * r))))
            setattr(mat, attr, new)


def process(path):
    before = os.path.getsize(path)
    is_robot = os.path.basename(path).startswith("robot_")
    loaded = trimesh.load(path)
    geoms = loaded.geometry if isinstance(loaded, trimesh.Scene) else {"g": loaded}
    if is_robot:
        # robot meshes are single, identity-graph -> safe to rebuild; decimate hard
        out = trimesh.Scene()
        for name, g in geoms.items():
            out.add_geometry(decimate(g, 6000), geom_name=name)
        export = out
    else:
        # objects rely on glTF node transforms (scale/placement) -> must NOT rebuild
        # the scene graph; only edit textures in place and re-export the SAME scene.
        maxsz = 1024 if "mouse" in path else 512
        for g in geoms.values():
            shrink_textures(g, maxsz)
        export = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    with open(path, "wb") as f:
        f.write(trimesh.exchange.gltf.export_glb(export))
    after = os.path.getsize(path)
    print(f"  {os.path.basename(path):24s} {before/1e6:7.2f} -> {after/1e6:6.2f} MB")


def main():
    total_b = total_a = 0
    for path in sorted(glob.glob(os.path.join(ASSETS, "*.glb"))):
        total_b += os.path.getsize(path)
        process(path)
        total_a += os.path.getsize(path)
    print(f"total assets: {total_b/1e6:.1f} -> {total_a/1e6:.1f} MB")


if __name__ == "__main__":
    main()
