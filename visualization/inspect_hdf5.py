#!/usr/bin/env python3
"""Print the full tree of a collected episode HDF5: keys, shapes, dtypes.

For byte-string datasets (encoded JPEG/PNG frames) the first frame is decoded
to report the real image shape.

Usage:
    python inspect_hdf5.py <path/to/episodeN.hdf5>
"""
import sys

import cv2
import h5py
import numpy as np


def describe(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[group]   /{name}")
        return
    ds = obj
    line = f"[dataset] /{name}  shape={ds.shape}  dtype={ds.dtype}"
    if ds.dtype.kind == "S" and len(ds) > 0:
        buf = np.frombuffer(bytes(ds[0]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is not None:
            line += f"  -> decodes to {img.shape} {img.dtype}"
        else:
            line += "  -> (bytes, not a decodable image)"
    print(line)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    with h5py.File(path, "r") as f:
        print(f"== {path} ==")
        if f.attrs:
            for k, v in f.attrs.items():
                print(f"[attr]    {k} = {v!r}")
        f.visititems(describe)


if __name__ == "__main__":
    main()
