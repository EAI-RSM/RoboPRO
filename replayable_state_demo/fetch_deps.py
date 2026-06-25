#!/usr/bin/env python
"""Download the (third-party) three.js runtime into web/vendor/ so the demo runs
offline. Not committed to git -- regenerate with `python fetch_deps.py`."""
import os
import urllib.request

BASE = "https://unpkg.com/three@0.160.0/"
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web/vendor")
FILES = {
    "build/three.module.js": "three.module.js",
    "examples/jsm/loaders/GLTFLoader.js": "addons/loaders/GLTFLoader.js",
    "examples/jsm/controls/OrbitControls.js": "addons/controls/OrbitControls.js",
    "examples/jsm/utils/BufferGeometryUtils.js": "addons/utils/BufferGeometryUtils.js",
}

for src, dst in FILES.items():
    out = os.path.join(VENDOR, dst)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    urllib.request.urlretrieve(BASE + src, out)
    print(f"  {dst}  ({os.path.getsize(out)//1024} KB)")
print("vendored three.js r160 ->", VENDOR)
