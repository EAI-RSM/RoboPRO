#!/usr/bin/env python
"""Build a SINGLE self-contained HTML for the targeted-negative demo.

Inlines every rollout video as a base64 data: URI and embeds the manifest, so the
result needs no server and no external files — open it straight from disk or email
it around. Run build_demo.py first (it produces web/manifest.json + web/videos/).

  python build_demo.py            # -> web/manifest.json + web/videos/*.mp4
  python build_selfcontained.py   # -> negative_targeted_data_demo.html (standalone)
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
OUT = ROOT / "negative_targeted_data_demo.html"


def datauri(relpath: str) -> str:
    raw = (WEB / relpath).read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")


def main():
    manifest = json.loads((WEB / "manifest.json").read_text())
    n = 0
    for t in manifest["tasks"]:
        for cell in [t["baseline"]] + t["perturbations"]:
            if cell and cell.get("video"):
                cell["video"] = datauri(cell["video"])
                n += 1
    html = (WEB / "index.html").read_text()
    embed = "<script>window.__MANIFEST__=" + json.dumps(manifest, separators=(",", ":")) + ";</script>\n"
    # inject the embedded manifest right before the page's (only) <script> block
    assert html.count("<script>") >= 1, "no <script> in index.html"
    html = html.replace("<script>", embed + "<script>", 1)
    OUT.write_text(html)
    print(f"wrote {OUT.name} — {n} videos inlined, {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
