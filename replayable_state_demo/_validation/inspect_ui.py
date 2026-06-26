"""Headless inspection of the viewer: load it, capture console + page errors, screenshot.
Usage: python inspect_ui.py [url] [out.png] [wait_ms]"""
import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/claude-449600015/-work-mohammed-EAI-RSM-RoboPRO/cdca9023-751d-41ee-b9b5-cefcc2c936b5/scratchpad/ui.png"
WAIT = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=1)
    msgs, errs = [], []
    pg.on("console", lambda m: msgs.append(f"[{m.type}] {m.text}"))
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL, wait_until="networkidle", timeout=20000)
    pg.wait_for_timeout(WAIT)
    pg.screenshot(path=OUT, full_page=False)
    print("=== PAGE ERRORS ===")
    for e in errs:
        print("  ", e)
    print("=== CONSOLE (errors/warnings) ===")
    for m in msgs:
        if m.startswith("[error]") or m.startswith("[warning]"):
            print("  ", m[:300])
    # probe live state
    state = pg.evaluate("""() => {
        const cv=document.getElementById('cv');
        return {cvW:cv&&cv.width, cvH:cv&&cv.height, camtag:(document.getElementById('camtag')||{}).textContent,
                sceneOpts:[...document.querySelectorAll('#sceneSel option')].map(o=>o.textContent),
                bodyStartsPre: document.body.firstElementChild && document.body.firstElementChild.tagName};
    }""")
    print("=== STATE ===", state)
    print("screenshot:", OUT)
    b.close()
