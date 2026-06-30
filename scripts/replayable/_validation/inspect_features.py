"""Exercise the viewer's risky features headlessly and screenshot each, capturing errors."""
import sys
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000"
OUT = "/tmp/claude-449600015/-work-mohammed-EAI-RSM-RoboPRO/cdca9023-751d-41ee-b9b5-cefcc2c936b5/scratchpad/"
errs = []

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 900})
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto(URL, wait_until="networkidle", timeout=20000)
    pg.wait_for_timeout(3500)
    # seek to mid-episode where the mouse is grasped
    pg.eval_on_selector("#scrub", "el => { el.value = 90; el.dispatchEvent(new Event('input')); }")
    pg.wait_for_timeout(2500)

    def shot(name): pg.screenshot(path=OUT + name)

    shot("f_rgb.png")
    pg.click("#cmpBtn"); pg.wait_for_timeout(4000); shot("f_twin.png")          # compare twin
    pg.click("#cmpBtn"); pg.wait_for_timeout(500)                               # off
    pg.click('#baseSel .tbtn:nth-child(2)'); pg.wait_for_timeout(1500); shot("f_depth.png")   # Depth
    pg.click('#baseSel .tbtn:nth-child(3)'); pg.wait_for_timeout(1500); shot("f_seg.png")     # Segmentation
    pg.click('#baseSel .tbtn:nth-child(1)'); pg.wait_for_timeout(300)                          # RGB
    pg.check("#L-occ"); pg.check("#L-skel"); pg.wait_for_timeout(1500); shot("f_occ_skel.png") # occlusion + skeleton
    # switch to d6 (cluttered) scene
    pg.select_option("#sceneSel", label="d6_shift_obstacle (success_with_collision)")
    pg.wait_for_timeout(3500)
    pg.eval_on_selector("#scrub", "el => { el.value = 100; el.dispatchEvent(new Event('input')); }")
    pg.wait_for_timeout(2500); shot("f_d6.png")

    print("=== ERRORS during interaction ===")
    for e in dict.fromkeys(errs):
        print("  ", e[:200])
    print("none" if not errs else f"({len(errs)} total)")
    b.close()
