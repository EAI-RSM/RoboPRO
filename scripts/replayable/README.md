# Replayable State — interactive demo

A working demo of the **replayable-state** idea: take collected episodes and show that almost
everything interesting about them is a **deterministic offline projection of the logged state
trace** — new views, depth/segmentation, 3D boxes, point clouds, spatial-relation scene graphs,
robot kinematics, success labels — none of which needs the simulator or re-collection.

The data this consumes is exactly the per-frame actor-pose trace that
[`robo_tools`](../../robo_tools/) logs (`targeted_state`), plus the standard bench
`joint_action` / `endpose` / camera streams. The path-free projectors live in the importable
library [`robo_tools.replayable`](../../robo_tools/src/robo_tools/replayable.py); this directory
holds the CLI (`export_scene.py`) and the web bundle.

## What it is

A dependency-free, browser-based **layered viewer** (`web/index.html` + `web/layers.js`) over the
collected **camera streams**, with toggleable overlay layers — all projected from the state trace:

- **Base view:** RGB · **Depth** · **Segmentation** (per-object masks)
- **Overlays:** object point clouds, 3D bounding boxes, IDs, trajectories, velocity, grippers,
  **robot skeleton** (FK), **occlusion** (depth z-test), and a **gripper-relation scene graph**
  drawn on the image (heat = closest, edges labelled with distance + 3D direction)
- **Per-twin footers:** a top-down **minimap** + **scene graph** under each view
- **Multi-scene:** a left-rail **scene browser** groups episodes by task — each clean baseline is a
  *root* with its perturbations (shift_object / shift_target) and alternative good actions (e.g.
  high_arc) **branching** beneath it — and **compare twins** side-by-side (synced, Δ divergence readout)
- **Re-derive success** live with adjustable thresholds; **event timeline** + progress/safety/speed
  sparklines per view; per-layer and per-object toggles (with all/none)

The experimental **3D scene reconstruction** (three.js) lives at `web/scene3d.html`.

## Run it

```bash
./build.sh                 # regenerate the bundle for all scenes (git-ignored output)
python serve.py 8000       # threaded static server -> fast frame streaming
# open http://localhost:8000   (use the server, not file://)
```

`build.sh` runs three steps you can also run by hand:

1. `export_scene.py` — the **offline projector**. For each episode it reconstructs the manifest
   (actor→mesh + recovered scale, robot base pose), runs **FK** (`yourdfpy`), decodes **depth**,
   samples object **point clouds**, derives semantics + progress/safety curves, and writes
   `web/data/scenes/<id>/scene.json` (+ `rgb/`, `depth/`) and a `scenes.json` index. **No simulator.**
2. `slim_assets.py` — slims the glb meshes (only needed by the 3D `scene3d.html`).
3. `fetch_deps.py` — vendors three.js (only needed by `scene3d.html`).

## Layout

```
export_scene.py     offline projector: episodes -> web bundle (scenes + depth + skeleton + curves)
serve.py            threaded static server
slim_assets.py      mesh slimming for the 3D viewer
fetch_deps.py       three.js for the 3D viewer
build.sh            run export + slim + fetch
web/index.html      the layered viewer
web/layers.js       viewer engine (projection, layers, twin-compare, scene loader, preloading)
web/scene3d.html    experimental three.js reconstruction (+ web/viewer.js)
_validation/        numpy projection checks + headless (Playwright) UI inspection
```

## Validation

Geometry is validated by reprojecting the reconstructed overlays onto the real RGB
(`_validation/overlay_layers.py`, `reproject_check.py`). The UI itself is inspected headlessly with
Playwright (`_validation/inspect_ui.py`) — load the page, capture console errors, screenshot.

## Caveats (it's a demo)

- Distractor object **scales** are recovered heuristically (the collection-time manifest isn't logged
  — exactly the gap the proposal proposes to close), so some distractors may sit slightly off.
- Depth is exported for the head + external cameras only; occlusion/depth-view are limited to those.
- Cluttered scenes can have many objects — use the Objects **none** toggle or turn off "Relations on view" to declutter.
