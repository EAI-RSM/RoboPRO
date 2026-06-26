# Replayable State — interactive demo

A working, minimal demo of [`REPLAYABLE_STATE_PROPOSAL.md`](../REPLAYABLE_STATE_PROPOSAL.md).

It takes **one** already-collected episode (`put_mouse_on_pad`, clean baseline, seed 0) and
shows that everything interesting about it can be **reconstructed and re-derived offline from
the logged state trace** — no simulator, no re-collection. The whole 3D scene you scrub
through is rebuilt from an **80 KB** pose/joint trace; the original pixels for that episode
were **~40 MB**.

> Scope: deliberately one scenario. The point is to *demonstrate the idea end-to-end*, not to
> be a general pipeline.

![what it shows](_validation/reproject_demo_camera_f80.png)

*(validation render: the reconstruction — robot from FK in yellow, objects from the trace —
reprojected onto the real recorded RGB. They line up, which is the whole proof.)*

---

## What the demo shows (maps 1:1 to the proposal)

You open an HTML page with a 3D viewer and a timeline. From the state trace alone it derives,
live in the browser:

| In the UI | Proposal section | Why it matters |
|---|---|---|
| **Free-orbit to any camera** | §3 Tier A — render-derived | views that were *never collected*, on demand |
| **"Realistic ↔ Segmentation"** toggle | §3 Tier A / §8 | per-actor masks for free, no extra capture |
| **Robot reconstructed from `qpos`** (FK) | §4.2 state trace | full kinematic replay from 14 joint values/frame |
| **"Head cam / External cam" + RGB inset** | §11 success criterion 1 | round-trip: reconstruction matches the real pixels |
| **Spatial-relation sentence** ("mouse is left of … the pad") | §8 language re-annotation | instructions regenerated from poses |
| **Distance / height / placed / success** | §8 success & reward | success rule *recomputed* from poses post-hoc |
| **Camera frusta** overlay | §4.1 camera rig | where the collected cameras actually were |
| **Storage banner** (0.2 % of the pixels) | §5 storage inverts | the trace is ~1–2 % of the pixel footprint |

The scale of two distractor objects was **never recorded** (the "manifest gap" the proposal
calls out, §4.1). The exporter *recovers* it from the trace — the scale that drops the mesh
onto the table at frame 0 — which is itself a small instance of "derive what wasn't logged."

---

## Run it

```bash
./build.sh                       # regenerates the bundle (assets are git-ignored)
cd web && python -m http.server 8000
# open http://localhost:8000   (use a server, not file://, for ES-module + fetch)
```

`build.sh` runs three steps you can also run by hand:

1. `export_scene.py` — the **offline projector**. Reads the episode HDF5 (`targeted_state`
   pose trace + `joint_action` + `endpose` + cameras) and `episode.json`, reconstructs the
   scene manifest (actor → mesh + recovered scale, robot base pose from the embodiment
   config), runs **FK** (`yourdfpy` on `arx5_description_isaac.urdf`) for the robot, computes
   the derived semantics, and writes `web/data/scene.json` + `web/assets/*.glb` +
   collected RGB for the round-trip panel. **Nothing re-runs the simulator.**
2. `slim_assets.py` — downscales textures / decimates robot meshes so the bundle is ~13 MB.
3. `fetch_deps.py` — vendors three.js r160 into `web/vendor/` (so it runs offline).

## Layout

```
export_scene.py     offline projector: episode -> web bundle
slim_assets.py      mesh/texture slimming for the web
fetch_deps.py       download three.js runtime
build.sh            run all three
web/index.html      the UI
web/viewer.js       three.js viewer (Z-up SAPIEN world, scrubber, toggles, derived panel)
_validation/        numpy reprojection checks (proof the geometry is right)
```

## How it's validated (no GPU / no browser needed to trust it)

`_validation/reproject_check.py` projects the **reconstructed** mesh vertices (objects posed
by the trace, robot posed by FK) into a collected camera and overlays them on the real RGB.
Because the viewer consumes the *same* glbs and transforms, this numpy render is a faithful
predictor of the 3D view. Robot FK was independently checked against the recorded `endpose`
(x,y match to ~1 mm; a constant 0.15 m z-offset is just the tool-frame definition).

```bash
python _validation/reproject_check.py demo_camera 80
python _validation/project_check.py   head_camera 100   # state -> 2D keypoints on real RGB
```

## Caveats (it's a demo)

- One episode, one task. Environment (table/floor/wall) and the pad are rendered as
  primitives; the mouse/robot/distractors use real meshes.
- Distractor scales are *recovered* heuristically (resting-on-table / extent-normalized),
  because collection-time scale isn't in the data — exactly the gap the proposal proposes to
  close by writing a manifest.
- "Re-render" here is browser rasterization of the reconstructed scene, not a SAPIEN
  re-render. The proposal's Tier-A path would render in SAPIEN for pixel-exact depth/seg; the
  browser version is enough to *show the idea* interactively.
