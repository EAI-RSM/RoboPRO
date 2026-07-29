---
name: repo_sapien_gotchas
description: "SAPIEN 3.0.0b1 API breaks vs the 2.x docs, and why the interactive viewer is unusable here"
metadata:
  type: project
---

**API (3.0.0b1 differs from the docs):**
- `sapien.Engine()` and `sapien.SapienRenderer()` are deprecated — use `sapien.Scene()` directly.
- `actor.get_velocity()` does not exist. Use
  `actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).linear_velocity`.
- `set_camera_shader_dir("default")` = rasterization (fast), `"rt"` = ray tracing (slow, high
  quality). This repo runs `"rt"` at 32 spp.

**The interactive viewer is broken in the VSCode terminal** — it freezes, reports "not
responding", and renders a snapshot of the VSCode window. Cause: RT shading is too heavy for
interactive use and SAPIEN's Vulkan surface conflicts with VSCode's GPU rendering.
*How to apply:* always use `--no-render --save_data` for headless video. If you truly need the
viewer, run from a standalone terminal outside VSCode. The `Large_D435` camera variant in
`_camera_config.yml` gives 640×480 instead of the default 320×240.

Do not try to switch the shader to rasterizer as a speedup — see [[domain_visibility]] for why
that dead-ends.
