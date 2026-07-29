---
name: repo_task_assets
description: "Editing the object catalog: task_objects.yml pitfalls, OOD ID rule, unregistered objaverse assets, and the expert-first eval gate"
metadata:
  type: project
---

From the June 2026 object/OOD work stream. Still true; consult before touching the catalog.

**`benchmark/bench_task_config/task_objects.yml` structural quirks:**
- `scales` and `z_offset` sections are FULLY DUPLICATED with identical content, so Edit often
  fails with "found 2 matches" — locate the right section by line number first.
- `yaml.dump()` REFORMATS the whole file (expands compact `[0,1,2]` into block style). Never
  round-trip this file through yaml.load/dump; use sed or manual edits.
- Duplicate top-level keys silently take the LAST value (the file once had two `object_ood:`
  blocks; only the second was live).
- Verify any edit with `--no-physics` validation.

**OOD ID rule:** each variant ID for a category within a scene must belong to exactly ONE
distribution (seen OR OOD) — overlap contaminates evaluation. Two pre-existing bugs were fixed
this way (`077_phone/4`, `120_plant/0`). Check with
`python ../scripts/validation/validate_ood_objects.py --no-physics` (runs in seconds).

**152 objaverse objects sit unregistered** at `benchmark/assets/objects/objaverse/` (26
categories). They are URDF format (`textured.obj` + `coacd_collision.obj` + `model.urdf`), not the
standard glb; the load path exists (`create_cluttered_urdf_obj()` in
`rand_create_cluttered_actor.py`) but `get_obstacle_objects_subset()` reads the standard
`model_data{id}.json` path. Don't register them without resolving that format mismatch.

**Eval runs the expert FIRST as a feasibility gate** (`eval_policy.py`, the `play_once()` call in the seed loop): `play_once()` runs,
and if the expert fails the seed is skipped and the policy never sees it. So any new object or
task must be expert-solvable before it can be used for policy evaluation, and high-density
(d10+) expert failures are normal. This gate is what blocked the pi05 smoke test — see
[[tool_vla_pi05_port]]. Test feasibility with
`visualize_task_scene.py --rollout --no-render`.
