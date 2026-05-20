# benchmark/asset_unsure/

Bench-team asset variants that existed in the old `benchmark/bench_assets/`
location but were **never wired into the runtime loaders**. Recovered here
during the asset consolidation refactor (PR / commit `d97bb67`) so the work
isn't lost while we triage.

All three remaining items are variants of objects that the 80 tasks *do*
use — but the tasks load the copy in `benchmark/assets/objects/<name>/`,
not these. Triage whether any variant should replace the loaded copy.

| Item | Files | Why preserved |
|---|---|---|
| `044_microwave/` | 316 | Custom `mobility.urdf` referencing 134 collision meshes (CoACD output) — runtime uses the upstream URDF with 16 meshes from `benchmark/assets/objects/044_microwave/`. Used by all 40 kitchen tasks (scene furniture). |
| `034_knife/` | 4 | `model_data0.json` had different `contact_points_pose` (likely re-annotated grasp points). Used as an object-OOD / distractor object. |
| `122_file-holder/` | 1 | Bench had a sub-named `122_file_holder/base.glb`; runtime expects `base.glb` at the dir root. Used by all 20 office tasks (scene furniture). |

Four other items (`121_cabinet_cjcyed`, `123_drawer_dsbcxl`,
`126_fridge_hivvdfn`, `120_storage-rack`) were also stashed here but
confirmed dead — zero references in any `bench_envs/` file or
`bench_task_config/` — and deleted.

## To use one

If a bench task should actually load one of these, either:
1. **Replace runtime version**: copy the variant into `benchmark/assets/objects/<name>/` (overwrites upstream copy). Re-run `python scripts/install/download_assets.py` would undo this — see (2) for a more durable fix.
2. **Wire up loader**: add explicit handling in the relevant `bench_envs/` file to point at `BENCH_ROOT/asset_unsure/<name>/...`.

## To discard

If after triage you decide an item is truly dead, `rm -rf benchmark/asset_unsure/<name>` and commit.
