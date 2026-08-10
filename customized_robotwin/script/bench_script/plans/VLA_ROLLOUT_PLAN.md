# VLA rollout on the occluder scene — plan

Executable plan for Codex. Written 2026-07-30 against branch `codex/bench-script-refactor`,
HEAD `73679af`.

**Goal.** One script the user can run that drives the ported **pi05 RoboPRO checkpoint** through the
occluder-ring scene and **saves a video per episode**, in the same run-folder style as
`analyze_occluder_visibility.py`. This is a *validation* deliverable: the question it answers is
"does the VLA actually run end to end in our scene and produce sane videos", not "how good is it".

**Read first:** `agent-memory/tool_vla_pi05_port.md` (port status, checkpoint spec, the blocker),
`agent-memory/domain_bench_script_layout.md` (the `lib/` dependency rule),
`agent-memory/status_current.md`.

---

## 0. Ground rules

**No expert.** The prior smoke test died in the two-pass expert feasibility gate
(`eval_policy_client.py` runs `play_once()` to validate a seed before handing control to the
policy; `put_mouse_on_pad.grasp_actor_from_table` raises `IndexError` there). **A VLA rollout does
not need the expert.** Do not call `play_once`. Do not "fix" the grasp problem to get past the gate
— that is a different, long-standing workstream and the user has directed it be left alone.

**Do not modify the expert / seeding / clearance path.** No edits to `task/*_mixin.py`,
`lib/widest_path.py`, `lib/seed_from_clearance.py`, or anything the Phase 4 A/B run depends on. This
work must not move any already-collected number.

**Dependency rule holds.** `lib/` imports nothing from a CLI script. New shared code goes in `lib/`.

**Reuse, don't reinvent.** Almost every piece already exists somewhere: the scene loop in
`analyze_occluder_visibility.py`, the policy eval loop in `eval_policy_client.py`, the RPC client in
both of those plus `collect_rollout_client.py`, the dual-env launcher in
`policy/pi05/eval_double_env.sh`. This is mostly assembly.

**Environment.** Run from `customized_robotwin/` with `source set_env.sh` and
`export ROBOTWIN_BENCH_TASK=bench`. Single GPU (RTX 4080, 16 GB) — `GPU_SPEC` must be `0:0`.

**One stage per commit.** Stages are ordered so Stage 1 can kill the whole plan cheaply.

---

## 1. Stage 1 — inference smoke test (GATE; do this before writing anything)

The port is code-complete but **model load + inference has never once succeeded**. Everything below
assumes `policy.infer()` works. Settle that first, standalone, with no SAPIEN in the process.

Write a throwaway `script/bench_script/smoke_test_pi05_infer.py` (or a scratch script — it does not
need to be committed):

1. Start `script/policy_model_server.py` in `policy/pi05/.venv` with the args from
   `policy/pi05/eval_double_env.sh` (train config `pi05_robopro_top_cam_jax`, model `robopro`,
   checkpoint `30000`).
2. Connect a `ModelClient`, call `set_language("Put the mouse onto the mouse pad.")`, then
   `update_observation_window` with three synthetic `uint8` 240×320×3 images and a
   `float32[14]` state.
3. Call `get_action()`.

**Pass condition:** returns a finite array of shape `[50, 14]`.

Record in the commit message (and in `agent-memory/tool_vla_pi05_port.md`) the peak VRAM the server
alone occupies — Stage 4 depends on it.

**If this fails, STOP and report.** Do not proceed to build a driver for a model that cannot infer.

---

## 2. Stage 2 — env plumbing gaps

Three small, contained changes. Each is needed by Stage 3 and none of them touches expert logic.

**2a. `task_config/_eval_step_limit.yml`** — `put_mouse_on_pad` is absent, so `eval_mode` falls back
to `step_lim = 1000`. Add an entry at **600**, in line with comparable office tasks.

**2b. `lib/scene_build.py::build_cfg`** — add a mode for policy rollouts. It currently has exactly
two shapes (`rollout=False` measurement, `rollout=True` expert). Add a third that sets:

- `eval_mode: True` (without it `step_lim` is `None` and the rollout loop is a no-op)
- `eval_video_save_dir: <episode video dir>`
- `need_plan: False`, `save_data: False`, `measurement_only: False`
- `render_freq: 0`

Do this by extending the existing signature (e.g. `mode="measure"|"expert"|"policy"`) rather than
adding a fourth boolean. **The existing two call paths must produce byte-identical cfgs.**

**2c. VRAM guard (only if Stage 4 shows you need it).** `Robot.set_planner` builds a `CuroboPlanner`
for *both* arms unconditionally (`envs/robot/robot.py`) — `need_plan=False` only short-circuits
`move_to_pose`, not construction. So a policy rollout pays for curobo it never uses, co-resident
with a ~7 GB bf16 pi05 on a 16 GB card. If needed, gate planner construction behind a flag that
defaults to today's behaviour. **Do not change the default.**

---

## 3. Stage 3 — `script/bench_script/vla_rollout.py` (the deliverable)

The main artifact. Structurally `analyze_occluder_visibility.py` with the visibility work removed and
the expert rollout replaced by a policy loop.

**Keep from `analyze_occluder_visibility.py`:**
- `make_occluder_task()` env construction, reused across episodes
- per-seed ring formation via `draw_ring_config` / `occluder_ring_xy`, drawn **once** and re-asserted
  on the env before the build (the "measure one scene, roll out another" trap)
- the pad-distance rejection (`OCC_PAD_MIN_DIST`) and the redraw-on-unstable-seed loop, so
  `--num-seeds N` yields N usable scenes
- the run-folder convention: `<out-dir>/<run-type>/<YYYYmmdd-HHMMSS>/` with `records.jsonl`,
  `log/episode{N}.log` (tee'd stdout), and success/fail bucketing of videos
- the CLI surface for scene geometry: `--seed-start`, `--num-seeds`, `--offsets`,
  `--num-occluders`, `--random-ring-rotation`, `--clutter-densities`, `--no-occluder-prob`

**Delete:** the clean-denominator build, `measure_target_visibility`, `save_overlay`,
`--save-images`, `--bins`, `--group-by`, `--plot-only`, and both `analyze*` calls. Also collapse the
two-pass buffering — its only purpose was keeping measurement and rollout on the same scene. Keep the
per-build stability gate, drop the buffering.

**Add — the policy loop.** Lifted from `eval_policy_client.py`:

```
model.call(func_name='reset_model')
while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
    observation = TASK_ENV.get_obs()
    eval_func(TASK_ENV, model, observation)      # policy/pi05/deploy_policy.py::eval
    if TASK_ENV.eval_success:
        break
```

Plus, around it:
- **ffmpeg pipe** exactly as `eval_policy_client.py` does it: `video_size="320x240"` (countertop is
  D435; `_base_task.take_action` already writes `countertop_camera` rgb into the pipe every step),
  then `TASK_ENV._set_eval_video_ffmpeg(ffmpeg)` before the loop and `_del_eval_video_ffmpeg()` after.
- **Instruction, without the expert.** `put_mouse_on_pad.play_once` has its `self.info["info"]` block
  commented out, so `generate_episode_descriptions` yields nothing useful anyway. Read
  `benchmark/bench_task_config/instruction_bank.json` directly — it has 10 phrasings under
  `put_mouse_on_pad`. Sample deterministically from the episode seed. Add `--instruction` to pin one
  string for debugging.
- **Outcome:** `env.check_success()`. Record per episode.
- **Failure isolation:** an exception in one episode must be caught, logged, marked failed, and the
  loop continues to the next seed — one bad scene may not abort a 50-episode run.

**`records.jsonl`, one line per episode:** `seed`, `offset`, `num_occluders`, `occluder_radii`,
`occluder_angle0`, `occluder_shown`, `clutter_density`, `instruction`, `success`, `steps_taken`,
`step_lim`, `wall_seconds`, `video_relpath`, `failure_reason`, and `policy` /`checkpoint_id` so the
folder is self-describing. Also `collision_metrics` via `env.get_collision_metrics()` if it is cheap
— it is already enabled in `bench_demo_office_clean`.

**New CLI:** `--port` (model server), `--pi0-step`, `--max-steps` (override `step_lim`),
`--run-type` (default `vla`).

**`ModelClient`:** three copies already exist (`eval_policy_client.py`, `collect_rollout_client.py`,
and inline). Do **not** add a fourth — move one verbatim into `lib/model_client.py` and have the new
script import it. Leave the existing two copies alone; deduplicating them is out of scope.

---

## 4. Stage 4 — launcher `policy/pi05/vla_occluder_rollout.sh`

Clone `policy/pi05/eval_double_env.sh` and change two things:

- the client half runs `script/bench_script/vla_rollout.py` instead of `script/eval_policy_client.py`,
  forwarding `--port` and the scene args
- **Leave the JAX memory env vars as they are, and confirm why.** The server subshell already sets
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, which makes JAX grow on demand — and `MEM_FRACTION` only
  governs the *preallocation* fraction, so the `0.85` sitting next to it should be inert. That
  on-demand growth is most likely what makes single-GPU co-residency work at all. **Verify this in
  Stage 1 with `nvidia-smi` while the server is idle-loaded**: if the server sits near its ~7 GB
  weight footprint rather than near 0.85 × card, the reading is confirmed and nothing needs changing.
  If it instead grabs most of the card, `PREALLOCATE=false` is not taking effect and *that* is the
  bug to fix — do not paper over it by tuning `MEM_FRACTION`.

Keep everything else verbatim — the free-port discovery, the `exec` trick for a clean `SERVER_PID`,
the `trap` cleanup, the explicit `export CUDA_VISIBLE_DEVICES` (the inline-prefix form was found
unreliable inside the `( ... ) &` subshell).

Default `GPU_SPEC` to `0:0`.

---

## 5. Verification

No test runner is configured. The bar, in order:

1. `python script/bench_script/vla_rollout.py --help` — exercises the full import chain, no GPU.
2. `python script/bench_script/analyze_occluder_visibility.py --help` and
   `cd script/bench_script && python -m lib.seed_from_clearance --help` still works — proves the `build_cfg`
   change in 2b did not break the existing callers.
3. `python script/bench_script/test_ring_config.py` and `test_obstacle_set.py` — CPU, fast. Must pass
   unchanged; the ring one is what guarantees the scene is reproducible.
4. **`--num-seeds 1 --num-occluders 1`, GPU, end to end.** Success criteria: an `.mp4` exists, is
   non-empty, and plays; `records.jsonl` has one complete line; VRAM does not OOM.
5. **`--num-seeds 5`.** Watch VRAM across episodes — the leak history in `status_current.md` is that
   per-episode rebuilds have leaked before. Report whether reserved memory is flat.

---

## 6. Deliverables

| # | Artifact | Stage |
|---|---|---|
| 1 | Confirmation that pi05 loads and returns `[50,14]`, + measured server VRAM | 1 |
| 2 | `put_mouse_on_pad: 600` in `task_config/_eval_step_limit.yml` | 2a |
| 3 | `build_cfg` policy mode, existing call paths byte-identical | 2b |
| 4 | `lib/model_client.py` | 3 |
| 5 | **`script/bench_script/vla_rollout.py`** | 3 |
| 6 | `policy/pi05/vla_occluder_rollout.sh` | 4 |
| 7 | A committed 5-episode run folder's `records.jsonl` + one video, or a report of where it broke | 5 |
| 8 | `agent-memory/tool_vla_pi05_port.md` and `status_current.md` updated | — |

---

## 7. What NOT to do

- **Do not chase success rate.** The checkpoint is finetuned on RoboPRO *real* data
  (`roboreal_lerobot`), so there is a sim2real gap on top of an out-of-distribution occluder scene.
  Near-floor success is the expected outcome and is **not** a bug in this script. Report the number,
  do not tune against it.
- **Do not touch placement.** Standing user directive (2026-07-28): `place_actor` / landing-search /
  object-ejection failures are reported and left alone.
- **Do not add the clearance metric to the loop.** Joining rollout outcomes to precomputed eps*
  buckets is the clean next experiment, but it is a *separate* offline join and explicitly out of
  scope here.
- **Do not port other VLAs.** Every other policy needs a finetuned checkpoint for this embodiment
  that does not exist. pi05 is the only one with weights on disk.
- **Do not add a plotting/analysis layer.** `records.jsonl` + videos is the deliverable; figures come
  later once there is something worth plotting.

---

## 8. Known risks

| Risk | Signal | Response |
|---|---|---|
| pi05 never inferred | Stage 1 fails | Stop, report the traceback — do not build on it |
| VRAM: 7 GB jax + curobo + SAPIEN on 16 GB | OOM or CUDA illegal-memory-access | Confirm `PREALLOCATE=false` is holding (Stage 4); then 2c's planner gate. Single-GPU is a supported mode, not a workaround — `eval_double_env.sh` has always accepted `0:0`. |
| torch/jax co-residency instability | 169k-identical-error spin (seen before) | Add a hard failure ceiling to the seed loop — never `while` without one |
| Per-episode VRAM creep | Stage 5.5 shows climbing reserved | Report it; do not fix blind |
| `step_lim=600` too short for the policy | Every episode ends at the cap | Report — it is a finding, not a bug to tune away |
