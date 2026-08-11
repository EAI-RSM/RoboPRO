---
name: feedback_script_conventions
description: "Checklist every analysis script must satisfy: timestamped run folder, per-component timings.json, legible figures under scripts/validation/results/<topic>/"
metadata:
  type: feedback
---

Three standing requirements for every script that produces artifacts. Apply by default, unasked.

**1. Timestamped run folder.** Each run writes to `<out-dir>/<YYYYmmdd-HHMMSS>/`, never a flat
shared dir. Build it once at the top of `run()` with
`Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")`, print it, route everything
(PNGs, mp4 `save_path`, caches) underneath. `--out-dir` stays the stable parent.
*Why:* re-running a seed would overwrite the previous run, and rollout outcomes for the SAME seed
are not reproducible — two runs are genuinely different data points that must both survive.

**2. Per-component timing, saved with the run.** A small `Timings` helper with a `section(name)`
context manager: wrap each logical phase (scene setup, grid build, solver build, each heavy loop,
reporting), print live, and write `timings.json` (component → seconds + total) into the run dir.
Heavy per-item loops must also print live per-iteration progress so a long silent phase never
looks hung. *Why:* these are long GPU runs; without a breakdown you cannot tell where time goes
(the warm-start loop was silently ~half the labelling cost). First implemented in
`clearance_metric_3d.py`.

**3. Results go to `scripts/validation/results/<topic>/`, and must be legible.** Figures AND data
outputs (records.jsonl, csv/json summaries, videos). Default `--out-dir` should point at
`../scripts/validation/results/<topic>` (relative to robotwin_root after the script's os.chdir).
*Why:* the user interprets results themselves, keeps every run in that one folder, and could not
read a cramped multi-panel chart rendered inline. That dir is gitignored, so outputs stay local
and reproducible from the generating script.
*How:* few panels, large figsize, big fonts, on-figure annotations so the takeaway needs no zoom.
`/tmp` is only for throwaway smoke tests, never real results.

**4. Save initialized-scene views for no-rollout scene generators.** If a script builds a live
scene but deliberately executes no expert or policy, save static camera PNGs in the timestamped
run so the user can verify what was measured. Do not create a one-frame/repeated-frame MP4 and
call it a rollout video; label the artifact as an initialized-scene image.
