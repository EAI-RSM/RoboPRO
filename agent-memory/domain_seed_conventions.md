---
name: domain_seed_conventions
description: "How seeds are chosen for the finetune-then-evaluate workflow: the three disjoint numeric bands, the expert-feasibility gate, and the fact that the frozen eval set is not wired into any eval path"
metadata:
  type: project
---

Established 2026-08-10 by reading the scripts. The user was told "RoboPRO has a specific way seeds
are chosen" for train-a-VLA-then-evaluate; this is what that actually is in this tree.

**Three DISJOINT numeric bands. Train/eval separation is structural, not a random split of one
pool.**

| band | who writes it | rule |
|---|---|---|
| `0, 1, 2, …` | `collect_data.py` | training / demo collection |
| `40000+` | `precollect_eval_seeds.py` | frozen evaluation set |
| `100000 * (1 + seed)` | `eval_policy.py`, `eval_policy_client.py` | upstream RoboTwin runtime |

**1. Training seeds ascend from 0 and land in `seed.txt`.** `collect_data.py` is TWO passes.
Pass 1 (`use_seed: false`) starts `epid = 0`, +1 per attempt, keeps a seed only if
`plan_success and check_success()`, until it has `episode_num` successes — then writes the accepted
list to `<save_path>/seed.txt`. Pass 2 (`use_seed: true`) re-runs EXACTLY that list to write HDF5.
So the training set is the expert-feasible SUBSET of `[0, N)` — non-contiguous, and `seed.txt` is
the authoritative record, not the range. Resumable: an existing `seed.txt` restarts from
`max(seed_list) + 1`.

**2. Eval seeds start at 40000 and are frozen on disk.** `precollect_eval_seeds.py` — the
RoboPRO-specific piece, and the answer to "the standard". `SEED_BASE = 40000`, chosen in its own
docstring "to guarantee no collision with training seeds (0..~30)". Same expert-feasibility accept
rule, but writes NO artifacts (forces `collect_data=False`, `save_data=False`, `render_freq=0`,
scratch tempdir for anything incidental). Output `benchmark/eval_seeds/<task>/<config>.txt`, written
atomically (tmp + `os.replace`). **Target: 20 seeds for `*_clean` configs, 2 for everything else**
(the `d6`..`d15` clutter configs). Budgets `MAX_TRIES_CLEAN=500` / `MAX_TRIES_CLUTTER=200`, `[WARN]`
on shortfall. **Idempotent — it SKIPS a (task, config) already at target rather than regenerating**,
which is what keeps the eval set from drifting between checkpoints.

Realized on disk 2026-08-10: 80 tasks, 3158 seeds, range **40000–40297**. Both scenes of interest
are covered — `put_cup_on_coaster/bench_demo_study_clean.txt` 20 seeds (40000–40024, gaps),
`put_mouse_on_pad/bench_demo_office_clean.txt` 20 (40001–40027). 792 clutter configs have exactly 2;
three fell short at 1; `move_pen_to_box/bench_demo_study_clean.txt` reached only 11.

**The unifying principle: the expert-feasibility gate.** Every seed in BOTH bands is one the
scripted curobo expert solved. Seeds the expert fails are discarded and never enter either set —
so evaluation never asks the policy for something the demonstrator could not do. Same gate as
[[repo_task_assets]]'s "eval runs the expert FIRST"; this is where the gate gets applied *ahead of
time* instead of inline.

**FOUR TRAPS — check these before claiming the standard is being followed.**

1. **Nothing reads `benchmark/eval_seeds/`.** `precollect_eval_seeds.py` is the ONLY file in the
   tree that mentions the path (`grep -rn eval_seeds` → 3 hits, all in that one script). The eval
   drivers ignore it and re-derive seeds at runtime from `st_seed = 100000 * (1 + seed)`, re-running
   the expert check inline. The frozen set is an artifact, not a wired-in input. Honoring the
   standard requires making the eval driver read the file.
2. **No `seed.txt` exists anywhere on disk** — no training collection has ever been run here,
   consistent with there being zero checkpoints ([[tool_vla_pi05_port]]).
3. **2 eval seeds per clutter config is a smoke gate, not a measurement.** The 3000-rollout VLA
   association campaign side-stepped the scheme entirely: `seed_start: 3000, num_seeds: 50`, a band
   colliding with neither 0.. nor 40000+, and not the precollected set either.
4. **The olive-oil ring scene is outside the scheme.** `analyze_occluder_visibility.py` draws its
   own seeds from `--seed-start`/`--num-seeds` and gates on stability / occluder tilt / pad-blocked
   ([[domain_scene]]), NOT on expert feasibility. That is a fourth, ad-hoc convention — reconcile it
   before any ring finetune/eval is claimed to follow the RoboPRO standard.
