# Deterministic rebuild-and-replay

Reaching a mid-episode state *exactly, more than once* — so two policy branches
can be compared from an identical starting point, or a labelled episode can be
re-derived later.

## The result

| operation | reproduces the state? | reproduces the outcome? |
| --- | --- | --- |
| `pack()` / `unpack()` snapshot-restore | **yes, bit-exact** | **no — 38.9% of labels flip** |
| rebuild + replay actions from step 0 | **yes, bit-exact** | **yes — 0% flip** |
| ...across separate processes | **yes, bit-exact** | yes |
| ...with the policy in the loop | **yes, bit-exact** | yes |

The contract:

```
deterministic given      (scene seed, action rows from step 0)
NOT deterministic given  (mid-episode observable state, action)
```

## Why restore fails, and why it is dangerous

`PhysxSystem.pack()` saves every pose, quaternion, joint angle and joint velocity.
Restoring reproduces all of them **bit-exactly** — a 336-component state
fingerprint comes back at `max|diff| = 0.000e+00`. Nothing looks wrong.

What it does not save is the PhysX contact / warm-start cache: the persistent
contact manifolds and accumulated solver impulses that warm-start each step. Those
are a function of *how the state was reached*. Restore the same state from a
different history and the solver converges differently, the arm tracks slightly
differently, and over 150–250 remaining steps that amplifies.

Measured on `put_milktea_next_to_laptop`, 18 states × 4 identical replays:

```
7 / 18 states flipped their success label   (38.9%)
divergence up to 0.67 rad on arm joints
flips are systematically 1 -> 0: restored branches fail MORE than natural ones
```

That last point matters most. It is not symmetric noise — it is a **dynamics
shift**, so a dataset built on restore would carry a success rate below the
policy's true one.

The determinism itself is intact throughout: repeats sharing the same pre-restore
history come out byte-identical to each other. The simulator is deterministic;
`pack()` is simply a lossy capture of what determines the future.

## The protocol

```
spine        one plain policy rollout per scene; store its commanded action rows
branch point close_env(clear_cache=True) + setup_demo(seed) + replay rows[:T]
             -- no policy, no observations consumed, bit-exact
branch       run the policy closed-loop from T to terminal
```

`close_env(clear_cache=True)` + `setup_demo()` **in the same process** resets the
solver cache. A fresh process is not required.

Prefix replay needs no policy, so a branch *point* is reproducible regardless of
anything in the inference path — and since closed-loop replay also measured
bit-exact, a whole branch is reproducible from `(seed, prefix rows, noise key)`.

## Recording the flow noise

pi05 draws its flow-matching noise from `Policy`'s internal jax key, so a rollout
cannot be reproduced or paired with the noise that produced it. Two env vars make
it explicit (both inert when unset, so every existing caller is unaffected):

| var | effect |
| --- | --- |
| `PI05_NOISE_SEED` | episode key `K` draws from `default_rng(seed + crc32(K))` |
| `PI05_NOISE_DIR` | writes `<dir>/<K>_noise.npz` with the noise **and** the action chunk it produced |

`crc32`, not `hash()` — `hash()` is salted per process and would not reproduce.
Callers select the stream with `model.set_noise_episode(key)`.

## Language: one fixed instruction per task (new default)

`_maybe_apply_language_perturbation` now returns **`instruction_bank.json` entry 0**
for the task — one deterministic instruction — instead of sampling a paraphrase.
`ROBOPRO_LANG_PERTURB=1` restores the old sampling.

Why the old behaviour was a trap:

* The sample was drawn from the **scene RNG**, so the instruction was a
  deterministic function of the scene seed. Verified across two independent
  collections months apart: **38/38 scenes got identical instructions**. So
  instruction and scene were *perfectly confounded* — no per-scene result could
  separate scene difficulty from prompt wording, and a chi-square across
  instructions was really measuring scene difficulty.
* A single task drew **8–17 distinct phrasings** across one 25-episode eval.
  Anything trained on one phrasing per task — a Q-function, say — was then scored
  under phrasings it had never seen.
* Setting `enabled: false` did **not** fix this. It returned `None` and let the
  caller fall back to `np.random.choice(generated_descriptions)`: still random,
  just from a different pool. Returning `bank[0]` makes "no perturbation" mean
  what it says, since every caller prefers this return value.

All 14 benchmark tasks have bank entries, so this is deterministic for each.

## Files

| path | role |
| --- | --- |
| `script/replay_utils.py` | `rebuild` / `replay_actions` / `state_fingerprint` / `build_args` |
| `script/verify_replay_determinism.py` | the harness that measures all of the above |
| `script/collect_qeval_pairs.py` | Q-eval pair collection built on the protocol |
| `qeval_collect.sh` | launcher (model server + sim client) |
| `policy/pi05/pi_model.py` | `set_noise_episode` / noise recording |
| `script/collect_rollout_client.py` | opts existing collection into noise recording |

## Reproducing the measurements

```bash
# core claim: rebuilds are bit-exact
python script/verify_replay_determinism.py --phase rebuild \
    --task_name put_milktea_next_to_laptop --task_config bench_demo_office_d14 \
    --seed 0 --actions <episode>_noise.npz --repeats 3

# survives a process boundary
python script/verify_replay_determinism.py --phase crossproc ... --out /tmp/a.npz
python script/verify_replay_determinism.py --phase crossproc ... --out /tmp/b.npz
python script/verify_replay_determinism.py --compare /tmp/a.npz /tmp/b.npz

# with the policy in the loop (needs a running model server)
python script/verify_replay_determinism.py --phase closedloop --port <PORT> ...

# the control: what restore does instead (needs the `lookahead` package)
python script/verify_replay_determinism.py --phase restore ... --T 100 --repeats 4
```

## Caveats

* Two *natural* episodes reaching the same observable state by different routes
  would also carry different solver caches, so the outcome is a function of more
  than `(images, proprio, action)`. Exact state collisions do not occur in
  continuous space, so this never shows up as contradictory labels — it shows up
  as the target function being rougher than its observable inputs explain. The
  38.9% is an upper bound: restore constructs a maximally mismatched cache.
* Sensitivity falls sharply as an episode progresses — divergence at T=150 was
  `1e-4` where T=50 gave `0.5`. Later branch points are far more determined.
* Success criteria live in each task's `check_success`, and changing one silently
  redefines every label. Record the collection's git SHA alongside the data.
