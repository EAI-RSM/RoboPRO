# Failure annotation — redesign brainstorm

**Status: brainstorm canvas, nothing final.** Target design (per direction):

1. **Per-timestep** annotation with three channels:
   - **WHAT happened** — grasp failure, placement failure, collision, … (the symptom,
     derived from sim).
   - **WHY** — mislocalized object / size misdetection / undetected object / … (the cause,
     *known by construction* because we are the ones who shifted / resized / hid it).
   - **HOW MUCH** — the exact numbers of the change and its effects every frame (error
     vectors, distances, clearances, contact impulses, task progress). Categorical labels
     train a classifier; these continuous signals are what let the data train a **reward
     or value model**.
2. **Episode-level quality** — `good | suboptimal | bad`.

Lineage: `negative_data_brainstorming.md` (desync idea), `.claude/negative_data_collection.md`
(contract), code in `benchmark/bench_envs/targeted/{labels.py,runtime.py}` and
`script/bench_script/run_targeted_episode.py`.

---

## 1. Why the current labels fall short (motivation)

Today: one per-episode `outcome` ∈ {clean_success, success_with_collision, empty_grasp,
wrong_place_pose, collision_with_object, plan_aborted}, chosen by a precedence ladder
(`derive_outcome`). Three problems this redesign fixes:

- **Lossy + cause-blind.** The ladder gates on `plan_success` first (`labels.py:67`): our
  `clean/seed0/shift_object` episode mislocalized the mouse 4.5 cm, *missed the grasp*,
  **and** aborted the transport plan — we recorded only `plan_aborted`. Both the grasp
  failure and the cause vanished.
- **One verdict for a whole trajectory.** No way to say *when* it went wrong or point a
  monitor at the failing frames.
- **No notion of trajectory quality** for filtering training data.

---

## 2. The reframe that makes WHY trustworthy

A perturbation **is** an injected fault in the executor's world model — and because *we*
inject it, the WHY is ground truth, not inference:

| Intervention | WHY (cause), known by construction |
|---|---|
| `shift_object` | `object_mislocalized` (target), axis ∈ {depth,lateral,vertical,mixed} |
| `shift_target` | `destination_mislocalized` |
| `shift_obstacle` | `obstacle_mislocalized` (believed off-path, truly on-path) |
| `hide_obstacle` | `object_undetected` (missed detection) |
| `resize_*` (future) | `size_misdetected` |
| `swap_*` (future) | `object_misidentified` |

The baseline twin injects nothing → WHY = `none` for every frame, which is what lets us
attribute the perturbed twin's WHAT to the cause (§5).

---

## 3. Per-timestep annotation — two aligned channels

For each saved frame `t` (saved-frame index space; the deferred-shift boundary is masked
per `negative_data_collection.md` principle 7), emit:

```yaml
t:
  phase:   reach | grasp | lift | transport | place | retreat   # from the scripted sub-actions
  what:    ok | grasp_failure | placement_failure | collision | planning_failure
  what_evidence:                       # only when what != ok
    collision: {bodies: [...], impulse, link}        # for `collision`
    planning_failure: {status: "INVALID_START_STATE_WORLD_COLLISION", stage: transport}
  why:     none | object_mislocalized | destination_mislocalized | obstacle_mislocalized
           | object_undetected | size_misdetected | object_misidentified
  metrics:                             # HOW MUCH — exact numbers, for reward/value models
    active_error_cm: 4.50              # |believed − true| of the faulted entity at t (0 pre-inject)
    active_error_vector_world: [dx,dy,dz]
    believed_pose / true_pose: {p:[...], q:[...]}     # faulted entity (privileged)
    gripper_to_target_cm: ...          # progress toward the grasp
    object_to_destination_cm: ...      # progress toward the place (== live placement error)
    object_height_above_table_cm: ...  # grasp-secured proxy
    min_robot_obstacle_clearance_cm: ...   # safety margin
    contact_impulse_N: ...             # collision severity at t (0 if no contact)
    progress: ...                      # 1 − dist_to_goal / dist_to_goal(0)  ∈ [0,1]
    twin_divergence_cm: ...            # ‖state_perturbed(t) − state_baseline(t)‖ (counterfactual)
```

`metrics` are **privileged sim measurements** — a deployed reward/value model never sees
them; we use them to *construct the training target* it learns to predict from the
policy's own observation (images + proprio, already in the HDF5).

Plus a **constant** `why_params` for the episode (the injected fault, recorded once):

```yaml
why_params:
  entity_role: target | destination | obstacle
  entity_name: "047_mouse"
  axis_class: lateral
  error_vector_world: [dx,dy,dz]        # believed − true
  error_camera_frame: {depth_cm, lateral_cm, vertical_cm}
  magnitude_cm: 4.50
  active_from_frame: <shift_frame_idx>  # WHY = none before this, the cause after it
```

### Channel semantics

- **WHAT** is per-frame and *derived from sim*. Most frames are `ok`. Adverse states are
  spans, not instants: `collision` over the contact span (already in `contact_log`, keyed
  by frame), `planning_failure` from the abort frame to the end, `grasp_failure` from the
  failed grasp onward, `placement_failure` over the final placement frames. WHAT can
  co-occur (collision *and* planning_failure) — store the per-frame value as a small set,
  or a primary + flags (open Q1).
- **WHY** is a step function: `none` until `active_from_frame`, then the injected cause to
  the end. For `object_undetected` it is active the whole episode (the obstacle was never
  in the planner's world). It is **read off the intervention**, never inferred.

### Worked timeline — `clean/seed0/shift_object` (4.5 cm lateral, deferred)

| frames | phase | what | why |
|---|---|---|---|
| 0 .. grasp-lock | reach | `ok` | `none` (object not yet shifted; planner correct) |
| grasp-lock (inject) | grasp | `ok` | `object_mislocalized` (4.5 cm lateral) — active from here |
| close → lift | grasp/lift | `grasp_failure` | `object_mislocalized` |
| ~frame 83 → end | transport | `planning_failure` (INVALID_START_STATE_WORLD_COLLISION) | `object_mislocalized` |

One trajectory, two readable channels, the cause attached to the symptom — vs. the old
single `plan_aborted`.

---

## 4. Episode quality — `good | suboptimal | bad`

A rollup of the WHAT channel + task success, for filtering training data:

- **good** — task succeeded and every frame is `ok` (clean success, or fault fully
  absorbed with no adverse frame). Usable as a positive demo.
- **suboptimal** — task succeeded **but** some adverse WHAT occurred (e.g. collision then
  recovered, a near-miss, inefficiency). Use with care.
- **bad** — task failed: any `grasp_failure` / `placement_failure` / `planning_failure`
  frame, or `task_success == false`. The negative.

(Maps onto the absorbed / degraded / caused_failure idea, but stated as the quality grade
you asked for. Open Q4: where exactly is the good/suboptimal line for collision-but-success?)

---

## 5. Causal attribution (per twin-pair)

Because the baseline twin has WHY = `none` everywhere, the perturbed twin's adverse WHAT
that the baseline lacks is **attributable to the injected cause**:

```
fault_attributable_what = {adverse WHAT in perturbed} − {adverse WHAT in baseline}
```

Derived at the pair level (the collector owns both records). This is the honest
"the mislocalization *caused* the grasp failure + plan abort," and it powers the
magnitude → quality sensitivity curves per (task, axis_class).

**Quantitative counterfactual — the reward/value gold.** The twins share the nominal plan,
so their per-frame `metrics` gap is a clean, low-variance learning signal:

```
advantage(t)     = progress_baseline(t) − progress_perturbed(t)   # what the fault cost, per step
return_to_go(t)  = discounted future reward from t                # value target
success          ∈ {0,1}                                          # sparse terminal target
```

A reward/value model can then be trained to predict these from observations alone, with the
fault magnitude (`why_params.magnitude_cm`) as a known dose for calibration.

---

## 6. Pipeline implications (the real work)

Per-timestep annotation means logging a **time series**, not just episode-end signals.

1. **Inject** (`runtime.py`) — already applies shift + override; *add* recording of
   `active_from_frame` and the believed/true poses so `why_params` is exact.
2. **Observe** (during `play_once`) — the gap to close:
   - **per-frame WHAT signals**: contacts are already per-frame (`contact_log`); add a
     per-frame **grasp-secured** check (object lifted / between fingers — the current
     `object_moved_cm < 3` heuristic is end-of-episode and mislabels: case2 seed2 moved
     3.96 cm yet wasn't grasped) and capture the **plan-abort frame + `MotionGenStatus` +
     sub-action** (today only a `plan_success` bool survives to the record; the *why* is
     lost to `run.log`).
   - **frame-index integrity**: `shift_frame_idx` came out `0` for a deferred shift —
     verify `FRAME_IDX` is the saved-frame counter we think it is, since the whole
     per-timestep alignment rides on it.
3. **Derive per-episode** (`labels.annotate`, pure) — build the `what[]` / `why[]` arrays
   from the logged series + `why_params`; roll up `quality`.
4. **Derive per-pair** — `fault_attributable_what`, sensitivity rows.
5. **Annotate** (`hdf5_annot.py`) — store the per-frame arrays (aligned to the saved
   frames), `why_params`, `quality`, and keep raw signals as evidence.

---

## 7. Stored fields

Keep all provenance (seed, config, executor, pair_id/role, sampler, camera_axes,
achieved shift, instruction). Replace the single `outcome` with:

- `what[]`, `why[]`, `metrics[]` — per-frame arrays (length = saved frames).
- `why_params` — the injected fault (constant).
- `quality` — `good|suboptimal|bad`.
- `reward_value` — `success`, `return_to_go[]`, optional `shaped_reward[]` (+ components),
  `advantage[]` (per-pair). Targets for a reward/value model.
- `episode_what_summary` — the set of adverse WHAT that occurred (for flat filtering).
- `fault_attributable_what` — per-pair.
- raw `signals` + `contact_log` retained as evidence.

Flat keys consumers filter on: `why` (modality), `entity_role`, `axis_class`,
`magnitude_cm`, `quality`, `task_success`, one bool per adverse WHAT.

---

## 8. Open questions (your call)

1. **WHAT per frame: a set, or primary + flags?** Co-occurrence (collision + planning_failure)
   is real. Set = lossless; primary = simpler CSV.
2. **WHAT vocabulary** — is {grasp_failure, placement_failure, collision, planning_failure}
   complete, or add `near_miss` / `drop` (grasped then lost) / `wrong_object`?
3. **Granularity of `why`** — keep axis (depth/lateral/…) inside `why_params`, or split
   `object_mislocalized_depth` vs `_lateral` into the `why` value itself?
4. **good/suboptimal/bad cut points** — is collision-but-success `suboptimal`? Is a
   recovered grasp `suboptimal` or `good`? Is `plan_aborted`-but-then-succeeds possible?
5. **Storage of per-frame arrays** — dense arrays (ML-friendly) vs. run-length segments
   (human-friendly). Both?
6. **Quality without a fault** — does the baseline twin get `good` automatically, or do we
   still grade it (a baseline can be suboptimal too)?
7. **Reward target: precompute or raw?** Store only raw `metrics[]` and let the consumer
   shape the reward, or also precompute a `shaped_reward[]` / `return_to_go[]`? And what is
   `success`/return — sparse terminal only, or progress-shaped (and is "progress" distance
   to the live destination, or task-defined milestones)?
