# Collision-Aware pi05: Clutter Perception + Critic-Driven Policy Optimization

**Status:** design doc, 2026-06-10.
**This branch (`collison_free_data_gen`):** the critic and the policy-optimization stage.
**Sibling branch:** clutter understanding (Depth Anything 3 cross-attention + auxiliary
distance heads) — summarized in §1, spec in `auxiliary_collision_heads_rgb_only_spec.md`.

---

## 0. The method at a glance

Two stages, two branches, one converging design:

```
Stage 1 (perception branch)         Stage 2 (this branch)
─────────────────────────           ────────────────────────────────
pi05 + DA3 features cross-          Differentiable critic
attended into the action     ──►    f(s, a_1:H) → ℝ  (ground-truth
head + aux head: RGB →              FK × SDF, validated in sim)
EE/link clearance                          │
                                           ▼
   BC training with clutter-        Fine-tune the flow head so its
   aware representations            SAMPLES are low-cost under f
                                    (adjoint matching, §4)
                                           │
                                           ▼
                                    Deployment: aux head becomes the
                                    learned f̂ (no ground-truth scene
                                    needed at test time)
```

The critic assesses a robot **configuration** per waypoint (every collision sphere of
every link, against the scene SDF), sums over the chunk into one scalar, and is exactly
differentiable w.r.t. the action chunk. Its gradient is the perturbation signal — at
inference time today (guidance, the validation harness), at training time next (the
actual goal).

---

## 1. Stage 1 — clutter understanding (perception branch)

- **DA3 cross-attention:** Depth Anything 3 features are cross-attended with the action
  tokens inside the pi05 action expert, giving the action head direct access to metric
  scene geometry rather than relying on the VLM backbone to encode it.
- **Auxiliary distance head:** trained to predict per-link distance (and direction) to
  nearby collisions, supervised from the simulator's ground-truth clearances (the same
  FK × SDF machinery as the critic — one source of truth for "distance to collision").
  Verified to date: top-3 distances for the end effector. Target spec for the per-link
  extension below (§1.1).
- Output of this stage: a BC policy that *represents* clutter. It does not yet *optimize*
  against it — BC imitates the data, including the data's contacts.

The aux head matters for Stage 2 twice over: (a) it shapes representations so the
fine-tuning gradient has somewhere useful to flow, and (b) extended to condition on the
action chunk, it **is** the learned critic f̂ — the deployable stand-in for f when there
is no ground-truth scene (real robot). Same function signature, same semantics.

### 1.1 Aux-head target spec: per-link top-3 (distance + away-direction)

**Why per-link, empirically:** in the 2026-06-10 paired study the contacting links in
the worst episode (kettle grind, seed 300011) were `fr_link3/4/5/6/8` — almost none of
the damage went through the end effector. An EE-only head is blind to the majority of
real contacts; the critic's worst offenders are routinely mid-chain links (`fr_link3`,
the upper arm, is also the hardest for the optimizer to move — early warning matters
most exactly there).

**Why directions:** the gradient of clearance w.r.t. a link position is exactly the
unit away-vector û from the nearest obstacle (closed form: û = (x−p)/‖x−p‖ to the
nearest surface point), and the critic's action gradient factors as Jᵀû per link with
J the known FK Jacobian. A head that predicts (d, û) per link therefore supplies a
first-order local model of the clearance field, `d(x+δ) ≈ d + ûᵀδ`, which is enough to
run **inference-time guidance from RGB alone** — no ground-truth scene — before any
action-conditioned critic exists. It also makes the eventual f̂ gradient-correct, not
just value-correct: a distance-only head can fit values perfectly while its implicit
spatial gradient is garbage.

**Why top-3:** a nearest-only prediction flips discontinuously across the medial axis
and reports "clear on both sides" in exactly the bracketing failure we observed (arm
trapped between two kettles, repulsion gradients cancelling). Three distance-sorted
slots represent "close on both sides"; sorting by distance resolves target matching
for free. Top-3 is already verified for the EE — same target shape, replicated.

**Output spec** (≈ 7 links/arm × 2 arms = 14 link tokens — emit one token per link so
the action expert's cross-attention can attend to links spatially, not a flat vector):

```
per link ℓ, for the 3 nearest DISTINCT obstacles i (distinct per_scene_id —
two samples of the same kettle must not fill all three slots):
    d_ℓ,i        clearance from the link sphere surface, clipped at d_max = 0.3 m
    û_ℓ,i ∈ S²   unit away-vector (obstacle → link), robot base frame
plus per link:   contact logit  P(engine contact on link ℓ within next W steps)
                 — supervised by RECORDED contacts, not by d_model < margin: this is
                 the learned calibration (§4.5 gating) and the deployment-time GATE;
                 the distance regressions supply direction/standoff, the logit
                 decides whether to act
later:           a 4th slot for the HELD object once grasped-object geometry
                 is attached to the EE frame (the ~20 % blind spot)
```

**Supervision** — generate labels with the critic pipeline verbatim
(`differentiable_proximity.py`: FK + CuRobo spheres + scene samples): per link, min
over the link's spheres; û from the nearest surface sample via the KD-tree closed form
(exact — no voxel smoothing in labels). Supervise the *sphere-model* clearance, not
true mesh clearance: it is what the analytic critic computes, so the learned head is a
drop-in replacement with the same calibration (the 2–2.5 cm margin was tuned for this
quantity), and it is conservative by construction.

**Losses:** Huber on clipped d with distance-decaying weights (w = exp(−d/5 cm)) so
accuracy concentrates inside the margin shell where the hinge has gradient; cosine
loss on û gated to d < 15 cm (far directions are noise and never used); BCE on the
contact logit against recorded engine contacts (this head IS the calibration —
regression near zero is where the geometric model itself under-reads up to 1.5 cm,
so the gate must be learned from contacts, not thresholded from distances; monitor
AUC against engine contacts per checkpoint).

**Deliberately NOT in this stage:** a fully queryable neural SDF (image → distance at
arbitrary 3D points) — elegant endgame (the critic would transfer unchanged by
FK-querying the predicted field) but a much harder learning problem; the per-link head
covers the configurations the policy actually visits. And no action-chunk conditioning
yet — that is the Q̂ distillation step (§4.5), second stage on the same trunk, so
perception and credit assignment are never debugged simultaneously.

**The staircase:** EE top-3 (verified) → per-link top-3 + directions (this step;
enables RGB-only guidance) → chunk-conditioned Q̂ on the same trunk (§4.5–4.6; enables
training-time critic without sim) → optional neural SDF only if per-link directions
prove too coarse. Every stage is supervised by the same ground-truth pipeline, so
calibration carries forward instead of restarting.

## 2. Stage 2 — the critic f(s, a_1:H)

Implemented and validated in `customized_robotwin/script/differentiable_proximity.py`:

```
f(s, a_1:H) = Σ_{k=1..H} Σ_{s∈spheres} h²·(1 + γ·h/margin) / 2β
              with h = relu(r_s + margin − SDF(c_s(FK(a_k))))
```

- **FK** extracted numerically from the live articulation (0.001–0.043 mm vs SAPIEN),
  batched torch, exact gradients.
- **Robot body** = CuRobo sphere decomposition (~130 spheres/arm).
- **Scene** = per-episode signed-distance voxel grid (1 cm), built from ground-truth
  meshes at their current poses; trilinear interpolation → differentiable queries;
  rebuilt when an obstacle moves > 3 cm.
- **Hinge** is exactly zero outside the margin shell (a softplus tail provably biases
  the arms away from harmless obstacles); the cubic depth weighting (γ) makes the
  gradient ~2.5× steeper at contact and ~4× under penetration while leaving the shell
  boundary untouched — "bulge more at small distances, barely alter at large ones."

**Validation status (paired-seed sim studies, 2026-06-10):**

| Property | Evidence |
|---|---|
| Predictive | model-negative clearances predicted 35–49 steps before the real hit; zero false alarms; clean episodes → zero predictions |
| Calibrated | predicted vs realized depth within ~0.5 cm; engine-forceful contacts occur at model clearance −2.5…+1.5 cm (sets the margin) |
| Differentiable | whole chain is torch autodiff; C¹ at the hinge boundary |
| Actionable | optimizing chunks with it cleared −2.2 cm planned penetrations to +2.7 cm in ~370 ms |

**Known gaps:** ~20 % of forceful contacts are the *held* object hitting clutter
(invisible to arm spheres — attach grasped-object geometry to the EE frame, open item);
gripper columns untouched; unsigned-grid fallback weakens gradients under deep
penetration when meshes aren't watertight.

## 3. Why inference-time guidance is brittle — and why that doesn't doom the critic

Guidance (optimize each arriving chunk with Adam, frozen policy) works often but fails
in characteristic ways, all observed today:

1. **It cannot edit the present.** When an episode *starts* in contact (ep3,
   pick_bottle_from_fridge: −0.07 cm measured at t=0, first chunk predicted −2.5 cm),
   the chunk-start continuity ramp correctly forbids a step jump, and the optimizer
   recovered 1 mm. Guidance halves the damage (contact 116 vs 250 steps) but can't
   prevent it.
2. **Authority budget.** steps × lr ≈ 0.16 rad of total correction; deep escapes at
   upper-arm links (short lever arm) need more.
3. **Medial-axis conflicts.** Bracketed between two kettles, the repulsion gradients
   cancel; per-instance optimization has no way to "choose a side" early.
4. **Zero gradient outside the shell.** By design f cannot reshape a trajectory until
   it is already close — fine for a safety filter, useless for preference.
5. **Closed-loop divergence.** The policy reacts to its own corrected history; the
   correction fights the policy's intent every chunk, forever.

None of these is an indictment of the *signal* — every failure above happens **after**
f correctly predicted the collision with 35–50 steps of lead. They are failures of
*per-instance, test-time, budget-limited* optimization. Training-time optimization is
categorically different:

- The gradient accumulates over **epochs and thousands of scenes**, not 32 Adam steps.
- The policy learns to **not enter** the bad states at all — including the t=0
  emergencies, because the *previous* chunk's distribution is also being shaped.
- Medial-axis ambiguities average out over scene variations, and the perception stack
  (Stage 1) lets the policy resolve them from the image *before* getting trapped.
- There is no inference-time fight: the corrected behavior **is** the policy.

Guidance therefore remains exactly what it was built to be — the validation harness for
f and a deployable safety net — while the policy improvement happens in training.

## 4. Using the critic to optimize the policy

### 4.1 The objective

pi05's action head is a flow-matching sampler: given state s (images, language,
proprioception) it transports noise x₀ ~ N(0,I) along a learned velocity field
v_θ(x_τ, τ, s) to an action chunk x₁ = a_1:H. We want the *fine-tuned sampler* to draw
from the reward-tilted distribution

```
π*(a|s) ∝ π_base(a|s) · exp( (η·Q̂_task(s, a) − f(s, a)) / λ )
```

i.e. keep the BC behavior except (a) suppress probability mass in the margin shell and
inside obstacles, and (b) **push mass toward chunks that advance the task** — the
task-progress critic Q̂_task (§4.6) is the counterweight that stops "safe" from
collapsing into "stalled". λ trades the whole tilt against staying near the base
policy; η balances task progress against collision cost inside the tilt.
This is the standard "RL as sampling from a tilted posterior" objective, and it is the
problem **adjoint matching** solves for flow/diffusion models.

Why two critics instead of relying on the BC anchor alone: the anchor is *passive* — it
pulls toward the base distribution everywhere, including toward the base policy's
collisions, and it does not distinguish "deviated safely and finished the task" from
"deviated safely and froze". We have hit the freeze mode twice already (the absolute-
smoothing regression: success 9/10 → 5/10 while contact dropped 100×; the guided stall
episodes in pick_can_from_basket). A collision-only tilt has the same gradient
structure as that bug — zero cost for not moving — so the objective must contain an
explicit term that is maximized by *completing* the task.

### 4.2 Route A (primary): adjoint matching with the known differentiable cost

The "Q-adjoint matching" idea, made concrete. The key simplification for us: adjoint
matching normally needs a learned reward/Q model and differentiates through it. **Our
terminal cost f is already known, differentiable, and validated** — so the first
incarnation needs no Q-learning at all (see §4.5 for when a real Q enters).

Per gradient step (Domingo-Enrich et al., *Adjoint Matching*, arXiv:2409.08861, adapted
to pi05):

```
1. Sample a state s from the fine-tuning state buffer (§5).
2. Roll the MEMORYLESS noised sampler of the current policy θ:
     x_{τ+Δ} = x_τ + v_θ(x_τ, τ, s)Δ + σ(τ)·ε,   τ: 0 → 1
   storing the trajectory {x_τ}. (Memoryless schedule = the noise injection that
   makes the fine-tuning objective unbiased; without it you optimize a biased
   functional of the sampler.)
3. Terminal adjoint:  ã(1) = ∇_{x₁} [ f(s, x₁) − η·Q̂_task(s, x₁) ]
                                                     ← one autodiff call through
                                                       FK + SDF (collision critic)
                                                       + one through the task critic
   (Phase 2 may start with η = 0 — collision-only — and switch the task term on
    once Q̂_task is fitted; the gate in §5.1 protects success either way.)
4. Lean adjoint ODE backward along the FROZEN trajectory:
     ã(τ−Δ) = ã(τ) + ã(τ)ᵀ ∂_x v_base(x_τ, τ, s) Δ
   (vector–Jacobian products through the BASE velocity field; no second-order terms)
5. Regression loss (the actual parameter update):
     L_AM(θ) = Σ_τ ‖ v_θ(x_τ, τ, s) − v_base(x_τ, τ, s) + (σ(τ)²/2λ)·ã(τ) ‖²
6. Total loss:  L = L_AM + λ_BC · L_FM(expert data)
   where L_FM is pi05's ordinary flow-matching loss on the BC dataset — the anchor
   that protects task success (see §5.4).
```

pi05-specific decisions:

- **What to train:** the action expert (and aux/DA3 cross-attention) only; freeze the
  VLM backbone. LoRA on the action expert is the cheap, reversible default — matches
  the existing fine-tuning configs in `policy/pi05/src/openpi/training/config.py`.
- **Chunk shape:** the adjoint is a (H, 14) tensor like the chunk; gripper columns
  (6, 13) get zero adjoint (f never touches them — same rule as guidance).
- **Cost of step 3:** one f evaluation + backward ≈ the per-chunk cost we already pay
  in guidance (sub-second on CPU, far less batched on GPU with grids resident).
- **σ(τ)/λ:** start with the schedule from the paper and λ such that the *typical*
  per-chunk f at initialization maps to an O(1) tilt; calibrate from the predlogs —
  we know the distribution of f over guided/unguided chunks empirically.

Why this beats score/policy-gradient RL here: no reward model, no value bootstrap, no
high-variance likelihood-ratio gradients — the critic's analytic gradient enters the
update directly, which is precisely the asset we spent this branch building.

### 4.3 Route B (cheap auxiliary): one-step x̂₁ penalty

During ordinary flow-matching training, at the sampled noise level τ form the one-step
denoised estimate x̂₁ = x_τ + (1−τ)·v_θ(x_τ, τ, s) and add `w(τ)·f(s, x̂₁)` to the loss.

- Pros: trivially cheap (no sampling rollout), dense signal each batch.
- Cons: x̂₁ at high noise is the conditional mean — blurry, can under-report
  penetration; the objective is a heuristic, not the tilted posterior.
- Use: as a regularizer during Phase 1 (warm start) with w(τ) concentrated at low
  noise (τ → 1), not as the main mechanism.

### 4.4 Route C (offline warm start): corrected-label fine-tuning

Run `optimize_chunk` as a **corrector** offline over dataset chunks (it is exactly the
trust-region argmin of f around a₀), producing corrected labels a*. Fine-tune with the
ordinary flow loss toward a* on contact-adjacent states, ordinary labels elsewhere.

**Engine-gated variant (preferred — see §4.5 gating):** generate corrected labels only
for chunks overlapping *recorded contact windows*, with the hinge restricted to the
contacted link/sphere (from the contact pair attribution in the dataset), pushed to
the standoff target and scaled by window impulse. Chunks with model-negative clearance
but no recorded contact keep their original labels — no conservatism tax on model
false positives.

**Spread corrections over the contact window, never a single frame.** Optimizing only
the contact step produces exactly the failure mode to avoid: one frame's joints bulge
out of an otherwise unchanged trajectory — a kink the policy then has to imitate
(this is guidance bugs #8/#9 reappearing as label corruption). The rule, inherited
from `optimize_chunk`: the unit of correction is the **whole chunk containing the
contact window**. The hinge fires only at the contacted sphere/steps (the gate), but
the velocity/acceleration/anchor regularizers act on the entire chunk's correction
field d = a − a₀, so the answer to "sphere 7 of link3 penetrates at t=143" is a
smooth detour ramping in ~20–40 steps earlier and blending back after. Concretely:
(a) corrected-label generation selects every chunk overlapping `in_contact_window`
(±50 steps), which includes the APPROACH chunks — real avoidance is a different
approach, not a swerve at the moment of contact; (b) all H waypoints of a selected
chunk are optimized jointly and saved jointly — never a partial-chunk edit; (c) the
chunk-start continuity ramp stays on, so corrected labels remain consistent with the
executed prefix at training time exactly as corrections do at inference time.

- Pros: pure supervised learning — no sampler in the loop, very stable, reuses today's
  optimizer verbatim; bakes in the same corrections guidance makes online, minus the
  closed-loop fight.
- Cons: inherits guidance's per-instance limits (§3) — it can only teach corrections
  the chunk optimizer can find. It shifts the policy *toward* safe behavior; it cannot
  reshape the distribution beyond the corrector's authority.
- Use: Phase 1. It is the cheapest way to take the big first step before on-policy
  fine-tuning (Route A) does the precise work.

### 4.5 When a learned Q actually becomes necessary

f is **chunk-local**: it scores the configurations commanded *within* this chunk. Two
situations exceed it:

1. **Cross-chunk credit:** the fatal approach is committed in chunk k, the contact
   happens in chunk k+1 (the ep3 t=0 emergency is exactly this — the previous episode
   structure put the arm there). Adjoint matching with chunk-local f still helps
   (every chunk of every rollout is trained, including chunk k at its own state), but
   the *gradient at chunk k* doesn't see the future contact.
2. **Deployment without ground truth:** the real robot has no SDF.

Both are solved by the same object: a learned critic **Q̂(s, a_1:H)** — the Stage-1 aux
head extended to condition on the action chunk, trained to regress (a) f's value at
(s, a) for the local term and (b) a TD/discounted-sum target over rollout data for the
closed-loop term. Then ∇_a Q̂ replaces ∇_a f in step 3 of §4.2 with no other change.

**Target choice for the chunk-local term — commanded, not realized.** The geometric
regression target is f computed on the *commanded* chunk (free-space FK clearance),
NOT the clearance realized in the rollout, even though the rollout is recorded.
Reason: contact **censors** realized clearance — commanding 2 cm or 5 cm through an
obstacle realizes nearly the same ≈ −1 cm (the obstacle stops the arm; the excess
becomes impulse), so a realized-clearance critic is flat exactly in the penetration
regime where the avoidance gradient must be steepest. Realized outcomes enter through
the channels where they carry unique information instead: chunk impulse/force totals
and the closed-loop success/contact-to-go targets, as separate heads on the same
trunk. The analytic target also permits **counterfactual augmentation**: perturb
stored actions offline and label f(s, a′) exactly — unlimited (s, a) pairs around the
data distribution, which is what makes Q̂'s *gradients* (not just values) accurate.

**Gating — engine contact, not a model threshold.** Model distance values carry the
±1.5 cm calibration error; engine contacts do not. The supervision/correction rule is
therefore **engine-gated, geometry-directed**:

- *When*: a correction (or a contact-loss term) applies only where the engine
  recorded contact — model-negative clearance with no recorded contact gets NO
  perturbation (false positive: don't pay the conservatism tax); model-positive
  clearance WITH recorded contact does (false negative: sphere gap / under-read —
  perturb the contacted sphere identified from the contact pair, e.g. link3/sphere7,
  regardless of what the distance value claims).
- *Which way / where*: û at the contacted sphere, point Jacobian at that sphere's
  center. Direction from measured-pose geometry at the contact step is robust even
  when the distance *value* is miscalibrated — direction is relative geometry, not a
  zero point.
- *How far*: NOT realized depth (censored) and not a bare "stop touching" (a pure
  contact gate trains a grazer that surfs the boundary with zero buffer) — push the
  contacted sphere to a **standoff target** (clearance ≥ margin at that sphere),
  with magnitude scaled by the window's accumulated **impulse** (the engine's own
  severity signal). The margin is thus demoted from trigger to target depth — a far
  less calibration-sensitive role.
- *Learned version*: the aux head's gate logit is supervised by ENGINE contact
  (P(contact on link ℓ within the next W steps)), not by `d_model < margin` — the
  network learns the calibration from data instead of inheriting the hand-set
  threshold (§1.1).
Sequencing: do **known-f adjoint matching first** (no model error in the gradient),
distill Q̂ from the same rollout data in parallel, swap in Q̂ only when (1) or (2) bites.

### 4.6 The task-progress critic Q̂_task — making the policy finish the job

A separate value function trained to score **task progression and success**, used as
the positive term in the tilt (§4.1) so the fine-tuned policy is optimized to avoid
collisions *while completing the task*, not instead of it.

**Signature.** Q̂_task(s, a_1:H) → ℝ: given the current observation and a candidate
action chunk, predict the discounted task outcome after executing the chunk. It must
condition on the chunk (not just s) and be differentiable w.r.t. it, because its
gradient enters the terminal adjoint — same requirement as the collision critic, so it
shares the same architecture pattern: the Stage-1 aux-head trunk with action tokens
cross-attended, two output heads (clearance → f̂/Q̂ in §4.5, progress → Q̂_task). One
backbone, two critics.

**Training targets — all already in the logs:**

1. **Success-to-go (primary).** From every logged rollout (BC data, eval rollouts,
   Phase-2 collections — failures included), label each chunk boundary t with
   `y_t = γ^(T_succ − t)` if the episode succeeded at step T_succ, else 0.
   Regress Q̂_task(s_t, a_t:t+H) → y_t where a_t:t+H is the chunk **actually executed**
   (Monte-Carlo, no bootstrapping — stable, unbiased, and the episodes are short
   enough that MC variance is acceptable).
2. **Stage progress (dense shaping, optional).** The bench envs expose stage flags
   (`stage_success_tag`, per-task `check_success` sub-stages) and the expert
   demonstrations give normalized time-to-completion; either provides a dense
   progress fraction p_t ∈ [0,1] as an auxiliary regression target. Dense targets
   matter early in fine-tuning, when most sampled chunks neither collide nor finish —
   without them, both tilt terms are flat for the bulk of samples.
3. **Hindsight failure labels.** Episodes that stalled or timed out (the exact mode we
   are guarding against) are *negative* examples with y = 0 at every t — these teach
   Q̂_task that low-motion chunks near an unfinished task have low value, which is
   precisely the gradient that pushes back against over-conservative collision
   avoidance.

**Fitting protocol.** Train on the same buffer as §6.3 but *without* the
contact-region weighting (success signal lives everywhere, not just near contacts);
refit (or fine-tune) at every Phase-2 buffer refresh so the critic tracks the current
policy's distribution — a Q̂_task fitted only on π_base goes stale exactly like the
state buffer does. Hold out seeds for critic validation; a Q̂_task whose ranking of
(success vs failure) holdout chunks is poor (AUC ≲ 0.8) is not ready to put in the
adjoint.

**Why a learned critic is unavoidable here** (unlike the collision side, where f is
analytic): "task progress" has no closed-form differentiable expression — it lives in
the environment's success predicate. The asymmetry is deliberate: the collision term
enters with zero model error (known f), and the learned term enters only as the
*positive* drive, where exploitation is bounded by the base-policy tilt and the BC
anchor (a hacked Q̂_task can at worst pull toward base-policy-like behavior, which is
exactly what the anchor already does).

**Guardrails specific to a learned value in the gradient:**
- Small ensemble (2–3 heads), use the min or mean-minus-std as the adjoint target —
  cheap pessimism against value hacking.
- Clip ‖∇_a Q̂_task‖ per chunk to the same order as typical ‖∇_a f‖ on active chunks
  (we know that scale empirically from the predlogs) so η stays interpretable.
- The §5.1 gate already measures real success; if real success and Q̂_task's predicted
  success diverge across an iteration, the critic is being exploited — refit before
  continuing.

**Relation to existing repo infra:** the advantage-conditioned training path
(`src/openpi/training/advantage_data_loader.py`, `scripts/train_divl.py`) already
computes per-rollout return/advantage labels for pi05; Q̂_task's success-to-go targets
are the same bookkeeping with a chunk-conditioned head, so the data plumbing can be
reused nearly as-is.

## 5. The three plans

Three interchangeable inner steps for the same outer loop (collect → relabel →
train offline → gate → repeat); they differ in which critic supplies the
correction signal. §5.4 compares them and gives the sequencing.

### 5.1 Plan v1 — engine-gated corrected labels, iterative offline RL

The correction mechanics are §4.4 (engine-gated variant, corrections spread
over contact windows); the state buffers and anchors are §6. The outer loop:

Collect → relabel → optimize
offline → collect with the improved policy → repeat. The triplet dataset
(`data_generation_documentation.md`) is iteration 0; later iterations re-collect only
the policy leg — the planner sources are policy-independent and collected once.

```
D_expert  = curobo_collision_free episodes        (fixed, ALWAYS in every D_k)
D_neg     = curobo_collision episodes             (fixed, calibration positives)
π_0       = BC policy (clutter-aware, Stage 1)

for k = 0..K:
    1. Collect M rollouts with π_k (triplet collector, pi05 leg; paired seeds,
       contact streams on, guidance OFF — measure the policy, not the safety net).
    2. Relabel pass → per-link labels, contact attribution, windows.
    3. D_k = D_expert + rollouts(π_k) [+ decayed rollouts(π_{k-1},...)] + D_neg.
       Fresh rollouts must DOMINATE the correction signal: the failure
       distribution moves every iteration (after the policy stops grazing the
       kettle, its residual failures are elsewhere) — stale corrections fix
       problems that no longer exist.
    4. Engine-gated corrected labels on D_k's contact-window chunks (§4.4,
       spread over the window); refit Q̂_task on D_k (§4.6).
    5. Train π_{k+1} offline: flow loss toward corrected labels on gated
       chunks + λ_BC·L_FM on D_expert (+ optionally the x̂₁ penalty §4.3).
       [Upgrade path: replace this inner step with adjoint matching §4.2 —
        the outer loop is unchanged.]
    6. Gate (§7) BEFORE π_{k+1}'s rollouts become the next dataset: paired-seed
       eval vs π_base AND vs π_k.  Success drop > 3 pts absolute → restore
       checkpoint, weaken corrections (smaller standoff / fewer gated chunks).
       Success flat AND contact flat → raise η (task drive) first.
       A regressed policy admitted to step 1 poisons every later iteration.
```

K = 3–5 iterations of M ≈ 100–200 episodes is the expected scale before returns
flatten. This is iterated corrected-label fine-tuning — DAgger with the chunk
optimizer as the expert — phrased as offline RL: every optimization step runs on a
static, relabeled dataset, and on-policy data enters only through the collection
step, where it is fully logged and gated.


### 5.2 Plan v2 — global analytic critic: optimize policy SAMPLES under f

The complementary plan to v1 (engine-gated iterative offline RL, §5.1). Where v1
trusts the **engine** — correct only what actually made contact, at recorded
trajectories — v2 trusts the **model** everywhere: sample action chunks from the
current policy, evaluate the produced sequence with the analytic critic
(FK → spheres → SDF → f), and optimize the policy on the predicted collision
distance directly.

```
v2 inner step:
    s  ~ state buffer (scene record + image observation from the dataset)
    a  ~ π_θ(·|s)            # sample the policy's own chunk
    L  = w · f(s, a)         # analytic critic, differentiable end-to-end
    ∂L/∂θ through the sampling chain
```

**What it buys over v1:** dense and *preventive* — gradient flows wherever the
policy's samples come near obstacles, including collisions no rollout has realized
yet; inherently on-policy (every gradient step samples the current π_θ — no
DAgger-style distribution chase); no corrected-label generation in the loop.

**Three instantiations of "∂L/∂θ through the sampling chain"** (increasing rigor):
1. **One-step x̂₁ estimate** (§4.3) — cheapest, biased at high noise.
2. **Pathwise backprop through the full sampling ODE** — the literal version of
   this plan; works, but memory grows with integration steps and unconstrained
   reward-on-samples optimization is the classic value-collapse setup; needs a
   strong BC anchor.
3. **Adjoint matching** (§4.2) — the stabilized form of exactly this signal:
   same "sample, assess, differentiate," with the memoryless schedule and the
   base-policy tie that make it a well-posed objective. v2 taken to its rigorous
   conclusion IS Route A.

**The cost — calibration exposure.** v2 reinstates the global threshold that v1's
engine gating was designed to remove: f fires wherever *predicted* clearance is
small, model false positives included (conservatism tax returns), and the model's
under-read blind spots (sphere gaps, held object) stay invisible since no engine
ever checks the samples. Mitigation — the **hybrid gate**: weight v2's loss by the
engine-calibrated contact logit from the aux head (§1.1), so the learned,
contact-supervised gate decides *where* to perturb and analytic f supplies the
*direction and magnitude*. Engine decides when, geometry decides which way — the
§4.5 principle lifted from recorded contacts to sampled actions.

**Requirements & guards:** scene geometry per training state (the stored
`scene/seed_N` + `fk_basis.npz` cover dataset states — no live sim in the training
loop); the same anti-stall protections as everywhere else (λ_BC anchor, Q̂_task
term) — a collision-only global tilt finds "smooth but parked" minima even faster
than gated corrections do.

**How v1 and v2 compose:** they are not rivals. The v1 outer loop (§5.1) stays;
v2 enters as an additional loss term in its training step (states from the same
buffer, chunks sampled fresh from π_θ, f evaluated globally with the hybrid gate) —
or replaces the corrected-label inner step entirely once instantiation 3 is wired.
Sensible sequencing: run v1 alone first (it is calibration-proof and cheap to
debug), add v2 as a regularizer second, promote to full adjoint matching last.

### 5.3 Plan v3 — learned collision value function + iterative offline RL

The third plan: train a **value function for collision** — Q̂_col(s, a_1:H) on the
aux-head trunk (§4.5/§1.1), observation in, action chunk cross-attended — and run
the same iterative offline RL outer loop (§5.1) with Q̂_col as the critic instead of
corrected labels (v1) or analytic f (v2).

**Targets** (multi-head, all from the triplet dataset + each iteration's rollouts):
1. *Value matching:* analytic f on commanded chunks — dense, exact, augmentable
   with counterfactual perturbed actions (§4.5 target choice).
2. *Closed-loop contact-to-go:* discounted future ENGINE contact/impulse after the
   chunk — the credit-assignment signal nothing else provides.
3. *Contact logit:* P(engine contact | s, a) — the learned calibration (§4.6 twin).

**Inner step options** (replacing v1's corrected-label training):
- **AWR-style (recommended first):** weight the flow-matching loss on dataset/rollout
  chunks by exp(−β·Â_col) with Â_col = Q̂_col(s,a) − mean_a'~π Q̂_col(s,a') — no
  gradient ever taken *through* Q̂, so model error can mis-weight but not fabricate
  directions. Plugs directly into the existing advantage-conditioning infra
  (`advantage_data_loader.py`, `train_divl.py`).
- **Pathwise:** sample a ~ π_θ, descend ∇_a Q̂_col through the sampler (v2's
  mechanics with Q̂ in place of f) — stronger signal, full value-hacking exposure;
  requires the §4.6 guardrails (ensemble pessimism, gradient clipping, refit every
  iteration) plus the BC anchor.

**What v3 uniquely buys:**
1. **Closed-loop credit assignment** — trained on contact-to-go, Q̂_col blames the
   *approach* chunk for a collision realized two chunks later (the ep3 chunk-start
   emergency: f is structurally blind to it, v1 only reaches it via window-overlap
   heuristics).
2. **No ground-truth scene at query time** — Q̂_col reads the observation, so the
   inner step needs no stored geometry, and this is the ONLY plan that transfers to
   the real robot as-is. v3 is the deployment endgame, converging with the
   perception branch by construction (same trunk).

**What v3 uniquely risks:** a learned model inside the optimization loop — OOD
exploitation at sampled actions (§6 discussion: never *trained* OOD, but *queried* near-OOD
every inner step), plus standard offline-RL distribution-shift pitfalls as π_k moves
between refreshes. Mitigations are the §4.6 set + AWR-first sequencing + the
counterfactual-augmentation trick (label perturbed actions with exact f) to widen
Q̂'s trustworthy neighborhood before any gradient flows through it.

### 5.4 The three plans side by side

| | v1 (§5.1) | v2 (§5.2) | v3 (§5.3) |
|---|---|---|---|
| Critic | engine contacts + chunk optimizer | analytic f (FK×SDF) | learned Q̂_col |
| Signal coverage | realized contacts only | wherever samples enter the margin shell | everywhere Q̂ generalizes, incl. cross-chunk |
| Calibration risk | none (engine truth) | full (model thresholds) | learned (engine-supervised), but exploitable |
| Model error in gradient | none | none (geometry exact, scope chunk-local) | yes — the defining risk |
| Scene needed at train time | for labels only | yes (per-state geometry) | no (observation-based) |
| Real-robot transfer | no (needs engine) | no (needs scene) | **yes** |
| Sequencing | first | second (regularizer → adjoint matching) | last; consumes v1/v2's data & supervision |

Same outer loop for all three (collect → relabel → train offline → gate → repeat);
they differ only in the inner step's critic, and each earlier plan generates exactly
the data and supervision the next one needs.

## 6. Data strategy: train on the dataset, sample from the policy, or both?

**Both — with a strict division of labor: states come from data, actions come from the
policy, and the loop iterates.** This is the concrete answer to the question.

### 6.1 Why dataset-actions-only is not enough

- The hinge is ~zero on most expert/successful actions — **no gradient** flows from f
  at dataset actions. (Empirically: clear chunks skip free in guidance; most chunks
  are clear.)
- The collisions you need to remove are produced by the *policy's own* samples in the
  *policy's own* visitation distribution — off-data by definition. Optimizing f at
  dataset actions changes nothing about what the policy will actually sample.
- Exception: Route C's corrected labels work offline precisely because the corrector
  *generates* the improved action — but its reach is bounded (§4.4).

### 6.2 Why pure on-policy collection is wasteful

- Sim rollouts are the expensive resource (minutes/episode); most timesteps of most
  rollouts have zero f-gradient (clear chunks).
- We already possess exactly the prioritization signal needed: the dataset and the
  predlogs record, per timestep, contact pairs, impulses, force_steps, and predicted
  clearances (`sdf_debug/predlog_*.jsonl`, `episodes.jsonl`, the rollout collector's
  proximity stream).

### 6.3 The concrete recipe

**State buffer construction (offline, once per refresh):**

```
for every logged rollout (BC data + eval rollouts + collected proximity rollouts):
    for every chunk boundary t:
        w(s_t) = 1                                   # base: keep full coverage
                 + α · 1[recorded contact within t ± N steps]      (α≈4, N≈50)
                 + β · 1[predicted clearance < margin at t]        (β≈2)
                 + δ · 1[force_steps > 0 within t ± N]             (δ≈4)
    store (s_t, scene snapshot: object ids+poses, cached SDF grid key, w)
```

Precompute SDF grids per (episode, scene-version) — rebuild events are already logged —
and serialize the FK basis once per embodiment. Grids are ~2–15 MB each; a training
batch holds its grids on GPU.

**Phase 1 — offline warm start (Routes C + B):** corrected-label fine-tuning on
contact-weighted states, x̂₁ penalty as regularizer, λ_BC·L_FM on the full dataset.
Cheap, stable; expect the bulk of the contact-rate drop here.

### 6.4 Anti-forgetting anchors (non-negotiable)

- λ_BC·L_FM on the *unweighted* expert dataset in every batch (not just contact
  regions) — this is what prevents the "smooth but stalled" failure we already
  produced once with over-regularized guidance (success 9/10 → 5/10).
- Keep ≥ 30 % of buffer samples at base weight 1 regardless of contact annotations.
- LoRA + checkpoint gating make every phase cheaply reversible.

## 7. Evaluation protocol

Reuse the harness from this branch unchanged:

- **Paired-seed comparison** (`seed_rng`, identical flow noise) of fine-tuned vs base
  policy, guidance off: success, CONTACT (primary: contact_steps, force_steps/ep),
  collisions (displacement metric, provenance-filtered as of 2026-06-10).
- **Predlog calibration curve** per checkpoint: precision–recall of predicted vs
  realized contact over clearance thresholds — confirms the policy is *moving* the
  clearance distribution, not gaming the hinge.
- **Ablations:** Phase 1 only vs Phase 1+2; chunk-local f vs Q̂ swap; λ sweep.
- Holdout tasks/scenes never used for buffer harvesting.

Secondary check: fine-tuned policy **plus** guidance at inference — the residual
correction magnitude (`max|dq|` in the sdf logs) should shrink toward zero as training
succeeds; it is a free, per-chunk measure of how much unsafe intent remains.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Success collapse from over-tilting (the training-time analogue of the smoothing bug) | Q̂_task term in the tilt (§4.6) — the objective itself rewards finishing; λ_BC anchor on full data; gate in §5.1; tilt λ is the single knob to retreat on |
| Q̂_task exploitation (policy maximizes predicted, not real, success) | ensemble pessimism + gradient clipping (§4.6); refit every buffer refresh; gate compares predicted vs realized success |
| Hinge gaming: policy hugs the exact margin boundary | depth-weighted hinge already penalizes proximity superlinearly; monitor clearance histogram, not just f |
| Held-object blind spot trains in as "safe" | attach grasped-object spheres to the EE frame before Phase 2 (open item, ~20 % of forceful contacts) |
| Unsigned-grid fallback gives weak gradients exactly on the worst (penetrating) samples | watertight-mesh audit for training assets; assert signed grids in the data pipeline, don't silently fall back |
| Memoryless-sampler mismatch with pi05's deterministic inference sampler | fine-tune with memoryless SDE, deploy with the usual ODE — adjoint matching is designed for exactly this split; verify with paired-seed eval |
| Sim-only critic (no real-world f) | that is Stage 1's aux head / Q̂ distillation path (§4.5), not a blocker for sim validation |

## 9. Roadmap

1. **Now:** finish the clean guidance comparison (correction-field fix + provenance
   metric) — that number is f's final validation certificate.
2. Data pipeline: state buffer + cached grids + FK serialization (§6.3); grasped-object
   spheres.
3. Phase 1: corrected-label warm start + x̂₁ regularizer; gate on paired eval.
   In parallel: fit Q̂_task v0 on existing rollout logs (success-to-go targets reuse
   the advantage-conditioning plumbing) and validate its holdout ranking.
4. Phase 2: the iterative offline RL loop (§5.1, plan v1) — collect with π_k,
   relabel, engine-gated corrected labels spread over contact windows, train
   offline with the expert anchor, gate, repeat.
5. Phase 2b: add the global-critic term (§5.2, plan v2) as a regularizer inside
   the v1 training step — policy samples assessed by analytic f with the hybrid
   (contact-logit) gate; promote to full adjoint matching with terminal cost
   f − η·Q̂_task once the simple loop plateaus (start η = 0, enable after Q̂_task
   validates).
6. Phase 3 (plan v3, §5.3): fit Q̂_col on accumulated rollouts (aux trunk, action
   tokens cross-attended; targets = analytic f + contact-to-go + contact logit);
   run the same outer loop with the AWR inner step first, pathwise ∇Q̂ after it
   validates; re-gate.
7. Merge with perception branch: DA3 + aux head + fine-tuned action expert, end-to-end
   eval; real-robot transfer uses Q̂ only.

## 10. Code pointers

| Component | Location |
|---|---|
| Critic f, FK, SDF, corrector (`optimize_chunk`) | `customized_robotwin/script/differentiable_proximity.py` |
| Guidance/eval harness, paired RNG, predlogs | `customized_robotwin/script/eval_policy_proximity_guidance.py` |
| Contact/collision metrics (provenance-filtered) | `benchmark/bench_envs/_bench_base_task.py` |
| pi05 flow head / action expert | `customized_robotwin/policy/pi05/src/openpi/models/pi0.py` |
| Training entry points to extend | `customized_robotwin/policy/pi05/scripts/train_collision.py`, `scripts/train.py`, `src/openpi/training/config.py` |
| Return/advantage labeling to reuse for Q̂_task targets | `customized_robotwin/policy/pi05/src/openpi/training/advantage_data_loader.py`, `scripts/train_divl.py` |
| Rollout collection with proximity stream | `customized_robotwin/script/collect_rollout_proximity_client.py` |
| Aux-head spec (perception branch) | `auxiliary_collision_heads_rgb_only_spec.md` |
