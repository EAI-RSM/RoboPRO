---
name: tool_vla_pi05_port
description: "Testing VLAs on this setup: the RoboPRO pi05 checkpoint port (wired, blocked on the expert gate) and why every other VLA is expensive"
metadata:
  type: project
---

2026-07-24 investigation: how feasible is testing the benchmark's vendored VLAs on the custom
occluder scene + clearance metric.

**Harness is already solved.** Uniform RoboTwin policy interface — every `policy/<X>/deploy_policy.py`
exposes `get_model` / `eval` / `reset_model`, and `script/eval_policy.py` drives them generically.
Adding a policy = drop a checkpoint + pick `policy_name`. Scaffolded: RDT, pi0, pi05, openvla-oft,
DexVLA, TinyVLA, LLaVA-VLA (+ non-VLA ACT/DP/DP3/GO1).

**The real cost is weights, not glue.** There are ZERO checkpoints on disk (gitignored on purpose:
`**/checkpoints/`, `policy/weights/*`, `*.pt`, `*.safetensors`), so "we have many VLAs" means
adapters are vendored, not trained models. Two distinct kinds of weights: base VLM **backbones**
have direct HF/S3 links in the per-policy READMEs (DexVLA→Qwen2-VL-2B, TinyVLA→InternVL3-1B,
openvla-oft→OpenVLA-7B, pi0/pi05→openpi S3, RDT→RDT-1B), but the **task policy** for this dual-arm
aloha embodiment is not provided for anyone — you finetune it (collect demos → finetune → merge).
Existing finetuned robot checkpoints are the wrong embodiment/task (openvla-oft→LIBERO,
pi0→DROID), so zero-shot ≈ floor. Each policy also wants its own conda env. That per-model
finetune+env loop dominates, not the code.

**Exception — the RoboPRO pi05 checkpoint exists and is now wired.** HF `mzxuan/robopro_jax_30000`
is a **π₀.₅** checkpoint (config `pi05_robopro_top_cam`), ~3.6B (PaliGemma gemma_2b + gemma_300m
action expert), 19.9 GB, shipping BOTH `jax_30000/` (orbax) and `pytorch_30000/` (safetensors).
Spec: action_dim 32 (14 physical, padded), horizon 50 @ 25 Hz, output [50,14] ABSOLUTE joint
targets; state float32[14] absolute joints, Aloha convention; cams `cam_high` / `cam_left_wrist` /
`cam_right_wrist` at 224²; norm stats at `assets/roboreal_lerobot/norm_stats.json` (REQUIRED).

Port status — **code complete and standalone inference verified.** Done and verified: `.venv` built via
`uv sync --python 3.11` (must pin 3.11; miniconda's 3.13 has no open3d wheel), JAX 0.5.0 on the
RTX 4080; the JAX half downloaded (~15.6 GB) and laid out at
`checkpoints/pi05_robopro_top_cam_jax/robopro/30000/`; `TrainConfig(name="pi05_robopro_top_cam_jax")`
registered in `src/openpi/training/config.py`; obs/action conventions verified to MATCH (model
outputs absolute Aloha-14 and `take_action(action_type='qpos')` treats actions as absolute targets;
state is 14-dim absolute same order; `left_camera`/`right_camera` ARE the wrist cams;
`AbsoluteActions` + `AlohaOutputs(adapt_to_pi)` are applied at inference); `deploy_policy.py`
`encode_obs` switched `head_camera` → **`countertop_camera`** (the model trained on countertop, and
they are DISTINCT cams in this env).

**Standalone model load + inference passed on 2026-07-30.** With no SAPIEN process, a synthetic
observation (three uint8 240x320 RGB images + float32[14] state) returned a finite float64 action
array of shape `[50,14]`. The server alone peaked at **8,454 MiB VRAM** with
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`; use 0.45 when it
must share the 16 GiB RTX 4080 with SAPIEN.

**The old eval smoke path is BLOCKED by the expert feasibility gate, not by the port.** Env boots,
SAPIEN inits, and the client handshakes, but `eval_policy_client.py` crashes before policy inference
in the two-pass expert gate ([[repo_task_assets]]):
`eval_policy_client.py` `play_once()` → `put_mouse_on_pad.grasp_actor_from_table` →
`IndexError: list index out of range` — the known parked mouse-scene grasp problem
([[archive_planner_comparison]]). Two port-independent ENGINEERING bugs also surfaced:
(a) `_office_base_task.py`'s `move()` indexes an empty grasp list without a guard, hard-crashing instead
of cleanly skipping an infeasible seed; (b) the retry loop `while succ_seed < test_num` has NO
failure ceiling — after ~4 fails a curobo/torch CUDA illegal-memory-access corrupted the GPU-0
context (torch + jax co-resident) and it spun 169k identical errors forever.
**`GPU_SPEC` MUST be `0:0`** (single 16 GB card). The VLA occluder rollout driver must bypass
`play_once`; do not repair the expert gate as part of that validation path.

**How the custom metric connects.** The occluder/clearance work is OFFLINE scene analysis with no
policy in the loop, so: using clearance/visibility as a **stratifier** of VLA success is trivial
(run the VLA over seeds, join each to its precomputed eps*/bucket, plot success-vs-bucket — no
metric change; this is the clean experiment). Making VLAs actually FACE the occluder is a modest
port (the spawn must move into the task env `eval_policy` instantiates, config-gated). Clearance as
a per-ROLLOUT outcome (did the executed trajectory keep clearance) is NEW code.

**Open:** zero-shot vs finetuned; which subset of the 7 VLAs; and whether upstream RoboTwin
publishes a checkpoint zoo already finetuned on their aloha tasks (never checked — would be a much
closer starting point than generic backbones).
