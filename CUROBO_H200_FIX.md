# Fixing curobo `CUDA error: an illegal instruction was encountered` on Hopper (sm_90)

**Applies to:** NVIDIA H100 / H200 / GH200 — any `sm_90+` GPU running cuRobo.
**Status:** root cause confirmed and fixed; two related issues documented but unfixed (see [Remaining issues](#5-remaining-issues)).

> This document is written to be executed by a human **or** an AI agent.
> [Patch procedure](#3-patch-procedure) contains exact, unique search/replace
> pairs. Do not paraphrase them.

---

## 1. Does this apply to you?

You are hitting this bug if **all** of the following are true:

- GPU compute capability is **9.0 or higher**
  (`python -c "import torch; print(torch.cuda.get_device_capability(0))"`).
- A cuRobo call — most often `motion_gen.warmup()` — dies with:
  ```
  RuntimeError: CUDA error: an illegal instruction was encountered
  ```
  often followed by `Warp CUDA error 715: an illegal instruction was encountered`.
- The reported Python frame **varies between runs** (e.g. `curobolib/ls.py`
  `update_best`, `curobolib/opt.py` `lbfgs_step_cu.forward`). CUDA reports
  kernel faults asynchronously, so the reported frame is *not* the faulting
  kernel. Do not chase it.

### Not this bug

- **Vulkan / SAPIEN warnings** (`Failed to find system libvulkan`,
  `Failed to find Vulkan ICD file`) printed near the traceback are cosmetic and
  unrelated. Check `ls /etc/vulkan/icd.d/ /usr/share/vulkan/icd.d/` — if an
  `nvidia_icd.json` is present, Vulkan is fine.
- **Arch mismatch** normally raises `no kernel image is available`, not
  `illegal instruction`.

### Hypotheses to skip — already tested and disproven

Each of these was investigated on a confirmed reproduction. Do not re-derive them.

| Hypothesis | Verdict | How it was ruled out |
|---|---|---|
| Extensions built for the wrong arch | **No** | `cuobjdump` showed correct `sm_90` SASS in every `curobolib/*.so`; torch reported `sm_90` and detected the GPU. |
| CUDA graph capture | **No** | `use_cuda_graph=False` failed *identically*, in the same kernel. Plain torch CUDA graph capture succeeds on the same box. |
| Robot/task config specific | **No** | Reproduced with cuRobo's stock `franka.yml`. |
| Broken CUDA / driver install | **No** | torch matmul, `nvcc`, and a standalone call to cuRobo's `update_best` kernel all succeed. |
| Device-side `assert` firing | **No** | The asserts in `line_search_kernel.cu` are never reached; sanitizer pointed at a shuffle instruction. |
| `~/.local` shadowing the env | **No** | Real hygiene problem (see §5.3), but not the cause. |

---

## 2. Root cause

In `src/curobo/curobolib/cpp/lbfgs_step_kernel.cu`, warp shuffle reductions run
under a **partial** warp mask while **all 32 lanes** reach the shuffle:

```cuda
unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < m);   // e.g. m=15 -> lanes 0..14
val += __shfl_down_sync(mask, val, 1);                       // executed by ALL 32 lanes
```

Lanes `m..31` participate in a sync whose mask excludes them — undefined
behavior. Pre-Hopper architectures tolerated it; **sm_90 enforces it and traps.**

Evidence from `compute-sanitizer`:

```
========= Illegal instruction
=========     at 0x480 in void Curobo::Optimization::lbfgs_update_buffer_and_step_v1_compile_m<float, float, (bool)0, (int)15>(...)
=========     by thread (0,0,0) in block (0,0,0)
```

Thread (0,0,0) of block (0,0,0) — the very first thread — rules out races and
out-of-bounds indexing. The SASS at that offset, where `R15` holds the partial
mask produced by `VOTE.ANY`:

```
/*0470*/  BRA.DIV UR7, 0x50d70 ;
/*0480*/  WARPSYNC R15 ;              <-- traps here
/*0490*/  SHFL.DOWN PT, R11, R16, 0x1, 0x1f ;
```

**Fix strategy:** reduce under `FULL_MASK` so every lane is named in the mask,
and zero the lanes that must not contribute. Numerically identical to the
intended masked reduction, and legal on sm_90.

---

## 3. Patch procedure

### 3.0 Locate the file

cuRobo may be vendored in-tree or pip-installed:

```bash
# vendored checkout
find . -name lbfgs_step_kernel.cu 2>/dev/null
# or installed package
python -c "import curobo, pathlib; print(pathlib.Path(curobo.__file__).parent)"
```

Target: `<curobo_root>/src/curobo/curobolib/cpp/lbfgs_step_kernel.cu`
(a pip-installed copy may place it under `curobo/curobolib/cpp/`).

**All four edits are in this one file.**

### 3.1 Check whether the patch is already applied

```bash
grep -c "__shfl_down_sync(FULL_MASK" <path>/lbfgs_step_kernel.cu
```

- `0` → unpatched, proceed.
- `7` → already patched, skip to [§4](#4-rebuild-and-verify).
- anything between → partially applied; inspect each site before editing.

### 3.2 Edit 1 — `reduce_v0`, first-stage warp reduction

**Find:**
```cuda
      psum_t   val  = v;
      unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < m);

      val += __shfl_down_sync(mask, val, 1);
      val += __shfl_down_sync(mask, val, 2);
      val += __shfl_down_sync(mask, val, 4);
      val += __shfl_down_sync(mask, val, 8);
      val += __shfl_down_sync(mask, val, 16);
```

**Replace with:**
```cuda
      // All 32 lanes reach the shuffles below, so they must all be named in the
      // mask; a partial mask traps as an illegal instruction on sm_90+. Zeroing
      // the out-of-range lanes keeps the sum identical to a masked reduction.
      psum_t val = (threadIdx.x < m) ? (psum_t)v : (psum_t)0;

      val += __shfl_down_sync(FULL_MASK, val, 1);
      val += __shfl_down_sync(FULL_MASK, val, 2);
      val += __shfl_down_sync(FULL_MASK, val, 4);
      val += __shfl_down_sync(FULL_MASK, val, 8);
      val += __shfl_down_sync(FULL_MASK, val, 16);
```

### 3.3 Edit 2 — `reduce_v0`, second-stage cross-warp reduction

> Distinguished from Edit 4 by the **absence** of `#pragma unroll` before the loop.

**Find:**
```cuda
        int elems      = (m + 31) / 32;
        unsigned mask2 = __ballot_sync(FULL_MASK, threadIdx.x < elems);

        if (threadIdx.x / 32 == 0) // only the first warp will do this work
        {
          psum_t val2  = data[threadIdx.x % 32];
          int    shift = 1;

          for (int i = elems - 1; i > 0; i /= 2)
          {
            val2  += __shfl_down_sync(mask2, val2, shift);
            shift *= 2;
          }

          // int leader = __ffs(mask2) – 1;    // select a leader lane
```

**Replace with:**
```cuda
        int elems = (m + 31) / 32;

        if (threadIdx.x / 32 == 0) // only the first warp will do this work
        {
          psum_t val2  = (threadIdx.x < elems) ? data[threadIdx.x % 32] : (psum_t)0;
          int    shift = 1;

          for (int i = elems - 1; i > 0; i /= 2)
          {
            val2  += __shfl_down_sync(FULL_MASK, val2, shift);
            shift *= 2;
          }

          // int leader = __ffs(mask2) – 1;    // select a leader lane
```

### 3.4 Edit 3 — `reduce_v1`, first-stage reduction via `warpReduce`

**Find:**
```cuda
      unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < m);
      psum_t   val  = warpReduce(v, 32, mask);
```

**Replace with:**
```cuda
      // See reduce_v0: every lane reaches the shuffles, so reduce under
      // FULL_MASK and zero the lanes that must not contribute.
      psum_t val = warpReduce((psum_t)((threadIdx.x < m) ? (psum_t)v : (psum_t)0),
                              32, FULL_MASK);
```

### 3.5 Edit 4 — `reduce_v1`, second-stage cross-warp reduction

> Distinguished from Edit 2 by the **presence** of `#pragma unroll`.

**Find:**
```cuda
        int elems      = (m + 31) / 32;
        unsigned mask2 = __ballot_sync(FULL_MASK, threadIdx.x < elems);

        if (threadIdx.x / 32 == 0) // only the first warp will do this work
        {
          psum_t val2  = data[threadIdx.x % 32];
          int    shift = 1;

          #pragma unroll
          for (int i = elems - 1; i > 0; i /= 2)
          {
            val2  += __shfl_down_sync(mask2, val2, shift);
            shift *= 2;
          }
```

**Replace with:**
```cuda
        int elems = (m + 31) / 32;

        if (threadIdx.x / 32 == 0) // only the first warp will do this work
        {
          psum_t val2  = (threadIdx.x < elems) ? data[threadIdx.x % 32] : (psum_t)0;
          int    shift = 1;

          #pragma unroll
          for (int i = elems - 1; i > 0; i /= 2)
          {
            val2  += __shfl_down_sync(FULL_MASK, val2, shift);
            shift *= 2;
          }
```

> **Agent note:** Edits 2 and 4 have nearly identical search text. Apply them as
> **separate, non-global** replacements, and keep the distinguishing
> `#pragma unroll` line inside the match. A replace-all will corrupt both sites.

### 3.6 Expected post-edit state

```bash
grep -c "__shfl_down_sync(FULL_MASK" <path>/lbfgs_step_kernel.cu   # -> 7
grep -c "warpReduce(.*FULL_MASK\|32, FULL_MASK" <path>/lbfgs_step_kernel.cu  # -> 1 (Edit 3)
grep -n "mask2" <path>/lbfgs_step_kernel.cu                        # -> 3 hits, ALL in comments
```

The 7 breaks down as 5 (Edit 1) + 1 (Edit 2) + 1 (Edit 4). Edit 3 does not add a
`__shfl_down_sync(FULL_MASK` call of its own — it passes `FULL_MASK` into
`warpReduce`, which is why it is counted separately.

The 3 remaining `mask2` hits must all be commented-out lines; no live code should
reference `mask2` after patching:

```
// int leader = __ffs(mask2) – 1;    // select a leader lane
//psum_t val2 = warpReduce(data[threadIdx.x % 32], elems - 1, mask2);
// // int leader = __ffs(mask2) – 1;    // select a leader lane
```

---

## 4. Rebuild and verify

Set the arch to match your GPU (`9.0` for H100/H200) and point `CUDA_HOME` at
your toolkit:

```bash
cd <curobo_root>
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME="$CONDA_PREFIX"          # or your CUDA toolkit root
python setup.py build_ext --inplace -j 8  # several minutes
```

Confirm the rebuild produced sm_90 code:

```bash
cuobjdump src/curobo/curobolib/lbfgs_step_cu*.so | grep -oE 'sm_[0-9]+' | sort -u
```

### Reproducer

Save as `repro.py`, run from a directory where `curobo` is importable:

```python
import sys, torch
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

use_graph = (len(sys.argv) > 1 and sys.argv[1] == "graph")
cfg = MotionGenConfig.load_from_robot_config(
    "franka.yml",
    {"cuboid": {"table": {"dims": [2, 2, 0.2], "pose": [0, 0, -0.1, 1, 0, 0, 0]}}, "mesh": {}},
    interpolation_dt=1 / 250, num_trajopt_seeds=4,
    use_cuda_graph=use_graph,
)
mg = MotionGen(cfg)
print(f"--- warmup with use_cuda_graph={use_graph} ---", flush=True)
mg.warmup()
torch.cuda.synchronize()
print(f"RESULT use_cuda_graph={use_graph}: OK", flush=True)
```

```bash
python repro.py graph
```

**Expected after the fix:**
```
--- warmup with use_cuda_graph=True ---
RESULT use_cuda_graph=True: OK
```

Warmup succeeds with CUDA graphs **enabled** (the default), so no config change
is needed. Optional deeper check:

```bash
compute-sanitizer --tool memcheck python repro.py nograph 2>&1 | grep -E "Illegal|Invalid|ERROR SUMMARY"
```

`Illegal instruction` must be gone. One `Invalid __shared__ write` may remain —
see §5.1.

---

## 5. Remaining issues

Identified but **not** fixed. Independent of the patch above.

### 5.1 Shared-memory under-allocation by one float (unfixed)

`compute-sanitizer` still reports:

```
========= Invalid __shared__ write of size 4 bytes
=========     at 0x1170 in ...lbfgs_update_buffer_and_step_v1_compile_m<float, float, (bool)0, (int)15>(...)
=========     Address 0x8e8c is out of bounds
```

The host-side dynamic shared-memory size in `lbfgs_step_kernel.cu` allocates:

```cuda
const int shared_buffer_smemsize = (((3 * v_dim)  + 1)  * history_m + 32) * sizeof(float);
// = 3*M*v_dim + M + 32 floats
```

The kernel layout requires `s_buffer_sh` + `y_buffer_sh` + `alpha_buffer_sh`
(`3*M*v_dim`) + `rho_buffer_sh` (`M`) + `data[32]` + `result[1]`
= `3*M*v_dim + M + 33` floats. The trailing `result` element is the
out-of-bounds write.

Candidate fix (**not applied — verify before using**):

```cuda
const int shared_buffer_smemsize = (((3 * v_dim)  + 1)  * history_m + 32 + 1) * sizeof(float);
```

Pre-existing upstream bug. Runs succeed without `compute-sanitizer`, but it is a
genuine out-of-bounds write that can corrupt adjacent shared memory.

### 5.2 Unaudited warp-shuffle site (unverified)

`curobolib/cpp/kinematics_fused_kernel.cu` contains a similar
`__ballot_sync(0xffffffff, batch < batchSize)` feeding `__shfl_down_sync` calls
in `kin_fused_backward_kernel3`. **Not verified, not changed.**
`compute-sanitizer` has not flagged it, but it was not proven safe either. If
you see the same illegal-instruction signature pointing at a kinematics kernel,
apply the §2 fix strategy there.

`line_search_kernel.cu` was examined and is **correct**: threads that must not
participate `return` before the reduce (`if (threadIdx.x >= l2) return;`), so its
masks match the active lanes. Do not "fix" it.

### 5.3 Environment hygiene: user-site shadowing (not a cause of this bug)

```bash
python -c "import sys; [print(p) for p in sys.path]"
python -c "import torch; print(torch.__file__)"
```

If `~/.local/lib/pythonX.Y/site-packages` appears **before** the env's
site-packages, packages installed there override the env. This is Python's
user-site feature (on by default), not `PYTHONPATH`; it comes from running
`pip install` without the env activated.

If the shadowing and env copies are the same version this is harmless here, but
the env is not self-contained. To disable:

```bash
conda env config vars set PYTHONNOUSERSITE=1 -n <env>
conda activate <env>   # re-activate to apply
```

---

## 6. Reference environment

Where this was diagnosed and fixed. The fix is **not** specific to these
versions — it corrects undefined behavior in the CUDA source that any sm_90+
device will trap on.

| Component | Version |
|---|---|
| GPU | 8× NVIDIA H200 (sm_90) |
| Driver | 580.126.20 |
| Python | 3.10 (conda) |
| torch | 2.4.1+cu121 |
| CUDA toolkit | 12.1 |
| cuRobo | vendored in-tree checkout |
