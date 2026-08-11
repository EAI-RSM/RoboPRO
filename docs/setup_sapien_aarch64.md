# GB10 / ARM64 setup with Docker and SLURM

RoboPRO's original environment is an amd64, Python 3.10 stack. The GB10 path uses
an ARM64 NVIDIA container because CUDA-enabled PyTorch for this machine must come
from NVIDIA's GB10-compatible image rather than the generic PyPI wheel.

## Supported environments

| Host | Python | PyTorch | SAPIEN / mplib |
|---|---:|---|---|
| Linux amd64 | 3.10 | PyPI `torch==2.4.1` | Existing pinned PyPI wheels |
| Linux ARM64 / GB10 | 3.12 | NGC PyTorch 25.11 | Repaired/rebuilt during the Docker build |

The architecture markers are in the root `pyproject.toml`. On ARM64, Torch,
Torchvision, SAPIEN, mplib, Open3D, and PyTorch3D are deliberately not resolved
from PyPI:

- Torch and Torchvision are supplied by `nvcr.io/nvidia/pytorch:25.11-py3`.
- SAPIEN's official ARM wheel requires runtime repair on Ubuntu 24.04.
- mplib is compiled against a single FCL/OctoMap 1.10 stack.
- Open3D has no ARM64 wheel; RoboPRO uses Trimesh for point-cloud export.
- A Torch farthest-point sampler is used when PyTorch3D is unavailable.

## Build the image

Run this on an ARM64 GB10 node with Docker and NVIDIA Container Toolkit:

```bash
IMAGE=robopro:gb10 bash scripts/docker/build_gb10.sh
```

CuRobo v0.7.8 is installed by default. For a faster core-image build while
working only on SAPIEN/mplib integration:

```bash
IMAGE=robopro:gb10-core INSTALL_CUROBO=0 bash scripts/docker/build_gb10.sh
```

The multi-stage build performs these reproducible operations:

1. Starts from NGC PyTorch 25.11 (Python 3.12, CUDA 13).
2. Builds FCL 0.7 against the OctoMap 1.10 libraries used by Pinocchio 2.7.
3. Builds mplib 0.2.1 for CPython 3.12/aarch64 against that FCL build.
4. Combines the current official SAPIEN ARM nightly with the denoiser libraries
   from SAPIEN 3.0.3 and relinks glibc/GCC runtime dependencies to Ubuntu 24.04.
5. Installs the architecture-neutral dependencies from `pyproject.toml` without
   replacing NVIDIA's Torch.

## Local smoke test

```bash
IMAGE=robopro:gb10-core bash scripts/docker/smoke_gb10.sh
```

The expected line contains `aarch64`, `True`, `NVIDIA GB10`, the SAPIEN and
mplib versions, and `Scene`. This creates a PhysX-only scene so native-code and
CUDA validation are independent of the host's Vulkan installation.

Test the renderer separately:

```bash
IMAGE=robopro:gb10-core TEST_RENDER=1 bash scripts/docker/smoke_gb10.sh
```

The wrapper passes `/dev/dri`, `/dev/nvidia-modeset`, and NVIDIA Vulkan/GLVND
manifests into Docker when they exist on the host.

For an interactive shell with the current checkout mounted:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
  -v "$PWD:/workspace/RoboPRO" \
  -w /workspace/RoboPRO \
  robopro:gb10
```

## SLURM

Docker does not request a GPU from SLURM. Request the resource with `sbatch` or
`srun`, then pass only SLURM's allocated device to Docker. The supplied wrapper
does this with `--gpus "device=${CUDA_VISIBLE_DEVICES}"`:

```bash
IMAGE=robopro:gb10 sbatch scripts/slurm/slurm_docker_gb10.sh
```

Run a repository command by placing it after the script name:

```bash
IMAGE=robopro:gb10 sbatch scripts/slurm/slurm_docker_gb10.sh \
  bash -lc 'cd customized_robotwin && source set_env.sh && python -c "import sapien, mplib; print(sapien.__version__, mplib.__version__)"'
```

Docker images normally live in a node-local daemon. On a multi-node cluster,
either build/load the image on every possible GB10 node or publish it to a
registry. For a registry image, ask the job to pull it on the allocated node:

```bash
sbatch --export=ALL,IMAGE=registry.example/robopro:gb10,PULL_IMAGE=1 \
  scripts/slurm/slurm_docker_gb10.sh
```

The wrapper mounts the checkout at `/workspace/RoboPRO` and runs as the calling
UID/GID so generated results are not owned by root.

## Reproducibility note

The amd64 benchmark remains pinned to SAPIEN 3.0.0b1. The validated ARM64 path
uses the June 2026 SAPIEN nightly because the older ARM artifacts are not usable
with the GB10 Python/CUDA stack. Physics or rendering results from these two
SAPIEN versions should not be treated as directly comparable benchmark numbers.
The SAPIEN maintainers also describe Linux ARM64 wheels as not yet fully tested
or manylinux-compliant: <https://github.com/haosulab/SAPIEN/issues/197>.

## Troubleshooting

- `GLIBC_PRIVATE` from `librt`: ensure the image was built with
  `docker/gb10/repair_sapien_wheel.sh`; do not install the upstream wheel over it.
- Heap corruption at Python shutdown: ensure mplib resolves
  `/opt/fcl-cmeel/lib/libfcl.so.0.7`, not Ubuntu's FCL linked to OctoMap 1.9.
- `libucc.so.1: undefined symbol`: retain the image's `LD_LIBRARY_PATH` ordering;
  NVIDIA HPC-X must precede Ubuntu's UCX libraries pulled in by OMPL.
- SAPIEN cannot create a rendering device: first test the host itself with
  `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json vulkaninfo --summary`.
  If that fails with `ERROR_INCOMPATIBLE_DRIVER`, the NVIDIA Vulkan userspace
  package/driver must be repaired by the cluster administrator; Docker cannot
  fix a host ICD that does not initialize. After the host test passes, retain
  `NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute`, `libvulkan1`, the DRM
  device arguments, and manifest mounts supplied by the wrappers.

On the GB10 node used for validation, CUDA and SAPIEN PhysX pass, but the host
NVIDIA 580.95.05 Vulkan ICD currently fails that host-side `vulkaninfo` check.
Rendering therefore remains unavailable until the host driver is corrected.
