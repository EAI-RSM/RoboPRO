#!/usr/bin/env bash
#SBATCH --job-name=robopro-gb10
#SBATCH --partition=gb10
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT_DIR}"
IMAGE="${IMAGE:-robopro:gb10}"
PULL_IMAGE="${PULL_IMAGE:-0}"
IMAGE_TAR="${IMAGE_TAR:-$PROJECT_ROOT/docker/gb10/robopro-gb10.tar}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: this wrapper must run inside a SLURM allocation." >&2
  echo "Submit it with sbatch, or use scripts/slurm/submit_pi05_eval.sh." >&2
  exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: SLURM did not expose an allocated GPU" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: project root is not visible on $(hostname): $PROJECT_ROOT" >&2
  exit 1
fi
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)

if [[ $# -eq 0 ]]; then
  command=(python -c "import torch, sapien, mplib; print(torch.cuda.get_device_name(0), sapien.__version__, mplib.__version__)")
else
  command=("$@")
fi

mkdir -p "$PROJECT_ROOT/logs"

echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES image=$IMAGE"
if ! srun --ntasks=1 --nodes=1 docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [[ -n "$IMAGE_TAR" ]]; then
    if [[ ! -r "$IMAGE_TAR" ]]; then
      echo "ERROR: IMAGE_TAR is not readable on $(hostname): $IMAGE_TAR" >&2
      exit 1
    fi
    echo "loading node-local image from $IMAGE_TAR"
    srun --ntasks=1 --nodes=1 docker load --input "$IMAGE_TAR"
  elif [[ "$PULL_IMAGE" == "1" ]]; then
    echo "pulling node-local image $IMAGE"
    srun --ntasks=1 --nodes=1 docker pull "$IMAGE"
  else
    echo "ERROR: image $IMAGE is absent from node $(hostname)." >&2
    echo "Set IMAGE_TAR=/shared/path/robopro-gb10.tar or use a registry image with PULL_IMAGE=1." >&2
    exit 1
  fi
fi

graphics_args=()
if [[ -e /dev/dri ]]; then
  graphics_args+=(--device /dev/dri)
fi
if [[ -e /dev/dri/renderD128 ]]; then
  graphics_args+=(--group-add "$(stat -c %g /dev/dri/renderD128)")
fi
if [[ -e /dev/nvidia-modeset ]]; then
  graphics_args+=(--device /dev/nvidia-modeset)
fi
for manifest in \
  /usr/share/vulkan/icd.d/nvidia_icd.json \
  /usr/share/vulkan/implicit_layer.d/nvidia_layers.json \
  /usr/share/glvnd/egl_vendor.d/10_nvidia.json; do
  if [[ -f "$manifest" ]]; then
    graphics_args+=(--volume "$manifest:$manifest:ro")
  fi
done

srun --ntasks=1 --nodes=1 docker run --rm \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  "${graphics_args[@]}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env PI05_VENV="${PI05_VENV:-}" \
  --env OPENPI_DATA_HOME=/workspace/RoboPRO/customized_robotwin/policy/pi05/.cache/openpi \
  --env TORCH_EXTENSIONS_DIR=/workspace/RoboPRO/customized_robotwin/envs/curobo/.cache/torch-extensions \
  --env TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-${UID}" \
  --env XDG_CACHE_HOME="/tmp/robopro-cache-${UID}" \
  --env PI05_ASSET_ID="${PI05_ASSET_ID:-trossen}" \
  --env EVAL_RUN_TAG="${EVAL_RUN_TAG:-}" \
  --env PYTHONUNBUFFERED=1 \
  --env NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
  --volume "$PROJECT_ROOT:/workspace/RoboPRO" \
  --workdir /workspace/RoboPRO \
  "$IMAGE" \
  "${command[@]}"
