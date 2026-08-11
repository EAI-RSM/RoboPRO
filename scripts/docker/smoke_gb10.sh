#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-robopro:gb10}"
gpu_args=(--gpus all)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_args=(--gpus "device=${CUDA_VISIBLE_DEVICES}")
fi

graphics_args=()
if [[ -e /dev/dri ]]; then
  graphics_args+=(--device /dev/dri)
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

docker run --rm \
  "${gpu_args[@]}" \
  "${graphics_args[@]}" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
  "$IMAGE" \
  python -c "import platform, torch, sapien, mplib; scene = sapien.Scene([sapien.physx.PhysxCpuSystem()]); pose = mplib.Pose(); print(platform.machine(), torch.cuda.is_available(), torch.cuda.get_device_name(0), sapien.__version__, mplib.__version__, type(scene).__name__, pose)"

if [[ "${TEST_RENDER:-0}" == "1" ]]; then
  docker run --rm \
    "${gpu_args[@]}" \
    "${graphics_args[@]}" \
    --env NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    "$IMAGE" \
    python -c "import sapien; print(type(sapien.Scene()).__name__)"
fi
