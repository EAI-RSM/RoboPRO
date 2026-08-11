"""RoboPRO-owned OpenPI configuration overlays for pi0.5 evaluation."""

from __future__ import annotations

import os

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config
from openpi.training import optimizer
from openpi.training import weight_loaders

ROBOPRO_CONFIG_NAME = "pi05_robopro_top_cam_jax"


def build_robopro_config() -> config.TrainConfig:
    """Build the exact input/action contract used by robopro_jax_30000."""
    return config.TrainConfig(
        name=ROBOPRO_CONFIG_NAME,
        # JAX 0.5's CUDA 12.6 backend cannot lower the checkpoint's BF16
        # inference graph on GB10/CC 12.1 (LLVM "Unsupported rounding mode").
        # Float32 uses the same architecture and checkpoint values while
        # avoiding that unsupported compiler path. Keep this runtime-selectable
        # so a future Blackwell-capable JAX stack can return to bfloat16.
        model=pi0_config.Pi0Config(
            pi05=True,
            dtype=os.environ.get("PI05_COMPUTE_DTYPE", "bfloat16"),
        ),
        data=config.LeRobotAlohaDataConfig(
            repo_id="roboreal_lerobot",
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.countertop",
                                "cam_left_wrist": "observation.images.left",
                                "cam_right_wrist": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=optimizer.CosineDecaySchedule(decay_steps=30_000),
        num_train_steps=30_000,
        batch_size=192,
        num_workers=16,
        fsdp_devices=1,
    )


def register_robopro_config() -> config.TrainConfig:
    """Idempotently register the RoboPRO config with the installed OpenPI."""
    existing = config._CONFIGS_DICT.get(ROBOPRO_CONFIG_NAME)
    if existing is not None:
        return existing
    robopro_config = build_robopro_config()
    config._CONFIGS.append(robopro_config)
    config._CONFIGS_DICT[ROBOPRO_CONFIG_NAME] = robopro_config
    return robopro_config
