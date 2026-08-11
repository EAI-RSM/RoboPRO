#!/usr/bin/env python3
"""Load the RoboPRO pi0.5 checkpoint and execute one synthetic inference."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="pi05_robopro_top_cam_jax")
    parser.add_argument("--model-name", default="robopro_jax")
    parser.add_argument("--checkpoint-id", default="30000")
    args = parser.parse_args()

    robotwin_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "customized_robotwin")
    sys.path.insert(0, robotwin_root)
    os.chdir(robotwin_root)

    from policy.pi05.pi_model import PI0

    model = PI0(args.train_config, args.model_name, args.checkpoint_id, pi0_step=50)
    model.set_language("put the sauce can in the basket")
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(14, dtype=np.float32)
    model.update_observation_window([image, image, image], state)
    actions = np.asarray(model.get_action())
    if actions.shape != (50, 14):
        raise RuntimeError(f"unexpected action shape: {actions.shape}; expected (50, 14)")
    if not np.isfinite(actions).all():
        raise RuntimeError("inference returned non-finite actions")
    print(
        "[inference-smoke] success "
        f"shape={actions.shape} dtype={actions.dtype} "
        f"min={actions.min():.6f} max={actions.max():.6f}"
    )


if __name__ == "__main__":
    main()
