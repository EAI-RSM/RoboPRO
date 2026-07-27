#!/usr/bin/env python
"""Visualize obstacle-guided attention: base_0_rgb with the GT obstacle heatmap and
the model's predicted (query-head) attention map overlaid, side by side.

JAX/LeRobot port of SafeVLA's vis_seg_heatmap / vis_weighted_heatmap. Saves one PNG
per sample.

Usage (from the pi05-obs-attn dir, in the project env):
    python scripts/vis_attention_heatmap.py \
        --config pi05_robopro_obstacle \
        --params /path/to/checkpoints/<exp>/<step>/params \
        --num-samples 8 --role obstacle --out ./attn_vis
"""

from __future__ import annotations

import os

import jax
import matplotlib.pyplot as plt
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _grid_to_image(dist_1d: np.ndarray, grid: int, size: int = 224) -> np.ndarray:
    """[N] distribution -> [size, size] nearest-upsampled, max-normalized for display."""
    hm = dist_1d.reshape(grid, grid)
    hm = np.kron(hm, np.ones((size // grid, size // grid)))
    m = hm.max()
    return hm / m if m > 0 else hm


def main(
    config: str,
    params: str,
    *,
    num_samples: int = 8,
    role: str = "obstacle",
    seed: int = 0,
    out: str = "./attn_vis",
) -> None:
    train_config = _config.get_config(config)
    model = train_config.model.load(_model.restore_params(os.path.expanduser(params)))
    if model.obstacle_attention is None or not model.obstacle_attention.enabled:
        raise ValueError("obstacle_attention is not enabled on this config")
    role_names = model.obstacle_attention.role_names
    if role not in role_names:
        raise ValueError(f"role must be enabled by the config; got {role!r}, enabled={role_names}")
    grid = model.obstacle_attention.attn_grid_h

    out_dir = os.path.expanduser(out)
    os.makedirs(out_dir, exist_ok=True)

    data_loader = _data_loader.create_data_loader(train_config, shuffle=False, num_batches=None)
    observation, _ = next(iter(data_loader))

    rng = jax.random.key(seed)
    preds = model.predict_attention(rng, observation, train=False)
    role_probs = np.asarray(preds["role_probs"])  # [S, B, R, N]
    gt_key = f"gt_{role}"
    gt = np.asarray(preds[gt_key]) if gt_key in preds else None
    replacement_mask = getattr(observation, f"{role}_mask")
    replacement_mask = np.asarray(replacement_mask) if replacement_mask is not None else None
    if role == "obstacle" and replacement_mask is not None and observation.beta_mask is not None:
        replacement_mask = replacement_mask * np.asarray(observation.beta_mask)

    # base_0_rgb is [-1, 1] -> [0, 1]; predict_attention preprocesses a copy so the
    # loader observation still holds the (already-resized) image.
    rgb = np.asarray(observation.images["base_0_rgb"])
    rgb = np.clip((rgb + 1.0) / 2.0, 0.0, 1.0)

    layer = role_probs.shape[0] - 1  # last supervised layer
    r = role_names.index(role)
    n = min(num_samples, rgb.shape[0])
    for i in range(n):
        model_map = _grid_to_image(role_probs[layer, i, r], grid)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(rgb[i])
        axes[0].set_title("base_0_rgb")
        axes[1].imshow(rgb[i])
        if replacement_mask is not None:
            axes[1].imshow(replacement_mask[i], cmap="jet", alpha=0.5)
        axes[1].set_title(f"replacement {role} mask")
        axes[2].imshow(rgb[i])
        if gt is not None:
            axes[2].imshow(_grid_to_image(gt[i], grid), cmap="jet", alpha=0.5)
        axes[2].set_title(f"patchified GT {role}")
        axes[3].imshow(rgb[i])
        axes[3].imshow(model_map, cmap="jet", alpha=0.5)
        axes[3].set_title(f"model attn (layer {layer})")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        path = os.path.join(out_dir, f"attn_{role}_{i:03d}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    tyro.cli(main)
