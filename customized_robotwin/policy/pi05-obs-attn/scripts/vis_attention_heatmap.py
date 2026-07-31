#!/usr/bin/env python
"""Visualize obstacle-guided attention: base_0_rgb with the GT obstacle heatmap and
the model's predicted (query-head) attention map overlaid, side by side.

JAX/LeRobot port of SafeVLA's vis_seg_heatmap / vis_weighted_heatmap. Saves one PNG
per sample. Uses the same renderer as W&B train visualizations.

Usage (from the pi05-obs-attn dir, in the project env):
    python scripts/vis_attention_heatmap.py \
        --config pi05_robopro_obstacle \
        --params /path/to/checkpoints/<exp>/<step>/params \
        --num-samples 8 --role obstacle --out ./attn_vis

    # Also export GT vs pred EE Cartesian figure for the first batch:
    python scripts/vis_attention_heatmap.py ... --actions
"""

from __future__ import annotations

import os

import jax
import numpy as np
import tyro
from PIL import Image

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.viz_logging as _viz_logging


def main(
    config: str,
    params: str,
    *,
    num_samples: int = 8,
    role: str = "obstacle",
    seed: int = 0,
    out: str = "./attn_vis",
    actions: bool = False,
) -> None:
    train_config = _config.get_config(config)
    model = train_config.model.load(_model.restore_params(os.path.expanduser(params)))
    if model.obstacle_attention is None or not model.obstacle_attention.enabled:
        raise ValueError("obstacle_attention is not enabled on this config")
    role_names = model.obstacle_attention.role_names
    if role not in role_names:
        raise ValueError(f"role must be enabled by the config; got {role!r}, enabled={role_names}")

    out_dir = os.path.expanduser(out)
    os.makedirs(out_dir, exist_ok=True)

    data_loader = _data_loader.create_data_loader(train_config, shuffle=False, num_batches=None)
    observation, gt_actions = next(iter(data_loader))

    rng = jax.random.key(seed)
    preds = model.predict_attention(rng, observation, train=False)
    panels = _viz_logging.render_attention_panels(
        observation,
        preds,
        (role,),
        num_samples=num_samples,
        grid=model.obstacle_attention.attn_grid_h,
    )
    for i, img in enumerate(panels[role]):
        path = os.path.join(out_dir, f"attn_{role}_{i:03d}.png")
        Image.fromarray(img).save(path)
        print(f"wrote {path}")

    if actions:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        adapt_to_pi = getattr(train_config.data, "adapt_to_pi", True)
        use_delta = getattr(train_config.data, "use_delta_joint_actions", True)
        n = min(num_samples, int(np.asarray(observation.state).shape[0]))
        observation_n, gt_actions_n = _viz_logging._slice_batch(observation, gt_actions, n)
        rng_act = jax.random.fold_in(rng, 1)
        pred_actions = model.sample_actions(rng_act, observation_n)
        pred_actions_np = np.asarray(jax.device_get(pred_actions))
        gt_actions_np = np.asarray(jax.device_get(gt_actions_n))
        state_np = np.asarray(jax.device_get(observation_n.state))
        urdf = _viz_logging.load_urdf()
        for i in range(n):
            gt_l, gt_r = _viz_logging.actions_to_ee_xyz(
                state_np[i],
                gt_actions_np[i],
                adapt_to_pi=adapt_to_pi,
                use_delta=use_delta,
                norm_stats=data_config.norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                urdf=urdf,
            )
            pred_l, pred_r = _viz_logging.actions_to_ee_xyz(
                state_np[i],
                pred_actions_np[i],
                adapt_to_pi=adapt_to_pi,
                use_delta=use_delta,
                norm_stats=data_config.norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                urdf=urdf,
            )
            img = _viz_logging.render_ee_trajectories(gt_l, gt_r, pred_l, pred_r, sample_idx=i)
            path = os.path.join(out_dir, f"ee_cartesian_{i:03d}.png")
            Image.fromarray(img).save(path)
            print(f"wrote {path}")


if __name__ == "__main__":
    tyro.cli(main)
