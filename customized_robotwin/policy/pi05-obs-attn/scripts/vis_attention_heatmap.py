#!/usr/bin/env python
"""Visualize obstacle-guided attention on cluttered scenes (unified multi-role panel).

Saves per-sample scene PNGs (masks | GTs | preds across supervised layers) and
optional layer-animated GIFs. Uses the same renderer as W&B train visualizations.

Usage (from the pi05-obs-attn dir, in the project env):
    python scripts/vis_attention_heatmap.py \
        --config pi05_obs_attn \
        --params /path/to/checkpoints/<exp>/<step>/params \
        --num-samples 8 --out ./attn_vis

    # Also export GT vs pred EE Cartesian figure:
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
    seed: int = 0,
    out: str = "./attn_vis",
    actions: bool = False,
    gif: bool = True,
) -> None:
    train_config = _config.get_config(config)
    model = train_config.model.load(_model.restore_params(os.path.expanduser(params)))
    if model.obstacle_attention is None or not model.obstacle_attention.enabled:
        raise ValueError("obstacle_attention is not enabled on this config")
    oa = model.obstacle_attention

    out_dir = os.path.expanduser(out)
    os.makedirs(out_dir, exist_ok=True)

    data_loader = _data_loader.create_data_loader(train_config, shuffle=False, num_batches=None)
    observation, gt_actions = next(iter(data_loader))

    clutter_idxs = _viz_logging.select_cluttered_indices(
        observation, num_samples, role_names=oa.role_names
    )
    if not clutter_idxs:
        raise RuntimeError(
            "No cluttered samples in the first batch "
            f"(need non-empty beta, obstacle, and masks for enabled roles {oa.role_names})."
        )
    observation, gt_actions = _viz_logging._take_indices(observation, gt_actions, clutter_idxs)
    n = len(clutter_idxs)
    print(f"selected cluttered indices (from first batch): {clutter_idxs}")

    rng = jax.random.key(seed)
    preds = model.predict_attention(rng, observation, train=False)
    panels = _viz_logging.render_attention_scene_panels(
        observation,
        preds,
        oa.role_names,
        grid=oa.attn_grid_h,
        supervised_layers=oa.supervised_layers,
    )
    for i, img in enumerate(panels):
        path = os.path.join(out_dir, f"attn_scene_{i:03d}.png")
        Image.fromarray(img).save(path)
        print(f"wrote {path}")

    if gif:
        role_probs = np.asarray(jax.device_get(preds["role_probs"]))
        rgb = np.asarray(jax.device_get(observation.images["base_0_rgb"]))
        rgb = np.clip((rgb + 1.0) / 2.0, 0.0, 1.0)
        layer_ids = list(oa.supervised_layers)
        for i in range(n):
            gt_key = "gt_obstacle" if "obstacle" in oa.role_names else f"gt_{oa.role_names[0]}"
            gt_i = np.asarray(jax.device_get(preds[gt_key]))[i] if gt_key in preds else None
            frames = _viz_logging.render_attention_gif_frames(
                rgb[i],
                role_probs[:, i],
                role_names=oa.role_names,
                layer_ids=layer_ids,
                grid=oa.attn_grid_h,
                gt_dists=gt_i,
            )
            path = os.path.join(out_dir, f"attn_live_{i:03d}.gif")
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=500, loop=0)
            print(f"wrote {path}")

    if actions:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        adapt_to_pi = getattr(train_config.data, "adapt_to_pi", True)
        use_delta = getattr(train_config.data, "use_delta_joint_actions", True)
        rng_act = jax.random.fold_in(rng, 1)
        pred_actions = model.sample_actions(rng_act, observation)
        pred_actions_np = np.asarray(jax.device_get(pred_actions))
        gt_actions_np = np.asarray(jax.device_get(gt_actions))
        state_np = np.asarray(jax.device_get(observation.state))
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
