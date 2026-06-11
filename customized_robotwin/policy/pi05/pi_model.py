#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import os
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        specified_path = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        # asset_id is the path (relative to assets/) of the dir holding
        # norm_stats.json. It can be nested when repo_id contains a "/"
        # (e.g. "local/kitchenl_d15_combined"), so walk to find it rather
        # than assuming a single top-level entry.
        asset_id = None
        if os.path.isdir(specified_path):
            for root, _dirs, files in os.walk(specified_path):
                if "norm_stats.json" in files:
                    asset_id = os.path.relpath(root, specified_path)
                    break

        config = _config.get_config(self.train_config_name)

        policy_kwargs = {}
        if asset_id is not None:
            policy_kwargs["robotwin_repo_id"] = asset_id
        else:
            # Checkpoint saved without norm stats (trained unnormalized) —
            # serve with empty stats so inference matches training.
            policy_kwargs["norm_stats"] = {}
            print("no norm stats in checkpoint assets; serving unnormalized")

        # Advantage-token models take an integer conditioning index at inference:
        # 0=negative, 1=positive, 2=null/unconditional. Default to positive.
        if getattr(config.model, "advantage_token", False):
            indicator = int(os.environ.get("ADV_INDICATOR", "1"))
            policy_kwargs["sample_kwargs"] = {
                "advantage_indicator": np.array([indicator], dtype=np.int32)
            }
            print(f"advantage_token model: conditioning with indicator={indicator}")

        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            **policy_kwargs,
            )
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

    # Server proxies methods by attribute lookup; eval client calls
    # `reset_model` whereas deploy_policy.py keeps the legacy typo. Alias.
    def reset_model(self):
        self.reset_obsrvationwindows()

    def seed_rng(self, seed):
        """Reset the flow-matching sampling RNG.

        Called by the eval client before each rollout so paired comparisons
        (guided vs unguided on the same env seed) draw IDENTICAL noise — the
        policy's own sampling stochasticity otherwise differs between the two
        rollouts and masquerades as a guidance effect.
        """
        import jax
        self.policy._rng = jax.random.key(int(seed))
        return True
