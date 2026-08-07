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
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=assets_id,
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

    # ------------------- controlled / recorded flow noise --------------------
    # By default pi05 draws its flow-matching noise from Policy's internal jax
    # key, so a rollout cannot be reproduced or paired with the noise that
    # produced it. Setting PI05_NOISE_SEED makes the noise explicit: episode key
    # K draws from np.random.default_rng(PI05_NOISE_SEED + crc32(K)), and with
    # PI05_NOISE_DIR the noise plus the action chunk it produced are written to
    # <dir>/<K>_noise.npz. Both are no-ops when the env vars are unset, so this
    # is inert for every existing caller.
    def set_noise_episode(self, ep_key):
        """Begin a new episode's noise stream.

        ep_key is an int index (-> "ep00007_noise.npz") or a scene-qualified
        string such as "d14_seed1_ep003" (-> "<key>_noise.npz"), so one
        long-lived model server can serve many scenes without collisions.
        crc32 rather than hash(): hash() is salted per process and would not
        reproduce across runs.
        """
        import zlib
        is_int = isinstance(ep_key, (int, float)) or str(ep_key).lstrip("-").isdigit()
        self._noise_key = f"ep{int(ep_key):05d}" if is_int else str(ep_key)
        self._noise_buf = []
        self._action_buf = []
        base = os.environ.get("PI05_NOISE_SEED")
        if base is None:
            self._noise_rng = None
            return "disabled"
        off = int(ep_key) if is_int else zlib.crc32(str(ep_key).encode())
        self._noise_seed = int(base) + int(off)
        self._noise_rng = np.random.default_rng(self._noise_seed)
        return self._noise_key

    def _draw_noise(self):
        if getattr(self, "_noise_rng", None) is None:
            return None
        m = self.policy._model
        return self._noise_rng.standard_normal(
            (m.action_horizon, m.action_dim)).astype(np.float32)

    def _save_noise_record(self, noise, actions):
        d = os.environ.get("PI05_NOISE_DIR")
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        self._noise_buf.append(noise)
        self._action_buf.append(np.asarray(actions, dtype=np.float32))
        # rewritten after every inference (~6 KB each) so a crashed episode
        # still leaves a usable file
        np.savez_compressed(
            os.path.join(d, f"{self._noise_key}_noise.npz"),
            noise=np.stack(self._noise_buf),
            actions=np.stack(self._action_buf),
            ep_key=self._noise_key,
            noise_seed=self._noise_seed,
        )

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        noise = self._draw_noise()
        if noise is None:                       # unchanged default path
            return self.policy.infer(self.observation_window)["actions"]
        result = self.policy.infer(self.observation_window, noise=noise)
        self._save_noise_record(noise, result["actions"])
        return result["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")

    # Server proxies methods by attribute lookup; eval client calls
    # `reset_model` whereas deploy_policy.py keeps the legacy typo. Alias.
    def reset_model(self):
        self.reset_obsrvationwindows()
