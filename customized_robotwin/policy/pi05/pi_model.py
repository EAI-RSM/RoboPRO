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
from openpi.models import tokenizer as _tokenizer
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

from experiments.graph_conditioned_pi05.graph_serializer import PackedItem, pack_items
from policy.pi05_robopro_config import register_robopro_config
from policy.pi05_robopro_orbax import install_modern_orbax_restore


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        specified_path = f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        entries = sorted(
            entry for entry in os.listdir(specified_path)
            if os.path.isdir(os.path.join(specified_path, entry))
        )
        requested_asset_id = os.environ.get("PI05_ASSET_ID", "trossen")
        if requested_asset_id in entries:
            assets_id = requested_asset_id
        elif len(entries) == 1:
            assets_id = entries[0]
        else:
            raise ValueError(
                f"PI05_ASSET_ID={requested_asset_id!r} is unavailable in "
                f"{specified_path}; choose one of {entries}"
            )
        print(f"using pi05 normalization asset: {assets_id}")

        register_robopro_config()
        config = _config.get_config(self.train_config_name)
        install_modern_orbax_restore(_model, config.model)
        print(f"pi05 compute dtype: {config.model.dtype}")
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi05/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=assets_id,
            )
        self._graph_prompt_tokenizer = _tokenizer.PaligemmaTokenizer(
            config.model.max_token_len
        )
        self._max_prompt_tokens = int(config.model.max_token_len)
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

    def fit_graph_prompt(self, payload):
        """Pack ranked graph nodes/facts with the checkpoint tokenizer."""
        instruction = str(payload["instruction"]).strip()
        state = np.asarray(payload["state"], dtype=np.float32)
        graph_budget = int(payload.get("graph_token_budget", 120))
        if graph_budget < 1:
            raise ValueError("graph_token_budget must be positive")

        items = [
            PackedItem(
                text=str(item["text"]),
                rank=int(item["rank"]),
                section=str(item["section"]),
                provides=str(item.get("provides", "")),
                requires=tuple(str(value) for value in item.get("requires", ())),
                mandatory=bool(item.get("mandatory", False)),
            )
            for item in payload.get("items", [])
        ]
        sentencepiece = self._graph_prompt_tokenizer._tokenizer

        def token_count(text):
            cleaned = text.strip().replace("_", " ").replace("\n", " ")
            return len(sentencepiece.encode(cleaned))

        def full_prompt_token_count(prompt):
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            discretized = np.digitize(
                state, bins=np.linspace(-1, 1, 256 + 1)[:-1]
            ) - 1
            state_text = " ".join(map(str, discretized))
            full_prompt = f"Task: {cleaned}, State: {state_text};\nAction: "
            return len(sentencepiece.encode(full_prompt, add_bos=True))

        packed = pack_items(items, graph_budget, token_counter=token_count)
        while packed.text:
            prompt = instruction + "\n" + packed.text
            if full_prompt_token_count(prompt) <= self._max_prompt_tokens:
                break
            graph_budget -= 1
            if graph_budget < 1:
                packed = pack_items([], 1, token_counter=token_count)
                break
            packed = pack_items(items, graph_budget, token_counter=token_count)

        prompt = instruction + "\n" + packed.text if packed.text else instruction
        selected_nodes = sum(item.section == "node" for item in packed.selected)
        selected_facts = sum(
            item.section in {"goal", "fact"} for item in packed.selected
        )
        return {
            "instruction": prompt,
            "selected_node_count": selected_nodes,
            "selected_fact_count": selected_facts,
            "graph_token_count": packed.token_count,
            "full_prompt_token_count_estimate": full_prompt_token_count(prompt),
            "max_prompt_tokens": self._max_prompt_tokens,
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
