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

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        result = self.policy.infer(self.observation_window)
        if "embeds" in result and os.environ.get("PI05_EMBEDS_DIR"):
            self._save_embed_record(result)
        if "cand_actions" in result and os.environ.get("PI05_SAFE_SELECT") == "1":
            return self._safe_select(result)
        if "cand_actions" in result and os.environ.get("PI05_PROGRESS_SELECT") == "1":
            return self._progress_select(result)
        return result["actions"]

    def _q_client_score(self, imgs, state, prompt, actions):
        import socket as _sock, struct as _st, pickle as _pk
        if not hasattr(self, "_q_conn"):
            c = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
            c.connect(os.environ["PI05_Q_SOCK"])
            self._q_conn = c
        req = _pk.dumps({"imgs": imgs, "state": state, "prompt": prompt,
                         "actions": actions}, protocol=_pk.HIGHEST_PROTOCOL)
        self._q_conn.sendall(_st.pack(">I", len(req)) + req)
        def _rd(n):
            b = b""
            while len(b) < n:
                ch = self._q_conn.recv(n - len(b))
                if not ch:
                    raise ConnectionError("q-scorer closed")
                b += ch
            return b
        (n,) = _st.unpack(">I", _rd(4))
        return _pk.loads(_rd(n))["q"]

    def _progress_select(self, result):
        import numpy as _np
        ow = self.observation_window
        imgs = {k: _np.asarray(v) for k, v in ow["images"].items()}
        state = _np.asarray(ow["state"], _np.float32).reshape(-1)
        prompt = ow.get("prompt") or "perform the task"
        cands = _np.asarray(result["cand_actions"], _np.float32)   # (N,H,dim)
        q = self._q_client_score(imgs, state, prompt, cands)
        if os.environ.get("PI05_Q_ARGMIN") == "1":
            best = int(_np.argmin(q))                              # collision-Q: min P(collision)
        else:
            best = int(_np.argmax(q))                              # progress-Q: higher = better
        if not hasattr(self, "_progress_logged"):
            print(f"[progress-select] img{list(imgs.values())[0].shape} "
                  f"state{state.shape} cands{cands.shape}", flush=True)
            self._progress_logged = True
        print(f"[progress-select] Q={[round(float(x),3) for x in q]} -> cand {best}", flush=True)
        return cands[best]

    def _safe_select(self, result):
        import numpy as _np
        import torch as _t
        if not hasattr(self, "_safe_lstm"):
            import torch.nn as _nn
            class _H(_nn.Module):
                def __init__(s, d=1024, h=256):
                    super().__init__(); s.l = _nn.LSTM(d, h, batch_first=True); s.f = _nn.Linear(h, 1)
                def forward(s, x):
                    o, _ = s.l(x); return s.f(o).squeeze(-1)
            m = _H()
            m.load_state_dict(_t.load(os.environ["PI05_SAFE_CKPT"], map_location="cpu"))
            m.eval(); self._safe_lstm = m
        cands = result["cand_actions"]                    # (N,H,dim)
        feats = result["cand_embeds"][:, -1].mean(axis=1).astype(_np.float32)  # (N,1024): final denoise, horizon mean
        hist = getattr(self, "_safe_hist", [])
        with _t.no_grad():
            scores = []
            for i in range(len(cands)):
                seq = _np.stack(hist + [feats[i]])[None]  # (1,T,1024)
                logit = self._safe_lstm(_t.from_numpy(seq))[0, -1]
                scores.append(float(_t.sigmoid(logit)))   # higher = predicted SUCCESS
        best = int(_np.argmax(scores))
        self._safe_hist = hist + [feats[best]]
        print(f"[safe-select] scores={[round(s,3) for s in scores]} -> cand {best}", flush=True)
        return cands[best]

    def _save_embed_record(self, result):
        import pickle
        import numpy as _np
        ep = getattr(self, "_embed_ep_idx", 0)
        idx = getattr(self, "_embed_infer_idx", 0)
        d = os.path.join(os.environ["PI05_EMBEDS_DIR"], f"ep{ep:05d}")
        os.makedirs(d, exist_ok=True)
        rec = {
            "pre_velocity": result["embeds"],                 # (n_denoise, horizon, width) fp16
            "actions": _np.asarray(result["actions"]),        # (horizon, action_dim)
            "state": (_np.asarray(result["state"]) if "state" in result else None),  # output transform may drop it
            "infer_idx": idx,
        }
        with open(os.path.join(d, f"infer{idx:04d}_meta.pkl"), "wb") as f:
            pickle.dump(rec, f)
        self._embed_infer_idx = idx + 1

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        self._safe_hist = []
        if getattr(self, "_embed_infer_idx", 0) > 0:
            self._embed_ep_idx = getattr(self, "_embed_ep_idx", 0) + 1
            self._embed_infer_idx = 0
        print("successfully unset obs and language intruction")

    # Server proxies methods by attribute lookup; eval client calls
    # `reset_model` whereas deploy_policy.py keeps the legacy typo. Alias.
    def reset_model(self):
        self.reset_obsrvationwindows()
