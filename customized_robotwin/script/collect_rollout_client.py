"""
collect_rollout_client.py — dedicated policy rollout data collection.

Runs in the RoboPRO conda env (client side). Connects to a model server
(pi05/.venv). For each episode: finds a valid scene via expert check, rolls
out the policy, saves per-frame observations + joint states + labels to HDF5.

Usage (via shell script):
    bash policy/pi05/collect_rollout.sh <task> <task_config> <train_config> \
        <model_name> <checkpoint_id> <seed> <server_gpu>[:<client_gpu>]

Env vars:
    COLLECT_NUM         — episodes to collect (default 100)
    COLLECT_START_SEED  — starting seed (default 100000*(1+seed))
"""

import sys
import os

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

from script.bench_script.setup_paths import setup_paths
setup_paths()

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import socket
import json
import subprocess
import h5py
import cv2
import time
import traceback
import yaml
import argparse
import importlib
import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from generate_episode_instructions import generate_episode_descriptions


# ── Numpy ↔ JSON serialization (needed for model server comms) ─────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            dtype = str(obj.dtype) if obj.dtype not in (np.float32, np.float64, np.int32, np.int64) else str(obj.dtype)
            return {'__numpy_array__': True, 'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                    'dtype': dtype, 'shape': obj.shape}
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    def object_hook(dct):
        if '__numpy_array__' in dct:
            return np.frombuffer(base64.b64decode(dct['data']), dtype=dct['dtype']).reshape(dct['shape'])
        return dct
    return json.loads(json_str, object_hook=object_hook)


# ── ModelClient (TCP proxy to model server) ────────────────────────────────

class ModelClient:
    _MIRRORED_DEFAULTS = {"observation_window": None}

    def __init__(self, host='localhost', port=9999, timeout=30, **mirrored):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        for k, v in self._MIRRORED_DEFAULTS.items():
            object.__setattr__(self, k, v)
        for k, v in mirrored.items():
            object.__setattr__(self, k, v)
        self._connect()

    def _connect(self):
        for attempt in range(1000):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                if self.sock: self.sock.close()
                print(f"⚠️ Connection attempt {attempt+1} failed: {e}. Retrying in 5s...")
                time.sleep(5)
        raise ConnectionError("Failed to connect to model server after 1000 attempts")

    def _send_recv(self, data):
        json_data = numpy_to_json(data).encode('utf-8')
        self.sock.sendall(len(json_data).to_bytes(4, 'big'))
        self.sock.sendall(json_data)
        return self._recv_response()

    def _recv_response(self):
        size = int.from_bytes(self.sock.recv(4), 'big')
        chunks, received = [], 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk: raise ConnectionError("Incomplete response")
            chunks.append(chunk); received += len(chunk)
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def call(self, func_name=None, obs=None):
        return self._send_recv({"cmd": func_name, "obs": obs})['res']

    def __getattr__(self, name):
        if name.startswith("_") or name in {"host", "port", "timeout", "sock"}:
            raise AttributeError(name)
        def _proxy(*args, **kwargs):
            obs = None if not args else (args[0] if len(args) == 1 else {"_args": list(args)})
            result = self.call(func_name=name, obs=obs)
            if name in ("reset_obsrvationwindows", "reset_model"):
                object.__setattr__(self, "observation_window", None)
            elif name == "update_observation_window":
                object.__setattr__(self, "observation_window", True)
            return result
        return _proxy

    def close(self):
        if self.sock:
            try: self.sock.close()
            except: pass
            finally: self.sock = None; print("🔌 Connection closed")


# ── Helpers ────────────────────────────────────────────────────────────────

def class_decorator(task_name):
    envs_module = None
    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        for mod_path in [f"bench_envs.{task_name}", f"bench_envs.study.{task_name}",
                         f"bench_envs.office.{task_name}", f"bench_envs.kitchenl.{task_name}",
                         f"bench_envs.kitchens.{task_name}"]:
            try:
                envs_module = importlib.import_module(mod_path); break
            except ModuleNotFoundError:
                continue
    if envs_module is None:
        envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        return getattr(envs_module, task_name)()
    except AttributeError:
        raise SystemExit("No such task")


def eval_function_decorator(policy_name, model_name):
    return getattr(importlib.import_module(policy_name), model_name)


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../task_config/_camera_config.yml")
    with open(camera_config_path, "r") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg[camera_type]


# ── HDF5 writer ────────────────────────────────────────────────────────────

# Episode-invariant provenance stamped into every hdf5's attrs (populated in
# main(): collector id, policy/ckpt, collision-metric + planner regime).
_PROVENANCE = {}


def _flush_step_flags(TASK_ENV):
    """Read-and-reset the per-window contact/collision flags accumulated by
    Bench_base_task.check_collisions() over every scene.step() since the
    previous saved frame (same semantics as Base_Task._take_picture)."""
    contact = np.uint8(getattr(TASK_ENV, '_win_contact', False))
    collision = np.uint8(getattr(TASK_ENV, '_win_collision', False))
    TASK_ENV._win_contact = False
    TASK_ENV._win_collision = False
    return contact, collision


def save_episode_hdf5(buffer, save_dir, ep_idx, instruction, success, collision):
    """Write one rollout episode to HDF5.

    Structure:
        obs/{cam}/rgb  — JPEG-encoded frames, shape (T,)
        state          — float32 (T, 14)
        contact        — uint8 (T,)  any robot<->world PhysX contact in frame window
        collision      — uint8 (T,)  filtered collision event in frame window
        attrs: instruction, episode, success, collision (+ _PROVENANCE)
    """
    hdf5_path = Path(save_dir) / f"episode_{ep_idx}.hdf5"
    cam_names = list(buffer[0]['obs'].keys())
    cam_images = {cam: [] for cam in cam_names}
    states = []
    for frame in buffer:
        for cam in cam_names:
            cam_images[cam].append(frame['obs'][cam])
        states.append(frame['state'])

    with h5py.File(hdf5_path, 'w') as f:
        f.attrs['instruction'] = np.bytes_(instruction)
        f.attrs['episode'] = ep_idx
        f.attrs['success'] = success
        f.attrs['collision'] = collision
        for k, v in _PROVENANCE.items():
            f.attrs[k] = v

        obs_grp = f.create_group('obs')
        for cam in cam_names:
            encoded, max_len = [], 0
            for img in cam_images[cam]:
                _, buf = cv2.imencode('.jpg', img)
                data = buf.tobytes()
                encoded.append(data)
                max_len = max(max_len, len(data))
            padded = [d.ljust(max_len, b'\0') for d in encoded]
            obs_grp.create_dataset(f'{cam}/rgb', data=padded, dtype=f'S{max_len}')

        f.create_dataset('state', data=np.array(states, dtype=np.float32))
        # per-timestep engine flags, 1:1 with state rows (0 for frames collected
        # before the flags existed or with enable_collision_metrics off)
        f.create_dataset('contact', data=np.array(
            [frame.get('contact', 0) for frame in buffer], dtype=np.uint8))
        f.create_dataset('collision', data=np.array(
            [frame.get('collision', 0) for frame in buffer], dtype=np.uint8))

    return hdf5_path


# ── Scene state snapshot / restore ────────────────────────────────────────

def collect_rollouts(task_name, TASK_ENV, args, model, st_seed,
                     collect_num=100, save_dir=None, instruction_type="seen",
                     video_size=None, save_seed_fn=None):
    save_dir = Path(save_dir)
    data_dir = save_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    eval_func = eval_function_decorator(args["policy_name"], "eval")
    args["eval_mode"] = True
    clear_cache_freq = args["clear_cache_freq"]

    from policy.pi05.deploy_policy import encode_obs as _encode_obs

    # Inject Gaussian noise into every action to collect diverse near-policy trajectories.
    _noise_var = float(args.get("action_noise_var", os.environ.get("ACTION_NOISE_VAR", 0.001)))
    _noise_std = np.sqrt(_noise_var)
    _orig_take_action = TASK_ENV.take_action
    def _noisy_take_action(action):
        noisy = np.asarray(action, dtype=np.float32).copy()
        # Arm joints only: indices 0-5 (left arm) and 7-12 (right arm)
        noisy[[*range(6), *range(7, 13)]] += np.random.normal(0, _noise_std, size=12).astype(np.float32)
        result = _orig_take_action(noisy)
        # Record one frame per policy action (same cadence as branching/stitched
        # modes); per-timestep flags flushed over this action's scene.step()s.
        if getattr(TASK_ENV, '_contrastive_buffer', None) is not None:
            _obs = TASK_ENV.get_obs()
            _, _state = _encode_obs(_obs)
            _c, _k = _flush_step_flags(TASK_ENV)
            TASK_ENV._contrastive_buffer.append({
                'obs':   {cam: _obs['observation'][cam]['rgb'].copy()
                          for cam in ('head_camera', 'right_camera', 'left_camera')
                          if cam in _obs['observation']},
                'state': _state.copy(),
                'contact': _c,
                'collision': _k,
            })
        return result
    TASK_ENV.take_action = _noisy_take_action
    print(f"\033[33mAction noise: N(0, {_noise_std:.4f}) (var={_noise_std**2:.4f})\033[0m")

    now_seed = st_seed
    ep_idx = 0
    succ_count = 0
    collision_count = 0
    episode_log = []
    fixed_seed = bool(os.environ.get("COLLECT_FIXED_SEED"))

    print(f"\033[34mTask: {task_name}  |  Collecting {collect_num} episodes"
          f"{'  |  fixed seed (no expert check)' if fixed_seed else ''}\033[0m")

    while ep_idx < collect_num:
        args["render_freq"] = 0

        if not fixed_seed:
            # ── Expert check: find a valid, solvable scene ───────────────
            try:
                TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                if episode_info is None:
                    episode_info = getattr(TASK_ENV, "info", {"info": {}})
                TASK_ENV.close_env()
            except UnStableError:
                TASK_ENV.close_env()
                now_seed += 1
                continue
            except Exception:
                TASK_ENV.close_env()
                now_seed += 1
                continue

            if not (TASK_ENV.plan_success and TASK_ENV.check_success()):
                now_seed += 1
                continue

        # ── Policy rollout setup ─────────────────────────────────────────
        TASK_ENV.setup_demo(now_ep_num=ep_idx, seed=now_seed, is_test=True, **args)
        if fixed_seed:
            episode_info = getattr(TASK_ENV, "info", {"info": {}})

        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, collect_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        if video_size is not None and TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-f", "rawvideo", "-pixel_format", "rgb24",
                 "-video_size", video_size, "-framerate", "10",
                 "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                 f"{TASK_ENV.eval_video_path}/episode{ep_idx}.mp4"],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        # ── Roll out policy, populate buffer ─────────────────────────────
        model.reset_model()
        TASK_ENV._contrastive_buffer = []
        # fresh per-timestep flag window for this episode
        TASK_ENV._win_contact = False
        TASK_ENV._win_collision = False

        obs0 = TASK_ENV.get_obs()
        _, state0 = _encode_obs(obs0)
        TASK_ENV._contrastive_buffer.append({
            'obs': {cam: obs0['observation'][cam]['rgb'].copy()
                    for cam in ('head_camera', 'right_camera', 'left_camera')
                    if cam in obs0['observation']},
            'state': state0.copy(),
            'contact': np.uint8(0),
            'collision': np.uint8(0),
        })

        succ = False
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break

        if video_size is not None and TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        # ── Collision check ──────────────────────────────────────────────
        is_collision = False
        col_metrics = {}
        if hasattr(TASK_ENV, 'get_collision_metrics'):
            col_metrics = TASK_ENV.get_collision_metrics()
            is_collision = col_metrics.get("is_collision", False)
            if is_collision:
                collision_count += 1

        # ── Save HDF5 ────────────────────────────────────────────────────
        n_frames = len(TASK_ENV._contrastive_buffer)
        if n_frames >= 2:
            hdf5_path = save_episode_hdf5(
                TASK_ENV._contrastive_buffer, data_dir,
                ep_idx, instruction, succ, is_collision,
            )
        TASK_ENV._contrastive_buffer = None

        if succ:
            succ_count += 1
            print(f"  ep{ep_idx:04d} seed={now_seed} \033[92mSuccess\033[0m  frames={n_frames}")
        else:
            col_detail = ""
            if is_collision:
                parts = []
                if col_metrics.get("robot_to_furniture_names"):
                    parts.append("furn=" + ",".join(col_metrics["robot_to_furniture_names"]))
                if col_metrics.get("robot_to_static_object_names"):
                    parts.append("obj=" + ",".join(col_metrics["robot_to_static_object_names"]))
                if col_metrics.get("target_to_static_object_names"):
                    parts.append("tgt=" + ",".join(col_metrics["target_to_static_object_names"]))
                col_detail = f"  \033[91mCollision: {' | '.join(parts)}\033[0m"
            print(f"  ep{ep_idx:04d} seed={now_seed} \033[91mFail\033[0m      frames={n_frames}{col_detail}")

        episode_log.append({
            "episode": ep_idx, "seed": now_seed, "n_frames": n_frames,
            "success": succ, "collision": is_collision,
            "instruction": instruction,
            **{k: col_metrics.get(k, v) for k, v in {
                "robot_to_furniture": 0, "robot_to_static_object": 0,
                "target_to_static_object": 0,
                "robot_to_furniture_names": [], "robot_to_static_object_names": [],
                "target_to_static_object_names": [],
            }.items()},
        })

        ep_idx += 1
        TASK_ENV.close_env(clear_cache=(ep_idx % clear_cache_freq == 0))
        if not fixed_seed:
            now_seed += 1
        if save_seed_fn:
            save_seed_fn(now_seed)

        print(f"  → {ep_idx}/{collect_num} collected | "
              f"SR \033[95m{succ_count/ep_idx*100:.1f}%\033[0m | "
              f"CR \033[95m{collision_count/ep_idx*100:.1f}%\033[0m")

    with open(save_dir / "collect_summary.json", "w") as f:
        json.dump(episode_log, f, indent=2)

    return episode_log


# ── Branching rollout collection ───────────────────────────────────────────

def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args.get("instruction_type", "seen")
    port = usr_args["port"]
    collect_num = int(os.environ.get("COLLECT_NUM", usr_args.get("collect_num", 100)))

    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        cfg_path = f"{os.getenv('BENCH_ROOT')}/bench_task_config/{task_config}.yml"
    else:
        cfg_path = f"./task_config/{task_config}.yml"
    with open(cfg_path, "r") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args.update({"task_name": task_name, "task_config": task_config, "ckpt_setting": ckpt_setting})

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(et):
        path = _embodiment_types[et]["file_path"]
        if path is None: raise ValueError(f"missing embodiment file for {et}")
        return path

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment must have 1 or 3 entries")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # Merge deploy_policy.yml collect params into args (task config takes priority)
    for k in ("collect_fixed_seed", "action_noise_var", "collect_num"):
        if k in usr_args and k not in args:
            args[k] = usr_args[k]

    save_dir = Path(f"rollout_data/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Episode provenance stamped into every hdf5 (mirrors collect_data.py's
    # _stamp_provenance_attrs): collector identity + collision/planner regime.
    _peo = args.get("planner_exclude_obstacles", None)
    _PROVENANCE.update({
        # human-readable producer label, e.g. "pi05" — matches the CuRobo
        # collector's "curobo_collision_{aware,unaware}" convention
        "generator": np.bytes_(str(policy_name)),
        "collector": np.bytes_("policy_rollout"),
        "policy_name": np.bytes_(str(policy_name)),
        "ckpt_setting": np.bytes_(str(ckpt_setting)),
        "enable_collision_metrics": bool(args.get("enable_collision_metrics", False)),
        "planner_exclude_obstacles": -1 if _peo is None else int(bool(_peo)),
    })

    video_size = None
    if args.get("eval_video_log"):
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = f"{camera_config['w']}x{camera_config['h']}"
        args["eval_video_save_dir"] = save_dir / "videos"
        (save_dir / "videos").mkdir(parents=True, exist_ok=True)

    TASK_ENV = class_decorator(task_name)
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args.get("seed", 0)
    # Persistent seed state: resume from where the last run left off so
    # repeated invocations don't re-collect the same scenes.
    _seed_state_dir  = Path(f"rollout_data/{task_name}/{policy_name}/{task_config}/{ckpt_setting}")
    _seed_state_file = _seed_state_dir / "seed_state.txt"
    _seed_state_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("COLLECT_START_SEED"):
        st_seed = int(os.environ["COLLECT_START_SEED"])
    elif _seed_state_file.exists():
        st_seed = int(_seed_state_file.read_text().strip())
        print(f"\033[33mResuming from seed {st_seed} (loaded from {_seed_state_file})\033[0m")
    else:
        st_seed = 100000 + (1 + seed)

    model_mirrored = {}
    if "pi0_step" in usr_args:
        model_mirrored["pi0_step"] = usr_args["pi0_step"]
    model = ModelClient(port=port, **model_mirrored)

    def _save_seed(seed):
        _seed_state_file.write_text(str(seed))

    episode_log = collect_rollouts(
        task_name, TASK_ENV, args, model, st_seed,
        collect_num=collect_num,
        save_dir=str(save_dir),
        instruction_type=instruction_type,
        video_size=video_size,
        save_seed_fn=_save_seed,
    )

    succ_count = sum(1 for e in episode_log if e["success"])
    coll_count = sum(1 for e in episode_log if e["collision"])
    n = len(episode_log)
    print(f"\nDone. {n} episodes saved to {save_dir}")
    print(f"SR: {succ_count/n*100:.1f}%  |  CR: {coll_count/n*100:.1f}%")


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["port"] = args.port

    if args.overrides:
        pairs = args.overrides
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            try: value = eval(pairs[i + 1])
            except: value = pairs[i + 1]
            config[key] = value

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()
    main(usr_args)
