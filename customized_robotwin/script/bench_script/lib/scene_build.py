"""Configuration and environment-class construction shared by bench tools."""

import importlib
import os
from pathlib import Path

import yaml
from envs import CONFIGS_PATH

bench_root = Path(os.environ["BENCH_ROOT"])
robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])


def build_cfg(task_name, base_config, seed, dr_overrides, rollout=False, ep_num=0,
              save_path=None, mode=None):
    """Build a setup_demo cfg.

    mode=None preserves the legacy ``rollout`` switch exactly:
    rollout=False (default): a fast t=0 measurement build -- no planning, no saved
    data, single-camera render (measurement_only).
    rollout=True: an expert curobo rollout build -- need_plan=True + save_data=True
    so play_once() plans/executes and frames are captured for merge_pkl_to_hdf5_video()
    (writes <save_path>/video/episode{ep_num}.mp4). Full render (not measurement_only).
    mode="policy": a no-expert VLA build -- eval loop/video enabled, no curobo calls
    and no dataset capture. ``save_path`` is the episode video directory.
    """
    if mode is None:
        mode = "expert" if rollout else "measure"
    elif rollout:
        raise ValueError("pass either rollout=True or mode=..., not both")
    if mode not in {"measure", "expert", "policy"}:
        raise ValueError(f"unknown build mode: {mode!r}")

    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        config_path = bench_root / "bench_task_config" / f"{base_config}.yml"
    else:
        config_path = Path(f"./task_config/{base_config}.yml")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = 0
    cfg["now_ep_num"] = int(ep_num)
    cfg["seed"] = int(seed)
    if mode == "expert":
        cfg["need_plan"] = True        # expert curobo plans + executes the task
        cfg["save_data"] = True        # capture frames so a video can be written
        cfg["measurement_only"] = False
        cfg.setdefault("save_freq", 15)
        if save_path is not None:
            cfg["save_path"] = str(save_path)
    elif mode == "measure":
        cfg["need_plan"] = False       # no planning/rollout for a t=0 measurement
        cfg["save_data"] = False
        cfg["measurement_only"] = True  # t=0 measurement: render only the measured camera
    else:
        cfg["eval_mode"] = True
        cfg["build_planner"] = False
        cfg["need_plan"] = False
        cfg["save_data"] = False
        cfg["measurement_only"] = False
        if save_path is not None:
            cfg["eval_video_save_dir"] = str(save_path)

    cfg.setdefault("domain_randomization", {})
    cfg["domain_randomization"].update(dr_overrides)

    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(name):
        robot_file = _embodiment_types[name]["file_path"]
        if robot_file is None:
            raise SystemExit("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = emb_file(embodiment_type[0])
        cfg["right_robot_file"] = emb_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])
    return cfg


DR_CLEAN = {"cluttered_table": False, "obstacle_density": 0, "clean_background_rate": 0}


def dr_measure(clutter_density):
    """Domain-randomization for the measurement build: optional default (non-curated)
    table clutter at the given density. Density 0 -> clean (occluder only)."""
    if clutter_density and clutter_density > 0:
        return {"cluttered_table": True, "obstacle_density": int(clutter_density),
                "clean_background_rate": 0}
    return dict(DR_CLEAN)


def _extract_task_class(envs_module, task_name):
    """Extract task class from module, handling class names that differ from module name."""
    try:
        return getattr(envs_module, task_name)
    except AttributeError:
        from envs._base_task import Base_Task
        for name in dir(envs_module):
            obj = getattr(envs_module, name)
            if isinstance(obj, type) and issubclass(obj, Base_Task) and obj is not Base_Task:
                return obj
        raise SystemExit(f"No task class found in {envs_module.__name__}")


def get_env_class(task_name, bench_subdir=None):
    """Load task env class from bench_envs, or envs if not in bench_envs."""
    # Known bench_envs subpackages (office, study, etc.)
    BENCH_SUBDIRS = ["office", "study", "kitchenl", "kitchens"]

    if bench_subdir:
        # Explicit subdir: try only bench_envs.{subdir}.{task_name}
        try:
            envs_module = importlib.import_module(f"bench_envs.{bench_subdir}.{task_name}")
            return _extract_task_class(envs_module, task_name)
        except ModuleNotFoundError:
            raise SystemExit(f"Task '{task_name}' not found in bench_envs.{bench_subdir}")

    # Try bench_envs.{task_name} first (flat structure)
    try:
        envs_module = importlib.import_module(f"bench_envs.{task_name}")
        return _extract_task_class(envs_module, task_name)
    except ModuleNotFoundError:
        pass

    # Try bench_envs.{subdir}.{task_name} for each known subdir
    for subdir in BENCH_SUBDIRS:
        try:
            envs_module = importlib.import_module(f"bench_envs.{subdir}.{task_name}")
            return _extract_task_class(envs_module, task_name)
        except ModuleNotFoundError:
            continue

    # Fallback to envs
    try:
        envs_module = importlib.import_module(f"envs.{task_name}")
        return _extract_task_class(envs_module, task_name)
    except ModuleNotFoundError:
        raise SystemExit(f"No task class found for '{task_name}' in bench_envs or envs")


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)
