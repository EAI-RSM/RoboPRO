"""Run ONE targeted-collection episode (baseline or perturbed).

Production form of the PoC runner (TARGETED_DATA_COLLECTION.md). Always
executed as a subprocess of collect_targeted_data.py so CuRobo warm-start
state and SAPIEN GPU state never leak across episodes.

Only tasks in bench_envs.targeted.registry are run; everything else is
skipped with a logged reason (incremental task enablement).

Writes into --out-dir:
    episode.json            full label record (schema v1, debug sidecar)
    data/episode0.hdf5      standard bench episode + `targeted_labels` group
    video/episode0.mp4      rollout video

Exit codes: 0 recorded, 2 skip (unsupported task / unstable scene / no
obstacle), 3 discard (integrity guard), 4 runtime error, 5 GPU occupied,
6 scene-pairing config mismatch.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR  # this script lives at the repo root

TRIGGER_BY_PTYPE = {"shift_object": "after_grasp_plan", "shift_target": "immediate",
                    "shift_obstacle": "immediate"}
ACTOR_ATTR_BY_PTYPE = {"shift_object": "target_obj", "shift_target": "des_obj"}

# An episode needs roughly 7-8 GiB (CuRobo + SAPIEN RT renderer); below this we
# abort immediately and loudly instead of OOMing minutes into CuRobo warmup.
# GPU assignment is the user's job (CUDA_VISIBLE_DEVICES) — this code never
# selects or reassigns GPUs.
MIN_FREE_GIB = 12.0

GPU_OCCUPIED_MSG = (
    "FATAL: GPU occupied — {note}. An episode needs ~{need:.0f} GiB free. "
    "GPUs are user-managed: free the GPU or set CUDA_VISIBLE_DEVICES to a free one, "
    "then rerun (recorded episodes resume from cache).")


def probe_visible_gpus():
    """Report free memory on the GPU(s) this process may use. Read-only."""
    import subprocess
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=30)
        rows = [r.split(",") for r in out.strip().splitlines() if r.strip()]
        if preset:
            ids = {i.strip() for i in preset.split(",")}
            rows = [r for r in rows if r[0].strip() in ids] or rows
        free = max(float(r[1]) for r in rows)
        ids_str = ",".join(r[0].strip() for r in rows)
        note = f"gpu(s) {ids_str}, max {free/1024.0:.1f} GiB free"
        if preset:
            note += f" (CUDA_VISIBLE_DEVICES={preset})"
        return free / 1024.0, note
    except Exception as e:  # noqa: BLE001
        return None, f"gpu probe failed: {e}"


def is_cuda_oom(tb: str) -> bool:
    return "OutOfMemoryError" in tb or "CUDA out of memory" in tb


def ensure_env():
    os.environ.setdefault("ROBOTWIN_ROOT", str(REPO_ROOT / "customized_robotwin"))
    os.environ.setdefault("BENCH_ROOT", str(REPO_ROOT / "benchmark"))
    os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    # CuRobo's editable .pth holds an absolute path that may not resolve on
    # other nodes — resolve from this checkout instead.
    curobo_src = str(Path(os.environ["ROBOTWIN_ROOT"]) / "envs" / "curobo" / "src")
    for p in (curobo_src, os.environ["ROBOTWIN_ROOT"], os.environ["BENCH_ROOT"]):
        if p not in sys.path:
            sys.path.insert(0, p)
    # ffmpeg lives in the conda env's bin; jobs invoke this interpreter directly.
    env_bin = str(Path(sys.executable).parent)
    if env_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")


def build_cfg(args, out_dir: Path) -> dict:
    import yaml
    from envs import CONFIGS_PATH

    config_path = Path(os.environ["BENCH_ROOT"]) / "bench_task_config" / f"{args.task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["task_name"] = args.task_name
    cfg["render_freq"] = 0
    cfg["now_ep_num"] = 0
    cfg["seed"] = args.seed
    cfg["need_plan"] = True
    cfg["save_data"] = args.save_data
    cfg["save_path"] = str(out_dir)
    if args.save_freq:
        cfg["save_freq"] = args.save_freq

    # Embodiment wiring — mirrors visualize_task_scene.py.
    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)

    def robot_file(name):
        rf = embodiment_types[name]["file_path"]
        if rf is None:
            raise SystemExit("missing embodiment files")
        return rf

    def embodiment_cfg(rf):
        with open(os.path.join(rf, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = cfg["right_robot_file"] = robot_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = robot_file(embodiment_type[0])
        cfg["right_robot_file"] = robot_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")
    cfg["left_embodiment_config"] = embodiment_cfg(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = embodiment_cfg(cfg["right_robot_file"])
    return cfg


def load_env_class(task_name: str, bench_subdir: str):
    import importlib
    module = importlib.import_module(f"bench_envs.{bench_subdir}.{task_name}")
    return getattr(module, task_name)


def table_plane_camera_axes(env):
    """Head-camera optical axis projected into the table plane (z=0), plus the
    horizontal perpendicular. SAPIEN cameras look along +x of the mount pose."""
    import numpy as np

    forward, source = None, "head_camera"
    try:
        cams = env.cameras
        idx = list(cams.static_camera_name).index("head_camera")
        cam = cams.static_camera_list[idx]
        ent = getattr(cam, "entity", cam)
        rot = ent.get_pose().to_transformation_matrix()[:3, :3]
        forward = rot[:, 0]
    except Exception as e:  # noqa: BLE001 — fall back to the known aloha head-camera forward
        print(f"[targeted] head camera lookup failed ({e}); using constant forward")
        forward, source = np.array([0.0, 0.6, -0.8]), "fallback_constant"

    flat = np.array([forward[0], forward[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(flat))
    if norm < 1e-6:
        flat, norm, source = np.array([0.0, 1.0, 0.0]), 1.0, source + "+degenerate"
    depth_axis = flat / norm
    lateral_axis = np.cross([0.0, 0.0, 1.0], depth_axis)
    return depth_axis, lateral_axis, [float(x) for x in forward], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="put_mouse_on_pad")
    parser.add_argument("--task-config", default="bench_demo_office_clean")
    parser.add_argument("--bench-subdir", default=None,
                        help="default: from the supported-task registry")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", choices=["baseline", "perturbed"], required=True)
    parser.add_argument("--ptype", choices=["shift_object", "shift_target", "shift_obstacle", "none"],
                        default="none")
    parser.add_argument("--params-json", default=None,
                        help="JSON params (sampler.sample_shift_params dict, or "
                             "{'mode':'corridor','corridor_t':..} for shift_obstacle); "
                             "falls back to the config's `targeted:` block, then the sampler")
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-data", dest="save_data", action="store_true", default=True)
    parser.add_argument("--no-save-data", dest="save_data", action="store_false")
    parser.add_argument("--save-freq", type=int, default=None)
    args = parser.parse_args()

    if args.role == "perturbed" and args.ptype == "none":
        parser.error("--role perturbed requires a --ptype")

    ensure_env()
    gpu_free_gib, gpu_note = probe_visible_gpus()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(os.environ["ROBOTWIN_ROOT"])

    import numpy as np
    import robo_negative as targeted  # installed package (the single-file PoC library)
    labels = sampler = registry = targeted
    from robo_negative import EpisodeDiscard, TargetedRuntime

    record = {
        "label_schema_version": labels.LABEL_SCHEMA_VERSION,
        "task_name": args.task_name,
        "task_config": args.task_config,
        "scene_seed": args.seed,
        "executor_type": "scripted_expert",
        "executor_checkpoint": None,
        "pair_id": args.pair_id,
        "pair_role": args.role,
        "perturbation_type": args.ptype if args.role == "perturbed" else None,
        "trigger": TRIGGER_BY_PTYPE.get(args.ptype) if args.role == "perturbed" else None,
        "status": "running",
        "outcome": None,
        "failure_reason": None,
        "sampler": None,
        "commanded": None,
        "achieved": None,
        "camera_axes": None,
        "shift_frame_idx": None,
        "magnitude_bins_cm": [list(b) for b in sampler.MAGNITUDE_BINS_CM],
        "planner_visible_actors": None,
        "contact_log": [],
        "signals": {},
        "instruction": None,
        "gpu": gpu_note,
    }

    def flush(exit_code=None):
        with open(out_dir / "episode.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        if exit_code is not None:
            print(f"[targeted] {args.role} ptype={args.ptype} seed={args.seed} "
                  f"status={record['status']} outcome={record['outcome']}")
            sys.exit(exit_code)

    # ---- supported-task registry gate (incremental enablement) ----
    entry = registry.task_entry(args.task_name)
    if entry is None or (args.role == "perturbed" and args.ptype not in entry["ptypes"]):
        what = (f"task '{args.task_name}'" if entry is None
                else f"ptype '{args.ptype}' for task '{args.task_name}'")
        record["status"] = "skip_unsupported_task"
        record["failure_reason"] = (f"{what} is not in bench_envs.targeted.registry — "
                                    f"add it after the desync audit passes")
        flush(exit_code=2)
    bench_subdir = args.bench_subdir or entry["bench_subdir"]

    if gpu_free_gib is not None and gpu_free_gib < MIN_FREE_GIB:
        msg = GPU_OCCUPIED_MSG.format(note=gpu_note, need=MIN_FREE_GIB)
        print(f"[targeted] {msg}", file=sys.stderr)
        record["status"] = "gpu_occupied"
        record["failure_reason"] = msg
        flush(exit_code=5)

    from envs.utils.create_actor import UnStableError

    cfg = build_cfg(args, out_dir)
    dr = cfg.get("domain_randomization") or {}
    scene_kind = ("cluttered" if (dr.get("cluttered_table") and dr.get("obstacle_density", 0) > 0)
                  else "clean")
    record["scene_kind"] = scene_kind
    if args.role == "perturbed":
        required = registry.REQUIRED_SCENE_BY_PTYPE[args.ptype]
        if required != scene_kind:
            msg = (f"FATAL: {args.ptype} requires a {required} scene but config "
                   f"'{args.task_config}' is {scene_kind} (scene-pairing rule: object/target "
                   f"shifts in clean scenes, obstacle shifts/hides in cluttered scenes)")
            print(f"[targeted] {msg}", file=sys.stderr)
            record["status"] = "config_mismatch"
            record["failure_reason"] = msg
            flush(exit_code=6)

    env_class = load_env_class(args.task_name, bench_subdir)
    env = env_class()
    try:
        env.setup_demo(**cfg)
    except UnStableError as e:
        record["status"] = "skip_unstable"
        record["failure_reason"] = str(e)
        flush(exit_code=2)
    except Exception:
        tb = traceback.format_exc(limit=20)
        if is_cuda_oom(tb):
            print(f"[targeted] {GPU_OCCUPIED_MSG.format(note=gpu_note, need=MIN_FREE_GIB)}",
                  file=sys.stderr)
            record["status"] = "gpu_occupied"
        else:
            record["status"] = "runtime_error"
        record["failure_reason"] = tb
        flush(exit_code=5 if record["status"] == "gpu_occupied" else 4)

    rt = TargetedRuntime()
    rt.attach(env)

    record["planner_visible_actors"] = sorted(
        info["actor"].get_name() for info in getattr(env, "collision_list", [])
        if not (getattr(env, "enable_collision_metrics", False) and info.get("is_obstacle", False)))

    depth_axis, lateral_axis, cam_forward, axes_source = table_plane_camera_axes(env)
    record["camera_axes"] = {
        "depth_axis_world": depth_axis.tolist(),
        "lateral_axis_world": lateral_axis.tolist(),
        "head_camera_forward_world": cam_forward,
        "source": axes_source,
    }

    # Params: CLI > config `targeted:` block > sampler defaults.
    cli_params = json.loads(args.params_json) if args.params_json else None
    cfg_params = (cfg.get("targeted") or {}).get(args.ptype) if args.role == "perturbed" else None

    world_shift = np.zeros(3)
    perturbed_actor = None
    if args.role == "perturbed" and args.ptype == "shift_obstacle":
        params = cli_params or cfg_params or {"mode": "corridor", "corridor_t": 0.55}
        record["sampler"] = params
        record["perceptual_failure_class"] = "obstacle_localization_error"
        try:
            perturbed_actor = rt.select_corridor_obstacle(env)
        except EpisodeDiscard as d:
            record["status"] = f"skip_{d.reason}"
            record["failure_reason"] = d.detail
            flush(exit_code=2)
        record["planner_entries_hidden"] = rt.hide_from_planner(env, perturbed_actor)
        record["planner_visible_actors_after_hide"] = sorted(
            info["actor"].get_name() for info in env.collision_list
            if not (getattr(env, "enable_collision_metrics", False) and info.get("is_obstacle", False))
            and not rt.is_excluded(info["actor"]))
        world_shift = rt.corridor_shift_vector(env, perturbed_actor,
                                               float(params.get("corridor_t", 0.55)))
        record["commanded"] = {
            "magnitude_cm": float(np.linalg.norm(world_shift) * 100.0),
            "shift_world": world_shift.tolist(),
            "camera_frame": labels.camera_frame_decomposition(world_shift, depth_axis, lateral_axis),
        }
    elif args.role == "perturbed":
        params = cli_params or cfg_params or sampler.sample_shift_params(args.ptype, args.seed, 0)
        record["sampler"] = params
        record["perceptual_failure_class"] = params["target_class"]
        world_shift = sampler.construct_world_shift(params, depth_axis, lateral_axis)
        record["commanded"] = {
            "magnitude_cm": params["magnitude_cm"],
            "shift_world": world_shift.tolist(),
            "camera_frame": labels.camera_frame_decomposition(world_shift, depth_axis, lateral_axis),
        }
        perturbed_actor = getattr(env, ACTOR_ATTR_BY_PTYPE[args.ptype])

    if perturbed_actor is not None:
        record["perturbed_actor"] = perturbed_actor.get_name()

    deferred_is_real = (args.role == "perturbed"
                        and TRIGGER_BY_PTYPE.get(args.ptype) == "after_grasp_plan")

    def on_deferred_shift(rec):
        if deferred_is_real:
            record["achieved"] = rec
            record["shift_frame_idx"] = rec.get("shift_frame_idx")
        else:
            record["baseline_deferred_settle"] = {
                k: rec[k] for k in ("max_other_actor_move_m", "max_other_actor_name") if k in rec}

    status, play_error = "recorded", None
    start_target_p = None
    try:
        # Both twins run the identical settle protocol at BOTH trigger points;
        # the shift is zero everywhere except the perturbed episode's own
        # trigger (principle 5).
        immediate_shift = world_shift if (args.role == "perturbed"
                                          and TRIGGER_BY_PTYPE[args.ptype] == "immediate") else np.zeros(3)
        immediate_actor = perturbed_actor if float(np.linalg.norm(immediate_shift)) > 0 else None
        rec = rt.apply_shift(env, immediate_actor, immediate_shift)
        if immediate_actor is not None:
            rec["applied"] = True
            record["achieved"] = rec
            record["shift_frame_idx"] = None  # immediate trigger: null per principle 7

        deferred_shift = world_shift if deferred_is_real else np.zeros(3)
        deferred_actor = perturbed_actor if float(np.linalg.norm(deferred_shift)) > 0 else None
        rt.arm("after_grasp_plan", deferred_actor, deferred_shift, on_deferred_shift)

        # Live (post-immediate-shift) start poses for outcome signals.
        start_target_p = np.array(env.target_obj.get_pose().p, dtype=np.float64)

        env.play_once()
    except EpisodeDiscard as d:
        status = f"discard_{d.reason}"
        play_error = d.detail
    except Exception:
        play_error = traceback.format_exc(limit=20)
        status = "gpu_occupied" if is_cuda_oom(play_error) else "runtime_error"
        if status == "gpu_occupied":
            print(f"[targeted] {GPU_OCCUPIED_MSG.format(note=gpu_note, need=MIN_FREE_GIB)}",
                  file=sys.stderr)

    # ---- Outcome signals (always gathered; the scene is still alive) ----
    try:
        target_p = np.array(env.target_obj.get_pose().p, dtype=np.float64)
        des_p = np.array(env.des_obj.get_pose().p, dtype=np.float64)
        object_moved_cm = None
        if start_target_p is not None:
            post_shift_start = start_target_p
            if (record.get("achieved") and record["achieved"].get("achieved_world")
                    and args.ptype == "shift_object"):
                post_shift_start = start_target_p + np.asarray(record["achieved"]["achieved_world"])
            object_moved_cm = float(np.linalg.norm(target_p - post_shift_start) * 100.0)
        place_error_cm = float(np.linalg.norm(target_p[:2] - des_p[:2]) * 100.0)
        success = bool(env.check_success())
        record["signals"] = {
            "success_flag": success,
            "plan_success": bool(env.plan_success),
            "object_moved_cm": object_moved_cm,
            "place_error_cm": place_error_cm,
            "final_target_obj_p": target_p.tolist(),
            "final_des_obj_p": des_p.tolist(),
        }
        if getattr(env, "enable_collision_metrics", False) and hasattr(env, "robot_link_names"):
            record["signals"]["collision_metrics"] = env.get_collision_metrics()
        record["contact_log"] = rt.contact_log
        # Per-frame raw state goes into the HDF5 (written below); here we only
        # record the entity roles so offline metrics know which actor is which.
        record["n_frames"] = len(rt.frame_state)
        record["progress_entities"] = {
            "target": env.target_obj.get_name() if hasattr(env, "target_obj") else None,
            "destination": env.des_obj.get_name() if hasattr(env, "des_obj") else None,
        }
        try:
            record["instruction"] = env.get_instruction()
        except Exception:
            record["instruction"] = None
    except Exception:
        if status == "recorded":
            status = "runtime_error"
            play_error = traceback.format_exc(limit=20)

    record["status"] = status
    if status == "recorded":
        metrics = record["signals"].get("collision_metrics") or {}
        record["outcome"], record["failure_reason"] = labels.derive_outcome(
            plan_success=record["signals"]["plan_success"],
            success=record["signals"]["success_flag"],
            is_collision=bool(metrics.get("is_collision", False)),
            object_moved_cm=record["signals"]["object_moved_cm"],
            place_error_cm=record["signals"]["place_error_cm"],
        )
    else:
        record["failure_reason"] = play_error

    # ---- Persist sim artifacts + HDF5 label annotation (component 0.4) ----
    try:
        env.close_env(clear_cache=True)
        if cfg["save_data"]:
            env.merge_pkl_to_hdf5_video()
            env.remove_data_cache()
            hdf5_path = out_dir / "data" / f"episode{env.ep_num}.hdf5"
            if not targeted.write_labels_hdf5(hdf5_path, record):
                record.setdefault("warnings", []).append("hdf5 annotation skipped: file missing")
            # Raw per-frame state for offline metrics (targeted.compute_progress/safety).
            targeted.write_frame_state(hdf5_path, rt.frame_state, record["progress_entities"])
    except Exception:
        record.setdefault("warnings", []).append(
            "artifact save failed: " + traceback.format_exc(limit=10))

    exit_code = {"recorded": 0, "skip_unstable": 2, "runtime_error": 4, "gpu_occupied": 5}.get(status, 3)
    flush(exit_code=exit_code)


if __name__ == "__main__":
    main()
