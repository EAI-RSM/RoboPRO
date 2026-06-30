"""Run ONE multimodal episode: a (task, mode) pair on a fixed scene seed.

Reuses run_targeted_episode.py's env setup (ensure_env / build_cfg / load_env_class)
and robo_negative's frame-state logger + HDF5 writers, but instead of a perturbation
it attaches a MultimodalRuntime that makes the planner take a mode-specific path.

Always a subprocess of collect_multimodal.py (CuRobo/SAPIEN state never leaks).
Writes into --out-dir: episode.json, data/episode0.hdf5 (+ targeted_state trace +
a `multimodal` label group), video/episode0.mp4.
Exit: 0 recorded, 2 skip(unstable), 4 runtime, 5 gpu.
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent              # scripts/
REPO = HERE.parent                                  # repo root
sys.path.insert(0, str(HERE))                       # run_targeted_episode (sibling script)
import run_targeted_episode as RT
from robo_tools.multimodal import MODES, MultimodalRuntime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--task-config", required=True)
    ap.add_argument("--bench-subdir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    RT.ensure_env()
    gpu_free_gib, gpu_note = RT.probe_visible_gpus()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(os.environ["ROBOTWIN_ROOT"])

    import robo_tools as targeted
    from robo_tools import TargetedRuntime, EpisodeDiscard

    record = {
        "task_name": args.task_name, "task_config": args.task_config,
        "scene_seed": args.seed, "executor_type": "scripted_expert",
        "mode": args.mode, "mode_params": MODES[args.mode],
        "pair_id": f"{args.task_name}__seed{args.seed:05d}",   # all modes of a scene share this
        "pair_role": "mode", "status": "running", "outcome": None,
        "failure_reason": None, "signals": {}, "instruction": None, "gpu": gpu_note,
    }

    def flush(code=None):
        (out_dir / "episode.json").write_text(json.dumps(record, indent=2))
        if code is not None:
            print(f"[mode] {args.task_name} mode={args.mode} seed={args.seed} "
                  f"status={record['status']} outcome={record['outcome']}")
            sys.exit(code)

    if gpu_free_gib is not None and gpu_free_gib < RT.MIN_FREE_GIB:
        record["status"] = "gpu_occupied"
        record["failure_reason"] = RT.GPU_OCCUPIED_MSG.format(note=gpu_note, need=RT.MIN_FREE_GIB)
        flush(5)

    from envs.utils.create_actor import UnStableError

    cfg = RT.build_cfg(SimpleNamespace(task_name=args.task_name, task_config=args.task_config,
                                       seed=args.seed, save_data=True, save_freq=None), out_dir)
    env = RT.load_env_class(args.task_name, args.bench_subdir)()
    try:
        env.setup_demo(**cfg)
    except UnStableError as e:
        record["status"] = "skip_unstable"; record["failure_reason"] = str(e); flush(2)
    except Exception:
        tb = traceback.format_exc(limit=20)
        record["status"] = "gpu_occupied" if RT.is_cuda_oom(tb) else "runtime_error"
        record["failure_reason"] = tb
        flush(5 if record["status"] == "gpu_occupied" else 4)

    rt = TargetedRuntime(); rt.attach(env)            # per-frame actor trace (reused, task-agnostic)
    mm = MultimodalRuntime(MODES[args.mode], args.mode); mm.attach(env)  # planner mode

    status, play_error = "recorded", None
    try:
        env.play_once()
    except EpisodeDiscard as d:
        status = f"discard_{d.reason}"; play_error = d.detail
    except Exception:
        play_error = traceback.format_exc(limit=20)
        status = "gpu_occupied" if RT.is_cuda_oom(play_error) else "runtime_error"

    # ---- signals (defensive: no task-specific attrs assumed) ----
    try:
        success = None
        try:
            success = bool(env.check_success())
        except Exception:
            pass
        plan_ok = bool(getattr(env, "plan_success", False))
        record["signals"] = {"success_flag": success, "plan_success": plan_ok,
                             "n_injected_waypoints": len(mm.injected_waypoints)}
        record["n_frames"] = len(rt.frame_state)
        record["injected_waypoints"] = mm.injected_waypoints
        tgt, dst = getattr(env, "target_obj", None), getattr(env, "des_obj", None)
        record["progress_entities"] = {
            "target": tgt.get_name() if tgt is not None and hasattr(tgt, "get_name") else None,
            "destination": dst.get_name() if dst is not None and hasattr(dst, "get_name") else None}
        try:
            record["instruction"] = env.get_instruction()
        except Exception:
            record["instruction"] = None
    except Exception:
        if status == "recorded":
            status = "runtime_error"; play_error = traceback.format_exc(limit=20)

    record["status"] = status
    if status == "recorded":
        s = record["signals"].get("success_flag")
        # A mode "realized" if the planner executed it AND the task still succeeded.
        record["outcome"] = ("mode_success" if s
                             else "mode_unrealized" if not record["signals"]["plan_success"]
                             else "mode_failed")
    else:
        record["failure_reason"] = play_error

    # ---- persist artifacts + enriched groups ----
    try:
        env.close_env(clear_cache=True)
        if cfg["save_data"]:
            env.merge_pkl_to_hdf5_video()
            env.remove_data_cache()
            hdf5 = out_dir / "data" / f"episode{env.ep_num}.hdf5"
            targeted.write_labels_hdf5(hdf5, record)
            targeted.write_frame_state(hdf5, rt.frame_state, record["progress_entities"])
            _write_mode_label(hdf5, record)
    except Exception:
        record.setdefault("warnings", []).append("artifact save failed: " + traceback.format_exc(limit=10))

    code = {"recorded": 0, "skip_unstable": 2, "runtime_error": 4, "gpu_occupied": 5}.get(status, 3)
    flush(code)


def _write_mode_label(hdf5_path, record):
    """A small `multimodal` group so consumers can filter by mode without parsing JSON."""
    import h5py
    if not Path(hdf5_path).exists():
        return
    with h5py.File(hdf5_path, "a") as f:
        g = f.require_group("multimodal")
        g.attrs["mode"] = record["mode"]
        g.attrs["pair_id"] = record["pair_id"]
        g.attrs["outcome"] = str(record["outcome"])
        g.attrs["n_injected_waypoints"] = record["signals"].get("n_injected_waypoints", 0)


if __name__ == "__main__":
    main()
