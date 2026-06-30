#!/usr/bin/env python
"""Collect a multimodal PoC set: each task on a FIXED seed, run through every mode
(MODE_ORDER). Same scene, N distinct behaviors. Each episode is an isolated
subprocess of run_mode_episode.py, parallelized one episode per GPU.

  python collect_multimodal.py --tasks-file tasks.json --seed 0 \
      --gpus 0,1,2,3 --out-root ../multimodal_data
  # tasks.json = [["put_stapler_on_book","office"], ["put_bread_on_board_ks","kitchens"]]
  #   (tasks must be in robo_negative.SUPPORTED_TASKS)
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN = HERE / "run_mode_episode.py"
PY = sys.executable
SCENE_CLEAN = {"office": "bench_demo_office_clean", "study": "bench_demo_study_clean",
               "kitchenl": "bench_demo_kitchenl_clean", "kitchens": "bench_demo_kitchens_clean"}
sys.path.insert(0, str(HERE))
from robo_tools.multimodal import MODE_ORDER

_lock = threading.Lock()


def log(m):
    with _lock:
        print(m, flush=True)


def run_episode(task, subdir, mode, seed, gpu_pool, out_root, timeout):
    gpu = gpu_pool.get()
    try:
        out_dir = out_root / "runs" / task / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-u", str(RUN), "--task-name", task, "--task-config", SCENE_CLEAN[subdir],
               "--bench-subdir", subdir, "--seed", str(seed), "--mode", mode,
               "--out-dir", str(out_dir)]
        env = dict(os.environ)
        env.update({"ROBOTWIN_ROOT": str(REPO / "customized_robotwin"),
                    "BENCH_ROOT": str(REPO / "benchmark"), "ROBOTWIN_BENCH_TASK": "bench",
                    "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu)})
        log(f"[launch] {task:28s} {mode:12s} -> gpu {gpu}")
        t0 = time.time()
        with open(out_dir / "run.log", "w") as lf:
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False)
        dur = time.time() - t0
        ep = {}
        if (out_dir / "episode.json").exists():
            try:
                ep = json.loads((out_dir / "episode.json").read_text())
            except Exception:
                pass
        sig = ep.get("signals") or {}
        row = {"task": task, "mode": mode, "gpu": gpu, "rc": proc.returncode,
               "status": ep.get("status"), "outcome": ep.get("outcome"),
               "n_waypoints": sig.get("n_injected_waypoints"), "n_frames": ep.get("n_frames"),
               "dur_s": round(dur, 1), "dir": str(out_dir)}
        log(f"[done]   {task:28s} {mode:12s} status={str(row['status']):>14} "
            f"outcome={str(row['outcome']):>16} wp={row['n_waypoints']} ({row['dur_s']:.0f}s)")
        return row
    except subprocess.TimeoutExpired:
        log(f"[TIMEOUT] {task} {mode}")
        return {"task": task, "mode": mode, "status": "timeout"}
    finally:
        gpu_pool.put(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out-root", default=str(REPO / "multimodal_data"))
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks_file).read_text())  # [[task, subdir], ...]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    gpu_pool = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)

    jobs = [(t, sd, m) for (t, sd) in tasks for m in MODE_ORDER]
    log(f"[batch] {len(tasks)} tasks x {len(MODE_ORDER)} modes = {len(jobs)} episodes "
        f"(seed {args.seed}); gpus={gpus}; out={out_root}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        rows = list(pool.map(lambda j: run_episode(j[0], j[1], j[2], args.seed, gpu_pool,
                                                   out_root, args.timeout), jobs))
    dur = time.time() - t0
    realized = [r for r in rows if r.get("outcome") == "mode_success"]
    (out_root / "summary.json").write_text(json.dumps(
        {"rows": rows, "mode_success": len(realized), "total": len(rows),
         "wall_s": round(dur, 1)}, indent=2))
    log(f"\n[batch] DONE in {dur:.0f}s — {len(realized)}/{len(rows)} modes realized (success)")
    for r in sorted(rows, key=lambda x: (x["task"], x["mode"])):
        log(f"   {r['task']:28s} {r['mode']:12s} {str(r.get('outcome')):>16}  "
            f"{r.get('n_frames')} frames")


if __name__ == "__main__":
    main()
