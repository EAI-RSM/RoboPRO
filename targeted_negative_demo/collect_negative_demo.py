#!/usr/bin/env python
"""Collect the targeted-negative DEMO set: a subset of atomic tasks, each with a
clean baseline + the 4 perturbations (shift_object, shift_target, shift_obstacle,
hide_obstacle) — ALL in the same clean scene (obstacle perturbations inject their
own corridor obstacle). 5 episodes per task.

Each episode is an isolated subprocess of run_targeted_episode.py. Parallel one
episode per GPU. Perturbations use explicit, reliably-failing magnitudes so the
demo shows clean before/after pairs.

  python collect_negative_demo.py --tasks-file tasks.json --gpus 0,1,.. --out-root DIR
"""
import argparse
import json
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/work/mohammed/EAI_RSM/RoboPRO")
RUN = REPO / "run_targeted_episode.py"
PY = str(REPO / ".venv" / "bin" / "python")
SCENE_CLEAN = {"office": "bench_demo_office_clean", "study": "bench_demo_study_clean",
               "kitchenl": "bench_demo_kitchenl_clean", "kitchens": "bench_demo_kitchens_clean"}

# Explicit perturbation params -> reliable, clearly-labelled failures for the demo.
PTYPE_PARAMS = {
    "shift_object": {"target_class": "lateral", "magnitude_cm": 5.0, "sign": 1,
                     "mixed_angle_deg": 0.0, "magnitude_bin_index": 3, "magnitude_bin_cm": [3.5, 5.0]},
    "shift_target": {"target_class": "lateral", "magnitude_cm": 9.0, "sign": 1,
                     "mixed_angle_deg": 0.0, "magnitude_bin_index": 4, "magnitude_bin_cm": [5.0, 8.0]},
    "shift_obstacle": {"mode": "corridor", "corridor_t": 0.55},
    "hide_obstacle": {"mode": "corridor", "corridor_t": 0.5},
}
PTYPES = ["shift_object", "shift_target", "shift_obstacle", "hide_obstacle"]
_lock = threading.Lock()


def log(m):
    with _lock:
        print(m, flush=True)


def run_episode(task, subdir, role, ptype, seed, gpu_pool, out_root, timeout):
    gpu = gpu_pool.get()
    try:
        config = SCENE_CLEAN[subdir]
        label = "baseline" if role == "baseline" else ptype
        out_dir = out_root / "runs" / task / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-u", str(RUN), "--task-name", task, "--task-config", config,
               "--bench-subdir", subdir, "--seed", str(seed), "--role", role,
               "--ptype", (ptype or "none"), "--out-dir", str(out_dir),
               "--pair-id", f"{task}__seed{seed:05d}"]
        if role == "perturbed":
            cmd += ["--params-json", json.dumps(PTYPE_PARAMS[ptype])]
        env = dict(os.environ)
        env.update({"ROBOTWIN_ROOT": str(REPO / "customized_robotwin"),
                    "BENCH_ROOT": str(REPO / "benchmark"), "ROBOTWIN_BENCH_TASK": "bench",
                    "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                    "CUDA_VISIBLE_DEVICES": str(gpu)})
        log(f"[launch] {task:30s} {label:14s} -> gpu {gpu}")
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
        row = {"task": task, "label": label, "role": role, "ptype": ptype, "gpu": gpu,
               "rc": proc.returncode, "status": ep.get("status"), "outcome": ep.get("outcome"),
               "perceptual_failure_class": ep.get("perceptual_failure_class"),
               "obstacle_source": ep.get("obstacle_source"), "n_frames": ep.get("n_frames"),
               "dur_s": round(dur, 1), "dir": str(out_dir)}
        log(f"[done]   {task:30s} {label:14s} status={str(row['status']):>14} "
            f"outcome={str(row['outcome']):>22} ({row['dur_s']:.0f}s)")
        return row
    except subprocess.TimeoutExpired:
        log(f"[TIMEOUT] {task} {ptype or 'baseline'}")
        return {"task": task, "label": ptype or "baseline", "status": "timeout"}
    finally:
        gpu_pool.put(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks_file).read_text())  # [[task, subdir], ...]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = args.gpus.split(",")
    gpu_pool = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)

    jobs = []  # (task, subdir, role, ptype)
    for task, subdir in tasks:
        jobs.append((task, subdir, "baseline", None))
        for pt in PTYPES:
            jobs.append((task, subdir, "perturbed", pt))

    log(f"[batch] {len(tasks)} tasks x 5 = {len(jobs)} episodes; gpus={gpus}; out={out_root}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futs = [pool.submit(run_episode, t, sd, role, pt, args.seed, gpu_pool, out_root, args.timeout)
                for (t, sd, role, pt) in jobs]
        rows = [f.result() for f in futs]
    dur = time.time() - t0

    rec = [r for r in rows if r.get("status") == "recorded"]
    (out_root / "summary.json").write_text(json.dumps(
        {"rows": rows, "recorded": len(rec), "total": len(rows), "wall_s": round(dur, 1)}, indent=2))
    log(f"\n[batch] DONE in {dur:.0f}s — {len(rec)}/{len(rows)} recorded")
    for r in sorted(rows, key=lambda x: (x["task"], x["label"])):
        log(f"   {r['task']:30s} {r['label']:14s} {str(r.get('status')):>14} "
            f"{str(r.get('outcome')):>22}  src={r.get('obstacle_source')}")


if __name__ == "__main__":
    main()
