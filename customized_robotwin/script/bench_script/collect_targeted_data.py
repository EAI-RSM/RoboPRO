"""Targeted-collection orchestrator (production form of the PoC collector).

No sim imports here — every episode runs in its own subprocess
(run_targeted_episode.py), so CuRobo warm-start state never leaks between the
baseline runs and the perturbed twin.

Work is organized by the supported-task registry (bench_envs/targeted/
registry.py) and the scene-pairing rule: for each requested task, the clean
config carries shift_object/shift_target and the cluttered config carries
shift_obstacle/hide_obstacle, each group with its own n-baseline seed
qualification. Unsupported tasks/ptypes are skipped with a logged reason —
support is enabled incrementally by extending the registry.

Per (task, config) group and seed:
  1. n baseline runs (default 3); the seed qualifies only if all n are
     recorded with outcome == clean_success.
  2. One perturbed twin per requested ptype, params from the stratified
     sampler (isolated RNG).

GPUs are user-managed: episodes use --gpus / --gpu / inherited
CUDA_VISIBLE_DEVICES; an occupied GPU aborts the whole batch loudly. Recorded
episodes resume from cache on rerun.

Example (robopro conda env, from anywhere):
    python customized_robotwin/script/bench_script/collect_targeted_data.py \
        --tasks put_mouse_on_pad --seeds 0:40 --target-perturbed 30 \
        --workers 5 --gpus 3,4,5,6,7
"""
import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RUN_EPISODE = SCRIPT_DIR / "run_targeted_episode.py"

# Flat import of the registry/sampler modules — avoids pulling the sim-heavy
# bench_envs package into the orchestrator process.
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "bench_envs" / "targeted"))
import registry  # noqa: E402
from sampler import sample_shift_params  # noqa: E402

_INDEX_LOCK = threading.Lock()


class GpuOccupiedError(RuntimeError):
    """A visible GPU was occupied. GPUs are user-managed — abort the batch."""


def parse_seeds(spec: str) -> list[int]:
    if ":" in spec:
        lo, hi = spec.split(":")
        return list(range(int(lo), int(hi)))
    return [int(s) for s in spec.split(",")]


def append_index(index_path: Path, row: dict):
    with _INDEX_LOCK:
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def patch_json(path: Path, updates: dict):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(updates)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except FileNotFoundError:
        pass


def cached_row(args, out_dir: Path, seed: int, role: str, ptype: str) -> dict | None:
    """Resume support: reuse a cleanly recorded episode instead of re-running."""
    ep_path = out_dir / "episode.json"
    if not ep_path.exists():
        return None
    try:
        episode = json.loads(ep_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if episode.get("status") != "recorded":
        return None
    row = {
        "task": episode.get("task_name"),
        "config": episode.get("task_config"),
        "seed": seed,
        "role": role,
        "ptype": ptype if role == "perturbed" else None,
        "status": "recorded",
        "outcome": episode.get("outcome"),
        "target_class": (episode.get("sampler") or {}).get("target_class"),
        "magnitude_cm": (episode.get("sampler") or {}).get("magnitude_cm"),
        "achieved_magnitude_cm": (episode.get("achieved") or {}).get("achieved_magnitude_cm"),
        "dir": str(out_dir),
        "returncode": 0,
        "duration_s": 0.0,
        "cached": True,
    }
    append_index(args.index_path, row)
    print(f"  [{role:9s}] seed={seed} ptype={row['ptype'] or '-':12s} "
          f"status=recorded (cached)        outcome={row['outcome'] or '-':22s}")
    return row


def run_episode(args, *, group: dict, seed: int, role: str, out_dir: Path,
                ptype: str = "none", params: dict | None = None,
                pair_id: str | None = None, gpu: str | None = None) -> dict:
    cmd = shlex.split(args.launcher_prefix) + [
        args.python, "-u", str(RUN_EPISODE),
        "--task-name", group["task"],
        "--task-config", group["config"],
        "--bench-subdir", group["subdir"],
        "--seed", str(seed),
        "--role", role,
        "--ptype", ptype,
        "--out-dir", str(out_dir),
    ]
    if params is not None:
        cmd += ["--params-json", json.dumps(params)]
    if pair_id is not None:
        cmd += ["--pair-id", pair_id]
    if not args.save_data:
        cmd += ["--no-save-data"]

    env = dict(os.environ)
    env.setdefault("ROBOTWIN_ROOT", str(REPO_ROOT / "customized_robotwin"))
    env.setdefault("BENCH_ROOT", str(REPO_ROOT / "benchmark"))
    env["ROBOTWIN_BENCH_TASK"] = "bench"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    effective_gpu = gpu if gpu is not None else args.gpu
    if effective_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(effective_gpu)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_dir / "run.log", "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=args.timeout, check=False)
    duration = time.time() - t0

    episode = {}
    ep_path = out_dir / "episode.json"
    if ep_path.exists():
        episode = json.loads(ep_path.read_text(encoding="utf-8"))
    row = {
        "task": group["task"],
        "config": group["config"],
        "seed": seed,
        "role": role,
        "ptype": ptype if role == "perturbed" else None,
        "status": episode.get("status", f"no_episode_json_rc{proc.returncode}"),
        "outcome": episode.get("outcome"),
        "target_class": (episode.get("sampler") or {}).get("target_class"),
        "magnitude_cm": (episode.get("sampler") or {}).get("magnitude_cm"),
        "achieved_magnitude_cm": (episode.get("achieved") or {}).get("achieved_magnitude_cm"),
        "dir": str(out_dir),
        "returncode": proc.returncode,
        "duration_s": round(duration, 1),
    }
    append_index(args.index_path, row)
    print(f"  [{row['role']:9s}] seed={seed} ptype={row['ptype'] or '-':12s} "
          f"status={row['status']:24s} outcome={row['outcome'] or '-':22s} "
          f"({row['duration_s']:.0f}s)")
    if row["status"] == "gpu_occupied" or proc.returncode == 5:
        raise GpuOccupiedError(episode.get("failure_reason")
                               or f"episode at {out_dir} reported an occupied GPU")
    return row


def collect_group(args, group: dict):
    """Run one (task, config, ptypes) group: per-seed qualification then twins."""
    runs_root = args.out_root / "runs" / group["task"] / group["config"]
    ep_counter = {pt: 0 for pt in group["ptypes"]}
    state = {"perturbed_recorded": 0}
    counter_lock = threading.Lock()
    stop = threading.Event()

    gpu_pool: queue.Queue = queue.Queue()
    gpu_ids = args.gpus.split(",") if args.gpus else [None] * args.workers
    for i in range(args.workers):
        gpu_pool.put(gpu_ids[i % len(gpu_ids)])

    print(f"[collector] group: task={group['task']} config={group['config']} "
          f"ptypes={group['ptypes']} target={args.target_perturbed}")

    def process_seed(seed: int):
        if stop.is_set():
            return
        gpu = gpu_pool.get()
        try:
            return _process_seed_inner(seed, gpu)
        except GpuOccupiedError:
            stop.set()  # halt all chains before propagating
            raise
        finally:
            gpu_pool.put(gpu)

    def _process_seed_inner(seed: int, gpu):
        seed_dir = runs_root / f"seed{seed:05d}"
        pair_id = f"{group['task']}__{group['config']}__seed{seed:05d}"
        print(f"[collector] seed {seed} — qualifying baseline x{args.n_baseline}"
              + (f" on gpu {gpu}" if gpu is not None else ""))

        baseline_rows = []
        for t in range(args.n_baseline):
            if stop.is_set():
                return
            out_dir = seed_dir / f"baseline_t{t}"
            row = (cached_row(args, out_dir, seed, "baseline", "none")
                   or run_episode(args, group=group, seed=seed, role="baseline",
                                  out_dir=out_dir, pair_id=pair_id, gpu=gpu))
            baseline_rows.append(row)
            if not (row["status"] == "recorded" and row["outcome"] == "clean_success"):
                break  # seed disqualified; skip remaining trials

        successes = sum(1 for r in baseline_rows
                        if r["status"] == "recorded" and r["outcome"] == "clean_success")
        qualified = successes == args.n_baseline
        qual = {"baseline_trials": len(baseline_rows), "baseline_successes": successes,
                "seed_qualified": qualified}
        for t in range(len(baseline_rows)):
            patch_json(seed_dir / f"baseline_t{t}" / "episode.json", qual)
        if not qualified:
            append_index(args.index_path, {"task": group["task"], "config": group["config"],
                                           "seed": seed, "role": "qualification",
                                           "status": "seed_disqualified", **qual})
            print(f"[collector] seed {seed} disqualified "
                  f"({successes}/{len(baseline_rows)} clean)")
            return

        for ptype in group["ptypes"]:
            out_dir = seed_dir / ptype
            row = cached_row(args, out_dir, seed, "perturbed", ptype)
            if row is None:
                with counter_lock:
                    if state["perturbed_recorded"] >= args.target_perturbed:
                        stop.set()
                        return
                    ep_index = ep_counter[ptype]
                    ep_counter[ptype] += 1
                params = (sample_shift_params(ptype, seed, ep_index)
                          if ptype != "shift_obstacle" else None)
                row = run_episode(args, group=group, seed=seed, role="perturbed", ptype=ptype,
                                  out_dir=out_dir, params=params, pair_id=pair_id, gpu=gpu)
            patch_json(out_dir / "episode.json", {
                **qual,
                "twin_relpaths": [f"baseline_t{t}" for t in range(args.n_baseline)],
            })
            if row["status"] == "recorded":
                with counter_lock:
                    state["perturbed_recorded"] += 1
                    if state["perturbed_recorded"] >= args.target_perturbed:
                        stop.set()

    seeds = parse_seeds(args.seeds)
    if args.workers <= 1:
        for seed in seeds:
            if stop.is_set():
                break
            process_seed(seed)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_seed, seed) for seed in seeds]
            for fut in futures:
                fut.result()
    print(f"[collector] group done — {state['perturbed_recorded']} perturbed episodes recorded")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=["put_mouse_on_pad"])
    parser.add_argument("--ptypes", nargs="+",
                        default=["shift_object", "shift_target", "shift_obstacle"])
    parser.add_argument("--seeds", default="0:40", help="'lo:hi' range or comma list")
    parser.add_argument("--n-baseline", type=int, default=3)
    parser.add_argument("--target-perturbed", type=int, default=30,
                        help="per (task, config) group")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "targetted_dataset"))
    parser.add_argument("--save-data", dest="save_data", action="store_true", default=True)
    parser.add_argument("--no-save-data", dest="save_data", action="store_false")
    parser.add_argument("--timeout", type=int, default=2400, help="per-episode seconds")
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES for all episodes (user-managed; default: inherit)")
    parser.add_argument("--gpus", default=None,
                        help="comma list of GPU ids round-robined across workers. GPUs are "
                             "user-managed: an occupied GPU aborts the batch.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--launcher-prefix", default="",
                        help="optional command prefix for each episode run")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    args.out_root = Path(args.out_root).resolve()
    args.index_path = args.out_root / "index.jsonl"
    args.out_root.mkdir(parents=True, exist_ok=True)

    # Build the work list from the registry + scene-pairing rule; skip and log
    # anything unsupported (incremental task enablement).
    groups = []
    for task in args.tasks:
        entry = registry.task_entry(task)
        if entry is None:
            append_index(args.index_path, {"task": task, "role": "registry",
                                           "status": "skip_unsupported_task"})
            print(f"[collector] SKIP task '{task}' — not in bench_envs.targeted.registry "
                  f"(supported: {sorted(registry.SUPPORTED_TASKS)})")
            continue
        unsupported = [pt for pt in args.ptypes if pt not in entry["ptypes"]]
        for pt in unsupported:
            append_index(args.index_path, {"task": task, "ptype": pt, "role": "registry",
                                           "status": "skip_unsupported_ptype"})
            print(f"[collector] SKIP ptype '{pt}' for '{task}' — not in registry entry")
        for scene_kind, config_key in (("clean", "clean_config"),
                                       ("cluttered", "cluttered_config")):
            ptypes = [pt for pt in args.ptypes if pt in entry["ptypes"]
                      and registry.REQUIRED_SCENE_BY_PTYPE.get(pt) == scene_kind]
            if ptypes:
                groups.append({"task": task, "subdir": entry["bench_subdir"],
                               "config": entry[config_key], "ptypes": ptypes})

    if not groups:
        print("[collector] nothing to do — no supported (task, ptype) combinations")
        return

    print(f"[collector] {len(groups)} group(s); gpus={args.gpus or args.gpu or 'inherited'} "
          f"workers={args.workers} out={args.out_root}")
    try:
        for group in groups:
            collect_group(args, group)
    except GpuOccupiedError as e:
        print(f"\n[collector] ABORTED — {e}\n"
              f"[collector] GPUs are user-managed: free a GPU (or set CUDA_VISIBLE_DEVICES/"
              f"--gpus) and rerun the same command; recorded episodes resume from cache.",
              file=sys.stderr)
        sys.exit(2)
    print(f"[collector] all groups done; index at {args.index_path}")


if __name__ == "__main__":
    main()
