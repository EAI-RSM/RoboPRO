#!/usr/bin/env python
"""Q-function evaluation set: success/failure branches from identical states.

Protocol rationale and the measurements behind it: see script/replay_utils.py.

PROTOCOL (settled by the determinism tests on 2026-08-07):
  * spine   -- one plain policy rollout per scene; its commanded actions are stored.
  * branch  -- rebuild the scene from seed (close_env + setup_demo, which resets the
               PhysX solver cache), replay the spine's actions to step T with NO
               policy (pure action replay, proven bit-exact), then run the policy
               closed-loop from T to terminal with a fresh noise key.
  * NEVER `rollback.restore`: it restores poses/velocities bit-exactly but not the
    solver warm-start cache, which flips the success label in 38.9% of states and
    biases systematically toward failure. Fresh rebuild is 0.000e+00 / 0% flips.

Every attempt is retained, pairing or not: 8/8 successes is the measurement
p_hat(success|s) > 0.8, not a wasted slot. Pairs are formed post-hoc.

Output under --outdir/<task>/<density>/seed<seed>/
    spine.npz                       spine actions + label
    T<T>/attempt<k>_{SUCCESS,FAIL}.mp4
    T<T>/attempt<k>.npz             branch actions, first chunk, noise key
    attempts.jsonl                  one row per attempt (appended live)
    summary.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
from script.bench_script.setup_paths import setup_paths  # noqa: E402

setup_paths()
import eval_policy_client as E  # noqa: E402

from replay_utils import build_args, rebuild, replay_actions, task_environment  # noqa: E402


def camera_wh(args):
    cfg = Path(__file__).resolve().parents[1] / "task_config" / "_camera_config.yml"
    c = yaml.safe_load(cfg.read_text())[args["camera"]["head_camera_type"]]
    return c["w"], c["h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--densities", default="clean,clean,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15")
    ap.add_argument("--branch_steps", default="0,50,100")
    ap.add_argument("--max_attempts", type=int, default=8)
    ap.add_argument("--max_seeds", type=int, default=4, help="1 + 3 retries")
    ap.add_argument("--seed_base", type=int, default=5000, help="unseen: HF used 0,1,2,4,6,7")
    ap.add_argument("--step_lim", type=int, default=300)
    ap.add_argument("--pi0_step", type=int, default=50)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--instruction", default=None, help="default: bank entry 0 (fixed, no perturbation)")
    a = ap.parse_args()

    env = task_environment(a.task_name)
    if env is None:
        raise SystemExit(f"cannot resolve environment for {a.task_name}")
    Ts = [int(x) for x in a.branch_steps.split(",")]
    out = Path(a.outdir) / a.task_name
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "attempts.jsonl"

    instr = a.instruction
    if instr is None:
        bank = json.loads((Path(os.environ["BENCH_ROOT"]) / "bench_task_config"
                           / "instruction_bank.json").read_text())
        pool = bank.get(a.task_name)
        instr = pool[0] if pool else "perform the task"
    print(f"[cfg] task={a.task_name} env={env} Ts={Ts} attempts={a.max_attempts} "
          f"seeds={a.max_seeds} step_lim={a.step_lim}", flush=True)
    print(f"[cfg] instruction (FIXED) = {instr!r}", flush=True)

    eval_func = E.eval_function_decorator("pi05", "eval")
    model = E.ModelClient(port=a.port, pi0_step=a.pi0_step)
    task = None

    done = set()
    if jsonl.exists():
        for ln in jsonl.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done.add((r["density"], r["seed"], r["T"], r["k"]))
        print(f"[resume] {len(done)} attempts already recorded", flush=True)

    def rpc(name, *rpc_args):
        try:
            return getattr(model, name)(*rpc_args)
        except (ConnectionError, BrokenPipeError, OSError) as e:
            print(f"[rpc] reconnecting after {e}", flush=True)
            model._connect()
            return getattr(model, name)(*rpc_args)

    def _rebuild(args, seed):
        nonlocal task
        task = rebuild(task, a.task_name, args, seed,
                       step_lim=a.step_lim, instruction=instr)
        return task

    def run_policy(task, noise_key, record_from=None):
        """Closed-loop policy to terminal. Returns (actions, label, steps)."""
        rpc("reset_model")
        rpc("set_noise_episode", noise_key)
        acts = []
        orig = task.take_action

        def rec(action, *aa, **kw):
            acts.append(np.asarray(action, dtype=np.float32).copy())
            return orig(action, *aa, **kw)

        task.take_action = rec
        try:
            while task.take_action_cnt < task.step_lim and not task.eval_success:
                eval_func(task, model, task.get_obs())
        finally:
            task.take_action = orig
        lab = bool(task.eval_success or task.check_success())
        return (np.stack(acts) if acts else np.zeros((0, 14), np.float32)), lab, int(task.take_action_cnt)

    densities = a.densities.split(",")
    seen_dens = {}
    t_start = time.time()
    summary = []

    for di, dens in enumerate(densities):
        n_rep = seen_dens.get(dens, 0)
        seen_dens[dens] = n_rep + 1
        cfg = f"bench_demo_{env}_{dens}"
        args = build_args(a.task_name, cfg)
        W, H = camera_wh(args)
        paired = {T: False for T in Ts}
        slot = f"{dens}#{n_rep}"

        for s_try in range(a.max_seeds):
            if all(paired.values()):
                break
            seed = a.seed_base + di * 100 + n_rep * 10 + s_try
            sdir = out / dens / f"seed{seed}"
            try:
                task = _rebuild(args, seed)
            except Exception:
                print(f"[skip] {slot} seed {seed} scene build failed", flush=True)
                traceback.print_exc()
                continue

            # ---- spine: one plain policy rollout; its actions define the states ----
            sp_acts, sp_lab, sp_steps = run_policy(task, f"{a.task_name}_{dens}_{seed}_spine")
            sdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sdir / "spine.npz", actions=sp_acts, label=sp_lab, steps=sp_steps)
            print(f"[spine] {slot} seed={seed}: {'SUCCESS' if sp_lab else 'FAIL'} "
                  f"steps={sp_steps}", flush=True)

            for T in Ts:
                if paired[T] or T >= len(sp_acts):
                    continue
                tdir = sdir / f"T{T}"
                tdir.mkdir(parents=True, exist_ok=True)
                labs = []
                for k in range(a.max_attempts):
                    if (dens, seed, T, k) in done:
                        continue
                    key = f"{a.task_name}_{dens}_{seed}_T{T}_k{k}"
                    task = _rebuild(args, seed)
                    tmp = tdir / f"attempt{k}.tmp.mp4"
                    ff = subprocess.Popen(
                        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                         "-pixel_format", "rgb24", "-video_size", f"{W}x{H}",
                         "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                         "-vcodec", "libx264", "-crf", "23", str(tmp)],
                        stdin=subprocess.PIPE)
                    task._set_eval_video_ffmpeg(ff)
                    task.eval_video_path = str(tdir)
                    # prefix: pure action replay, NO policy (bit-exact from cold start)
                    replay_actions(task, sp_acts[:T])
                    br_acts, lab, steps = run_policy(task, key)
                    task._del_eval_video_ffmpeg()
                    task.eval_video_path = None
                    fin = tdir / f"attempt{k}_{'SUCCESS' if lab else 'FAIL'}.mp4"
                    if tmp.exists():
                        tmp.replace(fin)
                    np.savez_compressed(tdir / f"attempt{k}.npz",
                                        branch_actions=br_acts,
                                        first_chunk=br_acts[:a.pi0_step] if len(br_acts) else br_acts,
                                        prefix_len=T, noise_key=key, label=lab)
                    labs.append(lab)
                    row = {"task": a.task_name, "env": env, "density": dens, "slot": slot,
                           "seed": seed, "T": T, "k": k, "label": bool(lab), "steps": steps,
                           "noise_key": key, "spine_label": bool(sp_lab),
                           "video": fin.name, "wall_s": round(time.time() - t_start, 1)}
                    with open(jsonl, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"  [{slot} seed{seed} T{T}] attempt {k}: "
                          f"{'SUCCESS' if lab else 'FAIL'} steps={steps} "
                          f"({'+'.join(str(int(x)) for x in labs)})", flush=True)
                    if len(set(labs)) == 2:
                        paired[T] = True
                        break
                p_hat = float(np.mean(labs)) if labs else float("nan")
                summary.append({"density": dens, "slot": slot, "seed": seed, "T": T,
                                "n_attempts": len(labs), "p_hat": p_hat,
                                "paired": bool(paired[T])})
                print(f"  -> {slot} seed{seed} T{T}: paired={paired[T]} "
                      f"p_hat={p_hat:.2f} after {len(labs)} attempts", flush=True)

        for T in Ts:
            if not paired[T]:
                print(f"[unfilled] {slot} T{T}: no pair after {a.max_seeds} seeds", flush=True)

    (out / "summary.json").write_text(json.dumps(
        {"task": a.task_name, "env": env, "instruction": instr, "branch_steps": Ts,
         "max_attempts": a.max_attempts, "max_seeds": a.max_seeds,
         "step_lim": a.step_lim, "records": summary,
         "wall_min": round((time.time() - t_start) / 60, 1)}, indent=2))
    n_pair = sum(1 for r in summary if r["paired"])
    print(f"\n[done] {a.task_name}: {n_pair}/{len(summary)} branch points paired, "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)
    if task is not None:
        task.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
