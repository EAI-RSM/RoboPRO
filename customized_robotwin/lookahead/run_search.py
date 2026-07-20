"""CLI: run a lookahead search and emit a verified motion Plan.

Flow
----
1. Build the task via ``collect_data.build_task_and_args`` + ``setup_demo`` (fresh,
   deterministic scene at ``--seed``).
2. Capture the FULL :class:`~lookahead.plan.TaskSpec` (embodiment, the complete
   ``domain_randomization`` dict, control/action semantics) + the t0 fingerprint +
   provenance (git HEAD of the worktree, task-config hash).
3. Load the candidate policy's model (``--candidate_policy`` deploy stack) and build
   a :class:`~lookahead.candidates.PolicyCandidates` source.
4. Run the chosen :class:`~lookahead.search.SearchStrategy` (default BeamSearch)
   scored by the chosen :class:`~lookahead.fitness.Fitness` (default OracleFitness).
5. VERIFY the winning raw actions with a from-start replay in a FRESH scene (same
   seed): apply the plan's actions from the pristine root and assert they reproduce
   the searched terminal fingerprint / success.
6. Write the :class:`~lookahead.plan.Plan` to
   ``<out>/<task>/<task_config>/seed<N>.json``.

Heavy sim / policy imports are deferred into function bodies so this module
byte-compiles and imports without the RoboTwin / jax environment. Run it from
``customized_robotwin`` with the pi05 venv, e.g.::

    python -m lookahead.run_search \
        --task put_milktea_on_shelf --task_config bench_demo_office_d8 --seed 40000 \
        --candidate_policy pi05 --policy_config policy/pi05/deploy_policy.yml \
        --strategy beam --width 3 --k 6 --depth 4 --fitness oracle \
        --out /work/mohammed/datasets/lookahead_plans
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Import the (pure) lookahead package whether run as a module or a bare script.
try:
    from . import fitness as fit_mod
    from . import pins as pins_mod
    from . import plan as plan_mod
    from . import rollback as rb
    from . import search as search_mod
    from .candidates import PolicyCandidates
except ImportError:  # pragma: no cover - executed only as a top-level script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lookahead import fitness as fit_mod
    from lookahead import pins as pins_mod
    from lookahead import plan as plan_mod
    from lookahead import rollback as rb
    from lookahead import search as search_mod
    from lookahead.candidates import PolicyCandidates


# ------------------------------------------------------------------ provenance
def git_head(repo_dir: str) -> Optional[str]:
    """``git rev-parse HEAD`` of the worktree containing this file, or None."""
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def config_hash(task_config: str) -> Optional[str]:
    """sha256 of the resolved task-config yml (mirrors build_task_and_args paths)."""
    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench" and os.getenv("BENCH_ROOT"):
        path = os.path.join(os.environ["BENCH_ROOT"], "bench_task_config", f"{task_config}.yml")
    else:
        path = os.path.join("task_config", f"{task_config}.yml")
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def now_iso() -> str:
    """Current UTC timestamp (ISO-8601). Kept OUT of the pure plan module."""
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ scene building
def build_scene(task_name: str, task_config: str, seed: int):
    """Build + set up a deterministic test scene. Returns ``(task, args)``.

    Mirrors the reference scripts: ``need_plan`` on, ``save_data`` off,
    ``render_freq`` 0, collision metrics on (so the fitness clutter proxy works).
    """
    import collect_data as cd  # deferred: pulls in sapien/envs
    task, args = cd.build_task_and_args(task_name, task_config)
    args.update(need_plan=True, save_data=False, render_freq=0)
    task.setup_demo(now_ep_num=0, seed=seed, is_test=True,
                    **{**args, "enable_collision_metrics": True})
    return task, args


def load_candidate(candidate_policy: str, usr_args: Dict[str, Any]):
    """Import the policy module and load its model. Returns ``(model, encode_obs)``."""
    import importlib
    for p in ("./", "./policy", f"./policy/{candidate_policy}"):
        if p not in sys.path:
            sys.path.append(p)
    mod = importlib.import_module(candidate_policy)
    return mod.get_model(usr_args), mod.encode_obs


def build_task_spec(task_name: str, task_config: str, seed: int, args: Dict[str, Any],
                    action_dim: int, chunk: int, repo_dir: str,
                    created_at: Optional[str] = None) -> "plan_mod.TaskSpec":
    """Assemble the PINNED TaskSpec from the resolved task args + provenance."""
    control = {
        "action_dim": int(action_dim),
        "chunk": int(chunk),
        "control_freq": args.get("control_freq"),
    }
    return plan_mod.TaskSpec(
        task_name=task_name,
        task_config=task_config,
        seed=int(seed),
        embodiment={
            "embodiment": args.get("embodiment"),
            "embodiment_name": args.get("embodiment_name"),
        },
        domain_randomization=dict(args.get("domain_randomization", {})),
        control=control,
        provenance={
            "robotwin_commit": git_head(repo_dir),
            "bench_config_hash": config_hash(task_config),
            "created_at": created_at or now_iso(),
        },
    )


# ------------------------------------------------------------------ verification
def verify_from_start(task_name: str, task_config: str, seed: int,
                      actions: np.ndarray, atol: float
                      ) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Fresh-scene, from-start raw-action replay of ``actions``.

    Builds a brand-new scene at the same seed, snapshots the pristine root, applies
    the plan's raw actions from t0 (no policy inference), and returns
    ``(t0_fingerprint, terminal_fingerprint, success)``. The caller compares the
    terminal fingerprint against the searched terminal (within ``atol``) to confirm
    the plan reproduces the searched trajectory before it is written.
    """
    task, _args = build_scene(task_name, task_config, seed)
    try:
        root = rb.snapshot(task)
        t0_fp = np.asarray(root["fingerprint"], dtype=np.float64)
        rb.apply_chunk(task, actions)
        terminal_fp = rb.state_fingerprint(task)
        success = bool(task.check_success())
        return t0_fp, terminal_fp, success
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------ main
def run(args: argparse.Namespace) -> str:
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    # usr_args for the candidate policy's get_model (deploy_policy.yml + overrides)
    import yaml
    with open(args.policy_config, "r", encoding="utf-8") as f:
        usr_args = yaml.safe_load(f) or {}
    usr_args.update(task_name=args.task, task_config=args.task_config, seed=args.seed,
                    policy_name=args.candidate_policy)
    chunk = int(usr_args.get("pi0_step", args.chunk))

    # 1) build the search scene + capture the root/provenance
    task, task_args = build_scene(args.task, args.task_config, args.seed)
    root = rb.snapshot(task)

    # 2) load candidate policy + build the search components
    model, encode_obs = load_candidate(args.candidate_policy, usr_args)
    candidates = PolicyCandidates(model, encode_obs, mode=args.candidate_mode,
                                  chunk=chunk, noise_seed=args.noise_seed)
    fitness = fit_mod.build_fitness(args.fitness, use_dist_tiebreak=not args.no_dist_tiebreak)
    strategy = search_mod.build_strategy(args.strategy, width=args.width, k=args.k,
                                         depth=args.depth)

    # 3) search the future-tree for the top-M distinct plans (best first)
    top_m = max(1, int(args.top_m))
    results = strategy.search(task, root, candidates, fitness, depth=args.depth, m=top_m)
    print(f"[run_search] {args.task}/{args.task_config} seed{args.seed}: "
          f"strategy={args.strategy} top_m={top_m} got={len(results)} "
          f"best_success={results[0].success} best_score={results[0].score}", flush=True)
    task.close_env()

    # 4+5) VERIFY each ranked plan by a fresh from-start replay, then write it.
    atol = args.atol
    created_at = now_iso()                          # one timestamp across all ranks
    out_dir = os.path.join(args.out, args.task, args.task_config)
    out_paths = []
    for rank, result in enumerate(results):
        action_dim = int(result.actions.shape[1]) if result.actions.size else 0
        t0_fp, term_fp, replay_success = verify_from_start(
            args.task, args.task_config, args.seed, result.actions, atol)
        fp_match = (term_fp.shape == result.terminal_fingerprint.shape
                    and bool(np.allclose(term_fp, result.terminal_fingerprint, atol=atol, rtol=0.0)))
        success_match = (replay_success == result.success)
        verified = bool(fp_match and success_match)
        if not verified:
            print(f"\033[93m[run_search] rank{rank} VERIFY MISMATCH fp_match={fp_match} "
                  f"success_match={success_match} (search={result.success} "
                  f"replay={replay_success}) — writing plan with verified=False\033[0m",
                  flush=True)

        spec = build_task_spec(args.task, args.task_config, args.seed, task_args,
                               action_dim, chunk, repo_dir, created_at=created_at)
        plan = plan_mod.Plan(
            task_spec=spec,
            actions=result.actions.tolist(),
            fingerprints={"t0": t0_fp.tolist(), "terminal": term_fp.tolist()},
            meta={
                "strategy": args.strategy,
                "strategy_params": {"width": args.width, "k": args.k, "depth": args.depth},
                "fitness": args.fitness,
                "candidate_policy": args.candidate_policy,
                "candidate_mode": args.candidate_mode,
                "rank": rank,
                "top_m": top_m,
                "score": list(result.score),
                "outcome": result.outcome,
                "success": bool(result.success),
                "candidate_indices": result.candidate_indices,
                "verified": verified,
                "verify": {"fingerprint_match": fp_match, "success_match": success_match,
                           "replay_success": replay_success, "atol": atol},
            },
        )
        # rank-0 -> seed<N>.json; rank-r -> seed<N>_rank<r>.json
        fname = f"seed{args.seed}.json" if rank == 0 else f"seed{args.seed}_rank{rank}.json"
        out_path = os.path.join(out_dir, fname)
        plan_mod.save(plan, out_path)
        print(f"[run_search] wrote {out_path} (rank={rank} verified={verified} "
              f"success={result.success} score={result.score})", flush=True)
        out_paths.append(out_path)

    return out_paths[0]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a lookahead search -> verified motion Plan.")
    p.add_argument("--task", required=True, help="task name (envs/<task>.py)")
    p.add_argument("--task_config", required=True, help="task config yml stem")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--candidate_policy", default="pi05",
                   help="policy module name under ./policy (implements get_model/encode_obs)")
    p.add_argument("--policy_config", required=True, help="path to the policy's deploy_policy.yml")
    p.add_argument("--candidate_mode", default="modes", choices=["modes", "noise"],
                   help="draw candidates by latent mode z (WTA) or by noise sampling")
    p.add_argument("--strategy", default="beam", choices=sorted(search_mod.STRATEGY_REGISTRY))
    p.add_argument("--width", type=int, default=3, help="beam width B")
    p.add_argument("--k", type=int, default=6, help="candidates per node K")
    p.add_argument("--depth", type=int, default=4, help="search depth D (chunks)")
    p.add_argument("--fitness", default="oracle", choices=sorted(fit_mod.FITNESS_REGISTRY))
    p.add_argument("--no_dist_tiebreak", action="store_true",
                   help="disable the distance-to-goal tiebreak in OracleFitness")
    p.add_argument("--noise_seed", type=int, default=1234)
    p.add_argument("--atol", type=float, default=1e-5, help="fingerprint match tolerance")
    p.add_argument("--top-m", "--top_m", dest="top_m", type=int, default=1,
                   help="number of top distinct plans to emit (rank-0 -> seed<N>.json, "
                        "rank-r -> seed<N>_rank<r>.json)")
    p.add_argument("--out", required=True, help="output dir root for plans")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
