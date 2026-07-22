"""CLI: render a ghost-futures branching video for one scene (mirrors run_search).

Two modes — viz NEVER re-runs a search it was given the output of:

* ``--trace <path>`` (policy-free): load a saved
  :class:`~lookahead.trace.SearchTrace` (from ``run_search --dump-trace`` or a
  previous search-mode run), build a fresh scene at the trace's seed, and
  :func:`lookahead.viz.render_trace` it by pure raw-action playback — NO search,
  NO candidate model, NO policy inference. This is the "viz from a pre-done
  search output" path, exactly like replay consumes a plan.
* default (search mode, GPU + policy): build the deterministic test scene at
  ``--seed`` via :func:`lookahead.run_search.build_scene` (need_plan on,
  save_data off, render_freq 0, ``enable_collision_metrics`` ON —
  :class:`OracleFitness` reads ``task.get_collision_metrics()["is_collision"]``,
  so the flag is required for correct ghost tiers), load the candidate policy
  exactly like ``run_search`` (``load_candidate`` + ``--candidate-opt``
  passthrough), run ``BeamSearch(width, k, depth)`` for the committed spine,
  :func:`~lookahead.viz.capture_trace` the fans, SAVE the trace to
  ``<out>/traces/<scene>.json`` (so later runs can use ``--trace``), then
  :func:`~lookahead.viz.render_trace`.

Both modes finish with the CPU composite:
:func:`lookahead.viz.compose_scene_video` -> annotated MP4 + preview PNGs + a
self-contained embeddable HTML fragment (base64 data-URI video). A pre-existing
bundle under ``<out>/bundles/`` short-circuits straight to the composite.

``--depth`` defaults to the task's bench eval step limit // chunk
(``benchmark/bench_task_config/_bench_eval_step_limit.yml`` value // ``pi0_step``).

Run from ``customized_robotwin`` with the pi05 venv (beam width 1, K=6 fan,
depth = step_lim/50)::

    python -m lookahead.run_viz \
        --task put_stapler_in_drawer --task_config bench_demo_office_clean \
        --seed 40000 --candidate_policy pi05 \
        --policy_config policy/pi05/deploy_policy.yml \
        --width 1 --k 6 \
        --out /work/mohammed/datasets/branching_video/lookahead

    # policy-free re-render from the saved trace:
    python -m lookahead.run_viz \
        --trace /work/mohammed/datasets/branching_video/lookahead/traces/\
put_stapler_in_drawer__bench_demo_office_clean__seed40000.json \
        --out /work/mohammed/datasets/branching_video/lookahead
"""

from __future__ import annotations

import argparse
import functools
import os
import sys

import cv2

# Import the lookahead package whether run as a module or a bare script.
try:
    from . import fitness as fit_mod
    from . import rollback as rb
    from . import run_search as rs
    from . import trace as trace_mod
    from . import viz
    from .candidates import PolicyCandidates, noise_propose
    from .search import BeamSearch
except ImportError:  # pragma: no cover - executed only as a top-level script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lookahead import fitness as fit_mod
    from lookahead import rollback as rb
    from lookahead import run_search as rs
    from lookahead import trace as trace_mod
    from lookahead import viz
    from lookahead.candidates import PolicyCandidates, noise_propose
    from lookahead.search import BeamSearch


def default_depth(task_name: str, chunk: int) -> int:
    """Task's bench eval step limit // chunk (the eval-pipeline convention)."""
    import yaml
    bench_root = os.environ.get("BENCH_ROOT")
    if not bench_root:
        raise SystemExit("--depth omitted and BENCH_ROOT unset; pass --depth "
                         "or set BENCH_ROOT")
    path = os.path.join(bench_root, "bench_task_config", "_bench_eval_step_limit.yml")
    with open(path, "r", encoding="utf-8") as f:
        step_lim = int((yaml.safe_load(f) or {}).get(task_name, 1000))
    return max(1, step_lim // int(chunk))


def search_and_capture(args: argparse.Namespace, usr_args: dict, chunk: int,
                       depth: int, trace_path: str) -> "trace_mod.SearchTrace":
    """Search mode's POLICY stage: BeamSearch the spine, capture + save the trace."""
    task, task_args = rs.build_scene(args.task, args.task_config, args.seed)
    try:
        root = rb.snapshot(task)
        model, encode_obs, propose_fn = rs.load_candidate(args.candidate_policy, usr_args)
        # prefer --instruction, else the task's get_instruction, else generate one
        # the way the eval does (office tasks have no get_instruction).
        instruction = rs.resolve_instruction(task, args.task, args.instruction,
                                             args.instruction_type)
        if hasattr(model, "set_language") and instruction:
            model.set_language(instruction)
        print(f"[run_viz] instruction: {instruction!r}", flush=True)
        opts = rs._parse_candidate_opts(args.candidate_opt)
        has_hook = propose_fn is not None
        print(f"[run_viz] candidate proposer: "
              f"{args.candidate_policy}.propose_candidates" if has_hook
              else "[run_viz] candidate proposer: default noise proposer", flush=True)
        if opts:
            propose_fn = functools.partial(propose_fn or noise_propose, **opts)
        candidates = PolicyCandidates(model, encode_obs, k=args.k,
                                      propose_fn=propose_fn, chunk=chunk)
        fitness = fit_mod.build_fitness(args.fitness)
        strategy = BeamSearch(width=args.width, k=args.k, depth=depth)

        results = strategy.search(task, root, candidates, fitness, depth=depth, m=1)
        committed = results[0]
        print(f"[run_viz] committed spine: depth={committed.depth} "
              f"tier={(committed.outcome or {}).get('tier')} "
              f"success={committed.success} indices={committed.candidate_indices}",
              flush=True)

        action_dim = int(committed.actions.shape[1]) if committed.actions.size else 0
        spec = rs.build_task_spec(args.task, args.task_config, args.seed, task_args,
                                  action_dim, chunk,
                                  os.path.dirname(os.path.abspath(__file__)))
        rb.restore(task, root)               # capture starts from the pristine t0
        trace = viz.capture_trace(
            task, candidates, fitness, committed, horizon=depth, k=args.k,
            task_spec=spec,
            meta={"strategy": "beam",
                  "strategy_params": {"width": args.width, "k": args.k, "depth": depth},
                  "fitness": args.fitness,
                  "candidate_policy": args.candidate_policy,
                  "candidate_hook": has_hook, "candidate_opt": opts or None,
                  "instruction": instruction})
        trace_mod.save(trace, trace_path)
        print(f"[run_viz] wrote trace {trace_path}", flush=True)
        return trace
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


def render_bundle(trace: "trace_mod.SearchTrace", args: argparse.Namespace,
                  bundle_dir: str) -> None:
    """Deterministic, policy-free render: fresh scene at the trace's seed."""
    spec = trace.task_spec
    task, _task_args = rs.build_scene(spec.task_name, spec.task_config, spec.seed)
    try:
        viz.render_trace(task, trace, out_dir=bundle_dir,
                         frame_every=args.frame_every, camera=args.camera)
    finally:
        try:
            task.close_env()
        except Exception:  # noqa: BLE001
            pass


def run(args: argparse.Namespace) -> str:
    if args.trace:                            # policy-free mode: pre-done search output
        trace = trace_mod.load(args.trace)
        spec = trace.task_spec
        print(f"[run_viz] loaded trace {args.trace}: {spec.task_name}/"
              f"{spec.task_config} seed{spec.seed} decisions={trace.n_decisions}",
              flush=True)
        scene_key = f"{spec.task_name}__{spec.task_config}__seed{spec.seed}"
        bundle_dir = os.path.join(args.out, "bundles", scene_key)
        if os.path.exists(os.path.join(bundle_dir, "meta.json")):
            print(f"[run_viz] bundle exists, skipping render: {bundle_dir}", flush=True)
        else:
            render_bundle(trace, args, bundle_dir)
        scene_tag = f"{spec.task_name}  -  {spec.task_config}  seed{spec.seed}"
    else:                                     # search mode
        missing = [n for n in ("task", "task_config", "seed", "policy_config")
                   if getattr(args, n) is None]
        if missing:
            raise SystemExit(f"search mode requires --{' --'.join(missing)} "
                             f"(or pass --trace <path> for the policy-free mode)")
        import yaml
        with open(args.policy_config, "r", encoding="utf-8") as f:
            usr_args = yaml.safe_load(f) or {}
        usr_args.update(task_name=args.task, task_config=args.task_config,
                        seed=args.seed, policy_name=args.candidate_policy)
        chunk = int(usr_args.get("pi0_step") or 50)
        depth = args.depth if args.depth is not None else default_depth(args.task, chunk)
        print(f"[run_viz] {args.task}/{args.task_config} seed{args.seed}: "
              f"width={args.width} k={args.k} depth={depth} chunk={chunk}", flush=True)

        scene_key = f"{args.task}__{args.task_config}__seed{args.seed}"
        bundle_dir = os.path.join(args.out, "bundles", scene_key)
        if os.path.exists(os.path.join(bundle_dir, "meta.json")):
            print(f"[run_viz] bundle exists, skipping sim phase: {bundle_dir}", flush=True)
        else:
            trace_path = os.path.join(args.out, "traces", f"{scene_key}.json")
            trace = search_and_capture(args, usr_args, chunk, depth, trace_path)
            render_bundle(trace, args, bundle_dir)
        scene_tag = f"{args.task}  -  {args.task_config}  seed{args.seed}"

    frames, meta, previews = viz.compose_scene_video(bundle_dir, scene_tag)
    os.makedirs(args.out, exist_ok=True)
    for name, frame in previews.items():
        path = os.path.join(args.out, f"preview_{scene_key}_{name}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[run_viz] preview  {path}", flush=True)
    clip = os.path.join(args.out, f"futures_{scene_key}.mp4")
    viz.write_mp4(frames, clip)
    fragment = os.path.join(args.out, f"fragment_{scene_key}.html")
    viz.write_fragment([clip], fragment)
    print(f"[run_viz] decisions={meta['decisions']}", flush=True)
    print(f"[run_viz] bundle   {bundle_dir}", flush=True)
    print(f"[run_viz] video    {clip}", flush=True)
    print(f"[run_viz] fragment {fragment}", flush=True)
    return clip


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a ghost-futures branching video from a lookahead "
                    "search (or, with --trace, from a pre-done SearchTrace).")
    p.add_argument("--trace", default=None, metavar="PATH",
                   help="saved SearchTrace JSON -> policy-free render (no search, "
                        "no model); scene identity comes from the trace")
    p.add_argument("--task", default=None, help="task name (search mode)")
    p.add_argument("--task_config", default=None, help="task config yml stem (search mode)")
    p.add_argument("--seed", type=int, default=None, help="scene seed (search mode)")
    p.add_argument("--candidate_policy", default="pi05",
                   help="policy module name under ./policy (implements get_model/"
                        "encode_obs; may expose a propose_candidates hook)")
    p.add_argument("--policy_config", default=None,
                   help="path to the policy's deploy_policy.yml (search mode)")
    p.add_argument("--candidate-opt", "--candidate_opt", dest="candidate_opt",
                   action="append", default=None, metavar="KEY=VAL",
                   help="opaque key=val passed to the candidate proposer (repeatable)")
    p.add_argument("--instruction", default=None,
                   help="language instruction for the candidate policy (else the task's "
                        "get_instruction, else generated like the eval)")
    p.add_argument("--instruction_type", default="unseen",
                   help="instruction pool when generating (seen/unseen; default unseen)")
    p.add_argument("--k", type=int, default=6, help="candidate futures per decision K")
    p.add_argument("--width", type=int, default=1,
                   help="beam width B for the committed spine")
    p.add_argument("--depth", type=int, default=None,
                   help="horizon in chunks (default: bench eval step limit // chunk)")
    p.add_argument("--fitness", default="oracle", choices=sorted(fit_mod.FITNESS_REGISTRY))
    p.add_argument("--out", required=True,
                   help="output root (bundles/ + traces/ + previews + MP4 + fragment)")
    p.add_argument("--frame_every", type=int, default=3,
                   help="record a frame every N control steps")
    p.add_argument("--camera", default="countertop_camera",
                   help="static camera name to record")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
