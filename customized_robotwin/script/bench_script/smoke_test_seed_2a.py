#!/usr/bin/env python3
"""
Smoke test for Phase 2a: seed_from_clearance.compute_route_configs / build_seed on a REAL scene.

Validates the one piece of the seed pipeline that has never executed (0a/0b/1 proved the plumbing
bit-exact; 2b/2c are CPU-unit-tested). It builds one occluder scene exactly like
clearance_metric_3d.run(), runs the grasp-approach route builder, and reports whether:
  1. the current-gripper and grasp positions SNAP to FREE voxels (else no seed ever),
  2. the GATED widest-path connects gripper -> grasp (else seeding never fires),
  3. the route is sane (climbs, reasonable length) and 2a runs without exceptions,
  4. the produced seed tensor is (1,1,H,dof), and the planner's real action_horizon == the
     builder's hardcoded default (28) -- if not, Phase 3 must pass the real value.

COARSE grid by default (res 0.03 / zres 0.06) so this is a fast one-off, NOT the full ~1.7e5-voxel
metric run. A NO-ROUTE result is diagnostic, not a crash: it tells us exactly which of 1-3 to fix
before wiring Phase 3.

USAGE (env sourced + ROBOTWIN_BENCH_TASK=bench):
    python smoke_test_seed_2a.py --seed 1 --arm auto
    python smoke_test_seed_2a.py --seed 1 --arm right --res 0.02 --zres 0.04   # finer, slower
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import clearance_metric_3d as cm
import seed_from_clearance as sfc


def main():
    ap = argparse.ArgumentParser(description="Phase 2a smoke test (build_seed on a real scene)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arm", choices=["left", "right", "auto"], default="auto")
    ap.add_argument("--offset", type=float, default=0.2, help="occluder ring radius (m)")
    ap.add_argument("--num-occluders", type=int, default=1)
    ap.add_argument("--occluder-angle0", type=float, default=0.0)
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--topdown", action="store_true")
    ap.add_argument("--chunk", type=int, default=256)
    # coarse metric knobs -> fast smoke test (override for a finer check)
    ap.add_argument("--res", type=float, default=0.03, help="grid resolution (m); coarse for speed")
    ap.add_argument("--zres", type=float, default=0.06, help="vertical slice spacing (m); coarse for speed")
    ap.add_argument("--ik-seeds", type=int, default=20)
    ap.add_argument("--out-dir", default=str(cm.RESULTS_DIR / "seed_smoke_2a"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] writing to {out_dir}")
    tm = cm.Timings()

    # --- scene (identical setup to clearance_metric_3d.run) ---
    with tm.section("scene_setup"):
        env = cm.make_occluder_task()()
        env.spawn_occluder = True
        env.occluder_offset = args.offset
        env.num_occluders = args.num_occluders
        env.occluder_angle0 = args.occluder_angle0
        env.setup_demo(**cm.build_cfg("put_mouse_on_pad", args.base_config, args.seed, cm.DR_CLEAN))

    # --- resolve arm + grasp orientation/pose + its IK solver (same as run) ---
    with tm.section("select_arm"):
        sel_args = SimpleNamespace(arm=args.arm, topdown=args.topdown, chunk=args.chunk)
        arm, planner, grasp_q, grasp_pose, ik = cm.select_arm(env, sel_args)
    print(f"[smoke] arm={arm}  grasp_pose={'None' if grasp_pose is None else np.round(grasp_pose, 3)}")

    # start (current gripper) + goal (grasp) world positions -> the two widest-path seeds
    try:
        ee_pose = env.robot.get_left_ee_pose() if arm == "left" else env.robot.get_right_ee_pose()
        start_xyz = np.asarray(ee_pose[:3], dtype=float)
    except Exception as e:
        print(f"[smoke] ABORT: could not read gripper pose ({e})")
        return
    if grasp_pose is None:
        print("[smoke] ABORT: no grasp pose (topdown/unreachable) -> nothing to approach")
        return
    goal_xyz = np.asarray(grasp_pose[:3], dtype=float)
    print(f"[smoke] start(gripper) xyz={np.round(start_xyz, 3)}   goal(grasp) xyz={np.round(goal_xyz, 3)}")

    # planner's real action_horizon -> validate the builder's hardcoded 28
    try:
        H = int(planner.motion_gen.trajopt_solver.action_horizon)
    except Exception as e:
        H = 28
        print(f"[smoke] could not read action_horizon ({e}); using default {H}")
    print(f"[smoke] planner action_horizon = {H}   (seed builder default = 28"
          + ("  OK" if H == 28 else "  MISMATCH -> Phase 3 must pass the real value") + ")")

    # --- 2a: compute the grasp-approach route (coarse grid) ---
    cfg = sfc.SeedMetricConfig(res=args.res, zres=args.zres, ik_seeds=args.ik_seeds, chunk=args.chunk)
    print(f"[smoke] running compute_route_configs (coarse res={cfg.res} zres={cfg.zres} "
          f"ik_seeds={cfg.ik_seeds}) -- this is the slow part ...")
    with tm.section("compute_route_configs"):
        res = sfc.compute_route_configs(env, planner, arm, ik, grasp_q, start_xyz, goal_xyz, cfg)

    print("\n===== RouteResult =====")
    print(f"  merged (gated connect) : {res.merged}")
    print(f"  eps_gated (m)          : {res.eps_gated}")
    print(f"  reason (if no route)   : {res.reason}")
    n_route = 0 if res.route_qs is None else len(res.route_qs)
    print(f"  route length (voxels)  : {n_route}")
    if res.route_world:
        zc = [p[2] for p in res.route_world]
        print(f"  route z span (m)       : {min(zc):.3f} -> {max(zc):.3f}")
    print(f"  metric wall time       : {res.seconds:.1f}s")

    # --- gate diagnostic: geometry-fail vs gate-seam-fail, and the tau threshold that connects ---
    # reuses the ONE metric run (res.edt / res.q_warm_3d / seeds) -- no extra IK.
    tau_connect = None
    if res.edt is not None and res.seed_start is not None:
        print("\n----- gate diagnostic (reuses the same metric run) -----")
        print(f"  ungated (reach+clear only): merged={res.merged_ungated}  eps={res.eps_ungated:.3f} m")
        if not res.merged_ungated:
            print("  -> even WITHOUT the gate the seeds don't connect: this is GEOMETRY/REACHABILITY, "
                  "not the gate (try finer --res/--zres or wider bounds), the tau sweep won't help")
        free_vol = res.label == cm.FREE
        for tau in (0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
            e, _b, m = cm.widest_path_eps_3d(res.label, res.edt, res.q_warm_3d,
                                             res.seed_start, res.seed_goal, tau)
            n = 0
            if m:
                rt = cm.reconstruct_widest_path_3d(free_vol, res.edt, res.q_warm_3d,
                                                   res.seed_start, res.seed_goal, e, tau)
                n = len(rt) if rt else 0
                if tau_connect is None:
                    tau_connect = tau
            print(f"  gated tau={tau:<4}: merged={m}  eps={e:.3f}  route_voxels={n}")
        if tau_connect is not None:
            print(f"  -> the gated route CONNECTS at tau>={tau_connect} (builder default tau={cfg.gate_tau})")

    # --- 2c: shape into the seed tensor (route endpoints as stand-in start/goal configs; the real
    #     start_q/goal_q ordering is a Phase-3 concern -- here we only confirm the tensor shapes out) ---
    seed_shape = None
    if res.route_qs is not None:
        seed = sfc.route_qs_to_seed_tensor(res.route_qs, res.route_qs[0], res.route_qs[-1], H)
        seed_shape = tuple(seed.shape)
        dof = res.route_qs.shape[1]
        print(f"  seed tensor            : shape={seed_shape} dtype={seed.dtype} dev={seed.device.type}")
        assert seed_shape == (1, 1, H, dof), f"seed tensor shape {seed_shape} != (1,1,{H},{dof})"

    ok = res.route_qs is not None
    summary = {
        "SMOKE_PASS": bool(ok),
        "arm": arm, "seed": args.seed, "offset": args.offset,
        "merged": bool(res.merged),
        "eps_gated_m": (None if not np.isfinite(res.eps_gated) else round(float(res.eps_gated), 4)),
        "eps_gated_unbounded": bool(np.isinf(res.eps_gated)),
        "route_len_voxels": int(n_route),
        "reason": res.reason,
        "action_horizon": H, "action_horizon_matches_default_28": (H == 28),
        "seed_tensor_shape": (None if seed_shape is None else list(seed_shape)),
        "start_xyz": [round(float(c), 4) for c in start_xyz],
        "goal_xyz": [round(float(c), 4) for c in goal_xyz],
        "cfg": {"res": cfg.res, "zres": cfg.zres, "ik_seeds": cfg.ik_seeds},
    }
    (out_dir / "smoke_result.json").write_text(json.dumps(summary, indent=2))
    tm.save(out_dir)

    if ok:
        print(f"\n[smoke] PASS -- 2a produced a {n_route}-voxel route and a {seed_shape} seed tensor. "
              f"Wrote smoke_result.json + timings.json")
    else:
        print(f"\n[smoke] NO-ROUTE (diagnostic, not a crash). reason: {res.reason}")
        print("[smoke] Before Phase 3, this points at which link to fix:")
        print("        - 'unsnappable': gripper/grasp is outside the grid or not FREE at grasp orientation")
        print("          -> widen grid bounds / raise --seed-snap / check the grasp orientation")
        print("        - 'could not connect': the joint gate disconnects gripper->grasp")
        print("          -> the climb-over has a real branch seam (raise --gate-tau or inspect the scene)")
        print("        Wrote smoke_result.json + timings.json")


if __name__ == "__main__":
    main()
