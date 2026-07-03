#!/usr/bin/env python3
"""
Collision-free reachability map via curobo batched IK (issue #35, subgoal search).

Builds the occluder scene (milk-box in the collision world), then for a grid of
(x, y) GRIPPER positions at a fixed height and orientation, batch-solves
collision-free IK per arm and plots which cells are reachable. Reachable ==
curobo found a joint solution within limits, self-collision-free AND world-
collision-free (box + table). Overlays the box footprint, target and pad.

This tells you where a SUBGOAL may legally sit (a valid collision-free EE pose),
NOT whether a path exists between two cells -- but a chain of nearby reachable
cells from behind the box to the pad is what trajopt can actually connect.

USAGE (from the benchmark folder, env sourced + ROBOTWIN_BENCH_TASK=bench):
    python reachability_map.py --seed 1 --offset 0.2 --res 0.02
    python reachability_map.py --arms right --z 0.90

NOTE: reachability is orientation-specific. By default we use the target's own
horizontal side-grasp orientation (per arm). Pass --topdown for a top-down quat.
"""

import argparse
from pathlib import Path

import numpy as np
import transforms3d as t3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setup_paths import setup_paths
setup_paths()

import torch  # noqa: E402  (after setup_paths so curobo's torch is on the path)
from analyze_occluder_visibility import make_occluder_task, PAD_XY, OCC_HALF_FOOTPRINT  # noqa: E402
from analyze_natural_visibility import build_cfg, DR_CLEAN  # noqa: E402


# ----------------------------------------------------------------------------- frames
def _world_gripper_to_curobo(robot, planner, arm_tag, gripper_pose):
    """Replicate robot.left/right_plan_path + planner.plan_path frame chain for a single
    WORLD gripper pose [x,y,z,qw,qx,qy,qz] -> (pos3, quat4) in curobo's IK frame.
    (aloha-agilex branch, since the bench uses that embodiment.)"""
    # 1) gripper -> endlink (robot frame convention)
    endlink = robot._trans_from_gripper_to_endlink(gripper_pose, arm_tag=str(arm_tag))
    # 2) world -> arm base
    world_base = np.concatenate([np.array(planner.robot_origion_pose.p),
                                 np.array(planner.robot_origion_pose.q)])
    world_target = np.concatenate([np.array(endlink.p), np.array(endlink.q)])
    p, q = planner._trans_from_world_to_base(world_base, world_target)
    # 3) frame_bias + small per-arm yaw (aloha-agilex patch, see planner.plan_path)
    T_target = t3d.affines.compose(p, t3d.quaternions.quat2mat(q), [1, 1, 1])
    T_bias = t3d.affines.compose(planner.frame_bias, np.eye(3), [1, 1, 1])
    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02 if str(arm_tag) == "left" else -0.01)
    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
    T_new = T_rot @ T_bias @ T_target
    return T_new[:3, 3], t3d.quaternions.mat2quat(T_new[:3, :3])


def _build_ik_solver(planner):
    """Fresh IKSolver that REUSES the planner's already-loaded collision world, with
    use_cuda_graph=False so we can pass an arbitrary-size batch (the motion_gen ik_solver
    locks its batch size via cuda-graph, so we can't reuse it directly)."""
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    mg = planner.motion_gen
    cfg = IKSolverConfig.load_from_robot_config(
        planner.yml_path,
        None,                                   # world comes from the shared checker below
        tensor_args=mg.tensor_args,
        world_coll_checker=mg.world_coll_checker,   # <- already has the box + table
        use_cuda_graph=False,
        self_collision_check=True,
        self_collision_opt=False,
    )
    return IKSolver(cfg)


def _solve_grid(robot, planner, ik, arm_tag, gp_world, chunk=256):
    """gp_world: (N,7) world gripper poses -> (N,) bool reachable+collision-free.
    Solved in chunks: batched IK spawns many seeds per pose, so the whole grid at once
    is a multi-GB allocation -> chunk to cap peak memory (safe since use_cuda_graph=False)."""
    from curobo.types.math import Pose as CuroboPose
    ta = planner.motion_gen.tensor_args
    N = len(gp_world)
    pos = np.empty((N, 3), dtype=np.float32)
    quat = np.empty((N, 4), dtype=np.float32)
    for i, gp in enumerate(gp_world):
        p, q = _world_gripper_to_curobo(robot, planner, arm_tag, gp)
        pos[i], quat[i] = p, q
    out = np.zeros(N, dtype=bool)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        goal = CuroboPose(position=ta.to_device(pos[s:e]), quaternion=ta.to_device(quat[s:e]))
        result = ik.solve_batch(goal)
        out[s:e] = result.success.detach().cpu().numpy().reshape(e - s, -1)[:, 0].astype(bool)
        del result, goal
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- run
def run(args):
    env = make_occluder_task()()
    env.spawn_occluder = True
    env.occluder_offset = args.offset
    env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, args.seed, DR_CLEAN))

    box_p = np.array(env.occluder.get_pose().p)
    tgt_p = np.array(env.target_obj.get_pose().p)
    pad_xy = np.array(PAD_XY)
    print(f"box=({box_p[0]:.3f},{box_p[1]:.3f})  target=({tgt_p[0]:.3f},{tgt_p[1]:.3f})  pad={pad_xy}")

    arms = {"left": "left", "right": "right"} if args.arms == "both" else {args.arms: args.arms}

    # grid
    xs = np.arange(args.xmin, args.xmax + 1e-9, args.res)
    ys = np.arange(args.ymin, args.ymax + 1e-9, args.res)
    XX, YY = np.meshgrid(xs, ys)                      # (ny, nx)
    z = args.z

    reach_any = np.zeros(XX.shape, dtype=bool)
    per_arm = {}
    for name in arms:
        planner = env.robot.left_planner if name == "left" else env.robot.right_planner

        # orientation: the target's horizontal side grasp for this arm (or top-down)
        if args.topdown:
            grasp_q = np.array([0, 1, 0, 0], dtype=float)      # gripper pointing down
            grasp_pose = None
        else:
            cp_id = env._pick_side_grasp_id(env.target_obj, name)
            grasp_pose = env._geometric_grasp_pose(env.target_obj, cp_id, pre_dis=0.0) if cp_id is not None else None
            grasp_q = np.array(grasp_pose[-4:]) if grasp_pose is not None else np.array([0, 1, 0, 0], dtype=float)

        gp = np.zeros((XX.size, 7))
        gp[:, 0] = XX.ravel(); gp[:, 1] = YY.ravel(); gp[:, 2] = z
        gp[:, 3:] = grasp_q

        ik = _build_ik_solver(planner)

        # --- self-checks: real grasp pose should be reachable; box centre should not ---
        if grasp_pose is not None:
            chk = _solve_grid(env.robot, planner, ik, name, np.array([grasp_pose]), chunk=args.chunk)
            print(f"[{name}] self-check grasp pose reachable = {bool(chk[0])}  (expect True)")
        inbox = np.array([[box_p[0], box_p[1], z, *grasp_q]])
        chk2 = _solve_grid(env.robot, planner, ik, name, inbox, chunk=args.chunk)
        print(f"[{name}] self-check box-centre reachable = {bool(chk2[0])}  (expect False)")

        succ = _solve_grid(env.robot, planner, ik, name, gp, chunk=args.chunk).reshape(XX.shape)
        per_arm[name] = succ
        reach_any |= succ
        print(f"[{name}] reachable cells: {succ.sum()}/{succ.size}")
        del ik
        torch.cuda.empty_cache()

    _plot(XX, YY, reach_any, per_arm, box_p, tgt_p, pad_xy, z, args)
    try:
        env.close_env()
    except Exception:
        pass


def _plot(XX, YY, reach_any, per_arm, box_p, tgt_p, pad_xy, z, args):
    fig, ax = plt.subplots(figsize=(8, 7))
    extent = [XX.min(), XX.max(), YY.min(), YY.max()]
    ax.imshow(reach_any, origin="lower", extent=extent, cmap="Greens", alpha=0.8, aspect="equal")
    # box footprint (approx square of half-diagonal OCC_HALF_FOOTPRINT)
    h = OCC_HALF_FOOTPRINT
    ax.add_patch(plt.Rectangle((box_p[0] - h, box_p[1] - h), 2 * h, 2 * h,
                               fill=False, edgecolor="red", lw=2, label="occluder"))
    ax.plot(tgt_p[0], tgt_p[1], "b*", ms=16, label="target")
    ax.plot(pad_xy[0], pad_xy[1], "ms", ms=12, label="pad")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Collision-free reachability @ z={z:.2f}m, arms={args.arms}"
                 f"{' (top-down)' if args.topdown else ' (side-grasp quat)'}\n"
                 f"green = reachable+collision-free (curobo IK)")
    ax.legend(loc="upper right")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = "topdown" if args.topdown else "sidegrasp"
    p = out / f"reach_seed{args.seed}_off{args.offset}_z{z:.2f}_{args.arms}_{tag}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean",
                    help="bench_task_config/<name>.yml (same scene config as the occluder experiment)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--offset", type=float, default=0.2, help="occluder offset in front of target (m)")
    ap.add_argument("--arms", choices=["both", "left", "right"], default="both")
    ap.add_argument("--z", type=float, default=0.90, help="EE height for the grid (m)")
    ap.add_argument("--topdown", action="store_true", help="use a top-down quat instead of the side grasp")
    ap.add_argument("--xmin", type=float, default=-0.6)
    ap.add_argument("--xmax", type=float, default=0.6)
    ap.add_argument("--ymin", type=float, default=-0.35)
    ap.add_argument("--ymax", type=float, default=0.35)
    ap.add_argument("--res", type=float, default=0.02, help="grid resolution (m)")
    ap.add_argument("--chunk", type=int, default=256,
                    help="IK poses per batch; lower if you hit CUDA OOM (planners already use ~9GB)")
    ap.add_argument("--out-dir", default="../scripts/validation/results/reachability",
                    help="same repo-root results location as the other bench scripts")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
