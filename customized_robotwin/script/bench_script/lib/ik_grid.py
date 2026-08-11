"""Frame transforms and batched IK-grid solvers."""

import numpy as np
import transforms3d as t3d


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
    import torch
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


def build_grid(args):
    """Regular (x, y) lattice of gripper positions (constant across z) plus the z-slice axis for the
    stack. Returns xs, ys, zs (1D axes) and XX, YY (ny, nx meshgrids); the stack is swept over zs at
    run time (x,y do not depend on z, so we keep XX/YY 2D and vary z as a scalar per slice)."""
    xs = np.arange(args.xmin, args.xmax + 1e-9, args.res)
    ys = np.arange(args.ymin, args.ymax + 1e-9, args.res)
    zs = np.arange(args.zmin, args.zmax + 1e-9, args.zres)
    XX, YY = np.meshgrid(xs, ys)                      # (ny, nx)
    return xs, ys, zs, XX, YY


def _build_ik_solver_no_world(planner, num_seeds=100):
    """Companion to reachability_map._build_ik_solver, but with NO collision world at all,
    so IK success == pure kinematic reachability + self-collision. This is the 'collisions-OFF'
    sweep: a cell that fails here is BEYOND-REACH (kinematic or self-collision), not
    clutter-blocked. The table therefore falls into OBSTACLE, as intended.

    We pass world_model=None (and no world_coll_checker): curobo only builds the primitive
    collision cost when a checker exists (arm_base.py: `... and world_coll_checker is not None`),
    so with both None there is no world-collision cost -- an EMPTY world would instead keep the
    cost active and error with 'Primitive Collision has no obstacles'."""
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    mg = planner.motion_gen
    cfg = IKSolverConfig.load_from_robot_config(
        planner.yml_path,
        None,                                   # world_model=None -> no checker, no world-collision cost
        tensor_args=mg.tensor_args,
        num_seeds=num_seeds,                    # fewer seeds = faster IK (Tier-1 speedup knob)
        use_cuda_graph=False,
        self_collision_check=True,
        self_collision_opt=False,
    )
    return IKSolver(cfg)


def _solve_grid_q(robot, planner, ik, arm_tag, gp_world, chunk=256, num_seeds=None):
    """Like reachability_map._solve_grid, but ALSO returns the IK joint solution per pose -- the
    fuel for the joint-space gate. curobo already computes it inside solve_batch, so returning it
    costs no extra IK; the 2D tool simply throws it away by reading only result.success.
    num_seeds (None = solver default 100) caps the per-pose seed optimisation (Tier-1 speedup).

    Returns (success (N,) bool, q (N, dof) float32; rows where success is False are NaN)."""
    import torch
    from curobo.types.math import Pose as CuroboPose
    ta = planner.motion_gen.tensor_args
    N = len(gp_world)
    pos = np.empty((N, 3), dtype=np.float32)
    quat = np.empty((N, 4), dtype=np.float32)
    for i, gp in enumerate(gp_world):
        p, q = _world_gripper_to_curobo(robot, planner, arm_tag, gp)
        pos[i], quat[i] = p, q
    succ = np.zeros(N, dtype=bool)
    qout = None
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        goal = CuroboPose(position=ta.to_device(pos[s:e]), quaternion=ta.to_device(quat[s:e]))
        result = ik.solve_batch(goal, num_seeds=num_seeds)
        sc = result.success.detach().cpu().numpy().reshape(e - s, -1)[:, 0].astype(bool)
        # result.solution is (batch, n_seeds, dof); take seed 0 to match the [:, 0] used on success.
        sol = result.solution.detach().cpu().numpy()
        sol = sol.reshape(e - s, -1, sol.shape[-1])[:, 0, :]        # (batch, dof)
        if qout is None:
            qout = np.full((N, sol.shape[-1]), np.nan, dtype=np.float32)
        succ[s:e] = sc
        qout[s:e] = sol
        del result, goal
        torch.cuda.empty_cache()
    return succ, qout


def _solve_grid_q_multi(robot, planner, ik, arm_tag, gp_world, chunk=256, return_seeds=8, num_seeds=None):
    """Multi-branch IK: like _solve_grid_q but returns the top-`return_seeds` CONVERGED solutions
    per pose. curobo already optimises ~100 seeds internally per pose; asking for K of them back
    (instead of only the best) is nearly free and hands us the distinct IK branches at each cell --
    the menu the warm-start propagation chooses among to enforce branch continuity.

    Returns (cand_q (N, K, dof) float32, NaN where that candidate did not converge; cand_ok (N, K) bool)."""
    import torch
    from curobo.types.math import Pose as CuroboPose
    ta = planner.motion_gen.tensor_args
    N = len(gp_world)
    pos = np.empty((N, 3), dtype=np.float32)
    quat = np.empty((N, 4), dtype=np.float32)
    for i, gp in enumerate(gp_world):
        p, q = _world_gripper_to_curobo(robot, planner, arm_tag, gp)
        pos[i], quat[i] = p, q
    cand_q = None
    cand_ok = np.zeros((N, return_seeds), dtype=bool)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        goal = CuroboPose(position=ta.to_device(pos[s:e]), quaternion=ta.to_device(quat[s:e]))
        result = ik.solve_batch(goal, return_seeds=return_seeds, num_seeds=num_seeds)   # top-K branches per pose
        sol = result.solution.detach().cpu().numpy()                 # (b, K, dof)
        ok = result.success.detach().cpu().numpy().reshape(e - s, -1).astype(bool)  # (b, K)
        if cand_q is None:
            cand_q = np.full((N, return_seeds, sol.shape[-1]), np.nan, dtype=np.float32)
        cand_q[s:e] = sol[:, :return_seeds]
        cand_ok[s:e] = ok[:, :return_seeds]
        del result, goal
        torch.cuda.empty_cache()
    cand_q[~cand_ok] = np.nan
    return cand_q, cand_ok


def grasp_orientation(env, arm_tag, topdown):
    """The fixed EE orientation the whole slice is evaluated at: the target's own
    horizontal side-grasp for this arm (or top-down). Returns (grasp_q, grasp_pose)
    where grasp_pose is the full [x,y,z,qw,qx,qy,qz] world grasp (or None)."""
    if topdown:
        return np.array([0, 1, 0, 0], dtype=float), None
    cp_id = env._pick_side_grasp_id(env.target_obj, arm_tag)
    grasp_pose = env._geometric_grasp_pose(env.target_obj, cp_id, pre_dis=0.0) if cp_id is not None else None
    grasp_q = np.array(grasp_pose[-4:]) if grasp_pose is not None else np.array([0, 1, 0, 0], dtype=float)
    return grasp_q, grasp_pose


def select_arm(env, arm_choice, topdown, chunk):
    """Resolve which arm the metric runs on and return (arm, planner, grasp_q, grasp_pose, ik),
    with the chosen arm's IK solver already built (so run() doesn't rebuild it).

    arm_choice == 'left'/'right' -> use that arm.
    arm_choice == 'auto'         -> probe BOTH arms' grasp reachability and pick a reachable one;
                                  nearest arm-base breaks ties (prefer the arm that isn't
                                  over-extending). If neither grasp is reachable, fall back to the
                                  nearest arm and warn (the metric will then read inaccessible)."""
    import torch

    def planner_for(arm):
        return env.robot.left_planner if arm == "left" else env.robot.right_planner

    if arm_choice in ("left", "right"):
        planner = planner_for(arm_choice)
        grasp_q, grasp_pose = grasp_orientation(env, arm_choice, topdown)
        return arm_choice, planner, grasp_q, grasp_pose, _build_ik_solver(planner)

    tgt_xy = np.array(env.target_obj.get_pose().p)[:2]
    cands = []
    for arm in ("left", "right"):
        planner = planner_for(arm)
        grasp_q, grasp_pose = grasp_orientation(env, arm, topdown)
        base = np.array(planner.robot_origion_pose.p)[:2]
        ref = np.array(grasp_pose[:2]) if grasp_pose is not None else tgt_xy
        dist = float(np.hypot(ref[0] - base[0], ref[1] - base[1]))
        reachable, ik = None, None
        if grasp_pose is not None:
            ik = _build_ik_solver(planner)
            reachable = bool(_solve_grid(env.robot, planner, ik, arm,
                                         np.array([grasp_pose]), chunk=chunk)[0])
        cands.append(dict(arm=arm, planner=planner, grasp_q=grasp_q, grasp_pose=grasp_pose,
                          dist=dist, reachable=reachable, ik=ik))
        print(f"[auto-arm] {arm}: grasp reachable={reachable}  base-dist={dist:.3f}m")

    reach = [c for c in cands if c["reachable"]]
    pool = reach if reach else cands
    pool.sort(key=lambda c: c["dist"])
    chosen = pool[0]
    if not reach:
        print(f"[auto-arm] WARNING no arm's grasp is reachable; falling back to nearest "
              f"({chosen['arm']}) -> metric will likely read inaccessible")
    # release IK solvers built for the arm(s) we did not choose
    for c in cands:
        if c is not chosen and c["ik"] is not None:
            del c["ik"]
    torch.cuda.empty_cache()
    ik = chosen["ik"] if chosen["ik"] is not None else _build_ik_solver(chosen["planner"])
    print(f"[auto-arm] chosen: {chosen['arm']}")
    return chosen["arm"], chosen["planner"], chosen["grasp_q"], chosen["grasp_pose"], ik
