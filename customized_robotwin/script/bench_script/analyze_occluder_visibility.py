"""
Phase 2 part 1 (#35): deterministic single-occluder spawn + visibility distribution.

Places the target (001_bottle id 9) in the upper third of the reachable region (in
front of the back furniture) and spawns ONE tall milk-box occluder (038_milk-box id 2,
scale 1.0) in a narrow band just in front (-y, robot/camera side) of the target -- no
clutter, no binary search. Then measures the t=0 countertop `visible_fraction`
distribution across seeds, reusing the Phase 1 harness (measurement + plotting).

visible_fraction = visible_target_px(with occluder) / full_target_px(no occluder),
same seed (target pose is fixed before the occluder is added).

USAGE (from the benchmark folder):
    cd benchmark
    source set_env.sh
    export ROBOTWIN_BENCH_TASK=bench
    python script/bench_script/analyze_occluder_visibility.py \
        --seed-start 0 --num-seeds 50 --offsets 0.10 --bins 20 \
        --out-dir ../scripts/validation/results/phase2_occluder

    # narrow-region sweep (a few offsets) to see how distance affects occlusion:
    python script/bench_script/analyze_occluder_visibility.py \
        --num-seeds 40 --offsets 0.07,0.10,0.13 \
        --out-dir ../scripts/validation/results/phase2_occluder

    # re-plot only:
    python script/bench_script/analyze_occluder_visibility.py --plot-only \
        --out-dir ../scripts/validation/results/phase2_occluder
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import transforms3d as t3d

from setup_paths import setup_paths
setup_paths()
# these analysis scripts are bench-only; default the task mode so build_cfg looks
# under bench_task_config/ (explicit ROBOTWIN_BENCH_TASK still wins)
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
os.chdir(robotwin_root)

from envs.utils import rand_pose, create_actor, create_box, ArmTag
from visualize_task_scene import get_env_class
# reuse the Phase 1 harness verbatim
from analyze_natural_visibility import (build_cfg, DR_CLEAN, save_overlay, analyze, _resolve_target,
                                        CAMERA, run_rollout, analyze_rollout, effective_out_dir)

# target object: 001_bottle id 9 (8-way grasp ring -> side-reaches around the occluder).
# Swap TARGET_MODEL/TARGET_ID back to 047_mouse id 0 for the stock top-down mouse.
TARGET_MODEL = "001_bottle"
TARGET_ID = 9
# target spawn: back half of the table (back furniture removed). Capped at y=0.20 so it
# stays 0.15 m off the back edge (table y in [-0.35, 0.35]); x kept to [-0.15, 0.15] so it
# stays 0.45 m off each side edge (table x in [-0.6, 0.6]) -> leaves room for the around-
# the-box side waypoint and keeps target + waypoint inside the arm's reach envelope.
TARGET_XLIM = (-0.15, 0.15)
TARGET_YLIM = (0.0, 0.20)
PAD_XY = (0.0, -0.28)          # destination pad parked at the front, out of the occluder zone
OCCLUDER_QPOS = [0.66, 0.66, -0.25, -0.25]   # same upright carton orientation as the stock milk-box

# Two-step (around-the-box) planning: curobo won't reliably route around a tall obstacle
# between start and goal, so before the grasp (and before the place) we send the gripper
# to a waypoint beside the occluder on the arm's own side, then reach in. The waypoint's
# x-offset from the box CENTRE = OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP, i.e. SIDE_WAYPOINT
# _GAP is the clearance from the box EDGE to the gripper (not from centre). If the waypoint
# would leave the reachable x-range we flip to the other (table-centre) side. All tunable.
OCC_HALF_FOOTPRINT = 0.08      # milk-box base half-diagonal (~0.11 x 0.122, incl. yaw)
SIDE_WAYPOINT_GAP = 0.24       # clearance from the box EDGE to the waypoint (gripper)
REACH_X_LIMIT = 0.5

# Reject a (seed, offset) build if the occluder would sit within OCC_PAD_CLEARANCE
# (edge-to-edge) of the destination pad, so the occluder can never block/overlap the
# target's final placement. Center-to-center threshold = pad_half + occluder_half + gap.
OCC_PAD_CLEARANCE = 0.05
_PAD_HALF = 0.06               # pad half-size (create_box half_size xy)
_OCC_HALF = 0.06               # milk-box base half-extent (~0.11 x 0.122 footprint)
OCC_PAD_MIN_DIST = _PAD_HALF + _OCC_HALF + OCC_PAD_CLEARANCE


def dr_measure(clutter_density):
    """Domain-randomization for the measurement build: optional default (non-curated)
    table clutter at the given density. Density 0 -> clean (occluder only)."""
    if clutter_density and clutter_density > 0:
        return {"cluttered_table": True, "obstacle_density": int(clutter_density),
                "clean_background_rate": 0}
    return dict(DR_CLEAN)


def make_occluder_task():
    Base = get_env_class("put_mouse_on_pad", bench_subdir="office")

    class OccluderTask(Base):
        # Remove the back furniture (shelf/cabinet/file-holder) for now to free up
        # gripper workspace. Flip back to True (or delete) to restore the full office.
        SPAWN_BACK_FURNITURE = False
        # Lever A: drop the orientation-hold constraint on the final grasp approach so
        # curobo won't reject it with INVALID_PARTIAL_POSE_COST_METRIC. Set False (or
        # delete this flag + the grasp_actor_from_table override below) to restore the
        # stock straight-in, wrist-locked constrained approach.
        DROP_GRASP_ORIENTATION_CONSTRAINT = True
        # When True, play_once stops right after the lift (pickup only) -- used by
        # pickup_reachability_map.py to reach the picked-up state, then probe IK.
        PICKUP_ONLY = False
        spawn_occluder = False
        occluder_offset = 0.2         # metres in front (-y) of the target
        target_model = TARGET_MODEL
        target_id = TARGET_ID
        target_xlim = TARGET_XLIM
        target_ylim = TARGET_YLIM
        fixed_pad_xy = PAD_XY

        def load_actors(self):
            # target: random within the upper-third band (varies per seed -> distribution)
            target_pose = rand_pose(xlim=list(self.target_xlim), ylim=list(self.target_ylim),
                                    qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])
            # scale from the task yaml if present, else None -> model's own scale
            self.target_obj = create_actor(
                scene=self, pose=target_pose, modelname=self.target_model, convex=True,
                model_id=self.target_id,
                scale=self.item_info["scales"].get(self.target_model, {}).get(f"{self.target_id}", None),
            )
            self.target_obj.set_mass(0.05)

            # destination pad (flat; parked at the front so it never conflicts with the occluder)
            px, py = self.fixed_pad_xy
            pad_pose = rand_pose(xlim=[px], ylim=[py], qpos=[1, 0, 0, 0], rotate_rand=False)
            self.color_name, self.color_value = "Gray", (0.5, 0.5, 0.5)
            self.des_obj = create_box(scene=self, pose=pad_pose, half_size=[0.06, 0.06, 0.0005],
                                      color=self.color_value, name="box", is_static=True)
            self.add_prohibit_area(self.des_obj, padding=0.01, area="table")
            self.add_prohibit_area(self.target_obj, padding=0.02, area="table")
            self.des_obj_pose = self.des_obj.get_pose().p.tolist() + [0, 0, 0, 1]
            self.des_obj_pose[2] += 0.02

            # occluder: one tall milk-box, deterministically at mouse-x, just in front (-y)
            if self.spawn_occluder:
                mp = self.target_obj.get_pose().p
                ox, oy = float(mp[0]), float(mp[1]) - self.occluder_offset
                occ_pose = rand_pose(xlim=[ox], ylim=[oy], qpos=OCCLUDER_QPOS,
                                     rotate_rand=True, rotate_lim=[0, 3.14, 0])  # random yaw
                self.occluder = create_actor(
                    scene=self, pose=occ_pose, modelname="038_milk-box", convex=True,
                    model_id=2, scale=[1.0, 1.0, 1.0],
                )
                self.occluder.set_mass(0.1)
                # Register the occluder in curobo's collision world so the planner
                # actually avoids it. NOT flagged is_obstacle -> it survives the
                # exclude_obstacles=True pass used in collision-metrics/eval mode
                # ("... curobo planner skips clutter obstacles"), which only drops
                # is_obstacle=True procedural clutter.
                self.collision_list.append({
                    "actor": self.occluder,
                    "collision_path": f"{os.environ['BENCH_ROOT']}/assets/objects/038_milk-box/collision/base2.glb",
                })

        def play_once(self):
            # Expert plan. FORWARD (grasp) below is fixed. BACKWARD (placement) is a list
            # of subgoals returned by self._backward_subgoal_poses() -- EDIT THAT METHOD to
            # add / remove / reorder placement subgoals; each subgoal is one curobo move.
            target_p0 = self.target_obj.get_pose().p        # original target location (pre-grasp)
            self._place_target_y = float(target_p0[1])      # available to backward subgoals
            arm_tag = ArmTag("right" if target_p0[0] > 0 else "left")

            # Force a horizontal, arm-side-facing grasp. choose_grasp_pose otherwise
            # biases toward top-down / away-facing grips (its score is 0.7*top_down +
            # 0.3*side and it isn't occlusion/side aware), which gives a bad grasp AND a
            # bad first-subgoal orientation, since the waypoint inherits the grasp quat.
            cp_id = self._pick_side_grasp_id(self.target_obj, arm_tag)

            # grasp: go beside the box on the arm's side first, then reach in.
            # NB: compute the waypoint from a GEOMETRIC grasp pose, not choose_grasp_pose
            # -- choose_grasp_pose validates by planning a direct path (from the rest pose,
            # through the box) and returns None when blocked, which would skip the waypoint
            # and cause "can't find a valid pre_grasp_pose". We move to the waypoint first,
            # then grasp_actor_from_table plans the grasp FROM the waypoint (reachable).
            if cp_id is not None and self.spawn_occluder and getattr(self, "occluder", None) is not None:
                gpose = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.1)
                wp = self._around_box_waypoint(arm_tag, gpose) if gpose is not None else None
                if wp is not None:
                    self.move(self.move_to_pose(arm_tag, wp))
            self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1,
                                        contact_point_id=cp_id)

            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
            if self.PICKUP_ONLY:        # stop at the picked-up state (reachability probe)
                return
            self.attach_object(
                self.target_obj,
                f"{os.environ['BENCH_ROOT']}/assets/objects/{self.target_model}/collision/base{self.target_id}.glb",
                str(arm_tag),
            )
            self.enable_table(enable=True)

            # ---- BACKWARD (placement): edit self._backward_subgoal_poses() to change ----
            for name, pose in self._backward_subgoal_poses(arm_tag):
                self.move(self.move_to_pose(arm_tag, pose))
            # place_actor lowers the object from the last subgoal onto the pad.
            # constrain="free": check_success is position-only, so alignment buys nothing
            # and was a likely IK_FAIL cause for a tall object held near its top.
            self.move(self.place_actor(
                self.target_obj, arm_tag=arm_tag, target_pose=self.des_obj_pose,
                constrain="free", pre_dis=0.05, dis=0.005,
            ))

        def _backward_subgoal_poses(self, arm_tag):
            """Ordered BACKWARD (placement) subgoals -> list of (name, EE_pose), each run as
            one curobo move before place_actor.

            >>> EDIT HERE to change the placement path <<<
              - ADD a subgoal:    append a (name, [x, y, z, qw, qx, qy, qz]) tuple
              - REMOVE a subgoal: delete / comment its append line
              - REORDER:          move the append lines around

            Locals already computed for you to build poses from:
              cur    # current EE pose (holding the bottle); quat = cur[3:] (grasp orient.)
              tgt_y  # original target depth, behind the box (self._place_target_y)
              pad    # destination pad (x, y)  (self.fixed_pad_xy)
              x_side # x beside the box on the arm's own side, reach-clamped
            Also available: self.occluder.get_pose().p (milk-box x,y,z), self.des_obj_pose.
            Poses are ABSOLUTE world [x, y, z, qw, qx, qy, qz]; z world (table top ~0.74 m)."""
            if not (self.spawn_occluder and getattr(self, "occluder", None) is not None):
                return []
            cur = list(self.get_arm_pose(arm_tag))
            quat = cur[3:]
            tgt_y = self._place_target_y
            pad = self.fixed_pad_xy
            x_side = self._box_side_x(arm_tag)     # beside the box on the arm's side

            subgoals = []
            # 1) back beside the milk box, arm's side, at the original target depth
            subgoals.append(("beside_box",        [x_side, tgt_y,  cur[2], *quat]))
            # 2) up over the box (z=1.15 m) and forward to the pad's y, same x
            subgoals.append(("over_box_to_pad_y", [x_side, pad[1], 1.15,   *quat]))
            return subgoals

        def _box_side_x(self, arm_tag):
            """World x beside the occluder on the arm's OWN side (right -> +x, left -> -x),
            OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP from the box centre, clamped to the reach
            limit on that same side (never flipped across to the wrong side)."""
            box_x = float(self.occluder.get_pose().p[0])
            side = 1.0 if str(arm_tag) == "right" else -1.0
            x = box_x + side * (OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP)
            return side * REACH_X_LIMIT if abs(x) > REACH_X_LIMIT else x

        def _around_box_waypoint(self, arm_tag, ref_pose):
            """Grasp subgoal: a pose beside the occluder (via _box_side_x), ON the box's
            horizontal (y) line, keeping ref_pose's height/orientation so the reach-in
            sweeps around the box side to the target. None when no occluder is present."""
            if not (self.spawn_occluder and getattr(self, "occluder", None) is not None):
                return None
            wp = list(ref_pose)
            wp[0] = self._box_side_x(arm_tag)
            wp[1] = float(self.occluder.get_pose().p[1])   # box's y-line
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[around_box] arm={arm_tag} -> waypoint "
                      f"x={wp[0]:.3f} y={wp[1]:.3f} z={wp[2]:.3f}")
            return wp

        def _pick_side_grasp_id(self, actor, arm_tag):
            """Contact-point id whose grasp is horizontal AND approaches from the arm's
            own side, so the gripper grabs the target sideways facing it -- instead of the
            top-down / away-facing grip choose_grasp_pose defaults to. Geometric (no
            planning): replicates get_grasp_pose's contact_matrix @ conv, then scores the
            gripper +x approach axis. Returns None if none qualify (-> default selection)."""
            side = 1.0 if str(arm_tag) == "right" else -1.0
            cx = float(actor.get_pose().p[0])
            conv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            best_i, best_score = None, None
            for i, cm in actor.iter_contact_points("matrix"):
                if cm is None:
                    continue
                g = np.asarray(cm, dtype=float) @ conv
                R = g[:3, :3]
                approach = R @ np.array([1.0, 0.0, 0.0])      # gripper +x approach (world)
                horiz = abs(float(approach[2]))               # 0 = perfectly horizontal
                if horiz > 0.35:                              # skip clearly tilted/vertical
                    continue
                grasp_x = float(g[0, 3]) + float((R @ np.array([-0.12, 0.0, 0.0]))[0])
                arm_side = side * (grasp_x - cx)              # >0 = gripper on arm's side
                score = arm_side - horiz                      # arm-side + as horizontal as possible
                if best_score is None or score > best_score:
                    best_score, best_i = score, i
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[grasp_id] arm={arm_tag} picked contact_point_id={best_i}")
            return best_i

        def _geometric_grasp_pose(self, actor, cp_id, pre_dis=0.0):
            """Grasp pose for a contact point, computed geometrically (same math as
            get_grasp_pose but WITHOUT choose_best_pose's planning gate). Used only to
            orient/place the side waypoint, so a box-blocked direct plan can't veto it."""
            cm = actor.get_contact_point(cp_id, "matrix")
            if cm is None:
                return None
            conv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            g = np.asarray(cm, dtype=float) @ conv
            R = g[:3, :3]
            p = g[:3, 3] + R @ np.array([-0.12 - pre_dis, 0.0, 0.0])
            q = t3d.quaternions.mat2quat(R)
            return list(p) + list(q)

        def grasp_actor_from_table(self, actor, arm_tag, pre_grasp_dis=0.1, grasp_dis=0,
                                   gripper_pos=0.0, contact_point_id=None):
            # Same as Office_base_task.grasp_actor_from_table, but (Lever A) strips the
            # orientation-hold constraint_pose from the approach actions before executing.
            if not self.DROP_GRASP_ORIENTATION_CONSTRAINT:
                return super().grasp_actor_from_table(
                    actor, arm_tag, pre_grasp_dis=pre_grasp_dis, grasp_dis=grasp_dis,
                    gripper_pos=gripper_pos, contact_point_id=contact_point_id)
            _, actions = self.grasp_actor(
                actor=actor, arm_tag=arm_tag, pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis, gripper_pos=gripper_pos, contact_point_id=contact_point_id)
            if not actions:            # grasp_actor bails (e.g. an earlier move set
                return                 # plan_success=False) -> avoid actions[0] IndexError
            for a in actions:
                if getattr(a, "args", None):
                    a.args.pop("constraint_pose", None)  # -> unconstrained plan in move()
            self.move((arm_tag, [actions[0]]))
            self.enable_table(enable=False)
            self.move((arm_tag, actions[1:]))

    return OccluderTask


def run(args):
    out_dir = effective_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    offsets = [float(o) for o in args.offsets.split(",")]
    clutter_densities = [int(d) for d in args.clutter_densities.split(",")]
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    rollout = getattr(args, "rollout", False)
    ep_counter = 0   # unique episode id per rollout (-> video/episode{N}.mp4)

    env = make_occluder_task()()
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  offsets={offsets}  "
          f"clutter_densities={clutter_densities}  rollout={rollout}  camera={CAMERA}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        for si, seed in enumerate(seeds):
            # --- denominator: same scene, NO occluder ---
            env.spawn_occluder = False
            try:
                env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed, DR_CLEAN))
                target = _resolve_target(env)
                clean_pose = np.array(target.actor.get_pose().p)
                res_clean = env.measure_target_visibility(target, camera_name=CAMERA)
                full_px = res_clean["visible_pixel_count"]
                if args.save_images:
                    save_overlay(env, res_clean["mask"], img_dir / f"seed{seed:04d}_clean.png",
                                 f"seed {seed} CLEAN (denominator)  full_px={full_px}")
            except Exception as e:
                print(f"[seed {seed}] clean build failed ({type(e).__name__}: {e}); skipping seed")
                safe_close()
                continue
            safe_close()
            if full_px <= 0:
                print(f"[seed {seed}] full_target_px=0 on clean scene; skipping seed.")
                continue

            for off in offsets:
                # per-build coin flip: drop the occluder with prob no_occluder_prob
                # (keyed on seed+offset so the decision is shared across densities)
                show = bool(np.random.default_rng(int(seed) * 1000 + int(round(off * 100))).random()
                            >= args.no_occluder_prob)
                # Reject scenes where the occluder (at target_x, target_y - off) would land
                # too close to / on the destination pad, blocking the target's placement.
                if show:
                    occ_xy = np.array([clean_pose[0], clean_pose[1] - off])
                    pad_dist = float(np.linalg.norm(occ_xy - np.array(PAD_XY)))
                    if pad_dist < OCC_PAD_MIN_DIST:
                        print(f"[seed {seed}] occluder@off={off} is {pad_dist:.3f}m from pad "
                              f"(< {OCC_PAD_MIN_DIST:.3f}m); rejecting this build.")
                        continue
                env.spawn_occluder = show
                env.occluder_offset = off
                for cd in clutter_densities:
                    try:
                        env.setup_demo(**build_cfg("put_mouse_on_pad", args.base_config, seed,
                                                   dr_measure(cd)))
                        target = _resolve_target(env)
                        pose_ok = bool(np.allclose(np.array(target.actor.get_pose().p), clean_pose, atol=1e-4))
                        res = env.measure_target_visibility(target, camera_name=CAMERA, denominator=full_px)
                        if args.save_images:
                            tag = "occ" if show else "noocc"
                            save_overlay(
                                env, res["mask"],
                                img_dir / f"seed{seed:04d}_off{off:.2f}_cd{cd:02d}_{tag}.png",
                                f"seed {seed} off={off:.2f} {tag} clut={cd}  "
                                f"vis_px={res['visible_pixel_count']} full={full_px} "
                                f"frac={res['visible_fraction']:.3f} {res['bucket']} pose_match={pose_ok}",
                            )
                    except Exception as e:
                        print(f"[seed {seed} off{off:.2f} cd{cd}] build failed ({type(e).__name__}: {e}); skipping")
                        safe_close()
                        continue
                    safe_close()

                    rec = {"seed": int(seed), "offset": float(off), "full_px": int(full_px),
                           "visible_px": int(res["visible_pixel_count"]),
                           "visible_fraction": float(res["visible_fraction"]),
                           "bucket": res["bucket"], "in_fov": bool(res["in_fov"]),
                           "pose_match": pose_ok, "occluder_shown": show,
                           "clutter_density": int(cd)}
                    # --- expert curobo rollout on the same scene (video + success) ---
                    # env.spawn_occluder / occluder_offset persist, so the rollout
                    # build reproduces the same occluder placement as the measurement.
                    if rollout:
                        success = run_rollout(env, "put_mouse_on_pad", args.base_config, seed,
                                              dr_measure(cd), out_dir, ep_counter)
                        rec["rollout_success"] = bool(success)
                        rec["rollout_ep"] = ep_counter
                        rec["rollout_video"] = f"video/episode{ep_counter}.mp4"
                        print(f"    seed {seed} off={off:.2f} cd={cd} {res['bucket']}: "
                              f"rollout {'SUCCESS' if success else 'FAIL'} -> episode{ep_counter}.mp4")
                        ep_counter += 1
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
            print(f"[{si+1}/{len(seeds)}] seed {seed}: full_px={full_px} done")

    safe_close()
    print(f"\nsweep complete -> {jsonl_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="bench_demo_office_clean")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--num-seeds", type=int, default=50)
    ap.add_argument("--offsets", default="0.2", help="occluder offset(s) in m in front of the target")
    ap.add_argument("--clutter-densities", default="0",
                    help="table clutter density/densities to sweep in the measurement scene, "
                         "comma-separated (0=off). e.g. 0,8,15")
    ap.add_argument("--group-by", default="offset", choices=["offset", "clutter_density"],
                    help="analysis grouping variable for the bucket/histogram figures")
    ap.add_argument("--no-occluder-prob", type=float, default=0.2,
                    help="probability the milk-box occluder is NOT spawned for a build (default 0.2)")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="../scripts/validation/results/phase2_occluder")
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--rollout", action="store_true",
                    help="run an expert curobo rollout per scene (writes to <out-dir>_rollout, "
                         "saves videos, and adds success-only distribution + P(success) per bucket)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if not args.plot_only:
        run(args)
    group_label = {"offset": "occluder offset (m)",
                   "clutter_density": "table clutter density"}[args.group_by]
    out_dir = effective_out_dir(args)
    analyze_kwargs = dict(group_key=args.group_by, group_label=group_label,
                          suptitle="Visibility with one milk-box occluder (countertop)",
                          bar_title=f"Bucket proportions vs {group_label}")
    if args.rollout:
        analyze_rollout(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)
    else:
        analyze(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)


if __name__ == "__main__":
    main()
