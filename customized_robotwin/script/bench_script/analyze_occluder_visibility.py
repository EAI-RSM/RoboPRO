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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
TARGET_YLIM = (0.1, 0.15)      # lower bound raised to y>=0.1 (keeps the bottle further back)
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
FORWARD_SUBGOAL_Z = 0.85       # fixed EE height for the forward waypoint + lift (higher = more IK-reachable)
REACH_X_LIMIT = 0.5
PAD_BOTTLE_X_INSET = 0.05      # <-- TUNE ME: pad+bottle subgoals shifted this far INWARD in x
                               # (toward table centre). Sign is handled per-arm, so a single
                               # positive number works for BOTH the left and right grasper.

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
        # Optional callback fired just BEFORE each subgoal move during the rollout, with
        # the REAL executed pose. Set by subgoal_reachability_map.py to build a per-subgoal
        # IK reachability map. Signature: hook(name, target_pose7, current_ee7, arm_tag).
        # None -> no-op (normal rollouts are unaffected).
        subgoal_hook = None
        spawn_occluder = False
        occluder_offset = 0.2         # metres in front (-y) of the target
        target_model = TARGET_MODEL
        target_id = TARGET_ID
        target_xlim = TARGET_XLIM
        target_ylim = TARGET_YLIM
        fixed_pad_xy = PAD_XY
        # Reject a scene when the target bottle's long axis tilts more than this from vertical
        # (see check_stable). Upright ~0 deg; a bottle that toppled during settling ~90 deg.
        TARGET_MAX_TILT_DEG = 25.0

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

        def _target_tilt_deg(self):
            """Tilt (deg) of the target bottle's long axis (model-local +y) from world +z.
            ~0 = upright, ~90 = lying on its side. None if the target isn't built yet.
            The spawn pose is always perfectly upright, so any large tilt means it toppled
            during the physics settle."""
            tgt = getattr(self, "target_obj", None)
            if tgt is None:
                return None
            R = t3d.quaternions.quat2mat(np.array(tgt.get_pose().q))
            up = float(np.clip(R[2, 1], -1.0, 1.0))       # world-z component of the local +y axis
            return float(np.degrees(np.arccos(up)))

        def check_stable(self):
            """Stock settle + motion-stability check, PLUS an uprightness gate on the target.
            The base check only rejects actors still MOVING at the end, so a bottle that fell
            and came to rest reads as 'stable'. We additionally reject a target tilted past
            TARGET_MAX_TILT_DEG -> setup_demo raises UnStableError -> the scene is discarded."""
            is_stable, unstable_list = super().check_stable()
            tilt = self._target_tilt_deg()
            if tilt is not None and tilt > self.TARGET_MAX_TILT_DEG:
                is_stable = False
                unstable_list.append(f"{self.target_obj.get_name()}(tilted {tilt:.0f}deg)")
            return is_stable, unstable_list

        def play_once(self):
            # Expert plan. FORWARD (approach) and BACKWARD (placement) are each a list of
            # subgoals returned by self._forward_subgoal_poses() / self._backward_subgoal_poses()
            # -- EDIT THOSE METHODS to add / remove / reorder subgoals; each is one curobo move.
            target_p0 = self.target_obj.get_pose().p        # original target location (pre-grasp)
            self._place_target_y = float(target_p0[1])      # available to backward subgoals
            arm_tag = ArmTag("right" if target_p0[0] > 0 else "left")

            # No occluder in the scene -> nothing to route around. Skip the forced side grasp
            # AND the around-box subgoals, and run a plain grasp -> lift -> place (a normal
            # rollout). Both _forward_/_backward_subgoal_poses already return [] with no box,
            # so nothing below this dereferences self.occluder.
            has_box = self.spawn_occluder and getattr(self, "occluder", None) is not None

            # Force a horizontal, arm-side-facing grasp -- but ONLY when reaching around the
            # box. choose_grasp_pose otherwise biases toward top-down / away-facing grips (its
            # score is 0.7*top_down + 0.3*side and it isn't occlusion/side aware), a bad grasp
            # AND a bad first-subgoal orientation (the waypoint inherits the grasp quat). With
            # no box we want the normal grasp, so cp_id stays None (default selection).
            cp_id = self._pick_side_grasp_id(self.target_obj, arm_tag) if has_box else None

            # ---- FORWARD (approach): edit self._forward_subgoal_poses() to change ----
            # Go beside the box on the arm's side and sweep pad -> box -> bottle at the fixed
            # height, then reach in. Grasp orientation comes from a GEOMETRIC grasp pose (not
            # choose_grasp_pose, which validates via a direct box-blocked plan and returns
            # None): the subgoals move us beside the box first, then grasp_actor_from_table
            # plans the grasp FROM the last subgoal (reachable).
            if has_box and cp_id is not None:
                gpose = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.1)
                quat = gpose[-4:] if gpose is not None else None
                if quat is not None:
                    for name, pose in self._forward_subgoal_poses(arm_tag, quat):
                        frm = list(self.get_arm_pose(arm_tag)) if self.subgoal_hook is not None else None
                        self.move(self.move_to_pose(arm_tag, pose))
                        self._emit_subgoal(name, pose, arm_tag, frm)
            # grasp: emit the geometric contact pose AFTER the grasp (only if reached).
            # Hook-only extra work is guarded so a normal rollout is byte-for-byte unchanged.
            grasp_tp = frm_g = None
            if self.subgoal_hook is not None and cp_id is not None:
                grasp_tp = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.0)
                frm_g = list(self.get_arm_pose(arm_tag))
            self.grasp_actor_from_table(self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1,
                                        contact_point_id=cp_id)
            if grasp_tp is not None:
                self._emit_subgoal("grasp", grasp_tp, arm_tag, frm_g)

            # lift: straight up to the fixed forward height (more IK-reachable than +0.10)
            cur = list(self.get_arm_pose(arm_tag))
            lift_pose = [cur[0], cur[1], FORWARD_SUBGOAL_Z, *cur[3:]]
            self.move(self.move_to_pose(arm_tag, lift_pose))
            self._emit_subgoal("lift", lift_pose, arm_tag, cur)
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
                frm = list(self.get_arm_pose(arm_tag)) if self.subgoal_hook is not None else None
                self.move(self.move_to_pose(arm_tag, pose))
                self._emit_subgoal(name, pose, arm_tag, frm)
            # place_actor lowers the object from the last subgoal onto the pad.
            # constrain="free": check_success is position-only, so alignment buys nothing
            # and was a likely IK_FAIL cause for a tall object held near its top.
            # place subgoal: the pad target at the held orientation (bottle ignored, so the
            # EE reach target is approximated by the object placement pose + held quat).
            place_pose = place_cur = None
            if self.subgoal_hook is not None:
                place_cur = list(self.get_arm_pose(arm_tag))
                place_pose = [self.des_obj_pose[0], self.des_obj_pose[1],
                              self.des_obj_pose[2], *place_cur[3:]]
            self.move(self.place_actor(
                self.target_obj, arm_tag=arm_tag, target_pose=self.des_obj_pose,
                constrain="free", pre_dis=0.05, dis=0.005,
            ))
            if place_pose is not None:
                self._emit_subgoal("place", place_pose, arm_tag, place_cur)

        def _forward_subgoal_poses(self, arm_tag, quat):
            """Ordered FORWARD (approach) subgoals -> list of (name, EE_pose), each run as one
            curobo move BEFORE the grasp. Mirrors _backward_subgoal_poses but all at the fixed
            forward height (FORWARD_SUBGOAL_Z) and sweeping pad -> box -> bottle. `quat` is the
            grasp orientation (so the reach-in faces the target). The last one (fwd_bottle) is
            the pose just before the grasp -- NOT the lift.

            >>> EDIT HERE to change the approach path <<< (same rules as _backward_subgoal_poses)"""
            if not (self.spawn_occluder and getattr(self, "occluder", None) is not None):
                return []
            quat = self._upright_side_quat(quat)   # forward = side pickup orientation, NO spin
            tgt_y = self._place_target_y
            box_y = float(self.occluder.get_pose().p[1])
            pad = self.fixed_pad_xy
            x_side = self._box_side_x(arm_tag)     # beside the box on the arm's side
            x_in = self._x_inset(arm_tag, x_side)  # x_side shifted INWARD (PAD_BOTTLE_X_INSET, either arm)
            z = FORWARD_SUBGOAL_Z
            subgoals = []
            # subgoals.append(("fwd_pad",    [x_in, pad[1], z, *quat]))   # pad y, inset in x
            subgoals.append(("fwd_box",    [x_side, box_y, z, *quat]))  # over to the milk-box y
            subgoals.append(("fwd_bottle", [x_in, tgt_y,  z, *quat]))   # bottle y, inset in x (pre-grasp)
            return subgoals

        def _emit_subgoal(self, name, target_pose, arm_tag, from_ee):
            """Fire self.subgoal_hook (if set) AFTER a subgoal's move, and ONLY if it actually
            reached (plan_success) -- we never probe a goalpoint past what the rollout reached.
            from_ee is the EE pose captured right before this move ('position right before').
            No-op otherwise, so normal rollouts are unaffected."""
            hook = getattr(self, "subgoal_hook", None)
            if hook is None or not getattr(self, "plan_success", True):
                return
            hook(name, list(target_pose),
                 (list(from_ee) if from_ee is not None else None), str(arm_tag))

        def _backward_subgoal_poses(self, arm_tag, quat=None, held_z=None):
            """Ordered BACKWARD (placement) subgoals -> list of (name, EE_pose), each run as
            one curobo move before place_actor.

            >>> EDIT HERE to change the placement path <<<
              - ADD a subgoal:    append a (name, [x, y, z, qw, qx, qy, qz]) tuple
              - REMOVE a subgoal: delete / comment its append line
              - REORDER:          move the append lines around

            Locals already computed for you to build poses from:
              quat   # held-bottle orientation (grasp orient.)
              held_z # current EE height (holding the bottle after the lift)
              tgt_y  # original target/bottle depth (self._place_target_y)
              box_y  # milk-box depth (self.occluder.get_pose().p[1])
              pad    # destination pad (x, y)  (self.fixed_pad_xy)
              x_side # x beside the box on the arm's own side, reach-clamped
              x_in   # x_side shifted INWARD by PAD_BOTTLE_X_INSET (correct sign for either arm)
            Also available: self.occluder.get_pose().p (milk-box x,y,z), self.des_obj_pose.
            Poses are ABSOLUTE world [x, y, z, qw, qx, qy, qz]; z world (table top ~0.74 m).

            quat/held_z default to the live held pose (self.get_arm_pose) for the rollout;
            _planned_subgoals() passes them in so the poses are computable WITHOUT a rollout."""
            if not (self.spawn_occluder and getattr(self, "occluder", None) is not None):
                return []
            if quat is None or held_z is None:
                cur = list(self.get_arm_pose(arm_tag))   # holding the bottle after the lift
                quat = cur[3:] if quat is None else quat
                held_z = cur[2] if held_z is None else held_z
            tgt_y = self._place_target_y
            box_y = float(self.occluder.get_pose().p[1])
            pad = self.fixed_pad_xy
            x_side = self._box_side_x(arm_tag)     # beside the box on the arm's side
            x_in = self._x_inset(arm_tag, x_side)  # x_side shifted INWARD (PAD_BOTTLE_X_INSET, either arm)

            # TEST: lay the bottle FLAT for the carry. After the first backward subgoal we
            # rotate the held orientation 90 deg about world y (upright -> horizontal) and keep
            # that from box_mid through pad_high (place_actor then lowers it). To change which
            # way it lies, flip FLAT_AXIS.
            #
            # Either sign of the 90 deg tilt lays it flat, but they differ in whether the gripper
            # ends up ABOVE the bottle (approach points down = top grasp) or BELOW it (approach
            # points up = bottom grasp). ENFORCE a bottom grasp: of the two candidates keep the one
            # whose gripper +x approach axis (R[:,0]; +x = approach, per _pick_side_grasp_id) has
            # the larger +z component. Without this the top/bottom choice is arm-dependent, since
            # the side-grasp approach-x sign flips between the two arms.
            FLAT_AXIS = [0.0, 1.0, 0.0]
            _flat_cands = [list(t3d.quaternions.qmult(
                               t3d.quaternions.axangle2quat(FLAT_AXIS, s * np.pi / 2), quat))
                           for s in (1.0, -1.0)]
            flat_quat = max(_flat_cands, key=lambda q: t3d.quaternions.quat2mat(q)[2, 0])

            # reverse of the forward curve, on one vertical line (x = x_side): bottle -> over
            # the box -> pad, arcing higher the closer it gets to the pad.
            subgoals = []
            subgoals.append(("bottle",   [x_in, tgt_y,  held_z, *quat]))       # grasp orientation (upright)
            subgoals.append(("box_mid",  [x_side, box_y, 1.1,   *flat_quat]))  # rotate FLAT, over the box
            subgoals.append(("pad_high", [x_in, pad[1], 1.3,   *flat_quat]))  # stay flat, high above pad
            return subgoals

        def _box_side_x(self, arm_tag):
            """World x beside the occluder on the arm's OWN side (right -> +x, left -> -x),
            OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP from the box centre, clamped to the reach
            limit on that same side (never flipped across to the wrong side)."""
            box_x = float(self.occluder.get_pose().p[0])
            side = 1.0 if str(arm_tag) == "right" else -1.0
            x = box_x + side * (OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP)
            return side * REACH_X_LIMIT if abs(x) > REACH_X_LIMIT else x

        def _x_inset(self, arm_tag, x_side):
            """x_side moved INWARD (toward the table centre) by PAD_BOTTLE_X_INSET. Inward is
            -x for the right arm (x_side>0) and +x for the left (x_side<0), so the same
            positive constant works for both graspers."""
            side = 1.0 if str(arm_tag) == "right" else -1.0
            return x_side - side * PAD_BOTTLE_X_INSET

        def _upright_side_quat(self, quat):
            """Enforce an UPRIGHT side-grasp orientation: keep the grasp's (horizontal) APPROACH
            axis but zero any spin/roll about it, so every forward subgoal holds the gripper the
            same clean sideways way it grabs the bottle -- no per-seed wrist twist. The raw grasp
            quat carries the contact-point's roll, which is the likely reason planning fails on
            the first forward subgoals. Approach ~vertical -> returned unchanged (nothing to de-roll)."""
            R = t3d.quaternions.quat2mat(quat)
            x = R[:, 0]                                  # gripper +x = approach direction (world)
            nx = np.linalg.norm(x)
            if nx < 1e-9:
                return list(quat)
            x = x / nx
            y = np.cross([0.0, 0.0, 1.0], x)             # world-up x approach -> horizontal, no roll
            ny = np.linalg.norm(y)
            if ny < 1e-6:                                # approach nearly vertical: leave as-is
                return list(quat)
            y = y / ny
            z = np.cross(x, y)
            return list(t3d.quaternions.mat2quat(np.column_stack([x, y, z])))

        def _planned_subgoals(self, arm_tag=None):
            """Full ordered list of ALL planned subgoals [(name, pose7), ...], computed
            STATICALLY from the scene -- NO rollout needed, since every subgoal x/y/z is a
            function of the target / occluder / pad geometry. Reuses the same _forward_ and
            _backward_subgoal_poses the rollout uses, so edits there flow straight into the
            overview plot. Order = execution order:
                fwd_pad, fwd_box, fwd_bottle, grasp, lift, bottle, box_mid, pad_high, place.
            Orientation is the geometric grasp quat; the held height is FORWARD_SUBGOAL_Z (the
            lift height). Used by subgoal_reachability_map.py. Empty list if no occluder."""
            if not (self.spawn_occluder and getattr(self, "occluder", None) is not None):
                return []
            self._place_target_y = float(self.target_obj.get_pose().p[1])
            if arm_tag is None:
                arm_tag = ArmTag("right" if self.target_obj.get_pose().p[0] > 0 else "left")
            cp_id = self._pick_side_grasp_id(self.target_obj, arm_tag)
            grasp = (self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.0)
                     if cp_id is not None else None)
            quat = list(grasp[-4:]) if grasp is not None else [1.0, 0.0, 0.0, 0.0]
            subs = list(self._forward_subgoal_poses(arm_tag, quat))
            if grasp is not None:
                subs.append(("grasp", list(grasp)))
                subs.append(("lift", [grasp[0], grasp[1], FORWARD_SUBGOAL_Z, *quat]))
            subs += self._backward_subgoal_poses(arm_tag, quat, FORWARD_SUBGOAL_Z)
            subs.append(("place", [self.des_obj_pose[0], self.des_obj_pose[1],
                                   self.des_obj_pose[2], *quat]))
            return subs


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


def save_rollout_video(env, save_path, ep_num):
    """Merge the frames captured during a save_data=True rollout into
    <save_path>/video/episode{ep_num}.mp4 -- the SAME machinery analyze_natural_visibility.
    run_rollout uses. Call AFTER close_env(); no-op if no frames were captured. Keeps only
    the mp4, dropping the heavy per-frame hdf5 byproduct. Returns True if a video was written."""
    try:
        if getattr(env, "FRAME_IDX", 0) > 0:
            env.merge_pkl_to_hdf5_video()
            env.remove_data_cache()
            hdf5 = Path(str(save_path)) / "data" / f"episode{ep_num}.hdf5"
            if hdf5.exists():
                hdf5.unlink()
            print(f"    saved {Path(str(save_path)) / 'video' / f'episode{ep_num}.mp4'}")
            return True
    except Exception as e:
        print(f"    [video ep{ep_num}] merge failed ({type(e).__name__}: {e})")
    return False


def _plot_scene(seed, off, cd, box_p, tgt_p, pad_xy, success, res, out_dir):
    """Top-down scene layout (occluder box / target / pad) for one build, borrowed from
    pickup_reachability_map._plot_seed but WITHOUT the IK reachability grid. Saved for
    every seed so each scene can be eyeballed; labelled SUCCESS/FAILED from the rollout
    (or "no rollout" when --rollout is off). Table extent is x in [-0.6, 0.6], y in
    [-0.35, 0.35]. (Per-subgoal IK reachability MAPS live in subgoal_reachability_map.py.)"""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.35, 0.35)
    if success is None:
        status = "no rollout"
    else:
        status = "SUCCESS" if success else "FAILED"
    if box_p is not None:
        h = OCC_HALF_FOOTPRINT
        ax.add_patch(plt.Rectangle((box_p[0] - h, box_p[1] - h), 2 * h, 2 * h,
                                   fill=False, edgecolor="red", lw=2, label="occluder"))
    ax.plot(tgt_p[0], tgt_p[1], "b*", ms=16, label="target")
    ax.plot(pad_xy[0], pad_xy[1], "ms", ms=12, label="pad")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"seed {seed}  off={off:.2f}  cd={cd}  {status}\n"
                 f"frac={res['visible_fraction']:.3f} {res['bucket']} "
                 f"(vis={res['visible_pixel_count']}px)")
    ax.legend(loc="upper right"); ax.set_aspect("equal")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / f"scene_seed{seed:04d}_off{off:.2f}_cd{cd:02d}.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"    saved scene plot {p}")


def _plot_rollout_target_positions(out_dir):
    """Top-down scatter of every rolled-out target's spawn position, coloured by rollout
    outcome (green=success, red=fail). Reads records.jsonl so it also regenerates under
    --plot-only. One point per rollout record; the same seed can appear more than once
    (different offsets/clutter) at the same position with different outcomes. Saved as
    <out-dir>/rollout_target_positions.png."""
    jsonl = Path(out_dir) / "records.jsonl"
    if not jsonl.exists():
        print(f"[target-pos plot] no records at {jsonl}; skipping"); return
    ok, bad = [], []
    with open(jsonl) as f:
        for line in f:
            r = json.loads(line)
            if "rollout_success" not in r or "target_x" not in r:
                continue          # non-rollout records (or older files) have no position/outcome
            (ok if r["rollout_success"] else bad).append((r["target_x"], r["target_y"]))
    if not ok and not bad:
        print("[target-pos plot] no rollout records with target positions; skipping"); return

    fig, ax = plt.subplots(figsize=(8, 7))
    if bad:
        bx, by = zip(*bad)
        ax.scatter(bx, by, c="tab:red", s=70, edgecolors="black", linewidths=0.6,
                   alpha=0.75, label=f"fail ({len(bad)})", zorder=4)
    if ok:
        ox, oy = zip(*ok)
        ax.scatter(ox, oy, c="tab:green", s=70, edgecolors="black", linewidths=0.6,
                   alpha=0.75, label=f"success ({len(ok)})", zorder=5)
    # zoom to the actual target spread (+margin) so the spawn band fills the plot
    allx = [p[0] for p in ok + bad]; ally = [p[1] for p in ok + bad]
    m = 0.05
    ax.set_xlim(min(allx) - m, max(allx) + m); ax.set_ylim(min(ally) - m, max(ally) + m)
    n = len(ok) + len(bad)
    ax.set_title(f"Rollout target positions by outcome  (n={n}, "
                 f"success {len(ok)}/{n} = {len(ok) / n:.0%})\ngreen = success   red = fail")
    ax.set_xlabel("target x (m)"); ax.set_ylabel("target y (m)")
    ax.legend(loc="upper right"); ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    p = out / "rollout_target_positions.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved {p}")


def run(args):
    out_dir = effective_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    offsets = [float(o) for o in args.offsets.split(",")]
    clutter_densities = [int(d) for d in args.clutter_densities.split(",")]
    rollout = getattr(args, "rollout", False)
    ep_counter = 0   # unique episode id per rollout (-> video/episode{N}.mp4)

    env = make_occluder_task()()
    print(f"seeds from {args.seed_start}, want {args.num_seeds} STABLE  offsets={offsets}  "
          f"clutter_densities={clutter_densities}  rollout={rollout}  camera={CAMERA}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        # Draw seeds until we have args.num_seeds STABLE, usable clean scenes. A seed whose
        # clean build fails or is rejected (e.g. a toppled bottle -> UnStableError from
        # check_stable) is replaced by the next seed, so the run always yields num_seeds good
        # scenes instead of silently doing fewer. Seeds stay contiguous from seed_start apart
        # from the rejected ones, which are skipped entirely (never rolled out).
        produced = 0
        draw = args.seed_start          # incrementing seed to draw from (rejected ones skipped)
        max_draws = args.num_seeds * 10 + 50   # safety cap: don't loop forever if builds keep failing
        while produced < args.num_seeds and (draw - args.seed_start) < max_draws:
            seed = draw
            draw += 1
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
                print(f"[seed {seed}] clean build failed/rejected ({type(e).__name__}: {e}); "
                      f"drawing another seed")
                safe_close()
                continue
            safe_close()
            if full_px <= 0:
                print(f"[seed {seed}] full_target_px=0 on clean scene; drawing another seed.")
                continue
            produced += 1

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
                        # scene geometry for the top-down layout plot (captured before
                        # safe_close() tears the env down)
                        tgt_p = np.array(target.actor.get_pose().p)
                        box_p = (np.array(env.occluder.get_pose().p)
                                 if show and getattr(env, "occluder", None) is not None else None)
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
                           "clutter_density": int(cd),
                           # target spawn position (top-down), for the success/fail scatter
                           "target_x": float(tgt_p[0]), "target_y": float(tgt_p[1])}
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
                    # per-seed top-down layout plot (success/fail from the rollout when on)
                    if args.scene_plots:
                        _plot_scene(seed, off, cd, box_p, tgt_p, np.array(PAD_XY),
                                    rec.get("rollout_success") if rollout else None,
                                    res, out_dir / "scene_plots")
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
            print(f"[{produced}/{args.num_seeds}] seed {seed}: full_px={full_px} done")

        if produced < args.num_seeds:
            print(f"WARNING: only {produced}/{args.num_seeds} stable scenes after "
                  f"{draw - args.seed_start} seeds drawn (hit safety cap)")

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
    ap.add_argument("--no-scene-plots", dest="scene_plots", action="store_false",
                    help="disable the per-seed top-down scene layout plots "
                         "(occluder/target/pad, labelled success/fail) saved to <out-dir>/scene_plots")
    ap.set_defaults(scene_plots=True)
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
        _plot_rollout_target_positions(out_dir)   # target spawn positions, coloured success/fail
    else:
        analyze(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)


if __name__ == "__main__":
    main()
