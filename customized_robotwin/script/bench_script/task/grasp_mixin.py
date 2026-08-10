"""Methods extracted mechanically from analyze_occluder_visibility.py."""

import contextlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import transforms3d as t3d
from curobo.types.state import JointState

from envs.utils import ArmTag
from lib.ik_grid import _build_ik_solver, grasp_orientation
from lib.planning_tuning import *  # noqa: F403
from lib.run_io import CLEARANCE_RESULTS_DIR
from lib.scene_constants import *  # noqa: F403
import seed_from_clearance as sfc


class GraspMixin:
    def _rank_side_grasp_ids(self, actor, arm_tag, limit=GRASP_CANDIDATE_LIMIT):
        """Rank the grasps so we can search across candidates instead of committing to
            exactly one local optimum. Ranking only -- nothing is discarded, so this sets
            the order candidates are tried in, not which ones exist. `horiz` used to hard-
            drop non-horizontal approaches; that was an assumption about this object and
            has to be off for a scene-agnostic baseline, so it now only breaks ties."""
        side = 1.0 if str(arm_tag) == "right" else -1.0
        cx = float(actor.get_pose().p[0])
        conv = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        ranked = []
        for i, cm in actor.iter_contact_points("matrix"):
            if cm is None:
                continue
            g = np.asarray(cm, dtype=float) @ conv
            R = g[:3, :3]
            approach = R @ np.array([1.0, 0.0, 0.0])      # gripper +x approach (world)
            horiz = abs(float(approach[2]))               # 0 = perfectly horizontal
            grasp_x = float(g[0, 3]) + float((R @ np.array([-0.12, 0.0, 0.0]))[0])
            arm_side = side * (grasp_x - cx)              # >0 = gripper on arm's side
            score = arm_side - horiz                      # arm-side + as horizontal as possible
            ranked.append((score, i))
        ranked.sort(reverse=True)
        ids = [i for _, i in ranked[:limit]]
        if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
            print(f"[grasp_id] arm={arm_tag} candidates={ids}")
        return ids


    def _pick_side_grasp_id(self, actor, arm_tag):
        """Top-ranked grasp only -- the single orientation the clearance grid is labelled at.

            DO NOT DELETE AS DEAD CODE. Nothing in task/ calls this; both callers reach it
            duck-typed through the env object, so a by-name scan of this package finds zero
            references and is WRONG: lib/ik_grid.py grasp_orientation() and
            reachability_map.py both do env._pick_side_grasp_id(...). Removing it on that
            basis (a301e2e, 2026-07-29) killed the approach seed silently for every run after
            -- _get_approach_seed catches the AttributeError and falls back to stock, so the
            only symptom was a firing rate of 0. Restored 2026-07-30, verbatim from ea31499."""
        ranked = self._rank_side_grasp_ids(actor, arm_tag, limit=1)
        return ranked[0] if ranked else None


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


    def _grasp_via_cached_trajectories(self, arm_tag, candidate_plan, gripper_pos=0.0):
        """Replay the dry-run-verified pre-grasp and grasp trajectories.

            Matching the legacy execution path, disable the table after reaching
            pre-grasp and before replaying the final cached approach. Fall back to
            live planning only when the candidate has no cached trajectories.
            """
        pre_traj = candidate_plan.get("pre_grasp_trajectory")
        grasp_traj = candidate_plan.get("grasp_trajectory")
        if pre_traj is None or grasp_traj is None:
            return self.grasp_actor_from_table(
                self.target_obj, arm_tag=arm_tag, pre_grasp_dis=0.1,
                contact_point_id=candidate_plan["contact_point_id"])
        self._replay_planned_sequence(arm_tag, pre_traj)
        self.enable_table(enable=False)
        self._replay_planned_sequence(arm_tag, grasp_traj)
        self.move(self.close_gripper(arm_tag, pos=gripper_pos))


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


    def _candidate_specs(self, arm_tag, cp_ids):
        for cp_id in cp_ids:
            grasp_pre_pose = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.1)
            grasp_pose = self._geometric_grasp_pose(self.target_obj, cp_id, pre_dis=0.0)
            if grasp_pre_pose is None or grasp_pose is None:
                continue
            lift_pose = list(grasp_pose)
            lift_pose[2] += GRASP_LIFT_HEIGHT
            occluder_present = self.spawn_occluder and getattr(self, "occluder", None) is not None
            # ROBOPRO Phase 3 fix: in 'direct'/'seed' the around-box waypoint must not exist
            # AT ALL, not merely go unplanned. play_once branches on the waypoint POSE and
            # live-re-plans it whenever the pose is present but its trajectory is None
            # (see its `else: self._plan_and_replay_pose(...)`), so leaving the pose set made
            # those modes still execute the heuristic waypoint, and then snap back to rest
            # when the from-rest pre_grasp trajectory was replayed from a different qpos.
            # Emitting None here kills it on every return path -- verified candidates AND the
            # deepest-progress fallback, which never reaches _plan_candidate's tail.
            waypoints_off = self._approach_mode() != "off"
            placement_off = self._placement_mode() == "direct"
            # z_lift/orient/y_offset only ever shape the waypoint, so with no waypoint they
            # would expand the candidate list into exact duplicates -- each with its own
            # _plan_grasp_side cache key, so each re-running get_grasp_pose's 10-rotation
            # batch-plan for nothing. `gap` and `clearance_z` survive that collapse only
            # because they ALSO feed _box_side_x / _backward_subgoal_poses; with the
            # placement chain off they too are pure duplicate multipliers, so collapse them.
            #
            # Collapsing both also removes the last way the candidate SPACE depends on the
            # scene: `occluder_present` conditioning below is what made a curated cell
            # search 3 gaps and a standard cell search 1, i.e. the planner behaved
            # differently by scene type before any planning happened. With waypoints and
            # the placement chain both off, every list here is single-valued regardless of
            # what is on the table, and a candidate is just a contact point. The
            # occluder_present branches are left intact for APPROACH_MODE=off, which is
            # the acknowledged scene-specific reference mode.
            gaps = ((SIDE_WAYPOINT_GAPS[1],) if (waypoints_off and placement_off) else
                    (SIDE_WAYPOINT_GAPS if occluder_present else (SIDE_WAYPOINT_GAPS[1],)))
            clearance_zs = (PLACE_CLEARANCE_ZS[1],) if placement_off else PLACE_CLEARANCE_ZS
            z_lifts = ((0.0,) if waypoints_off else
                       (SIDE_WAYPOINT_Z_LIFTS if occluder_present else (0.0,)))
            orients = WAYPOINT_ORIENTATIONS if occluder_present else ("grasp_aligned",)
            y_offsets = WAYPOINT_Y_OFFSETS if occluder_present else (0.0,)
            for gap in gaps:
                x_side = self._box_side_x(arm_tag, gap=gap)
                for z_lift in z_lifts:
                    for orient in orients:
                        for y_offset in y_offsets:
                            grasp_waypoint = None if waypoints_off else \
                                self._around_box_waypoint(
                                    arm_tag, grasp_pre_pose, gap=gap, z_lift=z_lift,
                                    orient=orient, y_offset=y_offset)
                            for clearance_z in clearance_zs:
                                placement_subgoals = self._backward_subgoal_poses(
                                    arm_tag,
                                    x_side=x_side,
                                    clearance_z=clearance_z,
                                    lift_pose=lift_pose,
                                )
                                yield {
                                    "contact_point_id": cp_id,
                                    "grasp_waypoint": grasp_waypoint,
                                    "lift_pose": lift_pose,
                                    "placement_subgoals": placement_subgoals,
                                    "gap": gap,
                                    "y_offset": y_offset,
                                    "z_lift": z_lift,
                                    "orient": orient,
                                    "clearance_z": clearance_z,
                                }


    def _plan_grasp_side(self, arm_tag, candidate, cache):
        """Waypoint -> pre_grasp -> grasp -> lift, cached per (cp_id, gap, z_lift,
            orient, y_offset) -- this doesn't depend on clearance_z, but
            _candidate_specs yields one candidate per clearance_z sharing the same
            waypoint/grasp, so without caching this (expensive: get_grasp_pose does a
            real 10-rotation CuRobo batch-plan) work would be redone 3x for nothing.
            Returns (ok, failed_stage, fail_reason, qpos, lift_pose, trajectories).
            trajectories is {"waypoint"/"pre_grasp"/"grasp": <raw plan_path result>},
            captured so real execution can REPLAY these EXACT verified trajectories
            instead of independently re-planning to the same poses: confirmed
            empirically that an independent re-plan to the same Cartesian pose can
            converge to a genuinely different (though equally valid) joint
            configuration -- CuRobo's trajopt has no uniqueness guarantee -- with
            qpos drift up to 1.47 rad between the two, which flips downstream grasp
            reachability in ~50% of cases with large drift. This already fixed the
            waypoint; pre_grasp/grasp trajectories are captured the same way here
            because grasp_actor_from_table's own independent re-plan to the SAME
            pose choose_best_pose already verified is exactly the same failure mode,
            one step later in the chain.
            pre_grasp/grasp trajectories are planned CHAINED (pre_grasp's landed
            qpos feeds grasp's start), matching how they're physically executed one
            after another -- unlike the get_grasp_pose feasibility checks below,
            which mirror choose_grasp_pose's own unchained pose determination."""
        key = (candidate["contact_point_id"], candidate["gap"], candidate["z_lift"],
              candidate["orient"], candidate["y_offset"])
        if key in cache:
            return cache[key]

        cp_id = candidate["contact_point_id"]
        start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
        trajectories = {}

        # ROBOPRO Phase 3: grasp-approach MODE (env APPROACH_MODE; SEED_FROM_CLEARANCE=1 => "seed"):
        #   off    -> stock around-box waypoint (default; the scene-specific one-occluder heuristic).
        #   direct -> NO waypoint: plan pre_grasp straight from rest, NO seed  (generalization baseline).
        #   seed   -> NO waypoint: plan pre_grasp straight from rest, WITH the clearance-route seed.
        # direct vs seed differ ONLY by the seed (so any success delta is attributable to the seed), and
        # NEITHER falls back to the around-box waypoint -- a miss fails the candidate, so the baseline/
        # method numbers are not contaminated by the heuristic. The waypoint is suppressed at the
        # SOURCE (_candidate_specs emits grasp_waypoint=None outside 'off'). Skipping its PLAN here
        # is not enough on its own: play_once branches on the waypoint POSE and live-re-plans it
        # whenever the pose is present but its trajectory is None, which is how the heuristic kept
        # executing in these modes -- followed by a snap back to rest when the from-rest pre_grasp
        # trajectory was replayed from the waypoint's qpos.
        mode = self._approach_mode()
        pre_grasp_pose = grasp_pose = grasp_side_start_qpos = None
        qpos = np.array(start_qpos, dtype=np.float64, copy=True)
        if mode != "off":
            seed = self._get_approach_seed(arm_tag) if mode == "seed" else None
            # same grasp-pose selection real execution uses (rotation search + batch-plan check),
            # both determined from the REST qpos, unchained -- mirroring the stock path below.
            pre_grasp_pose = self.get_grasp_pose(self.target_obj, arm_tag, contact_point_id=cp_id,
                                                 pre_dis=0.1, last_qpos=start_qpos)
            if pre_grasp_pose is None:
                result = (False, "pre_grasp", "no_reachable_rotation", None, None, None)
                cache[key] = result
                return result
            grasp_pose = self.get_grasp_pose(self.target_obj, arm_tag, contact_point_id=cp_id,
                                             pre_dis=0.0, last_qpos=start_qpos)
            if grasp_pose is None:
                result = (False, "grasp", "no_reachable_rotation", None, None, None)
                cache[key] = result
                return result
            grasp_side_start_qpos = start_qpos
            ok, fail_reason, qpos, pre_grasp_trajectory = self._plan_pose_with_local_waypoint_retry(
                arm_tag, pre_grasp_pose, qpos, "pre_grasp", start_pose=None, seed_traj=seed)
            if not ok:
                result = (False, "pre_grasp", fail_reason, None, None, None)
                cache[key] = result
                return result
            trajectories["pre_grasp"] = pre_grasp_trajectory
        else:
            # ---- stock waypoint-based approach (default; unchanged) ----
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            if candidate["grasp_waypoint"] is not None:
                ok, fail_reason, qpos, waypoint_trajectory = self._plan_pose_with_local_waypoint_retry(
                    arm_tag, candidate["grasp_waypoint"], qpos, "waypoint")
                if not ok:
                    result = (False, "waypoint", fail_reason, None, None, None)
                    cache[key] = result
                    return result
                trajectories["waypoint"] = waypoint_trajectory
            grasp_side_start_qpos = qpos  # pre_grasp/grasp poses are both determined from here (unchained)

            # Validate the SAME grasp-pose selection real execution uses
            # (get_grasp_pose -> choose_best_pose's rotation search + batch-plan
            # check, now that choose_best_pose's shortest-plan logic is fixed)
            # instead of the geometric approximation -- a candidate that "passes"
            # here can no longer diverge from what grasp_actor_from_table() actually
            # executes. choose_grasp_pose's own check_pose tests pre_grasp/grasp
            # independently from the SAME starting qpos (it doesn't chain through
            # pre_grasp's landed state) -- mirrored here for consistency.
            pre_grasp_pose = self.get_grasp_pose(self.target_obj, arm_tag, contact_point_id=cp_id,
                                                 pre_dis=0.1, last_qpos=grasp_side_start_qpos)
            if pre_grasp_pose is None:
                result = (False, "pre_grasp", "no_reachable_rotation", None, None, None)
                cache[key] = result
                return result
            grasp_pose = self.get_grasp_pose(self.target_obj, arm_tag, contact_point_id=cp_id,
                                             pre_dis=0.0, last_qpos=grasp_side_start_qpos)
            if grasp_pose is None:
                result = (False, "grasp", "no_reachable_rotation", None, None, None)
                cache[key] = result
                return result

            # Now plan+capture the ACTUAL trajectories to replay, CHAINED (pre_grasp
            # then grasp from wherever pre_grasp lands), matching physical execution.
            ok, fail_reason, qpos, pre_grasp_trajectory = self._plan_pose_with_local_waypoint_retry(
                arm_tag, pre_grasp_pose, qpos, "pre_grasp", start_pose=candidate.get("grasp_waypoint"))
            if not ok:
                result = (False, "pre_grasp", fail_reason, None, None, None)
                cache[key] = result
                return result
            trajectories["pre_grasp"] = pre_grasp_trajectory

        # ---- common tail: grasp + lift, from wherever pre_grasp landed (seeded or stock) ----
        ok, fail_reason, qpos, grasp_trajectory = self._plan_pose_with_local_waypoint_retry(
            arm_tag, grasp_pose, qpos, "grasp", start_pose=pre_grasp_pose)
        if not ok:
            result = (False, "grasp", fail_reason, None, None, None)
            cache[key] = result
            return result
        trajectories["grasp"] = grasp_trajectory

        lift_pose = list(grasp_pose)
        lift_pose[2] += GRASP_LIFT_HEIGHT
        ok, failed_stage, fail_reason, qpos = self._plan_pose_sequence(
            arm_tag, [lift_pose], qpos, stage_labels=["lift"])
        if not ok:
            result = (ok, failed_stage, fail_reason, None, None, None)
            cache[key] = result
            return result

        result = (True, None, None, qpos, lift_pose, trajectories)
        cache[key] = result
        return result


    def _plan_candidate(self, arm_tag, candidate, grasp_side_cache):
        ok, failed_stage, fail_reason, qpos, lift_pose, trajectories = self._plan_grasp_side(
            arm_tag, candidate, grasp_side_cache)
        if not ok:
            return ok, failed_stage, fail_reason
        # Carried to play_once so real execution can REPLAY these exact verified
        # trajectories for the waypoint/pre_grasp/grasp moves instead of
        # independently re-planning to the same poses (see _plan_grasp_side's
        # docstring for why that matters).
        candidate["grasp_waypoint_trajectory"] = trajectories.get("waypoint")
        candidate["pre_grasp_trajectory"] = trajectories.get("pre_grasp")
        candidate["grasp_trajectory"] = trajectories.get("grasp")

        # Placement subgoal orientation is derived from the grasp orientation --
        # rebuild from the REAL lift_pose above (candidate["placement_subgoals"]
        # was built in _candidate_specs from the geometric approximation) and
        # write the corrected version back onto the candidate so real execution
        # (play_once) uses it too, not just this validation pass.
        placement_subgoals = self._backward_subgoal_poses(
            arm_tag, x_side=self._box_side_x(arm_tag, gap=candidate["gap"]),
            clearance_z=candidate["clearance_z"], lift_pose=lift_pose)
        candidate["placement_subgoals"] = placement_subgoals
        candidate["lift_pose"] = lift_pose
        if not placement_subgoals:
            # PLACEMENT_MODE=direct: there is no backward chain to dry-run. Return
            # before the attach/detach below rather than letting it certify an empty
            # pose list as trivially plannable -- the real placement work then happens
            # entirely inside place_actor, which does its own planning.
            return True, None, None

        # Real execution calls attach_object (holding the target) before planning
        # the placement subgoals; the check above doesn't, so it can wrongly
        # certify a placement pose that's actually blocked by the held object's
        # own collision volume. Confirmed empirically: "beside_box" flips from
        # IK-feasible to infeasible once attached with this same chained qpos.
        # Approximate the held object's pose as lift_pose (the gripper is
        # essentially at the object once it's holding it).
        planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
        object_dict = {
            "name": self.target_obj.get_name(),
            "pose": list(lift_pose),
            "file_path": f"{os.environ['BENCH_ROOT']}/assets/objects/{self.target_model}/collision/base{self.target_id}.glb",
            "scale": self.target_obj.scale,
        }
        placement_poses = [pose for _, pose in placement_subgoals]
        placement_labels = [name for name, _ in placement_subgoals]
        try:
            planner.attach_object(object_dict, qpos, arms_tag=str(arm_tag))
            ok2, failed_stage2, fail_reason2, _ = self._plan_pose_sequence(
                arm_tag, placement_poses, qpos, stage_labels=placement_labels)
        finally:
            planner.detach_object()
        return ok2, failed_stage2, fail_reason2


    def _select_pick_place_candidate(self, arm_tag, exclude_cp_ids=None):
        # Selected-candidate metadata for the structured failure recorder (see
        # play_once): whether the executed candidate was fully dry-run-verified
        # or a fallback, and if a fallback, how far it got in the dry run and
        # the full failure breakdown across everything tried. Lets records.jsonl
        # distinguish "the dry run found a working plan and it still failed for
        # real" from "the dry run never found anything workable to begin with"
        # without re-deriving it from ROBOTWIN_LOG_MOVE logs.
        #
        # exclude_cp_ids: contact points already tried and found to fail a REAL
        # grasp_verify check (see play_once) -- skipped so a retry after a
        # missed grasp picks a genuinely different contact point instead of
        # re-selecting the same one that just failed.
        cp_ids = self._rank_side_grasp_ids(self.target_obj, arm_tag)
        if exclude_cp_ids:
            cp_ids = [cp for cp in cp_ids if cp not in exclude_cp_ids]
        if not cp_ids:
            self.rollout_candidate_info = {"verified": False, "reason": "no_ranked_contact_points"}
            return {
                "contact_point_id": None,
                "grasp_waypoint": None,
                "placement_subgoals": self._backward_subgoal_poses(
                    arm_tag,
                    x_side=self._box_side_x(arm_tag),
                    clearance_z=PLACE_CLEARANCE_ZS[1],
                    lift_pose=list(self.get_arm_pose(arm_tag)),
                ),
            }

        log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
        fallback = None
        fallback_depth = -1
        fallback_stage = None
        tried = 0
        failure_breakdown = {}
        grasp_side_cache = {}
        for candidate in self._candidate_specs(arm_tag, cp_ids):
            tried += 1
            ok, failed_stage, fail_reason = self._plan_candidate(arm_tag, candidate, grasp_side_cache)
            if ok:
                if log_move:
                    print(
                        "[candidate] selected "
                        f"cp_id={candidate['contact_point_id']} gap={candidate['gap']:.2f} "
                        f"z_lift={candidate['z_lift']:.2f} orient={candidate['orient']} "
                        f"y_offset={candidate['y_offset']:.2f} "
                        f"clearance_z={candidate['clearance_z']:.2f} (tried {tried})"
                    )
                self.rollout_candidate_info = {
                    "verified": True, "contact_point_id": candidate["contact_point_id"], "tried": tried,
                }
                return candidate
            # Fallback = the candidate that progressed FURTHEST through the chain
            # before failing, not just the first one generated -- a later cp_id can
            # be far more reachable than the top-ranked one even though it's tried
            # later in the loop.
            depth = STAGE_ORDER.index(failed_stage) if failed_stage in STAGE_ORDER else -1
            if depth > fallback_depth:
                fallback, fallback_depth, fallback_stage = candidate, depth, failed_stage
            reasons = failure_breakdown.setdefault(failed_stage, {})
            reasons[fail_reason] = reasons.get(fail_reason, 0) + 1

        if fallback is not None and log_move:
            print(
                f"[candidate] no fully planned candidate found ({tried} tried); "
                f"using deepest-progress fallback cp_id={fallback['contact_point_id']} "
                f"(died at '{fallback_stage}', depth {fallback_depth}); "
                f"failure breakdown (stage -> {{reason: count}}): {failure_breakdown}"
            )
        self.rollout_candidate_info = {
            "verified": False,
            "contact_point_id": fallback["contact_point_id"] if fallback else None,
            "tried": tried,
            "dry_run_fallback_stage": fallback_stage,
            "dry_run_failure_breakdown": failure_breakdown,
        }
        return fallback
