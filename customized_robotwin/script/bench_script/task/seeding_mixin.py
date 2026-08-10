"""Methods extracted mechanically from analyze_occluder_visibility.py."""

import os

import torch

from lib.ik_grid import _build_ik_solver, grasp_orientation
from lib.metric_config import SeedMetricConfig
from lib.planning_tuning import *  # noqa: F403
from lib.run_io import CLEARANCE_RESULTS_DIR
from lib.scene_constants import *  # noqa: F403
import seed_from_clearance as sfc


class SeedingMixin:
    def _approach_mode(self):
        """Grasp-approach experiment mode (ROBOPRO Phase 3): 'off' | 'direct' | 'seed'.
              off    = stock around-box waypoint (default; scene-specific heuristic).
              direct = waypoints OFF, plan pre_grasp straight from rest, NO seed (generalization baseline).
              seed   = waypoints OFF, direct pre_grasp WITH the clearance-route seed (the method).
            Set via env APPROACH_MODE; legacy SEED_FROM_CLEARANCE=1 is honored as 'seed'."""
        m = os.environ.get("APPROACH_MODE", "").strip().lower()
        if m in ("off", "direct", "seed"):
            return m
        if os.environ.get("SEED_FROM_CLEARANCE", "") == "1":
            return "seed"
        return "off"


    def _placement_mode(self):
        """Carry/placement experiment mode: 'scripted' | 'direct'. Env PLACEMENT_MODE.

              scripted = the hand-authored backward chain (default; unchanged behaviour).
              direct   = NO placement subgoals at all: after the lift, go straight into
                         place_actor via the one generic blended intermediate.

            APPROACH_MODE only ever gated the FORWARD (rest -> pre_grasp) leg. The backward
            leg is scene-specific in the same way the around-box waypoint is: its x comes
            from _box_side_x, which reads occluder 0's world pose plus OCC_HALF_FOOTPRINT (a
            constant hand-entered for the olive-oil asset), and its heights come from the
            fixed PLACE_CLEARANCE_ZS ladder rather than anything measured off the scene. With
            more than one occluder the corridor it picks is only cleared against bottle 0; on
            an occluder-free scene _box_side_x silently falls back to the TARGET's x, so the
            chain is routed with reference to nothing that is actually in the way. None of the
            clutter is consulted anywhere. So for a generality claim the whole chain has to be
            switchable off, leaving only mechanisms that read no scene-specific state:
            _verified_intermediate (blend two poses, accept on IK), the local-offset bridge
            pool / shrinking-waypoint retry, and curobo's own collision world.

            Deliberately NOT folded into APPROACH_MODE: the two legs are independent knobs and
            the A/B pairs cells that differ by exactly one of them."""
        m = os.environ.get("PLACEMENT_MODE", "").strip().lower()
        return m if m in ("scripted", "direct") else "scripted"


    def _get_approach_seed(self, arm_tag):
        """ROBOPRO Phase 3: the clearance-metric route as a curobo trajopt seed for the grasp approach,
            computed ONCE per (scene, arm) and cached. Returns a CPU seed tensor (1,1,H,dof) or None
            (=> the direct plan runs unseeded). Only builds in 'seed' mode.

            The route depends on scene+arm+grasp orientation, not on the candidate's gap/z_lift/... so one
            seed is reused across candidates; 2b welds each plan's exact start config on. Any failure
            (wrong mode, no route, exception) returns None and the direct plan proceeds without a seed."""
        if self._approach_mode() != "seed":
            return None
        if not hasattr(self, "_approach_seed_cache"):
            self._approach_seed_cache = {}
        tag = str(arm_tag)
        # cache key = (arm, SCENE signature). The scene signature (target + occluder world poses)
        # makes the cache per-SCENE -- without it the env-level cache reused the FIRST scene's seed for
        # every later episode (stale-seed correctness bug; also why only one visual set was ever saved).
        import numpy as _np
        try:
            _sig = [tuple(_np.round(_np.asarray(self.target_obj.get_pose().p, float), 3))]
            for _o in (getattr(self, "occluders", None) or []):
                _sig.append(tuple(_np.round(_np.asarray(_o.get_pose().p, float), 3)))
            scene_sig = tuple(_sig)
        except Exception:
            scene_sig = None
        key = (tag, scene_sig)
        if key in self._approach_seed_cache:
            self._note_seed_stat(tag, self._approach_seed_cache[key] is not None,
                                 "cached", 0.0, None, None)
            return self._approach_seed_cache[key]

        seed = None
        ik = None
        try:
            import numpy as _np
            planner = self.robot.left_planner if tag == "left" else self.robot.right_planner
            grasp_q, grasp_pose = grasp_orientation(self, tag, topdown=False)
            if grasp_pose is None:
                print(f"[seed] arm={tag}: no grasp pose -> no seed")
                self._note_seed_stat(tag, False, "no_grasp_pose", 0.0, None, None)
                self._approach_seed_cache[key] = None
                return None
            ee = self.robot.get_left_ee_pose() if tag == "left" else self.robot.get_right_ee_pose()
            start_xyz = _np.asarray(ee[:3], dtype=float)
            goal_xyz = _np.asarray(grasp_pose[:3], dtype=float)
            # real current arm config in curobo active-joint order (the exact tstep-0 weld)
            full_q = (self.robot.left_entity.get_qpos() if tag == "left"
                      else self.robot.right_entity.get_qpos())
            idx = [planner.all_joints.index(n) for n in planner.active_joints_name]
            start_q = _np.asarray([full_q[i] for i in idx], dtype=float)
            ik = _build_ik_solver(planner)
            H = int(planner.motion_gen.trajopt_solver.action_horizon)
            # Obstacle set the clearance field measures against. Default "all" = every mesh
            # in env.collision_list except the target and pad, so table CLUTTER counts and
            # the seed has something to route around even with no occluder. SEED_OBSTACLES=
            # occluders restores the curated-ring-only field (eps* not comparable).
            # res defaults to 0.02 here, NOT clearance_metric_3d's own 0.01: the grid is
            # ~4x cheaper (x,y both halve) and the seed is only an INITIALIZATION that
            # trajopt refines, so it does not need the tool's full fidelity. The cost is
            # a coarser eps*, whose half-voxel optimism grows to res/2 = 10 mm.
            cfg = SeedMetricConfig.from_env()
            seed, res = sfc.build_seed(self, planner, tag, ik, grasp_q, start_q, None,
                                       start_xyz, goal_xyz, cfg=cfg, action_horizon=H)
            if seed is None:
                print(f"[seed] arm={tag}: build_seed produced NO route ({res.reason}) -> stock fallback "
                      f"({res.seconds:.1f}s)")
                self._note_seed_stat(tag, False, res.reason, res.seconds, None, None)
            else:
                print(f"[seed] arm={tag}: route {len(res.route_qs)} voxels, eps={res.eps_gated:.3f}m -> "
                      f"seed {tuple(seed.shape)} ({res.seconds:.1f}s)")
                self._note_seed_stat(tag, True, None, res.seconds,
                                     len(res.route_qs), res.eps_gated)
                # save the metric's route visuals for THIS rollout INSIDE the rollout's own output
                # folder (self._rollout_out_dir, set per episode in run()), in a dedicated
                # seed_route_visuals/episode<N>_<arm>/ subfolder so it doesn't mix with the rollout's
                # other files. Falls back to a standalone dir if no rollout out_dir is set. Own try --
                # a viz failure must NOT null the seed.
                try:
                    if os.environ.get("SEED_VISUALS", "1") != "0":
                        from datetime import datetime as _now
                        from pathlib import Path as _P
                        ep = getattr(self, "_rollout_ep", None)
                        out_root = getattr(self, "_rollout_out_dir", None)
                        if out_root is not None:
                            epname = (f"episode{ep}_{tag}" if ep is not None
                                      else f"{_now.now().strftime('%H%M%S')}_{tag}")
                            sub = _P(out_root) / "seed_route_visuals" / epname
                        else:
                            base = os.environ.get("SEED_VISUALS_DIR") or str(
                                _P(CLEARANCE_RESULTS_DIR).parent / "seed_visuals")
                            sub = _P(base) / f"{_now.now().strftime('%Y%m%d-%H%M%S')}_{tag}"
                        sfc.save_route_visuals(
                            res, sub, seed_label=(str(ep) if ep is not None
                                                  else _now.now().strftime('%H%M%S')), arm=tag)
                except Exception as _ve:
                    print(f"[seed-viz] skipped ({_ve})")
        except Exception as e:  # never let seeding break the expert
            print(f"[seed] arm={tag}: build_seed FAILED ({e}); stock fallback (no seed)")
            self._note_seed_stat(tag, False, f"exception:{type(e).__name__}", None, None, None)
            seed = None
        finally:
            # RELEASE THE IK SOLVER. _build_ik_solver constructs a fresh curobo IKSolver
            # with its own GPU rollout/seed buffers, and this runs once per (arm, SCENE) --
            # i.e. every episode, since the cache key includes the scene. Every other
            # call site in this repo pairs it with `del ik; empty_cache()`
            # (reachability_map.py:192, clearance_metric_3d.py:440); this one did not, so
            # solvers accumulated across a run until CUDA ran out. That is what produced
            # the 43 exception:OutOfMemoryError no-route reasons, and why the GPU stayed
            # saturated even once the process had stopped making progress -- torch's
            # caching allocator holds reserved memory away from the driver (and so from
            # every other process) until empty_cache() gives it back.
            try:
                del ik
                torch.cuda.empty_cache()
            except Exception:
                pass
            if os.environ.get("SEED_MEM_LOG", "1") != "0":
                # One line per build so unbounded growth is visible in the cell log
                # instead of only surfacing as an OOM tens of episodes later.
                try:
                    _al = torch.cuda.memory_allocated() / 2**30
                    _rs = torch.cuda.memory_reserved() / 2**30
                    print(f"[seed-mem] arm={tag}: cuda allocated={_al:.2f}GiB "
                          f"reserved={_rs:.2f}GiB (after release)")
                except Exception:
                    pass
        self._approach_seed_cache[key] = seed
        return seed


    def _carry_seed_on(self):
        """Is the CARRY leg (lift -> pad transit) seeded? Env CARRY_SEED overrides ('0'/'1').

            Defaults to the approach leg's mode, so the A/B stays two cells that differ by exactly
            one thing -- 'does the clearance metric's route help' -- rather than four. The override
            exists for diagnosis (seed one leg, not the other) and must NOT be set per-cell in a
            real run, or the cells stop differing by a single variable.

            Only meaningful under PLACEMENT_MODE=direct: with the hand-authored chain on, the carry
            is a sequence of short scripted hops that a route seed has nothing to contribute to."""
        v = os.environ.get("CARRY_SEED", "").strip()
        if v in ("0", "1"):
            return v == "1"
        return self._approach_mode() == "seed"


    def _get_carry_seed(self, arm_tag, goal_xyz):
        """Phase C: the clearance-metric route as a curobo trajopt seed for the CARRY transit
            (the post-lift move toward the pad), built with the target object ATTACHED. Returns a
            CPU seed tensor (1,1,H,dof) or None (=> the plan runs unseeded, exactly as 'direct').

            Differences from _get_approach_seed, all forced by the object being in the gripper:

            1. HELD-OBJECT COLLISION. The grid's IK solver gets the SAME attached-object spheres
               curobo built for the motion_gen during the expert's own attach_object() -- copied
               across verbatim by carry_object_spheres (see that module for why nothing is
               approximated here). Without this the sweep would label voxels FREE that the bottle
               does not fit through: already observed in this file, where beside_box "flips from
               IK-feasible to infeasible once attached with this same chained qpos".

            2. ORIENTATION. The grid is a single-orientation slice, and the carry leg is labelled
               GRASP-ALIGNED: the orientation the arm is already holding after the lift. That costs
               no reorientation move before the transit, and the rotation into place_actor's
               handoff pose stays where it already happens -- at the end, in place_actor.

            3. NO CROSS-CANDIDATE REUSE, and none needed. The approach seed is cached per (arm,
               scene) and shared across the candidate search; the carry runs ONCE per episode,
               after a grasp has already succeeded, so the exact attached geometry is simply
               available by then. The cache below only guards the grasp-retry loop re-entering
               with the same scene and goal.

            Any failure returns None; the transit then plans unseeded, which is the 'direct'
            behaviour, so a miss costs firing rate and never correctness."""
        if not self._carry_seed_on():
            return None
        if not hasattr(self, "_carry_seed_cache"):
            self._carry_seed_cache = {}
        tag = str(arm_tag)
        import numpy as _np
        try:
            _sig = [tuple(_np.round(_np.asarray(self.target_obj.get_pose().p, float), 3))]
            for _o in (getattr(self, "occluders", None) or []):
                _sig.append(tuple(_np.round(_np.asarray(_o.get_pose().p, float), 3)))
            scene_sig = tuple(_sig)
        except Exception:
            scene_sig = None
        key = (tag, scene_sig, tuple(_np.round(_np.asarray(goal_xyz, float), 3)))
        if key in self._carry_seed_cache:
            self._note_seed_stat(tag, self._carry_seed_cache[key] is not None,
                                 "cached", 0.0, None, None, leg="carry")
            return self._carry_seed_cache[key]

        seed = None
        ik = None
        try:
            from lib import carry_object_spheres as cos
            planner = self.robot.left_planner if tag == "left" else self.robot.right_planner

            # The held object, exactly as the motion_gen models it. attach_object() ran in
            # play_once before the placement section, so these slots are populated; if they
            # are not, an unattached grid would silently promise routes the bottle cannot
            # take, so refuse to build rather than build a wrong one.
            spheres = cos.attached_spheres_from_planner(planner)
            if spheres is None:
                print(f"[carry-seed] arm={tag}: nothing attached -> no seed")
                self._note_seed_stat(tag, False, "no_attached_object", 0.0, None, None, leg="carry")
                self._carry_seed_cache[key] = None
                return None

            # grasp-aligned slice: the orientation the arm is holding right now (post-lift)
            live = list(self.get_arm_pose(arm_tag))
            carry_q = _np.asarray(live[3:], dtype=float)
            start_xyz = _np.asarray(live[:3], dtype=float)
            goal_xyz = _np.asarray(goal_xyz, dtype=float)

            full_q = (self.robot.left_entity.get_qpos() if tag == "left"
                      else self.robot.right_entity.get_qpos())
            idx = [planner.all_joints.index(n) for n in planner.active_joints_name]
            start_q = _np.asarray([full_q[i] for i in idx], dtype=float)

            ik = _build_ik_solver(planner)
            cos.apply_attached_spheres(ik, spheres)
            H = int(planner.motion_gen.trajopt_solver.action_horizon)
            # Same knobs as the approach seed so the two legs are measured on one grid
            # geometry; only the labelling differs (attached, carry orientation).
            cfg = SeedMetricConfig.from_env()
            seed, res = sfc.build_seed(self, planner, tag, ik, carry_q, start_q, None,
                                       start_xyz, goal_xyz, cfg=cfg, action_horizon=H)
            if seed is None:
                print(f"[carry-seed] arm={tag}: NO route ({res.reason}) -> unseeded transit "
                      f"({res.seconds:.1f}s)")
                self._note_seed_stat(tag, False, res.reason, res.seconds, None, None, leg="carry")
            else:
                print(f"[carry-seed] arm={tag}: route {len(res.route_qs)} voxels, "
                      f"eps={res.eps_gated:.3f}m (object extent "
                      f"{cos.carry_sphere_extent(spheres):.3f}m) -> seed {tuple(seed.shape)} "
                      f"({res.seconds:.1f}s)")
                self._note_seed_stat(tag, True, None, res.seconds, len(res.route_qs),
                                     res.eps_gated, leg="carry")
                try:
                    if os.environ.get("SEED_VISUALS", "1") != "0":
                        from pathlib import Path as _P
                        ep = getattr(self, "_rollout_ep", None)
                        out_root = getattr(self, "_rollout_out_dir", None)
                        if out_root is not None:
                            sub = (_P(out_root) / "seed_route_visuals"
                                   / f"episode{ep}_{tag}_carry")
                            sfc.save_route_visuals(res, sub,
                                                   seed_label=f"{ep}-carry", arm=tag,
                                                   start_label="post-lift gripper",
                                                   goal_label="pad transit")
                except Exception as _ve:
                    print(f"[carry-seed-viz] skipped ({_ve})")
        except Exception as e:      # never let seeding break the expert
            print(f"[carry-seed] arm={tag}: FAILED ({e}); unseeded transit")
            self._note_seed_stat(tag, False, f"exception:{type(e).__name__}", None, None, None,
                                 leg="carry")
            seed = None
        finally:
            # Same release discipline as _get_approach_seed -- a fresh IKSolver per episode
            # that is never freed is what saturated the GPU on the first full A/B run. Detach
            # first: the solver is about to go, but the detach also documents the pairing and
            # keeps this correct if the solver is ever made reusable.
            try:
                if ik is not None:
                    from lib import carry_object_spheres as _cos
                    _cos.detach_attached_spheres(ik)
            except Exception:
                pass
            try:
                del ik
                torch.cuda.empty_cache()
            except Exception:
                pass
            if os.environ.get("SEED_MEM_LOG", "1") != "0":
                try:
                    _al = torch.cuda.memory_allocated() / 2**30
                    _rs = torch.cuda.memory_reserved() / 2**30
                    print(f"[seed-mem] arm={tag} carry: cuda allocated={_al:.2f}GiB "
                          f"reserved={_rs:.2f}GiB (after release)")
                except Exception:
                    pass
        self._carry_seed_cache[key] = seed
        return seed


    def _note_seed_stat(self, arm_tag, built, reason, seconds, voxels, eps, leg="approach"):
        """ROBOPRO Phase 4: record one _get_approach_seed outcome so the A/B can
            report the seed's FIRING RATE (the 2a smoke test showed IK is
            nondeterministic, so an identical scene can build a route on one run and not
            the next -- a miss is absorbed silently by the unseeded plan, which would
            otherwise make 'seed' mode look like it did nothing). reason='cached' marks a
            cache hit rather than a fresh build, so build cost isn't double-counted.
            Read by run() into records.jsonl. Never raises.

            leg: 'approach' (rest -> pre_grasp) or 'carry' (post-lift -> pad). The two legs have
            very different firing rates and build costs -- the carry grid is labelled with the
            object attached, which shrinks its FREE set -- so they must be reported separately or
            a healthy approach rate would mask a dead carry one. Defaults to 'approach' so records
            written before the carry leg existed read back correctly."""
        try:
            if not hasattr(self, "rollout_seed_stats"):
                self.rollout_seed_stats = []
            self.rollout_seed_stats.append({
                "arm": str(arm_tag), "leg": str(leg), "built": bool(built), "reason": reason,
                "seconds": (float(seconds) if seconds is not None else None),
                "route_voxels": (int(voxels) if voxels is not None else None),
                "eps_gated": (float(eps) if eps is not None else None),
            })
        except Exception:
            pass
