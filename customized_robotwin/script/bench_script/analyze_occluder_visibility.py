"""
Phase 2 part 1 (#35): deterministic occluder-ring spawn + visibility distribution.

Places the target (001_bottle id 9) in the upper third of the reachable region (in
front of the back furniture) and spawns a RING of `--num-occluders` tall, skinny olive-oil
occluders (029_olive-oil id 3, scale 1.0) equally spaced (2*pi/n apart) on a circle of
radius `--offsets` around the target -- n=1 is a single box directly in front (-y, the
robot/camera side). No clutter, no binary search. Then measures the t=0 countertop
`visible_fraction` distribution across seeds, reusing the Phase 1 harness (measurement +
plotting).

visible_fraction = visible_target_px(with occluder) / full_target_px(no occluder),
same seed (target pose is fixed before the occluder is added).

OUTPUT LAYOUT: every run writes to its OWN timestamped folder under a single
dedicated results folder, grouped by run type, so re-running never clobbers a
previous run:
    <out-dir>/<type>/<YYYYmmdd-HHMMSS>/
where <type> is 'rollout' vs 'no_rollout' by default (override with --run-type,
e.g. a planner name like 'hamid' / 'baseline'). --plot-only re-reads an existing
timestamped folder, so pass its full path as --out-dir.

USAGE (from the benchmark folder):
    cd benchmark
    source set_env.sh
    export ROBOTWIN_BENCH_TASK=bench
    python script/bench_script/analyze_occluder_visibility.py \
        --seed-start 0 --num-seeds 50 --offsets 0.10 --bins 20
    # -> ../scripts/validation/results/occluder_visibility/no_rollout/<timestamp>/

    # narrow-region sweep (a few offsets) to see how distance affects occlusion:
    python script/bench_script/analyze_occluder_visibility.py \
        --num-seeds 40 --offsets 0.07,0.10,0.13

    # re-plot only (point at an existing timestamped run folder):
    python script/bench_script/analyze_occluder_visibility.py --plot-only \
        --out-dir ../scripts/validation/results/occluder_visibility/no_rollout/<timestamp>
"""
import os
import sys
import json
import time
import argparse
import contextlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import transforms3d as t3d
from curobo.types.state import JointState

from setup_paths import setup_paths
setup_paths()

from lib.occluder_ring import (
    draw_ring_config, occluder_ring_xy, parse_count_choices, parse_offset_specs,
)
from lib.planning_tuning import *  # noqa: F403
from lib.run_io import CLEARANCE_RESULTS_DIR, _Tee, _prune_empty_topdirs, effective_out_dir
from lib.scene_build import DR_CLEAN, build_cfg, dr_measure, get_env_class
from lib.scene_constants import *
from lib.visibility import (
    CAMERA, _resolve_target, analyze, analyze_rollout, run_rollout, save_overlay,
)
from lib.ik_grid import _build_ik_solver, grasp_orientation
import seed_from_clearance as sfc
# these analysis scripts are bench-only; default the task mode so build_cfg looks
# under bench_task_config/ (explicit ROBOTWIN_BENCH_TASK still wins)
os.environ.setdefault("ROBOTWIN_BENCH_TASK", "bench")

robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])
os.chdir(robotwin_root)

from envs.utils import rand_pose, create_actor, create_box, ArmTag
from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC

# target object: 001_bottle id 9 (8-way grasp ring -> side-reaches around the occluder).
# Swap TARGET_MODEL/TARGET_ID back to 047_mouse id 0 for the stock top-down mouse.
# target spawn: back half of the table (back furniture removed). Capped at y=0.20 so it
# stays 0.15 m off the back edge (table y in [-0.35, 0.35]); x kept to [-0.15, 0.15] so it
# stays 0.45 m off each side edge (table x in [-0.6, 0.6]) -> leaves room for the around-
# the-box side waypoint and keeps target + waypoint inside the arm's reach envelope.
# Full tabletop extent (see the target-spawn note above). A ring occluder whose CENTRE
# would fall outside this is simply not spawned (fewer than n bottles, same formation)
# rather than rejecting the whole seed -- see occluder_ring_xy(... xlim/ylim ...).
# Occluder object: 029_olive-oil id 3 -- a tall, skinny, round bottle (raw extents
# H 0.305 m, footprint 0.080 x 0.080 m, aspect 3.8). Switched from the old 038_milk-box
# id 2 (H 0.254 m, 0.110 x 0.122 m rectangular footprint, aspect 2.1), which was too fat.
# Both share the same mesh convention (tall axis = local Y, base at origin), so OCCLUDER_QPOS
# stands either upright unchanged. Olive-oil's round footprint has no yaw dependence -- ideal
# for the planned circle-of-occluders. To swap back to the milk box: model "038_milk-box",
# id 2, and bump OCC_HALF_FOOTPRINT/_OCC_HALF back to 0.08/0.06.









# Two-step (around-the-box) planning: curobo won't reliably route around a tall obstacle
# between start and goal, so before the grasp (and before the place) we send the gripper
# to a waypoint beside the occluder on the arm's own side, then reach in. The waypoint's
# x-offset from the box CENTRE = OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP, i.e. SIDE_WAYPOINT
# _GAP is the clearance from the box EDGE to the gripper (not from centre). If the waypoint
# would leave the reachable x-range we flip to the other (table-centre) side. All tunable.
# The pre_beside_box move (right after lift+attach_object+enable_table) was found
# empirically to fail with MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION in
# 100% of its failures (5/5 in a 24-seed sweep) -- meaning the STARTING qpos for
# that plan is already colliding with the just-re-enabled table, before any target
# pose even matters. A 0.1m lift isn't always enough clearance for the held
# object's true collision volume (voxelized via attach_object) to clear the table
# once its collision is turned back on. Bumping the post-grasp lift height directly
# targets this; must stay consistent across dry-run and real execution.
# 0.15 (first attempt) reduced this failure 5/5 -> 4/24 seeds and raised the
# occluder-present success rate to 5/16. Pushing further to 0.2 was WORSE on both
# counts (4/16 successes, 6 pre_beside_box collisions) -- not monotonic, likely
# because more lift shifts the arm into a different (sometimes worse) posture
# rather than simply buying more clearance. Reverted to the better-validated 0.15;
# some of the 0.2 regression is plausibly sampling noise (documented CuRobo
# run-to-run nondeterminism) rather than a true causal effect, but 0.15 is the
# only value with two consistent, positive data points.
# Minimum fraction of GRASP_LIFT_HEIGHT the TARGET OBJECT itself (not just the
# gripper) must actually rise for the grasp to be considered real. Confirmed
# empirically (seeds 11/19/26): a fully missed grasp still reports the lift
# MOVE as plan_success=True (CuRobo found a valid plan for the commanded
# gripper motion) while the object's z stays flat (0.0-0.6% of the commanded
# rise) -- a wide margin below any plausible "successful but marginal" grasp,
# so 0.5 has a lot of room without risking false positives on a real grasp.
# Max drift (meters) of the object's position, expressed in the gripper's LOCAL
# frame, from the baseline captured right after attach_object, before a stage
# is considered to have lost the grasp. Confirmed empirically (seed 27): every
# placement stage reported plan_success=True while the object physically fell
# mid-transit (~10-60cm drift) -- plan_success only reflects the arm's motion
# plan, never the held object's actual state. 0.03 gives room for simulation
# jitter while still catching any real slip/drop, which in every observed case
# was at least an order of magnitude larger.
# Max rotation drift (radians) of the object's orientation, expressed in the
# gripper's local frame, from the baseline -- translation alone misses a
# held object rotating loose without translating past OBJECT_RETENTION_
# TOLERANCE, leaving CuRobo's attached collision model (which assumes a
# fixed rigid transform) inconsistent with the physical object. ~11 degrees.
# How close (meters) the held object's actual pose must be to the intended
# final placement pose (self.des_obj_pose) to treat place_actor's descent as
# genuinely DONE (skip remaining moves, open the gripper) rather than a loss
# or an ambiguous mid-descent contact (see DESCENT_CONTACT_DRIFT_TOLERANCE).
# Deliberately mirrors put_mouse_on_pad.check_success()'s own criterion
# (independent x/y error each < 0.02) with a small safety margin, INSTEAD of
# a looser combined radius: an earlier 8cm combined-radius version accepted
# seed 9 (dx=3.24cm) and seed 12 (dx=6.11cm, dy=2.85cm) as "arrived" even
# though check_success() would still reject both on at least one axis --
# stopping the descent there only guaranteed a failure the remaining slices
# might otherwise have corrected.
# Contact handling uses the task's exact per-axis success threshold. Once the
# object has settled, a gripper-relative retention check is no longer meaningful;
# release only when the task's own XY predicate already passes.
# Gripper-relative drift (meters) beyond OBJECT_RETENTION_TOLERANCE that's
# still treated as ambiguous-possible-contact rather than an outright loss,
# for place_actor's descent specifically: the object settling onto the pad/
# table a bit early (before reaching the precise final target) changes the
# gripper-relative transform similarly to a drop, but isn't one. Confirmed
# empirically: benign early-contact cases (seeds 9/12) showed ~6-7cm drift,
# while genuine catastrophic drops (seeds 20/25/27) showed 30cm-1.4m -- a
# wide, unambiguous gap. Below this, the descent continues toward the real
# target instead of failing early; above it, it's an unambiguous loss.
# How close (meters) the held object's world-frame z must be to the intended
# resting height (self.des_obj_pose[2]) for a DESCENT_CONTACT_DRIFT_TOLERANCE-
# sized drift to be plausibly explained by contact with the placement surface,
# rather than the early stage of a genuine drop -- a real drop passes through
# the same 4-15cm drift band while still well above the surface (falling
# under gravity, unconstrained by the descent's planned trajectory), so
# position drift magnitude ALONE can't distinguish the two. Looser than
# OBJECT_PLACEMENT_Z_TOLERANCE (which gates a full "already placed" stop)
# since this only needs to confirm plausible surface contact, not a precise
# final resting pose.
# create_grasp_approach_metric's tstep_fraction for place_actor's descent
# slices specifically -- lower than CuRobo's own default (0.8, appropriate for
# a single large grasp-approach jump) because these slices are already tiny
# (DESCENT_SLICE_SIZE). tstep_fraction only holds the straight-line/
# orientation-locked cost for the trajectory's LAST (1-tstep_fraction)
# fraction of timesteps -- the rest is a soft-cost-only "free" portion.
# Confirmed empirically (seed 8, place_actor:move1 slice 4): with a FIXED
# 0.8, a 3cm slice's trajopt solve ballooned to 980 timesteps and swung the
# gripper 56cm away (up to z=0.70m) before snapping back onto the line for
# the final 20% -- the free portion has no incentive to stay short once
# trajopt escalates to a much longer time horizon (which it does more
# readily right at the final approach, where collision margins against the
# placement surface are tightest).
#
# A single fixed tight value isn't enough either: re-testing with a FIXED
# 0.2 fixed the excursion (deviation dropped from 56cm to ~2cm) but made one
# slice fail FINETUNE_TRAJOPT_FAIL on all 10 retries -- an overconstrained
# geometry where the exact straight line genuinely conflicts with collision
# margins needs at least some slack, and retrying the identical
# overconstrained problem 10 times can't discover that. Progressive
# relaxation instead: start tight (least detour room) for a few attempts,
# then relax toward CuRobo's own default across the remaining retry budget
# -- giving a genuinely-infeasible-under-tight-constraint slice a real
# chance to solve, while a hard path-safety filter (DESCENT_MAX_PATH_
# DEVIATION and friends below) still rejects any candidate at ANY fraction
# that takes an implausibly circuitous path, so relaxing never re-admits
# the original 56cm-loop failure mode.
# Max perpendicular distance (meters) a descent slice's planned Cartesian EE
# path may stray from the FINITE straight-line SEGMENT between its own start
# and end pose before being rejected -- a backstop against the pathological-
# detour failure mode above, independent of which tstep_fraction produced
# it. Well above the sub-2mm wobble seen on well-behaved slices and well
# below the 56cm pathological case, and larger than a single slice's own
# straight-line distance (DESCENT_SLICE_SIZE) to leave room for legitimate
# small detours. Must be measured against the CLAMPED segment, not the
# infinite line through it -- an unclamped projection would report near-zero
# deviation for a trajectory that overshoots straight past the target along
# the same axis, which is exactly the kind of detour this needs to catch.
# Max ratio of a descent slice's actual Cartesian path length (sum of
# consecutive EE waypoint distances) to its straight-line distance --
# catches a trajectory that stays geometrically close to the line (small
# DESCENT_MAX_PATH_DEVIATION) but travels back and forth along it far more
# than the direct distance requires. A conservative multiple of 1.0 (a
# perfectly straight path); not tuned against a large empirical sample yet,
# so treat as provisional pending broader validation.
# Max ratio of a descent slice's total joint-space path length (sum of
# consecutive joint-position deltas) to the direct start->end joint-space
# distance -- catches joint-space winding that an EE-only Cartesian check
# can miss entirely (a redundant/near-singular arm can reach nearly the same
# EE pose through very different, looping joint configurations). Same
# provisional-pending-validation caveat as DESCENT_MAX_PATH_LENGTH_RATIO.
# Max excursion (radians) of any SINGLE joint's range (max-min position)
# across a descent slice's trajectory -- flags one joint spinning through an
# implausible range for what should be a tiny slice even when the aggregate
# DESCENT_MAX_JOINT_TRAVEL_RATIO looks acceptable (that ratio can stay small
# if only one joint winds while the others move normally, since it's an
# L2-norm aggregate). Same provisional-pending-validation caveat.
# Max absolute joint-space distance (radians, L2 norm across all active
# joints) between a descent slice's own trajectory start and end -- catches
# a monotonic (non-winding) move to a DISTANT IK branch that the ratio-only
# checks above can miss entirely: a straight-line jump has joint_travel_ratio
# near 1.0 regardless of magnitude, and several joints each moving under
# DESCENT_MAX_JOINT_RANGE individually can still sum to a large aggregate
# displacement (e.g. six joints at ~0.9 rad each is an L2 norm of ~2.2 rad,
# passing max_joint_range=1.0 easily). A tiny Cartesian slice should need
# only a small joint-space move; not tuned against a large empirical sample
# yet, so treat as provisional pending broader validation.
# Max absolute joint-space path length (radians) traveled during a descent
# slice -- an absolute companion to DESCENT_MAX_JOINT_TRAVEL_RATIO (which
# only bounds the path as a MULTIPLE of the endpoint distance, so it can't
# catch a monotonic jump to a distant branch on its own, same gap as
# DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT above). Slightly above that
# endpoint-displacement cap since path length is always >= endpoint
# distance. Same provisional-pending-validation caveat.
# Local landing-region search (place_actor's final approach): instead of
# forcing CuRobo to reach one exact final pose -- which kept failing
# FINETUNE_TRAJOPT_FAIL right at the surface, where the attached object's
# collision margin against the table is tightest -- search a small grid of
# nearby landing poses and accept the first one (at the lowest feasible
# release height) that plans cleanly and passes the same trajectory-safety
# filters used elsewhere, then let physics settle the final contact instead
# of commanding CuRobo all the way down to it.
#
# XY offsets (meters) tried per axis around the nominal target, combined
# into a grid -- matches check_success's own per-axis 2cm tolerance with a
# safety margin (max offset 1.5cm, comfortably under OBJECT_PLACEMENT_XY_
# TOLERANCE's 1.8cm).
# Candidate release heights (meters above the target's resting height),
# tried lowest first -- a lower release means less for physics to settle
# and stays closest to the nominal target, but a higher one gives CuRobo
# more collision-margin room to actually plan to when the exact surface
# height is too tight to solve.
# Max number of IK-feasible candidates (closest-to-nominal-target first) to
# actually PLAN per release height -- IK checks are cheap, full CuRobo plans
# aren't, so this bounds cost instead of exhaustively planning every
# candidate in the grid.
# Stop planning further candidates at a given height once this many have
# been accepted (passed both plan success and the path-safety filter) --
# picks among a reasonable sample instead of paying for the full budget
# above every time.
# Max XY distance (meters) from the target, and required object retention,
# for _local_landing_search_and_place to even attempt its search -- its
# candidate moves are short by design, so if it were ever invoked from
# farther away its filters would reject a legitimate longer approach
# instead of a genuinely bad one. The caller currently only reaches this
# function once already close (via the placement chain + pre_place_descent),
# but this makes that assumption an explicit, checked precondition rather
# than an unstated one.
#
# Deliberately XY-only, not a combined 3D radius: XY offset directly bounds
# which of the +-1.5cm candidate offsets could plausibly land within
# check_success's own tolerance, while the remaining VERTICAL gap is a
# different concern (how far the arm still has to descend) governed by
# LANDING_SEARCH_MAX_Z_DISTANCE below instead -- a combined 3D check would
# treat "5cm lateral, no vertical drop left" the same as "no lateral offset,
# 5cm still to descend", which aren't comparable for this search.
# Max vertical (Z) distance (meters) from the target for the same gate --
# looser than the XY condition since the whole point of this search is
# picking a release height (LANDING_RELEASE_HEIGHTS, up to 8cm) rather than
# requiring the arm to already be at the exact final height.
# Max number of DIFFERENT contact-point candidates to actually try grasping in
# play_once when grasp_verify detects a missed grasp, before giving up on the
# episode entirely. _select_pick_place_candidate already ranks/generates
# several candidates for the dry-run search -- this reuses that same ranking
# to pick a genuinely different contact point on retry (via exclude_cp_ids)
# instead of ending the episode after the first missed grasp.
# Replay attached-object trajectories more slowly without changing their
# planned geometric path. A factor of 2 inserts one interpolated control point
# between each pair and scales commanded velocity accordingly.
# Slice length (meters) for place_actor's deterministic descent (see
# _plan_pose_with_descent_slices). place_actor's moves are short, mostly-
# straight-line final approaches with KNOWN structure (fixed target
# orientation, held via approach_axis) -- a scalar-error-informed shrink
# doesn't exploit that the way a fixed-size slice along the straight line to
# the target does. 0.04 sits in the requested 3-5cm range.
# Fallback gaps tried ONLY when none of SIDE_WAYPOINT_GAPS's beside_box target verifies
# reachable (check_ik_batch, no trajopt) -- confirmed empirically (seed 9, right arm,
# box far from the shoulder in x): a check_ik_batch sweep at fixed y/z showed a clean
# reachable/unreachable cutoff at gap<=0.16 vs gap>=0.18, i.e. ALL of SIDE_WAYPOINT_GAPS
# (0.20-0.28) were unreachable for this box position while smaller gaps were fine. The
# fixed 3-value set silently assumed the arm can always reach OCC_HALF_FOOTPRINT+gap
# from the box on its own side, which breaks down for boxes positioned far from the
# shoulder. Only consulted as a fallback (extra search cost) when the preferred, more
# clearance-safe gaps all fail to verify.
# Direct-plan retries for the post-attach placement search specifically (seed 8 class):
# a check_ik_batch/live-replay probe showed the SAME post_grasp_escape/pre_beside_box
# move -- confirmed independently reachable via pure IK -- failed 180/180 times with
# INVALID_START_STATE_WORLD_COLLISION in one background run, then succeeded on EVERY
# attempt in a fresh replay of the identical seed: the live qpos right after lift+
# enable_table sits close enough to the table that whether it's flagged in collision is
# sensitive to CuRobo's own internal trajopt seeding (the same marginal-feasibility
# nondeterminism already documented for place_actor, just showing up one stage earlier
# now that the search finds this stage instead of silently skipping it). Retrying the
# direct plan a few times before falling back to bridge waypoints gives that seeding
# more chances to land on a working solution, at far lower cost than enabling bridges
# (each retry is 1 extra trajopt call per combo; each bridge is ~local_attempts more).
# A direct IK-reachability sweep (check_ik_batch diagnostic) showed the waypoint's
# feasibility roughly DOUBLES at +0.15m above grasp height (64%) vs the +0.0/+0.06m
# band this used to test alone (29-36%) -- x/gap and going even higher (+0.30, 54%)
# both matter far less. +0.15 is the empirically-found sweet spot, not a guess.
# Fallback clearance heights tried when NONE of PLACE_CLEARANCE_ZS verifies reachable
# for lift_above_box (mirrors SIDE_WAYPOINT_GAPS_FALLBACK's reasoning for gaps): a
# check_ik_batch(relax_orientation=True) height sweep at each seed's beside_box x/y
# showed the REAL reachable window sits around z~0.8-1.05, i.e. PLACE_CLEARANCE_ZS's
# two higher values (1.15, 1.25) are unreachable outright regardless of orientation,
# and even 1.05 needs orientation relaxed to be reachable for some gaps. These lower
# values give the search somewhere to fall back to if 1.05 isn't enough either.
# Stages that must plan with a fixed (non-relaxed) orientation even when the rest
# of the placement chain uses relax_orientation=True: center_over_pad is the last
# placement subgoal before place_actor's own strict final descent, so it needs to
# hand off a KNOWN, expected orientation, not whatever CuRobo happened to converge
# to under a free-orientation goal. See _plan_pose_trajectory_sequence.
# Max iterations for the adaptive waypoint-shrink retry (see
# _plan_pose_with_shrinking_waypoint), used whenever a direct Cartesian plan
# fails. 0 preserves pure direct planning. Replaces the previous fixed-offset
# bridge-pool fallback: instead of blindly trying hand-picked offset directions,
# a failed attempt's target is iteratively pulled halfway toward the arm's
# current pose and retried, chaining a follow-up attempt at the REAL target
# once a shrunk intermediate succeeds. Default lowered from the old pool's 9 to
# 5 since this converges toward an always-eventually-reachable point (the
# current pose itself), needing fewer attempts than searching fixed offsets
# blindly. POST_GRASP_ESCAPE_ATTEMPTS is accepted as a backward-compatible
# alias for runs launched before this became universal.
# Give up shrinking once the remaining distance from the current pose to the
# (possibly already-shrunk) target drops below this -- not worth chasing an
# even-tinier hop that makes no real progress toward the actual destination.
# Both a second ("top_down") waypoint orientation and a +/-0.05m y-offset search were
# tried here and measured empirically to fail at the SAME rate as the baseline (both
# orientations: ~equal IK_FAIL proportion; y-offset: no change in failure proportion
# either) -- i.e. neither is the actual bottleneck (that's pure kinematic unreachability
# at the position itself). Collapsed back to single values so this search space doesn't
# multiply the cost of the (expensive, real CuRobo batch-plan) grasp-pose check below
# for no measured benefit. Re-widen only with new evidence, not speculatively.
# Ordered chain stages (see _plan_candidate) -- used to rank how far a candidate got
# before failing, so the fallback (used when NO candidate fully plans) can pick the
# one that progressed furthest instead of whichever was generated first. Confirmed via
# a check_ik_batch reach-envelope probe: a fully-reachable contact point (both pre_grasp
# and grasp) can exist among the ones tried, but the old "first-generated" fallback
# picked an unreachable one instead because it happened to rank #1 geometrically.

# Reject a (seed, offset) build if the occluder would sit within OCC_PAD_CLEARANCE
# (edge-to-edge) of the destination pad, so the occluder can never block/overlap the
# target's final placement. Center-to-center threshold = pad_half + occluder_half + gap.




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
        spawn_occluder = False
        # Occluder ring: num_occluders bottles equally spaced (2*pi/n apart) on a circle of
        # radius occluder_offset centred on the target. occluder_offset is the RING RADIUS
        # (kept under its old name so the reachability/gripper/swept tools that set
        # env.occluder_offset still get a single front box). occluder_angle0 rotates the
        # whole ring; 0 puts occluder k=0 directly in front (-y). num_occluders=1 == the
        # original single front occluder. See occluder_ring_xy().
        occluder_offset = 0.2         # ring radius in metres (target at centre)
        num_occluders = 1
        occluder_angle0 = 0.0         # radians; angle of occluder k=0, measured from -y (front)
        occluder_radii = None         # per-occluder radii (list); None -> occluder_offset for all
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

            # occluder ring: num_occluders tall olive-oil bottles equally spaced (2*pi/n
            # apart) on a circle of radius occluder_offset around the target. k=0 sits
            # directly in front (-y) when occluder_angle0=0; num_occluders=1 reproduces the
            # original single front box. self.occluder aliases ring bottle 0 so the
            # single-box tools (reachability/gripper/swept) keep working unchanged.
            self.occluders = []
            self.occluder = None
            if self.spawn_occluder and self.num_occluders > 0:
                mp = self.target_obj.get_pose().p
                # Per-occluder radii when the caller supplied them (--offsets given a range),
                # else the single scalar radius. The scalar path keeps every external tool
                # that only sets env.occluder_offset (reachability / gripper / swept volume)
                # working unchanged.
                radii = getattr(self, "occluder_radii", None) or self.occluder_offset
                # Drop any ring position whose centre would leave the tabletop -- keep the
                # formation with fewer bottles rather than spawning one off the table.
                ring = occluder_ring_xy(float(mp[0]), float(mp[1]),
                                        radii, self.num_occluders,
                                        self.occluder_angle0,
                                        xlim=TABLE_XLIM, ylim=TABLE_YLIM)
                for ox, oy in ring:
                    occ_pose = rand_pose(xlim=[ox], ylim=[oy], qpos=OCCLUDER_QPOS,
                                         rotate_rand=True, rotate_lim=[0, 3.14, 0])  # random yaw
                    occ = create_actor(
                        scene=self, pose=occ_pose, modelname=OCCLUDER_MODEL, convex=True,
                        model_id=OCCLUDER_ID, scale=[1.0, 1.0, 1.0],
                    )
                    occ.set_mass(0.1)
                    # Register each occluder in curobo's collision world so the planner
                    # actually avoids it. NOT flagged is_obstacle -> it survives the
                    # exclude_obstacles=True pass used in collision-metrics/eval mode
                    # ("... curobo planner skips clutter obstacles"), which only drops
                    # is_obstacle=True procedural clutter.
                    self.collision_list.append({
                        "actor": occ,
                        "collision_path": f"{os.environ['BENCH_ROOT']}/{OCCLUDER_COLLISION}",
                    })
                    self.occluders.append(occ)
                # self.occluder aliases ring bottle 0 for the single-box tools; stays None
                # if every ring position was filtered off-table (an empty formation).
                if self.occluders:
                    self.occluder = self.occluders[0]

        def play_once(self):
            # Expert plan. FORWARD (grasp) below is fixed. BACKWARD (placement) is a list
            # of subgoals returned by self._backward_subgoal_poses() -- EDIT THAT METHOD to
            # add / remove / reorder placement subgoals; each subgoal is one curobo move.
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"

            # Structured failure recorder (replaces having to grep '[play_once] after'
            # lines out of ROBOTWIN_LOG_MOVE logs by hand): the FIRST stage whose
            # checkpoint sees plan_success go False, plus whatever specific CuRobo
            # fail_reason our own plan+replay helpers captured for it (None for
            # stages still going through the standard self.move() path, which
            # doesn't expose a reason). Read by run() after run_rollout() and written
            # into records.jsonl. Reset per-episode since env is reused across seeds.
            self.rollout_failure_stage = None
            self.rollout_failure_reason = None
            self.rollout_candidate_info = None
            self._last_fail_reason = None
            self._grasp_baseline_transform = None
            self._slow_attached_replay = False
            # Richer diagnostics read by run() into records.jsonl alongside the
            # single-value failure/candidate fields above: per-attempt grasp
            # history (which contact points were tried and why each failed, not
            # just the last one), per-check retention drift (position + rotation,
            # not just the pass/fail it produced), and per-slice descent detail
            # (CuRobo's own position/rotation error on each place_actor segment).
            self.rollout_grasp_attempts = []
            self.rollout_retention_checks = []
            self.rollout_descent_slices = []
            # ROBOPRO Phase 4 (APPROACH_MODE A/B): per-plan curobo effort and per-build
            # seed outcomes for this episode. rollout_plan_effort answers "did the seed
            # let curobo converge in fewer attempts"; rollout_seed_stats answers "how
            # often did the seed actually fire" (seeding is opportunistic -- IK is
            # nondeterministic, so build_seed sometimes returns no route for inputs that
            # worked a moment earlier). Both reset per-episode; env is reused across seeds.
            self.rollout_plan_effort = []
            self.rollout_seed_stats = []
            # Per-phase wall-clock timing: each checkpoint(name) records how long the
            # expert spent since the previous checkpoint. Read by run() and written
            # into the per-rollout log file. Reset per-episode (env reused across seeds).
            self.rollout_stage_times = []
            self._stage_clock = time.perf_counter()

            # Re-register the FULL collision world (clutter included) before any
            # planning. setup_demo ran update_world(exclude_obstacles=True) because the
            # bench config sets enable_collision_metrics=true -- that drops every
            # is_obstacle=True procedural clutter object from curobo's world, so the
            # expert would plan straight THROUGH clutter and knock it over (even on
            # "successful" rollouts). The whole point here is to test how well the
            # expert AVOIDS clutter, so curobo must see it: update_world() (no exclude)
            # rebuilds the world from collision_list including clutter. The physics-based
            # collision metrics (check_collisions, gated by enable_collision_metrics) are
            # independent of curobo's world, so they still measure any residual hits.
            self.update_world()

            def checkpoint(name):
                # Diagnostic: the dry-run candidate search only verifies the
                # waypoint/pre_grasp/grasp/lift/placement POSES are independently
                # reachable via a chained last_qpos rollout -- it never calls
                # attach_object/enable_table (which change the live collision world)
                # and never tests place_actor's final descent at all. A candidate can
                # pass the dry run and still die at any of these real-execution-only
                # steps, or at a step the dry run "verified" but which replans from a
                # genuinely different qpos in real execution. This pinpoints which.
                if not self.plan_success and self.rollout_failure_stage is None:
                    self.rollout_failure_stage = name
                    self.rollout_failure_reason = self._last_fail_reason
                now = time.perf_counter()
                self.rollout_stage_times.append(
                    {"stage": name, "seconds": round(now - self._stage_clock, 4)})
                self._stage_clock = now
                if log_move:
                    print(f"[play_once] after {name}: plan_success={self.plan_success}")
                # Optional stage observer (set by tooling, e.g. swept_volume_3d.py, to slice
                # the rollout into named stages). Only read via getattr; never set normally.
                stage_hook = getattr(self, "stage_hook", None)
                if stage_hook is not None:
                    stage_hook(name)

            target_p0 = self.target_obj.get_pose().p        # original target location (pre-grasp)
            self._place_target_y = float(target_p0[1])      # available to backward subgoals
            arm_tag = ArmTag("right" if target_p0[0] > 0 else "left")
            # Retry with a genuinely different contact-point candidate when
            # grasp_verify (below) detects a missed grasp, instead of ending the
            # episode after the first miss. _select_pick_place_candidate already
            # ranks/generates several candidates for the dry-run search --
            # exclude_cp_ids reuses that same ranking rather than re-deriving it.
            # Intermediate failed attempts reset plan_success/rollout_failure_*
            # so a LATER successful attempt isn't left alongside a stale failure
            # recording from an earlier, since-recovered-from miss.
            tried_cp_ids = set()
            candidate_plan = None
            cp_id = None
            grasp_succeeded = False
            retry_reset_pose = list(self.get_arm_pose(arm_tag))
            for grasp_attempt in range(1, GRASP_VERIFY_MAX_CANDIDATES + 1):
                candidate_plan = self._select_pick_place_candidate(arm_tag, exclude_cp_ids=tried_cp_ids)
                cp_id = candidate_plan["contact_point_id"]
                if cp_id is None:
                    # No viable contact-point candidate at all (e.g. the object's
                    # current geometry/pose gives _rank_side_grasp_ids nothing to
                    # rank) -- retrying can't help since nothing about the object
                    # or arm has changed. Executing the grasp anyway would fall
                    # through to an unconstrained default grasp
                    # (grasp_actor_from_table with contact_point_id=None) whose
                    # failure then overwrites the real cause with a generic
                    # grasp/lift error and reason=None. Fail fast with the real
                    # reason instead.
                    self.plan_success = False
                    self._last_fail_reason = "no_ranked_contact_points"
                    checkpoint("grasp_candidate_selection")
                    self.rollout_grasp_attempts.append({
                        "attempt": grasp_attempt,
                        "contact_point_id": None,
                        "success": False,
                        "failed_stage": self.rollout_failure_stage,
                        "fail_reason": self.rollout_failure_reason,
                    })
                    break
                tried_cp_ids.add(cp_id)

                # Belt-and-braces: _candidate_specs already emits grasp_waypoint=None outside
                # 'off' mode. If one ever leaks through again, the `else` below would live-plan
                # and EXECUTE the around-box heuristic, silently putting it back into the
                # waypoint-free modes and invalidating the A/B. Skip it loudly instead.
                if candidate_plan["grasp_waypoint"] is not None and self._approach_mode() != "off":
                    print("\033[93m[waypoint] BUG: a grasp_waypoint survived in "
                          f"APPROACH_MODE={self._approach_mode()}; skipping it\033[0m")
                elif candidate_plan["grasp_waypoint"] is not None:
                    wp_trajectory = candidate_plan.get("grasp_waypoint_trajectory")
                    if wp_trajectory is not None:
                        self._replay_planned_sequence(arm_tag, wp_trajectory)
                    else:
                        self._plan_and_replay_pose(arm_tag, candidate_plan["grasp_waypoint"])
                    checkpoint("waypoint_move")
                self._grasp_via_cached_trajectories(arm_tag, candidate_plan)
                checkpoint("grasp")

                pre_lift_object_z = float(self.target_obj.get_pose().p[2])
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=GRASP_LIFT_HEIGHT))
                checkpoint("lift")
                # Verify the grasp actually captured the object before trusting it. plan_success
                # only reflects whether CuRobo found a valid motion plan for the COMMANDED
                # gripper move -- it says nothing about whether the object moved with it.
                # Confirmed empirically (seeds 11/19/26): the gripper lifted the full commanded
                # GRASP_LIFT_HEIGHT while the object stayed frozen at its original table pose
                # (z-rise 0.0-0.6% of the commanded height) -- a fully missed grasp, not a
                # marginal one. Left unchecked, attach_object below bakes the object's real
                # (still-on-table) pose into CuRobo's collision model as if it were rigidly
                # held by the now-lifted gripper, producing a nonsensical attached-mesh
                # position that immediately collides with the table the instant
                # enable_table(True) runs a few lines down -- misattributed as a placement/
                # pre_beside_box planning bug rather than what it actually is: a failed grasp.
                if self.plan_success:
                    object_rise = float(self.target_obj.get_pose().p[2] - pre_lift_object_z)
                    if object_rise < GRASP_LIFT_HEIGHT * GRASP_VERIFY_MIN_RISE_FRACTION:
                        self.plan_success = False
                        self._last_fail_reason = f"grasp_missed(object_z_rise={object_rise:.3f}m)"
                checkpoint("grasp_verify")
                attempt_record = {
                    "attempt": grasp_attempt,
                    "contact_point_id": cp_id,
                    "success": self.plan_success,
                    "failed_stage": self.rollout_failure_stage,
                    "fail_reason": self.rollout_failure_reason,
                }
                self.rollout_grasp_attempts.append(attempt_record)
                if self.plan_success:
                    grasp_succeeded = True
                    break
                if grasp_attempt < GRASP_VERIFY_MAX_CANDIDATES and cp_id is not None:
                    if log_move:
                        print(f"[grasp_verify] attempt {grasp_attempt} missed (cp_id={cp_id}); "
                              f"retrying with next candidate")
                    self.plan_success = True
                    self.rollout_failure_stage = None
                    self.rollout_failure_reason = None
                    self._last_fail_reason = None
                    # The failed attempt closed the gripper; restore both the
                    # gripper and table state before selecting another contact point. The table must go back to
                    # ENABLED here, not stay disabled: _grasp_via_cached_
                    # trajectories' own waypoint_move/pre_grasp leg expects to
                    # plan WITH table collision active (that's why it only
                    # disables the table partway through, right before the
                    # close final approach) -- leaving it disabled would let
                    # the next candidate's waypoint/pre_grasp plan straight
                    # through the table.
                    self.move(self.open_gripper(arm_tag))
                    self.enable_table(enable=True)
                    # Candidate plans are generated from the live arm state. A
                    # missed grasp leaves the arm at its lifted endpoint, so
                    # selecting the next candidate there made every retry solve
                    # a different, frequently colliding problem. Return to the
                    # common pre-attempt pose before selecting another contact.
                    self._plan_and_replay_pose(arm_tag, retry_reset_pose)
                    attempt_record["reset_success"] = bool(self.plan_success)
                    if not self.plan_success:
                        reset_reason = self._last_fail_reason or "unknown"
                        attempt_record["reset_fail_reason"] = reset_reason
                        self._last_fail_reason = f"grasp_retry_reset_failed:{reset_reason}"
                        checkpoint("grasp_retry_reset")
                        break
                else:
                    break
            if not grasp_succeeded:
                # Retry cleanup can temporarily restore plan_success so the next
                # candidate can run. Exhausting the loop is still a failed grasp.
                self.plan_success = False
                if self._last_fail_reason is None:
                    self._last_fail_reason = "grasp_candidates_exhausted"
                checkpoint("grasp_verify")
                return
            self.attach_object(
                self.target_obj,
                f"{os.environ['BENCH_ROOT']}/assets/objects/{self.target_model}/collision/base{self.target_id}.glb",
                str(arm_tag),
            )
            checkpoint("attach_object")
            # Baseline for _object_retained: the object's full pose relative to the
            # gripper right now, while we still trust the grasp (grasp_verify just
            # passed). Every subsequent replay step compares against this.
            self._grasp_baseline_transform = self._gripper_relative_object_transform(arm_tag)
            self._slow_attached_replay = True
            self.enable_table(enable=True)
            checkpoint("enable_table")

            # Re-select the placement geometry under the REAL attached-object world,
            # but rank candidates by whether the FULL post-attach motion chain plans
            # successfully, not just by endpoint IK. Cache the winning trajectories so
            # these stages are replayed exactly instead of being planned again live.
            # PLACEMENT_MODE=direct: no subgoals, so there is no geometry to re-select --
            # skip the whole attached-world search (it is the scene-specific part: it scores
            # _box_side_x corridors against the PLACE_CLEARANCE_ZS ladder) and hand the lift
            # straight to place_actor below, via the one generic blended intermediate.
            if not candidate_plan["placement_subgoals"]:
                placement_plan = {"placement_subgoals": [], "trajectories": [], "verified": True}
                if log_move:
                    print("[placement] PLACEMENT_MODE=direct -- no placement subgoals; "
                          "lift -> intermediate -> place_actor")
            else:
                _beside_box_pose = candidate_plan["placement_subgoals"][0][1]
                _lift_z = _beside_box_pose[2]
                _quat = list(_beside_box_pose[3:])
                placement_plan = self._select_attached_placement_plan(
                    arm_tag,
                    tgt_y=self._place_target_y,
                    lift_z=_lift_z,
                    quat=_quat,
                )
                if log_move:
                    print(f"[placement] verified x_side={placement_plan['x_side']:.3f} "
                          f"clearance_z={placement_plan['clearance_z']:.2f} "
                          f"escape={placement_plan.get('escape_idx', 0)} "
                          f"quat={placement_plan['quat']}")
            candidate_plan["placement_subgoals"] = placement_plan["placement_subgoals"]
            for stage_name, traj in placement_plan["trajectories"]:
                self._replay_planned_move(arm_tag, traj)
                if self.plan_success and not self._object_retained(arm_tag, context=stage_name):
                    self.plan_success = False
                    self._last_fail_reason = f"object_lost:{stage_name}"
                checkpoint(stage_name)
            # _select_attached_placement_plan's search can exhaust every geometry/
            # escape combo without ever finding one that plans ALL the way through
            # (verified=False): it then falls back to the DEEPEST partial chain it
            # found, whose "trajectories" list is truncated at the stage that kept
            # failing (confirmed empirically: seed 8/9 checkpoint traces jumped
            # straight from an early placement subgoal to place_actor's
            # pre_place_descent, skipping lift_above_box/over_box_to_pad_y/
            # center_over_pad entirely, yet still reported rollout_success=True).
            # The loop above only iterates what's actually IN that truncated list,
            # so it silently "completes" without ever calling checkpoint() for the
            # missing subgoals -- plan_success never goes False. Make the fallback
            # an explicit, correctly-attributed failure instead of a silent skip.
            if self.plan_success and not placement_plan.get("verified", False):
                self.plan_success = False
                self._last_fail_reason = placement_plan.get("fail_reason", "unverified_placement_fallback")
                checkpoint(placement_plan.get("failed_stage") or "placement:unverified_fallback")
            # place_actor lowers the object from the last subgoal onto the pad.
            # constrain="free": check_success is position-only, so alignment buys nothing
            # and was a likely IK_FAIL cause for a tall object held near its top.
            # dis=0.02 (was 0.005): the wide-sweep checkpoints showed 3/3 episodes that
            # reached this step failed here with MotionGenStatus.FINETUNE_TRAJOPT_FAIL --
            # a 5mm final gap puts the held object's collision volume so close to the
            # table that trajopt's finetune can't converge on a smooth, collision-margin
            # -respecting descent. check_success() (put_mouse_on_pad.py) only checks x/y
            # position + gripper state, no z/height requirement, so a looser release
            # height costs nothing and gives finetune room to converge.
            # That alone didn't fix it: [place_actor] logging showed place_pre_pose ->
            # place_pose isn't a pure vertical drop, it also shifts diagonally in x/y,
            # and still hit FINETUNE_TRAJOPT_FAIL. CuRobo's own stacking example
            # (examples/isaac_sim/simple_stacking.py) uses PoseCostMetric.
            # create_grasp_approach_metric for exactly this: a straight-line approach
            # blended into only the last tstep_fraction of the trajectory. Unlike the
            # plain constraint_pose path (why DROP_GRASP_ORIENTATION_CONSTRAINT exists),
            # it sets offset_tstep_fraction >= 0, which skips CuRobo's start-vs-goal
            # orientation match check, so it won't reject a differing start/goal
            # orientation the way constraint_pose did. approach_axis=2 asks for a
            # straight-line descent (z) into both the pre-pose and final pose.
            place_arm_tag, place_actions = self.place_actor(
                self.target_obj, arm_tag=arm_tag, target_pose=self.des_obj_pose,
                constrain="free", pre_dis=0.05, dis=0.02,
            )
            if not place_actions:        # place_actor bails (plan_success already False)
                return                   # -> avoid place_actions[0] IndexError below
            for a in place_actions:
                if a.action == "move":
                    a.args["approach_axis"] = 2
            # None of the above fixed it: a pure-IK check (object attached, live qpos)
            # showed place_pre_pose/place_pose are BOTH independently reachable and
            # collision-free -- the endpoint is fine. So FINETUNE_TRAJOPT_FAIL here is a
            # path-smoothness problem, not a reachability/collision one: center_over_pad
            # -> place_pre_pose is one big trajopt problem (large vertical drop + lateral
            # shift + orientation change all at once) that the optimizer can't
            # interpolate smoothly, even though both ends are individually valid. Same
            # fix pattern as the grasp side: break it into a smaller intermediate step
            # (position+orientation blend) instead of one large jump.
            # The post-lift transit to place_pre_pose, which the two modes reach very differently.
            #
            # SCRIPTED (unchanged): the chain already ended at center_over_pad, directly above
            # the pad, so all that remains is the short blended hop described above.
            #
            # DIRECT: one move, straight from the lift pose to place_pre_pose, seeded.
            # This replaced a two-hop version (blend, then pad) for a reason worth recording.
            # The blend is an INTERMEDIATE, not a destination. With the chain on, that is
            # invisible -- starting above the pad, the blend lands at the pad. With the chain
            # off it starts at the LIFT pose and stops half way, which is what the first smoke
            # run hit: carry_transit planned Success and was the episode's LAST plan, then
            # landing_search_preconditions_not_met(xy_distance=0.206 m). Appending a second hop
            # fixed the reach but left the seed covering only the first half -- and on the
            # rerun that unseeded second hop was exactly where the seeded cell died
            # (FINETUNE_TRAJOPT_FAIL, 24/24 attempts), while its seeded first hop had converged
            # in ONE attempt with the object held to 4e-5 m.
            #
            # Splitting the move is also the wrong shape for the experiment. The seed's whole
            # claim is that it makes the big jump tractable, so the big jump is what has to be
            # attempted -- unseeded in `direct`, seeded in `seed`. Both cells keep the generic
            # replacement for the hand-placed midpoint: _plan_pose_with_local_waypoint_retry's
            # shrinking-waypoint fallback, which breaks a failed move into smaller hops using
            # CuRobo's own reported position error, reads no scene state, and applies equally
            # to both. So the midpoint is not lost, it is earned rather than assumed.
            if candidate_plan["placement_subgoals"]:
                transit_pose = self._verified_intermediate(
                    arm_tag, candidate_plan["placement_subgoals"][-1][1],
                    place_actions[0].target_pose)
            else:
                transit_pose = list(place_actions[0].target_pose)
            # Phase C seed: goal is the pose actually being planned to, which is now the whole
            # transit. None in every other mode, so this call is byte-identical to the unseeded
            # one there. (Scripted's plan-effort label changes from "pose" to "carry_transit";
            # that is a record label only, no behaviour rides on it.)
            _carry_seed = self._get_carry_seed(arm_tag, transit_pose[:3])
            self._plan_and_replay_pose(arm_tag, transit_pose, seed_traj=_carry_seed,
                                       stage_label="carry_transit")
            if self.plan_success and not self._object_retained(arm_tag, context="placement:pre_place_descent"):
                self.plan_success = False
                self._last_fail_reason = "object_lost:placement:pre_place_descent"
            checkpoint("placement:pre_place_descent")
            if log_move:
                # Diagnostic: is the endpoint itself reachable (pure collision-aware IK,
                # no trajectory optimization/smoothness), at the LIVE qpos with the
                # object actually attached right now? Distinguishes "the pose itself is
                # unreachable/colliding" from "the pose is fine but trajopt can't find a
                # smooth path there" -- four different trajopt/collision-margin fixes
                # all failed identically, so this tells us whether to keep tuning the
                # solver or reconsider the target pose itself.
                check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
                move_poses = [a.target_pose for a in place_actions if a.action == "move"]
                ik_ok = check_ik(move_poses)
                print(f"[place_actor] pure-IK reachability (attached, live qpos): {list(ik_ok)}")
            # Forcing CuRobo all the way down to place_actor's one exact final
            # pose (via _execute_actions_via_plan_and_replay, retries 4 -> 10)
            # was the prior approach here; the broad-sweep validation (seeds
            # 8-31) confirmed it made place_actor the dominant remaining
            # failure (100% MotionGenStatus.FINETUNE_TRAJOPT_FAIL), since the
            # attached object's collision margin against the table is
            # tightest exactly at that one pose. Searching a small grid of
            # nearby landing poses instead -- accepting any that plans
            # cleanly within check_success's own tolerance, then letting
            # physics settle the final contact -- avoids ever asking CuRobo
            # to solve that one hardest point.
            self._local_landing_search_and_place(place_arm_tag)
            checkpoint("place_actor")

        def _blend_pose(self, pose_a, pose_b, t=0.5):
            """Position lerp + quaternion nlerp (normalized linear interpolation --
            a cheap, good-enough slerp approximation for a transit waypoint that
            doesn't need to be exact) between two flat [x,y,z,qw,qx,qy,qz] poses.
            Used to insert an intermediate step in a large single move so trajopt
            gets a smaller, easier motion to solve instead of one big jump."""
            pa, pb = np.asarray(pose_a, dtype=float), np.asarray(pose_b, dtype=float)
            pos = (1 - t) * pa[:3] + t * pb[:3]
            qa, qb = pa[3:], pb[3:]
            if np.dot(qa, qb) < 0:      # quaternion double-cover: align signs before blending
                qb = -qb
            q = (1 - t) * qa + t * qb
            q = q / np.linalg.norm(q)
            return list(pos) + list(q)

        def _verified_intermediate(self, arm_tag, pose_a, pose_b, fractions=(0.5, 0.35, 0.65, 0.25, 0.75)):
            """Same verify-before-committing principle as
            _select_attached_placement_plan, applied to a transit waypoint: a naive
            midpoint blend can itself be unreachable (confirmed empirically -- seed
            11's pre_beside_box waypoint failed at the plain t=0.5 blend). Try a small
            spread of blend fractions and use whichever verifies reachable via
            check_ik_batch, falling back to the plain midpoint if none do."""
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            candidates = [self._blend_pose(pose_a, pose_b, t) for t in fractions]
            ok = check_ik(candidates)
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[intermediate] fractions={fractions} ok={list(ok)}")
            for candidate, reachable in zip(candidates, ok):
                if reachable:
                    return candidate
            return candidates[0]  # none verified reachable; fall back to the plain midpoint

        def _backward_subgoal_poses(self, arm_tag, x_side, clearance_z, lift_pose):
            """Ordered BACKWARD (placement) subgoals for a chosen candidate path.

            Empty in PLACEMENT_MODE=direct. Killed HERE, at the single source every caller
            goes through (_candidate_specs, _plan_candidate, _select_attached_placement_plan
            and its no-geometry fallback, the no-contact-point fallback), rather than at each
            call site -- the same reason the around-box waypoint is suppressed inside
            _candidate_specs: a chain that merely goes unplanned still gets live-re-planned
            downstream by whatever branches on its poses."""
            if self._placement_mode() == "direct":
                return []
            quat = list(lift_pose[3:])
            tgt_y = self._place_target_y
            pad_x, pad_y = self.fixed_pad_xy
            lift_z = float(lift_pose[2])
            subgoals = []
            subgoals.append(("beside_box", [x_side, tgt_y, lift_z, *quat]))
            subgoals.append(("lift_above_box", [x_side, tgt_y, clearance_z, *quat]))
            subgoals.append(("over_box_to_pad_y", [x_side, pad_y, clearance_z, *quat]))
            subgoals.append(("center_over_pad", [pad_x, pad_y, clearance_z, *quat]))
            return subgoals

        def _post_grasp_escape_poses(self, arm_tag, attempts=LOCAL_WAYPOINT_ATTEMPTS):
            """Local escape waypoint candidates from the current held-object pose.

            Candidate zero is `None`, preserving the direct route. The remaining
            candidates are the bounded local waypoint family for this specific
            attached-object transition out of the grasped state.
            """
            current = list(self.get_arm_pose(arm_tag))
            return [None] + self._local_waypoint_candidates(
                arm_tag, current, current, attempts=attempts)

        def _placement_execution_steps(self, arm_tag, placement_subgoals, escape_pose=None):
            """Build the exact post-attach placement chain we want to execute."""
            current_pose = list(self.get_arm_pose(arm_tag))
            steps = []
            if escape_pose is not None:
                steps.append(("placement:post_grasp_escape", escape_pose))
                current_pose = escape_pose
            if placement_subgoals:
                beside_box_pose = placement_subgoals[0][1]
                mid_to_beside_box = self._verified_intermediate(arm_tag, current_pose, beside_box_pose)
                steps.append(("placement:pre_beside_box", mid_to_beside_box))

            prev_pose = None
            for name, pose in placement_subgoals:
                if name == "lift_above_box" and prev_pose is not None:
                    mid_to_lift_above_box = self._verified_intermediate(arm_tag, prev_pose, pose)
                    steps.append(("placement:pre_lift_above_box", mid_to_lift_above_box))
                steps.append((f"placement:{name}", pose))
                prev_pose = pose
            return steps

        def _plan_pose_trajectory_sequence(self, arm_tag, stage_poses, start_qpos,
                                           local_attempts=LOCAL_WAYPOINT_ATTEMPTS, retries=1,
                                           relax_orientation=False):
            """Plan a labeled pose chain and keep the exact successful trajectories.

            relax_orientation is withheld for PLACEMENT_STRICT_ORIENTATION_STAGES
            regardless of the caller's request: relaxing orientation on the LAST
            placement subgoal (center_over_pad, the handoff into place_actor's own
            strict, orientation-sensitive final descent) let CuRobo converge to an
            arbitrary held-object orientation there -- confirmed empirically to
            cause catastrophic placement misses (one seed landed the object ~51m
            from the pad, almost certainly a real collision/physics ejection from
            an orientation the attached-object collision approximation never
            validated) despite every stage reporting plan_success=True. Interior
            transit stages stay relaxed (that's where the real IK_FAIL bottleneck
            was, and they're always followed by at least one more stage -- ending
            at this forced-strict one -- so any orientation drift there gets
            corrected before the place_actor handoff."""
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            trajectories = []
            start_pose = list(self.get_arm_pose(arm_tag))
            for stage_name, pose in stage_poses:
                stage_relax_orientation = relax_orientation and stage_name not in PLACEMENT_STRICT_ORIENTATION_STAGES
                ok, fail_reason, qpos, stage_trajs = self._plan_pose_with_local_waypoint_retry(
                    arm_tag, pose, qpos, stage_name, start_pose=start_pose,
                    local_attempts=local_attempts, retries=retries,
                    relax_orientation=stage_relax_orientation)
                if not ok:
                    return False, stage_name, fail_reason, trajectories
                trajectories.extend(stage_trajs)
                start_pose = pose
            return True, None, None, trajectories

        def _select_attached_placement_plan(self, arm_tag, tgt_y, lift_z, quat,
                                            gaps=SIDE_WAYPOINT_GAPS, clearance_options=PLACE_CLEARANCE_ZS,
                                            fallback_gaps=SIDE_WAYPOINT_GAPS_FALLBACK,
                                            clearance_fallback=PLACE_CLEARANCE_ZS_FALLBACK):
            """Pick placement geometry by FULL attached-world motion-plan success.

            Endpoint IK is a useful prefilter, but the remaining failures here have
            increasingly been about the chained motion itself. This helper keeps the
            same geometry search space, then scores candidates by whether the exact
            post-attach execution chain plans successfully from the current live qpos.
            The winning trajectories are cached for immediate replay.

            Every stage this searches over (post_grasp_escape, beside_box,
            lift_above_box, over_box_to_pad_y, center_over_pad, and the blended
            pre_* transit waypoints between them) is a pure carry-the-held-object
            move -- check_success() never looks at orientation en route, only the
            final x/y placement. So both the prefilters and the real trajopt calls
            below use relax_orientation=True (position-only goals): a
            check_ik_batch(relax_orientation=True) sweep confirmed this opens up
            real, currently-dead combos at lift_above_box that strict full-pose IK
            rejects outright."""
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            quat_options = (quat, GRASP_DIRECTION_DIC["top_down"])
            start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            fallback = None
            fallback_depth = -1
            failure_breakdown = {}
            escape_poses = self._post_grasp_escape_poses(arm_tag, attempts=LOCAL_WAYPOINT_ATTEMPTS)

            # gaps first, fallback_gaps ONLY if NOTHING in gaps ever verified reachable
            # (the seed-9 case: box far enough from the shoulder that ALL of SIDE_
            # WAYPOINT_GAPS's beside_box targets are IK-infeasible, so the full trajopt
            # chain never even had a chance regardless of clearance/escape/quat).
            for gap_values in (gaps, fallback_gaps):
                any_reachable_x = False
                for quat_choice in quat_options:
                    for gap in gap_values:
                        x = self._box_side_x(arm_tag, gap=gap)
                        # Cheap prefilter (no trajopt): beside_box's height (lift_z) and
                        # y (tgt_y) don't depend on clearance_z/escape_idx, so checking
                        # reachability once per (gap, quat) here skips the ENTIRE cz x
                        # escape_idx loop (up to 30 full trajopt chains) for a gap whose
                        # target could never succeed regardless of how it's reached.
                        if not bool(check_ik([[x, tgt_y, lift_z, *quat_choice]], relax_orientation=True)[0]):
                            failure_breakdown[("placement:beside_box", "IK_prefilter_unreachable")] = (
                                failure_breakdown.get(("placement:beside_box", "IK_prefilter_unreachable"), 0) + 1)
                            if log_move:
                                print(f"[placement-plan] gap={gap:.2f} x_side={x:.3f} quat={quat_choice} "
                                      f"skipped: beside_box IK-unreachable (prefilter)")
                            continue
                        any_reachable_x = True
                        for cz in list(clearance_options) + list(clearance_fallback):
                            # Same idea as the beside_box prefilter above, one level
                            # deeper: lift_above_box shares this (x, tgt_y) and only
                            # varies in z (clearance) -- skip a clearance value here
                            # (cheaply) rather than discovering it's dead only after
                            # the full escape_idx loop of trajopt chains.
                            if not bool(check_ik([[x, tgt_y, cz, *quat_choice]], relax_orientation=True)[0]):
                                failure_breakdown[("placement:lift_above_box", "IK_prefilter_unreachable")] = (
                                    failure_breakdown.get(("placement:lift_above_box", "IK_prefilter_unreachable"), 0) + 1)
                                if log_move:
                                    print(f"[placement-plan] gap={gap:.2f} x_side={x:.3f} clearance_z={cz:.2f} "
                                          f"quat={quat_choice} skipped: lift_above_box IK-unreachable (prefilter)")
                                continue
                            placement_subgoals = self._backward_subgoal_poses(
                                arm_tag,
                                x_side=x,
                                clearance_z=cz,
                                lift_pose=[0, 0, lift_z, *quat_choice],
                            )
                            for escape_idx, escape_pose in enumerate(escape_poses):
                                stage_poses = self._placement_execution_steps(
                                    arm_tag, placement_subgoals, escape_pose=escape_pose)
                                ok, failed_stage, fail_reason, trajectories = self._plan_pose_trajectory_sequence(
                                    arm_tag, stage_poses, start_qpos, local_attempts=0,
                                    retries=PLACEMENT_SEARCH_RETRIES, relax_orientation=True)
                                if ok:
                                    return {
                                        "x_side": x,
                                        "clearance_z": cz,
                                        "quat": quat_choice,
                                        "escape_idx": escape_idx,
                                        "placement_subgoals": placement_subgoals,
                                        "trajectories": trajectories,
                                        "verified": True,
                                    }
                                depth = STAGE_ORDER.index(failed_stage.split("placement:")[-1]) if (
                                    failed_stage and failed_stage.startswith("placement:")
                                    and failed_stage.split("placement:")[-1] in STAGE_ORDER
                                ) else -1
                                if depth > fallback_depth:
                                    fallback = {
                                        "x_side": x,
                                        "clearance_z": cz,
                                        "quat": quat_choice,
                                        "escape_idx": escape_idx,
                                        "placement_subgoals": placement_subgoals,
                                        "trajectories": trajectories,
                                        "verified": False,
                                        "failed_stage": failed_stage,
                                        "fail_reason": fail_reason,
                                    }
                                    fallback_depth = depth
                                failure_breakdown[(failed_stage, fail_reason)] = failure_breakdown.get((failed_stage, fail_reason), 0) + 1
                                if log_move:
                                    print(f"[placement-plan] x_side={x:.3f} clearance_z={cz:.2f} "
                                          f"escape={escape_idx} quat={quat_choice} failed at "
                                          f"{failed_stage} ({fail_reason})")
                if any_reachable_x or gap_values is not gaps:
                    break
                if log_move:
                    print(f"[placement-plan] none of gaps={gap_values} verified beside_box-reachable "
                          f"for any quat; trying fallback gaps={fallback_gaps}")
            if log_move and fallback is not None:
                print(f"[placement-plan] no fully planned geometry; using deepest fallback "
                      f"(depth={fallback_depth}) with breakdown={failure_breakdown}")
            if fallback is not None:
                return fallback
            placement_subgoals = self._backward_subgoal_poses(
                arm_tag,
                x_side=self._box_side_x(arm_tag, gap=gaps[0]),
                clearance_z=clearance_options[0],
                lift_pose=[0, 0, lift_z, *quat],
            )
            return {
                "x_side": self._box_side_x(arm_tag, gap=gaps[0]),
                "clearance_z": clearance_options[0],
                "quat": quat,
                "placement_subgoals": placement_subgoals,
                "trajectories": [],
                "verified": False,
                "failed_stage": None,
                "fail_reason": "no_geometry_attempted",
            }

        def _box_side_x(self, arm_tag, gap=SIDE_WAYPOINT_GAPS[1]):
            """World x beside the occluder on the arm's OWN side (right -> +x, left -> -x),
            OCC_HALF_FOOTPRINT + SIDE_WAYPOINT_GAP from the box centre, clamped to the reach
            limit on that same side (never flipped across to the wrong side)."""
            if self.spawn_occluder and getattr(self, "occluder", None) is not None:
                box_x = float(self.occluder.get_pose().p[0])
            else:
                box_x = float(self.target_obj.get_pose().p[0])
            side = 1.0 if str(arm_tag) == "right" else -1.0
            x = box_x + side * (OCC_HALF_FOOTPRINT + gap)
            return side * REACH_X_LIMIT if abs(x) > REACH_X_LIMIT else x

        def _around_box_waypoint(self, arm_tag, ref_pose, gap=SIDE_WAYPOINT_GAPS[1], z_lift=0.0,
                                  orient="grasp_aligned", y_offset=0.0):
            """Grasp subgoal: a pose beside the occluder (via _box_side_x), near the box's
            horizontal (y) line. Position comes from ref_pose's height + the side-waypoint
            offset, with y = box's y-line + y_offset (searched around 0 -- pinning exactly
            to the box's y-line was found to be IK-unreachable at some seeds regardless of
            x/orientation). Orientation is either ref_pose's own (grasp_aligned -- matches
            the eventual reach-in angle) or a neutral top-down orientation (orient=
            "top_down"). None when no occluder is present."""
            if not (self.spawn_occluder and getattr(self, "occluder", None) is not None):
                return None
            wp = list(ref_pose)
            wp[0] = self._box_side_x(arm_tag, gap=gap)
            wp[1] = float(self.occluder.get_pose().p[1]) + float(y_offset)   # box's y-line +/- offset
            wp[2] += float(z_lift)
            if orient == "top_down":
                wp[3:] = GRASP_DIRECTION_DIC["top_down"]
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[around_box] arm={arm_tag} orient={orient} y_offset={y_offset:.2f} -> waypoint "
                      f"x={wp[0]:.3f} y={wp[1]:.3f} z={wp[2]:.3f}")
            return wp

        def _rank_side_grasp_ids(self, actor, arm_tag, limit=GRASP_CANDIDATE_LIMIT):
            """Rank a few side grasps so we can search across candidates instead of
            committing to exactly one local optimum."""
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
                if horiz > 0.35:                              # skip clearly tilted/vertical
                    continue
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

        def _time_stretch_trajectory(self, result, factor):
            """Densify a planned joint path while preserving its geometry."""
            factor = max(1, int(factor))
            positions = np.asarray(result.get("position"))
            if factor == 1 or positions.ndim < 2 or len(positions) < 2:
                return result

            stretched = deepcopy(result)
            sample = np.linspace(0.0, len(positions) - 1,
                                 (len(positions) - 1) * factor + 1)
            lo = np.floor(sample).astype(int)
            hi = np.minimum(lo + 1, len(positions) - 1)
            alpha = (sample - lo).reshape((-1,) + (1,) * (positions.ndim - 1))
            for key, derivative_order in (("position", 0), ("velocity", 1),
                                          ("acceleration", 2), ("jerk", 3)):
                values = result.get(key)
                if values is None:
                    continue
                values = np.asarray(values)
                if values.ndim == 0 or len(values) != len(positions):
                    continue
                blended = values[lo] * (1.0 - alpha) + values[hi] * alpha
                if derivative_order:
                    blended = blended / (factor ** derivative_order)
                stretched[key] = blended.astype(values.dtype, copy=False)
            return stretched

        def _replay_planned_move(self, arm_tag, cached_result):
            """Execute an ALREADY-PLANNED trajectory (position/velocity arrays from a
            prior plan_path call) directly via take_dense_action, skipping a fresh
            plan_path call entirely. Mirrors left_move_to_pose/right_move_to_pose's
            need_plan/joint_path bookkeeping but never re-plans.

            Why this exists: CuRobo's trajopt has no uniqueness guarantee for a given
            Cartesian target -- re-planning to the SAME pose independently (as
            left_move_to_pose/right_move_to_pose normally do) can converge to a
            genuinely different, equally-valid arm posture. Confirmed empirically
            (qpos drift up to 1.47 rad between a dry-run-verified plan and a fresh
            live re-plan to the identical waypoint pose), and that drift flips
            downstream grasp reachability in ~50% of cases -- explaining most of the
            grasp-stage failures seen despite the dry run having just verified a
            working grasp. Replaying the exact verified trajectory removes the
            divergence: the live qpos after this call is guaranteed to match what the
            dry run assumed, since it's the same trajectory, not a new one."""
            if not self.plan_success:
                return
            if cached_result is None or cached_result.get("status") != "Success":
                self.plan_success = False
                self._last_fail_reason = (cached_result or {}).get("fail_reason", "unknown")
                return
            if str(arm_tag) == "left":
                if self.need_plan:
                    self.left_joint_path.append(deepcopy(cached_result))
                else:
                    cached_result = deepcopy(self.left_joint_path[self.left_cnt])
                    self.left_cnt += 1
                replay_result = self._time_stretch_trajectory(
                    cached_result, ATTACHED_TRAJECTORY_SLOWDOWN
                    if self._slow_attached_replay else 1)
                control_seq = {"left_arm": replay_result, "left_gripper": None, "right_arm": None, "right_gripper": None}
            else:
                if self.need_plan:
                    self.right_joint_path.append(deepcopy(cached_result))
                else:
                    cached_result = deepcopy(self.right_joint_path[self.right_cnt])
                    self.right_cnt += 1
                replay_result = self._time_stretch_trajectory(
                    cached_result, ATTACHED_TRAJECTORY_SLOWDOWN
                    if self._slow_attached_replay else 1)
                control_seq = {"left_arm": None, "left_gripper": None, "right_arm": replay_result, "right_gripper": None}
            self.take_dense_action(control_seq)

        def _gripper_relative_object_transform(self, arm_tag):
            """The held object's FULL pose (position AND orientation) expressed in
            the gripper's LOCAL frame -- stable under gripper translation/rotation
            as long as the object is still rigidly held, unlike a raw world-frame
            offset (which changes with gripper orientation even with zero slip).
            Returns (relative_position, relative_quaternion)."""
            gripper_pose = np.asarray(self.get_arm_pose(arm_tag), dtype=float)
            gripper_rot = t3d.quaternions.quat2mat(gripper_pose[3:])
            object_pose = self.target_obj.get_pose()
            object_pos = np.asarray(object_pose.p, dtype=float)
            object_rot = t3d.quaternions.quat2mat(np.asarray(object_pose.q, dtype=float))
            relative_pos = gripper_rot.T @ (object_pos - gripper_pose[:3])
            relative_quat = t3d.quaternions.mat2quat(gripper_rot.T @ object_rot)
            return relative_pos, relative_quat

        def _object_near_placement_target(self, xy_tolerance=OBJECT_PLACEMENT_XY_TOLERANCE,
                                          z_tolerance=OBJECT_PLACEMENT_Z_TOLERANCE):
            """Has the held object already reached a pose that WOULD PASS
            put_mouse_on_pad.check_success() -- independent x/y error each under
            xy_tolerance (mirroring check_success's own per-axis 0.02 criterion,
            not a looser combined radius)? Used by place_actor's descent to stop
            and open the gripper immediately once this is genuinely true, rather
            than continuing to descend past a placement that's already good.
            Deliberately conservative: an earlier combined-radius version
            accepted seed 9 (dx=3.24cm) and seed 12 (dx=6.11cm) as "arrived" even
            though check_success() would still reject both on at least one
            axis -- see DESCENT_CONTACT_DRIFT_TOLERANCE for how a drift that
            ISN'T yet this precise, but also isn't a genuine loss, is handled."""
            if not getattr(self, "des_obj_pose", None):
                return False
            obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
            target = np.asarray(self.des_obj_pose, dtype=float)
            dx = abs(float(obj_pos[0] - target[0]))
            dy = abs(float(obj_pos[1] - target[1]))
            z_dist = float(obj_pos[2] - target[2])  # positive: object sits above the target resting height
            return dx <= xy_tolerance and dy <= xy_tolerance and -z_tolerance <= z_dist <= z_tolerance

        def _object_near_support_height(self, height_tolerance=DESCENT_CONTACT_HEIGHT_TOLERANCE):
            """Is the held object's world-frame z plausibly at the placement
            surface right now (independent of x/y)? Used to gate
            DESCENT_CONTACT_DRIFT_TOLERANCE's ambiguous-contact branch: a
            gripper-relative drift in that band is only explainable as early
            contact if the object is actually near the resting height it would
            be contacting -- a genuine drop shows the same drift magnitude
            while still high above the surface, since it's falling under
            gravity rather than following the descent's planned trajectory."""
            if not getattr(self, "des_obj_pose", None):
                return False
            obj_z = float(self.target_obj.get_pose().p[2])
            target_z = float(self.des_obj_pose[2])
            return abs(obj_z - target_z) <= height_tolerance

        def _placement_xy_error(self):
            """Signed target-minus-object XY error in world coordinates."""
            obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
            target = np.asarray(self.des_obj_pose, dtype=float)
            return target[:2] - obj_pos[:2]


        def _object_retained(self, arm_tag, tolerance=OBJECT_RETENTION_TOLERANCE,
                             rotation_tolerance=OBJECT_RETENTION_ROTATION_TOLERANCE, context=None,
                             return_drift=False):
            """True if the held object is still where it was (position AND
            orientation) relative to the gripper when _grasp_baseline_transform
            was captured (right after attach_object in play_once) -- i.e. the
            grasp hasn't slipped, rotated loose, or been dropped since. Confirmed
            empirically (seed 27): every placement stage reported
            plan_success=True while the object physically fell mid-transit, since
            plan_success only reflects the ARM's motion plan, never the HELD
            OBJECT's actual state. Returns True (nothing to check) if no baseline
            has been captured yet.

            Strict everywhere, including place_actor's descent -- callers there
            should check _object_near_placement_target FIRST and short-circuit
            before ever calling this, since a legitimate early placement also
            changes the gripper-relative transform and would otherwise look
            identical to a drop."""
            if self._grasp_baseline_transform is None:
                return (True, 0.0, 0.0) if return_drift else True
            baseline_pos, baseline_quat = self._grasp_baseline_transform
            current_pos, current_quat = self._gripper_relative_object_transform(arm_tag)
            pos_drift = float(np.linalg.norm(current_pos - baseline_pos))
            rot_dot = float(np.clip(abs(np.dot(baseline_quat, current_quat)), -1.0, 1.0))
            rot_drift = float(2.0 * np.arccos(rot_dot))
            retained = pos_drift <= tolerance and rot_drift <= rotation_tolerance
            self.rollout_retention_checks.append({
                "context": context, "pos_drift": pos_drift, "rot_drift": rot_drift, "retained": retained,
            })
            if os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1":
                print(f"[object-retained] context={context} pos_drift={pos_drift:.4f}m (tol={tolerance}) "
                      f"rot_drift={rot_drift:.4f}rad (tol={rotation_tolerance}) retained={retained}")
            if return_drift:
                return retained, pos_drift, rot_drift
            return retained

        def _trajectory_path_metrics(self, arm_tag, result):
            """Safety-filter metrics for a planned descent-slice trajectory,
            via the arm's planner's own forward kinematics. Used by
            _plan_pose_with_descent_slices to catch CuRobo trajopt solves
            that are technically valid but take an implausibly circuitous
            path for what should be a tiny straight-line slice -- see
            DESCENT_MAX_PATH_DEVIATION and friends for the empirical case
            that motivated this (a 3cm slice's solve swinging the gripper
            56cm away before snapping back). Returns None if the trajectory
            has fewer than 2 waypoints (nothing to check).

            - max_perp_deviation: max distance any waypoint's EE position
              strays from the closest point on the FINITE segment between
              the trajectory's own start/end EE pose (projection clamped to
              [0, line_len] -- an unclamped projection would report
              near-zero deviation for a trajectory that overshoots straight
              past the target along the same axis).
            - path_length_ratio: actual Cartesian path length (sum of
              consecutive EE waypoint distances) over the straight-line
              distance -- catches back-and-forth paths that stay close to
              the line but travel much farther than the direct distance.
            - joint_path_length / joint_direct_dist: absolute joint-space
              path length and direct start->end joint-space distance
              (radians, L2 norm) -- exposed directly (not just as a ratio)
              so a monotonic jump to a distant IK branch can be caught by
              its own absolute magnitude, since such a jump has
              joint_travel_ratio near 1.0 regardless of how far it goes.
            - joint_travel_ratio: joint_path_length / joint_direct_dist --
              catches joint-space WINDING an EE-only check can miss (a
              redundant/near-singular arm can reach nearly the same EE pose
              via very different, looping joint paths). A near-zero
              joint_direct_dist (or line_len for path_length_ratio) with a
              non-negligible numerator returns inf rather than silently
              collapsing to a "perfect" 1.0 -- a real loop with almost
              identical start/end shouldn't look the same as "nothing
              moved."
            - max_joint_range: the largest single joint's (max-min)
              excursion across the trajectory, in radians -- flags one
              joint spinning through an implausible range even when the
              aggregate joint_travel_ratio looks acceptable."""
            positions = result.get("position")
            if positions is None or len(positions) < 2:
                return None
            planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
            joint_state = JointState.from_position(
                torch.tensor(positions, dtype=torch.float32).cuda(),
                joint_names=planner.active_joints_name,
            )
            kin_state = planner.motion_gen.compute_kinematics(joint_state)
            ee_pos = np.array(kin_state.ee_pos_seq.to("cpu"), dtype=float)
            start_pos, end_pos = ee_pos[0], ee_pos[-1]
            line_vec = end_pos - start_pos
            line_len = float(np.linalg.norm(line_vec))
            if line_len < 1e-6:
                max_perp_deviation = float(np.max(np.linalg.norm(ee_pos - start_pos, axis=1)))
            else:
                line_unit = line_vec / line_len
                rel = ee_pos - start_pos
                t = np.clip(rel @ line_unit, 0.0, line_len)
                closest = start_pos + np.outer(t, line_unit)
                max_perp_deviation = float(np.max(np.linalg.norm(ee_pos - closest, axis=1)))
            path_length = float(np.sum(np.linalg.norm(np.diff(ee_pos, axis=0), axis=1)))
            if line_len > 1e-6:
                path_length_ratio = path_length / line_len
            elif path_length > 1e-6:
                path_length_ratio = float("inf")
            else:
                path_length_ratio = 1.0

            joint_pos = np.asarray(positions, dtype=float)
            joint_path_length = float(np.sum(np.linalg.norm(np.diff(joint_pos, axis=0), axis=1)))
            joint_direct_dist = float(np.linalg.norm(joint_pos[-1] - joint_pos[0]))
            if joint_direct_dist > 1e-6:
                joint_travel_ratio = joint_path_length / joint_direct_dist
            elif joint_path_length > 1e-6:
                joint_travel_ratio = float("inf")
            else:
                joint_travel_ratio = 1.0
            max_joint_range = float(np.max(joint_pos.max(axis=0) - joint_pos.min(axis=0)))

            return {
                "max_perp_deviation": max_perp_deviation,
                "path_length": path_length,
                "path_length_ratio": path_length_ratio,
                "joint_path_length": joint_path_length,
                "joint_direct_dist": joint_direct_dist,
                "joint_travel_ratio": joint_travel_ratio,
                "max_joint_range": max_joint_range,
            }

        def _descent_tstep_fraction_for_attempt(self, attempt_idx, total_attempts,
                                               fractions=DESCENT_APPROACH_TSTEP_FRACTIONS):
            """Progressive relaxation schedule across a descent slice's retry
            budget: front-load attempts on the tightest (least detour room)
            fraction, then relax toward CuRobo's own default across the
            remaining attempts -- see DESCENT_APPROACH_TSTEP_FRACTIONS for why
            a single fixed value (either tight or loose) isn't enough on its
            own."""
            tier = min(len(fractions) - 1, attempt_idx * len(fractions) // max(1, total_attempts))
            return fractions[tier]

        def _plan_pose_with_descent_slices(self, arm_tag, target_pose, qpos, stage_label, start_pose,
                                          retries=1, constraint_pose=None, approach_axis=None,
                                          slice_size=DESCENT_SLICE_SIZE, near_contact=False):
            """Deterministic incremental descent, used for place_actor's moves
            instead of the generic scalar-error-informed shrink retry
            (_plan_pose_with_shrinking_waypoint). place_actor's moves are short,
            mostly-straight-line final approaches with KNOWN structure (fixed
            target orientation, held via approach_axis) -- exploiting that
            structure with fixed-size slices along the straight line to the
            target is more appropriate than an adaptive shrink that doesn't know
            it. Plans AND REPLAYS each slice immediately (not a virtual dry-run
            chain), checking _object_retained after every segment -- catches a
            mid-descent drop at the EXACT slice it happens (confirmed
            empirically: seed 27 dropped the object mid-transit while every
            stage still reported plan_success=True) instead of only at the end,
            and naturally preserves whatever slices already succeeded if a
            later one fails, without needing a separate replay-before-return
            step (each slice is already committed by the time the next runs).

            Returns (ok, fail_reason, qpos, placed). placed=True means the
            object has already reached a pose that would pass check_success --
            the caller (_execute_actions_via_plan_and_replay) must skip any
            remaining 'move' actions and go straight to opening the gripper,
            since continuing to descend past an already-good placement could
            only make it worse."""
            plan_func = self._arm_plan_func(arm_tag)
            qpos0 = np.array(qpos, dtype=np.float64, copy=True)
            start = np.asarray(start_pose, dtype=float)
            target = np.asarray(target_pose, dtype=float)
            total_dist = float(np.linalg.norm(target[:3] - start[:3]))
            num_slices = max(1, int(np.ceil(total_dist / slice_size)))
            segment_length = total_dist / num_slices
            # create_grasp_approach_metric's offset_position must stay smaller
            # than the segment it's applied to -- CuRobo's own default (0.05) is
            # LARGER than a 4cm slice, asking it to set up a straight-line
            # pre-approach point farther away than the entire move itself.
            # Confirmed to matter: 6 episodes still failed FINETUNE_TRAJOPT_FAIL
            # with the unscaled 5cm offset on every slice.
            approach_offset = min(0.05, segment_length * 0.8) if approach_axis is not None else 0.05
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            last_reason = "unknown"
            for i in range(1, num_slices + 1):
                # Interpolate BOTH position and orientation (_blend_pose does
                # position lerp + quaternion nlerp) -- copying the target's
                # quaternion onto every slice made the FIRST slice perform the
                # entire orientation change within just one small position step.
                waypoint = np.asarray(self._blend_pose(start, target, i / num_slices), dtype=float)
                total_attempts = max(1, int(retries))
                # Collect every attempt that plans successfully AND passes the
                # path-safety filter, across the whole progressive-relaxation
                # schedule, then replay the SHORTEST accepted one -- not just
                # the first success. Trying every attempt regardless of an
                # early pass (rather than stopping at the first accepted
                # candidate) is what lets a tight-fraction attempt lose to a
                # looser one that happens to find a genuinely shorter path,
                # and is bounded by retries (already small, e.g. 10).
                accepted = []  # list of (path_length, joint_path_length, max_joint_range, result, attempt_idx, attempt_record)
                for attempt in range(total_attempts):
                    fraction = self._descent_tstep_fraction_for_attempt(attempt, total_attempts)
                    candidate = plan_func(list(waypoint), last_qpos=qpos0,
                                         constraint_pose=constraint_pose, approach_axis=approach_axis,
                                         approach_offset=approach_offset,
                                         tstep_fraction=fraction, near_contact=near_contact)
                    candidate["tstep_fraction"] = fraction
                    attempt_record = {
                        "stage": stage_label, "slice": i, "num_slices": num_slices,
                        "attempt": attempt, "tstep_fraction": fraction,
                        "near_contact": near_contact,
                        "status": candidate.get("status"),
                        "position_error": candidate.get("position_error"),
                        "rotation_error": candidate.get("rotation_error"),
                    }
                    if candidate.get("status") != "Success":
                        last_reason = candidate.get("fail_reason", "unknown")
                        attempt_record["fail_reason"] = last_reason
                        self.rollout_descent_slices.append(attempt_record)
                        continue
                    metrics = self._trajectory_path_metrics(arm_tag, candidate)
                    attempt_record["path_metrics"] = metrics
                    if metrics is None:
                        # Nothing to measure (degenerate <2-waypoint plan) --
                        # trust it, there's no path to have gone wrong.
                        accepted.append((0.0, 0.0, 0.0, candidate, attempt, attempt_record))
                        attempt_record["accepted"] = True
                    elif (metrics["max_perp_deviation"] <= DESCENT_MAX_PATH_DEVIATION
                            and metrics["path_length_ratio"] <= DESCENT_MAX_PATH_LENGTH_RATIO
                            and metrics["joint_travel_ratio"] <= DESCENT_MAX_JOINT_TRAVEL_RATIO
                            and metrics["max_joint_range"] <= DESCENT_MAX_JOINT_RANGE
                            and metrics["joint_direct_dist"] <= DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT
                            and metrics["joint_path_length"] <= DESCENT_MAX_JOINT_PATH_LENGTH):
                        accepted.append((metrics["path_length"], metrics["joint_path_length"],
                                        metrics["max_joint_range"], candidate, attempt, attempt_record))
                        attempt_record["accepted"] = True
                    else:
                        last_reason = (
                            f"path_filter_rejected(dev={metrics['max_perp_deviation']:.3f},"
                            f"len_ratio={metrics['path_length_ratio']:.2f},"
                            f"joint_ratio={metrics['joint_travel_ratio']:.2f},"
                            f"max_joint_range={metrics['max_joint_range']:.3f},"
                            f"joint_dist={metrics['joint_direct_dist']:.3f},"
                            f"joint_path={metrics['joint_path_length']:.3f})")
                        attempt_record["accepted"] = False
                        attempt_record["reject_reason"] = last_reason
                        if log_move:
                            print(f"[descent-slice] {stage_label} slice {i}/{num_slices} attempt "
                                  f"{attempt} (tstep_fraction={fraction}): rejected -- {last_reason}")
                    self.rollout_descent_slices.append(attempt_record)
                if not accepted:
                    if log_move:
                        print(f"[descent-slice] {stage_label} slice {i}/{num_slices} failed "
                              f"({total_attempts} attempts, none accepted; last={last_reason})")
                    return False, last_reason, qpos0, False
                # Lexicographic: shortest Cartesian path first, then shortest
                # joint-space path, then smallest single-joint excursion --
                # two candidates can have near-identical Cartesian paths but
                # very different joint motion, so Cartesian length alone
                # isn't enough to prefer the more sensible one.
                accepted.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
                best_path_length, best_joint_path_length, best_max_joint_range, result, best_attempt, best_record = accepted[0]
                best_record["selected"] = True
                if log_move:
                    print(f"[descent-slice] {stage_label} slice {i}/{num_slices} succeeded "
                          f"(attempt {best_attempt}, tstep_fraction={result['tstep_fraction']}, "
                          f"path_length={best_path_length:.3f}m, "
                          f"joint_path_length={best_joint_path_length:.3f}rad, "
                          f"{len(accepted)}/{total_attempts} attempts accepted)")
                qpos0 = self._roll_qpos_forward(arm_tag, qpos0, result)
                self._replay_planned_move(arm_tag, result)
                if not self.plan_success:
                    return False, self._last_fail_reason, qpos0, False
                # Check arrival BEFORE retention, not as a fallback after a
                # failed retention check: the object settling onto the target
                # surface early (before the gripper's own nominal endpoint)
                # changes the gripper-relative transform exactly like a drop
                # would, but it's the intended outcome -- stop descending
                # further (which could otherwise ram the held object into the
                # surface it already reached) and report this move done, via
                # placed=True (the caller must skip any remaining moves).
                if self._object_near_placement_target():
                    if log_move:
                        print(f"[descent-slice] {stage_label} slice {i}/{num_slices}: object "
                              f"already at a placement that would pass check_success -- done")
                    return True, None, qpos0, True
                retained, pos_drift, rot_drift = self._object_retained(
                    arm_tag, context=f"{stage_label}:slice{i}", return_drift=True)
                if not retained:
                    xy_error = self._placement_xy_error()
                    supported_contact = (
                        pos_drift <= DESCENT_CONTACT_DRIFT_TOLERANCE
                        and self._object_near_support_height())
                    if (supported_contact
                            and np.all(np.abs(xy_error) < CONTACT_RELEASE_XY_TOLERANCE)):
                        self.rollout_descent_slices.append({
                            "stage": f"{stage_label}:contact-release",
                            "dx": float(xy_error[0]), "dy": float(xy_error[1]),
                            "correction_needed": False,
                        })
                        return True, None, qpos0, True
                    # Ambiguous-contact requires ALL of: bounded position
                    # drift, bounded ROTATION drift (a rotation-only loss --
                    # object tipped/rotated loose without much translating --
                    # would otherwise sail through on position alone, quietly
                    # defeating the retention check's own rotation term), and
                    # the object plausibly near the support surface it would
                    # be contacting (a genuine drop shows the same 4-15cm
                    # position drift while still high above the surface,
                    # since it's falling under gravity rather than following
                    # the descent's planned trajectory).
                    if (pos_drift <= DESCENT_CONTACT_DRIFT_TOLERANCE
                            and rot_drift <= OBJECT_RETENTION_ROTATION_TOLERANCE
                            and self._object_near_support_height()):
                        # Ambiguous: not yet at a passing placement (the check
                        # above already ruled that out) and not within strict
                        # retention tolerance either, but the drift magnitude is
                        # well below anything a genuine drop has shown (30cm-
                        # 1.4m) and consistent with early, imprecise contact.
                        # Don't fail -- let the remaining slices keep trying to
                        # converge on the real target instead of giving up on
                        # a placement that's already known to be too far off.
                        if log_move:
                            print(f"[descent-slice] {stage_label} slice {i}/{num_slices}: drift "
                                  f"{pos_drift:.3f}m looks like early contact, not a loss -- continuing")
                    else:
                        self.plan_success = False
                        self._last_fail_reason = f"object_lost:{stage_label}:slice{i}"
                        return False, self._last_fail_reason, qpos0, False
                elif log_move:
                    print(f"[descent-slice] {stage_label} slice {i}/{num_slices} succeeded")
            return True, None, qpos0, False

        def _local_landing_search_and_place(self, arm_tag):
            """Replaces place_actor's forced descent to one exact final pose
            (move1/move2 via _execute_actions_via_plan_and_replay), which kept
            failing MotionGenStatus.FINETUNE_TRAJOPT_FAIL right at the surface
            -- the attached object's collision margin against the table is
            tightest exactly there, and CuRobo's trajopt struggles to converge
            on ANY smooth path into it, let alone the one exact pose asked for.

            Instead searches a small grid of nearby landing poses -- XY
            offsets within check_success's own tolerance, release heights
            from low to high -- and requires only ONE of them to actually
            plan and pass the same trajectory-safety filters used by
            _plan_pose_with_descent_slices, then replays ONLY that one and
            opens the gripper, letting physics settle the final contact
            instead of commanding CuRobo all the way down to it. Explicitly
            gated on the object being retained and within
            LANDING_SEARCH_TRIGGER_DISTANCE (XY) / LANDING_SEARCH_MAX_Z_
            DISTANCE (vertical) of the target -- this function's
            candidate moves are short by design, so calling it from farther
            away would apply the same short-path filters to a legitimately
            longer approach and reject it.

            Heights are grouped into two tiers ([1cm,3cm] then [5cm,8cm]):
            ALL accepted candidates across an entire tier are compared
            before picking a winner, not just the first height with any
            success -- otherwise a marginal 1cm solve could beat a cleanly
            planned 3cm one that was never even tried, since candidates
            aren't compared across different heights independently.

            Candidate object poses are converted to gripper poses via
            get_place_pose, NOT a fixed offset above the pad -- get_place_pose
            derives the required gripper pose from the ACTUAL current grasp
            transform (actor pose vs. end-effector pose), so it's correct
            regardless of which grasp this particular episode happened to
            use, unlike a fixed "gripper N cm above the pad" assumption which
            would put the object at the wrong height for a different grasp.

            Sets plan_success=False with a descriptive fail_reason if no
            candidate at any height passes; the caller's checkpoint() records
            it as an ordinary failure, same as any other stage."""
            plan_func = self._arm_plan_func(arm_tag)
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            check_ik = self.robot.left_check_ik_batch if str(arm_tag) == "left" else self.robot.right_check_ik_batch
            target = np.asarray(self.des_obj_pose, dtype=float)

            obj_pos = np.asarray(self.target_obj.get_pose().p, dtype=float)
            xy_distance = float(np.hypot(obj_pos[0] - target[0], obj_pos[1] - target[1]))
            z_distance = float(abs(obj_pos[2] - target[2]))
            retained = self._object_retained(arm_tag, context="landing-search:precondition")
            if not retained or xy_distance > LANDING_SEARCH_TRIGGER_DISTANCE or z_distance > LANDING_SEARCH_MAX_Z_DISTANCE:
                self.plan_success = False
                self._last_fail_reason = (f"landing_search_preconditions_not_met(retained={retained},"
                                          f"xy_distance={xy_distance:.3f}m,z_distance={z_distance:.3f}m)")
                if log_move:
                    print(f"[landing-search] preconditions not met -- retained={retained} "
                          f"xy_distance={xy_distance:.3f}m (max {LANDING_SEARCH_TRIGGER_DISTANCE}m) "
                          f"z_distance={z_distance:.3f}m (max {LANDING_SEARCH_MAX_Z_DISTANCE}m)")
                return

            # Skip direct 1--5 cm grids: the sweep produced 509 solver failures.
            # Stage at 8 cm, then descend sequentially with the near-contact profile.
            height_tiers = [[max(LANDING_RELEASE_HEIGHTS)]]
            for tier in height_tiers:
                tier_accepted = []  # (path_length, joint_path_length, max_joint_range, result, height, record)
                for height in tier:
                    candidates = []  # (dx, dy, obj_pose)
                    for dx in LANDING_XY_OFFSETS:
                        for dy in LANDING_XY_OFFSETS:
                            obj_pose = target.copy()
                            obj_pose[0] += dx
                            obj_pose[1] += dy
                            obj_pose[2] = target[2] + height
                            candidates.append((dx, dy, obj_pose))
                    # Sort by radius (closest to nominal target first), then
                    # by angle -- groups by distance WITHOUT letting ties at
                    # the same radius systematically favor one sign over
                    # another (plain grid iteration order puts every
                    # negative-dx candidate before any positive-dx one at
                    # the same radius, since LANDING_XY_OFFSETS lists
                    # negatives first and Python's sort is stable).
                    candidates.sort(key=lambda c: (np.hypot(c[0], c[1]), np.arctan2(c[1], c[0])))

                    gripper_poses = [
                        self.get_place_pose(self.target_obj, arm_tag, obj_pose, constrain="free", pre_dis=0.0)
                        for _, _, obj_pose in candidates
                    ]
                    ik_ok = list(check_ik(gripper_poses))
                    current_ee_pose = np.asarray(self.get_arm_pose(arm_tag), dtype=float)
                    qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
                    qpos0 = np.array(qpos, dtype=np.float64, copy=True)

                    planned = 0
                    height_accepted = 0
                    for (dx, dy, _), gripper_pose, feasible in zip(candidates, gripper_poses, ik_ok):
                        record = {"stage": "landing-search", "height": height, "dx": dx, "dy": dy,
                                  "ik_feasible": bool(feasible)}
                        if not feasible:
                            self.rollout_descent_slices.append(record)
                            continue
                        if planned >= LANDING_MAX_CANDIDATES_PER_HEIGHT:
                            continue  # never attempted -- not a rejection, don't record
                        planned += 1
                        segment_length = float(np.linalg.norm(np.asarray(gripper_pose[:3]) - current_ee_pose[:3]))
                        approach_offset = min(0.05, segment_length * 0.8) if segment_length > 1e-6 else 0.02
                        if height == max(LANDING_RELEASE_HEIGHTS):
                            # This collision-clear target is a staging transit,
                            # not a descent. The approach metric caused the
                            # repeatable 2--2.5x loops seen in landing_v2.
                            candidate_result = plan_func(list(gripper_pose), last_qpos=qpos0)
                        else:
                            candidate_result = plan_func(
                                list(gripper_pose), last_qpos=qpos0, approach_axis=2,
                                approach_offset=approach_offset, tstep_fraction=0.4)
                        record["status"] = candidate_result.get("status")
                        if candidate_result.get("status") != "Success":
                            record["fail_reason"] = candidate_result.get("fail_reason", "unknown")
                            record["accepted"] = False
                            self.rollout_descent_slices.append(record)
                            continue
                        metrics = self._trajectory_path_metrics(arm_tag, candidate_result)
                        record["path_metrics"] = metrics
                        # DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT/PATH_LENGTH
                        # were calibrated for descent slices (~DESCENT_SLICE_
                        # SIZE=4cm each) -- a landing-search candidate can be
                        # up to ~10cm away, which naturally needs more joint
                        # travel even for a perfectly clean plan. Reusing the
                        # unscaled absolute caps rejected almost everything
                        # (confirmed empirically: 1/288 candidates accepted
                        # across a 6-seed sanity run). Scale by how much
                        # longer this candidate's segment is than a nominal
                        # slice, same way approach_offset already scales.
                        joint_scale = max(1.0, segment_length / DESCENT_SLICE_SIZE)
                        max_joint_endpoint_displacement = DESCENT_MAX_JOINT_ENDPOINT_DISPLACEMENT * joint_scale
                        max_joint_path_length = DESCENT_MAX_JOINT_PATH_LENGTH * joint_scale
                        record["joint_scale"] = joint_scale
                        if metrics is None:
                            tier_accepted.append((0.0, 0.0, 0.0, candidate_result, height, dx, dy, record))
                            record["accepted"] = True
                            height_accepted += 1
                        elif (metrics["max_perp_deviation"] <= DESCENT_MAX_PATH_DEVIATION
                                and metrics["path_length_ratio"] <= DESCENT_MAX_PATH_LENGTH_RATIO
                                and metrics["joint_travel_ratio"] <= DESCENT_MAX_JOINT_TRAVEL_RATIO
                                and metrics["max_joint_range"] <= DESCENT_MAX_JOINT_RANGE
                                and metrics["joint_direct_dist"] <= max_joint_endpoint_displacement
                                and metrics["joint_path_length"] <= max_joint_path_length):
                            tier_accepted.append((metrics["path_length"], metrics["joint_path_length"],
                                                  metrics["max_joint_range"], candidate_result, height, dx, dy, record))
                            record["accepted"] = True
                            height_accepted += 1
                        else:
                            record["accepted"] = False
                            record["reject_reason"] = (
                                f"path_filter_rejected(dev={metrics['max_perp_deviation']:.3f},"
                                f"len_ratio={metrics['path_length_ratio']:.2f},"
                                f"joint_ratio={metrics['joint_travel_ratio']:.2f},"
                                f"max_joint_range={metrics['max_joint_range']:.3f},"
                                f"joint_dist={metrics['joint_direct_dist']:.3f}(max={max_joint_endpoint_displacement:.3f}),"
                                f"joint_path={metrics['joint_path_length']:.3f}(max={max_joint_path_length:.3f}))")
                        self.rollout_descent_slices.append(record)
                        if height_accepted >= LANDING_MIN_ACCEPTED_TO_STOP:
                            break
                    if log_move:
                        print(f"[landing-search] height={height:.3f}m: {height_accepted} accepted "
                              f"({sum(ik_ok)}/{len(candidates)} IK-feasible, {planned} planned)")

                if not tier_accepted:
                    continue

                tier_accepted.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
                best_path_length, best_joint_path_length, _, result, best_height, best_dx, best_dy, best_record = tier_accepted[0]
                best_record["selected"] = True
                if log_move:
                    print(f"[landing-search] tier {tier} succeeded: {len(tier_accepted)} candidates "
                          f"accepted across the tier, chose height={best_height:.3f}m "
                          f"path_length={best_path_length:.3f}m joint_path_length={best_joint_path_length:.3f}rad")
                self._replay_planned_move(arm_tag, result)
                if not self.plan_success:
                    return
                if not self._object_retained(arm_tag, context="landing-search"):
                    self.plan_success = False
                    self._last_fail_reason = "object_lost:landing-search"
                    return

                release_height = best_height
                descent_failure = None
                if best_height == max(LANDING_RELEASE_HEIGHTS):
                    # The collision-clear solve is a staging pose. From this
                    # verified state, make only short constrained downward moves.
                    for lower_height in sorted(
                            (h for h in LANDING_RELEASE_HEIGHTS if h < best_height), reverse=True):
                        lower_obj_pose = target.copy()
                        lower_obj_pose[:3] += np.array([best_dx, best_dy, lower_height])
                        lower_gripper_pose = self.get_place_pose(
                            self.target_obj, arm_tag, lower_obj_pose,
                            constrain="free", pre_dis=0.0)
                        live_qpos = (self.robot.left_entity.get_qpos() if str(arm_tag) == "left"
                                     else self.robot.right_entity.get_qpos())
                        ok, reason, _, placed = self._plan_pose_with_descent_slices(
                            arm_tag, lower_gripper_pose,
                            np.asarray(live_qpos, dtype=np.float64),
                            f"landing-descent:{lower_height:.3f}",
                            start_pose=self.get_arm_pose(arm_tag),
                            retries=PLACEMENT_SEARCH_RETRIES, approach_axis=2,
                            near_contact=True)
                        if not ok:
                            descent_failure = reason
                            if not self.plan_success:
                                return
                            break
                        release_height = lower_height
                        if placed:
                            break

                self.rollout_descent_slices.append({
                    "stage": "landing-release", "height": release_height,
                    "dx": best_dx, "dy": best_dy,
                    "descent_failure": descent_failure,
                })
                self.move(self.open_gripper(arm_tag))
                return

            self.plan_success = False
            self._last_fail_reason = "no_valid_landing_candidate"
            if log_move:
                print(f"[landing-search] failed: no candidate at any height "
                      f"{LANDING_RELEASE_HEIGHTS} planned and passed the filter")

        def _execute_actions_via_plan_and_replay(self, arm_tag, actions, retries=1):
            """Execute each 'move' Action via deterministic descent slicing (see
            _plan_pose_with_descent_slices), which plans AND replays incrementally
            and checks grasp retention after every slice; non-move actions
            (gripper open/close) execute normally through self.move.

            retries: place_actor's failures are 100% MotionGenStatus.FINETUNE_TRAJOPT_FAIL
            (confirmed empirically) -- a marginal-feasibility trajopt-difficulty problem,
            not a reachability one. We've directly observed the SAME setup succeed on
            one run and fail on another (CuRobo's trajopt has internal randomized
            seeding), so retrying a failed plan_func call with fresh internal seeding
            has a real chance of finding a solution the first attempt missed. Each
            retry re-attempts ONLY the failed slice, not the whole chain."""
            if not self.plan_success:
                return
            start_pose = list(self.get_arm_pose(arm_tag))
            move_idx = 0
            already_placed = False
            for a in actions:
                if a.action != "move":
                    self.move((arm_tag, [a]))
                    if not self.plan_success:
                        return
                    continue
                if already_placed:
                    # A previous move already reached a placement that would
                    # pass check_success (placed=True) -- skip any remaining
                    # descent moves (continuing could only push an already-good
                    # placement out of tolerance or into the surface) but still
                    # fall through to whatever non-move actions (gripper open)
                    # come after this one.
                    continue
                move_idx += 1
                qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
                ok, fail_reason, _, placed = self._plan_pose_with_descent_slices(
                    arm_tag, a.target_pose, np.array(qpos, dtype=np.float64, copy=True),
                    f"place_actor:move{move_idx}", start_pose=start_pose, retries=retries,
                    constraint_pose=a.args.get("constraint_pose"),
                    approach_axis=a.args.get("approach_axis"))
                if not ok:
                    self.plan_success = False
                    self._last_fail_reason = fail_reason
                    return
                start_pose = list(self.get_arm_pose(arm_tag))
                already_placed = placed

        def _plan_and_replay_pose(self, arm_tag, pose, seed_traj=None, stage_label="pose"):
            """Plan a single pose from the CURRENT live qpos and immediately replay
            it via _replay_planned_move, instead of self.move(self.move_to_pose(...)).
            For a lone plan-then-immediately-execute call like this there's no
            separate earlier verification pass to diverge from (unlike waypoint/
            grasp/place_actor), so this isn't fixing a re-plan-divergence bug here --
            its value is exposing the specific CuRobo fail_reason via
            self._last_fail_reason for the structured failure recorder, and using the
            same execution primitive as the rest of this file for consistency.

            seed_traj: optional clearance-route trajopt seed (Phase C). Warm-starts only the
            DIRECT attempt inside the retry helper; the shrinking-waypoint fallback stays
            unseeded, matching how the approach leg is seeded. None => unseeded, i.e. identical
            to what this did before, which is what makes seeded-vs-unseeded a one-variable
            contrast at this call site.
            stage_label: what the plan-effort recorder files this attempt under. Defaults to the
            historical "pose" so existing call sites keep their labels."""
            if not self.plan_success:
                return
            start_qpos = self.robot.left_entity.get_qpos() if str(arm_tag) == "left" else self.robot.right_entity.get_qpos()
            ok, fail_reason, _, trajectories = self._plan_pose_with_local_waypoint_retry(
                arm_tag, pose, np.array(start_qpos, dtype=np.float64, copy=True), stage_label,
                seed_traj=seed_traj)
            if not ok:
                self.plan_success = False
                self._last_fail_reason = fail_reason
                return
            self._replay_planned_sequence(arm_tag, trajectories)

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

        def _arm_plan_func(self, arm_tag):
            return self.robot.left_plan_path if str(arm_tag) == "left" else self.robot.right_plan_path

        def _arm_active_joint_indices(self, arm_tag):
            planner = self.robot.left_planner if str(arm_tag) == "left" else self.robot.right_planner
            return [planner.all_joints.index(name) for name in planner.active_joints_name if name in planner.all_joints]

        def _roll_qpos_forward(self, arm_tag, qpos, plan_result):
            qpos_next = np.array(qpos, dtype=np.float64, copy=True)
            active_indices = self._arm_active_joint_indices(arm_tag)
            qpos_next[active_indices] = np.asarray(plan_result["position"][-1], dtype=np.float64)
            return qpos_next

        def _local_waypoint_candidates(self, arm_tag, start_pose, target_pose, attempts=LOCAL_WAYPOINT_ATTEMPTS):
            """Bridge candidates near the current/start pose for any Cartesian move."""
            try:
                attempts = max(0, int(attempts))
            except (TypeError, ValueError):
                attempts = LOCAL_WAYPOINT_ATTEMPTS
            if attempts == 0:
                return []
            start = np.asarray(start_pose, dtype=float)
            target = np.asarray(target_pose, dtype=float)
            side = 1.0 if str(arm_tag) == "right" else -1.0
            specs = [
                ("offset", 0.00, 0.00, 0.04, 0.00),
                ("offset", 0.04, 0.00, 0.04, 0.00),
                ("offset", 0.08, 0.00, 0.06, 0.00),
                ("offset", 0.00, -0.04, 0.06, 0.00),
                ("offset", 0.04, -0.04, 0.08, 0.00),
                ("offset", 0.08, -0.04, 0.08, 0.00),
                ("offset", 0.00, 0.04, 0.06, 0.00),
                ("blend", 0.00, 0.00, 0.04, 0.35),
                ("blend", 0.04, 0.00, 0.06, 0.35),
                ("blend", 0.00, -0.04, 0.08, 0.35),
                ("offset", 0.10, 0.00, 0.10, 0.00),
                ("offset", 0.10, -0.06, 0.10, 0.00),
                # Added when the bridge search gained an IK prefilter (see
                # _plan_pose_with_local_waypoint_retry): a bigger pure-vertical lift
                # (no lateral offset -- the direction that resolved the escape/
                # pre_beside_box INVALID_START_STATE_WORLD_COLLISION failures) and a
                # later blend fraction, both cheap to add now that infeasible
                # candidates are filtered before spending a trajopt call on them.
                ("offset", 0.00, 0.00, 0.10, 0.00),
                ("blend", 0.00, 0.00, 0.00, 0.50),
            ]
            candidates = []
            for mode, dx_side, dy, dz, blend_t in specs[:attempts]:
                pose = self._blend_pose(start, target, blend_t) if mode == "blend" else list(start)
                pose[0] = float(np.clip(pose[0] + side * dx_side, -REACH_X_LIMIT, REACH_X_LIMIT))
                pose[1] = float(np.clip(pose[1] + dy, -0.33, 0.30))
                pose[2] = float(np.clip(pose[2] + dz, 0.78, 1.30))
                candidates.append(pose)
            return candidates

        def _plan_pose_with_shrinking_waypoint(self, arm_tag, target_pose, qpos, stage_label,
                                              start_pose, constraint_pose=None, approach_axis=None,
                                              relax_orientation=False,
                                              max_iterations=LOCAL_WAYPOINT_ATTEMPTS,
                                              min_distance=WAYPOINT_SHRINK_MIN_DISTANCE):
            """Adaptive waypoint-shrink retry, used when a direct plan to target_pose
            has already failed.

            CuRobo's MotionGenResult sets position_error/rotation_error UNCONDITIONALLY
            even on FINETUNE_TRAJOPT_FAIL (confirmed via source read of motion_gen.py's
            _plan_from_solve_state) -- a failed attempt isn't pure noise, it tells us
            how far off the optimizer's best attempt landed. Instead of only trying
            hand-picked fixed offset directions (the old _local_waypoint_candidates
            bridge pool), each iteration here either (a) retries the ORIGINAL target
            from wherever the previous iteration actually landed, or (b) if that
            fails, shrinks the target toward the current pose and retries that
            smaller hop instead. The shrink amount is informed by position_error when
            CuRobo reports one: back the target off by roughly the residual gap
            (plus a safety margin) instead of blindly halving every time, so a
            near-miss shrinks only a little (another attempt might close a small
            gap) while a wild miss shrinks a lot. Falls back to a plain 50% halving
            when no usable position_error is available (e.g. IK_FAIL, where
            trajopt never ran). Shrinking toward the current pose (by definition
            already reachable) converges toward something plannable even for
            targets that are hard/far, unlike fixed offsets which can themselves be
            unreachable. Stops on success at the real target, on max_iterations, or
            once the remaining hop is already < min_distance from the current pose
            (not worth shrinking further). Note: this does NOT replay CuRobo's own
            failed near-miss trajectory (its full path was never verified
            collision-safe, only its final error magnitude is known) -- it only
            uses the error MAGNITUDE to size a fresh, independently-planned hop."""
            log_move = os.environ.get("ROBOTWIN_LOG_MOVE", "") == "1"
            plan_func = self._arm_plan_func(arm_tag)
            qpos0 = np.array(qpos, dtype=np.float64, copy=True)
            current_pose = np.asarray(start_pose, dtype=float)
            original_target = np.asarray(target_pose, dtype=float)
            goal = original_target.copy()
            trajectories = []
            last_reason = "unknown"
            for iteration in range(1, max(1, int(max_iterations)) + 1):
                result = plan_func(list(goal), last_qpos=qpos0, constraint_pose=constraint_pose,
                                   approach_axis=approach_axis, relax_orientation=relax_orientation)
                reached_original = bool(np.allclose(goal[:3], original_target[:3], atol=1e-6))
                if result.get("status") == "Success":
                    qpos0 = self._roll_qpos_forward(arm_tag, qpos0, result)
                    label = stage_label if reached_original else f"{stage_label}:shrink{iteration}"
                    trajectories.append((label, result))
                    if reached_original:
                        return True, None, qpos0, trajectories
                    if log_move:
                        print(f"[waypoint-shrink] {stage_label} iter {iteration}: reached shrunk "
                              f"intermediate, retrying real target from there")
                    current_pose = goal.copy()
                    goal = original_target.copy()
                    continue
                last_reason = result.get("fail_reason", "unknown")
                remaining_dist = float(np.linalg.norm(goal[:3] - current_pose[:3]))
                if remaining_dist < min_distance:
                    if log_move:
                        print(f"[waypoint-shrink] {stage_label} iter {iteration} failed ({last_reason}); "
                              f"remaining distance {remaining_dist:.3f}m < {min_distance}m, giving up")
                    break
                position_error = result.get("position_error")
                if position_error is not None and 0 < position_error < remaining_dist:
                    # Back off by roughly how far short CuRobo's own best attempt
                    # landed (2x margin so we don't retry right at the same edge
                    # of infeasibility), instead of blindly halving every time.
                    target_remaining = max(min_distance, remaining_dist - 2.0 * position_error)
                    shrink_fraction = target_remaining / remaining_dist
                else:
                    shrink_fraction = 0.5
                goal[:3] = current_pose[:3] + (goal[:3] - current_pose[:3]) * shrink_fraction
                if log_move:
                    print(f"[waypoint-shrink] {stage_label} iter {iteration} failed ({last_reason}, "
                          f"position_error={position_error}); shrinking target toward current pose "
                          f"(fraction={shrink_fraction:.2f}) -> new distance "
                          f"{float(np.linalg.norm(goal[:3] - current_pose[:3])):.3f}m")
            return False, last_reason, qpos0, trajectories

        def _record_plan_effort(self, stage_label, arm_tag, result):
            """ROBOPRO Phase 4: log curobo's effort for ONE plan_path call so the
            APPROACH_MODE A/B can report attempts, not just success rate. Recorded for
            the DIRECT attempt only (the stage the experiment varies); the
            shrinking-waypoint fallback is deliberately not counted here so 'attempts'
            stays comparable across modes. Read by run() into records.jsonl. Never raises
            -- this is measurement, it must not be able to fail a rollout."""
            try:
                if not hasattr(self, "rollout_plan_effort"):
                    self.rollout_plan_effort = []
                self.rollout_plan_effort.append({
                    "stage": stage_label,
                    "arm": str(arm_tag),
                    "status": result.get("status"),
                    "attempts": int(result.get("attempts", 0) or 0),
                    "trajopt_attempts": int(result.get("trajopt_attempts", 0) or 0),
                    "seeded": bool(result.get("seeded", False)),
                })
            except Exception:
                pass

        def _plan_pose_with_local_waypoint_retry(self, arm_tag, target_pose, qpos, stage_label,
                                                start_pose=None, retries=1, constraint_pose=None,
                                                approach_axis=None, local_attempts=LOCAL_WAYPOINT_ATTEMPTS,
                                                relax_orientation=False, seed_traj=None):
            """Plan one target; if direct planning fails, try the adaptive
            waypoint-shrink retry (see _plan_pose_with_shrinking_waypoint).

            relax_orientation=True: position-only goal (PoseCostMetric.
            reach_partial_pose, see planner.py's plan_path) -- for transit-only moves
            where the target's orientation doesn't matter, only its position. A
            check_ik_batch(relax_orientation=True) sweep confirmed this opens up
            real, currently-dead combos (e.g. lift_above_box at low clearance_z)."""
            qpos0 = np.array(qpos, dtype=np.float64, copy=True)
            start_pose = list(self.get_arm_pose(arm_tag)) if start_pose is None else list(start_pose)
            last_reason = "unknown"
            plan_func = self._arm_plan_func(arm_tag)
            for _ in range(max(1, int(retries))):
                # ROBOPRO Phase 3: the external seed (if any) warm-starts only the DIRECT attempt; the
                # shrinking-waypoint fallback below stays unseeded (it's the safety net). None => stock.
                result = plan_func(target_pose, last_qpos=qpos0,
                                   constraint_pose=constraint_pose, approach_axis=approach_axis,
                                   relax_orientation=relax_orientation, seed_traj=seed_traj)
                self._record_plan_effort(stage_label, arm_tag, result)
                if result.get("status") == "Success":
                    return True, None, self._roll_qpos_forward(arm_tag, qpos0, result), [(stage_label, result)]
                last_reason = result.get("fail_reason", "unknown")
            if local_attempts <= 0:
                return False, last_reason, qpos0, []
            return self._plan_pose_with_shrinking_waypoint(
                arm_tag, target_pose, qpos0, stage_label, start_pose,
                constraint_pose=constraint_pose, approach_axis=approach_axis,
                relax_orientation=relax_orientation, max_iterations=local_attempts)

        def _replay_planned_sequence(self, arm_tag, trajectories):
            """Replay one planned trajectory or a list of labeled trajectories."""
            if trajectories is None:
                return
            if isinstance(trajectories, dict):
                trajectories = [trajectories]
            for item in trajectories:
                traj = item[1] if isinstance(item, tuple) else item
                self._replay_planned_move(arm_tag, traj)
                if not self.plan_success:
                    return

        def _plan_pose_sequence(self, arm_tag, poses, start_qpos, stage_labels=None):
            """Returns (success, failed_stage_label, fail_reason, final_qpos).
            failed_stage_label/fail_reason are None on success; failed_stage_label is
            the label (from stage_labels, positionally aligned with poses) of the
            first pose in the chain that failed to plan, and fail_reason is CuRobo's
            own MotionGenStatus string for that failure (e.g. IK_FAIL vs a
            collision-specific status). final_qpos is the qpos reached by the last
            successfully-planned pose (== start_qpos if the very first pose failed),
            so a caller can resume planning a follow-on sub-chain from where this one
            left off (e.g. after toggling world state between segments)."""
            qpos = np.array(start_qpos, dtype=np.float64, copy=True)
            start_pose = list(self.get_arm_pose(arm_tag))
            for idx, pose in enumerate(poses):
                label = stage_labels[idx] if stage_labels else f"pose{idx}"
                ok, fail_reason, qpos, _ = self._plan_pose_with_local_waypoint_retry(
                    arm_tag, pose, qpos, label, start_pose=start_pose)
                if not ok:
                    return False, label, fail_reason, qpos
                start_pose = pose
            return True, None, None, qpos

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
                cfg = sfc.SeedMetricConfig(
                    obstacles=os.environ.get("SEED_OBSTACLES", "all").strip().lower(),
                    res=float(os.environ.get("SEED_RES", "0.02")),
                    zres=float(os.environ.get("SEED_ZRES", "0.03")),
                    zmax=float(os.environ.get("SEED_ZMAX", "1.23")),
                    chunk=int(os.environ.get("SEED_CHUNK", "256")),
                    # Joint-continuity gate. Exposed because the build's own tau sweep
                    # reports the smallest tau that would have connected a route it had
                    # to reject, and that number is only useful if it can be set back in.
                    # Raising it admits larger joint steps between adjacent voxels: too
                    # low and a smooth stretch reads as a branch seam (what coarse
                    # SEED_RES causes); too high and the route may hop IK branches, which
                    # makes the seed a worse initialization but never an unsafe one --
                    # trajopt still has to produce the trajectory it executes.
                    gate_tau=float(os.environ.get("SEED_GATE_TAU", "0.35")))
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
                import carry_object_spheres as cos
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
                cfg = sfc.SeedMetricConfig(
                    obstacles=os.environ.get("SEED_OBSTACLES", "all").strip().lower(),
                    res=float(os.environ.get("SEED_RES", "0.02")),
                    zres=float(os.environ.get("SEED_ZRES", "0.03")),
                    zmax=float(os.environ.get("SEED_ZMAX", "1.23")),
                    chunk=int(os.environ.get("SEED_CHUNK", "256")),
                    # Joint-continuity gate. Exposed because the build's own tau sweep
                    # reports the smallest tau that would have connected a route it had
                    # to reject, and that number is only useful if it can be set back in.
                    # Raising it admits larger joint steps between adjacent voxels: too
                    # low and a smooth stretch reads as a branch seam (what coarse
                    # SEED_RES causes); too high and the route may hop IK branches, which
                    # makes the seed a worse initialization but never an unsafe one --
                    # trajopt still has to produce the trajectory it executes.
                    gate_tau=float(os.environ.get("SEED_GATE_TAU", "0.35")))
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
                                                       seed_label=f"{ep}-carry", arm=tag)
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
                        import carry_object_spheres as _cos
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

    return OccluderTask


def run(args):
    out_dir = effective_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "records.jsonl"
    offset_specs = parse_offset_specs(args.offsets)
    count_choices = parse_count_choices(args.num_occluders)
    clutter_densities = [int(d) for d in args.clutter_densities.split(",")]
    rollout = getattr(args, "rollout", False)
    ep_counter = 0   # unique episode id per rollout (-> video/episode{N}.mp4)

    env = make_occluder_task()()
    spec_desc = ", ".join(f"{lo:.2f}" if lo == hi else f"{lo:.2f}-{hi:.2f}"
                          for lo, hi, _ in offset_specs)
    print(f"seeds from {args.seed_start}, want {args.num_seeds} STABLE  offsets=[{spec_desc}]  "
          f"n_occluders={count_choices}  ring_rotation="
          f"{'random' if args.random_ring_rotation else 'fixed(front)'}  "
          f"clutter_densities={clutter_densities}  rollout={rollout}  camera={CAMERA}")
    print(f"writing -> {jsonl_path}\n")

    def safe_close():
        try:
            env.close_env()
        except Exception:
            pass

    with open(jsonl_path, "w") as fout:
        # Draw seeds until we have args.num_seeds FULLY-USABLE seeds -> num_seeds complete
        # trajectory sets. "Fully usable" means EVERY build the seed needs succeeds and is
        # stable: the clean denominator build AND every (offset x clutter_density) measurement
        # build (target upright, occluder upright per check_stable, occluder clear of the pad).
        # If ANY is rejected (UnStableError from a toppled bottle/milk box, a too-close-to-pad
        # occluder, or any build error) the WHOLE seed is discarded and the next seed is drawn
        # in its place -- nothing partial is written and produced is not incremented. So
        # `--num-seeds N` always yields N seeds, each with the full (offset, density) grid
        # rolled out (== N rollouts for a single offset+density run), never fewer with silent
        # holes. Two passes per seed: (1) build+measure + stability gate, buffering; (2) only
        # if the whole seed passed, run rollouts and commit records. (Ported from the testbench.)
        produced = 0
        draw = args.seed_start          # incrementing seed to draw from (rejected ones skipped)
        max_draws = args.num_seeds * 20 + 50   # safety cap: don't loop forever if builds keep failing
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

            # ---- Pass 1: build + measure every (offset, density); gate stability ----
            seed_items = []       # buffered per-build results, committed only if the seed passes
            seed_ok = True
            for spec in offset_specs:
                if not seed_ok:
                    break
                off = spec[2]                  # nominal label for this spec (midpoint of a range)
                # per-build coin flip: drop the occluder with prob no_occluder_prob
                # (keyed on seed+offset so the decision is shared across densities)
                show = bool(np.random.default_rng(int(seed) * 1000 + int(round(off * 100))).random()
                            >= args.no_occluder_prob)
                # The formation (rotation, count, per-occluder radii) is drawn ONCE here and
                # then re-asserted verbatim on the env for both the measurement build and the
                # rollout build -- re-deriving it anywhere would risk measuring one scene and
                # rolling out another.
                angle0, n_occ, radii = draw_ring_config(seed, spec, count_choices,
                                                        args.random_ring_rotation)
                # Reject scenes where ANY occluder that WILL spawn (off-table ring positions
                # are dropped by the same xlim/ylim filter load_actors uses) lands too close
                # to / on the destination pad. Rejects the WHOLE seed (redraw) so the
                # trajectory count is still met.
                if show:
                    ring_xys = occluder_ring_xy(clean_pose[0], clean_pose[1], radii, n_occ,
                                                angle0, xlim=TABLE_XLIM, ylim=TABLE_YLIM)
                    pad_dist = min((float(np.linalg.norm(np.array(xy) - np.array(PAD_XY)))
                                    for xy in ring_xys), default=float("inf"))
                    if pad_dist < OCC_PAD_MIN_DIST:
                        print(f"[seed {seed}] a ring occluder@off={off:.2f} is {pad_dist:.3f}m from "
                              f"pad (< {OCC_PAD_MIN_DIST:.3f}m); rejecting seed, drawing another.")
                        seed_ok = False
                        break
                env.spawn_occluder = show
                env.occluder_offset = off
                env.num_occluders = n_occ
                env.occluder_angle0 = angle0
                env.occluder_radii = list(radii)
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
                        # A toppled bottle/milk box (check_stable -> UnStableError) or any build
                        # error rejects the ENTIRE seed so we redraw and still hit num_seeds.
                        print(f"[seed {seed} off{off:.2f} cd{cd}] build failed/unstable "
                              f"({type(e).__name__}: {e}); rejecting seed, drawing another.")
                        safe_close()
                        seed_ok = False
                        break
                    safe_close()

                    # "offset" stays the spec's NOMINAL value so --group-by offset buckets a
                    # range into one group; occluder_radii/angle0 carry what was actually
                    # spawned, which is the truth for anything reading the geometry back.
                    rec = {"seed": int(seed), "offset": float(off), "full_px": int(full_px),
                           "visible_px": int(res["visible_pixel_count"]),
                           "visible_fraction": float(res["visible_fraction"]),
                           "bucket": res["bucket"], "in_fov": bool(res["in_fov"]),
                           "pose_match": pose_ok, "occluder_shown": show,
                           "num_occluders": int(n_occ) if show else 0,
                           "occluder_radii": ([round(r, 4) for r in radii] if show else []),
                           "occluder_angle0": (round(float(angle0), 4) if show else None),
                           "clutter_density": int(cd)}
                    seed_items.append({"off": off, "cd": cd, "show": show,
                                       "n_occ": n_occ, "angle0": angle0, "radii": list(radii),
                                       "rec": rec, "res": res})

            if not seed_ok or not seed_items:
                continue   # rejected (unstable / pad-blocked / produced nothing) -> next seed

            # ---- Pass 2: seed accepted -> roll out and commit all its records ----
            for item in seed_items:
                off, cd, show, rec, res = item["off"], item["cd"], item["show"], item["rec"], item["res"]
                # --- expert curobo rollout on the same scene (video + success) ---
                # Re-assert spawn_occluder / occluder_offset so run_rollout's own build
                # reproduces this item's occluder placement (env is shared across items).
                if rollout:
                    env.spawn_occluder = show
                    env.occluder_offset = off
                    env.num_occluders = item["n_occ"]
                    env.occluder_angle0 = item["angle0"]
                    env.occluder_radii = list(item["radii"])
                    # One log file per rollout under <out-dir>/log/episode{N}.log: the
                    # usual stdout/stderr (play_once + run_rollout diagnostics) tee'd to
                    # both console and file, plus the expert's per-phase timing appended.
                    log_dir = out_dir / "log"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    # utf-8 (not the platform-default ASCII) so the rollout's emoji log
                    # lines ("Video is saved...") encode cleanly into the file.
                    rollout_log = open(log_dir / f"episode{ep_counter}.log", "w",
                                       encoding="utf-8")
                    rollout_log.write(
                        f"# rollout episode{ep_counter}  seed={seed} offset={off:.3f} "
                        f"clutter_density={cd} occluder_shown={show} "
                        f"num_occluders={item['n_occ'] if show else 0} "
                        f"radii={[round(r, 3) for r in item['radii']] if show else []} "
                        f"angle0={item['angle0']:.3f}  "
                        f"bucket_measured={res['bucket']}\n\n")
                    rollout_log.flush()
                    # expose this episode's output folder + number to the seed-route visuals
                    # (_get_approach_seed writes into out_dir/seed_route_visuals/episode<N>_<arm>/)
                    env._rollout_out_dir = out_dir
                    env._rollout_ep = ep_counter
                    _t_rollout = time.perf_counter()
                    with contextlib.redirect_stdout(_Tee(sys.stdout, rollout_log)), \
                         contextlib.redirect_stderr(_Tee(sys.stderr, rollout_log)):
                        rollout_result = run_rollout(env, "put_mouse_on_pad", args.base_config, seed,
                                                     dr_measure(cd), out_dir, ep_counter)
                    success = bool(rollout_result["success"])
                    rec["rollout_success"] = success
                    # Wall-clock for THIS rollout (successes and failures alike -- a failure
                    # costs time too). Lets the summary report usable-samples/hour, which is
                    # what actually matters when the seed buys success rate but costs a
                    # clearance-metric build per scene.
                    rec["rollout_seconds"] = float(time.perf_counter() - _t_rollout)
                    rec["attached_trajectory_slowdown"] = ATTACHED_TRAJECTORY_SLOWDOWN
                    rec["rollout_ep"] = ep_counter
                    artifact_info = rollout_result.get("artifact_info") or {}
                    rec["rollout_bucket"] = artifact_info.get("bucket", "success" if success else "fail")
                    rec["rollout_video"] = artifact_info.get(
                        "video_relpath", f"{rec['rollout_bucket']}/video/episode{ep_counter}.mp4"
                    )
                    rec["rollout_data"] = artifact_info.get(
                        "hdf5_relpath", f"{rec['rollout_bucket']}/data/episode{ep_counter}.hdf5"
                    )
                    # Structured failure/candidate metadata set by play_once (see
                    # its checkpoint()/_select_pick_place_candidate) -- survives
                    # run_rollout's internal close_env() since it's just plain
                    # attributes on the (reused) env object. None on success.
                    rec["rollout_failure_stage"] = getattr(env, "rollout_failure_stage", None)
                    rec["rollout_failure_reason"] = getattr(env, "rollout_failure_reason", None)
                    rec["rollout_candidate_info"] = getattr(env, "rollout_candidate_info", None)
                    rec["rollout_grasp_attempts"] = getattr(env, "rollout_grasp_attempts", None)
                    rec["rollout_retention_checks"] = getattr(env, "rollout_retention_checks", None)
                    rec["rollout_descent_slices"] = getattr(env, "rollout_descent_slices", None)
                    # ROBOPRO Phase 4 (APPROACH_MODE A/B). approach_mode is stamped on every
                    # record so a results folder is self-describing and the summary can refuse
                    # to compare cells that were not actually run in different modes.
                    rec["approach_mode"] = env._approach_mode()
                    # Same reason: the backward leg is an independent knob, so a cell is only
                    # fully identified by BOTH modes. Without this, a scripted-placement run
                    # and a direct-placement run are indistinguishable in the results folder.
                    rec["placement_mode"] = env._placement_mode()
                    rec["rollout_plan_effort"] = getattr(env, "rollout_plan_effort", None)
                    rec["rollout_seed_stats"] = getattr(env, "rollout_seed_stats", None)
                    # Physics collision metrics for THIS episode (accumulated by
                    # check_collisions during play_once; reset per build by
                    # _init_collision_metrics; survives close_env like the fields
                    # above). This is the direct clutter-avoidance measurement:
                    # robot_to_static_object / target_to_static_object count clutter
                    # the arm / held object hit, with the object names. Empty/zero
                    # when enable_collision_metrics is off.
                    try:
                        rec["collision_metrics"] = env.get_collision_metrics()
                    except Exception:
                        rec["collision_metrics"] = None
                    cm = rec["collision_metrics"] or {}
                    summary = (f"    seed {seed} off={off:.2f} cd={cd} {res['bucket']}: "
                               f"rollout {'SUCCESS' if success else 'FAIL'} "
                               f"collisions={cm.get('total_collision_count', '?')} "
                               f"(clutter={cm.get('robot_to_static_object', '?')}"
                               f"+{cm.get('target_to_static_object', '?')}) -> "
                               f"{rec['rollout_bucket']}/episode{ep_counter}")
                    print(summary)
                    # append the expert's per-phase timing, then close the per-rollout log
                    stage_times = getattr(env, "rollout_stage_times", None) or []
                    rollout_log.write(summary + "\n")
                    rollout_log.write("\n# expert phase timings (wall-clock seconds per phase)\n")
                    total_s = 0.0
                    for stg in stage_times:
                        rollout_log.write(f"  {stg['stage']:<34} {stg['seconds']:8.3f}s\n")
                        total_s += stg["seconds"]
                    rollout_log.write(f"  {'TOTAL':<34} {total_s:8.3f}s\n")
                    rollout_log.close()
                    # top-level data/ and video/ are emptied by the success/fail bucketing
                    _prune_empty_topdirs(out_dir)
                    ep_counter += 1
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            produced += 1
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
    ap.add_argument("--offsets", default="0.2",
                    help="occluder radius/radii in m (target at centre); one figure group per "
                         "token. A token is either a FIXED value (0.20 -- every occluder at that "
                         "radius, the original behaviour) or a RANGE (0.10-0.25 -- each occluder "
                         "independently draws its own radius from that interval, so one scene can "
                         "mix near and far occluders). e.g. 0.15,0.20,0.25 or 0.10-0.25")
    ap.add_argument("--num-occluders", default="1",
                    help="how many occluders to spawn, equally spaced in angle. A single value "
                         "(1 = the original single box in front) or a comma-separated menu "
                         "(2,3,4,5) from which each scene draws one, for a variety of "
                         "configurations.")
    ap.add_argument("--random-ring-rotation", action=argparse.BooleanOptionalAction, default=False,
                    help="rotate the whole formation by a random theta in [0, 2pi) per scene, so "
                         "the occluders are not always anchored with one directly in front of the "
                         "target. Off by default (theta=0), which keeps the historical layout.")
    ap.add_argument("--clutter-densities", default="0",
                    help="table clutter density/densities to sweep in the measurement scene, "
                         "comma-separated (0=off). e.g. 0,8,15")
    ap.add_argument("--group-by", default="offset", choices=["offset", "clutter_density"],
                    help="analysis grouping variable for the bucket/histogram figures")
    ap.add_argument("--no-occluder-prob", type=float, default=0.2,
                    help="probability the olive-oil occluder is NOT spawned for a build (default 0.2)")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--selectable-threshold", type=float, default=0.05)
    ap.add_argument("--out-dir", default="../scripts/validation/results/occluder_visibility",
                    help="dedicated results folder for this script; each run lands in its own "
                         "<out-dir>/<type>/<timestamp>/ subfolder (type = --run-type)")
    ap.add_argument("--run-type", default=None,
                    help="name of the <out-dir>/<type>/ subfolder grouping this run "
                         "(default: 'rollout' when --rollout else 'no_rollout'). Use to label "
                         "variants, e.g. a planner name like 'hamid' or 'baseline'.")
    # Tri-state: default None -> resolved after parsing to "on unless --rollout".
    # No-rollout runs save the per-scene overlay PNGs by default (they're the run's
    # primary artifact); --rollout runs default to video-only (run_rollout always
    # saves the episode mp4) and skip the stills. Pass --save-images / --no-save-images
    # to force either way regardless of --rollout.
    ap.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=None,
                    help="save per-scene overlay PNGs (default: on without --rollout, off with it)")
    ap.add_argument("--rollout", action="store_true",
                    help="run an expert curobo rollout per scene (goes under the 'rollout' type "
                         "folder, saves videos, and adds success-only distribution + P(success) per bucket)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    # Resolve the tri-state --save-images default: on for no-rollout runs (overlay PNGs
    # are their main artifact), off for --rollout runs (the episode video is). An explicit
    # --save-images / --no-save-images on the command line wins over this.
    if args.save_images is None:
        args.save_images = not args.rollout

    # One folder per run: <out-dir>/<type>/<timestamp>, so re-running never clobbers
    # previous results. --plot-only re-reads an existing run, so leave args.out-dir
    # exactly as the user pointed it (it should already be a .../<type>/<timestamp> dir).
    if not args.plot_only:
        run_type = args.run_type or ("rollout" if args.rollout else "no_rollout")
        args.out_dir = str(Path(args.out_dir) / run_type / datetime.now().strftime("%Y%m%d-%H%M%S"))
        print(f"[run] writing to {args.out_dir}")

    if not args.plot_only:
        run(args)
    group_label = {"offset": "occluder offset (m)",
                   "clutter_density": "table clutter density"}[args.group_by]
    out_dir = effective_out_dir(args)
    analyze_kwargs = dict(group_key=args.group_by, group_label=group_label,
                          suptitle="Visibility with one olive-oil occluder (countertop)",
                          bar_title=f"Bucket proportions vs {group_label}")
    if args.rollout:
        analyze_rollout(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)
    else:
        analyze(out_dir, args.bins, args.selectable_threshold, **analyze_kwargs)


if __name__ == "__main__":
    main()
