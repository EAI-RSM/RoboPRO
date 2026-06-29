"""robo_negative — Targeted Data Collection for RoboPRO (single-file PoC library).

Consolidated from the former `targeted/` package (registry + sampler + labels +
metrics + runtime + hdf5 annotation + tests) into one module for the PoC.

Layers (each its own section below):
  registry   supported (task, ptype) list + scene-pairing rule
  sampler    stratified, class-by-construction shift sampling (pure)
  labels     three-layer label derivation + WHAT/WHY/quality annotation (pure)
  metrics    offline per-frame progress / safety from logged state (pure)
  runtime    TargetedRuntime — per-episode perturbation state + triggers + the
             integrity guard; consulted by Bench_base_task hooks (env.targeted)
  hdf5       append the label record + raw frame state to the episode HDF5

The pure label/metric/sampler functions need only numpy (+ h5py for the HDF5
helpers); `TargetedRuntime` imports sapien lazily, so viewers can import this
module without a sim install. Run the unit tests with:
    python -m robo_negative
"""
import json
from pathlib import Path

import numpy as np

LABEL_SCHEMA_VERSION = 1


# ======================================================================
# Registry — incremental task enablement + scene-pairing rule.
# A task enters SUPPORTED_TASKS only after a desync audit (locked quantity does
# not follow the shift, check_success reads live state). Collectors/runners must
# SKIP unsupported (task, ptype) with a logged reason, never crash or mislabel.
# ======================================================================

# object/target desyncs are collected in clean scenes only; obstacle desyncs
# need clutter, so they run in cluttered scenes only.
REQUIRED_SCENE_BY_PTYPE = {"shift_object": "clean", "shift_target": "clean",
                           "shift_obstacle": "cluttered", "hide_obstacle": "cluttered"}

_ALL_PTYPES = frozenset({"shift_object", "shift_target", "shift_obstacle", "hide_obstacle"})

SUPPORTED_TASKS: dict[str, dict] = {
    # Verified 2026-06-10/12 (PoC batch: 84 episodes + demo quad):
    #   shift_object  after_grasp_plan, boundary ~2.9-3.5 cm (plan_aborted)
    #   shift_target  immediate, flips at the 2 cm check_success epsilon
    #   shift_obstacle corridor form, milk box, success_with_collision
    "put_mouse_on_pad": {
        "bench_subdir": "office",
        "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_office_clean",
        "cluttered_config": "bench_demo_office_d6",
    },
    "put_book_on_book": {
        "bench_subdir": "office", "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_office_clean", "cluttered_config": "bench_demo_office_d6"},
    "put_mouse_next_to_stapler": {
        "bench_subdir": "office", "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_office_clean", "cluttered_config": "bench_demo_office_d6"},
    "put_spoon_on_plate_ks": {
        "bench_subdir": "kitchens", "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_kitchens_clean", "cluttered_config": "bench_demo_kitchens_d6"},
    # Audited 2026-06-29 for the targeted-negative demo: all read live state in
    # check_success and carry self.target_obj + self.des_obj. shift_object /
    # shift_target use cached des_obj_pose (place at stale pose). The obstacle
    # perturbations need no per-task obstacle — the runtime injects a corridor
    # milk-box (inject_corridor_obstacle), so they run in the CLEAN scene too.
    # NOTE: put_phone_on_holder reads its destination LIVE in play_once, so
    # shift_target is typically *absorbed* (clean_success) — kept honest.
    "put_stapler_on_book": {
        "bench_subdir": "office", "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_office_clean", "cluttered_config": "bench_demo_office_d6"},
    "put_phone_on_holder": {
        "bench_subdir": "office", "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_office_clean", "cluttered_config": "bench_demo_office_d6"},
    "put_bread_on_board_ks": {
        "bench_subdir": "kitchens", "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_kitchens_clean", "cluttered_config": "bench_demo_kitchens_d6"},
    "drop_apple_in_bin_ks": {
        "bench_subdir": "kitchens", "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_kitchens_clean", "cluttered_config": "bench_demo_kitchens_d6"},
    "move_hamburger_onto_plate_ks": {
        "bench_subdir": "kitchens", "ptypes": _ALL_PTYPES,
        "clean_config": "bench_demo_kitchens_clean", "cluttered_config": "bench_demo_kitchens_d6"},
}


def task_entry(task_name: str) -> dict | None:
    return SUPPORTED_TASKS.get(task_name)


def is_supported(task_name: str, ptype: str | None = None) -> bool:
    entry = SUPPORTED_TASKS.get(task_name)
    if entry is None:
        return False
    return ptype is None or ptype in entry["ptypes"]


# ======================================================================
# Sampler — stratified, class-by-construction shift sampling. Isolated RNG that
# never touches the global np.random stream scene generation consumes.
# ======================================================================

MAGNITUDE_BINS_CM = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 8.0))
PERCEPTUAL_CLASSES = ("depth", "lateral", "mixed")
# Stable per-type RNG stream codes. Never hash() — it is salted per process.
PTYPE_CODES = {"shift_object": 1, "shift_target": 2, "shift_obstacle": 3, "hide_obstacle": 4}


def sample_shift_params(ptype: str, scene_seed: int, episode_index: int) -> dict:
    """Sample (class, magnitude, sign, mixed angle) for one perturbed episode.

    Deterministic given (ptype, scene_seed, episode_index). Class and magnitude
    bin are assigned by round-robin quota so small PoC batches cover the full
    class x bin grid; only the within-bin magnitude, the sign and the mixed
    angle are random draws.
    """
    rng = np.random.default_rng([PTYPE_CODES[ptype], int(scene_seed), int(episode_index)])
    target_class = PERCEPTUAL_CLASSES[episode_index % len(PERCEPTUAL_CLASSES)]
    bin_index = (episode_index // len(PERCEPTUAL_CLASSES)) % len(MAGNITUDE_BINS_CM)
    lo, hi = MAGNITUDE_BINS_CM[bin_index]
    return {
        "target_class": target_class,
        "magnitude_bin_index": int(bin_index),
        "magnitude_bin_cm": [lo, hi],
        "magnitude_cm": float(rng.uniform(lo, hi)),
        "sign": int(rng.choice([-1, 1])),
        # Deliberately off-axis angle for the `mixed` class (25-65 deg off the
        # depth axis, either side).
        "mixed_angle_deg": float(rng.uniform(25.0, 65.0)) * (-1 if rng.random() < 0.5 else 1),
    }


def construct_world_shift(params: dict, depth_axis: np.ndarray, lateral_axis: np.ndarray) -> np.ndarray:
    """Build the commanded world-frame shift (z = 0) from sampled params and the
    table-plane camera axes. Returns a (3,) vector in meters."""
    depth = np.asarray(depth_axis, dtype=np.float64)
    lateral = np.asarray(lateral_axis, dtype=np.float64)
    cls = params["target_class"]
    if cls == "depth":
        direction = depth.copy()
    elif cls == "lateral":
        direction = lateral.copy()
    elif cls == "mixed":
        a = np.deg2rad(params["mixed_angle_deg"])
        direction = np.cos(a) * depth + np.sin(a) * lateral
    else:
        raise ValueError(f"unknown perceptual class: {cls}")
    direction = direction / np.linalg.norm(direction)
    shift = params["sign"] * (params["magnitude_cm"] / 100.0) * direction
    return np.array([shift[0], shift[1], 0.0], dtype=np.float64)


# ======================================================================
# Labels — Layer 1 (cause) is recorded by the runner; this derives the Layer-2
# camera-frame consistency check and the Layer-3 mechanical outcome, plus the
# WHAT (flag-set) / WHY / quality annotation. Pure, unit-tested (bottom).
# ======================================================================

# Object moved less than this from its (post-shift) start pose => the grasp never picked it up.
GRASP_MOVED_THRESHOLD_CM = 3.0
# Mirrors put_mouse_on_pad.check_success epsilon (per-axis 2 cm on the live pad).
PLACE_SUCCESS_EPS_CM = 2.0
# Achieved-shift dominance threshold for the post-hoc consistency check (NOT the
# label — labels are by construction). Must exceed cos(25 deg) ~ 0.906 so
# constructed `mixed` shifts (25-65 deg off depth) never read as dominant.
DOMINANCE_FRACTION = 0.92

OUTCOMES = ("clean_success", "success_with_collision", "empty_grasp",
            "wrong_place_pose", "collision_with_object", "plan_aborted")

# WHAT vocabulary — co-occurring flags, NOT a precedence ladder.
WHAT_FLAGS = ("grasp_failure", "placement_failure", "collision", "planning_failure")

# WHY: the cause, known by construction from the intervention we applied.
WHY_BY_PTYPE = {
    "shift_object": ("object_mislocalized", "target"),
    "shift_target": ("destination_mislocalized", "destination"),
    "shift_obstacle": ("obstacle_mislocalized", "obstacle"),
    "hide_obstacle": ("object_undetected", "obstacle"),
}


def camera_frame_decomposition(shift_world, depth_axis, lateral_axis) -> dict:
    """Decompose a world-frame shift onto the table-plane camera axes. Returns
    cm components plus the post-hoc dominance classification (a consistency
    check against the constructed class)."""
    s = np.asarray(shift_world, dtype=np.float64)
    depth_cm = float(np.dot(s[:2], np.asarray(depth_axis)[:2]) * 100.0)
    lateral_cm = float(np.dot(s[:2], np.asarray(lateral_axis)[:2]) * 100.0)
    vertical_cm = float(s[2] * 100.0)
    horizontal_norm = float(np.hypot(depth_cm, lateral_cm))
    if horizontal_norm < 1e-9:
        dominant = "none"
    elif abs(depth_cm) >= DOMINANCE_FRACTION * horizontal_norm:
        dominant = "depth"
    elif abs(lateral_cm) >= DOMINANCE_FRACTION * horizontal_norm:
        dominant = "lateral"
    else:
        dominant = "mixed"
    return {"depth_cm": depth_cm, "lateral_cm": lateral_cm, "vertical_cm": vertical_cm,
            "posthoc_dominant_axis": dominant}


def derive_outcome(*, plan_success: bool, success: bool, is_collision: bool,
                   object_moved_cm: float | None, place_error_cm: float | None) -> tuple[str, str | None]:
    """Layer-3 mechanical outcome from sim signals. Success outcomes are
    first-class; object identity / planner visibility are separate attrs and
    never baked into the label."""
    if not plan_success:
        return "plan_aborted", "executor aborted: a motion plan failed"
    if success:
        if is_collision:
            return "success_with_collision", None
        return "clean_success", None
    if object_moved_cm is not None and object_moved_cm < GRASP_MOVED_THRESHOLD_CM:
        return "empty_grasp", (
            f"object moved only {object_moved_cm:.1f} cm from its start pose "
            f"(< {GRASP_MOVED_THRESHOLD_CM} cm threshold)")
    if is_collision:
        return "collision_with_object", "episode failed and a collision was recorded"
    detail = None
    if place_error_cm is not None:
        detail = f"object moved but ended {place_error_cm:.1f} cm (xy) from the live target"
    return "wrong_place_pose", detail


def manifestations(*, plan_success: bool, success: bool, is_collision: bool,
                   object_moved_cm: float | None, place_error_cm: float | None) -> list:
    """WHAT as a flag-set: flags co-occur (a mislocalized grasp that also trips
    the planner is {grasp_failure, planning_failure}), so nothing is masked by a
    single precedence-ordered outcome."""
    flags = []
    if not success:
        if object_moved_cm is not None and object_moved_cm < GRASP_MOVED_THRESHOLD_CM:
            flags.append("grasp_failure")
        elif place_error_cm is not None and place_error_cm > PLACE_SUCCESS_EPS_CM:
            flags.append("placement_failure")
    if is_collision:
        flags.append("collision")
    if not plan_success:
        flags.append("planning_failure")
    return flags


def quality(*, success: bool, what: list) -> str:
    """good = success, no adverse manifestation (incl. fault absorbed);
    suboptimal = success but something happened (e.g. collision); bad = failed."""
    if not success:
        return "bad"
    return "suboptimal" if what else "good"


def annotate(record: dict) -> dict:
    """Top-level annotation from a recorded episode dict (WHAT flag-set / WHY /
    quality). Pure / offline: reads `signals`, `contact_log`, `perturbation_type`."""
    sig = record.get("signals") or {}
    cm = sig.get("collision_metrics") or {}
    contact_frames = [c["frame_idx"] for c in record.get("contact_log", []) if c.get("contacts")]
    is_coll = bool(cm.get("is_collision")) or bool(contact_frames)
    success = bool(sig.get("success_flag"))
    what = manifestations(
        plan_success=bool(sig.get("plan_success")),
        success=success,
        is_collision=is_coll,
        object_moved_cm=sig.get("object_moved_cm"),
        place_error_cm=sig.get("place_error_cm"),
    )
    why_val, why_role = WHY_BY_PTYPE.get(record.get("perturbation_type"), ("none", None))
    shift_frame = record.get("shift_frame_idx")
    return {
        "what": what,                                   # flag-set (possibly empty)
        "what_frames": {"collision": contact_frames},   # per-timestep where available
        "why": why_val,
        "why_role": why_role,
        "why_axis": record.get("perceptual_failure_class"),
        "why_active_from_frame": int(shift_frame) if shift_frame is not None else 0,
        "quality": quality(success=success, what=what),
        "task_success": success,
    }


# ======================================================================
# Metrics — offline task-progress / safety from logged per-frame state. The sim
# loop logs ONLY raw actor poses (group `targeted_state`); every scalar metric
# is computed HERE so adding/changing one never requires re-simulating. Pure:
# numpy + h5py only.
# ======================================================================

TASK_FAMILY = {"put_mouse_on_pad": "pick_and_place"}  # unknown -> pick_and_place


def task_family(task_name: str) -> str:
    return TASK_FAMILY.get(task_name, "pick_and_place")


def load_frame_state(hdf5_path):
    """Load raw per-frame state written by write_frame_state, plus per-frame
    end-effector poses from the standard `endpose` group. Returns
    {names, pos[T,A,3], quat[T,A,4], frame_idx[T], entities, ee} or None."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        if "targeted_state" not in f:
            return None
        g = f["targeted_state"]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in g["actor_names"][:]]
        ent = json.loads(g.attrs["entities_json"]) if "entities_json" in g.attrs else None
        ee = None
        if "endpose" in f:
            ee = {"left": f["endpose/left_endpose"][:], "right": f["endpose/right_endpose"][:]}
        return {"names": names, "pos": g["actor_pos"][:], "quat": g["actor_quat"][:],
                "frame_idx": g["frame_idx"][:], "entities": ent, "ee": ee}


def _idx(state, name):
    return state["names"].index(name) if name and name in state["names"] else None


def object_to_destination_l2_cm(state, target, destination):
    """Pick-and-place progress: L2 distance object<->destination per frame (cm)."""
    ti, di = _idx(state, target), _idx(state, destination)
    if ti is None or di is None:
        return None
    return np.linalg.norm(state["pos"][:, ti, :] - state["pos"][:, di, :], axis=1) * 100.0


# Safety = 1 - proximity-of-an-end-effector-to-any-non-target-object scaled by EE
# speed, in [0,1] (1 = clear). All params are task-dependent.
SAFETY_PARAMS = {
    "pick_and_place": {
        "cutoff_cm": 15.0,         # >= this EE->object distance => 0 proximity term
        "ee_tip_offset_cm": 16.0,  # drop the flange pose to approximate the gripper tip
        "v_ref_cm": 0.9,           # EE displacement/frame that saturates the speed factor
        "speed_floor": 0.65,       # fraction of the proximity term kept at zero speed
        "structural": ("ground", "table", "wall"),  # never count as hazards
    },
}


def compute_safety(record: dict, state: dict):
    """Per-frame safety in [0,1] (1 = clear): 1 - proximity of either end-effector
    to the nearest non-target object (target/destination/structural excluded),
    scaled by EE speed. Returns {metric, unit, values, cutoff_cm} or None."""
    p = SAFETY_PARAMS.get(task_family(record.get("task_name")))
    if p is None or not state.get("ee"):
        return None
    ent = state.get("entities") or record.get("progress_entities") or {}
    exclude = set(p["structural"]) | {ent.get("target"), ent.get("destination")}
    haz_idx = [i for i, n in enumerate(state["names"]) if n not in exclude]
    if not haz_idx:
        return None
    haz = state["pos"][:, haz_idx, :]                         # [T, H, 3]
    T = haz.shape[0]
    cutoff = p["cutoff_cm"] / 100.0
    vref = p["v_ref_cm"] / 100.0
    floor = p["speed_floor"]
    tip = np.array([0.0, 0.0, -p["ee_tip_offset_cm"] / 100.0])  # flange -> approx tip
    risk = np.zeros(T)
    for arm in ("left", "right"):
        if state["ee"].get(arm) is None:
            continue
        ee = np.asarray(state["ee"][arm])[:, :3] + tip        # [T, 3]
        d = np.linalg.norm(haz - ee[:, None, :], axis=2)      # [T, H]
        nearest = np.nanmin(d, axis=1)
        nearest = np.where(np.isnan(nearest), cutoff * 10, nearest)
        prox = np.clip((cutoff - nearest) / cutoff, 0.0, 1.0)
        spd = np.zeros(T)
        spd[1:] = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        vfac = np.clip(spd / vref, 0.0, 1.0)
        risk = np.maximum(risk, prox * (floor + (1.0 - floor) * vfac))
    safety = [round(1.0 - min(1.0, float(x)), 3) for x in risk]
    return {"metric": "safety", "unit": "", "cutoff_cm": p["cutoff_cm"], "values": safety}


def compute_progress(record: dict, state: dict):
    """Task-type-dependent progress normalized to 0-100%: start distance -> 0%,
    reaching the destination (distance 0) -> 100%. Raw distance (cm) returned too.
    Returns {metric, unit, family, values, distance_cm, reference_cm} or None."""
    fam = task_family(record.get("task_name"))
    ent = state.get("entities") or record.get("progress_entities") or {}
    if fam == "pick_and_place":
        dist = object_to_destination_l2_cm(state, ent.get("target"), ent.get("destination"))
        if dist is not None and len(dist):
            ref = float(dist[0]) if float(dist[0]) > 1e-6 else (float(np.max(dist)) or 1.0)
            pct = [round(max(0.0, min(100.0, (1.0 - float(v) / ref) * 100.0)), 2) for v in dist]
            return {"metric": "task progress", "unit": "%", "family": fam, "values": pct,
                    "distance_cm": [round(float(v), 3) for v in dist], "reference_cm": round(ref, 2)}
    return None


# ======================================================================
# Runtime — TargetedRuntime, attached as `env.targeted`. Bench_base_task.update_world
# consults is_excluded / override_pose on every call (persistent planner blindness);
# grasp_actor fires notify("after_grasp_plan", env) once grasp poses are locked.
# sapien is imported lazily (only apply_shift constructs poses), so the pure
# functions above import without a sim install.
# ======================================================================

SETTLE_STEPS = 25
OTHER_ACTOR_TOL_M = 0.005          # principle 5(a): non-perturbed actors move < this during settle
ACHIEVED_TOL_FRACTION = 0.25       # principle 5(b): achieved within max(25% of commanded, 5 mm)
ACHIEVED_TOL_MIN_M = 0.005


class EpisodeDiscard(Exception):
    """Integrity guard rejected an episode. Reason ∈ {'scene_disturbed',
    'shift_not_achieved', 'no_eligible_obstacle'}."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def _entity(actor):
    """Unwrap envs.utils.actor_utils.Actor -> sapien Entity (raw entities pass through)."""
    return getattr(actor, "actor", actor)


def _is_dynamic(ent) -> bool:
    get_components = getattr(ent, "get_components", None)
    if get_components is None:
        return False  # articulations (cabinet/drawer furniture) are not shiftable obstacles
    return any("RigidDynamic" in type(c).__name__ for c in get_components())


def snapshot_poses(env) -> dict:
    return {e.per_scene_id: (e.get_name(), np.array(e.get_pose().p, dtype=np.float64))
            for e in env.scene.get_all_actors()}


def settle(env, steps: int = SETTLE_STEPS):
    for _ in range(steps):  # settle steps are never recorded (no _take_picture) — principle 7
        env.scene.step()
    env._update_render()


class TargetedRuntime:
    """Holds the episode's perturbation state and applies shifts.

    Lifecycle (driven by run_targeted_episode.py):
        rt = TargetedRuntime(); rt.attach(env)
        rt.hide_from_planner(env, actor)        # shift_obstacle / hide_obstacle
        rt.apply_shift(env, actor, shift)       # immediate trigger (or zero-shift parity)
        rt.arm("after_grasp_plan", actor, shift, on_applied)  # deferred trigger
    """

    def __init__(self):
        self._excluded_ids: set[int] = set()
        self._pose_overrides: dict = {}          # per_scene_id -> sapien.Pose (pre-shift belief)
        self._armed: dict | None = None
        self.contact_log: list = []
        # Raw per-saved-frame state (all actor poses), aligned 1:1 with the saved
        # video/HDF5 frames (logged by the _take_picture hook). Metrics are computed
        # OFFLINE from this (see compute_progress / compute_safety).
        self.frame_state: list = []

    # ------------- consulted by Bench_base_task hooks -------------

    def is_excluded(self, actor) -> bool:
        sid = getattr(_entity(actor), "per_scene_id", None)  # articulations have none
        return sid is not None and sid in self._excluded_ids

    def override_pose(self, actor, pose):
        """Pose the planner should believe for this actor (the pre-shift pose
        for shifted actors), or the live pose unchanged."""
        sid = getattr(_entity(actor), "per_scene_id", None)
        return pose if sid is None else self._pose_overrides.get(sid, pose)

    def notify(self, event: str, env):
        """Fire the armed deferred shift if it matches `event`. Fires once."""
        armed, self._armed = self._armed, None
        if armed is None or armed["event"] != event:
            self._armed = armed
            return
        rec = self.apply_shift(env, armed["actor"], armed["shift"])
        rec["applied"] = armed["actor"] is not None
        rec["shift_frame_idx"] = int(getattr(env, "FRAME_IDX", 0))
        if armed["on_applied"] is not None:
            armed["on_applied"](rec)

    # ------------- used by the episode runner -------------

    def attach(self, env):
        env.targeted = self
        self._install_frame_state_logger(env)
        if getattr(env, "enable_collision_metrics", False):
            self._install_contact_logger(env)

    def arm(self, event: str, actor, world_shift, on_applied=None):
        self._armed = {"event": event, "actor": actor,
                       "shift": np.asarray(world_shift, dtype=np.float64),
                       "on_applied": on_applied}

    def hide_from_planner(self, env, actor) -> int:
        """Persistently remove `actor` from the planner's collision world (physics
        body untouched) and rebuild CuRobo's world. Every later update_world keeps
        the exclusion."""
        self._excluded_ids.add(_entity(actor).per_scene_id)
        env.update_world(exclude_obstacles=getattr(env, "enable_collision_metrics", False))
        return 1

    def apply_shift(self, env, actor, world_shift) -> dict:
        """Teleport `actor` by `world_shift`, settle, run the two-sided integrity
        guard (principle 5). Call with actor=None / zero shift for the baseline
        twin: identical settle, no shift. After a real shift the planner keeps
        seeing the pre-shift pose via the override map.

        Raises EpisodeDiscard('scene_disturbed' | 'shift_not_achieved')."""
        import sapien.core as sapien

        world_shift = np.asarray(world_shift, dtype=np.float64)
        is_real = float(np.linalg.norm(world_shift)) > 0 and actor is not None

        pre = snapshot_poses(env)
        pid, start_p = None, None
        if is_real:
            ent = _entity(actor)
            pid = ent.per_scene_id
            pose = ent.get_pose()
            start_p = np.array(pose.p, dtype=np.float64)
            # Planner must keep believing the pre-shift pose (principle 8).
            self._pose_overrides.setdefault(pid, sapien.Pose(pose.p, pose.q))
            ent.set_pose(sapien.Pose(start_p + world_shift, pose.q))

        settle(env)

        post = snapshot_poses(env)
        max_other, worst = 0.0, None
        for k, (name, p) in pre.items():
            if k == pid or k not in post:
                continue
            d = float(np.linalg.norm(post[k][1] - p))
            if d > max_other:
                max_other, worst = d, name

        record = {"commanded_world": world_shift.tolist(),
                  "max_other_actor_move_m": max_other, "max_other_actor_name": worst}
        if is_real:
            achieved = np.array(_entity(actor).get_pose().p, dtype=np.float64) - start_p
            err = float(np.linalg.norm(achieved - world_shift))
            record.update({
                "achieved_world": achieved.tolist(),
                "achieved_magnitude_cm": float(np.linalg.norm(achieved[:2]) * 100.0),
                "achieved_vs_commanded_m": err,
            })

        if max_other > OTHER_ACTOR_TOL_M:
            raise EpisodeDiscard("scene_disturbed",
                                 f"'{worst}' moved {max_other * 100:.2f} cm during settle")
        if is_real:
            tol = max(ACHIEVED_TOL_MIN_M, ACHIEVED_TOL_FRACTION * float(np.linalg.norm(world_shift)))
            if err > tol:
                raise EpisodeDiscard(
                    "shift_not_achieved",
                    f"|achieved - commanded| = {err * 100:.2f} cm > tol {tol * 100:.2f} cm")
        return record

    # ------------- obstacle selection (shift_obstacle / hide_obstacle) -------------

    def planner_visible_dynamic_obstacles(self, env) -> list:
        """Collision-list actors the planner can currently see that are not task
        targets and are physically movable."""
        targets = env._get_target_object_names()
        metrics_on = getattr(env, "enable_collision_metrics", False)
        out = []
        for info in getattr(env, "collision_list", []):
            if metrics_on and info.get("is_obstacle", False):
                continue
            actor = info["actor"]
            if not _is_dynamic(_entity(actor)):  # filters articulated furniture first
                continue
            if actor.get_name() in targets or self.is_excluded(actor):
                continue
            if all(_entity(a) is not _entity(actor) for a in out):
                out.append(actor)
        return out

    def select_corridor_obstacle(self, env):
        """Default shift_obstacle target: the planner-visible dynamic obstacle
        nearest the grasp->place corridor midpoint."""
        cands = self.planner_visible_dynamic_obstacles(env)
        if not cands:
            raise EpisodeDiscard("no_eligible_obstacle",
                                 "no planner-visible dynamic obstacle in this scene")
        mid = 0.5 * (np.asarray(env.target_obj.get_pose().p[:2])
                     + np.asarray(env.des_obj.get_pose().p[:2]))
        return min(cands, key=lambda a: float(
            np.linalg.norm(np.asarray(_entity(a).get_pose().p[:2]) - mid)))

    @staticmethod
    def corridor_shift_vector(env, actor, corridor_t: float = 0.55) -> np.ndarray:
        """Shift that puts `actor` ON the straight grasp->place transport corridor
        at fraction `corridor_t` (xy; table support kept). With the actor hidden
        from the planner the executed sweep intersects it — collision by construction."""
        a = np.asarray(env.target_obj.get_pose().p[:2], dtype=np.float64)
        b = np.asarray(env.des_obj.get_pose().p[:2], dtype=np.float64)
        target_xy = (1.0 - corridor_t) * a + corridor_t * b
        cur = np.asarray(_entity(actor).get_pose().p[:2], dtype=np.float64)
        d = target_xy - cur
        return np.array([d[0], d[1], 0.0], dtype=np.float64)

    def inject_corridor_obstacle(self, env, model_id: int = 0, clear_min_m: float = 0.10,
                                 obstacle_scale: float = 1.4, mass: float = 3.0):
        """GENERAL corridor obstacle: spawn a STATIC milk-box on the table on the
        grasp->place transport corridor (between target_obj and des_obj), at the
        point that maximises clearance from every other actor. No per-task code:
        mirrors the proven put_mouse_on_pad milk-box so shift_obstacle /
        hide_obstacle work in ANY (incl. clean) scene.

        Static => it never falls or is pushed, so it cannot disturb the scene.
        Clearance-aware => it is not dropped onto the task objects. If the corridor
        is too short to fit it (clearance < clear_min_m) raises
        EpisodeDiscard('short_corridor'). Returns (actor, info)."""
        import os
        from envs.utils.rand_create_actor import rand_create_actor

        a = np.asarray(env.target_obj.get_pose().p[:2], dtype=np.float64)
        b = np.asarray(env.des_obj.get_pose().p[:2], dtype=np.float64)
        table_z = float(env.target_obj.get_pose().p[2])
        # other actors to clear (dynamic props + the two task objects)
        others = [a, b]
        try:
            for e in env.scene.get_all_actors():
                nm = e.get_name()
                if nm in ("ground", "table", "wall"):
                    continue
                others.append(np.asarray(e.get_pose().p[:2], dtype=np.float64))
        except Exception:  # noqa: BLE001
            pass
        d = b - a
        perp = np.array([-d[1], d[0]], dtype=np.float64)
        pn = float(np.linalg.norm(perp))
        perp = perp / pn if pn > 1e-6 else np.array([1.0, 0.0])
        # search points along the corridor (and a little to either side) for the
        # one with the largest min-distance to every other actor.
        best, best_clear = None, -1.0
        for t in np.linspace(0.25, 0.75, 11):
            for off in (0.0, 0.06, -0.06, 0.10, -0.10):
                p = (1.0 - t) * a + t * b + perp * off
                clr = min(float(np.linalg.norm(p - o)) for o in others)
                if clr > best_clear:
                    best, best_clear = p, clr
        if best is None or best_clear < clear_min_m:
            raise EpisodeDiscard("short_corridor",
                                 f"no corridor point clears all actors by {clear_min_m*100:.0f} cm "
                                 f"(best {best_clear*100:.1f} cm) — target/dest too close")
        s = float(obstacle_scale)
        obstacle = rand_create_actor(
            env, modelname="038_milk-box",
            xlim=[float(best[0])], ylim=[float(best[1])],
            rotate_rand=False, qpos=[0.66, 0.66, -0.25, -0.25],
            convex=True, scale=[s, s, s], model_id=int(model_id) % 4)
        obstacle.set_mass(float(mass))  # heavy => the arm cannot just shove it aside
        env.collision_list.append({
            "actor": obstacle,
            "collision_path": (f"{os.environ['BENCH_ROOT']}/assets/objects/038_milk-box/"
                               f"collision/base{int(model_id) % 4}.glb"),
        })
        settle(env)  # let it come to rest BEFORE the rollout (clearance => disturbs nothing)
        env.update_world(exclude_obstacles=getattr(env, "enable_collision_metrics", False))
        return obstacle, {"pos": best.tolist(), "clearance_cm": round(best_clear * 100, 1)}

    def believe_displaced(self, env, actor, displacement_xy):
        """Make the planner believe `actor` is at pose + displacement (xy) while it
        physically stays put — a localisation error. update_world then routes the
        belief pose through override_pose. No physics touched."""
        import sapien.core as sapien
        ent = _entity(actor)
        pose = ent.get_pose()
        bp = np.array(pose.p, dtype=np.float64)
        bp[0] += float(displacement_xy[0])
        bp[1] += float(displacement_xy[1])
        self._pose_overrides[ent.per_scene_id] = sapien.Pose(bp.tolist(), pose.q)
        env.update_world(exclude_obstacles=getattr(env, "enable_collision_metrics", False))

    # ------------- instrumentation -------------

    def _install_frame_state_logger(self, env):
        """Log raw per-saved-frame state (every actor's world pose) aligned 1:1
        with the saved video/HDF5 frames by hooking `_take_picture`. No metrics
        here — they are derived offline (compute_progress/compute_safety). Settle
        steps never call `_take_picture`, so this stays video-aligned."""
        original = env._take_picture

        def wrapped(*a, **kw):
            try:
                poses = {}
                for e in env.scene.get_all_actors():
                    p = e.get_pose()
                    poses[e.get_name()] = [float(p.p[0]), float(p.p[1]), float(p.p[2]),
                                           float(p.q[0]), float(p.q[1]), float(p.q[2]), float(p.q[3])]
                self.frame_state.append({
                    "frame_idx": int(getattr(env, "FRAME_IDX", len(self.frame_state))),
                    "poses": poses,
                })
            except Exception:  # noqa: BLE001 — never let logging break a rollout
                pass
            return original(*a, **kw)

        env._take_picture = wrapped

    def _install_contact_logger(self, env, max_entries: int = 500):
        """Persist per-frame contacts that pass the bench collision filters."""
        original = env.check_collisions

        def wrapped(*a, **kw):
            result = original(*a, **kw)
            contacts = getattr(env, "filtered_contacts_for_log", None)
            if contacts and len(self.contact_log) < max_entries:
                self.contact_log.append({
                    "frame_idx": int(getattr(env, "FRAME_IDX", 0)),
                    "contacts": [dict(c) for c in contacts],
                })
            return result

        env.check_collisions = wrapped


# ======================================================================
# HDF5 annotation — append the label record + raw frame state to a finished
# episode HDF5 (no upstream pkl2hdf5 changes).
# ======================================================================

FLAT_KEYS = (
    "label_schema_version", "task_name", "task_config", "scene_seed",
    "executor_type", "pair_id", "pair_role", "perturbation_type", "trigger",
    "scene_kind", "perceptual_failure_class", "status", "outcome",
    "perturbed_actor", "shift_frame_idx", "instruction",
)


def write_labels_hdf5(hdf5_path, record: dict) -> bool:
    """Store the full label record as one JSON attr (lossless) + flat scalar attrs
    consumers filter on, under group `targeted_labels`. (Formerly hdf5_annot.annotate.)"""
    import h5py

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        return False
    with h5py.File(hdf5_path, "a") as f:
        g = f.require_group("targeted_labels")
        g.attrs["record_json"] = json.dumps(record)
        for k in FLAT_KEYS:
            v = record.get(k)
            if v is not None:
                g.attrs[k] = v
    return True


def write_frame_state(hdf5_path, frame_state: list, entities: dict | None = None) -> bool:
    """Write the raw per-frame actor poses logged during the rollout into the
    episode HDF5 (group `targeted_state`), aligned 1:1 with the saved frames.
    Stores ONLY raw state (float32 + gzip); metrics are computed offline from it.
    `entities` maps task roles (target/destination/…) to actor names."""
    import h5py

    hdf5_path = Path(hdf5_path)
    if not frame_state or not hdf5_path.exists():
        return False
    names = sorted({n for fr in frame_state for n in fr["poses"]})
    col = {n: i for i, n in enumerate(names)}
    T, A = len(frame_state), len(names)
    pos = np.full((T, A, 3), np.nan, dtype=np.float32)
    quat = np.full((T, A, 4), np.nan, dtype=np.float32)
    fidx = np.zeros(T, dtype=np.int32)
    for t, fr in enumerate(frame_state):
        fidx[t] = fr.get("frame_idx", t)
        for n, v in fr["poses"].items():
            a = col[n]
            pos[t, a] = v[:3]
            quat[t, a] = v[3:7]
    with h5py.File(hdf5_path, "a") as f:
        if "targeted_state" in f:
            del f["targeted_state"]
        g = f.require_group("targeted_state")
        g.create_dataset("frame_idx", data=fidx)
        g.create_dataset("actor_names", data=np.array(names, dtype=h5py.string_dtype()))
        g.create_dataset("actor_pos", data=pos, compression="gzip")
        g.create_dataset("actor_quat", data=quat, compression="gzip")
        if entities:
            g.attrs["entities_json"] = json.dumps(entities)
    return True


# ======================================================================
# Unit tests for the pure functions (sampler + labels). No sim imports.
# Run:  python benchmark/bench_envs/targeted.py
# ======================================================================

def test_sampler_deterministic_and_isolated():
    a = sample_shift_params("shift_object", 7, 3)
    np.random.seed(123)  # touching the global stream must not change draws
    np.random.rand(100)
    b = sample_shift_params("shift_object", 7, 3)
    assert a == b, "sampler must be deterministic and isolated from global np.random"


def test_sampler_grid_coverage():
    classes, bins = set(), set()
    for i in range(15):
        p = sample_shift_params("shift_target", 0, i)
        classes.add(p["target_class"])
        bins.add(p["magnitude_bin_index"])
        lo, hi = p["magnitude_bin_cm"]
        assert lo <= p["magnitude_cm"] < hi
        assert p["sign"] in (-1, 1)
    assert classes == set(PERCEPTUAL_CLASSES)
    assert bins == set(range(len(MAGNITUDE_BINS_CM)))


def test_construct_world_shift_by_class():
    depth = np.array([0.0, 1.0, 0.0])
    lateral = np.cross([0.0, 0.0, 1.0], depth)
    base = {"magnitude_cm": 4.0, "sign": 1, "mixed_angle_deg": 40.0}

    s = construct_world_shift({**base, "target_class": "depth"}, depth, lateral)
    assert np.allclose(s, [0.0, 0.04, 0.0])

    s = construct_world_shift({**base, "target_class": "lateral", "sign": -1}, depth, lateral)
    assert np.allclose(s, [0.04, 0.0, 0.0])

    s = construct_world_shift({**base, "target_class": "mixed"}, depth, lateral)
    assert abs(np.linalg.norm(s) - 0.04) < 1e-9 and s[2] == 0.0
    d = camera_frame_decomposition(s, depth, lateral)
    assert d["posthoc_dominant_axis"] == "mixed", "40-degree off-axis shift must read as mixed"


def test_decomposition_consistency():
    depth = np.array([0.0, 1.0, 0.0])
    lateral = np.cross([0.0, 0.0, 1.0], depth)
    d = camera_frame_decomposition([0.0, 0.05, 0.0], depth, lateral)
    assert abs(d["depth_cm"] - 5.0) < 1e-9 and d["posthoc_dominant_axis"] == "depth"
    d = camera_frame_decomposition([-0.03, 0.0, 0.0], depth, lateral)
    assert abs(d["lateral_cm"] - 3.0) < 1e-9 and d["posthoc_dominant_axis"] == "lateral"


def test_derive_outcome_precedence():
    assert derive_outcome(plan_success=False, success=False, is_collision=True,
                          object_moved_cm=None, place_error_cm=None)[0] == "plan_aborted"
    assert derive_outcome(plan_success=True, success=True, is_collision=False,
                          object_moved_cm=30, place_error_cm=0.2)[0] == "clean_success"
    assert derive_outcome(plan_success=True, success=True, is_collision=True,
                          object_moved_cm=30, place_error_cm=0.2)[0] == "success_with_collision"
    assert derive_outcome(plan_success=True, success=False, is_collision=False,
                          object_moved_cm=0.5, place_error_cm=25)[0] == "empty_grasp"
    assert derive_outcome(plan_success=True, success=False, is_collision=True,
                          object_moved_cm=20, place_error_cm=8)[0] == "collision_with_object"
    assert derive_outcome(plan_success=True, success=False, is_collision=False,
                          object_moved_cm=20, place_error_cm=8)[0] == "wrong_place_pose"


if __name__ == "__main__":
    _fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for _fn in _fns:
        _fn()
        print(f"PASS {_fn.__name__}")
    print(f"{len(_fns)} tests passed")
