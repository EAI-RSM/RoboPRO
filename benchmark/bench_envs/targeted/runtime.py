"""TargetedRuntime — per-episode perturbation state attached as `env.targeted`.

This replaces the PoC's instance monkeypatching with first-class hooks:
`Bench_base_task.update_world` consults `is_excluded` / `override_pose` on
every call (persistent planner blindness — design principle 8, required for
tasks that re-call update_world mid-episode), and `Bench_base_task.grasp_actor`
fires `notify("after_grasp_plan", env)` once the grasp poses are locked.

When `env.targeted` is absent or None, the bench behaves exactly as stock.
"""
import numpy as np
import sapien.core as sapien

SETTLE_STEPS = 25
# Principle 5(a): every non-perturbed actor must move less than this during settle.
OTHER_ACTOR_TOL_M = 0.005
# Principle 5(b): achieved displacement within max(25% of commanded, 5 mm) of commanded.
ACHIEVED_TOL_FRACTION = 0.25
ACHIEVED_TOL_MIN_M = 0.005


class EpisodeDiscard(Exception):
    """Raised when the integrity guard rejects an episode. Reason is one of
    'scene_disturbed' / 'shift_not_achieved' / 'no_eligible_obstacle'."""

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
    snap = {}
    for e in env.scene.get_all_actors():
        snap[e.per_scene_id] = (e.get_name(), np.array(e.get_pose().p, dtype=np.float64))
    return snap


def settle(env, steps: int = SETTLE_STEPS):
    # Settle steps are never recorded (no _take_picture) — principle 7.
    for _ in range(steps):
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
        self._pose_overrides: dict[int, sapien.Pose] = {}
        self._armed: dict | None = None
        self.contact_log: list = []
        # Raw per-saved-frame simulation state (all actor poses), aligned 1:1
        # with the saved video/HDF5 frames. Logged by the _take_picture hook
        # (attach()); metrics are computed OFFLINE from this (targeted/metrics.py).
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
        """Persistently remove `actor` from the planner's collision world (the
        physics body is untouched) and rebuild CuRobo's world. Every later
        update_world call keeps the exclusion."""
        self._excluded_ids.add(_entity(actor).per_scene_id)
        env.update_world(exclude_obstacles=getattr(env, "enable_collision_metrics", False))
        return 1

    def apply_shift(self, env, actor, world_shift) -> dict:
        """Teleport `actor` by `world_shift`, settle, run the two-sided
        integrity guard (principle 5). Call with actor=None / zero shift for
        the baseline twin: identical settle, no shift. After a real shift the
        planner keeps seeing the pre-shift pose via the override map.

        Raises EpisodeDiscard('scene_disturbed' | 'shift_not_achieved')."""
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

        record = {
            "commanded_world": world_shift.tolist(),
            "max_other_actor_move_m": max_other,
            "max_other_actor_name": worst,
        }
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
        """Collision-list actors the planner can currently see that are not
        task targets and are physically movable."""
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
        """Shift that puts `actor` ON the straight grasp->place transport
        corridor at fraction `corridor_t` (xy; table support kept). With the
        actor hidden from the planner the executed sweep intersects it —
        collision by construction."""
        a = np.asarray(env.target_obj.get_pose().p[:2], dtype=np.float64)
        b = np.asarray(env.des_obj.get_pose().p[:2], dtype=np.float64)
        target_xy = (1.0 - corridor_t) * a + corridor_t * b
        cur = np.asarray(_entity(actor).get_pose().p[:2], dtype=np.float64)
        d = target_xy - cur
        return np.array([d[0], d[1], 0.0], dtype=np.float64)

    # ------------- instrumentation -------------

    def _install_frame_state_logger(self, env):
        """Log raw per-saved-frame simulation state — every actor's world pose —
        aligned 1:1 with the saved video/HDF5 frames by hooking `_take_picture`
        (the per-frame save). NO metrics are computed here: task progress,
        clearances and reward/value signals are derived OFFLINE from this state
        (targeted/metrics.py), so the sim loop stays cheap and adding a metric
        needs no re-simulation. Settle steps never call `_take_picture`, so this
        stays video-aligned."""
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
        """Persist per-frame contacts that pass the bench collision filters
        (cheap form of design component 0.3)."""
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
