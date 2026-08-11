"""Explicit semantic roles for stock tasks used by the geometric metric study."""

from dataclasses import dataclass
from pathlib import Path

from .obstacles import scene_obstacle_entries


SUPPORTED_TASK = "put_cup_on_coaster"


@dataclass(frozen=True)
class ActorRole:
    actor: object
    name: str
    model_id: str
    collision_path: str


@dataclass(frozen=True)
class TaskRoles:
    target: ActorRole
    destination: ActorRole
    obstacles: tuple


def _actor_name(actor):
    name = actor.get_name()
    if not name:
        raise ValueError("task-role actor has no name")
    return str(name)


def _asset_collision_path(name, model_id):
    import os

    bench_root = os.environ.get("BENCH_ROOT")
    if not bench_root:
        raise RuntimeError("BENCH_ROOT is required to resolve task-role assets")
    return str(
        Path(bench_root) / "assets" / "objects" / str(name)
        / "collision" / f"base{model_id}.glb"
    )


def _require_mesh(path, role):
    if not Path(path).is_file() and not Path(path).is_dir():
        raise FileNotFoundError(f"missing {role} collision mesh: {path}")


def resolve_task_roles(env, task_name):
    """Resolve the one selected stock task without guessing from collision ordering."""
    if task_name != SUPPORTED_TASK:
        raise ValueError(
            f"no task-role adapter for {task_name!r}; supported task is {SUPPORTED_TASK!r}"
        )
    for attr in ("target_obj", "target_name", "target_id", "des_obj", "des_obj_id"):
        if getattr(env, attr, None) is None:
            raise ValueError(f"{task_name} scene is missing required role attribute env.{attr}")

    target = env.target_obj
    destination = env.des_obj
    if target is destination:
        raise ValueError("target and destination resolved to the same actor")
    target_name = _actor_name(target)
    destination_name = _actor_name(destination)
    if target_name != str(env.target_name):
        raise ValueError(
            f"target identity mismatch: actor={target_name!r}, configured={env.target_name!r}"
        )

    target_path = _asset_collision_path(target_name, env.target_id)
    destination_path = _asset_collision_path(destination_name, env.des_obj_id)
    _require_mesh(target_path, "target")
    _require_mesh(destination_path, "destination")

    obstacles = []
    for actor, path in scene_obstacle_entries(env, "all"):
        _require_mesh(path, f"obstacle {_actor_name(actor)!r}")
        obstacles.append(ActorRole(actor, _actor_name(actor), "", str(path)))

    return TaskRoles(
        target=ActorRole(target, target_name, str(env.target_id), target_path),
        destination=ActorRole(
            destination, destination_name, str(env.des_obj_id), destination_path
        ),
        obstacles=tuple(obstacles),
    )
