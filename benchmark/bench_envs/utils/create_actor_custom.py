import sapien.core as sapien
import numpy as np
from pathlib import Path
import os
from .actor_utils_custom import Simple_Actor


def create_glb_actor(
    scene,
    pose: sapien.Pose,
    model_name: str,
    scale=(1.0, 1.0, 1.0),
    convex: bool = False,
    is_static: bool = False,
    mass: float = 0.01,
) -> Simple_Actor:
    """
    Create a SAPIEN actor from a single GLB file and wrap it in Simple_Actor.
    Loads the GLB from assets/objects/{model_name}.

    Args:
        scene: SAPIEN scene
        pose: Initial pose of the actor
        model_name: Model name; GLB is loaded from assets/objects/{model_name}/
        scale: Scale (tuple or list [x, y, z]). Default (1, 1, 1).
        convex: If True, use convex decomposition for collision; else nonconvex.
        is_static: If True, create static actor; else dynamic.
        mass: Mass for dynamic actor (used by Simple_Actor).
    Returns:
        Simple_Actor wrapping the built SAPIEN actor.
    """
    root = Path(os.environ.get("BENCH_ROOT", "."))
    model_dir = root / "assets" / "objects" / model_name
    model_dir = Path(model_dir)

    # Prefer base.glb at the root, otherwise fall back to the first GLB found
    # anywhere under the model directory because some benchmark assets are
    # nested one level deeper.
    glb_path = model_dir / "base.glb"
    if not glb_path.exists():
        glb_files = list(model_dir.rglob("*.glb"))
        if not glb_files:
            raise FileNotFoundError(f"No GLB file found in {model_dir}")
        glb_path = sorted(glb_files)[0]

    if isinstance(scale, (int, float)):
        scale = [float(scale), float(scale), float(scale)]

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex:
        builder.add_multiple_convex_collisions_from_file(filename=str(glb_path), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(filename=str(glb_path), scale=scale)
    builder.add_visual_from_file(filename=str(glb_path), scale=scale)

    actor = builder.build()
    actor.set_pose(pose)
    actor.set_name(model_name)

    return Simple_Actor(actor, mass=mass, scale=scale)
