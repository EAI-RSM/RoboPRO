"""Orbax compatibility for restoring legacy RoboPRO OpenPI checkpoints."""

from __future__ import annotations

import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import traverse_util


def install_modern_orbax_restore(model_module, model_config) -> None:
    """Teach the historical OpenPI loader to restore with modern Orbax metadata."""
    if getattr(model_module.restore_params, "_robopro_modern_orbax", False):
        return

    def restore_params(
        params_path: pathlib.Path | str,
        *,
        restore_type=jax.Array,
        dtype: jnp.dtype | None = None,
        sharding: jax.sharding.Sharding | None = None,
    ):
        params_path = (
            pathlib.Path(params_path).resolve()
            if not str(params_path).startswith("gs://")
            else params_path
        )
        if restore_type is jax.Array and sharding is None:
            mesh = jax.sharding.Mesh(jax.devices(), ("x",))
            sharding = jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()
            )

        abstract_model = nnx.eval_shape(model_config.create, jax.random.key(0))
        _, abstract_state = nnx.split(abstract_model)
        # This checkpoint was saved from an older NNX State, whose Orbax
        # leaves are wrapped as {"value": array}.
        wrapped_params = jax.tree.map(
            lambda leaf: {"value": leaf}, abstract_state.to_pure_dict()
        )
        item = {"params": wrapped_params}
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(
                sharding=sharding, restore_type=restore_type, dtype=dtype
            ),
            item,
        )
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored = checkpointer.restore(
                params_path,
                ocp.args.PyTreeRestore(item=item, restore_args=restore_args),
            )["params"]

        flat = traverse_util.flatten_dict(restored)
        if flat and all(path[-1] == "value" for path in flat):
            flat = {path[:-1]: value for path, value in flat.items()}
        return traverse_util.unflatten_dict(flat)

    restore_params._robopro_modern_orbax = True
    model_module.restore_params = restore_params
