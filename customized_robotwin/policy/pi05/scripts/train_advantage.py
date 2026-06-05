"""Advantage-conditioned pi05 training script (RECAP / π*0.6 style).

Drop-in replacement for scripts/train.py that adds:
  1. Advantage conditioning: appends "Advantage: positive" or "Advantage: negative"
     as text tokens to the language prefix.  No architecture changes required –
     pi05 already processes arbitrary-length language tokens in the prefix.
  2. Binary signal: per-episode success/failure label.
     No learned value function (equivalent to π*0.6 §Appendix F N=T case).
  3. 30% advantage-token dropout during training (enables CFG at inference).
  4. Option-4 chunk re-segmentation: when an action chunk spans a label boundary
     (e.g., a human intervention at step k flips I=False → I=True), the loss is
     masked to zero for steps [k, horizon).  Each "half" trains with a consistent
     label via the sliding window without any extra data copying.

Dataset requirements
--------------------
See openpi/training/advantage_data_loader.py for the full list of requirements.
Short version:
  - Your LeRobot dataset must have an ``advantage`` column (float32, 0.0 or 1.0).
  - Your config's repack_transforms must include ``"advantage": "advantage"``.
  - Your config's action_sequence_keys must be ``("action", "advantage")``.
  See ``pi05_aloha_advantage`` in config.py for a ready-made template.

Labeling convention
-------------------
  BC / expert-correction demonstrations  → advantage = 1.0 (all timesteps)
  Successful rollouts (above threshold)  → advantage = 1.0 (all timesteps)
  Failed rollouts (below threshold)      → advantage = 0.0 (all timesteps)
  Intervention episodes: autonomous part → advantage = 0.0
                         correction part → advantage = 1.0  (boundary splitting kicks in)

Usage
-----
    cd customized_robotwin/policy/pi05
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_advantage.py \\
        pi05_aloha_advantage \\
        --exp-name my_advantage_run \\
        --overwrite

To resume from checkpoint:
    uv run scripts/train_advantage.py pi05_aloha_advantage \\
        --exp-name my_advantage_run --resume

Advantage dropout (default 0.3) can be changed via ADVANTAGE_DROPOUT env var:
    ADVANTAGE_DROPOUT=0.0 uv run scripts/train_advantage.py pi05_aloha_advantage ...
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from openpi.training.advantage_data_loader import create_advantage_data_loader

# 30% dropout matches the π*0.6 paper; override via ADVANTAGE_DROPOUT env var.
_ADVANTAGE_DROPOUT: float = float(os.environ.get("ADVANTAGE_DROPOUT", "0.3"))


# ---------------------------------------------------------------------------
# Infrastructure (unchanged from train.py)
# ---------------------------------------------------------------------------

def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader, params_shape):
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict({
        k: v
        for k, v in traverse_util.flatten_dict(loaded_params).items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    })


@at.typecheck
def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# ---------------------------------------------------------------------------
# Advantage-conditioned train step
# ---------------------------------------------------------------------------

@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, at.Bool[at.Array, "*b ah"]],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single training step with advantage-conditioned masked loss.

    The ``action_mask`` in ``batch`` is a bool array of shape ``[B, action_horizon]``.
    Steps with mask=False are excluded from the loss (they lie beyond a within-chunk
    advantage-label boundary).  The per-sample loss is a weighted mean over valid steps.
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    observation, actions, action_mask = batch

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        # per_step_loss: [B, ah] — mean squared error per action timestep
        per_step_loss = model.compute_loss(rng, observation, actions, train=True)

        # Masked mean: zero out steps beyond the advantage-label boundary,
        # then divide by the number of valid steps per sample.
        mask_f = action_mask.astype(per_step_loss.dtype)          # [B, ah] float
        valid_counts = jnp.maximum(mask_f.sum(axis=-1), 1.0)      # [B]
        sample_loss = (per_step_loss * mask_f).sum(axis=-1) / valid_counts  # [B]
        return sample_loss.mean()

    train_rng = jax.random.fold_in(rng, state.step)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by device count {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Advantage data loader (yields (Observation, actions, action_mask) triples)
    data_loader = create_advantage_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
        advantage_dropout=_ADVANTAGE_DROPOUT,
        seed=config.seed,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized advantage data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # Batch sharding: (Observation pytree, actions, action_mask) all sharded along batch dim
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced, step=step)
            infos = []

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            save_step = step + 1 if step == config.num_train_steps - 1 else step
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, save_step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
