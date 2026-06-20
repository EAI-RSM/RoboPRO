"""Online-relabel collision-aware BC fine-tuning of pi05.

Standard pi05 flow-matching BC, except at each step the action-chunk TARGET is
perturbed online — contact-gated smooth push (online_relabel.make_online_perturb,
same logic as the offline smooth relabel) — when the conditioning frame is NOT a
recorded contact. Observations are never modified; only the target chunk is corrected,
so there is no image/action inconsistency (the reason offline relabeling was wrong).

    safe_actions = perturb(actions, obstacle_points, contact_chunk)   # stop-gradient target
    L = flow_bc_loss(observation, safe_actions)

Gate (per the design): contact_chunk[:,0] is the conditioning frame's RECORDED contact.
  * conditioning frame in contact  -> don't perturb (plain BC on the original chunk)
  * conditioning frame clean        -> push the chunk's recorded-contact steps to a
    depth-scaled standoff, smooth in time, last step free (not anchored to original).

Config: pi05_aloha_online_relabel (Pi0Config(pi05=True), full BC fine-tune).
Data:   da3_collision_am_succ with obstacle_points + windowed collision_per_step.

Usage (pi05 uv env):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_online_relabel.py \\
        --config-name=pi05_aloha_online_relabel --exp-name=online_relabel_run1 \\
        --fk-basis-path=$SCRATCH/datasets/fk_basis.npz \\
        --weight-loader-path=$SCRATCH/checkpoints/.../params --overwrite
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import platform

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
from flax.training import common_utils
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.weight_loaders as weight_loaders
import openpi.training.utils as training_utils
from openpi.training.adjoint_data_loader import create_online_relabel_data_loader
from openpi.training.collision_proximity import build_joint_affine
from openpi.training.online_relabel import make_online_perturb


@dataclasses.dataclass
class OnlineRelabelArgs:
    config_name: str = tyro.MISSING
    exp_name: str = tyro.MISSING
    fk_basis_path: str = tyro.MISSING        # fk_basis.npz for FK / sphere decomposition

    # online-perturbation hyperparameters (smooth push, same as the offline relabel)
    push: float = 0.02                       # push contacting spheres this far beyond measured (m)
    contact_clearance: float = 0.05          # at a contact step, spheres under this are active (m)
    max_dq: float = 0.30                     # per-arm joint displacement cap (rad)
    inner_steps: int = 30                    # inner push-optimization steps per batch
    inner_lr: float = 0.05

    # overrides applied to the named config (so it can be retargeted from CLI/sbatch)
    data_repo_id: str | None = None
    weight_loader_path: str | None = None
    batch_size: int | None = None
    num_train_steps: int | None = None
    fsdp_devices: int | None = None

    wandb_enabled: bool = True
    overwrite: bool = False
    resume: bool = False

    def __post_init__(self):
        if self.resume and self.overwrite:
            raise ValueError("--resume and --overwrite are mutually exclusive.")


def init_logging():
    logging.getLogger().setLevel(logging.INFO)


def _load_weights_and_validate(loader, params_shape):
    loaded = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict({
        k: v for k, v in traverse_util.flatten_dict(loaded).items()
        if not isinstance(v, jax.ShapeDtypeStruct)
    })


@at.typecheck
def init_train_state(config: _config.TrainConfig, init_rng: at.KeyArrayLike,
                     mesh: jax.sharding.Mesh, *, resume: bool):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter,
                                     lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0, params=params, model_def=nnx.graphdef(model), tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    if resume:
        return train_state_shape, state_sharding
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(init, donate_argnums=(1,), in_shardings=replicated,
                          out_shardings=state_sharding)(init_rng, partial_params)
    return train_state, state_sharding


@at.typecheck
def train_step(config: _config.TrainConfig, rng: at.KeyArrayLike,
               state: training_utils.TrainState, batch: tuple, *, perturb_fn):
    observation, actions, obstacle_points, contact_chunk = batch
    # Online target relabel (stop-gradient; pure geometry on the dataset chunk).
    safe_actions = perturb_fn(actions, obstacle_points, contact_chunk)

    def loss_fn(model):
        train_rng = jax.random.fold_in(rng, state.step)
        per = model.compute_loss(train_rng, observation, safe_actions, train=True)
        return per.mean(), {}

    model = nnx.merge(state.model_def, state.params)
    model.train()
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, _), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model),
                                    opt_state=new_opt_state)
    # fraction of samples whose conditioning frame was clean (i.e. perturbed)
    perturbed = (contact_chunk[:, 0] < 0.5).mean()
    info = {"loss/bc": loss, "grad_norm": optax.global_norm(grads), "frac_perturbed": perturbed}
    return new_state, info


def main(args: OnlineRelabelArgs):
    init_logging()
    logging.info(f"Running on: {platform.node()}  args={args}")
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, exp_name=args.exp_name, overwrite=args.overwrite)
    if args.data_repo_id is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=args.data_repo_id))
    if args.weight_loader_path is not None:
        config = dataclasses.replace(config, weight_loader=weight_loaders.CheckpointWeightLoader(args.weight_loader_path))
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.num_train_steps is not None:
        config = dataclasses.replace(config, num_train_steps=args.num_train_steps)
    if args.fsdp_devices is not None:
        config = dataclasses.replace(config, fsdp_devices=args.fsdp_devices)

    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    mesh = sharding.make_mesh(jax.device_count())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = create_online_relabel_data_loader(config, sharding=data_sharding, shuffle=True,
                                                    num_workers=config.num_workers, seed=config.seed)

    # online perturbation on the NORMALIZED action chunk (uses the dataset's action norm affine)
    data_config = data_loader.data_config()
    scale, bias = build_joint_affine(data_config.norm_stats["actions"], data_config.use_quantile_norm)
    perturb_fn = make_online_perturb(
        args.fk_basis_path, scale, bias,
        push=args.push, contact_clearance=args.contact_clearance, max_dq=args.max_dq,
        inner_steps=args.inner_steps, inner_lr=args.inner_lr)

    init_rng = jax.random.key(config.seed)
    train_state, state_sharding = init_train_state(config, init_rng, mesh, resume=args.resume)
    if args.resume:
        train_state = _checkpoints.restore_state(train_state, ckpt_dir)

    wandb.init(name=args.exp_name, project=config.project_name,
               config={**dataclasses.asdict(args), **dataclasses.asdict(config)},
               mode="online" if args.wandb_enabled else "disabled",
               resume="must" if args.resume else None)

    pstep = jax.jit(
        functools.partial(train_step, config, perturb_fn=perturb_fn),
        in_shardings=(None, state_sharding, data_sharding),
        out_shardings=(state_sharding, None),
    )

    data_iter = iter(data_loader)
    pbar = tqdm.tqdm(range(int(train_state.step), config.num_train_steps),
                     initial=int(train_state.step), total=config.num_train_steps, dynamic_ncols=True)
    step_rng = jax.random.key(0)
    infos: list[dict] = []
    for step in pbar:
        batch = next(data_iter)
        step_rng, train_rng = jax.random.split(step_rng)
        with sharding.set_mesh(mesh):
            train_state, info = pstep(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        if (step % config.save_interval == 0 and step > int(train_state.step) - 1) or \
                step == config.num_train_steps - 1:
            _checkpoints.save_state(train_state, ckpt_dir)
    logging.info("Online-relabel BC training complete.")


if __name__ == "__main__":
    main(tyro.cli(OnlineRelabelArgs))
