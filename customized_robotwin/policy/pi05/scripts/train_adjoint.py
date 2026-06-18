"""Adjoint-matching fine-tuning of pi05 toward differentiable proximity.

Plain Adjoint Matching (reward-only; no learned Q): a LoRA "fast" field on the
action expert is trained so the controlled flow's terminal samples maximize the
proximity reward (= −collision_cost), behavior-regularized to the frozen base.

    L = lam_adj · adjoint_loss(fast; ∇_a proximity)   [+ lam_bc · L_flow(fast)]

Pieces:
  - reward / ∇_a r : collision_proximity.make_action_reward  (FK + scene points)
  - AM core        : adjoint_matching.adjoint_loss          (validated on toy)
  - slow vs fast   : same params; slow = LoRA leaves zeroed, fast = full
  - τ-adapter      : v_τ(x,t) = −v_pi05(x, 1−t)  (pi05 t=1 noise → 0 data)
  - data           : adjoint_data_loader  (obs, actions, obstacle_points[B,M,3])

Config: pi05_aloha_adjoint (Pi0Config(pi05=True, action_expert_variant=
"gemma_300m_lora") → trainable_filter = LoRA params only).

⚠ This wires the 3B model and cannot be validated off-cluster; the reward and
the AM core ARE validated independently.  First-run checks to watch: the
velocity-wrapper leading-dim handling (unroll [B,H,A] vs loss [N,B,H,A]) and
that nnx.state(PathRegex(".*lora.*")) selects exactly the LoRA leaves.

Usage (pi05 uv env):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_adjoint.py \\
        pi05_aloha_adjoint --exp-name am_run1 \\
        --fk-basis-path ../../collision_dataset/<task>/cluttered/fk_basis.npz \\
        --lam-adj 1.0 --lam-bc 1.0 --inv-temp 1.0 --flow-steps 4 --overwrite
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import platform
from typing import Any

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

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from openpi.training.adjoint_data_loader import create_adjoint_data_loader
from openpi.training.adjoint_matching import AdjointConfig, adjoint_loss
from openpi.training.collision_proximity import build_joint_affine, make_action_reward

_LORA_REGEX = ".*lora.*"


@dataclasses.dataclass
class AdjointTrainArgs:
    config_name: str = tyro.MISSING
    exp_name: str = tyro.MISSING
    fk_basis_path: str = tyro.MISSING        # fk_basis.npz for FK / sphere decomposition

    lam_adj: float = 1.0                     # adjoint-matching loss weight
    lam_bc: float = 1.0                      # BC flow-loss weight (keep task competence; 0 = pure AM)
    inv_temp: float = 1.0                    # reward-gradient strength (1/KL-temperature)
    flow_steps: int = 4                      # SDE steps for the adjoint unroll (cost driver)
    margin: float = 0.03                     # proximity hinge margin (m)

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


def _make_velocity(model, observation_pp, kv_cache, prefix_mask):
    """τ-convention velocity callable v(x, t) for the adjoint core, backed by the
    cached prefix.  Handles x of rank 3 ([B,H,A], unroll step) and rank 4
    ([N,B,H,A], loss over flow steps).  pi05 t-convention is t=1 noise → 0 data,
    so v_τ(x, τ) = −v_pi05(x, 1−τ).
    """
    def v(x, t):
        if x.ndim == 3:                                   # [B,H,A] single unroll step
            tau = t[:, 0, 0]                              # [B]
            return -model.compute_velocity_cached(observation_pp, kv_cache, prefix_mask, x, 1.0 - tau)
        # [N,B,H,A] stacked over flow steps (N static at trace time)
        N = x.shape[0]
        outs = []
        for n in range(N):
            tau = t[n, :, 0, 0]                           # [B]
            outs.append(-model.compute_velocity_cached(observation_pp, kv_cache, prefix_mask, x[n], 1.0 - tau))
        return jnp.stack(outs, axis=0)
    return v


@at.typecheck
def train_step(config: _config.TrainConfig, rng: at.KeyArrayLike,
               state: training_utils.TrainState, batch: tuple, *,
               reward_fn, adj_cfg: AdjointConfig, lam_adj: float, lam_bc: float):
    observation, actions, obstacle_points = batch
    B = actions.shape[0]
    action_dim = config.model.action_dim
    horizon = config.model.action_horizon

    def loss_fn(model):
        train_rng = jax.random.fold_in(rng, state.step)
        bc_rng, unroll_rng = jax.random.split(train_rng)

        # ── base (slow): same model with LoRA leaves zeroed ───────────────
        base_model = nnx.merge(state.model_def, nnx.state(model))
        lora_state = nnx.state(base_model, nnx_utils.PathRegex(_LORA_REGEX))
        nnx.update(base_model, jax.tree.map(jnp.zeros_like, lora_state))

        # ── prefix KV (base-only path; shared by slow & fast) ─────────────
        obs_pp, kv_cache, prefix_mask = model.prefill_prefix_kv(observation, train=False)
        kv_cache = jax.lax.stop_gradient(kv_cache)

        v_fast = _make_velocity(model, obs_pp, kv_cache, prefix_mask)
        v_slow = _make_velocity(base_model, obs_pp, kv_cache, prefix_mask)

        # terminal adjoint: ∇_a reward at the generated normalized action chunk
        def reward_grad_fn(x):
            return jax.grad(lambda xx: reward_fn(xx, obstacle_points).sum())(x)

        x0 = jax.random.normal(unroll_rng, (B, horizon, action_dim))
        L_adj = adjoint_loss(v_slow, v_fast, reward_grad_fn, x0, unroll_rng, adj_cfg)

        # ── optional BC flow loss on the fast (LoRA) model ────────────────
        if lam_bc > 0.0:
            per = model.compute_loss(bc_rng, observation, actions, train=True)
            L_bc = per.mean()
        else:
            L_bc = jnp.zeros(())

        return lam_adj * L_adj + lam_bc * L_bc, {"loss/adj": L_adj, "loss/bc": L_bc}

    model = nnx.merge(state.model_def, state.params)
    model.train()
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model),
                                    opt_state=new_opt_state)
    info = {"loss/total": loss, "loss/adj": aux["loss/adj"], "loss/bc": aux["loss/bc"],
            "grad_norm": optax.global_norm(grads)}
    return new_state, info


def main(args: AdjointTrainArgs):
    init_logging()
    logging.info(f"Running on: {platform.node()}  args={args}")
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, exp_name=args.exp_name, overwrite=args.overwrite)

    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    mesh = sharding.make_mesh(jax.device_count())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    data_loader = create_adjoint_data_loader(config, sharding=data_sharding, shuffle=True,
                                             num_workers=config.num_workers, seed=config.seed)

    # reward (= −proximity cost) as a function of the NORMALIZED action chunk
    data_config = data_loader.data_config()
    scale, bias = build_joint_affine(data_config.norm_stats["actions"], data_config.use_quantile_norm)
    reward_fn = make_action_reward(args.fk_basis_path, scale, bias, margin=args.margin)
    adj_cfg = AdjointConfig(flow_steps=args.flow_steps, inv_temp=args.inv_temp, residual=False)

    init_rng = jax.random.key(config.seed)
    train_state, state_sharding = init_train_state(config, init_rng, mesh, resume=args.resume)
    if args.resume:
        train_state = _checkpoints.restore_state(train_state, ckpt_dir)

    wandb.init(name=args.exp_name, project=config.project_name,
               config={**dataclasses.asdict(args), **dataclasses.asdict(config)},
               mode="online" if args.wandb_enabled else "disabled",
               resume="must" if args.resume else None)

    pstep = jax.jit(
        functools.partial(train_step, config, reward_fn=reward_fn, adj_cfg=adj_cfg,
                          lam_adj=args.lam_adj, lam_bc=args.lam_bc),
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
    logging.info("Adjoint-matching training complete.")


if __name__ == "__main__":
    main(tyro.cli(AdjointTrainArgs))
