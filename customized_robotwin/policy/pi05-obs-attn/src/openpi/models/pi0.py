import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import obstacle_attention as _obstacle_attention
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.obstacle_attention = config.obstacle_attention
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # Obstacle-guided attention read-out token counts (base_0_rgb is the first image,
        # so its tokens are the first `num_image_tokens` keys; action tokens are the last
        # `action_horizon` queries). 0 disables the read-out slice.
        _oa_enabled = config.obstacle_attention is not None and config.obstacle_attention.enabled
        _num_image_tokens = config.obstacle_attention.num_image_tokens if _oa_enabled else 0
        _num_action_tokens = config.action_horizon if _oa_enabled else 0
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                num_action_tokens=_num_action_tokens,
                num_image_tokens=_num_image_tokens,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Single multi-query read-out head shared across layers. Role count follows
        # config flags (obstacle always; +target/+dest when enabled).
        if _oa_enabled:
            oa = config.obstacle_attention
            self.attn_query_head = _obstacle_attention.AttnQueryHead(
                in_h=oa.attn_grid_h,
                in_w=oa.attn_grid_w,
                num_roles=oa.num_roles,
                feat_dim=oa.head_dim,
                rngs=rngs,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # Embed images in IMAGE_KEYS order so base_0_rgb is always the first block of
        # keys (first `num_image_tokens` = IMAGE_RESOLUTION / patch). Obstacle-attention
        # slicing in gemma assumes that; do not iterate obs.images dict order.
        image_names = [n for n in _model.IMAGE_KEYS if n in obs.images]
        image_names += [n for n in obs.images if n not in image_names]
        for name in image_names:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Float[at.Array, ""]]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        oa_enabled = (
            self.obstacle_attention is not None
            and self.obstacle_attention.enabled
            and observation.obstacle_mask is not None
        )

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _, attn_stack = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            return_attn=oa_enabled,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        mse = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        aux_kl: dict[str, at.Float[at.Array, ""]] = {}
        if oa_enabled:
            aux_kl = self._obstacle_attn_loss(observation, attn_stack)

        return mse, aux_kl

    def _obstacle_attn_loss(
        self, observation: _model.Observation, attn_stack: at.Float[at.Array, "l b n"]
    ) -> dict[str, at.Float[at.Array, ""]]:
        """Per-supervised-layer KL between the query-head role maps and GT heatmaps."""
        oa = self.obstacle_attention
        role_index = {name: i for i, name in enumerate(oa.role_names)}
        # Select supervised layers -> [S, B, N]; run the shared head -> [S, B, R, N].
        model_attn = jnp.stack([attn_stack[i] for i in oa.supervised_layers], axis=0)
        role_probs = self.attn_query_head(model_attn)  # [S, B, R, N]

        def gt(mask: at.Float[at.Array, "b h w"]) -> at.Float[at.Array, "b n"]:
            hm = _obstacle_attention.generate_heatmap(mask, sigma=oa.heatmap_sigma)
            patched = _obstacle_attention.patchify_heatmap(hm, oa.patch_size_h, oa.patch_size_w)
            return patched.reshape(patched.shape[0], -1)

        # Obstacle gaze (always): beta_t applied to the mask BEFORE the blur.
        beta = observation.beta_mask if observation.beta_mask is not None else 1.0
        obstacle_gt = gt(observation.obstacle_mask * beta)
        aux: dict[str, at.Float[at.Array, ""]] = dict(
            _obstacle_attention.compute_attn_loss(
                role_probs[:, :, role_index["obstacle"], :],
                obstacle_gt,
                oa.layer_lr(),
                oa.reverse_kl,
                prefix="obstacle",
            )
        )
        if oa.target_attn and observation.target_mask is not None:
            aux.update(
                _obstacle_attention.compute_attn_loss(
                    role_probs[:, :, role_index["target"], :],
                    gt(observation.target_mask),
                    oa.layer_lr(oa.target_supervised_layer_lr),
                    oa.reverse_kl,
                    prefix="target",
                )
            )
        if oa.dest_attn and observation.dest_mask is not None:
            aux.update(
                _obstacle_attention.compute_attn_loss(
                    role_probs[:, :, role_index["dest"], :],
                    gt(observation.dest_mask),
                    oa.layer_lr(oa.dest_supervised_layer_lr),
                    oa.reverse_kl,
                    prefix="dest",
                )
            )
        return aux

    def predict_attention(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, train: bool = False
    ) -> dict[str, at.Array]:
        """Eval/visualization helper: returns the reduced model attention, the
        query-head role maps, and the GT heatmaps (obstacle/target/dest) for the
        supervised layers. Uses a mid-trajectory flow timestep with zero actions.
        """
        if self.obstacle_attention is None or not self.obstacle_attention.enabled:
            raise ValueError("obstacle_attention is not enabled on this model")
        oa = self.obstacle_attention
        observation = _model.preprocess_observation(rng, observation, train=train)

        batch = observation.state.shape[0]
        x_t = jnp.zeros((batch, self.action_horizon, self.action_dim), dtype=jnp.float32)
        time = jnp.full((batch,), 0.5, dtype=jnp.float32)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        _, _, attn_stack = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            return_attn=True,
        )
        model_attn = jnp.stack([attn_stack[i] for i in oa.supervised_layers], axis=0)  # [S, B, N]
        role_probs = self.attn_query_head(model_attn)  # [S, B, R, N]

        def gt(mask):
            hm = _obstacle_attention.generate_heatmap(mask, sigma=oa.heatmap_sigma)
            patched = _obstacle_attention.patchify_heatmap(hm, oa.patch_size_h, oa.patch_size_w)
            return patched.reshape(patched.shape[0], -1)

        out = {"model_attn": model_attn, "role_probs": role_probs}
        if observation.obstacle_mask is not None:
            beta = observation.beta_mask if observation.beta_mask is not None else 1.0
            out["gt_obstacle"] = gt(observation.obstacle_mask * beta)
        if observation.target_mask is not None:
            out["gt_target"] = gt(observation.target_mask)
        if observation.dest_mask is not None:
            out["gt_dest"] = gt(observation.dest_mask)
        return out

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
