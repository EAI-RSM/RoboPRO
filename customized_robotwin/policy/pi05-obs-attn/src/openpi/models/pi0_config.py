import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class ObstacleAttentionConfig:
    """Config for SafeVLA-style obstacle-guided attention supervision.

    The model's action->base_0_rgb attention (reduced to [depth, B, N]) is aligned
    via per-layer KL to a Gaussian heatmap built from the countertop-camera obstacle
    segmentation. A single `AttnQueryHead` with learnable role queries emits the
    obstacle/target/dest distributions from the shared reduced attention.
    """

    enabled: bool = True
    # Indices (into the transformer depth) whose attention is supervised.
    supervised_layers: tuple[int, ...] = (17,)
    # Per-supervised-layer loss weights (defaults to 0.005 each if None; 5x original 0.001).
    supervised_layer_lr: tuple[float, ...] | None = None
    # Gaussian blur std (pixels) for the GT heatmap.
    heatmap_sigma: float = 20.0
    # SigLIP patch size (So400m/14). Grid / token count are derived from
    # `_model.IMAGE_RESOLUTION` so they stay in sync with the vision encoder.
    patch_size_h: int = 14
    patch_size_w: int = 14
    # Feature width C of the single AttnQueryHead trunk/queries.
    head_dim: int = 128
    # Swap KL direction: forward D_KL(P||Q) if False, reverse if True.
    reverse_kl: bool = True
    # Optional extra role gaze heads (share the single AttnQueryHead's queries).
    # Obstacle is always supervised; target/dest add extra learnable queries when on.
    target_attn: bool = False
    dest_attn: bool = False
    target_supervised_layer_lr: tuple[float, ...] | None = None
    dest_supervised_layer_lr: tuple[float, ...] | None = None

    @property
    def role_names(self) -> tuple[str, ...]:
        """Ordered role query names for AttnQueryHead (obstacle always first)."""
        roles: tuple[str, ...] = ("obstacle",)
        if self.target_attn:
            roles += ("target",)
        if self.dest_attn:
            roles += ("dest",)
        return roles

    @property
    def num_roles(self) -> int:
        return len(self.role_names)

    @property
    def attn_grid_h(self) -> int:
        h, _ = _model.IMAGE_RESOLUTION
        if h % self.patch_size_h != 0:
            raise ValueError(
                f"IMAGE_RESOLUTION height {h} not divisible by patch_size_h {self.patch_size_h}"
            )
        return h // self.patch_size_h

    @property
    def attn_grid_w(self) -> int:
        _, w = _model.IMAGE_RESOLUTION
        if w % self.patch_size_w != 0:
            raise ValueError(
                f"IMAGE_RESOLUTION width {w} not divisible by patch_size_w {self.patch_size_w}"
            )
        return w // self.patch_size_w

    @property
    def num_image_tokens(self) -> int:
        return self.attn_grid_h * self.attn_grid_w

    def layer_lr(self, override: tuple[float, ...] | None = None) -> list[float]:
        n = len(self.supervised_layers)
        lr = override if override is not None else self.supervised_layer_lr
        if lr is None:
            return [0.005] * n
        if len(lr) != n:
            raise ValueError(f"layer_lr length {len(lr)} != num supervised layers {n}")
        return list(lr)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Optional obstacle-guided attention supervision (disabled unless provided).
    obstacle_attention: ObstacleAttentionConfig | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        obstacle_masks = {}
        if self.obstacle_attention is not None and self.obstacle_attention.enabled:
            mask_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION], jnp.float32)
            obstacle_masks = {
                "obstacle_mask": mask_spec,
                "beta_mask": mask_spec,
                "target_mask": mask_spec,
                "dest_mask": mask_spec,
            }

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                **obstacle_masks,
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
