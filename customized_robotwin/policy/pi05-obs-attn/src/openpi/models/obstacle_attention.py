"""JAX/NNX port of SafeVLA's obstacle-guided attention regularization.

Ported from safevla/models/obstacle_attention.py. The GT target is a Gaussian
heatmap over the obstacle segmentation, sum-pooled onto the SigLIP token grid and
normalized to a distribution; the model attention (action-query -> base_0_rgb-key,
already reduced over heads/actions to [depth, B, N] in gemma.Attention) is aligned
to it via per-layer KL.

Design change vs. SafeVLA: instead of three separate `AttnUpsampler` instances,
a single `AttnQueryHead` with a learnable query table emits all role maps
(obstacle / target / destination) from the shared reduced model attention.
"""

from math import ceil, isqrt

from flax import nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at

__all__ = [
    "generate_heatmap",
    "patchify_heatmap",
    "kl_div",
    "compute_attn_loss",
    "AttnQueryHead",
]


def _gaussian_kernel1d(sigma: float, kernel_size: int) -> jax.Array:
    ax = jnp.arange(kernel_size, dtype=jnp.float32) - (kernel_size - 1) / 2.0
    k = jnp.exp(-0.5 * (ax / sigma) ** 2)
    return k / jnp.sum(k)


def _gaussian_blur(x: jax.Array, sigma: float, kernel_size: int) -> jax.Array:
    """Separable Gaussian blur for [B, H, W] with zero padding (fp32)."""
    k = _gaussian_kernel1d(sigma, kernel_size)
    pad = kernel_size // 2
    x = x[:, None].astype(jnp.float32)  # [B, 1, H, W]
    kh = k[None, None, :, None]  # [O=1, I=1, kh, 1]
    kw = k[None, None, None, :]  # [O=1, I=1, 1, kw]
    dn = jax.lax.conv_dimension_numbers(x.shape, kh.shape, ("NCHW", "OIHW", "NCHW"))
    x = jax.lax.conv_general_dilated(x, kh, (1, 1), [(pad, pad), (0, 0)], dimension_numbers=dn)
    x = jax.lax.conv_general_dilated(x, kw, (1, 1), [(0, 0), (pad, pad)], dimension_numbers=dn)
    return x[:, 0]


@at.typecheck
def generate_heatmap(
    mask: at.Float[at.Array, "b h w"],
    sigma: float = 20.0,
) -> at.Float[at.Array, "b h w"]:
    """Gaussian heatmap from a (possibly weighted) mask, normalized to sum=1 per item.

    Empty masks (sum 0) stay all-zeros so downstream KL can skip them.
    """
    kernel_size = 2 * ceil(2 * sigma) + 1
    heatmap = _gaussian_blur(mask, sigma, kernel_size)
    total = jnp.sum(heatmap, axis=(1, 2), keepdims=True)
    return heatmap / jnp.clip(total, a_min=1e-8, a_max=None) * (total > 0)


@at.typecheck
def patchify_heatmap(
    hm: at.Float[at.Array, "b h w"],
    patch_size_h: int = 14,
    patch_size_w: int = 14,
) -> at.Float[at.Array, "b gh gw"]:
    """Sum-pool the heatmap into disjoint patches and renormalize to sum=1 per item."""
    b, h, w = hm.shape
    if h % patch_size_h != 0 or w % patch_size_w != 0:
        raise ValueError("Patch dimensions must divide heatmap dimensions")
    gh, gw = h // patch_size_h, w // patch_size_w
    pooled = hm.reshape(b, gh, patch_size_h, gw, patch_size_w).sum(axis=(2, 4))  # [B, gh, gw]
    total = jnp.sum(pooled, axis=(1, 2), keepdims=True)
    return pooled / jnp.clip(total, a_min=1e-8, a_max=None) * (total > 0)


@at.typecheck
def kl_div(
    P: at.Float[at.Array, "b n"],  # noqa: N803
    Q: at.Float[at.Array, "b n"],  # noqa: N803
    eps: float = 1e-8,
    reverse: bool = False,
) -> at.Float[at.Array, ""]:
    """D_KL(P || Q) averaged over batch items with non-empty target (fp32).

    Both distributions are eps-smoothed + renormalized to strictly positive so the
    KL is finite in both directions (see SafeVLA notes). Items whose target has zero
    mass before smoothing are excluded from the mean.
    """
    P = P.astype(jnp.float32)
    Q = Q.astype(jnp.float32)
    valid = jnp.sum(P, axis=-1) > 0  # [B]

    P = P + eps
    P = P / jnp.sum(P, axis=-1, keepdims=True)
    Q = Q + eps
    Q = Q / jnp.sum(Q, axis=-1, keepdims=True)

    target = P if not reverse else Q
    model_pred = Q if not reverse else P
    # target * (log target - log model_pred), summed over tokens.
    per_sample = jnp.sum(target * (jnp.log(target) - jnp.log(model_pred)), axis=-1)  # [B]

    valid_f = valid.astype(jnp.float32)
    denom = jnp.clip(jnp.sum(valid_f), a_min=1.0, a_max=None)
    return jnp.sum(per_sample * valid_f) / denom


def compute_attn_loss(
    model_attns: at.Float[at.Array, "s b n"],
    gt_attns: at.Float[at.Array, "b n"],
    supervised_layer_lr: list[float],
    reverse_kl: bool = False,
    prefix: str = "supervised",
) -> dict[str, jax.Array]:
    """Per-supervised-layer weighted KL between model attention and the GT heatmap.

    `model_attns` is [num supervised layers, B, N] already reduced over heads/actions
    (or, for the query-head variant, one role channel of [S, B, N]).
    """
    loss_dict: dict[str, jax.Array] = {}
    for layer_idx in range(model_attns.shape[0]):
        attn = model_attns[layer_idx]  # [B, N]
        attn = attn / jnp.clip(jnp.sum(attn, axis=-1, keepdims=True), a_min=1e-8, a_max=None)
        kl_loss = kl_div(gt_attns, attn, reverse=reverse_kl)
        loss_dict[f"{prefix}_kl_loss_layer_{layer_idx}"] = supervised_layer_lr[layer_idx] * kl_loss
    return loss_dict


class AttnQueryHead(nnx.Module):
    """Single multi-query read-out head shared across all supervised layers.

    Maps the reduced model attention over an (in_h x in_w) token grid to one
    distribution per role via a learnable query table. A light conv stem lifts the
    1-channel attention grid to a per-position feature map; each role query then
    scores every position (scaled dot-product) and softmaxes to a distribution.

    Input : [S, B, N]  (S supervised layers, N = in_h*in_w)
    Output: [S, B, R, N]  (R roles, fixed order [obstacle, target, dest])
    """

    def __init__(
        self,
        in_h: int,
        in_w: int,
        num_roles: int,
        feat_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.in_h = in_h
        self.in_w = in_w
        self.num_roles = num_roles
        self.feat_dim = feat_dim
        self.stem = nnx.Conv(
            in_features=1,
            out_features=feat_dim,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        # Learnable query table [R, C], one query per role.
        key = rngs.params()
        self.queries = nnx.Param(jax.random.normal(key, (num_roles, feat_dim)) * (feat_dim**-0.5))

    def __call__(self, attn: at.Float[at.Array, "s b n"]) -> at.Float[at.Array, "s b r n"]:
        s, b, n = attn.shape
        if n != self.in_h * self.in_w:
            raise ValueError(f"attn last dim {n} != in_h*in_w {self.in_h * self.in_w}")
        x = attn.reshape(s * b, self.in_h, self.in_w, 1)
        f = self.stem(x)  # [S*B, in_h, in_w, C]
        f = f.reshape(s * b, n, self.feat_dim)  # [S*B, N, C]
        q = self.queries.value  # [R, C]
        logits = jnp.einsum("mnc,rc->mrn", f, q) * (self.feat_dim**-0.5)  # [S*B, R, N]
        probs = jax.nn.softmax(logits, axis=-1)
        return probs.reshape(s, b, self.num_roles, n)


def infer_grid(num_tokens: int) -> tuple[int, int]:
    """Best-effort square grid for a token count (e.g. 256 -> 16x16)."""
    g = isqrt(num_tokens)
    if g * g != num_tokens:
        raise ValueError(f"num_tokens {num_tokens} is not a perfect square")
    return g, g
