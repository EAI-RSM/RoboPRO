"""Host-side W&B visualizations for obstacle-attention training.

Renders GT vs predicted attention panels and GT vs predicted end-effector
trajectories in Cartesian (robot base) space. Intended to run outside the
jitted train step.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3d projection

import openpi.models.model as _model
import openpi.policies.aloha_policy as aloha_policy
import openpi.shared.normalize as _normalize
import openpi.transforms as _transforms

# beta_geometry lives under scripts/ (not an installed package).
_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from beta_geometry import EEF_LINKS, fk_link_origins, load_urdf  # noqa: E402

_DELTA_MASK = _transforms.make_bool_mask(6, -1, 6, -1)
_logger = logging.getLogger(__name__)


def grid_to_image(dist_1d: np.ndarray, grid: int, size: int = 224) -> np.ndarray:
    """[N] distribution -> [size, size] nearest-upsampled, max-normalized for display."""
    hm = dist_1d.reshape(grid, grid)
    hm = np.kron(hm, np.ones((size // grid, size // grid)))
    m = hm.max()
    return hm / m if m > 0 else hm


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Render a matplotlib figure to an HxWx3 uint8 RGB array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    import PIL.Image

    return np.asarray(PIL.Image.open(buf).convert("RGB"))


def render_attention_panel(
    rgb: np.ndarray,
    replacement_mask: np.ndarray | None,
    gt_dist: np.ndarray | None,
    pred_dist: np.ndarray,
    *,
    role: str,
    grid: int,
    layer: int,
) -> np.ndarray:
    """One 4-column attention panel (HxWx3 RGB) for a single sample/role."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("base_0_rgb")
    axes[1].imshow(rgb)
    if replacement_mask is not None:
        axes[1].imshow(replacement_mask, cmap="jet", alpha=0.5)
    axes[1].set_title(f"replacement {role} mask")
    axes[2].imshow(rgb)
    if gt_dist is not None:
        axes[2].imshow(grid_to_image(gt_dist, grid), cmap="jet", alpha=0.5)
    axes[2].set_title(f"patchified GT {role}")
    axes[3].imshow(rgb)
    axes[3].imshow(grid_to_image(pred_dist, grid), cmap="jet", alpha=0.5)
    axes[3].set_title(f"model attn (layer {layer})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img


def render_attention_panels(
    observation: _model.Observation,
    preds: dict,
    role_names: tuple[str, ...],
    *,
    num_samples: int,
    grid: int,
) -> dict[str, list[np.ndarray]]:
    """Render attention panels for each enabled role.

    Returns dict role -> list of RGB images (one per sample).
    """
    role_probs = np.asarray(jax.device_get(preds["role_probs"]))  # [S, B, R, N]
    rgb = np.asarray(jax.device_get(observation.images["base_0_rgb"]))
    rgb = np.clip((rgb + 1.0) / 2.0, 0.0, 1.0)
    layer = role_probs.shape[0] - 1
    n = min(num_samples, rgb.shape[0])

    out: dict[str, list[np.ndarray]] = {}
    for role in role_names:
        r = role_names.index(role)
        gt_key = f"gt_{role}"
        gt = np.asarray(jax.device_get(preds[gt_key])) if gt_key in preds else None
        replacement = getattr(observation, f"{role}_mask", None)
        replacement = np.asarray(jax.device_get(replacement)) if replacement is not None else None
        if role == "obstacle" and replacement is not None and observation.beta_mask is not None:
            replacement = replacement * np.asarray(jax.device_get(observation.beta_mask))

        panels = []
        for i in range(n):
            panels.append(
                render_attention_panel(
                    rgb[i],
                    replacement[i] if replacement is not None else None,
                    gt[i] if gt is not None else None,
                    role_probs[layer, i, r],
                    role=role,
                    grid=grid,
                    layer=layer,
                )
            )
        out[role] = panels
    return out


def actions_to_ee_xyz(
    state: np.ndarray,
    actions: np.ndarray,
    *,
    adapt_to_pi: bool,
    use_delta: bool,
    norm_stats: dict[str, _normalize.NormStats] | None,
    use_quantiles: bool,
    urdf=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one sample's (state, action chunk) to left/right EE XYZ [T, 3].

    ``state`` is [S], ``actions`` is [T, A] in the model training space
    (normalized deltas when use_delta and norm_stats are set).
    """
    state = np.asarray(state, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)

    if norm_stats is not None:
        unnorm = _transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)
        data = unnorm({"state": state.astype(np.float32), "actions": actions.astype(np.float32)})
        state = np.asarray(data["state"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)

    if use_delta:
        abs_fn = _transforms.AbsoluteActions(mask=_DELTA_MASK)
        data = abs_fn({"state": state.astype(np.float32), "actions": actions.astype(np.float32)})
        actions = np.asarray(data["actions"], dtype=np.float64)

    # Model/pi space -> Aloha joint convention for the URDF (first 14 dims only).
    actions = np.asarray(actions[..., :14], dtype=np.float32)
    actions = aloha_policy._encode_actions(actions, adapt_to_pi=adapt_to_pi)
    actions = np.asarray(actions, dtype=np.float64)

    if urdf is None:
        urdf = load_urdf()

    left_ee = EEF_LINKS["left"]
    right_ee = EEF_LINKS["right"]
    left_xyz = []
    right_xyz = []
    for t in range(actions.shape[0]):
        la = actions[t, 0:6]
        ra = actions[t, 7:13]
        origins = fk_link_origins(urdf, la, ra, [left_ee, right_ee])
        left_xyz.append(origins[left_ee])
        right_xyz.append(origins[right_ee])
    return np.asarray(left_xyz), np.asarray(right_xyz)


def render_ee_trajectories(
    gt_l: np.ndarray,
    gt_r: np.ndarray,
    pred_l: np.ndarray,
    pred_r: np.ndarray,
    *,
    sample_idx: int = 0,
) -> np.ndarray:
    """Multi-panel GT vs pred EE paths (XYZ time series + 3D). Returns HxWx3 RGB."""
    T = gt_l.shape[0]
    t = np.arange(T)
    fig = plt.figure(figsize=(14, 8))

    # 3x2: rows X/Y/Z, cols left/right
    axes_labels = ("X", "Y", "Z")
    for dim, lab in enumerate(axes_labels):
        ax_l = fig.add_subplot(3, 3, dim * 3 + 1)
        ax_l.plot(t, gt_l[:, dim], "b-", label="GT", linewidth=1.5)
        ax_l.plot(t, pred_l[:, dim], "r--", label="pred", linewidth=1.5)
        ax_l.set_ylabel(f"L {lab} (m)")
        ax_l.grid(True, alpha=0.3)
        if dim == 0:
            ax_l.set_title("Left EE")
            ax_l.legend(fontsize=8)
        if dim == 2:
            ax_l.set_xlabel("t")

        ax_r = fig.add_subplot(3, 3, dim * 3 + 2)
        ax_r.plot(t, gt_r[:, dim], "b-", label="GT", linewidth=1.5)
        ax_r.plot(t, pred_r[:, dim], "r--", label="pred", linewidth=1.5)
        ax_r.set_ylabel(f"R {lab} (m)")
        ax_r.grid(True, alpha=0.3)
        if dim == 0:
            ax_r.set_title("Right EE")
            ax_r.legend(fontsize=8)
        if dim == 2:
            ax_r.set_xlabel("t")

    ax3d = fig.add_subplot(1, 3, 3, projection="3d")
    ax3d.plot(gt_l[:, 0], gt_l[:, 1], gt_l[:, 2], "b-", label="GT L")
    ax3d.plot(pred_l[:, 0], pred_l[:, 1], pred_l[:, 2], "c--", label="pred L")
    ax3d.plot(gt_r[:, 0], gt_r[:, 1], gt_r[:, 2], "g-", label="GT R")
    ax3d.plot(pred_r[:, 0], pred_r[:, 1], pred_r[:, 2], "m--", label="pred R")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(f"EE paths (sample {sample_idx})")
    ax3d.legend(fontsize=7)

    fig.tight_layout()
    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img


def _slice_batch(
    observation: _model.Observation, actions: _model.Actions, n: int
) -> tuple[_model.Observation, _model.Actions]:
    """Take the first n samples from a training batch (host-side pytree slice)."""

    def _take(x):
        if x is None:
            return None
        if hasattr(x, "shape") and len(getattr(x, "shape", ())) > 0:
            return x[:n]
        return x

    return jax.tree.map(_take, observation), actions[:n]


def log_train_visualizations(
    model: _model.BaseModel,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    rng: jax.Array,
    step: int,
    num_samples: int = 2,
    adapt_to_pi: bool = True,
    use_delta: bool = True,
    norm_stats: dict[str, _normalize.NormStats] | None = None,
    use_quantiles: bool = True,
) -> None:
    """Run predict_attention + sample_actions and log images/scalars to W&B."""
    observation, actions = batch
    batch_size = int(np.asarray(jax.device_get(observation.state)).shape[0])
    n = min(num_samples, batch_size)
    observation, actions = _slice_batch(observation, actions, n)

    model.eval()

    rng_attn, rng_act = jax.random.split(rng)
    log_payload: dict = {}

    # --- Attention panels ---
    oa = getattr(model, "obstacle_attention", None)
    if oa is not None and oa.enabled:
        preds = model.predict_attention(rng_attn, observation, train=False)
        panels = render_attention_panels(
            observation,
            preds,
            oa.role_names,
            num_samples=n,
            grid=oa.attn_grid_h,
        )
        for role, imgs in panels.items():
            log_payload[f"attn/{role}"] = [wandb.Image(im, caption=f"{role} s{i}") for i, im in enumerate(imgs)]

    # --- Cartesian EE trajectories ---
    pred_actions = model.sample_actions(rng_act, observation)
    pred_actions_np = np.asarray(jax.device_get(pred_actions))
    gt_actions_np = np.asarray(jax.device_get(actions))
    state_np = np.asarray(jax.device_get(observation.state))

    urdf = load_urdf()
    ee_images = []
    ee_l2s = []
    for i in range(n):
        gt_l, gt_r = actions_to_ee_xyz(
            state_np[i],
            gt_actions_np[i],
            adapt_to_pi=adapt_to_pi,
            use_delta=use_delta,
            norm_stats=norm_stats,
            use_quantiles=use_quantiles,
            urdf=urdf,
        )
        pred_l, pred_r = actions_to_ee_xyz(
            state_np[i],
            pred_actions_np[i],
            adapt_to_pi=adapt_to_pi,
            use_delta=use_delta,
            norm_stats=norm_stats,
            use_quantiles=use_quantiles,
            urdf=urdf,
        )
        ee_images.append(
            wandb.Image(
                render_ee_trajectories(gt_l, gt_r, pred_l, pred_r, sample_idx=i),
                caption=f"ee sample {i}",
            )
        )
        ee_l2s.append(
            0.5
            * (
                np.linalg.norm(gt_l - pred_l, axis=-1).mean()
                + np.linalg.norm(gt_r - pred_r, axis=-1).mean()
            )
        )

    log_payload["actions/ee_cartesian"] = ee_images
    if ee_l2s:
        log_payload["actions/ee_l2"] = float(np.mean(ee_l2s))

    wandb.log(log_payload, step=step)
    _logger.info("Logged train visualizations at step %s (%d samples)", step, n)
