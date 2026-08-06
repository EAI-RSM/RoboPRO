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


def grid_to_image(
    dist_1d: np.ndarray, grid: int, size: int = 224, *, normalize: bool = True
) -> np.ndarray:
    """[N] distribution -> [size, size] nearest-upsampled heatmap.

    Default max-normalizes so small spatial changes in mass remain visible (the
    original train-viz look). Pass ``normalize=False`` for absolute probabilities.
    """
    hm = np.asarray(dist_1d, dtype=np.float64).reshape(grid, grid)
    hm = np.kron(hm, np.ones((size // grid, size // grid)))
    if normalize:
        m = hm.max()
        return hm / m if m > 0 else hm
    return hm


def attention_stats(dist_1d: np.ndarray, *, eps: float = 1e-12) -> dict[str, float]:
    """Diagnostics for a patch distribution: H_norm, max, KL to uniform."""
    p = np.asarray(dist_1d, dtype=np.float64).reshape(-1)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0 or not np.isfinite(total):
        return {"H_norm": float("nan"), "max": float("nan"), "KL_u": float("nan")}
    p = p / total
    n = p.shape[0]
    log_n = np.log(n)
    entropy = float(-np.sum(p * np.log(p + eps)))
    h_norm = float(entropy / log_n) if log_n > 0 else 0.0
    return {
        "H_norm": h_norm,
        "max": float(p.max()),
        "KL_u": float(log_n - entropy),
    }


def _stats_title(prefix: str, stats: dict[str, float]) -> str:
    return (
        f"{prefix}\n"
        f"H_norm={stats['H_norm']:.2f} max={stats['max']:.3f} KL_u={stats['KL_u']:.2f}"
    )


def _fig_to_rgb(fig: plt.Figure, *, dpi: int = 80) -> np.ndarray:
    """Render a matplotlib figure to an HxWx3 uint8 RGB array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    import PIL.Image

    return np.asarray(PIL.Image.open(buf).convert("RGB"))


def _wandb_jpeg(
    rgb: np.ndarray,
    *,
    caption: str | None = None,
    max_side: int = 1280,
    quality: int = 75,
) -> wandb.Image:
    """Compress RGB to JPEG for W&B — large PNGs often 408 on slow uplinks."""
    import PIL.Image

    im = PIL.Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
    w, h = im.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), PIL.Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return wandb.Image(PIL.Image.open(buf), caption=caption)


def _mask_nonempty(mask: np.ndarray | None, *, eps: float = 0.0) -> np.ndarray:
    """Per-batch-item bool: mask has any mass. Missing mask -> all False."""
    if mask is None:
        return np.zeros((0,), dtype=bool)
    m = np.asarray(mask)
    if m.ndim < 3:
        return np.asarray([float(np.sum(m)) > eps], dtype=bool)
    return np.sum(m.reshape(m.shape[0], -1), axis=-1) > eps


def select_cluttered_indices(
    observation: _model.Observation,
    num_samples: int,
    role_names: tuple[str, ...] = ("obstacle", "target", "dest"),
) -> list[int]:
    """Indices with non-empty masks for clutter + each enabled role.

    Always requires non-empty ``obstacle_mask`` and ``beta_mask``. Target/dest
    masks are required only when those roles appear in ``role_names``.
    Scans the batch in order; returns up to ``num_samples`` matches (may be fewer).
    """
    batch_size = int(np.asarray(jax.device_get(observation.state)).shape[0])
    roles = set(role_names)

    def _as_mask(attr: str) -> np.ndarray | None:
        raw = getattr(observation, attr, None)
        if raw is None:
            return None
        return np.asarray(jax.device_get(raw))

    obs_m = _as_mask("obstacle_mask")
    beta_m = _as_mask("beta_mask")
    if obs_m is None or beta_m is None:
        return []

    ok = _mask_nonempty(obs_m) & _mask_nonempty(beta_m)

    if "target" in roles:
        tgt_m = _as_mask("target_mask")
        if tgt_m is None:
            return []
        ok = ok & _mask_nonempty(tgt_m)
    if "dest" in roles:
        dest_m = _as_mask("dest_mask")
        if dest_m is None:
            return []
        ok = ok & _mask_nonempty(dest_m)

    idxs = [int(i) for i in np.flatnonzero(ok) if i < batch_size]
    return idxs[:num_samples]


def _take_indices(
    observation: _model.Observation,
    actions: _model.Actions,
    indices: list[int] | np.ndarray,
) -> tuple[_model.Observation, _model.Actions]:
    """Gather batch items at ``indices`` (host-side pytree slice)."""
    idx = np.asarray(indices, dtype=np.int32)
    if idx.size == 0:
        raise ValueError("indices must be non-empty")

    def _take(x):
        if x is None:
            return None
        if hasattr(x, "shape") and len(getattr(x, "shape", ())) > 0:
            return x[idx]
        return x

    return jax.tree.map(_take, observation), actions[idx]


def _role_masks_for_sample(
    observation: _model.Observation, sample_i: int, role_names: tuple[str, ...]
) -> dict[str, np.ndarray | None]:
    """Pixel masks for one sample; obstacle is multiplied by beta when present."""
    out: dict[str, np.ndarray | None] = {}
    beta = None
    if observation.beta_mask is not None:
        beta = np.asarray(jax.device_get(observation.beta_mask))[sample_i]
    for role in role_names:
        attr = f"{role}_mask"
        raw = getattr(observation, attr, None)
        if raw is None:
            out[role] = None
            continue
        m = np.asarray(jax.device_get(raw))[sample_i]
        if role == "obstacle" and beta is not None:
            m = m * beta
        out[role] = m
    return out


def render_layer_role_panel(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray | None],
    gts: dict[str, np.ndarray | None],
    role_probs: np.ndarray,
    *,
    role_names: tuple[str, ...],
    layer_ids: list[int],
    grid: int,
) -> np.ndarray:
    """One sample: rows=supervised layers; per-role column groups.

    Column order (roles in ``role_names`` order)::
    ``mask_obs | GT_obs | pred_obs | mask_tgt | GT_tgt | pred_tgt | ...``
    so each GT sits next to its matching model map. GT/pred share one absolute
    ``vmax`` per row. ``role_probs`` is [S, R, N] for this sample.
    """
    n_roles = len(role_names)
    n_rows = len(layer_ids)
    # Layout: for each role -> [mask | gt | pred]
    n_cols = 3 * n_roles
    n_tok = grid * grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = np.asarray([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for row, layer_id in enumerate(layer_ids):
        gt_imgs = []
        pred_imgs = []
        for r, role in enumerate(role_names):
            gt = gts.get(role)
            pred = role_probs[row, r]
            gt_imgs.append(grid_to_image(gt, grid, normalize=False) if gt is not None else None)
            pred_imgs.append(grid_to_image(pred, grid, normalize=False))

        vmax = float(1.0 / n_tok)
        for im in gt_imgs + pred_imgs:
            if im is not None:
                vmax = max(vmax, float(np.max(im)), 1e-12)

        for r, role in enumerate(role_names):
            c0 = 3 * r
            ax_m = axes[row, c0]
            ax_m.imshow(rgb)
            m = masks.get(role)
            if m is not None:
                im0 = ax_m.imshow(m, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
                fig.colorbar(im0, ax=ax_m, fraction=0.046, pad=0.04)
            ax_m.set_title(f"L{layer_id}: mask {role}" if r == 0 else f"mask {role}", fontsize=8)
            ax_m.axis("off")

            ax_g = axes[row, c0 + 1]
            ax_g.imshow(rgb)
            gt = gts.get(role)
            gt_stats = attention_stats(gt) if gt is not None else None
            if gt_imgs[r] is not None:
                im1 = ax_g.imshow(gt_imgs[r], cmap="jet", alpha=0.5, vmin=0.0, vmax=vmax)
                fig.colorbar(im1, ax=ax_g, fraction=0.046, pad=0.04)
                ax_g.set_title(_stats_title(f"GT {role}", gt_stats), fontsize=7)
            else:
                ax_g.set_title(f"GT {role}", fontsize=8)
            ax_g.axis("off")

            ax_p = axes[row, c0 + 2]
            ax_p.imshow(rgb)
            pred_stats = attention_stats(role_probs[row, r])
            im2 = ax_p.imshow(pred_imgs[r], cmap="jet", alpha=0.5, vmin=0.0, vmax=vmax)
            fig.colorbar(im2, ax=ax_p, fraction=0.046, pad=0.04)
            ax_p.set_title(_stats_title(f"pred {role}", pred_stats), fontsize=7)
            ax_p.axis("off")

    fig.tight_layout()
    img = _fig_to_rgb(fig, dpi=72)
    plt.close(fig)
    return img


def render_attention_scene_panels(
    observation: _model.Observation,
    preds: dict,
    role_names: tuple[str, ...],
    *,
    grid: int,
    supervised_layers: tuple[int, ...] | list[int] | None = None,
) -> list[np.ndarray]:
    """Per-sample unified panels (all roles, all supervised layers).

    Expects ``observation`` / ``preds`` already sliced to the samples to render.
    """
    role_probs = np.asarray(jax.device_get(preds["role_probs"]))  # [S, B, R, N]
    rgb = np.asarray(jax.device_get(observation.images["base_0_rgb"]))
    rgb = np.clip((rgb + 1.0) / 2.0, 0.0, 1.0)
    n_stack, n_batch = role_probs.shape[0], role_probs.shape[1]
    if supervised_layers is None:
        layer_ids = list(range(n_stack))
    else:
        layer_ids = [
            int(supervised_layers[s]) if s < len(supervised_layers) else s for s in range(n_stack)
        ]

    gts: dict[str, np.ndarray | None] = {}
    for role in role_names:
        key = f"gt_{role}"
        gts[role] = np.asarray(jax.device_get(preds[key])) if key in preds else None

    panels: list[np.ndarray] = []
    for i in range(n_batch):
        masks = _role_masks_for_sample(observation, i, role_names)
        gts_i = {role: (gts[role][i] if gts[role] is not None else None) for role in role_names}
        panels.append(
            render_layer_role_panel(
                rgb[i],
                masks,
                gts_i,
                role_probs[:, i],
                role_names=role_names,
                layer_ids=layer_ids,
                grid=grid,
            )
        )
    return panels


def _panel_style_attn_frame(
    rgb: np.ndarray,
    dist_1d: np.ndarray,
    *,
    grid: int,
    vmax: float,
    size: int = 224,
) -> np.ndarray:
    """One frame matching a panel pred cell: RGB + jet overlay (alpha=0.5, absolute vmax)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    hm = grid_to_image(dist_1d, grid, size=size, normalize=False)

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.imshow(rgb)
    ax.imshow(hm, cmap="jet", alpha=0.5, vmin=0.0, vmax=vmax)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, pad_inches=0)
    buf.seek(0)
    import PIL.Image

    frame = np.asarray(PIL.Image.open(buf).convert("RGB").resize((size, size), PIL.Image.BILINEAR))
    plt.close(fig)
    return frame


def render_attention_gif_frames(
    rgb: np.ndarray,
    role_probs: np.ndarray,
    *,
    role_names: tuple[str, ...],
    layer_ids: list[int],
    grid: int,
    size: int = 224,
    gt_dists: np.ndarray | None = None,
) -> np.ndarray:
    """Stack of uint8 RGB frames [S, H, W, 3] matching panel pred-column appearance.

    ``role_probs`` is [S, R, N]. Each frame uses the same absolute jet overlay as
    ``render_layer_role_panel`` preds: ``alpha=0.5``, ``vmin=0``,
    ``vmax = max(1/N, GT, pred)`` for that layer. Shows obstacle if present, else
    the first role.
    """
    n_tok = grid * grid
    role_probs = np.asarray(role_probs, dtype=np.float64)
    role_idx = role_names.index("obstacle") if "obstacle" in role_names else 0
    gt_max = float(np.max(gt_dists)) if gt_dists is not None and np.size(gt_dists) else 0.0

    frames = []
    for s, layer_id in enumerate(layer_ids):
        pred = role_probs[s, role_idx]
        vmax = float(max(1.0 / n_tok, gt_max, float(np.max(pred)), 1e-12))
        frames.append(
            _panel_style_attn_frame(rgb, pred, grid=grid, vmax=vmax, size=size)
        )
        _ = layer_id
    return np.stack(frames, axis=0)


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


def _plot_ee_path_3d(ax, xyz: np.ndarray, *, color: str, linestyle: str, label: str) -> None:
    """Plot one EE trajectory with start (o) / end (s) markers."""
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linestyle=linestyle, label=label, linewidth=1.5)
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color=color, marker="o", s=36, depthshade=False)
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color=color, marker="s", s=36, depthshade=False)


def render_ee_trajectories(
    gt_l: np.ndarray,
    gt_r: np.ndarray,
    pred_l: np.ndarray,
    pred_r: np.ndarray,
    *,
    sample_idx: int = 0,
) -> np.ndarray:
    """XYZ signed error panels + 3D GT/pred L/R paths. Returns HxWx3 RGB."""
    T = gt_l.shape[0]
    t = np.arange(T)
    err_l = pred_l - gt_l
    err_r = pred_r - gt_r
    fig = plt.figure(figsize=(14, 8))

    # 3x2: rows X/Y/Z, cols left/right — signed pred-GT error vs horizon
    axes_labels = ("X", "Y", "Z")
    for dim, lab in enumerate(axes_labels):
        ax_l = fig.add_subplot(3, 3, dim * 3 + 1)
        ax_l.plot(t, err_l[:, dim], "r-", linewidth=1.5)
        ax_l.axhline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_l.set_ylabel(f"L Δ{lab} (m)")
        ax_l.grid(True, alpha=0.3)
        mae_l = float(np.mean(np.abs(err_l[:, dim])))
        if dim == 0:
            ax_l.set_title(f"Left EE  |mean|={mae_l:.4f}m")
        else:
            ax_l.set_title(f"|mean|={mae_l:.4f}m", fontsize=9)
        if dim == 2:
            ax_l.set_xlabel("t")

        ax_r = fig.add_subplot(3, 3, dim * 3 + 2)
        ax_r.plot(t, err_r[:, dim], "r-", linewidth=1.5)
        ax_r.axhline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_r.set_ylabel(f"R Δ{lab} (m)")
        ax_r.grid(True, alpha=0.3)
        mae_r = float(np.mean(np.abs(err_r[:, dim])))
        if dim == 0:
            ax_r.set_title(f"Right EE  |mean|={mae_r:.4f}m")
        else:
            ax_r.set_title(f"|mean|={mae_r:.4f}m", fontsize=9)
        if dim == 2:
            ax_r.set_xlabel("t")

    ax3d = fig.add_subplot(1, 3, 3, projection="3d")
    _plot_ee_path_3d(ax3d, gt_l, color="tab:blue", linestyle="-", label="GT L")
    _plot_ee_path_3d(ax3d, pred_l, color="tab:cyan", linestyle="--", label="pred L")
    _plot_ee_path_3d(ax3d, gt_r, color="tab:green", linestyle="-", label="GT R")
    _plot_ee_path_3d(ax3d, pred_r, color="tab:pink", linestyle="--", label="pred R")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(f"EE paths (sample {sample_idx})\nstart=o  end=s")
    ax3d.legend(fontsize=7)

    fig.tight_layout()
    img = _fig_to_rgb(fig)
    plt.close(fig)
    return img


def _slice_batch(
    observation: _model.Observation, actions: _model.Actions, n: int
) -> tuple[_model.Observation, _model.Actions]:
    """Take the first n samples from a training batch (host-side pytree slice)."""
    return _take_indices(observation, actions, list(range(n)))


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
    """Run predict_attention + sample_actions and log images/scalars to W&B.

    Selects cluttered samples (non-empty beta + obstacle, plus masks for each
    enabled role). If none match, skips viz for this step (no silent fall-back
    to clean scenes).
    """
    observation, actions = batch
    model.eval()
    rng_attn, rng_act = jax.random.split(rng)
    log_payload: dict = {}

    oa = getattr(model, "obstacle_attention", None)
    role_names = oa.role_names if oa is not None and oa.enabled else ("obstacle",)
    clutter_idxs = select_cluttered_indices(observation, num_samples, role_names=role_names)
    if not clutter_idxs:
        _logger.warning(
            "Train visualization skipped at step %s: no cluttered samples "
            "(need non-empty beta, obstacle, and masks for enabled roles %s) in batch",
            step,
            role_names,
        )
        return

    if len(clutter_idxs) < num_samples:
        _logger.warning(
            "Train visualization at step %s: only %d/%d cluttered samples in batch",
            step,
            len(clutter_idxs),
            num_samples,
        )

    observation, actions = _take_indices(observation, actions, clutter_idxs)
    n = len(clutter_idxs)

    if oa is not None and oa.enabled:
        preds = model.predict_attention(rng_attn, observation, train=False)
        role_names = oa.role_names
        layer_ids = list(oa.supervised_layers)
        panels = render_attention_scene_panels(
            observation,
            preds,
            role_names,
            grid=oa.attn_grid_h,
            supervised_layers=oa.supervised_layers,
        )
        # One JPEG per sample (no list + no duplicate keys): large PNG lists were
        # timing out on W&B GCS upload (408) while smaller actions/ee images synced.
        for i, im in enumerate(panels):
            log_payload[f"attn/scene_{i}"] = _wandb_jpeg(
                im, caption=f"clutter s{i} (batch[{clutter_idxs[i]}])"
            )

        role_probs = np.asarray(jax.device_get(preds["role_probs"]))  # [S, B, R, N]
        for s in range(role_probs.shape[0]):
            layer_id = int(layer_ids[s]) if s < len(layer_ids) else s
            for r_idx, role in enumerate(role_names):
                sample_stats = [attention_stats(role_probs[s, i, r_idx]) for i in range(n)]
                for key in ("H_norm", "max", "KL_u"):
                    vals = [st[key] for st in sample_stats if np.isfinite(st[key])]
                    if vals:
                        log_payload[f"attn_stats/{role}/L{layer_id}/{key}"] = float(np.mean(vals))

    # --- Cartesian EE trajectories (same cluttered samples) ---
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
            _wandb_jpeg(
                render_ee_trajectories(gt_l, gt_r, pred_l, pred_r, sample_idx=i),
                caption=f"ee clutter s{i}",
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
    _logger.info(
        "Logged train visualizations at step %s (%d cluttered samples, idxs=%s)",
        step,
        n,
        clutter_idxs,
    )
