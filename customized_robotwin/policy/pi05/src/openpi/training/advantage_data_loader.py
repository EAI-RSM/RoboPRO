"""Advantage-conditioned data loader for pi05 training.

Implements RECAP-style advantage conditioning (π*0.6 paper §3, Appendix F):
  - Appends "Advantage: positive" or "Advantage: negative" to the language prompt.
  - Uses binary per-episode success as the advantage signal (MC return, no value fn).
  - Handles chunk re-segmentation at label boundaries (Option 4):
      When the advantage label changes within a 50-step window (e.g., after a
      human intervention at step k), the loss is masked to zero for steps [k, 50).
      The model trains on a "short" chunk of length k with a consistent label.
  - 30% dropout on the advantage token (enables CFG-style inference).

Dataset requirements
--------------------
Your LeRobot dataset must have an ``advantage`` column (float32, values 0.0 or 1.0)
with one entry per timestep.

Your training DataConfig must satisfy two conditions:

1. ``action_sequence_keys`` must include ``"advantage"`` so the loader requests a
   50-step delta window for it:
       action_sequence_keys = ("action", "advantage")

2. The ``repack_transforms`` ``RepackTransform`` mapping must preserve the key:
       RepackTransform({..., "advantage": "advantage"})
   (RepackTransform drops all keys not listed; "advantage" would be silently lost
   without this entry.)

See ``pi05_aloha_advantage`` in the advantage training script for a complete example.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import typing
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AdvantageConditionTransform(_transforms.DataTransformFn):
    """Injects advantage conditioning into the prompt and builds the action loss mask.

    Must be inserted into the transform pipeline AFTER ``Normalize`` but BEFORE
    ``TokenizePrompt`` (i.e., between data_transforms and model_transforms) so
    that the modified prompt string is tokenized correctly.

    Input keys consumed / modified
    --------------------------------
    ``data["advantage"]``  : np.ndarray [action_horizon] or [action_horizon, 1]
        Per-timestep advantage labels for this sample's sliding window.
        Values are 0.0 (negative) or 1.0 (positive).  Removed from the dict.
    ``data["prompt"]``     : str  (modified in-place; may be absent if using
        prompt_from_task – in that case, advantage becomes a stand-alone prompt)

    Output keys added
    ------------------
    ``data["action_mask"]`` : np.ndarray bool [action_horizon]
        True for timesteps whose advantage label matches the label at t=0.
        False for all timesteps from the first label boundary onward.
        The training loss is masked to 0 for False positions.
    """

    dropout_rate: float = 0.3

    def __call__(self, data: dict) -> dict:
        advantage_seq = np.asarray(data.pop("advantage"), dtype=np.float32)

        # Handle shape [H, 1] from some LeRobot versions
        if advantage_seq.ndim == 2 and advantage_seq.shape[-1] == 1:
            advantage_seq = advantage_seq.squeeze(-1)

        H = len(advantage_seq)
        label = bool(round(float(advantage_seq[0])))

        # Find first within-chunk boundary (Option 4 re-segmentation)
        boundary = H
        for i in range(1, H):
            if bool(round(float(advantage_seq[i]))) != label:
                boundary = i
                break

        # Loss mask: valid for steps [0, boundary), zero for [boundary, H)
        data["action_mask"] = np.arange(H) < boundary  # bool [H]

        # Inject advantage token into prompt (with 30% dropout for CFG)
        if np.random.random() >= self.dropout_rate:
            adv_str = " Advantage: positive" if label else " Advantage: negative"
            prompt = data.get("prompt", "")
            if isinstance(prompt, np.ndarray):
                prompt = str(prompt.item())
            data["prompt"] = (prompt + adv_str) if prompt else adv_str.lstrip()

        return data


def create_advantage_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    advantage_dropout: float = 0.3,
) -> _data_loader.Dataset:
    """Create an advantage-conditioned map-style dataset.

    Mirrors ``create_torch_dataset`` but inserts ``AdvantageConditionTransform``
    between normalization and model transforms so the advantage token is tokenized
    as part of the prompt by ``TokenizePrompt``.

    See module docstring for dataset column and DataConfig requirements.
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("repo_id is required for advantage training.")
    if repo_id == "fake":
        return _data_loader.FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(
            dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )

    if data_config.norm_stats is None:
        raise ValueError(
            f"Norm stats not found for asset_id={data_config.asset_id!r}. Training without "
            "normalization is almost certainly unintended. Run "
            "`uv run scripts/compute_norm_stats.py --config-name <config>` first."
        )
    norm_stats = data_config.norm_stats

    transforms = [
        *data_config.repack_transforms.inputs,
        AdvantageConditionTransform(dropout_rate=advantage_dropout),  # must run before AlohaInputs drops "advantage"
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    return _data_loader.TransformedDataset(dataset, transforms)


class _AdvantageRawLoader:
    """Internal helper: wraps a torch DataLoader to yield JAX-sharded dicts."""

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        sharding: jax.sharding.Sharding,
        num_batches: int | None = None,
    ):
        self._loader = torch_loader
        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self) -> Iterator[dict]:
        num_items = 0
        while True:
            for batch in self._loader:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                num_items += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                    batch,
                )


class AdvantageDataLoaderImpl(_data_loader.DataLoader):
    """DataLoader that yields ``(Observation, actions, action_mask)`` triples.

    The ``action_mask`` is a bool array of shape ``[B, action_horizon]``.
    True = valid action step (contributes to loss); False = padded beyond the
    label boundary (loss is zeroed out).
    """

    def __init__(
        self,
        data_config: _config.DataConfig,
        raw_loader: _AdvantageRawLoader,
    ):
        self._data_config = data_config
        self._raw = raw_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[tuple]:
        """Yields ``(obs, actions, action_mask)``, or, when the dataset carries
        precomputed per-link proximity (collision-critic finetune),
        ``(obs, actions, action_mask, dists, deltas)`` with
        ``dists [B, H, 16, 3]`` and ``deltas [B, H, 16, 3, 3]``."""
        for batch in self._raw:
            action_mask = batch["action_mask"]  # [B, H] bool, already sharded
            obs = _model.Observation.from_dict(batch)
            if "link_distances.dist" in batch:
                B, H = action_mask.shape[0], action_mask.shape[1]
                dists = batch["link_distances.dist"].reshape(B, H, 16, 3)
                deltas = batch["link_distances.delta"].reshape(B, H, 16, 3, 3)
                yield obs, batch["actions"], action_mask, dists, deltas
            else:
                yield obs, batch["actions"], action_mask


def create_advantage_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    advantage_dropout: float = 0.3,
    seed: int = 0,
) -> AdvantageDataLoaderImpl:
    """Create an advantage-conditioned data loader for pi05 training.

    Drop-in replacement for ``create_data_loader`` except:
      - Loads the ``advantage`` column as a 50-step delta sequence.
      - Injects advantage conditioning into each prompt (with 30% dropout).
      - Handles chunk re-segmentation at label boundaries (Option 4).
      - Yields ``(Observation, actions, action_mask)`` triples instead of pairs.

    Arguments
    ---------
    config           : TrainConfig whose DataConfig satisfies the requirements in
                       the module docstring (action_sequence_keys includes "advantage",
                       repack_transforms preserves the "advantage" key).
    advantage_dropout: Probability of OMITTING the advantage token from the prompt.
                       0.3 matches the π*0.6 paper; set to 0.0 to disable dropout.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info(f"advantage data_config: {data_config}")

    if jax.process_count() > 1:
        raise NotImplementedError("Multi-process data loading is not supported.")

    local_batch_size = config.batch_size // jax.process_count()
    logger.info(f"local_batch_size: {local_batch_size}")

    dataset = create_advantage_torch_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
        advantage_dropout=advantage_dropout,
    )

    if sharding is None:
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.devices(), ("B",)),
            jax.sharding.PartitionSpec("B"),
        )

    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    generator = torch.Generator()
    generator.manual_seed(seed)

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=local_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_data_loader._collate_fn,
        worker_init_fn=_data_loader._worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    raw_loader = _AdvantageRawLoader(torch_loader, sharding=sharding, num_batches=num_batches)
    return AdvantageDataLoaderImpl(data_config, raw_loader)


# ---------------------------------------------------------------------------
# Token-mode: advantage as a learned embedding in the action expert suffix
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _InjectFakeAdvantage(_transforms.DataTransformFn):
    """Injects a constant all-positive advantage column for fake/smoke-test data."""
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        data["advantage"] = np.ones(self.action_horizon, dtype=np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class AdvantageTokenTransform(_transforms.DataTransformFn):
    """Produces an integer advantage_indicator for the model's embedding table.

    Unlike AdvantageConditionTransform, this does NOT modify the text prompt.
    The integer index is passed directly to Pi0.compute_loss / embed_suffix,
    where it selects a row from the learned advantage_token_embedding table:
      0 = negative (failed episode / autonomous portion before intervention)
      1 = positive (successful episode / BC demo / intervention correction)
      2 = null / unconditional  (used during dropout; also used for CFG inference)

    Reads:   data["advantage"] — float32 [action_horizon], values 0.0 or 1.0
    Writes:
      data["advantage_indicator"] — int32 scalar (0, 1, or 2)
      data["action_mask"]         — bool [action_horizon]  (Option-4 chunk mask)
    Removes: data["advantage"]
    """

    dropout_rate: float = 0.3

    def __call__(self, data: dict) -> dict:
        advantage_seq = np.asarray(data.pop("advantage"), dtype=np.float32)
        if advantage_seq.ndim == 2 and advantage_seq.shape[-1] == 1:
            advantage_seq = advantage_seq.squeeze(-1)

        H = len(advantage_seq)
        label = bool(round(float(advantage_seq[0])))

        # Option-4 re-segmentation: find first within-chunk label boundary
        boundary = H
        for i in range(1, H):
            if bool(round(float(advantage_seq[i]))) != label:
                boundary = i
                break

        data["action_mask"] = np.arange(H) < boundary  # bool [H]

        # 30% dropout: replace with null token (index 2) for CFG training
        if np.random.random() < self.dropout_rate:
            idx = 2
        else:
            idx = 1 if label else 0
        data["advantage_indicator"] = np.array(idx, dtype=np.int32)
        return data


def create_advantage_token_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    advantage_dropout: float = 0.3,
) -> _data_loader.Dataset:
    """Dataset variant for the token-mode advantage approach."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("repo_id is required for advantage training.")
    if repo_id == "fake":
        # FakeDataset doesn't include 'advantage'; inject it before AdvantageTokenTransform.
        fake = _data_loader.FakeDataset(model_config, num_samples=1024)
        return _data_loader.TransformedDataset(
            fake,
            [_InjectFakeAdvantage(action_horizon=action_horizon),
             AdvantageTokenTransform(dropout_rate=advantage_dropout)],
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(
            dataset,
            [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
        )

    if data_config.norm_stats is None:
        raise ValueError(
            f"Norm stats not found for asset_id={data_config.asset_id!r}. Training without "
            "normalization is almost certainly unintended. Run "
            "`uv run scripts/compute_norm_stats.py --config-name <config>` first."
        )
    norm_stats = data_config.norm_stats

    transforms = [
        *data_config.repack_transforms.inputs,
        AdvantageTokenTransform(dropout_rate=advantage_dropout),  # must run before AlohaInputs drops "advantage"
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]

    return _data_loader.TransformedDataset(dataset, transforms)


class AdvantageTokenDataLoaderImpl(_data_loader.DataLoader):
    """DataLoader that yields (Observation, actions, action_mask, advantage_indicator).

    action_mask        : bool [B, action_horizon]  — Option-4 chunk validity mask
    advantage_indicator: int32 [B]                 — row index into advantage_token_embedding
    """

    def __init__(
        self,
        data_config: _config.DataConfig,
        raw_loader: _AdvantageRawLoader,
    ):
        self._data_config = data_config
        self._raw = raw_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[tuple[_model.Observation, jax.Array, jax.Array, jax.Array]]:
        for batch in self._raw:
            action_mask = batch["action_mask"]           # [B, H] bool
            advantage_indicator = batch["advantage_indicator"]  # [B] int32
            obs = _model.Observation.from_dict(batch)
            yield obs, batch["actions"], action_mask, advantage_indicator


def create_advantage_token_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    advantage_dropout: float = 0.3,
    seed: int = 0,
) -> AdvantageTokenDataLoaderImpl:
    """Data loader for token-mode advantage conditioning.

    Yields (Observation, actions, action_mask, advantage_indicator) 4-tuples.
    The advantage_indicator [B] int32 is passed directly to
    Pi0.compute_loss(..., advantage_indicator=...) and selects a row from the
    learned advantage_token_embedding table (0=neg, 1=pos, 2=null).

    Dataset requirements are the same as for create_advantage_data_loader:
    the LeRobot dataset must have an "advantage" column, the repack_transforms
    must include "advantage": "advantage", and action_sequence_keys must contain
    "advantage".  See pi05_aloha_advantage_token in config.py.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info(f"advantage-token data_config: {data_config}")

    if jax.process_count() > 1:
        raise NotImplementedError("Multi-process data loading is not supported.")

    local_batch_size = config.batch_size // jax.process_count()
    logger.info(f"local_batch_size: {local_batch_size}")

    dataset = create_advantage_token_dataset(
        data_config,
        action_horizon=config.model.action_horizon,
        model_config=config.model,
        advantage_dropout=advantage_dropout,
    )

    if sharding is None:
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.devices(), ("B",)),
            jax.sharding.PartitionSpec("B"),
        )

    mp_context = None
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    generator = torch.Generator()
    generator.manual_seed(seed)

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=local_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_data_loader._collate_fn,
        worker_init_fn=_data_loader._worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    raw_loader = _AdvantageRawLoader(torch_loader, sharding=sharding, num_batches=num_batches)
    return AdvantageTokenDataLoaderImpl(data_config, raw_loader)
