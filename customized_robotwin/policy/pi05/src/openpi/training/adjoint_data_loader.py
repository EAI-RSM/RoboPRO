"""Data loader for adjoint-matching fine-tuning of pi05.

Yields ``(Observation, actions, obstacle_points)`` where
    actions          : [B, H, 14]      BC target (base flow loss)
    obstacle_points   : [B, M, 3]       per-sample scene obstacle points (AM reward)

Mirrors advantage_data_loader but without advantage conditioning / action masks —
plain BC inputs plus the ``obstacle_points`` per-frame column (NOT windowed; it is
the episode's static scene geometry).  Requires the dataset built by
convert_collision_am_to_lerobot.py and the ``pi05_aloha_adjoint`` config (repack
preserves ``obstacle_points``; AlohaInputs passes it through to the batch).
"""

from __future__ import annotations

import logging
import multiprocessing
import typing
from collections.abc import Iterator

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)


def create_adjoint_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
) -> _data_loader.Dataset:
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("repo_id is required for adjoint-matching training.")
    if repo_id == "fake":
        return _data_loader.FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    # Only the action-sequence keys are windowed; obstacle_points is a plain
    # per-frame feature returned as-is.
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )
    if data_config.prompt_from_task:
        dataset = _data_loader.TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )
    if data_config.norm_stats is None:
        raise ValueError(
            "Norm stats not found. Run `uv run scripts/compute_norm_stats.py "
            "--config-name pi05_aloha_adjoint` first."
        )
    transforms = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,   # AlohaInputs passes obstacle_points through
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    return _data_loader.TransformedDataset(dataset, transforms)


class _AdjointRawLoader:
    def __init__(self, torch_loader, sharding, num_batches=None):
        self._loader = torch_loader
        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self) -> Iterator[dict]:
        n = 0
        while True:
            for batch in self._loader:
                if self._num_batches is not None and n >= self._num_batches:
                    return
                n += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch
                )


class AdjointDataLoaderImpl(_data_loader.DataLoader):
    """Yields ``(Observation, actions, obstacle_points[B, M, 3])``."""

    def __init__(self, data_config: _config.DataConfig, raw_loader: _AdjointRawLoader):
        self._data_config = data_config
        self._raw = raw_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[tuple]:
        for batch in self._raw:
            obs = _model.Observation.from_dict(batch)
            actions = batch["actions"]
            pts = batch["obstacle_points"]            # [B, M*3]
            B = pts.shape[0]
            pts = pts.reshape(B, -1, 3)               # [B, M, 3]
            yield obs, actions, pts


def create_adjoint_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> AdjointDataLoaderImpl:
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info(f"adjoint data_config: {data_config}")
    if jax.process_count() > 1:
        raise NotImplementedError("Multi-process data loading is not supported.")

    local_batch_size = config.batch_size // jax.process_count()
    dataset = create_adjoint_torch_dataset(data_config, config.model.action_horizon, config.model)

    if sharding is None:
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(jax.devices(), ("B",)), jax.sharding.PartitionSpec("B")
        )

    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
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
    raw_loader = _AdjointRawLoader(torch_loader, sharding=sharding, num_batches=num_batches)
    return AdjointDataLoaderImpl(data_config, raw_loader)
