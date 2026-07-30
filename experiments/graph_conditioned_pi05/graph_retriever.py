"""Deterministic retrieval of a leakage-safe current-frame scene subgraph."""

from __future__ import annotations

from typing import Iterable

import h5py
import numpy as np

from .contract import (
    INVERSE_RELATIONS,
    RELATION_PRIORITY,
    SYMMETRIC_RELATIONS,
    VALIDITY_FIELD,
    GraphFact,
    RetrievalContract,
)


def _decode(values) -> list[str]:
    array = np.asarray(values)
    return [
        value.decode("utf-8", errors="replace")
        if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in array.tolist()
    ]



class HDF5GraphRetriever:
    """Read current-frame facts without exposing expert action nodes."""

    def __init__(self, root: h5py.File, contract: RetrievalContract | None = None):
        self.root = root
        self.contract = contract or RetrievalContract()
        support = root["benchmark_support"]
        self.catalog = support["object_catalog"]
        self.state = support["relation_state"]
        self.object_ids = np.asarray(self.catalog["object_ids"][()], dtype=np.int64)
        self.names = _decode(self.catalog["names"][()])
        self.roles = _decode(self.catalog["roles"][()])
        self.is_target = np.asarray(self.catalog["is_target"][()], dtype=np.bool_)
        relation_ids = np.asarray(self.state["object_ids"][()], dtype=np.int64)
        if not np.array_equal(self.object_ids, relation_ids):
            raise ValueError("Object catalog and relation-state IDs are not aligned")
        self.labels = [f"{name}#{int(object_id)}" for name, object_id in zip(self.names, self.object_ids)]
        self.effector_names = _decode(self.state["held_by_effector_names"][()])
        self.reachable_names = _decode(self.state["reachable_by_effector_names"][()])
        self.camera_names = _decode(self.state["visible_to_camera_names"][()])
        self.blocks_effector_names = (
            _decode(self.state["blocks_effector_names"][()])
            if "blocks_effector_names" in self.state else []
        )
        self.frame_count = int(self.state["near"].shape[0])

    def default_seed_indices(self) -> set[int]:
        seeds = set(np.flatnonzero(self.is_target).astype(int).tolist())
        for idx, object_id in enumerate(self.object_ids.tolist()):
            if int(object_id) in {-2, -3}:
                seeds.add(idx)
        return seeds

    def resolve_seed_ids(self, object_ids: Iterable[int] | None) -> set[int]:
        seeds = self.default_seed_indices()
        if object_ids is None:
            return seeds
        index_by_id = {int(object_id): idx for idx, object_id in enumerate(self.object_ids)}
        for object_id in object_ids:
            if int(object_id) not in index_by_id:
                raise ValueError(f"Seed object ID {object_id} is absent from the catalog")
            seeds.add(index_by_id[int(object_id)])
        return seeds

    def _valid_matrix(self, relation: str, frame: int, shape: tuple[int, ...]) -> np.ndarray:
        validity_name = VALIDITY_FIELD.get(relation)
        if validity_name is None or validity_name not in self.state:
            return np.ones(shape, dtype=np.bool_)
        valid = np.asarray(self.state[validity_name][frame], dtype=np.bool_)
        if valid.shape != shape:
            raise ValueError(
                f"{validity_name} shape {valid.shape} does not match {relation} shape {shape}"
            )
        return valid

    def retrieve(self, frame: int, seed_object_ids: Iterable[int] | None = None) -> list[GraphFact]:
        if frame < 0 or frame >= self.frame_count:
            raise IndexError(f"Frame {frame} is outside [0,{self.frame_count})")
        self._active_frame = frame
        seeds = self.resolve_seed_ids(seed_object_ids)
        facts: list[GraphFact] = []
        for priority, relation in enumerate(RELATION_PRIORITY):
            if relation in INVERSE_RELATIONS and not self.contract.include_inverse_relations:
                continue
            if relation not in self.state:
                continue
            values = np.asarray(self.state[relation][frame], dtype=np.bool_)
            if relation == "occludes":
                facts.extend(self._camera_conditioned_facts(priority, relation, values, seeds))
            elif relation in {"held_by", "reachable_by", "visible_to"}:
                facts.extend(self._bipartite_facts(priority, relation, values, seeds))
            elif values.ndim == 2:
                facts.extend(self._binary_facts(priority, relation, values, seeds))
        unique = {fact.key(): fact for fact in facts}
        return sorted(unique.values())

    def _binary_facts(
        self,
        priority: int,
        relation: str,
        values: np.ndarray,
        seeds: set[int],
    ) -> list[GraphFact]:
        valid = self._valid_matrix(relation, self._active_frame, values.shape)
        result = []
        for source, destination in np.argwhere(np.logical_and(values, valid)):
            source, destination = int(source), int(destination)
            if source not in seeds and destination not in seeds:
                continue
            if relation in SYMMETRIC_RELATIONS and destination <= source:
                continue
            qualifier = ""
            if relation == "blocks" and "blocks_by_effector" in self.state:
                per_effector = np.asarray(
                    self.state["blocks_by_effector"][
                        self._active_frame, source, destination
                    ],
                    dtype=np.bool_,
                )
                if "blocks_by_effector_valid" in self.state:
                    per_effector = np.logical_and(
                        per_effector,
                        np.asarray(
                            self.state["blocks_by_effector_valid"][
                                self._active_frame, source, destination
                            ],
                            dtype=np.bool_,
                        ),
                    )
                qualifier = ",".join(
                    name
                    for name, flag in zip(self.blocks_effector_names, per_effector)
                    if flag
                )
            result.append(
                GraphFact(
                    priority,
                    relation,
                    self.labels[source],
                    self.labels[destination],
                    qualifier,
                )
            )
        return result

    def _bipartite_facts(
        self,
        priority: int,
        relation: str,
        values: np.ndarray,
        seeds: set[int],
    ) -> list[GraphFact]:
        valid = self._valid_matrix(relation, self._active_frame, values.shape)
        labels = (
            self.camera_names
            if relation == "visible_to"
            else self.effector_names
            if relation == "held_by"
            else self.reachable_names
        )
        result = []
        for source, destination in np.argwhere(np.logical_and(values, valid)):
            source, destination = int(source), int(destination)
            if source not in seeds or destination >= len(labels):
                continue
            if relation == "visible_to" and labels[destination] != self.contract.default_camera:
                continue
            result.append(
                GraphFact(priority, relation, self.labels[source], labels[destination])
            )
        return result

    def _camera_conditioned_facts(
        self,
        priority: int,
        relation: str,
        values: np.ndarray,
        seeds: set[int],
    ) -> list[GraphFact]:
        if self.contract.default_camera not in self.camera_names:
            return []
        camera_idx = self.camera_names.index(self.contract.default_camera)
        matrix = values[:, :, camera_idx]
        valid = self._valid_matrix(
            relation, self._active_frame, values.shape
        )[:, :, camera_idx]
        result = []
        for source, destination in np.argwhere(np.logical_and(matrix, valid)):
            source, destination = int(source), int(destination)
            if source not in seeds and destination not in seeds:
                continue
            result.append(
                GraphFact(
                    priority,
                    relation,
                    self.labels[source],
                    self.labels[destination],
                    self.contract.default_camera,
                )
            )
        return result

    def retrieve_frame(
        self,
        frame: int,
        seed_object_ids: Iterable[int] | None = None,
    ) -> list[GraphFact]:
        return self.retrieve(frame, seed_object_ids)
