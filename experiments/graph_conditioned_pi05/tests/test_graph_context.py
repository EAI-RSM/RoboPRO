from pathlib import Path

import h5py
import numpy as np
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from experiments.graph_conditioned_pi05.contract import InputCondition, RetrievalContract
from experiments.graph_conditioned_pi05.graph_retriever import GraphFact, HDF5GraphRetriever
from experiments.graph_conditioned_pi05.graph_serializer import serialize_facts
from experiments.graph_conditioned_pi05.live_adapter import LiveGraphRetriever, prepare_instruction
from experiments.graph_conditioned_pi05.validate_alignment import FRAME_ALIGNED_PATHS, validate_episode


def _dataset(root, path, value):
    parent, name = path.rsplit("/", 1)
    root.require_group(parent).create_dataset(name, data=value)


def _graph_file(path: Path) -> h5py.File:
    root = h5py.File(path, "w")
    support = root.create_group("benchmark_support")
    catalog = support.create_group("object_catalog")
    state = support.create_group("relation_state")
    ids = np.array([10, 20, -2, -3])
    catalog.create_dataset("object_ids", data=ids)
    catalog.create_dataset("names", data=np.array([b"target", b"box", b"left", b"right"]))
    catalog.create_dataset("roles", data=np.array([b"target", b"object", b"effector", b"effector"]))
    catalog.create_dataset("is_target", data=np.array([1, 0, 0, 0], dtype=bool))
    state.create_dataset("object_ids", data=ids)
    state.create_dataset("held_by_effector_names", data=np.array([b"left_arm", b"right_arm"]))
    state.create_dataset("reachable_by_effector_names", data=np.array([b"left_arm", b"right_arm"]))
    state.create_dataset("visible_to_camera_names", data=np.array([b"countertop_camera", b"wrist_camera"]))
    state.create_dataset("blocks_effector_names", data=np.array([b"left_arm", b"right_arm"]))
    shape = (1, 4, 4)
    near = np.zeros(shape, dtype=bool)
    near[0, 0, 1] = near[0, 1, 0] = True
    state.create_dataset("near", data=near)
    state.create_dataset("contains", data=np.ones(shape, dtype=bool))
    state.create_dataset("contains_valid", data=np.ones(shape, dtype=bool))
    blocks = np.zeros(shape, dtype=bool)
    blocks[0, 1, 0] = True
    state.create_dataset("blocks", data=blocks)
    state.create_dataset("blocks_valid", data=np.ones(shape, dtype=bool))
    by_effector = np.zeros((1, 4, 4, 2), dtype=bool)
    by_effector[0, 1, 0] = [True, True]
    state.create_dataset("blocks_by_effector", data=by_effector)
    by_effector_valid = np.ones_like(by_effector)
    by_effector_valid[0, 1, 0, 1] = False
    state.create_dataset("blocks_by_effector_valid", data=by_effector_valid)
    reachable = np.zeros((1, 4, 2), dtype=bool)
    reachable[0, 0] = [True, True]
    state.create_dataset("reachable_by", data=reachable)
    reachable_valid = np.ones_like(reachable)
    reachable_valid[0, 0, 1] = False
    state.create_dataset("reachable_by_valid", data=reachable_valid)
    visible = np.zeros((1, 4, 2), dtype=bool)
    visible[0, 0] = [True, True]
    state.create_dataset("visible_to", data=visible)
    state.create_dataset("visible_to_valid", data=np.ones_like(visible))
    occludes = np.zeros((1, 4, 4, 2), dtype=bool)
    occludes[0, 1, 0] = [True, True]
    state.create_dataset("occludes", data=occludes)
    state.create_dataset("occludes_valid", data=np.ones_like(occludes))
    return root


def test_contract_rejects_leakage_and_invalid_facts():
    with raises(ValueError):
        RetrievalContract(include_action_history=True)
    with raises(ValueError):
        RetrievalContract(include_invalid=True)


def test_retrieval_is_deterministic_filtered_and_camera_specific(tmp_path):
    with _graph_file(tmp_path / "graph.hdf5") as root:
        retriever = HDF5GraphRetriever(root)
        first = retriever.retrieve(0)
        assert first == retriever.retrieve_frame(0)
    keys = [fact.key() for fact in first]
    assert ("reachable_by", "target#10", "left_arm", "") in keys
    assert ("reachable_by", "target#10", "right_arm", "") not in keys
    assert ("visible_to", "target#10", "countertop_camera", "") in keys
    assert not any(key[2] == "wrist_camera" for key in keys)
    assert ("occludes", "box#20", "target#10", "countertop_camera") in keys
    assert ("blocks", "box#20", "target#10", "left_arm") in keys
    assert not any(key[0] == "contains" for key in keys)
    assert sum(key[0] == "near" for key in keys) == 1
    assert not any(key[0].startswith("action") for key in keys)


def test_explicit_seed_and_unknown_seed(tmp_path):
    with _graph_file(tmp_path / "graph.hdf5") as root:
        retriever = HDF5GraphRetriever(root)
        assert retriever.resolve_seed_ids([20]) == {0, 1, 2, 3}
        with raises(ValueError):
            retriever.resolve_seed_ids([999])


def test_serialization_priority_and_budget():
    facts = [
        GraphFact(9, "near", "a", "b"),
        GraphFact(0, "held_by", "a", "left_arm"),
    ]
    full = serialize_facts(facts, 100)
    assert full.text.index("held_by") < full.text.index("near")
    small = serialize_facts(facts, 12)
    assert small.token_count <= 12
    assert small.dropped_fact_count > 0


def test_alignment_mismatch_fails(tmp_path):
    path = tmp_path / "episode0.hdf5"
    with h5py.File(path, "w") as root:
        root.attrs["schema_version"] = "1.9.0"
        root.attrs["task_name"] = "synthetic"
        root.attrs["success"] = True
        for index, name in enumerate(FRAME_ALIGNED_PATHS):
            length = 3 if index else 2
            _dataset(root, name, np.zeros((length, 1), dtype=np.float32))
    report = validate_episode(path, RetrievalContract(), 1)
    assert report["status"] == "FAIL"
    assert any("frame-axis mismatch" in failure for failure in report["failures"])

@contextmanager
def raises(exception):
    try:
        yield
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")



def _live_inputs(root):
    catalog = [
        {"object_id": 10, "name": "target", "is_target": True},
        {"object_id": 20, "name": "box", "is_target": False},
        {"object_id": -2, "name": "left", "is_target": False},
        {"object_id": -3, "name": "right", "is_target": False},
    ]
    state = {}
    for name, dataset in root["benchmark_support/relation_state"].items():
        if dataset.ndim > 0 and dataset.shape[0] == 1:
            state[name] = dataset[0]
        else:
            state[name] = dataset[()]
    return catalog, state


def test_live_retrieval_matches_hdf5(tmp_path):
    with _graph_file(tmp_path / "graph.hdf5") as root:
        expected = HDF5GraphRetriever(root).retrieve(0, [20])
        catalog, state = _live_inputs(root)
        actual = LiveGraphRetriever(catalog, state).retrieve([20])
    assert actual == expected


def test_prepare_instruction_preserves_visual_only_and_fits_graph(tmp_path):
    class Task:
        def __init__(self, catalog):
            self.catalog = catalog

        def get_instruction(self):
            return "put target in box"

        def get_role_names(self):
            return {"destination_id": 20}

        def _get_benchmark_object_catalog(self):
            return self.catalog

    class Model:
        def __init__(self):
            self.calls = []

        def fit_graph_prompt(self, payload):
            self.calls.append(payload)
            return {
                "instruction": payload["instruction"] + "\nScene graph: " + payload["fact_texts"][0],
                "selected_fact_count": 1,
                "graph_token_count": 9,
                "full_prompt_token_count_estimate": 40,
            }

    with _graph_file(tmp_path / "graph.hdf5") as root:
        catalog, state = _live_inputs(root)
    observation = {
        "benchmark_support": {"relation_state": state},
        "joint_action": {"vector": np.zeros(14, dtype=np.float32)},
    }
    task, model = Task(catalog), Model()
    visual = prepare_instruction(
        task, model, observation, InputCondition.VISUAL_ONLY, RetrievalContract()
    )
    assert visual.instruction == "put target in box"
    assert model.calls == []
    graph = prepare_instruction(
        task,
        model,
        observation,
        InputCondition.VISUAL_RETRIEVED_GRAPH,
        RetrievalContract(),
    )
    assert graph.selected_fact_count == 1
    assert graph.destination_seed_available
    assert len(model.calls) == 1
    assert all("action" not in fact for fact in model.calls[0]["fact_texts"])

def main():
    test_contract_rejects_leakage_and_invalid_facts()
    with TemporaryDirectory() as directory:
        test_retrieval_is_deterministic_filtered_and_camera_specific(Path(directory))
    with TemporaryDirectory() as directory:
        test_explicit_seed_and_unknown_seed(Path(directory))
    test_serialization_priority_and_budget()
    with TemporaryDirectory() as directory:
        test_live_retrieval_matches_hdf5(Path(directory))
    with TemporaryDirectory() as directory:
        test_prepare_instruction_preserves_visual_only_and_fits_graph(Path(directory))
    with TemporaryDirectory() as directory:
        test_alignment_mismatch_fails(Path(directory))
    print("7 graph-context tests passed")


if __name__ == "__main__":
    main()
