from pathlib import Path

import h5py
import numpy as np
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from experiments.graph_conditioned_pi05.contract import GraphNode, InputCondition, RetrievalContract
from experiments.graph_conditioned_pi05.graph_retriever import GraphFact, HDF5GraphRetriever
from experiments.graph_conditioned_pi05.graph_serializer import (
    NaturalLanguageGraphRenderer,
    format_fact,
    format_node,
    serialize_facts,
    serialize_graph,
)
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
    assert full.text.index("is held by") < full.text.index("is near")
    small = serialize_facts(facts, 11)
    assert small.token_count <= 11
    assert small.dropped_fact_count > 0


def test_packing_maximizes_same_tier_information_over_alphabetical_first_fit():
    # Regression test: within a single priority tier, a first-fit walk in
    # (relation, source, destination) order -- not size order -- can lock in
    # one long fact that happens to sort first alphabetically, leaving no
    # room for several shorter same-priority facts that would together carry
    # more information. `aaaa...` sorts before `ba`/`ca`/`da`, so the old
    # first-fit algorithm picked {long, ba} (2 facts) at this exact budget;
    # the knapsack packer picks {ba, ca, da} (3 facts) instead.
    long_fact = GraphFact(9, "near", "aaaa bbbb cccc dddd eeee", "target")
    short_facts = [GraphFact(9, "near", letter, "target") for letter in ("ba", "ca", "da")]
    result = serialize_facts([long_fact, *short_facts], token_budget=15)
    assert set(result.selected_facts) == set(short_facts)
    assert long_fact not in result.selected_facts


def test_serialize_graph_includes_nodes_with_grounding_attributes():
    bowl_a = GraphNode(object_id=23, label="bowl#23", kind="actor", position=(0.1, 0.2, 0.3), bbox_size=(0.2, 0.1, 0.1))
    bowl_b = GraphNode(object_id=41, label="bowl#41", kind="actor", position=(0.5, -0.1, 0.3), bbox_size=(0.2, 0.1, 0.1))
    fact = GraphFact(0, "near", "bowl#23", "bowl#41")
    result = serialize_graph([bowl_a, bowl_b], [fact], token_budget=80)
    assert bowl_a in result.selected_nodes and bowl_b in result.selected_nodes
    assert format_node(bowl_a) in result.text
    assert format_node(bowl_b) in result.text
    assert result.text.index("Nodes:") < result.text.index("Scene graph:")

    left_ee = GraphNode(object_id=-2, label="left_ee", kind="end_effector", position=(0.2, -0.1, 0.9))
    assert format_node(left_ee) == "left_ee is at (0.2, -0.1, 0.9)."


def test_natural_language_graph_renderer():
    renderer = NaturalLanguageGraphRenderer()
    assert renderer.render_fact(
        GraphFact(1, "reachable_by", "obj_1", "obj_2")
    ) == "obj_1 is reachable by obj_2."
    assert renderer.render_fact(
        GraphFact(2, "blocks", "obstacle", "target", "left_ee,right_ee")
    ) == "obstacle blocks target for left_ee and right_ee."
    assert format_fact(
        GraphFact(3, "occludes", "cup", "can", "countertop_camera")
    ) == "cup occludes can from countertop_camera."


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
    object_state = {
        "object_ids": np.array([10, 20, -2, -3], dtype=np.int64),
        "pose_world": np.array(
            [
                [0.1, 0.2, 0.3, 1, 0, 0, 0],
                [0.4, 0.5, 0.6, 1, 0, 0, 0],
                [0.7, 0.0, 0.5, 1, 0, 0, 0],
                [0.9, 0.0, 0.5, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "is_present": np.array([True, True, True, True]),
        "aabb_lower": np.array(
            [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "aabb_upper": np.array(
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "has_aabb": np.array([True, True, False, False]),
    }
    return catalog, state, object_state


def test_live_retrieval_matches_hdf5(tmp_path):
    with _graph_file(tmp_path / "graph.hdf5") as root:
        expected = HDF5GraphRetriever(root).retrieve(0, [20])
        catalog, state, object_state = _live_inputs(root)
        actual_nodes, actual_facts = LiveGraphRetriever(catalog, state, object_state).retrieve([20])
    assert actual_facts == expected
    node_ids = {node.object_id for node in actual_nodes}
    assert {10, 20} <= node_ids  # target + box, both seeds referenced by facts
    target_node = next(node for node in actual_nodes if node.object_id == 10)
    assert target_node.position == (0.1, 0.2, 0.3)
    assert target_node.bbox_size == (0.2, 0.2, 0.2)
    left_ee_node = next((node for node in actual_nodes if node.object_id == -2), None)
    assert left_ee_node is not None and left_ee_node.bbox_size is None


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
            fact_texts = [item["text"] for item in payload["items"] if item["section"] == "fact"]
            node_texts = [item["text"] for item in payload["items"] if item["section"] == "node"]
            return {
                "instruction": payload["instruction"] + "\nScene graph: " + fact_texts[0],
                "selected_node_count": len(node_texts),
                "selected_fact_count": 1,
                "graph_token_count": 9,
                "full_prompt_token_count_estimate": 40,
            }

    with _graph_file(tmp_path / "graph.hdf5") as root:
        catalog, state, object_state = _live_inputs(root)
    observation = {
        "benchmark_support": {"relation_state": state, "object_state": object_state},
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
    assert graph.selected_node_count > 0
    assert graph.destination_seed_available
    assert len(model.calls) == 1
    items = model.calls[0]["items"]
    assert all("action" not in item["text"] for item in items)
    assert all(item["text"].endswith(".") for item in items)
    fact_texts = [item["text"] for item in items if item["section"] == "fact"]
    node_texts = [item["text"] for item in items if item["section"] == "node"]
    assert any(" is reachable by " in text for text in fact_texts)
    assert node_texts  # nodes (target/box/effectors) are declared with grounding attributes
    assert any(" is at (" in text for text in node_texts)

def main():
    test_contract_rejects_leakage_and_invalid_facts()
    with TemporaryDirectory() as directory:
        test_retrieval_is_deterministic_filtered_and_camera_specific(Path(directory))
    with TemporaryDirectory() as directory:
        test_explicit_seed_and_unknown_seed(Path(directory))
    test_serialization_priority_and_budget()
    test_packing_maximizes_same_tier_information_over_alphabetical_first_fit()
    test_serialize_graph_includes_nodes_with_grounding_attributes()
    test_natural_language_graph_renderer()
    with TemporaryDirectory() as directory:
        test_live_retrieval_matches_hdf5(Path(directory))
    with TemporaryDirectory() as directory:
        test_prepare_instruction_preserves_visual_only_and_fits_graph(Path(directory))
    with TemporaryDirectory() as directory:
        test_alignment_mismatch_fails(Path(directory))
    print("10 graph-context tests passed")


if __name__ == "__main__":
    main()
