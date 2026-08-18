import math
from pathlib import Path

import h5py
import numpy as np
import transforms3d as t3d
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from experiments.graph_conditioned_pi05.action_intent import IntentOperation
from experiments.graph_conditioned_pi05.action_diagnostics import graph_evidence

from experiments.graph_conditioned_pi05.contract import GraphNode, InputCondition, RetrievalContract
from experiments.graph_conditioned_pi05.graph_retriever import GraphFact, HDF5GraphRetriever
from experiments.graph_conditioned_pi05.graph_serializer import (
    NaturalLanguageGraphRenderer,
    fact_pack_item,
    format_fact,
    format_node,
    node_pack_item,
    pack_items,
    serialize_facts,
    serialize_graph,
)
from experiments.graph_conditioned_pi05.graph_replanning import GraspSubstage
from experiments.graph_conditioned_pi05.live_adapter import (
    CONTACT_POINT_TO_GRASP_ROTATION,
    LiveGraphRetriever,
    action_graph_state,
    build_live_graph_context,
    compact_grasp_hint,
    destination_ids_from_task,
    goal_geometry_pack_item,
    keep_active_gripper_closed,
    live_task_state,
    prepare_instruction,
    vla_label_from_catalog_entry,
)
from experiments.graph_conditioned_pi05.simulator_evidence import (
    extract_simulator_evidence,
    _rotate_quat_about_own_local_axis,
)
from experiments.graph_conditioned_pi05.validate_alignment import FRAME_ALIGNED_PATHS, validate_episode


def _nan_safe_equal(a, b) -> bool:
    """Dict equality that treats NaN == NaN as equal (unlike plain ``==``)."""
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        other = b[key]
        if isinstance(value, float) and isinstance(other, float) and value != value and other != other:
            continue
        if value != other:
            return False
    return True


def _dataset(root, path, value):
    parent, name = path.rsplit("/", 1)
    root.require_group(parent).create_dataset(name, data=value)


def _graph_file(path: Path) -> h5py.File:
    root = h5py.File(path, "w")
    support = root.create_group("benchmark_support")
    catalog = support.create_group("object_catalog")
    state = support.create_group("relation_state")
    ids = np.array([10, 20, -1, -2, -3])
    catalog.create_dataset("object_ids", data=ids)
    catalog.create_dataset("names", data=np.array([b"target", b"box", b"robot", b"left", b"right"]))
    catalog.create_dataset("roles", data=np.array([b"target", b"destination", b"robot", b"effector", b"effector"]))
    catalog.create_dataset("is_target", data=np.array([1, 0, 0, 0, 0], dtype=bool))
    state.create_dataset("object_ids", data=ids)
    state.create_dataset("held_by_effector_names", data=np.array([b"left_arm", b"right_arm"]))
    state.create_dataset("reachable_by_effector_names", data=np.array([b"left_arm", b"right_arm"]))
    state.create_dataset("visible_to_camera_names", data=np.array([b"countertop_camera", b"wrist_camera"]))
    state.create_dataset("blocks_effector_names", data=np.array([b"left_arm", b"right_arm"]))
    shape = (1, 5, 5)
    near = np.zeros(shape, dtype=bool)
    near[0, 0, 1] = near[0, 1, 0] = True
    state.create_dataset("near", data=near)
    state.create_dataset("contains", data=np.ones(shape, dtype=bool))
    state.create_dataset("contains_valid", data=np.ones(shape, dtype=bool))
    blocks = np.zeros(shape, dtype=bool)
    blocks[0, 1, 0] = True
    state.create_dataset("blocks", data=blocks)
    state.create_dataset("blocks_valid", data=np.ones(shape, dtype=bool))
    by_effector = np.zeros((1, 5, 5, 2), dtype=bool)
    by_effector[0, 1, 0] = [True, True]
    state.create_dataset("blocks_by_effector", data=by_effector)
    by_effector_valid = np.ones_like(by_effector)
    by_effector_valid[0, 1, 0, 1] = False
    state.create_dataset("blocks_by_effector_valid", data=by_effector_valid)
    reachable = np.zeros((1, 5, 2), dtype=bool)
    reachable[0, 0] = [True, True]
    state.create_dataset("reachable_by", data=reachable)
    reachable_valid = np.ones_like(reachable)
    reachable_valid[0, 0, 1] = False
    state.create_dataset("reachable_by_valid", data=reachable_valid)
    visible = np.zeros((1, 5, 2), dtype=bool)
    visible[0, 0] = [True, True]
    state.create_dataset("visible_to", data=visible)
    state.create_dataset("visible_to_valid", data=np.ones_like(visible))
    occludes = np.zeros((1, 5, 5, 2), dtype=bool)
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
    assert ("reachable_by", "T1", "L", "") in keys
    assert ("reachable_by", "T1", "R", "") not in keys
    assert ("visible_to", "T1", "countertop_camera", "") in keys
    assert not any(key[2] == "wrist_camera" for key in keys)
    assert ("occludes", "D1", "T1", "countertop_camera") in keys
    assert ("blocks", "D1", "T1", "L") in keys
    assert not any(key[0] == "contains" for key in keys)
    assert sum(key[0] == "near" for key in keys) == 1
    assert not any(key[0].startswith("action") for key in keys)


def test_explicit_seed_and_unknown_seed(tmp_path):
    with _graph_file(tmp_path / "graph.hdf5") as root:
        retriever = HDF5GraphRetriever(root)
        assert retriever.resolve_seed_ids([20]) == {0, 1, 3, 4}
        with raises(ValueError):
            retriever.resolve_seed_ids([999])


def test_serialization_priority_and_budget():
    facts = [
        GraphFact(9, "near", "a", "b"),
        GraphFact(0, "held_by", "a", "left_arm"),
    ]
    full = serialize_facts(facts, 100)
    assert full.text.index("is held by") < full.text.index("is near")
    small = serialize_facts(facts, 10)
    assert small.token_count <= 10
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
    bowl_a = GraphNode(object_id=23, label="bowl", kind="actor", position=(0.1, 0.2, 0.3), bbox_size=(0.2, 0.1, 0.1), alias="T1")
    bowl_b = GraphNode(object_id=41, label="bowl", kind="actor", position=(0.5, -0.1, 0.3), bbox_size=(0.2, 0.1, 0.1), alias="O1")
    fact = GraphFact(0, "near", "T1", "O1", "", ("T1", "O1"))
    result = serialize_graph([bowl_a, bowl_b], [fact], token_budget=80)
    assert bowl_a in result.selected_nodes and bowl_b in result.selected_nodes
    assert format_node(bowl_a) in result.text
    assert format_node(bowl_b) in result.text
    assert result.text.index("Nodes:") < result.text.index("Relations:")

    left_ee = GraphNode(object_id=-2, label="left end effector", kind="end_effector", position=(0.2, -0.1, 0.9), alias="L")
    assert format_node(left_ee) == "L = left end effector at (0.2, -0.1, 0.9)."

    small_can = GraphNode(
        object_id=51,
        label="sauce can",
        kind="actor",
        position=(0.04, -0.06, 0.74),
        bbox_size=(0.047, 0.049, 0.123),
        alias="T1",
    )
    assert format_node(small_can) == "T1 = sauce can, size (0.05, 0.05, 0.12)."


def test_goal_geometry_is_robot_relative_and_mandatory():
    identity = goal_geometry_pack_item(
        "T1", "D1",
        np.array([0.0, 0.0, 0.0]),
        np.array([0.4, -0.2, 0.1]),
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "outside",
    )
    assert identity.mandatory and identity.section == "goal"
    assert identity.requires == ("T1", "D1")
    assert identity.text == (
        "T1 to D1 (robot base): forward 0.40m, "
        "right 0.20m, up 0.10m; outside."
    )

    # A +90-degree base yaw means world +x is base -y (right).
    half = np.sqrt(0.5)
    rotated = goal_geometry_pack_item(
        "T1", "D1",
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, half, 0.0, 0.0, half]),
        "inside",
    )
    assert "right 1.00m" in rotated.text
    assert rotated.text.endswith("inside.")


def test_mandatory_goal_and_dependencies_survive_tight_budget():
    target = GraphNode(10, "sauce can", "actor", (0.1, 0.2, 0.3), (0.05, 0.05, 0.12), "T1")
    destination = GraphNode(20, "basket", "actor", (0.4, 0.5, 0.6), (0.3, 0.2, 0.1), "D1")
    goal = goal_geometry_pack_item(
        "T1", "D1", np.zeros(3), np.ones(3),
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), "outside",
    )
    fact = GraphFact(0, "held_by", "T1", "L", "", ("T1", "L"))
    left = GraphNode(-2, "left end effector", "end_effector", (0.2, 0.0, 0.5), alias="L")
    items = [node_pack_item(target), node_pack_item(destination), node_pack_item(left), goal, fact_pack_item(fact)]

    mandatory_only = pack_items(items[:-1], 200)
    tight = pack_items(items, mandatory_only.token_count)
    assert {item.section for item in tight.selected} == {"node", "goal"}
    assert {item.provides for item in tight.selected if item.section == "node"} == {"T1", "D1"}
    assert goal in tight.selected and fact not in tight.selected

    with raises(ValueError):
        pack_items(items, mandatory_only.token_count - 1)


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
        {"object_id": -1, "name": "robot", "is_target": False},
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
        "object_ids": np.array([10, 20, -1, -2, -3], dtype=np.int64),
        "pose_world": np.array(
            [
                [0.1, 0.2, 0.3, 1, 0, 0, 0],
                [0.4, 0.5, 0.6, 1, 0, 0, 0],
                [0.0, 0.0, 0.0, 1, 0, 0, 0],
                [0.7, 0.0, 0.5, 1, 0, 0, 0],
                [0.9, 0.0, 0.5, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
        "is_present": np.array([True, True, True, True, True]),
        "aabb_lower": np.array(
            [[0.0, 0.1, 0.26], [0.3, 0.4, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "aabb_upper": np.array(
            [[0.2, 0.3, 0.34], [0.5, 0.6, 0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "has_aabb": np.array([True, True, False, False, False]),
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
    assert target_node.bbox_size == (0.2, 0.2, 0.08)
    left_ee_node = next((node for node in actual_nodes if node.object_id == -2), None)
    assert left_ee_node is not None and left_ee_node.bbox_size is None


def test_shared_live_context_has_value_parity_and_one_catalog_parse(tmp_path):
    class Task:
        def __init__(self, catalog):
            self.catalog = catalog
            self.catalog_reads = 0

        def get_instruction(self):
            return "put target in box"

        def get_role_names(self):
            return {"target_id": 10, "destination_id": 20}

        def _get_benchmark_object_catalog(self):
            self.catalog_reads += 1
            return self.catalog

    with _graph_file(tmp_path / "graph.hdf5") as root:
        catalog, state, object_state = _live_inputs(root)
    state["held_by"] = np.zeros((5, 2), dtype=bool)
    state["held_by"][0, 0] = True
    state["held_by_valid"] = np.ones((5, 2), dtype=bool)
    state["in"] = np.zeros((5, 5), dtype=bool)
    state["containment_valid"] = np.ones((5, 5), dtype=bool)
    state["raw_contact"] = np.zeros((5, 5), dtype=bool)
    state["raw_contact"][0, 3] = state["raw_contact"][3, 0] = True
    object_state["pose_world"][3, :3] = [0.10, 0.20, 0.35]
    observation = {
        "benchmark_support": {
            "relation_state": state,
            "object_state": object_state,
        }
    }
    task = Task(catalog)
    contract = RetrievalContract()

    independent_control = action_graph_state(task, observation, contract)
    independent_live = live_task_state(task, observation, contract)
    independent_diagnostics = graph_evidence(task, observation)
    assert task.catalog_reads == 3

    task.catalog_reads = 0
    context = build_live_graph_context(task, observation, contract)
    evidence = extract_simulator_evidence(context)
    shared_control = action_graph_state(
        task, observation, contract, context=context, evidence=evidence
    )
    shared_live = live_task_state(
        task, observation, contract, context=context, evidence=evidence
    )
    shared_diagnostics = graph_evidence(
        task, observation, context=context, evidence=evidence
    )
    assert task.catalog_reads == 1
    assert shared_control == independent_control
    assert shared_live == independent_live
    assert _nan_safe_equal(shared_diagnostics, independent_diagnostics)
    assert evidence.held_arm == "left"
    assert evidence.left.target_held and evidence.left.target_contact
    assert evidence.left.grasp_height_aligned
    assert not evidence.right.target_held and not evidence.right.target_contact
    assert evidence.grasp_substage is GraspSubstage.CLOSE
    assert evidence.grasp_arm == "left"
    assert evidence.grasp_close_immediate

    state["held_by"][:] = False
    object_state["pose_world"][3, :3] = [0.18, 0.20, 0.37]
    near_context = build_live_graph_context(task, observation, contract)
    near_evidence = extract_simulator_evidence(near_context)
    assert near_evidence.grasp_substage is GraspSubstage.MOVE_DOWN
    assert near_evidence.grasp_arm == "left"
    assert near_evidence.left.target_contact
    assert not near_evidence.left.grasp_height_aligned

    # Invalid-height contact on one arm must not mask a ready other arm.
    object_state["pose_world"][4, :3] = [0.15, 0.20, 0.34]
    alternate_context = build_live_graph_context(task, observation, contract)
    alternate_evidence = extract_simulator_evidence(alternate_context)
    assert alternate_evidence.grasp_substage is GraspSubstage.CLOSE
    assert alternate_evidence.grasp_arm == "right"
    assert alternate_evidence.grasp_close_immediate

    state["raw_contact"][:] = False
    object_state["pose_world"][4, :3] = [0.90, 0.00, 0.50]
    object_state["pose_world"][3, :3] = [0.18, 0.20, 0.31]
    below_context = build_live_graph_context(task, observation, contract)
    below_evidence = extract_simulator_evidence(below_context)
    assert below_evidence.grasp_substage is GraspSubstage.MOVE_UP
    assert below_evidence.grasp_arm == "left"

    object_state["pose_world"][3, :3] = [0.20, 0.20, 0.34]
    vicinity_context = build_live_graph_context(task, observation, contract)
    vicinity_evidence = extract_simulator_evidence(vicinity_context)
    assert vicinity_evidence.grasp_substage is GraspSubstage.MOVE_CLOSER
    assert vicinity_evidence.grasp_arm == "left"
    assert not vicinity_evidence.grasp_close_immediate

    object_state["pose_world"][3, :3] = [0.17, 0.20, 0.34]
    tolerated_context = build_live_graph_context(task, observation, contract)
    tolerated_evidence = extract_simulator_evidence(tolerated_context)
    assert tolerated_evidence.grasp_substage is GraspSubstage.CLOSE
    assert tolerated_evidence.grasp_arm == "left"
    assert not tolerated_evidence.grasp_close_immediate

    state["raw_contact"][0, 3] = state["raw_contact"][3, 0] = True
    contacted_context = build_live_graph_context(task, observation, contract)
    contacted_evidence = extract_simulator_evidence(contacted_context)
    assert contacted_evidence.grasp_substage is GraspSubstage.CLOSE
    assert contacted_evidence.grasp_arm == "left"
    assert contacted_evidence.grasp_close_immediate
    state["raw_contact"][:] = False

    object_state["pose_world"][3, :3] = [0.16, 0.20, 0.34]
    close_context = build_live_graph_context(task, observation, contract)
    close_evidence = extract_simulator_evidence(close_context)
    assert close_evidence.grasp_substage is GraspSubstage.CLOSE
    assert close_evidence.grasp_arm == "left"
    assert not close_evidence.left.target_contact
    assert close_evidence.left.grasp_height_aligned
    assert close_evidence.grasp_close_immediate


class _FakeGraspActor:
    """Minimal stand-in for the sim's wrapped ``Actor`` grasp-geometry API.

    Contact points are specified as the DESIRED post-transform grasp
    orientation -- what a real annotation's ``get_grasp_pose`` would
    resolve to -- and converted back through the inverse of
    ``CONTACT_POINT_TO_GRASP_ROTATION`` into the raw pre-transform contact
    matrix the real wrapped-Actor API returns. This exercises the same
    contact-to-grasp transform production code applies, rather than
    assuming it away by handing back an already-transformed quaternion.
    """

    def __init__(self, name: str, grasp_quats_wxyz):
        self._name = name
        self._grasp_quats = list(grasp_quats_wxyz)

    def get_name(self):
        return self._name

    def iter_contact_points(self, ret="matrix"):
        assert ret == "matrix", "fake actor only supports the matrix format"
        inverse_rotation = CONTACT_POINT_TO_GRASP_ROTATION.T
        for index, quat in enumerate(self._grasp_quats):
            grasp_matrix = np.eye(4)
            grasp_matrix[:3, :3] = t3d.quaternions.quat2mat(quat)
            yield index, grasp_matrix @ inverse_rotation


def test_grasp_orientation_gate_uses_annotated_contact_pose(tmp_path):
    """Orientation error is measured against an annotated grasp contact
    point -- not a fixed world-frame assumption -- but is diagnostics-only
    for now: it must never block CLOSE, aligned or not, until a real batch
    confirms it predicts outcomes and a corrective instruction exists for a
    misaligned orientation."""

    class Task:
        def __init__(self, catalog):
            self.catalog = catalog
            self.target = _FakeGraspActor("target", [(1.0, 0.0, 0.0, 0.0)])

        def get_instruction(self):
            return "put target in box"

        def get_role_names(self):
            return {"target_id": 10, "destination_id": 20}

        def _get_benchmark_object_catalog(self):
            return self.catalog

    with _graph_file(tmp_path / "graph.hdf5") as root:
        catalog, state, object_state = _live_inputs(root)
    state["held_by"] = np.zeros((5, 2), dtype=bool)
    state["held_by_valid"] = np.ones((5, 2), dtype=bool)
    state["in"] = np.zeros((5, 5), dtype=bool)
    state["containment_valid"] = np.ones((5, 5), dtype=bool)
    state["raw_contact"] = np.zeros((5, 5), dtype=bool)
    # Same close-ready position as the CLOSE case above: height-aligned,
    # within the close distance, no contact yet.
    object_state["pose_world"][3, :3] = [0.16, 0.20, 0.34]
    observation = {
        "benchmark_support": {"relation_state": state, "object_state": object_state},
    }
    contract = RetrievalContract()

    task = Task(catalog)
    aligned_context = build_live_graph_context(task, observation, contract)
    aligned_evidence = extract_simulator_evidence(aligned_context)
    assert aligned_evidence.grasp_substage is GraspSubstage.CLOSE
    assert aligned_evidence.left.grasp_orientation_aligned
    assert aligned_evidence.left.target_orientation_error_deg == 0.0

    # A contact point annotated 90 degrees away from the effector's actual
    # orientation is recorded as misaligned, but must NOT block CLOSE --
    # diagnostics-only, no gating -- since there is no corrective instruction
    # for a misaligned orientation and the fallback substages (move_closer)
    # give actively wrong advice for an orientation-only defect.
    task.target = _FakeGraspActor(
        "target", [(0.70710678, 0.70710678, 0.0, 0.0)]
    )
    misaligned_context = build_live_graph_context(task, observation, contract)
    misaligned_evidence = extract_simulator_evidence(misaligned_context)
    assert not misaligned_evidence.left.grasp_orientation_aligned
    assert misaligned_evidence.left.target_orientation_error_deg > 80.0
    assert misaligned_evidence.grasp_substage is GraspSubstage.CLOSE

    # No resolvable actor/annotation: fails open, identical to pre-existing
    # behavior for objects without annotated grasp geometry.
    del task.target
    unresolved_context = build_live_graph_context(task, observation, contract)
    unresolved_evidence = extract_simulator_evidence(unresolved_context)
    assert unresolved_evidence.left.grasp_orientation_aligned
    assert math.isnan(unresolved_evidence.left.target_orientation_error_deg)
    assert unresolved_evidence.grasp_substage is GraspSubstage.CLOSE


def test_grasp_orientation_error_covers_rotate_lim_arc_and_finger_swap_symmetry(tmp_path):
    """A grasp orientation elsewhere in the embodiment's rotate_lim arc, or
    the 180-degree finger-swap flip, must not read as misaligned -- both
    are orientations RoboTwin's own grasp-pose search treats as equally
    valid, not just the single annotated seed orientation."""

    class Robot:
        left_rotate_lim = [0.0, 1.0]
        right_rotate_lim = [0.0, 1.0]

    class Task:
        def __init__(self, catalog):
            self.catalog = catalog
            self.target = _FakeGraspActor("target", [(1.0, 0.0, 0.0, 0.0)])
            self.robot = Robot()

        def get_instruction(self):
            return "put target in box"

        def get_role_names(self):
            return {"target_id": 10, "destination_id": 20}

        def _get_benchmark_object_catalog(self):
            return self.catalog

    with _graph_file(tmp_path / "graph.hdf5") as root:
        catalog, state, object_state = _live_inputs(root)
    state["held_by"] = np.zeros((5, 2), dtype=bool)
    state["held_by_valid"] = np.ones((5, 2), dtype=bool)
    state["in"] = np.zeros((5, 5), dtype=bool)
    state["containment_valid"] = np.ones((5, 5), dtype=bool)
    state["raw_contact"] = np.zeros((5, 5), dtype=bool)
    object_state["pose_world"][3, :3] = [0.16, 0.20, 0.34]
    # 0.6 rad about the effector's own local Y (jaw-closing) axis: inside
    # the 0-1 rad rotate_lim arc, so a different-but-equally-valid grasp
    # orientation, not a bit-for-bit match of the annotated seed.
    within_arc_quat = _rotate_quat_about_own_local_axis(
        (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.6
    )
    object_state["pose_world"][3, 3:7] = within_arc_quat
    observation = {
        "benchmark_support": {"relation_state": state, "object_state": object_state},
    }
    contract = RetrievalContract()
    task = Task(catalog)

    within_arc_context = build_live_graph_context(task, observation, contract)
    within_arc_evidence = extract_simulator_evidence(within_arc_context)
    assert within_arc_evidence.left.grasp_orientation_aligned
    assert within_arc_evidence.left.target_orientation_error_deg < 1.0

    # 180 degrees about the local X (approach) axis: the finger-swap flip.
    flipped_quat = _rotate_quat_about_own_local_axis(
        (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0), math.pi
    )
    object_state["pose_world"][3, 3:7] = flipped_quat
    flipped_context = build_live_graph_context(task, observation, contract)
    flipped_evidence = extract_simulator_evidence(flipped_context)
    assert flipped_evidence.left.grasp_orientation_aligned
    assert flipped_evidence.left.target_orientation_error_deg < 1.0

    # A rotation about an unrelated axis (not the jaw axis, not the
    # approach-axis flip) is still correctly read as misaligned -- the
    # symmetry family is specific, not a free pass on any orientation.
    unrelated_quat = _rotate_quat_about_own_local_axis(
        (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 2
    )
    object_state["pose_world"][3, 3:7] = unrelated_quat
    unrelated_context = build_live_graph_context(task, observation, contract)
    unrelated_evidence = extract_simulator_evidence(unrelated_context)
    assert not unrelated_evidence.left.grasp_orientation_aligned
    assert unrelated_evidence.left.target_orientation_error_deg > 80.0


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
            goal_texts = [
                item["text"] for item in payload["items"]
                if item["section"] == "goal"
            ]
            return {
                "instruction": payload["instruction"] + (
                    "\n" + goal_texts[0] if goal_texts else ""
                ),
                "selected_node_count": 0,
                "selected_fact_count": len(goal_texts),
                "graph_token_count": 9 if goal_texts else 0,
                "full_prompt_token_count_estimate": 40 if goal_texts else 20,
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
    grasp = prepare_instruction(
        task,
        model,
        observation,
        InputCondition.VISUAL_RETRIEVED_GRAPH,
        RetrievalContract(),
    )
    assert grasp.instruction == (
        "Task objective: put target in box\n"
        "Current stage: Align the gripper with the target."
    )
    assert grasp.prompt_phase == "grasp"
    assert grasp.action_intent.operation is IntentOperation.GRASP
    assert grasp.action_intent.phase == grasp.prompt_phase
    assert grasp.retrieved_fact_count == 1
    assert grasp.selected_fact_count == 1
    assert grasp.selected_node_count == 0
    assert grasp.graph_token_count > 0
    assert grasp.destination_seed_available
    assert len(model.calls) == 1

    held = np.zeros((5, 2), dtype=bool)
    held[0, 0] = True
    state["held_by"] = held
    state["held_by_valid"] = np.ones_like(held)
    placement = prepare_instruction(
        task, model, observation, InputCondition.VISUAL_RETRIEVED_GRAPH,
        RetrievalContract(), previous_phase="placement",
    )
    assert placement.prompt_phase == "placement"
    assert placement.action_intent.operation is IntentOperation.PLACE
    assert placement.action_intent.phase == placement.prompt_phase
    assert placement.selected_fact_count == 1
    assert placement.instruction == (
        "Task objective: put target in box\n"
        "Current stage: Keep holding the object. "
        "Move it forward and left into the box."
    )
    assert len(model.calls) == 2
    held_event = live_task_state(task, observation, RetrievalContract())
    assert held_event.target_held and held_event.held_arm == "left"
    assert not held_event.target_inside_destination
    assert not held_event.release_ready
    items = model.calls[-1]["items"]
    assert len(items) == 1
    assert items[0]["section"] == "goal" and items[0]["mandatory"]
    assert not any(token in items[0]["text"] for token in ("T1", "D1", "0."))

    # A phase remains stable at a chunk boundary; the per-action controller
    # owns event-driven transitions and grasp-loss recovery.
    state["held_by"][:] = False
    latched = prepare_instruction(
        task, model, observation, InputCondition.VISUAL_RETRIEVED_GRAPH,
        RetrievalContract(), previous_phase=placement.prompt_phase,
    )
    assert latched.prompt_phase == "placement"
    assert len(model.calls) == 3

    state["in"] = np.zeros((5, 5), dtype=bool)
    state["containment_valid"] = np.ones((5, 5), dtype=bool)
    state["in"][0, 1] = True
    event = live_task_state(task, observation, RetrievalContract())
    assert not event.target_held
    assert event.held_arm is None
    assert event.target_inside_destination
    assert not event.release_ready

    release = prepare_instruction(
        task, model, observation, InputCondition.VISUAL_RETRIEVED_GRAPH,
        RetrievalContract(), previous_phase="release",
    )
    assert release.prompt_phase == "release"
    assert release.action_intent.operation is IntentOperation.RELEASE
    assert release.action_intent.phase == release.prompt_phase
    assert release.instruction == (
        "Task objective: put target in box\n"
        "Current stage: Release the held object in the box."
    )

    # Held above the container rim: not contained yet, but close enough that
    # gravity can finish insertion after a controlled release.
    state["in"][:] = False
    state["held_by"][0, 0] = True
    object_state["aabb_lower"][0] = [0.35, 0.45, 0.79]
    object_state["aabb_upper"][0] = [0.40, 0.50, 0.87]
    object_state["aabb_lower"][1] = [0.30, 0.40, 0.50]
    object_state["aabb_upper"][1] = [0.50, 0.60, 0.70]
    ready = live_task_state(task, observation, RetrievalContract())
    assert ready.target_held and not ready.target_inside_destination
    assert ready.release_ready

    object_state["aabb_lower"][0, 2] = 0.81
    object_state["aabb_upper"][0, 2] = 0.89
    too_high = live_task_state(task, observation, RetrievalContract())
    assert not too_high.release_ready

    # Job 10517 used center-only horizontal alignment: an object whose center
    # is inside the destination is release-ready even when its AABB touches an
    # edge.
    object_state["aabb_lower"][0] = [0.30, 0.45, 0.79]
    object_state["aabb_upper"][0] = [0.35, 0.50, 0.87]
    edge = live_task_state(task, observation, RetrievalContract())
    assert edge.release_ready


def test_grasp_hint_selects_only_a_valid_uniquely_clear_gripper(tmp_path):
    with _graph_file(tmp_path / "graph.hdf5") as root:
        catalog, state, object_state = _live_inputs(root)
    retriever = LiveGraphRetriever(
        catalog, state, object_state, RetrievalContract()
    )
    retriever.is_target = np.isin(retriever.object_ids, [10])

    # Left path blocked, right path valid and clear.
    state["blocks_by_effector"][:, 0, :] = False
    state["blocks_by_effector"][1, 0] = [True, False]
    state["blocks_by_effector_valid"][:, 0, :] = True
    # The box blocks the left corridor but is not imminent to the left
    # gripper, so choose the clear right arm without a distracting warning.
    assert compact_grasp_hint(retriever, 10) == (
        "Align the right gripper with the target."
    )

    # Only an imminent blocker produces a warning, and it is attributed to
    # the arm whose corridor it actually blocks.
    retriever._aabb_bounds[20] = (
        np.array([0.55, -0.10, 0.40]),
        np.array([0.65, 0.10, 0.60]),
    )
    assert compact_grasp_hint(retriever, 10) == (
        "Collision risk: the box blocks the left gripper. "
        "Align the right gripper with the target."
    )

    # Unknown opposite-side evidence must not be described as available.
    state["blocks_by_effector_valid"][:, 0, 1] = False
    assert compact_grasp_hint(retriever, 10) == "Align the gripper with the target."

    # Missing obstacle geometry preserves the approach-side instruction.
    state["blocks_by_effector"][1, 0] = [True, False]
    state["blocks_by_effector_valid"][:, 0, :] = True
    retriever._aabb_bounds.pop(20)
    assert compact_grasp_hint(retriever, 10) == (
        "Align the right gripper with the target."
    )

    # No preference when both paths are valid and clear.
    state["blocks_by_effector"][:, 0, :] = False
    state["blocks_by_effector_valid"][:, 0, :] = True
    assert compact_grasp_hint(retriever, 10) == "Align the gripper with the target."


def test_transport_gripper_latch_changes_only_the_active_channel():
    action = np.linspace(0.1, 1.4, 14)
    left = keep_active_gripper_closed(action, "left")
    right = keep_active_gripper_closed(action, "right")
    assert left[6] == 0.0 and np.array_equal(left[:6], action[:6])
    assert np.array_equal(left[7:], action[7:])
    assert right[13] == 0.0 and np.array_equal(right[:13], action[:13])
    assert action[6] != 0.0 and action[13] != 0.0  # input is never mutated

    with raises(ValueError):
        keep_active_gripper_closed(np.zeros(13), "left")


def test_destination_name_fallback_resolves_catalog_id_and_rejects_ambiguity():
    class Task:
        def get_role_names(self):
            return {"destination_object_names": ["basket_right"]}

    catalog = [
        {"object_id": 10, "name": "sauce_can", "semantic_label": "can"},
        {"object_id": 20, "name": "basket_right", "semantic_label": "basket"},
    ]
    assert destination_ids_from_task(Task(), catalog) == (20,)

    ambiguous = catalog + [
        {"object_id": 30, "name": "basket_right", "semantic_label": "basket"}
    ]
    with raises(ValueError):
        destination_ids_from_task(Task(), ambiguous)


def test_destination_ids_take_precedence_over_ambiguous_names():
    class Task:
        def get_role_names(self):
            return {
                "destination_ids": [30, 20],
                "destination_object_names": ["basket"],
            }

    catalog = [
        {"object_id": 20, "name": "basket"},
        {"object_id": 30, "name": "basket"},
    ]
    assert destination_ids_from_task(Task(), catalog) == (20, 30)


def test_vla_labels_hide_simulator_names_conservatively():
    assert vla_label_from_catalog_entry(
        {"name": "task_sauce_can", "semantic_label": "task_sauce_can"}
    ) == "sauce can"
    assert vla_label_from_catalog_entry(
        {"name": "basket_right", "semantic_label": "basket_right"}
    ) == "basket right"
    assert vla_label_from_catalog_entry(
        {"name": "model_red_bowl_3", "semantic_label": "model_red_bowl_3"}
    ) == "red bowl"
    assert vla_label_from_catalog_entry(
        {"name": "task_sauce_can", "semantic_label": "tomato sauce can"}
    ) == "tomato sauce can"
    assert vla_label_from_catalog_entry(
        {"name": "object_12", "semantic_label": "object_12"}
    ) == "object"


def test_aliases_are_stable_role_aware_and_duplicate_safe(tmp_path):
    with _graph_file(tmp_path / "graph.hdf5") as root:
        catalog, state, object_state = _live_inputs(root)
    catalog[0]["name"] = catalog[1]["name"] = "bowl"
    retriever = LiveGraphRetriever(catalog, state, object_state)
    nodes, facts = retriever.retrieve([20])
    aliases = {node.object_id: node.alias for node in nodes}
    assert aliases == {-3: "R", -2: "L", 10: "T1", 20: "D1"}
    assert {fact.source for fact in facts} | {fact.destination for fact in facts} >= {"T1", "D1", "L"}
    # Repeating retrieval does not renumber aliases based on the relation subset.
    nodes_again, _ = retriever.retrieve([20])
    assert {node.object_id: node.alias for node in nodes_again} == aliases


def test_dependency_safe_packing_is_atomic():
    target = GraphNode(10, "sauce can", "actor", (0.1, 0.2, 0.3), (0.1, 0.1, 0.2), "T1")
    destination = GraphNode(20, "basket", "actor", (0.4, 0.5, 0.6), (0.3, 0.2, 0.1), "D1")
    fact = GraphFact(2, "blocks", "D1", "T1", "", ("D1", "T1"))
    full = serialize_graph([target, destination], [fact], 100)
    assert full.selected_facts == (fact,)
    assert {node.alias for node in full.selected_nodes} == {"T1", "D1"}
    tight = serialize_graph([target, destination], [fact], full.token_count - 1)
    assert tight.selected_facts == ()
    assert {node.alias for node in tight.selected_nodes} == {"T1", "D1"}


def test_target_and_destination_nodes_are_mandatory_without_relations():
    target = GraphNode(10, "sauce can", "actor", (0.1, 0.2, 0.3), alias="T1")
    destination = GraphNode(20, "basket", "actor", (0.4, 0.5, 0.6), alias="D1")
    unrelated = GraphNode(30, "plate", "actor", (0.7, 0.8, 0.9), alias="O1")

    result = serialize_graph([target, destination, unrelated], [], 100)
    assert {node.alias for node in result.selected_nodes} == {"T1", "D1"}
    assert "T1 = sauce can" in result.text
    assert "D1 = basket" in result.text
    assert "O1 = plate" not in result.text

    with raises(ValueError):
        serialize_graph(
            [target, destination], [],
            token_budget=result.token_count - 1,
        )


def test_aliases_reduce_repeated_relation_tokens():
    aliased = [GraphFact(9, "near", "T1", "D1") for _ in range(3)]
    verbose = [GraphFact(9, "near", "sauce can instance 10", "large wicker basket instance 20") for _ in range(3)]
    from experiments.graph_conditioned_pi05.graph_serializer import conservative_token_count
    assert conservative_token_count(" ".join(map(format_fact, aliased))) < conservative_token_count(" ".join(map(format_fact, verbose)))


def main():
    test_contract_rejects_leakage_and_invalid_facts()
    with TemporaryDirectory() as directory:
        test_retrieval_is_deterministic_filtered_and_camera_specific(Path(directory))
    with TemporaryDirectory() as directory:
        test_explicit_seed_and_unknown_seed(Path(directory))
    test_serialization_priority_and_budget()
    test_packing_maximizes_same_tier_information_over_alphabetical_first_fit()
    test_serialize_graph_includes_nodes_with_grounding_attributes()
    test_goal_geometry_is_robot_relative_and_mandatory()
    test_mandatory_goal_and_dependencies_survive_tight_budget()
    test_natural_language_graph_renderer()
    with TemporaryDirectory() as directory:
        test_aliases_are_stable_role_aware_and_duplicate_safe(Path(directory))
    test_dependency_safe_packing_is_atomic()
    test_target_and_destination_nodes_are_mandatory_without_relations()
    test_aliases_reduce_repeated_relation_tokens()
    test_destination_name_fallback_resolves_catalog_id_and_rejects_ambiguity()
    test_destination_ids_take_precedence_over_ambiguous_names()
    test_vla_labels_hide_simulator_names_conservatively()
    with TemporaryDirectory() as directory:
        test_live_retrieval_matches_hdf5(Path(directory))
    with TemporaryDirectory() as directory:
        test_shared_live_context_has_value_parity_and_one_catalog_parse(Path(directory))
    with TemporaryDirectory() as directory:
        test_grasp_orientation_gate_uses_annotated_contact_pose(Path(directory))
    with TemporaryDirectory() as directory:
        test_grasp_orientation_error_covers_rotate_lim_arc_and_finger_swap_symmetry(Path(directory))
    with TemporaryDirectory() as directory:
        test_prepare_instruction_preserves_visual_only_and_fits_graph(Path(directory))
    with TemporaryDirectory() as directory:
        test_grasp_hint_selects_only_a_valid_uniquely_clear_gripper(Path(directory))
    test_transport_gripper_latch_changes_only_the_active_channel()
    with TemporaryDirectory() as directory:
        test_alignment_mismatch_fails(Path(directory))
    print("21 graph-context checks passed")


if __name__ == "__main__":
    main()
