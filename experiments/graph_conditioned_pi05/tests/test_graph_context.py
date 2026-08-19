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
    GRASP_APPROACH_STANDOFF_M,
    LiveGraphRetriever,
    action_graph_state,
    build_live_graph_context,
    compact_grasp_hint,
    destination_ids_from_task,
    goal_geometry_pack_item,
    grasp_pose_from_contact_matrix,
    grasp_quat_wxyz_from_contact_matrix,
    keep_active_gripper_closed,
    keep_active_gripper_open,
    live_task_state,
    prepare_instruction,
    vla_label_from_catalog_entry,
)
from experiments.graph_conditioned_pi05.simulator_evidence import (
    CandidateMatch,
    EffectorEvidence,
    _close_geometry_ready,
    _final_approach_geometry_ready,
    _final_approach_substage,
    GRASP_TOWARDS_AXIS,
    expand_grasp_pose_family,
    extract_simulator_evidence,
    placement_geometry_from_bounds,
    _min_orientation_error_deg,
    _rotate_pose_about_point,
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
    assert near_evidence.grasp_substage is GraspSubstage.GRASP_APPROACH
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
    assert below_evidence.grasp_substage is GraspSubstage.GRASP_APPROACH
    assert below_evidence.grasp_arm == "left"

    object_state["pose_world"][3, :3] = [0.20, 0.20, 0.34]
    vicinity_context = build_live_graph_context(task, observation, contract)
    vicinity_evidence = extract_simulator_evidence(vicinity_context)
    assert vicinity_evidence.grasp_substage is GraspSubstage.GRASP_APPROACH
    assert vicinity_evidence.grasp_arm == "left"
    assert not vicinity_evidence.grasp_close_immediate

    # dist=9cm from the fallback candidate (0.1, 0.2, 0.34): inside the 10cm
    # close ceiling but past the 8cm immediate-close radius, so this needs
    # persistence rather than an immediate close.
    object_state["pose_world"][3, :3] = [0.19, 0.20, 0.34]
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

    Contact points are specified as the DESIRED post-transform grasp pose
    (orientation, and optionally position) -- what a real annotation's
    ``grasp_pose_from_contact_matrix`` would resolve to -- and converted
    back through the inverse of ``CONTACT_POINT_TO_GRASP_ROTATION`` and the
    approach standoff into the raw pre-transform contact matrix the real
    wrapped-Actor API returns. This exercises the same transform production
    code applies, rather than assuming it away by handing back an
    already-transformed pose.

    Position defaults to the target fixture's own AABB-top height (0.34, at
    an arbitrary X/Y) so tests that only care about orientation get a grasp
    height reference consistent with what the AABB-based fallback already
    gave them, and don't fail a height check they were never exercising.
    """

    _DEFAULT_POSITION_WORLD = (0.1, 0.2, 0.34)

    def __init__(self, name: str, grasp_quats_wxyz, grasp_position_world=None):
        self._name = name
        self._grasp_quats = list(grasp_quats_wxyz)
        self._grasp_position = (
            tuple(grasp_position_world)
            if grasp_position_world is not None
            else self._DEFAULT_POSITION_WORLD
        )

    def get_name(self):
        return self._name

    def iter_contact_points(self, ret="matrix"):
        assert ret == "matrix", "fake actor only supports the matrix format"
        inverse_contact_rotation = CONTACT_POINT_TO_GRASP_ROTATION.T
        standoff = np.array([-GRASP_APPROACH_STANDOFF_M, 0.0, 0.0])
        for index, quat in enumerate(self._grasp_quats):
            grasp_rotation = t3d.quaternions.quat2mat(quat)
            grasp_matrix = np.eye(4)
            grasp_matrix[:3, :3] = grasp_rotation
            # Invert grasp_pose_from_contact_matrix's standoff addition
            # (position = grasp_matrix[:3,3] + grasp_rotation @ standoff).
            grasp_matrix[:3, 3] = np.array(self._grasp_position) - grasp_rotation @ standoff
            yield index, grasp_matrix @ inverse_contact_rotation


def _homogeneous_matrix(rotation, bottom_row=(0.0, 0.0, 0.0, 1.0)):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[3, :] = bottom_row
    return matrix


def _orientation_family(seed_quat_wxyz, rotate_lim_rad):
    """Orientation-only view of expand_grasp_pose_family, for tests that
    only care about orientation behavior. Passing the same point as both
    the seed position and the contact center makes the rotate_lim arc's
    position rotation a self-rotation (offset zero), so position stays at
    the origin throughout and these tests exercise orientation exactly as
    they did before position was added to the pose family."""
    pose_family = expand_grasp_pose_family(
        (0.0, 0.0, 0.0), seed_quat_wxyz, (0.0, 0.0, 0.0), rotate_lim_rad
    )
    return tuple(orientation for _, orientation in pose_family)


def test_grasp_quat_from_contact_matrix_rejects_malformed_input():
    """transforms3d's mat2quat does not itself validate its input: it
    raises for a non-finite matrix (which would otherwise propagate out of
    build_live_graph_context and halt live evaluation on one bad
    annotation) and silently returns a normalized-looking but physically
    meaningless quaternion for a reflection, a degenerate matrix, or an
    arbitrary non-rotation matrix (which would otherwise read as an
    ordinary, seemingly-valid AVAILABLE reference orientation). Every one
    of these must come back as None instead, and a genuinely valid matrix
    must still work."""
    assert grasp_quat_wxyz_from_contact_matrix(
        _homogeneous_matrix(np.eye(3))
    ) is not None

    # Non-finite: must not raise (mat2quat itself raises LinAlgError on this).
    assert grasp_quat_wxyz_from_contact_matrix(
        _homogeneous_matrix(np.full((3, 3), np.nan))
    ) is None

    # A reflection (det=-1): a valid orthonormal matrix, but not a proper
    # rotation -- mat2quat returns a plausible unit quaternion for it with
    # no error at all, which is exactly the silent-corruption case.
    assert grasp_quat_wxyz_from_contact_matrix(
        _homogeneous_matrix(np.diag([-1.0, 1.0, 1.0]))
    ) is None

    # Degenerate (all-zero rotation block).
    assert grasp_quat_wxyz_from_contact_matrix(
        _homogeneous_matrix(np.zeros((3, 3)))
    ) is None

    # Arbitrary non-orthonormal matrix (not a rotation at all).
    assert grasp_quat_wxyz_from_contact_matrix(
        _homogeneous_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]))
    ) is None

    # Invalid homogeneous bottom row.
    assert grasp_quat_wxyz_from_contact_matrix(
        _homogeneous_matrix(np.eye(3), bottom_row=(1.0, 2.0, 3.0, 4.0))
    ) is None

    # Wrong shape and None must both still be handled, not just malformed
    # 4x4s.
    assert grasp_quat_wxyz_from_contact_matrix(np.eye(3)) is None
    assert grasp_quat_wxyz_from_contact_matrix(None) is None


def test_grasp_pose_position_applies_approach_standoff():
    """grasp_pose_from_contact_matrix's position must match
    get_grasp_pose's actual formula -- contact position offset by
    GRASP_APPROACH_STANDOFF_M along the local approach axis (X) of the
    POST-transform grasp orientation, not the raw contact orientation, and
    not just the bare contact-point translation. Hand-derived expected
    value for an identity contact matrix: the grasp rotation is exactly
    CONTACT_POINT_TO_GRASP_ROTATION's own rotation block, so the offset is
    that matrix's local -X column times the standoff distance."""
    identity_contact_matrix = np.eye(4)
    pose = grasp_pose_from_contact_matrix(identity_contact_matrix)
    assert pose is not None

    grasp_rotation = CONTACT_POINT_TO_GRASP_ROTATION[:3, :3]
    expected_position = grasp_rotation @ np.array(
        [-GRASP_APPROACH_STANDOFF_M, 0.0, 0.0]
    )
    assert np.allclose(pose.position_world, expected_position, atol=1e-9)


def test_grasp_height_reference_prefers_annotation_over_aabb_top(tmp_path):
    """When an annotated grasp pose is available, its actual position must be
    the reference CLOSE is gated on -- not the object's AABB top surface --
    and the two must be tested at heights far enough apart that the AABB-top
    reference would give the opposite answer, so this test cannot pass by
    accident if the annotation were silently ignored. Also confirms distance
    is measured against this SAME candidate, not the object's own pose/center
    -- an effector sitting exactly at the annotated grasp point must read as
    both height-aligned AND at ~zero distance, not "aligned but still far."
    """

    class Task:
        def __init__(self, catalog):
            self.catalog = catalog
            # Annotated grasp height 10cm below the AABB top (0.34): well
            # outside the +/-3cm tolerance of either reference around the
            # other, so aligning with one and not the other is unambiguous.
            self.target = _FakeGraspActor(
                "target",
                [(1.0, 0.0, 0.0, 0.0)],
                grasp_position_world=(0.16, 0.20, 0.24),
            )

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
    # Left effector at the annotated grasp height (0.24), NOT the AABB top
    # (0.34) -- the AABB-based reference alone would call this misaligned.
    object_state["pose_world"][3, :3] = [0.16, 0.20, 0.24]
    observation = {
        "benchmark_support": {"relation_state": state, "object_state": object_state},
    }
    contract = RetrievalContract()
    task = Task(catalog)

    context = build_live_graph_context(task, observation, contract)
    evidence = extract_simulator_evidence(context)
    # No rotate_lim configured (this fake Task has no .robot), so the arc
    # is a single sample at the seed itself -- two entries (the sample and
    # its position-neutral 180-degree flip), both at the annotated position.
    assert len(context.left_reference_grasp_positions_m) == 2
    assert all(
        np.allclose(position, (0.16, 0.20, 0.24), atol=1e-6)
        for position in context.left_reference_grasp_positions_m
    )
    assert evidence.left.grasp_height_aligned
    assert abs(evidence.left.target_vertical_offset_m) < 1e-6
    # The effector sits exactly at the annotated grasp point, so distance
    # (now measured against that same point, not the object's own pose at
    # (0.1, 0.2, 0.3)) must be ~0 -- not the ~10cm a stale object-center
    # reference would report despite a perfect grasp-pose alignment.
    assert evidence.left.target_distance_m < 1e-6
    assert context.left_reference_candidate_metadata == ((0, 0, 0), (0, 0, 1))
    assert evidence.left.selected_candidate_index == 0
    assert evidence.left.selected_contact_point_index == 0
    assert evidence.left.selected_arc_sample_index == 0
    assert evidence.left.selected_finger_flip_index == 0
    assert evidence.left.orientation_best_candidate.index == 0
    assert evidence.left.joint_best_candidate.index == 0
    assert evidence.left.joint_best_selection_status == "orientation_band_then_nearest"
    assert np.allclose(evidence.left.selected_candidate_error_world, (0, 0, 0), atol=1e-6)
    assert np.allclose(evidence.left.selected_candidate_error_local, (0, 0, 0), atol=1e-6)
    assert abs(evidence.left.target_horizontal_offset_m) < 1e-6


def test_grasp_distance_uses_grasp_pose_reference_not_object_center(tmp_path):
    """Regression test for a live-batch failure (seed 40002 in the
    graph_delta_annotated_grasp_height_reference_v3 evaluation): the
    annotated grasp point sat ~20cm horizontally from the object's own pose
    center, so the TCP reached and held a fully height- and
    orientation-aligned grasp pose for 300+ frames, but the OLD
    object-center-based target_distance_m plateaued at ~0.20m -- comfortably
    over GRASP_CLOSE_MAX_DISTANCE_M (0.10) -- so CLOSE never became
    eligible and the episode stalled. This places the effector exactly at
    an annotated grasp point 20cm from the object's own pose in x, and
    checks that CLOSE is reached; a pre-fix implementation using
    target_pose[:3] as the distance reference would compute ~0.20m here and
    fail this."""

    class Task:
        def __init__(self, catalog):
            self.catalog = catalog
            # Object's own pose stays at the _live_inputs default (0.1, 0.2,
            # 0.3); the annotated grasp point sits 20cm away in x -- e.g. a
            # side contact point on a large or off-center object.
            self.target = _FakeGraspActor(
                "target",
                [(1.0, 0.0, 0.0, 0.0)],
                grasp_position_world=(0.30, 0.20, 0.30),
            )

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
    state["raw_contact"][0, 3] = state["raw_contact"][3, 0] = True
    # Left effector exactly at the annotated grasp point -- a physically
    # ideal grasp -- NOT at the object's own pose (0.1, 0.2, 0.3).
    object_state["pose_world"][3, :3] = [0.30, 0.20, 0.30]
    observation = {
        "benchmark_support": {"relation_state": state, "object_state": object_state},
    }
    contract = RetrievalContract()
    task = Task(catalog)

    context = build_live_graph_context(task, observation, contract)
    evidence = extract_simulator_evidence(context)

    # The bug this regresses against: an object-center reference would give
    # norm((0.1, 0.2, 0.3) - (0.30, 0.20, 0.30)) = 0.20m here, well past the
    # close ceiling, despite the effector already being at the ideal point.
    stale_object_center_distance = float(
        np.linalg.norm(np.array([0.1, 0.2, 0.3]) - np.array([0.30, 0.20, 0.30]))
    )
    assert stale_object_center_distance > 0.15

    assert evidence.left.target_distance_m < 1e-6
    assert context.left_reference_candidate_metadata == ((0, 0, 0), (0, 0, 1))
    assert evidence.left.selected_candidate_index == 0
    assert evidence.left.selected_contact_point_index == 0
    assert evidence.left.selected_arc_sample_index == 0
    assert evidence.left.selected_finger_flip_index == 0
    assert evidence.left.orientation_best_candidate.index == 0
    assert evidence.left.joint_best_candidate.index == 0
    assert evidence.left.joint_best_selection_status == "orientation_band_then_nearest"
    assert np.allclose(evidence.left.selected_candidate_error_world, (0, 0, 0), atol=1e-6)
    assert np.allclose(evidence.left.selected_candidate_error_local, (0, 0, 0), atol=1e-6)
    assert evidence.left.grasp_height_aligned
    assert evidence.grasp_substage is GraspSubstage.CLOSE
    assert evidence.grasp_arm == "left"


def test_annotated_close_requires_calibrated_approach_axis_and_distance():
    def annotated(distance, approach_x, contact=False, lateral=0.0):
        return EffectorEvidence(
            target_distance_m=distance,
            joint_best_candidate=CandidateMatch(
                error_local=(approach_x, lateral, 0.0), distance_m=distance
            ),
            joint_best_selection_status="orientation_band_then_nearest",
            grasp_height_aligned=True,
            grasp_orientation_aligned=True,
            target_contact=contact,
        )

    assert not _close_geometry_ready(annotated(0.04, -0.001))
    assert not _close_geometry_ready(annotated(0.04, 0.004))
    assert _close_geometry_ready(annotated(0.04, 0.006))
    assert _close_geometry_ready(annotated(0.04, 0.020))
    assert not _close_geometry_ready(annotated(0.04, 0.021))
    assert not _close_geometry_ready(annotated(0.06, 0.0))
    assert _close_geometry_ready(annotated(0.08, -0.04, contact=True))
    assert not _close_geometry_ready(
        annotated(0.08, -0.04, contact=True, lateral=0.016)
    )
    assert _close_geometry_ready(annotated(0.04, 0.01, lateral=0.015))
    assert not _close_geometry_ready(annotated(0.04, 0.01, lateral=0.016))
    assert not _close_geometry_ready(
        annotated(0.08, -0.04, contact=True, lateral=0.031)
    )
    assert _final_approach_geometry_ready(annotated(0.04, 0.004))
    assert not _final_approach_geometry_ready(annotated(0.04, 0.006))
    assert _final_approach_geometry_ready(annotated(0.06, -0.03))
    assert not _final_approach_geometry_ready(annotated(0.081, -0.03))

    above = annotated(0.06, -0.03)
    above = EffectorEvidence(**{**above.__dict__,
        "grasp_height_aligned": False, "target_vertical_offset_m": 0.04})
    below = EffectorEvidence(**{**above.__dict__, "target_vertical_offset_m": -0.04})
    assert _final_approach_geometry_ready(above)
    assert _final_approach_substage(above) is GraspSubstage.FINAL_APPROACH_DOWN
    assert _final_approach_substage(below) is GraspSubstage.FINAL_APPROACH_UP
    assert _final_approach_substage(annotated(0.04, -0.019)) is GraspSubstage.FINAL_APPROACH


def test_placement_geometry_requires_safe_footprint_and_descent():
    destination_lower = np.array([0.0, 0.0, 0.0])
    destination_upper = np.array([0.30, 0.30, 0.20])
    edge = placement_geometry_from_bounds(
        np.array([0.01, 0.12, 0.23]), np.array([0.07, 0.18, 0.35]),
        destination_lower, destination_upper, "in",
    )
    assert not edge.aligned and not edge.descent_ready

    centered_above = placement_geometry_from_bounds(
        np.array([0.12, 0.12, 0.26]), np.array([0.18, 0.18, 0.38]),
        destination_lower, destination_upper, "in",
    )
    assert centered_above.aligned
    assert not centered_above.descent_ready
    assert centered_above.stage == "final_descent"

    centered_near_rim = placement_geometry_from_bounds(
        np.array([0.12, 0.12, 0.215]), np.array([0.18, 0.18, 0.335]),
        destination_lower, destination_upper, "in",
    )
    assert centered_near_rim.aligned and centered_near_rim.descent_ready
    assert np.isclose(centered_near_rim.target_bottom_to_destination_rim_m, 0.015)

    at_clearance = placement_geometry_from_bounds(
        np.array([0.12, 0.12, 0.25]), np.array([0.18, 0.18, 0.37]),
        destination_lower, destination_upper, "in",
    )
    assert at_clearance.aligned and at_clearance.descent_ready

    above_clearance = placement_geometry_from_bounds(
        np.array([0.12, 0.12, 0.251]), np.array([0.18, 0.18, 0.371]),
        destination_lower, destination_upper, "in",
    )
    assert above_clearance.aligned and not above_clearance.descent_ready

    centered_inside = placement_geometry_from_bounds(
        np.array([0.12, 0.12, 0.18]), np.array([0.18, 0.18, 0.30]),
        destination_lower, destination_upper, "in",
    )
    assert centered_inside.aligned and centered_inside.descent_ready
    assert np.isclose(centered_inside.target_bottom_to_destination_rim_m, -0.02)

    on_surface = placement_geometry_from_bounds(
        np.array([0.12, 0.12, 0.21]), np.array([0.18, 0.18, 0.33]),
        destination_lower, destination_upper, "on",
    )
    assert on_surface.aligned and on_surface.descent_ready


def test_unannotated_close_preserves_distance_only_fallback():
    assert _close_geometry_ready(EffectorEvidence(
        target_distance_m=0.09,
        joint_best_selection_status="position_only_fallback",
        grasp_height_aligned=True,
        grasp_orientation_aligned=True,
    ))
    assert not _close_geometry_ready(EffectorEvidence(
        target_distance_m=0.11,
        joint_best_selection_status="position_only_fallback",
        grasp_height_aligned=True,
        grasp_orientation_aligned=True,
    ))


def test_grasp_height_reference_falls_back_to_aabb_top_without_annotation(tmp_path):
    """Without a resolvable actor, height alignment must behave exactly as
    it did before this feature existed -- the AABB-top reference, same
    tolerance -- not silently disable the height check. Distance now shares
    that same fallback point too (object's own (x, y), AABB-top z), not the
    raw object pose independently -- so a fully height-aligned effector
    reports a small, coherent distance instead of one derived from a
    different z than the height check just confirmed was aligned."""

    class Task:
        def __init__(self, catalog):
            self.catalog = catalog

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
    # AABB top is 0.34 (see _live_inputs); effector right at it.
    object_state["pose_world"][3, :3] = [0.16, 0.20, 0.34]
    observation = {
        "benchmark_support": {"relation_state": state, "object_state": object_state},
    }
    contract = RetrievalContract()
    task = Task(catalog)

    context = build_live_graph_context(task, observation, contract)
    evidence = extract_simulator_evidence(context)
    assert context.orientation_reference_status == "actor_unresolved"
    assert context.left_reference_grasp_positions_m == ()
    assert context.right_reference_grasp_positions_m == ()
    assert evidence.left.grasp_height_aligned
    assert abs(evidence.left.target_vertical_offset_m) < 1e-6
    # Fallback candidate is (target's x, y, AABB-top z) = (0.1, 0.2, 0.34);
    # effector is (0.16, 0.2, 0.34) -- horizontal offset accounts for all of
    # the resulting distance, vertical offset for none of it.
    assert abs(evidence.left.target_distance_m - 0.06) < 1e-6
    assert abs(evidence.left.target_horizontal_offset_m - 0.06) < 1e-6


def test_grasp_orientation_gate_uses_annotated_contact_pose(tmp_path):
    """Orientation and position must describe one coherent control pose.

    Annotated candidates outside the 20-degree compatibility band must block
    CLOSE and return to generic ALIGN; missing annotations still fail open.
    """

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
    assert aligned_evidence.orientation_reference_status == "available"
    assert aligned_evidence.orientation_reference_count == 1

    # A contact point annotated 90 degrees away has no compatible joint
    # candidate: geometry-specific readiness must not use its attractive but
    # physically incoherent position, so control falls back to ALIGN.
    task.target = _FakeGraspActor(
        "target", [(0.70710678, 0.70710678, 0.0, 0.0)]
    )
    misaligned_context = build_live_graph_context(task, observation, contract)
    misaligned_evidence = extract_simulator_evidence(misaligned_context)
    assert not misaligned_evidence.left.grasp_orientation_aligned
    assert misaligned_evidence.left.target_orientation_error_deg > 80.0
    assert misaligned_evidence.grasp_substage is GraspSubstage.ALIGN

    # No resolvable actor: fails open, identical to pre-existing behavior
    # for objects without annotated grasp geometry -- but the status must
    # say it's an actor-resolution failure, not "no annotation," since we
    # don't actually know the object lacks one, only that we couldn't find
    # its wrapped Actor.
    del task.target
    unresolved_context = build_live_graph_context(task, observation, contract)
    unresolved_evidence = extract_simulator_evidence(unresolved_context)
    assert unresolved_evidence.left.grasp_orientation_aligned
    assert math.isnan(unresolved_evidence.left.target_orientation_error_deg)
    assert unresolved_evidence.grasp_substage is GraspSubstage.CLOSE
    assert unresolved_evidence.orientation_reference_status == "actor_unresolved"
    assert unresolved_evidence.orientation_reference_count == 0


def test_grasp_orientation_arc_excludes_upper_endpoint_like_robotwin():
    """create_target_pose_list (customized_robotwin/envs/robot/robot.py)
    samples step * i for i in range(ROTATE_NUM) -- a half-open grid that
    never reaches the arc's upper endpoint. A candidate exactly at that
    endpoint is an orientation RoboTwin's own search never generates, so it
    must not read as aligned; the last real sample (one step short of the
    endpoint) must."""
    seed = (1.0, 0.0, 0.0, 0.0)
    rotate_lim = (0.0, 1.0)
    family = _orientation_family(seed, rotate_lim)
    assert len(family) == 20  # 10 arc samples x 2 (seed + 180-degree flip)

    exact_endpoint = _rotate_quat_about_own_local_axis(seed, (0.0, 1.0, 0.0), 1.0)
    assert _min_orientation_error_deg(np.array(exact_endpoint), family) > 1.0

    last_real_sample = _rotate_quat_about_own_local_axis(seed, (0.0, 1.0, 0.0), 0.9)
    assert _min_orientation_error_deg(np.array(last_real_sample), family) < 1e-6


def test_grasp_orientation_finger_flip_applies_to_each_rotated_candidate():
    """The 180-degree finger-swap flip must be composed AFTER each
    rotate_lim arc candidate, not applied to the unrotated seed before
    rotating -- 3D rotations don't commute, so rotate-then-flip and
    flip-then-rotate are different orientations for any nonzero arc angle.
    A finger-swapped version of the theta=0.6 rad candidate specifically
    (not the flip of the seed itself, and not the rotation of the
    already-flipped seed) must read as aligned, not just the flip at
    theta=0 or a bare rotation with no flip -- the combination is exactly
    what a same-order-only bug would miss."""
    seed = (1.0, 0.0, 0.0, 0.0)
    rotate_lim = (0.0, 1.0)
    family = _orientation_family(seed, rotate_lim)

    rotated = _rotate_quat_about_own_local_axis(seed, (0.0, 1.0, 0.0), 0.6)
    correctly_flipped = _rotate_quat_about_own_local_axis(
        rotated, (1.0, 0.0, 0.0), math.pi
    )
    assert _min_orientation_error_deg(np.array(correctly_flipped), family) < 1e-6

    # The wrong composition order (flip the seed first, then rotate the
    # flipped seed by the same theta) produces a materially different
    # orientation -- confirming this test would have caught the bug, not
    # just restated the fix.
    flip_then_rotate = _rotate_quat_about_own_local_axis(
        _rotate_quat_about_own_local_axis(seed, (1.0, 0.0, 0.0), math.pi),
        (0.0, 1.0, 0.0),
        0.6,
    )
    assert (
        _min_orientation_error_deg(np.array(flip_then_rotate), (correctly_flipped,))
        > 1.0
    )


def test_grasp_orientation_arc_preserves_configured_rotate_lim_order():
    """create_target_pose_list never reorders rotate_lim -- it uses the
    signed step (rotate_lim[1] - rotate_lim[0]) / ROTATE_NUM directly, so a
    reversed config like (1.0, 0.0) walks 1.0, 0.9, ..., 0.1 and excludes
    0.0, not 1.0. Sorting the limits first (as if only the numeric range
    mattered) would silently swap which endpoint is excluded relative to
    what was actually configured -- moot for every embodiment config in
    this repo today (all ascending), but not for a hypothetical one."""
    seed = (1.0, 0.0, 0.0, 0.0)
    family = _orientation_family(seed, (1.0, 0.0))

    # rotate_lim[0]=1.0 is the reversed arc's start point: included.
    start_point = _rotate_quat_about_own_local_axis(seed, (0.0, 1.0, 0.0), 1.0)
    assert _min_orientation_error_deg(np.array(start_point), family) < 1e-6

    # rotate_lim[1]=0.0 is the reversed arc's (never-reached) endpoint:
    # excluded, the same way the ascending case excludes its own endpoint.
    excluded_endpoint = _rotate_quat_about_own_local_axis(seed, (0.0, 1.0, 0.0), 0.0)
    assert _min_orientation_error_deg(np.array(excluded_endpoint), family) > 1.0


def test_grasp_pose_family_rotates_position_not_just_orientation():
    """create_target_pose_list rotates the seed's POSITION as an offset
    from the raw contact center, not just its orientation in place: the
    offset from center to the seed has magnitude GRASP_APPROACH_STANDOFF_M
    (12cm), so sweeping the rotate_lim arc moves the candidate's height by
    up to ~STANDOFF*sin(rotate_lim range) -- close to 10cm for a ~1 radian
    arc. An earlier version of this code treated every arc candidate's
    height as identical to the unrotated seed's, which is wrong by exactly
    this amount -- far larger than any reasonable height tolerance."""
    standoff = GRASP_APPROACH_STANDOFF_M
    seed_position = (-standoff, 0.0, 0.0)
    seed_orientation = (1.0, 0.0, 0.0, 0.0)
    contact_center = (0.0, 0.0, 0.0)

    family = expand_grasp_pose_family(
        seed_position, seed_orientation, contact_center, (0.0, 1.0)
    )
    heights = [position[2] for position, _ in family]

    # The theta=0 candidate must still be exactly the unrotated seed height.
    assert any(abs(h - seed_position[2]) < 1e-9 for h in heights)
    # But the arc must ALSO produce heights far from the seed's -- close to
    # the reviewer's ~10cm estimate for this standoff and range, not
    # clustered near zero as they would be if position were frozen at the
    # seed's own height across the whole arc.
    assert max(heights) - min(heights) > 0.09


def test_grasp_pose_family_applies_towards_sign_disambiguation():
    """create_target_pose_list never just uses the configured +theta: it
    computes that candidate first and, if it lands on the wrong side of the
    contact center (a negative dot product against towards=[0,-1,0]),
    discards it and recomputes the WHOLE candidate -- position and
    orientation -- from -theta instead. Getting the sign wrong moves a
    12cm-offset candidate's height by several cm, same order of magnitude
    as the bug test_grasp_pose_family_rotates_position_not_just_orientation
    regression-tests above, and unlike that fix this one flips which
    physical candidate is even generated, not just its position along a
    fixed direction.

    Geometry here is a synthetic worst case (chosen so the +theta candidate
    provably lands on the wrong side), not a claim about any real
    RoboTwin object; the seed orientation maps local Y to world X so the
    rotated offset's world-Y component (what towards actually measures)
    varies with theta instead of staying pinned at its unrotated value.
    """
    # Local Y -> world X, so rotating about (this seed's) local Y sweeps the
    # offset's Y/Z components -- unlike the identity-orientation test above,
    # where local Y already equals the rotation axis and the offset's Y
    # component is invariant under the rotation (so towards never fires).
    seed_orientation = tuple(
        float(v) for v in t3d.quaternions.axangle2quat((0.0, 0.0, 1.0), -math.pi / 2)
    )
    contact_center = (0.0, 0.0, 0.0)
    seed_position = (0.0, -0.05, -0.12)
    theta = 0.6

    naive_position, naive_orientation = _rotate_pose_about_point(
        seed_position, seed_orientation, contact_center, (0.0, 1.0, 0.0), theta
    )
    # Precondition: this synthetic geometry must actually trigger the rule
    # (the naive +theta candidate must land on the wrong side of the
    # contact center), or the rest of this test would trivially pass for
    # the wrong reason.
    assert np.dot(np.array(naive_position) - np.array(contact_center), GRASP_TOWARDS_AXIS) < 0

    expected_position, expected_orientation = _rotate_pose_about_point(
        seed_position, seed_orientation, contact_center, (0.0, 1.0, 0.0), -theta
    )

    family = expand_grasp_pose_family(
        seed_position, seed_orientation, contact_center, (theta, theta)
    )
    arc_position, arc_orientation = family[0]

    # The expanded family must contain the sign-corrected (-theta)
    # candidate, matching RoboTwin's own resolution exactly...
    assert all(
        abs(a - b) < 1e-9 for a, b in zip(arc_position, expected_position)
    )
    assert all(
        abs(a - b) < 1e-9 for a, b in zip(arc_orientation, expected_orientation)
    )
    # ...and must NOT contain the rejected +theta candidate RoboTwin itself
    # never generates for this geometry.
    assert any(abs(a - b) > 1e-6 for a, b in zip(arc_position, naive_position))
    # The height difference between the accepted and rejected candidates is
    # material -- several cm, not noise -- confirming a wrong sign choice
    # here is exactly the kind of error the height-reference gate cares
    # about, not a cosmetic discrepancy.
    assert abs(arc_position[2] - naive_position[2]) > 0.02


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


def test_grasp_orientation_error_is_genuinely_arm_specific(tmp_path):
    """Different rotate_lim per arm must produce different orientation
    errors for the SAME object and the SAME effector rotation -- proving
    left_reference_orientations_wxyz and right_reference_orientations_wxyz
    are genuinely separate tuples, not just structurally split but
    numerically identical (as they'd be if both arms shared one rotate_lim,
    as aloha-agilex happens to)."""

    class Robot:
        left_rotate_lim = [0.0, 1.0]
        right_rotate_lim = [0.0, 0.0]

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
    # The SAME 0.6-rad rotation applied to both effectors -- only the
    # arm-specific rotate_lim should determine whether it reads as aligned.
    same_quat = _rotate_quat_about_own_local_axis(
        (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.6
    )
    object_state["pose_world"][3, 3:7] = same_quat  # left effector
    object_state["pose_world"][4, 3:7] = same_quat  # right effector
    observation = {
        "benchmark_support": {"relation_state": state, "object_state": object_state},
    }
    contract = RetrievalContract()
    task = Task(catalog)

    context = build_live_graph_context(task, observation, contract)
    evidence = extract_simulator_evidence(context)
    assert evidence.left.grasp_orientation_aligned
    assert evidence.left.target_orientation_error_deg < 1.0
    assert not evidence.right.grasp_orientation_aligned
    assert evidence.right.target_orientation_error_deg > 20.0
    assert (
        context.left_reference_orientations_wxyz
        != context.right_reference_orientations_wxyz
    )


def test_orientation_reference_status_distinguishes_failure_modes(tmp_path):
    """An empty reference tuple must not be the only signal available: a
    goal that couldn't resolve to a single target id, a resolved target
    whose wrapped Actor can't be found at all, a genuinely un-annotated
    (but resolved) actor, a bug raised while iterating contact points, and
    an actor whose contact points are all malformed must all read as
    different statuses -- otherwise a smoke test could silently collect
    nothing but NaNs for an entire batch and look identical to "nothing to
    check." In particular, actor-resolution failure must NOT be mislabeled
    as "no annotation": we don't know the object lacks one, only that we
    couldn't find its wrapped Actor -- possibly because of a bug in the
    lookup itself, which is exactly the class of failure this status field
    exists to surface."""

    class ExtractionErrorActor:
        def get_name(self):
            return "target"

        def iter_contact_points(self, ret="matrix"):
            raise RuntimeError("boom")

    class EmptyAnnotationActor:
        def get_name(self):
            return "target"

        def iter_contact_points(self, ret="matrix"):
            return iter(())

    class InvalidAnnotationActor:
        def get_name(self):
            return "target"

        def iter_contact_points(self, ret="matrix"):
            yield 0, np.zeros((3, 3))  # malformed: not a 4x4 pose matrix

    class Task:
        def __init__(self, catalog, target=None, multi_target=False):
            self.catalog = catalog
            self.target = target
            self._multi_target = multi_target

        def get_instruction(self):
            return "put target in box"

        def get_role_names(self):
            if self._multi_target:
                return {"target_ids": [10, 20], "destination_id": 20}
            return {"target_id": 10, "destination_id": 20}

        def _get_benchmark_object_catalog(self):
            return self.catalog

    with _graph_file(tmp_path / "graph.hdf5") as root:
        catalog, state, object_state = _live_inputs(root)
    observation = {
        "benchmark_support": {"relation_state": state, "object_state": object_state},
    }
    contract = RetrievalContract()

    # target_unresolved: the goal doesn't resolve to exactly one target id,
    # so a target name is never even looked up.
    multi_target_task = Task(catalog, multi_target=True)
    multi_target_context = build_live_graph_context(multi_target_task, observation, contract)
    assert multi_target_context.orientation_reference_status == "target_unresolved"
    assert multi_target_context.orientation_reference_count == 0

    # actor_unresolved: the target name resolves fine, but no object in
    # task_env matches it via _resolve_wrapped_actor -- distinct from
    # "no annotation," since this could just as easily be a bug in the
    # lookup itself as a genuine absence.
    no_actor_task = Task(catalog, target=None, multi_target=False)
    no_actor_context = build_live_graph_context(no_actor_task, observation, contract)
    assert no_actor_context.orientation_reference_status == "actor_unresolved"
    assert no_actor_context.orientation_reference_count == 0

    # extraction_error: the actor resolves, but iterating its contact points
    # raises -- a bug/incompatibility, not "no annotation."
    error_task = Task(catalog, target=ExtractionErrorActor())
    error_context = build_live_graph_context(error_task, observation, contract)
    assert error_context.orientation_reference_status == "extraction_error"
    assert error_context.orientation_reference_count == 0

    # annotation_missing: the actor resolves but has zero annotated contact
    # points.
    empty_task = Task(catalog, target=EmptyAnnotationActor())
    empty_context = build_live_graph_context(empty_task, observation, contract)
    assert empty_context.orientation_reference_status == "annotation_missing"
    assert empty_context.orientation_reference_count == 0

    # annotation_invalid: the actor resolves and has contact points, but
    # every one is malformed, so no usable orientation exists despite the
    # annotation technically being present.
    invalid_task = Task(catalog, target=InvalidAnnotationActor())
    invalid_context = build_live_graph_context(invalid_task, observation, contract)
    assert invalid_context.orientation_reference_status == "annotation_invalid"
    assert invalid_context.orientation_reference_count == 0


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

    # Even when the alternate-arm blocker is physically close, it remains
    # selection evidence rather than distracting active-arm language.
    retriever._aabb_bounds[20] = (
        np.array([0.55, -0.10, 0.40]),
        np.array([0.65, 0.10, 0.60]),
    )
    assert compact_grasp_hint(retriever, 10) == (
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

    approach_left = keep_active_gripper_open(np.zeros(14), "left")
    approach_right = keep_active_gripper_open(np.zeros(14), "right")
    assert approach_left[6] == 1.0
    assert np.array_equal(approach_left[7:], np.zeros(7))
    assert approach_right[13] == 1.0
    assert np.array_equal(approach_right[:13], np.zeros(13))

    with raises(ValueError):
        keep_active_gripper_closed(np.zeros(13), "left")
    with raises(ValueError):
        keep_active_gripper_open(np.zeros(13), "left")


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
    ) == "basket"
    assert vla_label_from_catalog_entry(
        {"name": "model_red_bowl_3", "semantic_label": "model_red_bowl_3"}
    ) == "red bowl"
    assert vla_label_from_catalog_entry(
        {"name": "task_sauce_can", "semantic_label": "tomato sauce can"}
    ) == "tomato sauce can"
    assert vla_label_from_catalog_entry(
        {"name": "legacy_bowl", "semantic_label": "left blue bowl"}
    ) == "left blue bowl"
    assert vla_label_from_catalog_entry(
        {"name": "bright_cup", "semantic_label": "bright_cup"}
    ) == "bright cup"
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
    test_grasp_quat_from_contact_matrix_rejects_malformed_input()
    test_grasp_pose_position_applies_approach_standoff()
    with TemporaryDirectory() as directory:
        test_grasp_height_reference_prefers_annotation_over_aabb_top(Path(directory))
    with TemporaryDirectory() as directory:
        test_grasp_distance_uses_grasp_pose_reference_not_object_center(Path(directory))
    with TemporaryDirectory() as directory:
        test_grasp_height_reference_falls_back_to_aabb_top_without_annotation(Path(directory))
    with TemporaryDirectory() as directory:
        test_grasp_orientation_gate_uses_annotated_contact_pose(Path(directory))
    test_grasp_orientation_arc_excludes_upper_endpoint_like_robotwin()
    test_grasp_orientation_finger_flip_applies_to_each_rotated_candidate()
    test_grasp_orientation_arc_preserves_configured_rotate_lim_order()
    test_grasp_pose_family_rotates_position_not_just_orientation()
    test_grasp_pose_family_applies_towards_sign_disambiguation()
    with TemporaryDirectory() as directory:
        test_grasp_orientation_error_covers_rotate_lim_arc_and_finger_swap_symmetry(Path(directory))
    with TemporaryDirectory() as directory:
        test_grasp_orientation_error_is_genuinely_arm_specific(Path(directory))
    with TemporaryDirectory() as directory:
        test_orientation_reference_status_distinguishes_failure_modes(Path(directory))
    with TemporaryDirectory() as directory:
        test_prepare_instruction_preserves_visual_only_and_fits_graph(Path(directory))
    with TemporaryDirectory() as directory:
        test_grasp_hint_selects_only_a_valid_uniquely_clear_gripper(Path(directory))
    test_transport_gripper_latch_changes_only_the_active_channel()
    with TemporaryDirectory() as directory:
        test_alignment_mismatch_fails(Path(directory))
    print("33 graph-context checks passed")


if __name__ == "__main__":
    main()
