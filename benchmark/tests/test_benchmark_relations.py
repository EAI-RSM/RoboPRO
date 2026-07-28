import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT / "customized_robotwin" / "envs" / "utils" / "benchmark_relations.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_relations", MODULE_PATH)
RELATIONS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RELATIONS)


def _aabb(minimum, maximum):
    return np.asarray(minimum, dtype=float), np.asarray(maximum, dtype=float)


class NearRelationTest(unittest.TestCase):
    def test_symmetric_with_false_diagonal(self):
        matrix = RELATIONS.compute_near_relations(
            np.asarray([10, 20]),
            {
                10: _aabb([0, 0, 0], [0.1, 0.1, 0.2]),
                20: _aabb([0.15, 0, 0], [0.25, 0.1, 0.2]),
            },
        )
        np.testing.assert_array_equal(matrix, matrix.T)
        self.assertFalse(np.diag(matrix).any())
        self.assertTrue(matrix[0, 1])

    def test_horizontal_threshold_is_inclusive(self):
        matrix = RELATIONS.compute_near_relations(
            np.asarray([1, 2]),
            {
                1: _aabb([0, 0, 0], [0.1, 0.1, 0.1]),
                2: _aabb([0.2, 0, 0], [0.3, 0.1, 0.1]),
            },
            horizontal_threshold_m=0.10,
        )
        self.assertTrue(matrix[0, 1])

    def test_horizontal_separation_beyond_threshold_is_not_near(self):
        matrix = RELATIONS.compute_near_relations(
            np.asarray([1, 2]),
            {
                1: _aabb([0, 0, 0], [0.1, 0.1, 0.1]),
                2: _aabb([0.201, 0, 0], [0.3, 0.1, 0.1]),
            },
            horizontal_threshold_m=0.10,
        )
        self.assertFalse(matrix[0, 1])

    def test_excessive_vertical_separation_is_not_near(self):
        matrix = RELATIONS.compute_near_relations(
            np.asarray([1, 2]),
            {
                1: _aabb([0, 0, 0], [0.1, 0.1, 0.1]),
                2: _aabb([0, 0, 0.30], [0.1, 0.1, 0.40]),
            },
        )
        self.assertFalse(matrix[0, 1])

    def test_missing_aabb_leaves_relation_false(self):
        matrix = RELATIONS.compute_near_relations(
            np.asarray([1, 2]),
            {1: _aabb([0, 0, 0], [0.1, 0.1, 0.1])},
        )
        self.assertFalse(matrix.any())


class OnSupportsRelationTest(unittest.TestCase):
    def test_contact_gated_inverse_relations(self):
        contact = np.asarray([[False, True], [True, False]])
        on, supports = RELATIONS.compute_on_supports_relations(
            np.asarray([1, 2]),
            {
                1: _aabb([0.02, 0.02, 0.10], [0.08, 0.08, 0.20]),
                2: _aabb([0, 0, 0], [0.1, 0.1, 0.11]),
            },
            contact,
        )
        self.assertTrue(on[0, 1])
        self.assertFalse(on[1, 0])
        np.testing.assert_array_equal(supports, on.T)

    def test_geometry_without_contact_is_not_on(self):
        on, supports = RELATIONS.compute_on_supports_relations(
            np.asarray([1, 2]),
            {
                1: _aabb([0, 0, 0.10], [0.1, 0.1, 0.20]),
                2: _aabb([0, 0, 0], [0.1, 0.1, 0.10]),
            },
            np.zeros((2, 2), dtype=np.bool_),
        )
        self.assertFalse(on.any())
        self.assertFalse(supports.any())

    def test_overlap_threshold_is_configurable(self):
        aabbs = {
            1: _aabb([0, 0, 0.10], [0.1, 0.1, 0.20]),
            2: _aabb([0.08, 0, 0], [0.18, 0.1, 0.10]),
        }
        contact = np.asarray([[False, True], [True, False]])
        permissive, _ = RELATIONS.compute_on_supports_relations(
            np.asarray([1, 2]), aabbs, contact, min_xy_overlap_ratio=0.19,
        )
        strict, _ = RELATIONS.compute_on_supports_relations(
            np.asarray([1, 2]), aabbs, contact, min_xy_overlap_ratio=0.21,
        )
        self.assertTrue(permissive[0, 1])
        self.assertFalse(strict[0, 1])

    def test_raw_contact_shape_is_validated(self):
        with self.assertRaisesRegex(ValueError, "raw_contact has shape"):
            RELATIONS.compute_on_supports_relations(
                np.asarray([1, 2]), {}, np.zeros((1, 1), dtype=np.bool_)
            )


class InContainsRelationTest(unittest.TestCase):
    def test_center_based_inverse_and_validity(self):
        catalog = [
            {"object_id": 1, "name": "bottle"},
            {"object_id": 2, "name": "basket"},
        ]
        inside, contains, valid, contains_valid = RELATIONS.compute_in_contains_relations(
            catalog,
            {
                1: _aabb([0.02, 0.02, 0.02], [0.08, 0.08, 0.08]),
                2: _aabb([0, 0, 0], [0.1, 0.1, 0.1]),
            },
        )
        self.assertTrue(inside[0, 1])
        self.assertTrue(valid[0, 1])
        self.assertFalse(valid[1, 0])
        np.testing.assert_array_equal(contains, inside.T)
        np.testing.assert_array_equal(contains_valid, valid.T)

    def test_center_not_full_extent_defines_containment(self):
        catalog = [
            {"object_id": 1, "name": "wide_object"},
            {"object_id": 2, "name": "box"},
        ]
        inside, _, _, _ = RELATIONS.compute_in_contains_relations(
            catalog,
            {
                1: _aabb([-0.1, -0.1, -0.1], [0.2, 0.2, 0.2]),
                2: _aabb([0, 0, 0], [0.1, 0.1, 0.1]),
            },
        )
        self.assertTrue(inside[0, 1])

    def test_tolerance_is_configurable(self):
        catalog = [
            {"object_id": 1, "name": "object"},
            {"object_id": 2, "name": "bin"},
        ]
        aabbs = {
            1: _aabb([0.1001, 0.04, 0.04], [0.1001, 0.06, 0.06]),
            2: _aabb([0, 0, 0], [0.1, 0.1, 0.1]),
        }
        strict, _, _, _ = RELATIONS.compute_in_contains_relations(
            catalog, aabbs, center_tolerance_m=0,
        )
        tolerant, _, _, _ = RELATIONS.compute_in_contains_relations(
            catalog, aabbs, center_tolerance_m=0.0002,
        )
        self.assertFalse(strict[0, 1])
        self.assertTrue(tolerant[0, 1])

    def test_missing_geometry_is_invalid(self):
        catalog = [
            {"object_id": 1, "name": "object"},
            {"object_id": 2, "name": "drawer"},
        ]
        inside, _, valid, _ = RELATIONS.compute_in_contains_relations(
            catalog, {2: _aabb([0, 0, 0], [1, 1, 1])},
        )
        self.assertFalse(inside.any())
        self.assertFalse(valid.any())


class HeldByRelationTest(unittest.TestCase):
    def _compute(self, *, closed=(True, False), contact=((True, False),), distance=0.1):
        return RELATIONS.compute_held_by_relations(
            np.asarray([10]),
            {10: np.asarray([0.0, 0.0, 0.0])},
            np.asarray([[distance, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.asarray(closed),
            np.asarray(contact),
            max_object_tcp_distance_m=0.16,
        )

    def test_closed_contact_within_threshold_is_held(self):
        held, valid, codes = self._compute()
        self.assertTrue(held[0, 0])
        self.assertTrue(valid.all())
        self.assertEqual(int(codes[0]), 0)

    def test_open_gripper_is_not_held(self):
        held, _, codes = self._compute(closed=(False, False))
        self.assertFalse(held.any())
        self.assertEqual(int(codes[0]), -1)

    def test_missing_contact_is_not_held(self):
        held, _, _ = self._compute(contact=((False, False),))
        self.assertFalse(held.any())

    def test_beyond_threshold_is_not_held(self):
        held, _, _ = self._compute(distance=0.161)
        self.assertFalse(held.any())

    def test_threshold_boundary_is_inclusive(self):
        held, _, _ = self._compute(distance=0.16)
        self.assertTrue(held[0, 0])

    def test_missing_center_is_invalid_not_held(self):
        held, valid, codes = RELATIONS.compute_held_by_relations(
            np.asarray([10]), {}, np.zeros((2, 3)), np.ones(2), np.ones((1, 2))
        )
        self.assertFalse(held.any())
        self.assertFalse(valid.any())
        self.assertEqual(int(codes[0]), -1)

    def test_left_right_independence_and_dual_arm_code(self):
        held, _, codes = RELATIONS.compute_held_by_relations(
            np.asarray([10, 20]),
            {
                10: np.asarray([0.0, 0.0, 0.0]),
                20: np.asarray([0.0, 0.0, 0.0]),
            },
            np.zeros((2, 3)),
            np.ones(2),
            np.asarray([[True, False], [True, True]]),
        )
        np.testing.assert_array_equal(
            held, np.asarray([[True, False], [True, True]])
        )
        np.testing.assert_array_equal(codes, np.asarray([0, 2], dtype=np.int8))

    def test_input_shapes_are_validated(self):
        with self.assertRaisesRegex(ValueError, "effector_positions has shape"):
            RELATIONS.compute_held_by_relations(
                np.asarray([10]), {}, np.zeros((1, 3)), np.zeros(2), np.zeros((1, 2))
            )
        with self.assertRaisesRegex(ValueError, "object_effector_contact has shape"):
            RELATIONS.compute_held_by_relations(
                np.asarray([10]), {}, np.zeros((2, 3)), np.zeros(2), np.zeros((1, 1))
            )

    def test_threshold_is_validated(self):
        for invalid in (-0.1, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    RELATIONS.compute_held_by_relations(
                        np.asarray([10]), {}, np.zeros((2, 3)), np.zeros(2),
                        np.zeros((1, 2)), max_object_tcp_distance_m=invalid,
                    )


class VisibleToRelationTest(unittest.TestCase):
    def test_counts_pixels_and_sorts_cameras(self):
        visible, valid, counts, cameras = RELATIONS.compute_visible_to_relations(
            np.asarray([10, 20]),
            {10: {1}, 20: {2, 3}},
            {
                "z_camera": np.asarray([[1, 2], [3, 0]]),
                "a_camera": np.asarray([[0, 0], [0, 1]]),
            },
        )
        self.assertEqual(cameras, ["a_camera", "z_camera"])
        np.testing.assert_array_equal(counts, [[1, 1], [0, 2]])
        np.testing.assert_array_equal(visible, counts >= 1)
        self.assertTrue(valid.all())

    def test_threshold_is_inclusive_and_configurable(self):
        segmentation = {"camera": np.asarray([[7, 7], [0, 0]])}
        strict, _, counts, _ = RELATIONS.compute_visible_to_relations(
            np.asarray([10]), {10: {7}}, segmentation, min_visible_pixel_count=3,
        )
        boundary, _, _, _ = RELATIONS.compute_visible_to_relations(
            np.asarray([10]), {10: {7}}, segmentation, min_visible_pixel_count=2,
        )
        self.assertEqual(int(counts[0, 0]), 2)
        self.assertFalse(strict[0, 0])
        self.assertTrue(boundary[0, 0])

    def test_missing_segmentation_or_ids_is_invalid(self):
        visible, valid, counts, _ = RELATIONS.compute_visible_to_relations(
            np.asarray([10, 20]), {10: {1}},
            {"missing": None, "present": np.asarray([[0]])},
        )
        self.assertFalse(visible.any())
        self.assertFalse(valid[:, 0].any())
        self.assertTrue(valid[0, 1])
        self.assertFalse(valid[1, 1])
        self.assertFalse(counts.any())

    def test_malformed_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            RELATIONS.compute_visible_to_relations(
                np.zeros((1, 1)), {}, {},
            )
        with self.assertRaisesRegex(ValueError, "at least two dimensions"):
            RELATIONS.compute_visible_to_relations(
                np.asarray([1]), {1: {1}}, {"camera": np.asarray([1])},
            )
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "integer >= 1"):
                    RELATIONS.compute_visible_to_relations(
                        np.asarray([1]), {}, {},
                        min_visible_pixel_count=invalid,
                    )


class OccludesRelationTest(unittest.TestCase):
    def setUp(self):
        self.object_ids = np.asarray([10, 20])
        self.aabbs = {
            10: (np.asarray([-0.2, -0.2, 1.0]), np.asarray([0.2, 0.2, 1.2])),
            20: (np.asarray([-0.3, -0.3, 2.0]), np.asarray([0.3, 0.3, 2.2])),
        }
        self.segmentation = {
            "camera": np.asarray([
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ])
        }
        self.depth = {"camera": np.ones((5, 5), dtype=np.float64)}
        self.camera = {
            "camera": {
                "intrinsic_cv": np.asarray([
                    [5.0, 0.0, 2.0],
                    [0.0, 5.0, 2.0],
                    [0.0, 0.0, 1.0],
                ]),
                "extrinsic_cv": np.eye(4),
            }
        }

    def test_front_object_occludes_back_target(self):
        edges, valid, counts, fractions, source_depth, target_depth, area, cameras = RELATIONS.compute_occludes_relations(
            self.object_ids,
            self.aabbs,
            {10: {1}, 20: {2}},
            self.segmentation,
            self.depth,
            self.camera,
        )
        self.assertEqual(cameras, ["camera"])
        self.assertTrue(valid[0, 1, 0])
        self.assertTrue(edges[0, 1, 0])
        self.assertGreaterEqual(counts[0, 1, 0], 1)
        self.assertGreater(fractions[0, 1, 0], 0)
        self.assertAlmostEqual(source_depth[0, 1, 0], 1.0)
        self.assertGreater(target_depth[1, 0], 1.0)
        self.assertGreater(area[1, 0], 0)
        self.assertFalse(edges[1, 0, 0])

    def test_sapien_three_by_four_extrinsic_is_supported(self):
        camera = {
            "camera": {
                "intrinsic_cv": self.camera["camera"]["intrinsic_cv"],
                "extrinsic_cv": np.eye(4)[:3, :],
            }
        }

        edges, valid, counts, fractions, source_depth, target_depth, area, cameras = RELATIONS.compute_occludes_relations(
            self.object_ids,
            self.aabbs,
            {10: {1}, 20: {2}},
            self.segmentation,
            self.depth,
            camera,
        )

        self.assertEqual(cameras, ["camera"])
        self.assertTrue(valid[0, 1, 0])
        self.assertTrue(edges[0, 1, 0])
        self.assertGreaterEqual(counts[0, 1, 0], 1)

    def test_depth_order_and_overlap_threshold_are_enforced(self):
        strict, _, counts, _, _, _, _, _ = RELATIONS.compute_occludes_relations(
            self.object_ids,
            self.aabbs,
            {10: {1}, 20: {2}},
            self.segmentation,
            self.depth,
            self.camera,
            min_overlap_pixel_count=10,
        )
        self.assertLess(counts[0, 1, 0], 10)
        self.assertFalse(strict.any())

    def test_missing_geometry_or_camera_is_invalid(self):
        _, valid, _, _, _, _, _, _ = RELATIONS.compute_occludes_relations(
            self.object_ids,
            {20: self.aabbs[20]},
            {10: {1}, 20: {2}},
            self.segmentation,
            self.depth,
            {},
        )
        self.assertFalse(valid.any())

    def test_parameters_are_validated(self):
        for invalid in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "integer >= 1"):
                RELATIONS.compute_occludes_relations(
                    self.object_ids, self.aabbs, {}, {}, {}, {},
                    min_overlap_pixel_count=invalid,
                )
        for invalid in (-1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                RELATIONS.compute_occludes_relations(
                    self.object_ids, self.aabbs, {}, {}, {}, {},
                    min_depth_margin_m=invalid,
                )

    def test_ineligible_target_is_unknown_not_negative(self):
        _, valid, _, _, _, _, _, _ = RELATIONS.compute_occludes_relations(
            self.object_ids,
            self.aabbs,
            {10: {1}, 20: {2}},
            self.segmentation,
            self.depth,
            self.camera,
            target_eligible=np.asarray([True, False]),
        )
        self.assertFalse(valid[:, 1, :].any())

    def test_overlap_fraction_is_configurable(self):
        edges, _, _, fractions, _, _, _, _ = RELATIONS.compute_occludes_relations(
            self.object_ids,
            self.aabbs,
            {10: {1}, 20: {2}},
            {"camera": np.asarray([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ])},
            self.depth,
            self.camera,
            min_overlap_fraction=0.5,
        )
        self.assertGreater(fractions[0, 1, 0], 0)
        self.assertFalse(edges.any())



class BlocksRelationTest(unittest.TestCase):
    def test_directional_corridor_obstruction_and_effector_evidence(self):
        blocks, valid, by_effector, valid_by_effector = (
            RELATIONS.compute_blocks_relations(
                np.asarray([10, 20]),
                {
                    10: _aabb([0.45, -0.05, -0.05], [0.55, 0.05, 0.05]),
                    20: _aabb([0.95, -0.05, -0.05], [1.05, 0.05, 0.05]),
                },
                np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                corridor_clearance_m=0.0,
                endpoint_margin_m=0.02,
            )
        )
        self.assertTrue(blocks[0, 1])
        self.assertFalse(blocks[1, 0])
        self.assertTrue(by_effector[0, 1, 0])
        self.assertFalse(by_effector[0, 1, 1])
        self.assertTrue(valid[0, 1])
        self.assertTrue(valid_by_effector[0, 1].all())

    def test_clearance_controls_near_miss(self):
        kwargs = dict(
            object_ids=np.asarray([10, 20]),
            aabb_by_id={
                10: _aabb([0.45, 0.03, -0.01], [0.55, 0.04, 0.01]),
                20: _aabb([0.95, -0.01, -0.01], [1.05, 0.01, 0.01]),
            },
            effector_positions=np.asarray([[0.0, 0.0, 0.0]]),
            endpoint_margin_m=0.0,
        )
        strict = RELATIONS.compute_blocks_relations(
            **kwargs, corridor_clearance_m=0.0
        )[0]
        padded = RELATIONS.compute_blocks_relations(
            **kwargs, corridor_clearance_m=0.03
        )[0]
        self.assertFalse(strict[0, 1])
        self.assertTrue(padded[0, 1])

    def test_missing_geometry_is_invalid_not_negative_evidence(self):
        blocks, valid, _, _ = RELATIONS.compute_blocks_relations(
            np.asarray([10, 20]),
            {20: _aabb([0.9, 0, 0], [1.0, 0.1, 0.1])},
            np.asarray([[0.0, 0.0, 0.0]]),
        )
        self.assertFalse(blocks.any())
        self.assertFalse(valid[0, 1])

    def test_parameters_are_validated(self):
        for name, value in (("corridor_clearance_m", -1), ("endpoint_margin_m", np.nan)):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    RELATIONS.compute_blocks_relations(
                        np.asarray([1]),
                        {1: _aabb([0, 0, 0], [1, 1, 1])},
                        np.zeros((1, 3)),
                        **{name: value},
                    )
class CollisionSemanticsTest(unittest.TestCase):
    def test_partition_precedence_and_completeness(self):
        collisions = np.zeros((6, 6), dtype=np.bool_)
        for i, j in ((0, 1), (2, 3), (0, 4), (4, 5)):
            collisions[i, j] = collisions[j, i] = True
        intentional_mask = np.zeros_like(collisions)
        intentional_mask[0, 1] = intentional_mask[1, 0] = True
        static, intentional, robot, unexpected, valid = (
            RELATIONS.classify_collision_semantics(
                collisions,
                np.asarray([True, False, False, False, False, False]),
                np.asarray([False, False, True, True, False, False]),
                intentional_mask,
            )
        )
        self.assertTrue(intentional[0, 1])
        self.assertTrue(static[2, 3])
        self.assertTrue(robot[0, 4])
        self.assertTrue(unexpected[4, 5])
        union = static | intentional | robot | unexpected
        np.testing.assert_array_equal(union, collisions)
        self.assertTrue(valid.all())
        self.assertFalse(
            np.any(
                static.astype(int) + intentional.astype(int)
                + robot.astype(int) + unexpected.astype(int) > 1
            )
        )

    def test_frame0_baseline_contact_is_static(self):
        contact = np.asarray([[False, True], [True, False]])
        static, intentional, robot, unexpected, _ = (
            RELATIONS.classify_collision_semantics(
                contact,
                np.asarray([False, False]),
                np.asarray([False, False]),
                np.zeros((2, 2), dtype=np.bool_),
                baseline_static_contact=contact,
            )
        )
        np.testing.assert_array_equal(static, contact)
        self.assertFalse(intentional.any())
        self.assertFalse(robot.any())
        self.assertFalse(unexpected.any())

    def test_intentional_contact_requires_collision_evidence(self):
        with self.assertRaisesRegex(ValueError, "backed by non_support_contact"):
            RELATIONS.classify_collision_semantics(
                np.zeros((2, 2), dtype=np.bool_),
                np.zeros(2), np.zeros(2),
                np.asarray([[False, True], [True, False]]),
            )


class PartOfRelationTest(unittest.TestCase):
    def test_direct_membership_and_closed_world_validity(self):
        relation, valid = RELATIONS.compute_part_of_relations(
            np.asarray([-3, -2, -1, 10]), {-3: -1, -2: -1},
        )
        expected = np.zeros((4, 4), dtype=np.bool_)
        expected[0, 2] = True
        expected[1, 2] = True
        np.testing.assert_array_equal(relation, expected)
        self.assertTrue(valid.all())

    def test_empty_hierarchy_is_known_false(self):
        relation, valid = RELATIONS.compute_part_of_relations(
            np.asarray([1, 2]), {},
        )
        self.assertFalse(relation.any())
        self.assertTrue(valid.all())

    def test_unknown_references_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "child id 3"):
            RELATIONS.compute_part_of_relations(np.asarray([1, 2]), {3: 1})
        with self.assertRaisesRegex(ValueError, "parent id 3"):
            RELATIONS.compute_part_of_relations(np.asarray([1, 2]), {1: 3})

    def test_self_membership_and_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "self-membership"):
            RELATIONS.compute_part_of_relations(np.asarray([1]), {1: 1})
        with self.assertRaisesRegex(ValueError, "unique"):
            RELATIONS.compute_part_of_relations(np.asarray([1, 1]), {})

    def test_object_ids_must_be_one_dimensional(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            RELATIONS.compute_part_of_relations(np.zeros((1, 1)), {})


class RelationRegistryTest(unittest.TestCase):
    def test_derived_relation_groups_are_consistent(self):
        canonical = set(RELATIONS.CANONICAL_RELATION_NAMES)
        implemented = set(RELATIONS.IMPLEMENTED_RELATION_NAMES)
        binary = set(RELATIONS.IMPLEMENTED_BINARY_RELATION_NAMES)
        bipartite = set(RELATIONS.IMPLEMENTED_BIPARTITE_RELATION_NAMES)
        camera_conditioned = set(
            RELATIONS.IMPLEMENTED_CAMERA_CONDITIONED_RELATION_NAMES
        )
        self.assertLessEqual(implemented, canonical)
        self.assertEqual(binary | bipartite | camera_conditioned, implemented)
        self.assertFalse(binary & bipartite)
        self.assertFalse(binary & camera_conditioned)
        self.assertFalse(bipartite & camera_conditioned)

    def test_snapshot_validation_rejects_missing_relation(self):
        with self.assertRaisesRegex(RuntimeError, "missing=.*near"):
            RELATIONS.serialize_and_validate_relations({}, object_count=0)

    def test_snapshot_validation_rejects_wrong_shape(self):
        values = {
            spec.name: np.zeros((2, 2), dtype=np.bool_)
            for spec in RELATIONS.RELATION_SPECS
            if spec.implemented
        }
        with self.assertRaisesRegex(RuntimeError, "first dimension 3"):
            RELATIONS.serialize_and_validate_relations(values, object_count=3)


if __name__ == "__main__":
    unittest.main()
