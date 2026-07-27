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


class RelationRegistryTest(unittest.TestCase):
    def test_derived_relation_groups_are_consistent(self):
        canonical = set(RELATIONS.CANONICAL_RELATION_NAMES)
        implemented = set(RELATIONS.IMPLEMENTED_RELATION_NAMES)
        binary = set(RELATIONS.IMPLEMENTED_BINARY_RELATION_NAMES)
        bipartite = set(RELATIONS.IMPLEMENTED_BIPARTITE_RELATION_NAMES)
        self.assertLessEqual(implemented, canonical)
        self.assertEqual(binary | bipartite, implemented)
        self.assertFalse(binary & bipartite)

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
