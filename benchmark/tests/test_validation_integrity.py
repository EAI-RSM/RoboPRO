import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "benchmark/bench_script/run_action_validation_suite.py"
SPEC = importlib.util.spec_from_file_location("action_validation_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


class ValidationIntegrityTest(unittest.TestCase):
    def _relation_file(self, path, names=("source", "destination"), catalog_ids=(10, 20), relation_ids=(20, 10)):
        with h5py.File(path, "w") as root:
            support = root.create_group("benchmark_support")
            catalog = support.create_group("object_catalog")
            catalog.create_dataset("names", data=np.asarray(names, dtype="S32"))
            catalog.create_dataset("object_ids", data=np.asarray(catalog_ids, dtype=np.int64))
            state = support.create_group("relation_state")
            state.create_dataset("object_ids", data=np.asarray(relation_ids, dtype=np.int64))
            relation = np.zeros((1, 2, 2), dtype=np.bool_)
            relation[0, 1, 0] = True
            state.create_dataset("in", data=relation)

    def test_final_relation_uses_relation_object_id_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            self._relation_file(path)
            with h5py.File(path, "r") as root:
                self.assertTrue(RUNNER._final_relation_holds(
                    root["benchmark_support"],
                    {"relation": "in", "source_name": "source", "destination_name": "destination"},
                ))

    def test_duplicate_catalog_names_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            self._relation_file(path, names=("same", "same"))
            with h5py.File(path, "r") as root:
                with self.assertRaisesRegex(ValueError, "Acceptance object name 'same' is ambiguous"):
                    RUNNER._final_relation_holds(
                        root["benchmark_support"],
                        {"relation": "in", "source_name": "same", "destination_name": "same"},
                    )

    def test_duplicate_relation_ids_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            self._relation_file(path, relation_ids=(10, 10))
            with h5py.File(path, "r") as root:
                with self.assertRaisesRegex(ValueError, "Duplicate relation-state object id"):
                    RUNNER._final_relation_holds(
                        root["benchmark_support"],
                        {"relation": "in", "source_name": "source", "destination_name": "destination"},
                    )

    def test_blanket_xfail_matrix_raises(self):
        matrix = {
            "tasks": [{
                "id": "task", "task_name": "task", "episodes": 1,
                "expected_validation_failure": True,
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.yml"
            path.write_text(yaml.safe_dump(matrix), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blanket XFAIL is forbidden"):
                RUNNER._load_matrix(path)

    def test_corrupt_inverse_relation_makes_inspector_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.hdf5"
            with h5py.File(path, "w") as root:
                support = root.create_group("benchmark_support")
                state = support.create_group("relation_state")
                state.create_dataset("object_ids", data=np.asarray([10, 20], dtype=np.int64))
                inside = np.zeros((1, 2, 2), dtype=np.bool_)
                inside[0, 0, 1] = True
                state.create_dataset("in", data=inside)
                state.create_dataset("contains", data=np.zeros_like(inside))
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "benchmark/bench_script/inspect_benchmark_hdf5.py"),
                 "--file", str(path)],
                cwd=REPO_ROOT, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("contains is inverse of in", result.stderr)


    def test_sequence_requires_terminal_verify_success(self):
        failures = RUNNER._validate_action_sequence(
            ["grasp", "lift"], ["left", "left"], ["succeeded", "succeeded"],
            np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([10, 10]),
            np.asarray([True, True]),
        )
        self.assertIn("verify_success must occur exactly once as the terminal action", failures)

    def test_sequence_rejects_lift_before_grasp(self):
        failures = RUNNER._validate_action_sequence(
            ["lift", "verify_success"], ["left", "none"], ["succeeded", "succeeded"],
            np.asarray([0, 1]), np.asarray([0, 1]), np.asarray([10, -1]),
            np.asarray([True, False]),
        )
        self.assertTrue(any("lift occurs before a successful grasp" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
