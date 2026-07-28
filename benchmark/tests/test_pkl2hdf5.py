import tempfile
import unittest

import h5py
import numpy as np

from envs.utils.pkl2hdf5 import create_hdf5_from_dict


class Hdf5DepthRoutingTest(unittest.TestCase):
    def test_graph_depth_evidence_remains_numeric(self):
        evidence = [
            np.asarray([[1.25, np.nan], [0.75, 2.0]], dtype=np.float32),
            np.asarray([[1.5, 0.5], [np.nan, 2.25]], dtype=np.float32),
        ]
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as temporary:
            with h5py.File(temporary.name, "w") as root:
                create_hdf5_from_dict(
                    root,
                    {"occlusion_source_depth_m": evidence},
                )
            with h5py.File(temporary.name, "r") as root:
                stored = root["occlusion_source_depth_m"][()]

        self.assertEqual(stored.shape, (2, 2, 2))
        self.assertEqual(stored.dtype, np.dtype(np.float32))
        self.assertTrue(np.isnan(stored[0, 0, 1]))
        self.assertAlmostEqual(float(stored[1, 1, 1]), 2.25)

    def test_camera_depth_leaf_is_still_png_encoded(self):
        depth = [
            np.asarray([[1000.0, np.nan], [np.inf, -np.inf]], dtype=np.float64),
        ]
        with tempfile.NamedTemporaryFile(suffix=".hdf5") as temporary:
            with h5py.File(temporary.name, "w") as root:
                create_hdf5_from_dict(root, {"depth": depth})
            with h5py.File(temporary.name, "r") as root:
                dataset = root["depth"]
                self.assertEqual(dataset.shape, (1,))
                self.assertEqual(dataset.dtype.kind, "S")


if __name__ == "__main__":
    unittest.main()
