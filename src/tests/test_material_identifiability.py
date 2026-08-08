import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from utils.material_identifiability import (
    AuditSettings,
    NUISANCE_COLUMNS,
    PRIMARY_RESPONSE_COLUMNS,
    RESPONSE_COLUMNS,
    STATIC_COLUMNS,
    RecordValidationError,
    read_h5_record,
)


def find_row(rows, **criteria):
    matches = [
        row for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def serialise_folds(folds):
    return [(train.tolist(), test.tolist()) for train, test in folds]


class MaterialRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_h5(self, path: Path, *, frames: int = 25, include_dynamics: bool = True):
        cube = np.asarray([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float32)
        x = np.stack([cube + np.asarray([0, -0.01 * t, 0]) for t in range(frames)])
        with h5py.File(path, "w") as handle:
            handle["x"] = x
            handle["vol"] = np.ones(len(cube), dtype=np.float32) / len(cube)
            handle["E"] = np.asarray(1e6)
            handle["nu"] = np.asarray(0.3)
            handle["mat_type"] = np.asarray(1)
            handle["gravity"] = np.asarray(1)
            handle["floor_height"] = np.asarray(-0.1)
            handle["drag_force"] = np.zeros((0, 3), dtype=np.float32)
            handle["drag_mask"] = np.zeros((0, len(cube)), dtype=np.float32)
            if include_dynamics:
                handle["v"] = np.zeros_like(x)
                eye = np.eye(3, dtype=np.float32)
                handle["F"] = np.broadcast_to(eye, (frames, len(cube), 3, 3))
                handle["C"] = np.zeros((frames, len(cube), 3, 3), dtype=np.float32)

    def test_test_record_reads_only_static_frame_without_dynamics(self):
        test_path = self.root / "test_static.h5"
        self.write_h5(test_path, frames=1, include_dynamics=False)

        record = read_h5_record(test_path, split="test", settings=AuditSettings())

        self.assertEqual(record["split"], "test")
        self.assertTrue(all(column in record for column in STATIC_COLUMNS))
        self.assertNotIn("centered_shape_mse_f24", record)
        self.assertTrue(all(column not in record for column in RESPONSE_COLUMNS))
        self.assertNotIn("future_contact_fraction", NUISANCE_COLUMNS)

    def test_train_record_extracts_static_and_dynamic_features(self):
        train_path = self.root / "train_dynamic.h5"
        self.write_h5(train_path, frames=25, include_dynamics=True)

        record = read_h5_record(train_path, split="train", settings=AuditSettings())

        self.assertEqual(record["model"], "train_dynamic.h5")
        self.assertAlmostEqual(record["log10_e"], 6.0)
        self.assertIn("centered_shape_mse_f24", record)
        self.assertIn("future_contact_fraction", RESPONSE_COLUMNS)
        self.assertTrue(all(column in record for column in RESPONSE_COLUMNS))
        self.assertTrue(all(column in RESPONSE_COLUMNS for column in PRIMARY_RESPONSE_COLUMNS))
        self.assertAlmostEqual(record["centered_shape_mse_f24"], 0.0)
        self.assertAlmostEqual(record["centroid_displacement_f24"], 0.24)
        self.assertAlmostEqual(record["velocity_rms_trajectory"], 0.0)
        self.assertAlmostEqual(record["f_strain_norm_f24"], 0.0)
        self.assertAlmostEqual(record["volumetric_strain_f24"], 0.0)

    def test_train_record_requires_f_field(self):
        path_without_f = self.root / "missing_f.h5"
        self.write_h5(path_without_f)
        with h5py.File(path_without_f, "a") as handle:
            del handle["F"]

        with self.assertRaisesRegex(RecordValidationError, "missing.*F"):
            read_h5_record(path_without_f, split="train", settings=AuditSettings())

    def test_train_record_requires_at_least_25_frames(self):
        short_path = self.root / "short.h5"
        self.write_h5(short_path, frames=24)

        with self.assertRaisesRegex(RecordValidationError, "at least 25 frames"):
            read_h5_record(short_path, split="train", settings=AuditSettings())

    def test_train_record_requires_finite_positive_e(self):
        nonfinite_e_path = self.root / "nonfinite_e.h5"
        self.write_h5(nonfinite_e_path)
        with h5py.File(nonfinite_e_path, "a") as handle:
            del handle["E"]
            handle["E"] = np.asarray(np.inf)

        with self.assertRaisesRegex(RecordValidationError, "finite.*E"):
            read_h5_record(nonfinite_e_path, split="train", settings=AuditSettings())

    def test_record_requires_integer_material_code(self):
        invalid_material_path = self.root / "invalid_material.h5"
        self.write_h5(invalid_material_path)
        with h5py.File(invalid_material_path, "a") as handle:
            del handle["mat_type"]
            handle["mat_type"] = np.asarray(1.5)

        with self.assertRaisesRegex(RecordValidationError, "mat_type"):
            read_h5_record(
                invalid_material_path,
                split="train",
                settings=AuditSettings(),
            )

    def test_train_record_requires_aligned_particle_dimensions(self):
        misaligned_path = self.root / "misaligned.h5"
        self.write_h5(misaligned_path)
        with h5py.File(misaligned_path, "a") as handle:
            del handle["v"]
            handle["v"] = np.zeros((25, 7, 3), dtype=np.float32)

        with self.assertRaisesRegex(RecordValidationError, "particle dimension"):
            read_h5_record(misaligned_path, split="train", settings=AuditSettings())

    def test_coplanar_record_retains_valid_fields_when_hull_is_degenerate(self):
        coplanar_path = self.root / "coplanar.h5"
        self.write_h5(coplanar_path)
        with h5py.File(coplanar_path, "a") as handle:
            x = handle["x"][:]
            x[..., 2] = 0
            del handle["x"]
            handle["x"] = x

        record = read_h5_record(coplanar_path, split="train", settings=AuditSettings())

        self.assertTrue(np.isnan(record["initial_hull_volume"]))
        self.assertTrue(record["valid"])


if __name__ == "__main__":
    unittest.main()
