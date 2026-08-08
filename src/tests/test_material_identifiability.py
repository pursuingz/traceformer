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
    build_coverage_rows,
    build_support_rows,
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

    @staticmethod
    def _response_columns():
        frame_columns = tuple(
            f"{prefix}_f{frame}"
            for prefix in (
                "point_displacement_mse",
                "centroid_displacement",
                "centered_shape_mse",
                "extent_change_x",
                "extent_change_y",
                "extent_change_z",
                "future_contact_fraction",
            )
            for frame in (5, 10, 15, 20, 24)
        )
        return (
            "velocity_rms_trajectory",
            "velocity_rms_f24",
            "position_velocity_rms_trajectory",
            "position_acceleration_rms_trajectory",
            "f_strain_norm_f24",
            "f_strain_norm_trajectory",
            "volumetric_strain_f24",
            "volumetric_strain_trajectory",
            "c_norm_f24",
            "c_norm_trajectory",
            "contact_onset_frame",
            "future_contact_fraction",
            *frame_columns,
        )

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

    def test_response_columns_freeze_complete_secondary_schema(self):
        self.assertEqual(RESPONSE_COLUMNS, self._response_columns())
        self.assertNotIn("contact_onset_frame", NUISANCE_COLUMNS)
        self.assertNotIn("future_contact_fraction", NUISANCE_COLUMNS)

    def test_train_record_computes_complete_secondary_response_values(self):
        path = self.root / "secondary_responses.h5"
        self.write_h5(path)
        with h5py.File(path, "a") as handle:
            cube = handle["x"][0]
            frames = np.arange(25, dtype=np.float64)
            scale = 1.0 + 0.01 * frames[:, None, None]
            translation = frames[:, None, None] * np.asarray(
                [0.1, -0.02, 0.03], dtype=np.float64
            )
            x = cube[None, :, :] * scale + translation
            v = np.broadcast_to(frames[:, None, None], x.shape).copy()
            identity = np.eye(3, dtype=np.float64)
            f = np.broadcast_to(
                identity + frames[:, None, None, None] * 0.01 * identity,
                (25, len(cube), 3, 3),
            ).copy()
            c = np.broadcast_to(
                frames[:, None, None, None] * identity,
                (25, len(cube), 3, 3),
            ).copy()
            for field, values in (("x", x), ("v", v), ("F", f), ("C", c)):
                del handle[field]
                handle[field] = values

        record = read_h5_record(path, split="train", settings=AuditSettings())
        f_strain = np.linalg.norm(f - np.eye(3), axis=(-2, -1))
        j_error = np.abs(np.linalg.det(f) - 1.0)
        c_norm = np.linalg.norm(c, axis=(-2, -1))

        self.assertTrue(all(column in record for column in self._response_columns()))
        self.assertAlmostEqual(
            record["point_displacement_mse_f24"],
            np.mean((x[24] - x[0]) ** 2),
        )
        self.assertAlmostEqual(
            record["extent_change_x_f24"],
            np.ptp(x[24, :, 0]) - np.ptp(x[0, :, 0]),
        )
        self.assertAlmostEqual(
            record["extent_change_y_f24"],
            np.ptp(x[24, :, 1]) - np.ptp(x[0, :, 1]),
        )
        self.assertAlmostEqual(
            record["extent_change_z_f24"],
            np.ptp(x[24, :, 2]) - np.ptp(x[0, :, 2]),
        )
        self.assertAlmostEqual(record["velocity_rms_trajectory"], np.sqrt(np.mean(v ** 2)))
        self.assertAlmostEqual(record["velocity_rms_f24"], np.sqrt(np.mean(v[24] ** 2)))
        self.assertAlmostEqual(
            record["position_velocity_rms_trajectory"],
            np.sqrt(np.mean(np.diff(x, axis=0) ** 2)),
        )
        self.assertAlmostEqual(
            record["position_acceleration_rms_trajectory"],
            np.sqrt(np.mean(np.diff(x, n=2, axis=0) ** 2)),
        )
        self.assertAlmostEqual(record["f_strain_norm_f24"], np.mean(f_strain[24]))
        self.assertAlmostEqual(record["f_strain_norm_trajectory"], np.mean(f_strain))
        self.assertAlmostEqual(record["volumetric_strain_f24"], np.mean(j_error[24]))
        self.assertAlmostEqual(record["volumetric_strain_trajectory"], np.mean(j_error))
        self.assertAlmostEqual(record["c_norm_f24"], np.mean(c_norm[24]))
        self.assertAlmostEqual(record["c_norm_trajectory"], np.mean(c_norm))

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

    def test_record_rejects_nonvector_particle_volumes(self):
        path = self.root / "column_volume.h5"
        self.write_h5(path)
        with h5py.File(path, "a") as handle:
            del handle["vol"]
            handle["vol"] = np.ones((8, 1), dtype=np.float32) / 8

        with self.assertRaisesRegex(RecordValidationError, "vol.*shape"):
            read_h5_record(path, split="test", settings=AuditSettings())

    def test_record_rejects_invalid_drag_force_shape(self):
        path = self.root / "invalid_drag_force.h5"
        self.write_h5(path)
        with h5py.File(path, "a") as handle:
            del handle["drag_force"]
            del handle["drag_mask"]
            handle["drag_force"] = np.zeros((1, 2), dtype=np.float32)
            handle["drag_mask"] = np.zeros((1, 8), dtype=np.float32)

        with self.assertRaisesRegex(RecordValidationError, "drag_force.*shape"):
            read_h5_record(path, split="test", settings=AuditSettings())

    def test_record_rejects_misaligned_drag_mask(self):
        path = self.root / "misaligned_drag_mask.h5"
        self.write_h5(path)
        with h5py.File(path, "a") as handle:
            del handle["drag_force"]
            del handle["drag_mask"]
            handle["drag_force"] = np.zeros((1, 3), dtype=np.float32)
            handle["drag_mask"] = np.zeros((2, 7), dtype=np.float32)

        with self.assertRaisesRegex(RecordValidationError, "drag.*force count"):
            read_h5_record(path, split="test", settings=AuditSettings())

    def test_record_rejects_drag_mask_with_wrong_particle_dimension(self):
        path = self.root / "wrong_drag_particle_dimension.h5"
        self.write_h5(path)
        with h5py.File(path, "a") as handle:
            del handle["drag_force"]
            del handle["drag_mask"]
            handle["drag_force"] = np.zeros((1, 3), dtype=np.float32)
            handle["drag_mask"] = np.zeros((1, 7), dtype=np.float32)

        with self.assertRaisesRegex(RecordValidationError, "drag_mask.*particle dimension"):
            read_h5_record(path, split="test", settings=AuditSettings())

    def test_record_rejects_zero_frame_x_before_reading_initial_frame(self):
        path = self.root / "zero_frame.h5"
        self.write_h5(path)
        with h5py.File(path, "a") as handle:
            del handle["x"]
            handle["x"] = np.zeros((0, 8, 3), dtype=np.float32)

        with self.assertRaisesRegex(RecordValidationError, "x.*at least one frame"):
            read_h5_record(path, split="test", settings=AuditSettings())

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


class CoverageTests(unittest.TestCase):
    @staticmethod
    def _record(material, log10_e, nu, *, floor_gap=0.0):
        return {
            "model": f"{material}-{log10_e}-{nu}",
            "split": "train",
            "material": material,
            "valid": True,
            "log10_e": log10_e,
            "nu": nu,
            "initial_centroid_x": 0.0,
            "initial_centroid_y": 0.0,
            "initial_centroid_z": 0.0,
            "initial_extent_x": 1.0,
            "initial_extent_y": 1.0,
            "initial_extent_z": 1.0,
            "initial_cov_eig_0": 0.1,
            "initial_cov_eig_1": 0.2,
            "initial_cov_eig_2": 0.3,
            "radius_of_gyration": 0.4,
            "initial_hull_volume": 0.5,
            "total_particle_volume": 1.0,
            "floor_gap": floor_gap,
            "gravity": 1.0,
            "drag_magnitude": 0.0,
            "drag_count": 0,
            "drag_mask_ratio": 0.0,
            "initial_contact_fraction": 0.0,
            "centered_shape_mse_f24": 999.0,
        }

    def setUp(self):
        materials = ("elastic", "plasticine", "sand")
        self.train_records = []
        self.test_records = []
        for material_index, material in enumerate(materials):
            train_e = (4.0, 5.0, 6.0)
            test_e = (3.0, 5.0, 7.0) if material == "elastic" else train_e
            self.train_records.extend(
                self._record(
                    material,
                    log10_e,
                    0.10 + 0.10 * index,
                    floor_gap=float(material_index + index),
                )
                for index, log10_e in enumerate(train_e)
            )
            self.test_records.extend(
                self._record(
                    material,
                    log10_e,
                    0.10 + 0.10 * index,
                    floor_gap=float(material_index + index) + 10.0,
                )
                for index, log10_e in enumerate(test_e)
            )

    def test_coverage_rows_report_material_local_parameter_distributions(self):
        coverage = build_coverage_rows(self.train_records, self.test_records, bins=5)
        elastic_train_e = (4.0, 5.0, 6.0)
        elastic_e = find_row(
            coverage,
            split="train",
            material="elastic",
            parameter="log10_e",
        )

        self.assertEqual(elastic_e["unique_n"], len(set(elastic_train_e)))
        self.assertAlmostEqual(elastic_e["p50"], float(np.median(elastic_train_e)))
        self.assertGreaterEqual(elastic_e["joint_grid_occupancy"], 0.0)
        self.assertLessEqual(elastic_e["joint_grid_occupancy"], 1.0)
        self.assertAlmostEqual(elastic_e["pearson_e_nu"], 1.0)
        self.assertAlmostEqual(elastic_e["spearman_e_nu"], 1.0)

    def test_support_rows_separate_parameter_support_from_static_nuisance_shift(self):
        support = build_support_rows(self.train_records, self.test_records, bins=5)
        elastic_support = find_row(
            support,
            material="elastic",
            parameter="log10_e",
        )

        self.assertAlmostEqual(elastic_support["outside_train_fraction"], 2.0 / 3.0)
        self.assertIn("ks_statistic", elastic_support)
        self.assertIn("wasserstein_distance", elastic_support)
        self.assertIn("mahalanobis_outside_fraction", elastic_support)
        self.assertAlmostEqual(elastic_support["smd_floor_gap"], 10.0)
        self.assertEqual(elastic_support["support_status"], "out_of_support")
        self.assertTrue(all(name not in elastic_support for name in RESPONSE_COLUMNS))

    def test_test_nonfinite_nuisance_does_not_change_train_mahalanobis_reference(self):
        clean_support = build_support_rows(
            self.train_records,
            self.test_records,
            bins=5,
        )
        nonfinite_test_records = [dict(record) for record in self.test_records]
        nonfinite_test_records[0]["floor_gap"] = float("nan")

        nonfinite_support = build_support_rows(
            self.train_records,
            nonfinite_test_records,
            bins=5,
        )

        clean_elastic = find_row(
            clean_support,
            material="elastic",
            parameter="log10_e",
        )
        nonfinite_elastic = find_row(
            nonfinite_support,
            material="elastic",
            parameter="log10_e",
        )
        self.assertEqual(
            nonfinite_elastic["mahalanobis_feature_columns"],
            clean_elastic["mahalanobis_feature_columns"],
        )
        self.assertAlmostEqual(
            nonfinite_elastic["mahalanobis_train_p95"],
            clean_elastic["mahalanobis_train_p95"],
        )
        self.assertAlmostEqual(
            nonfinite_elastic["mahalanobis_nonfinite_test_fraction"],
            1.0 / 3.0,
        )
        self.assertEqual(nonfinite_elastic["support_status"], "out_of_support")

    def test_joint_grid_places_generation_maxima_in_final_bin(self):
        train_records = [
            self._record("elastic", 4.0, 0.05, floor_gap=0.0),
            self._record("elastic", 7.0, 0.45, floor_gap=1.0),
        ]

        coverage = build_coverage_rows(train_records, [], bins=5)

        elastic_e = find_row(
            coverage,
            split="train",
            material="elastic",
            parameter="log10_e",
        )
        self.assertAlmostEqual(elastic_e["joint_grid_occupancy"], 2.0 / 25.0)

    def test_support_metrics_allow_unequal_train_and_test_sample_counts(self):
        train_records = [
            self._record("elastic", 4.0, 0.10, floor_gap=0.0),
            self._record("elastic", 5.0, 0.20, floor_gap=1.0),
            self._record("elastic", 6.0, 0.30, floor_gap=2.0),
        ]
        test_records = [
            self._record("elastic", 4.0, 0.10, floor_gap=1.0),
            self._record("elastic", 7.0, 0.30, floor_gap=3.0),
        ]

        support = build_support_rows(train_records, test_records, bins=5)

        elastic_e = find_row(support, material="elastic", parameter="log10_e")
        self.assertEqual(elastic_e["n_train"], 3)
        self.assertEqual(elastic_e["n_test"], 2)
        self.assertAlmostEqual(elastic_e["ks_statistic"], 0.5)
        self.assertAlmostEqual(elastic_e["wasserstein_distance"], 5.0 / 6.0)
        self.assertAlmostEqual(elastic_e["smd_floor_gap"], 1.0)


if __name__ == "__main__":
    unittest.main()
