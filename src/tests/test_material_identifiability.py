import csv
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from scipy.stats import spearmanr

import utils.material_identifiability as material_identifiability
from utils.material_identifiability import (
    AuditSettings,
    NUISANCE_COLUMNS,
    PRIMARY_RESPONSE_COLUMNS,
    RESPONSE_COLUMNS,
    STATIC_COLUMNS,
    OUTPUT_NAMES,
    RecordValidationError,
    build_coverage_rows,
    build_support_rows,
    classify_identifiability,
    read_h5_record,
    render_markdown_report,
    write_audit_outputs,
)
from diagnose_material_identifiability import (
    build_parser,
    run_material_identifiability_audit,
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

    def test_train_record_accepts_flattened_legacy_f_and_c_matrices(self):
        train_path = self.root / "legacy_flattened_matrices.h5"
        self.write_h5(train_path, frames=25, include_dynamics=True)
        with h5py.File(train_path, "a") as handle:
            flattened_f = np.asarray(handle["F"][:]).reshape(25, 8, 9)
            flattened_c = np.asarray(handle["C"][:]).reshape(25, 8, 9)
            del handle["F"]
            del handle["C"]
            handle["F"] = flattened_f
            handle["C"] = flattened_c

        record = read_h5_record(
            train_path,
            split="train",
            settings=AuditSettings(),
        )

        self.assertAlmostEqual(record["f_strain_norm_f24"], 0.0)
        self.assertAlmostEqual(record["volumetric_strain_f24"], 0.0)
        self.assertAlmostEqual(record["c_norm_f24"], 0.0)

    def test_flattened_matrix_dynamics_preserves_row_major_component_order(self):
        x = np.zeros((2, 3, 3), dtype=np.float64)
        flattened = np.arange(54, dtype=np.float64).reshape(2, 3, 9)

        matrices = material_identifiability._normalise_matrix_dynamics(
            flattened,
            x,
            field="F",
        )

        np.testing.assert_array_equal(
            matrices,
            flattened.reshape(2, 3, 3, 3),
        )
        np.testing.assert_array_equal(matrices[0, 0, 0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(matrices[0, 0, 1], [3.0, 4.0, 5.0])
        np.testing.assert_array_equal(matrices[0, 0, 2], [6.0, 7.0, 8.0])

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

    def test_record_normalizes_generator_empty_list_drag_arrays(self):
        path = self.root / "zero_force_generator_schema.h5"
        self.write_h5(path)
        with h5py.File(path, "a") as handle:
            del handle["drag_force"]
            del handle["drag_mask"]
            handle["drag_force"] = []
            handle["drag_mask"] = []

        try:
            record = read_h5_record(path, split="test", settings=AuditSettings())
        except RecordValidationError as error:
            self.fail(f"generator empty-list drag arrays were rejected: {error}")

        self.assertEqual(record["drag_count"], 0)
        self.assertEqual(record["drag_magnitude"], 0.0)
        self.assertEqual(record["drag_mask_ratio"], 0.0)

    def test_record_rejects_nonempty_one_dimensional_drag_arrays(self):
        path = self.root / "nonempty_one_dimensional_drag.h5"
        self.write_h5(path)
        with h5py.File(path, "a") as handle:
            del handle["drag_force"]
            del handle["drag_mask"]
            handle["drag_force"] = [1.0, 0.0, 0.0]
            handle["drag_mask"] = [0.0] * 8

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

    def test_support_rows_mark_material_without_test_records_unknown(self):
        train_records = [
            self._record("elastic", 4.0, 0.10, floor_gap=0.0),
            self._record("elastic", 5.0, 0.20, floor_gap=1.0),
            self._record("elastic", 6.0, 0.30, floor_gap=2.0),
        ]

        support = build_support_rows(train_records, [], bins=5)

        elastic_e = find_row(support, material="elastic", parameter="log10_e")
        self.assertEqual(elastic_e["n_test"], 0)
        self.assertEqual(elastic_e["support_status"], "unknown")


class StatisticsHelperTests(unittest.TestCase):
    def test_audit_settings_freeze_production_sampling_defaults(self):
        settings = AuditSettings()

        self.assertEqual(settings.folds, 5)
        self.assertEqual(settings.permutations, 500)
        self.assertEqual(settings.bootstrap_samples, 1000)

    def test_object_folds_are_reproducible_complete_disjoint_and_name_stable(self):
        model_names = [f"object-{index:02d}.h5" for index in range(13)]

        folds_a = material_identifiability.make_object_folds(
            model_names, folds=5, seed=0
        )
        folds_b = material_identifiability.make_object_folds(
            model_names, folds=5, seed=0
        )

        self.assertEqual(serialise_folds(folds_a), serialise_folds(folds_b))
        self.assertEqual(
            sorted(np.concatenate([test for _, test in folds_a]).tolist()),
            list(range(len(model_names))),
        )
        for train, test in folds_a:
            self.assertTrue(set(train).isdisjoint(set(test)))

        reversed_names = list(reversed(model_names))
        reversed_folds = material_identifiability.make_object_folds(
            reversed_names, folds=5, seed=0
        )
        held_out_names = [
            tuple(sorted(model_names[index] for index in test))
            for _, test in folds_a
        ]
        reversed_held_out_names = [
            tuple(sorted(reversed_names[index] for index in test))
            for _, test in reversed_folds
        ]
        self.assertEqual(held_out_names, reversed_held_out_names)

    def test_piecewise_basis_uses_only_train_statistics(self):
        train_values = np.asarray([0.0, 1.0, 2.0, 3.0])
        train_basis, eval_basis = material_identifiability._piecewise_basis(
            train_values,
            np.asarray([100.0]),
        )
        train_basis_again, _ = material_identifiability._piecewise_basis(
            train_values,
            np.asarray([10000.0]),
        )

        self.assertEqual(train_basis.shape[1], 4)
        self.assertEqual(eval_basis.shape[1], 4)
        np.testing.assert_allclose(train_basis, train_basis_again)
        self.assertAlmostEqual(
            eval_basis[0, 0],
            (100.0 - train_values.mean()) / train_values.std(ddof=0),
        )

    def test_nested_predictions_do_not_use_sibling_held_out_values(self):
        model_names = [f"object-{index:02d}.h5" for index in range(20)]
        folds = material_identifiability.make_object_folds(
            model_names, folds=5, seed=3
        )
        held_out = folds[0][1]
        target_index, changed_index = int(held_out[0]), int(held_out[1])
        latent = np.linspace(-2.0, 2.0, len(model_names))
        nuisance = latent[:, None].copy()
        nuisance[target_index, 0] = np.nan
        response = 1.5 * latent + 0.1 * latent ** 2

        prediction_a = material_identifiability._nested_cv_predictions(
            nuisance,
            {},
            response,
            folds,
            augmented_parameter=None,
        )
        changed_nuisance = nuisance.copy()
        changed_nuisance[changed_index, 0] = 1e9
        changed_response = response.copy()
        changed_response[changed_index] = -1e9
        prediction_b = material_identifiability._nested_cv_predictions(
            changed_nuisance,
            {},
            changed_response,
            folds,
            augmented_parameter=None,
        )

        self.assertAlmostEqual(
            prediction_a[target_index], prediction_b[target_index], places=12
        )

    def test_permutation_pvalue_uses_inclusive_plus_one_formula(self):
        observed = 0.5
        null_values = np.asarray([0.5, 0.49, 0.8])

        p_value = material_identifiability._permutation_pvalue(
            observed, null_values
        )

        self.assertAlmostEqual(p_value, 0.75)
        self.assertAlmostEqual(
            material_identifiability._permutation_pvalue(
                2.0, np.asarray([0.0, 1.0, 1.5])
            ),
            0.25,
        )

    def test_bootstrap_resamples_paired_object_level_oof_tuples(self):
        base_prediction = np.asarray([0.0, 0.2, 0.5, 1.0, 1.5, 2.0])
        augmented_prediction = np.asarray([0.0, 1.1, 1.8, 3.2, 5.1, 8.2])
        response = np.asarray([0.0, 1.0, 2.0, 3.0, 5.0, 8.0])
        samples = 40
        seed = 17

        actual = material_identifiability._bootstrap_delta_r2(
            base_prediction,
            augmented_prediction,
            response,
            samples=samples,
            seed=seed,
        )
        rng = np.random.default_rng(seed)
        deltas = []
        for indices in rng.integers(0, len(response), size=(samples, len(response))):
            sampled_response = response[indices]
            total = np.sum((sampled_response - sampled_response.mean()) ** 2)
            if total < 1e-12:
                continue
            base_r2 = 1.0 - np.sum(
                (sampled_response - base_prediction[indices]) ** 2
            ) / total
            augmented_r2 = 1.0 - np.sum(
                (sampled_response - augmented_prediction[indices]) ** 2
            ) / total
            deltas.append(augmented_r2 - base_r2)
        expected = tuple(np.percentile(deltas, [2.5, 97.5]))

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)

    def test_conditional_residual_permutation_reconstructs_parameter(self):
        self.assertTrue(
            hasattr(material_identifiability, "_permute_conditional_residuals"),
            "conditional residual permutation helper is missing",
        )
        parameter = np.asarray([10.0, 20.0, 30.0, 40.0])
        conditional_prediction = np.asarray([9.0, 22.0, 27.0, 44.0])

        reconstructed = material_identifiability._permute_conditional_residuals(
            parameter,
            conditional_prediction,
            rng=np.random.default_rng(7),
        )

        np.testing.assert_allclose(reconstructed, [10.0, 25.0, 25.0, 40.0])

    def test_benjamini_hochberg_preserves_order_and_boundaries(self):
        pvalues = np.asarray([0.01, 0.04, 0.03, 0.002])

        qvalues = material_identifiability.benjamini_hochberg(pvalues)

        np.testing.assert_allclose(qvalues, [0.02, 0.04, 0.04, 0.008])
        self.assertTrue(np.all((qvalues >= 0.0) & (qvalues <= 1.0)))
        self.assertEqual(qvalues.shape, (4,))
        np.testing.assert_allclose(
            material_identifiability.benjamini_hochberg(
                np.asarray([0.0, 1.0])
            ),
            [0.0, 1.0],
        )

    def test_confounding_summary_reports_only_train_fold_varying_features(self):
        records = IdentifiabilityStatisticsTests.make_records(
            n=60,
            seed=9,
            response_kind="strong",
        )

        rows = material_identifiability.analyze_confounding(
            records,
            AuditSettings(seed=4, folds=5, permutations=4, bootstrap_samples=4),
        )
        summary = find_row(
            rows,
            row_type="summary",
            material="elastic",
            parameter="log10_e",
        )

        self.assertIn("initial_centroid_x", summary["fitted_features"])
        self.assertNotIn("gravity", summary["fitted_features"])


class IdentifiabilityStatisticsTests(unittest.TestCase):
    TEST_SETTINGS = AuditSettings(
        seed=0,
        folds=5,
        permutations=20,
        bootstrap_samples=40,
    )

    @staticmethod
    def make_records(*, n, seed, response_kind, material="elastic"):
        rng = np.random.default_rng(seed)
        nuisance = rng.normal(size=(n, 4))
        log10_e = rng.uniform(4.0, 7.0, size=n)
        nu = rng.uniform(0.05, 0.45, size=n)
        if response_kind == "strong":
            response = (
                2.0 * log10_e
                + 0.2 * nuisance[:, 0]
                + rng.normal(0.0, 0.1, n)
            )
        elif response_kind == "null":
            response = nuisance[:, 0] + rng.normal(0.0, 1.0, n)
        elif response_kind == "constant":
            response = np.ones(n)
        else:
            raise ValueError(response_kind)

        varying_columns = (
            "initial_centroid_x",
            "initial_centroid_y",
            "initial_centroid_z",
            "initial_extent_x",
        )
        records = []
        for index in range(n):
            record = {
                "model": f"{material}-{index:04d}.h5",
                "split": "train",
                "material": material,
                "valid": True,
                "log10_e": float(log10_e[index]),
                "nu": float(nu[index]),
                "centered_shape_mse_f24": float(response[index]),
            }
            record.update(
                {
                    column: float(nuisance[index, column_index])
                    for column_index, column in enumerate(varying_columns)
                }
            )
            for column in NUISANCE_COLUMNS:
                record.setdefault(column, 1.0 if column == "gravity" else 0.0)
            records.append(record)
        return records

    @staticmethod
    def make_correlated_parameter_records(*, n, seed, material="elastic"):
        rng = np.random.default_rng(seed)
        latent = rng.uniform(-1.0, 1.0, size=n)
        log10_e = 5.5 + 1.20 * latent + rng.normal(0.0, 0.002, size=n)
        nu = 0.25 + 0.16 * latent + rng.normal(0.0, 0.0002, size=n)
        nuisance = rng.normal(size=(n, 4))
        varying_columns = (
            "initial_centroid_x",
            "initial_centroid_y",
            "initial_centroid_z",
            "initial_extent_x",
        )
        records = []
        for index in range(n):
            record = {
                "model": f"{material}-correlated-{index:04d}.h5",
                "split": "train",
                "material": material,
                "valid": True,
                "log10_e": float(log10_e[index]),
                "nu": float(nu[index]),
                "centered_shape_mse_f24": float(nu[index]),
                "centroid_displacement_f24": float(log10_e[index]),
            }
            record.update(
                {
                    column: float(nuisance[index, column_index])
                    for column_index, column in enumerate(varying_columns)
                }
            )
            for column in NUISANCE_COLUMNS:
                record.setdefault(column, 1.0 if column == "gravity" else 0.0)
            records.append(record)
        return records

    def test_strong_parameter_signal_is_detected_with_reproducible_statistics(self):
        records = self.make_records(n=180, seed=123, response_kind="strong")

        rows = material_identifiability.analyze_responses(
            records, self.TEST_SETTINGS
        )
        repeated_rows = material_identifiability.analyze_responses(
            records, self.TEST_SETTINGS
        )
        strong = find_row(
            rows,
            material="elastic",
            parameter="log10_e",
            response="centered_shape_mse_f24",
        )

        self.assertEqual(rows, repeated_rows)
        self.assertGreater(strong["delta_r2"], 0.05)
        self.assertLessEqual(strong["permutation_p"], 0.05)
        self.assertGreater(strong["bootstrap_ci_low"], 0.0)
        self.assertEqual(strong["response_tier"], "primary")
        self.assertEqual(strong["status"], "ok")
        self.assertIn("r2_mboth", strong)
        self.assertEqual(strong.get("reference_model"), "Mnu")
        self.assertEqual(strong.get("augmented_model"), "Mboth")
        self.assertEqual(strong["r2_augmented"], strong["r2_mboth"])

    def test_correlated_parameters_do_not_create_pseudo_conditional_effects(self):
        records = self.make_correlated_parameter_records(
            n=120,
            seed=2026,
        )
        settings = AuditSettings(
            seed=0,
            folds=5,
            permutations=9,
            bootstrap_samples=16,
        )

        rows = material_identifiability.analyze_responses(records, settings)
        pseudo_effects = (
            find_row(
                rows,
                material="elastic",
                parameter="log10_e",
                response="centered_shape_mse_f24",
            ),
            find_row(
                rows,
                material="elastic",
                parameter="nu",
                response="centroid_displacement_f24",
            ),
        )

        for row in pseudo_effects:
            with self.subTest(parameter=row["parameter"]):
                self.assertLess(row["delta_r2"], 0.05)
                self.assertEqual(row["augmented_model"], "Mboth")

    def test_null_parameter_signal_remains_below_identifiability_threshold(self):
        records = self.make_records(n=180, seed=456, response_kind="null")

        rows = material_identifiability.analyze_responses(
            records, self.TEST_SETTINGS
        )
        null = find_row(
            rows,
            material="elastic",
            parameter="nu",
            response="centered_shape_mse_f24",
        )

        self.assertLess(null["delta_r2"], 0.05)

    def test_partial_spearman_uses_rank_correlation_of_cross_fitted_residuals(self):
        records = self.make_records(n=60, seed=654, response_kind="strong")
        settings = AuditSettings(
            seed=2,
            folds=5,
            permutations=4,
            bootstrap_samples=8,
        )

        rows = material_identifiability.analyze_responses(records, settings)
        row = find_row(
            rows,
            material="elastic",
            parameter="log10_e",
            response="centered_shape_mse_f24",
        )
        nuisance_columns = tuple(
            column
            for column in NUISANCE_COLUMNS
            if column not in ("log10_e", "nu")
        )
        nuisance = np.asarray(
            [[float(record[column]) for column in nuisance_columns] for record in records]
        )
        parameter = np.asarray([float(record["log10_e"]) for record in records])
        response = np.asarray(
            [float(record["centered_shape_mse_f24"]) for record in records]
        )
        folds = material_identifiability.make_object_folds(
            [str(record["model"]) for record in records],
            folds=settings.folds,
            seed=settings.seed,
        )
        parameter_prediction = material_identifiability._nested_cv_predictions(
            nuisance,
            {"log10_e": parameter, "nu": np.asarray([record["nu"] for record in records])},
            parameter,
            folds,
            augmented_parameter="nu",
        )
        response_prediction = material_identifiability._nested_cv_predictions(
            nuisance,
            {"log10_e": parameter, "nu": np.asarray([record["nu"] for record in records])},
            response,
            folds,
            augmented_parameter="nu",
        )
        expected = float(
            spearmanr(
                parameter - parameter_prediction,
                response - response_prediction,
            )[0]
        )

        self.assertAlmostEqual(row["partial_spearman"], expected, places=12)

    def test_nuisance_predictable_parameter_is_marked_confounding(self):
        records = self.make_records(n=180, seed=789, response_kind="null")
        rng = np.random.default_rng(987)
        for record in records:
            record["log10_e"] = float(
                5.5
                + 0.8 * record["initial_centroid_x"]
                + rng.normal(0.0, 0.05)
            )

        rows = material_identifiability.analyze_confounding(
            records, self.TEST_SETTINGS
        )
        summary = find_row(
            rows,
            row_type="summary",
            material="elastic",
            parameter="log10_e",
        )

        self.assertTrue(summary["confounded"])
        self.assertGreater(summary["cv_r2"], 0.05)
        self.assertLess(summary["permutation_p"], 0.05)

    def test_constant_response_is_not_emitted_as_evidence(self):
        records = self.make_records(n=60, seed=321, response_kind="constant")

        rows = material_identifiability.analyze_responses(
            records,
            AuditSettings(seed=0, folds=5, permutations=4, bootstrap_samples=8),
        )
        row = find_row(
            rows,
            material="elastic",
            parameter="log10_e",
            response="centered_shape_mse_f24",
        )

        self.assertEqual(row["status"], "constant_response")
        self.assertEqual(row["delta_r2"], 0.0)
        self.assertEqual(row["permutation_p"], 1.0)
        self.assertEqual(row["q_value"], 1.0)
        self.assertNotIn("future_contact_fraction", NUISANCE_COLUMNS)


class ClassificationTests(unittest.TestCase):
    @staticmethod
    def _response(**overrides):
        row = {
            "material": "elastic",
            "parameter": "log10_e",
            "response": "centered_shape_mse_f24",
            "response_tier": "primary",
            "delta_r2": 0.05,
            "permutation_p": 0.049,
            "q_value": 0.049,
            "bootstrap_ci_low": 0.001,
            "bootstrap_ci_high": 0.1,
            "partial_spearman": 0.2,
            "status": "ok",
        }
        row.update(overrides)
        return row

    @classmethod
    def _complete_responses(cls, **target_overrides):
        rows = [
            cls._response(
                response=response,
                delta_r2=0.0,
                permutation_p=1.0,
                q_value=1.0,
                bootstrap_ci_low=-0.1,
                bootstrap_ci_high=0.1,
                partial_spearman=0.0,
            )
            for response in PRIMARY_RESPONSE_COLUMNS
        ]
        rows[0].update(target_overrides)
        return rows

    @staticmethod
    def _confounding(**overrides):
        row = {
            "row_type": "summary",
            "material": "elastic",
            "parameter": "log10_e",
            "confounded": False,
            "cv_r2": 0.0,
            "permutation_p": 1.0,
            "status": "ok",
        }
        row.update(overrides)
        return row

    @staticmethod
    def _support(**overrides):
        row = {
            "material": "elastic",
            "parameter": "log10_e",
            "support_status": "in_support",
        }
        row.update(overrides)
        return row

    def test_classification_uses_primary_fdr_qualified_response(self):
        summary = classify_identifiability(
            self._complete_responses(
                delta_r2=0.05,
                permutation_p=0.049,
                q_value=0.049,
                bootstrap_ci_low=0.001,
                partial_spearman=0.2,
            ),
            [self._confounding()],
            [self._support()],
        )

        row = find_row(summary, material="elastic", parameter="log10_e")
        self.assertEqual(row["status"], "identifiable")
        self.assertEqual(row["support_status"], "in_support")
        self.assertIn("primary_delta_r2", row["reason_codes"])

    def test_classification_rejects_secondary_only_and_fdr_boundary_evidence(self):
        secondary_only = self._response(
            response="contact_onset_frame",
            response_tier="secondary",
        )
        summary = classify_identifiability(
            [
                *self._complete_responses(),
                secondary_only,
            ],
            [self._confounding()],
            [self._support()],
        )
        self.assertNotEqual(
            find_row(summary, material="elastic", parameter="log10_e")["status"],
            "identifiable",
        )

        summary = classify_identifiability(
            self._complete_responses(
                delta_r2=0.05,
                permutation_p=0.049,
                q_value=0.05,
                bootstrap_ci_low=0.001,
                partial_spearman=0.2,
            ),
            [self._confounding()],
            [self._support()],
        )
        self.assertEqual(
            find_row(summary, material="elastic", parameter="log10_e")["status"],
            "weak",
        )

    def test_classification_requires_strict_significance_and_positive_ci(self):
        for boundary in (
            {"permutation_p": 0.05},
            {"q_value": 0.05},
            {"bootstrap_ci_low": 0.0},
        ):
            with self.subTest(boundary=boundary):
                qualified = {
                    "delta_r2": 0.05,
                    "permutation_p": 0.049,
                    "q_value": 0.049,
                    "bootstrap_ci_low": 0.001,
                    "partial_spearman": 0.2,
                    **boundary,
                }
                summary = classify_identifiability(
                    self._complete_responses(**qualified),
                    [self._confounding()],
                    [self._support()],
                )
                self.assertEqual(
                    find_row(
                        summary,
                        material="elastic",
                        parameter="log10_e",
                    )["status"],
                    "weak",
                )

    def test_classification_treats_constant_primary_as_legal_no_evidence(self):
        responses = self._complete_responses(
            delta_r2=0.05,
            permutation_p=0.049,
            q_value=0.049,
            bootstrap_ci_low=0.001,
            partial_spearman=0.2,
        )
        responses[1].update(
            {
                "status": "constant_response",
                "delta_r2": 0.0,
                "permutation_p": 1.0,
                "q_value": 1.0,
                "bootstrap_ci_low": 0.0,
                "bootstrap_ci_high": 0.0,
                "partial_spearman": float("nan"),
            }
        )

        summary = classify_identifiability(
            responses,
            [self._confounding()],
            [self._support()],
        )

        row = find_row(summary, material="elastic", parameter="log10_e")
        self.assertEqual(row["status"], "identifiable")
        self.assertNotIn("invalid_primary_statistics", row["reason_codes"])

        constant_responses = [
            self._response(
                response=response,
                status="constant_response",
                delta_r2=0.0,
                permutation_p=1.0,
                q_value=1.0,
                bootstrap_ci_low=0.0,
                bootstrap_ci_high=0.0,
                partial_spearman=float("nan"),
            )
            for response in PRIMARY_RESPONSE_COLUMNS
        ]
        constant_summary = classify_identifiability(
            constant_responses,
            [self._confounding()],
            [self._support()],
        )
        constant_row = find_row(
            constant_summary,
            material="elastic",
            parameter="log10_e",
        )
        self.assertEqual(constant_row["status"], "not_detected")

    def test_classification_distinguishes_weak_not_detected_and_confounded(self):
        weak_rows = self._complete_responses(delta_r2=0.01)
        summary = classify_identifiability(
            weak_rows,
            [self._confounding()],
            [self._support()],
        )
        self.assertEqual(
            find_row(summary, material="elastic", parameter="log10_e")["status"],
            "weak",
        )

        null_rows = self._complete_responses(delta_r2=0.009)
        summary = classify_identifiability(
            null_rows,
            [self._confounding()],
            [self._support()],
        )
        self.assertEqual(
            find_row(summary, material="elastic", parameter="log10_e")["status"],
            "not_detected",
        )

        summary = classify_identifiability(
            self._complete_responses(
                delta_r2=0.05,
                permutation_p=0.049,
                q_value=0.049,
                bootstrap_ci_low=0.001,
                partial_spearman=0.2,
            ),
            [self._confounding(confounded=True)],
            [self._support(support_status="out_of_support")],
        )
        row = find_row(summary, material="elastic", parameter="log10_e")
        self.assertEqual(row["status"], "confounded")
        self.assertEqual(row["support_status"], "out_of_support")
        self.assertIn("nuisance_predictable", row["reason_codes"])

    def test_classification_keeps_support_independent(self):
        summary = classify_identifiability(
            self._complete_responses(
                delta_r2=0.05,
                permutation_p=0.049,
                q_value=0.049,
                bootstrap_ci_low=0.001,
                partial_spearman=0.2,
            ),
            [self._confounding()],
            [self._support(support_status="out_of_support")],
        )
        row = find_row(summary, material="elastic", parameter="log10_e")
        self.assertEqual(row["status"], "identifiable")
        self.assertEqual(row["support_status"], "out_of_support")

    def test_classification_rejects_invalid_file_counts_outside_metadata(self):
        with self.assertRaisesRegex(ValueError, "metadata.invalid_records"):
            classify_identifiability(
                self._complete_responses(invalid_record_count=1),
                [self._confounding(invalid_record_count=1)],
                [self._support(invalid_record_count=1)],
            )

    def test_classification_marks_incomplete_statistics_invalid(self):
        missing_primary = self._complete_responses()[:-1]
        insufficient_primary = self._complete_responses(status="insufficient_data")
        nonfinite_primary = self._complete_responses(partial_spearman=float("nan"))
        missing_primary_status = self._complete_responses()
        missing_primary_status[0].pop("status")
        duplicate_primary = [
            *self._complete_responses(),
            self._response(delta_r2=0.0),
        ]
        missing_confounding_status = self._confounding()
        missing_confounding_status.pop("status")
        missing_confounding_row_type = self._confounding()
        missing_confounding_row_type.pop("row_type")
        cases = (
            (
                "missing primary",
                missing_primary,
                [self._confounding()],
                "missing_primary_responses",
            ),
            (
                "insufficient primary",
                insufficient_primary,
                [self._confounding()],
                "invalid_primary_statistics",
            ),
            (
                "missing primary status",
                missing_primary_status,
                [self._confounding()],
                "invalid_primary_statistics",
            ),
            (
                "nonfinite primary",
                nonfinite_primary,
                [self._confounding()],
                "invalid_primary_statistics",
            ),
            (
                "duplicate primary",
                duplicate_primary,
                [self._confounding()],
                "duplicate_primary_responses",
            ),
            (
                "missing confounding",
                self._complete_responses(),
                [],
                "missing_confounding_summary",
            ),
            (
                "insufficient confounding",
                self._complete_responses(),
                [self._confounding(status="insufficient_data", cv_r2=float("nan"))],
                "invalid_confounding_statistics",
            ),
            (
                "missing confounding status",
                self._complete_responses(),
                [missing_confounding_status],
                "invalid_confounding_statistics",
            ),
            (
                "missing confounding row type",
                self._complete_responses(),
                [missing_confounding_row_type],
                "missing_confounding_summary",
            ),
        )

        for label, response_rows, confounding_rows, reason_code in cases:
            with self.subTest(label=label):
                summary = classify_identifiability(
                    response_rows,
                    confounding_rows,
                    [self._support()],
                )
                row = find_row(
                    summary,
                    material="elastic",
                    parameter="log10_e",
                )
                self.assertEqual(row["status"], "invalid")
                self.assertIn(reason_code, row["reason_codes"])

    def test_classification_emits_all_six_decisions_for_partial_inputs(self):
        summary = classify_identifiability(
            self._complete_responses(),
            [self._confounding()],
            [self._support()],
        )

        self.assertEqual(
            [(row["material"], row["parameter"]) for row in summary],
            [
                (material, parameter)
                for material in ("elastic", "plasticine", "sand")
                for parameter in ("log10_e", "nu")
            ],
        )
        absent = find_row(summary, material="sand", parameter="nu")
        self.assertEqual(absent["status"], "invalid")
        self.assertIn("missing_primary_responses", absent["reason_codes"])
        self.assertIn("missing_confounding_summary", absent["reason_codes"])


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.output_dir = self.root / "outputs"
        self.records = [
            {
                "model": "train.h5",
                "split": "train",
                "material": "elastic",
                "valid": True,
                "log10_e": 6.0,
                "nu": 0.3,
                "centered_shape_mse_f24": 0.2,
            },
            {
                "model": "test.h5",
                "split": "test",
                "material": "elastic",
                "valid": True,
                "log10_e": 6.1,
                "nu": 0.31,
            },
        ]
        self.summary_rows = [
            {
                "material": material,
                "parameter": parameter,
                "status": "not_detected",
                "support_status": "in_support",
                "reason_codes": ("no_detectable_response",),
                "invalid_record_count": 0,
            }
            for material in ("elastic", "plasticine", "sand")
            for parameter in ("log10_e", "nu")
        ]
        self.payload = {
            "records": self.records,
            "coverage_rows": [{
                "row_type": "wrong", "split": "train",
                "material": "elastic", "parameter": "log10_e",
                "n": 1, "unique_n": 1, "joint_grid_occupancy": 0.04,
            }],
            "support_rows": [{
                "row_type": "wrong", "material": "elastic", "parameter": "log10_e",
                "n_train": 1, "n_test": 1, "support_status": "in_support",
            }],
            "confounding_rows": [{
                "row_type": "summary", "material": "elastic", "parameter": "log10_e",
                "confounded": False, "status": "ok",
            }],
            "response_rows": [{
                "material": "elastic", "parameter": "log10_e",
                "response": "centered_shape_mse_f24", "response_tier": "primary",
                "delta_r2": 0.0, "permutation_p": 1.0, "q_value": 1.0,
            }],
            "summary_rows": self.summary_rows,
            "metadata": {
                "seed": 0,
                "note": "无效记录",
                "invalid_records": [
                    {"path": "bad.h5", "split": "train", "error": "missing F"}
                ],
            },
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_writer_emits_fixed_schema_refuses_overwrite_and_renders_honest_report(self):
        paths = write_audit_outputs(self.output_dir, overwrite=False, **self.payload)
        self.assertEqual(set(paths), set(OUTPUT_NAMES))
        self.assertTrue(all(path.exists() for path in paths.values()))
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            set(OUTPUT_NAMES.values()),
        )

        with self.assertRaises(FileExistsError):
            write_audit_outputs(self.output_dir, overwrite=False, **self.payload)

        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(len(metadata["invalid_records"]), 1)
        self.assertEqual(metadata["invalid_records"][0]["path"], "bad.h5")
        self.assertEqual(metadata["seed"], 0)
        self.assertEqual(metadata.get("audit_integrity_status"), "train_invalid")
        self.assertEqual(metadata.get("schema_version"), "1.0")
        generated_at = metadata.get("generated_at")
        self.assertIsInstance(generated_at, str)
        parsed_generated_at = datetime.fromisoformat(
            generated_at.replace("Z", "+00:00")
        )
        self.assertEqual(parsed_generated_at.tzinfo, timezone.utc)
        self.assertTrue(generated_at.endswith("Z"))
        self.assertIn(
            "无效记录",
            paths["metadata"].read_text(encoding="utf-8"),
        )

        with paths["summary"].open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        self.assertEqual({row["status"] for row in summary_rows}, {"invalid"})
        self.assertEqual(
            {row["invalid_record_count"] for row in summary_rows},
            {"1"},
        )
        self.assertTrue(
            all("train_invalid_records" in row["reason_codes"] for row in summary_rows)
        )
        self.assertEqual({row["support_status"] for row in summary_rows}, {"in_support"})

        with paths["records"].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, [*STATIC_COLUMNS, *RESPONSE_COLUMNS])
        test_row = next(row for row in rows if row["split"] == "test")
        self.assertEqual(test_row["centered_shape_mse_f24"], "")

        with paths["coverage"].open(newline="", encoding="utf-8") as handle:
            coverage_rows = list(csv.DictReader(handle))
        self.assertEqual(
            {row["row_type"] for row in coverage_rows},
            {"distribution", "support"},
        )

        report = paths["report"].read_text(encoding="utf-8")
        self.assertIn("np.random.randint(0, 1)", report)
        self.assertIn("elastic", report)
        self.assertIn("plasticine", report)
        self.assertIn("sand", report)
        self.assertIn("不能证明反事实物理正确", report)
        self.assertIn("不能构成配对反事实样本", report)
        self.assertIn("不使用 test 动力学", report)
        self.assertIn("observational conditional incremental association", report)
        self.assertIn("不是 counterfactual causality", report)
        self.assertIn("audit_integrity_status: train_invalid", report)
        self.assertIn("train invalid records: 1", report)
        self.assertIn("test invalid records: 0", report)

        with paths["response"].open(newline="", encoding="utf-8") as handle:
            response_reader = csv.DictReader(handle)
            list(response_reader)
        self.assertIn("reference_model", response_reader.fieldnames)
        self.assertIn("augmented_model", response_reader.fieldnames)

    def test_test_invalid_records_do_not_overwrite_train_status(self):
        payload = dict(self.payload)
        payload["metadata"] = {
            "seed": 0,
            "invalid_records": [
                {"path": "bad-test.h5", "split": "test", "error": "missing E"}
            ],
        }
        payload["summary_rows"] = [dict(row) for row in self.summary_rows]
        for row in payload["summary_rows"]:
            row["reason_codes"] = (
                *row["reason_codes"],
                "test_parameter_in_support",
            )
        payload["summary_rows"][0]["status"] = "identifiable"
        payload["summary_rows"][0]["reason_codes"] = (
            "primary_delta_r2",
            "test_parameter_in_support",
        )

        paths = write_audit_outputs(
            self.output_dir,
            overwrite=False,
            **payload,
        )

        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(metadata.get("audit_integrity_status"), "test_invalid")
        with paths["summary"].open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        elastic_e = find_row(
            summary_rows,
            material="elastic",
            parameter="log10_e",
        )
        self.assertEqual(elastic_e["status"], "identifiable")
        self.assertEqual({row["status"] for row in summary_rows[1:]}, {"not_detected"})
        self.assertEqual({row["support_status"] for row in summary_rows}, {"unknown"})
        self.assertTrue(
            all("test_invalid_records" in row["reason_codes"] for row in summary_rows)
        )
        self.assertTrue(
            all("test_support_unknown" in row["reason_codes"] for row in summary_rows)
        )
        self.assertTrue(
            all(
                "test_parameter_in_support" not in row["reason_codes"]
                and "test_parameter_extrapolation" not in row["reason_codes"]
                for row in summary_rows
            )
        )
        self.assertTrue(
            all("train_invalid_records" not in row["reason_codes"] for row in summary_rows)
        )
        report = paths["report"].read_text(encoding="utf-8")
        self.assertIn("audit_integrity_status: test_invalid", report)
        self.assertIn("train invalid records: 0", report)
        self.assertIn("test invalid records: 1", report)

    def test_test_invalid_support_override_is_consistent_across_outputs(self):
        payload = dict(self.payload)
        payload["metadata"] = {
            "seed": 0,
            "invalid_records": [
                {"path": "bad-test.h5", "split": "test", "error": "missing E"}
            ],
        }
        payload["summary_rows"] = [dict(row) for row in self.summary_rows]
        for row in payload["summary_rows"]:
            row["reason_codes"] = (
                *row["reason_codes"],
                "test_parameter_in_support",
            )
        payload["summary_rows"][0]["status"] = "identifiable"

        paths = write_audit_outputs(
            self.output_dir,
            overwrite=False,
            **payload,
        )

        with paths["summary"].open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        self.assertEqual(summary_rows[0]["status"], "identifiable")
        self.assertEqual({row["support_status"] for row in summary_rows}, {"unknown"})
        self.assertTrue(
            all(
                "test_parameter_in_support" not in row["reason_codes"]
                and "test_parameter_extrapolation" not in row["reason_codes"]
                for row in summary_rows
            )
        )

        with paths["coverage"].open(newline="", encoding="utf-8") as handle:
            coverage_rows = list(csv.DictReader(handle))
        support_rows = [row for row in coverage_rows if row["row_type"] == "support"]
        self.assertEqual({row["support_status"] for row in support_rows}, {"unknown"})

        report = paths["report"].read_text(encoding="utf-8")
        support_section = report.split("## Train/Test Support", 1)[1].split(
            "## Nuisance", 1
        )[0]
        self.assertIn("| elastic | log10_e |", support_section)
        self.assertIn("| unknown |", support_section)
        self.assertNotIn("| in_support |", support_section)
        self.assertNotIn("test_parameter_in_support", report)

    def test_writer_leaves_final_paths_untouched_when_rendering_fails(self):
        broken_payload = dict(self.payload)
        broken_payload["metadata"] = {"seed": 0, "unsupported": object()}

        with self.assertRaises(TypeError):
            write_audit_outputs(self.output_dir, overwrite=False, **broken_payload)

        self.assertFalse(
            self.output_dir.exists() and any(self.output_dir.glob("*"))
        )

    def test_writer_rejects_summary_invalid_count_inconsistent_with_metadata(self):
        inconsistent_payload = dict(self.payload)
        inconsistent_payload["summary_rows"] = [
            dict(self.summary_rows[0], invalid_record_count=2),
            *self.summary_rows[1:],
        ]

        with self.assertRaisesRegex(ValueError, "metadata.invalid_records"):
            write_audit_outputs(
                self.output_dir,
                overwrite=False,
                **inconsistent_payload,
            )

        self.assertFalse(
            self.output_dir.exists() and any(self.output_dir.glob("*"))
        )

    def test_writer_rejects_nonempty_test_response_columns(self):
        leaking_payload = dict(self.payload)
        leaking_payload["records"] = [dict(record) for record in self.records]
        leaking_payload["records"][1]["centered_shape_mse_f24"] = 0.0

        with self.assertRaisesRegex(
            ValueError,
            "test.*centered_shape_mse_f24",
        ):
            write_audit_outputs(
                self.output_dir,
                overwrite=False,
                **leaking_payload,
            )

        self.assertFalse(
            self.output_dir.exists() and any(self.output_dir.glob("*"))
        )

    def test_renderer_and_writer_require_six_unique_summary_decisions(self):
        invalid_summaries = (
            ("missing", self.summary_rows[:-1]),
            ("duplicate", [*self.summary_rows, dict(self.summary_rows[0])]),
        )

        for label, summary_rows in invalid_summaries:
            with self.subTest(label=label, entrypoint="renderer"):
                with self.assertRaisesRegex(ValueError, label):
                    render_markdown_report(
                        summary_rows,
                        self.payload["coverage_rows"],
                        self.payload["support_rows"],
                        self.payload["confounding_rows"],
                        self.payload["response_rows"],
                        self.payload["metadata"],
                    )

            with self.subTest(label=label, entrypoint="writer"):
                invalid_payload = dict(self.payload)
                invalid_payload["summary_rows"] = summary_rows
                invalid_output_dir = self.root / f"outputs-{label}"
                with self.assertRaisesRegex(ValueError, label):
                    write_audit_outputs(
                        invalid_output_dir,
                        overwrite=False,
                        **invalid_payload,
                    )
                self.assertFalse(
                    invalid_output_dir.exists() and any(invalid_output_dir.glob("*"))
                )

    def test_writer_does_not_partially_replace_outputs_when_activation_is_blocked(self):
        self.output_dir.mkdir()
        records_path = self.output_dir / OUTPUT_NAMES["records"]
        coverage_path = self.output_dir / OUTPUT_NAMES["coverage"]
        blocked_path = self.output_dir / OUTPUT_NAMES["confounding"]
        records_path.write_text("old records\n", encoding="utf-8")
        coverage_path.write_text("old coverage\n", encoding="utf-8")
        blocked_path.mkdir()

        with self.assertRaises(OSError):
            write_audit_outputs(self.output_dir, overwrite=True, **self.payload)

        self.assertEqual(records_path.read_text(encoding="utf-8"), "old records\n")
        self.assertEqual(coverage_path.read_text(encoding="utf-8"), "old coverage\n")
        self.assertTrue(blocked_path.is_dir())
        self.assertFalse(
            any(
                (self.output_dir / OUTPUT_NAMES[key]).exists()
                for key in ("response", "summary", "metadata", "report")
            )
        )

    def test_writer_refuses_any_existing_target_before_rendering(self):
        self.output_dir.mkdir()
        blocked_path = self.output_dir / OUTPUT_NAMES["summary"]
        blocked_path.mkdir()
        broken_payload = dict(self.payload)
        broken_payload["metadata"] = {"unsupported": object()}

        with self.assertRaises(FileExistsError):
            write_audit_outputs(
                self.output_dir,
                overwrite=False,
                **broken_payload,
            )

        self.assertTrue(blocked_path.is_dir())
        self.assertEqual(list(self.output_dir.iterdir()), [blocked_path])

    def test_writer_rechecks_output_targets_immediately_before_activation(self):
        raced_path = self.output_dir / OUTPUT_NAMES["summary"]
        original_render = material_identifiability.render_markdown_report

        def render_after_raced_target(*args, **kwargs):
            self.output_dir.mkdir()
            raced_path.write_text("concurrent output\n", encoding="utf-8")
            return original_render(*args, **kwargs)

        with mock.patch.object(
            material_identifiability,
            "render_markdown_report",
            side_effect=render_after_raced_target,
        ):
            with self.assertRaises(FileExistsError):
                write_audit_outputs(
                    self.output_dir,
                    overwrite=False,
                    **self.payload,
                )

        self.assertEqual(
            raced_path.read_text(encoding="utf-8"),
            "concurrent output\n",
        )
        self.assertEqual(list(self.output_dir.iterdir()), [raced_path])

    def test_writer_restores_all_old_outputs_after_mid_activation_failure(self):
        paths = write_audit_outputs(self.output_dir, overwrite=False, **self.payload)
        old_contents = {
            key: path.read_bytes()
            for key, path in paths.items()
        }
        replacement_payload = dict(self.payload)
        replacement_payload["metadata"] = {"seed": 1, "invalid_records": []}
        original_replace = Path.replace
        failed = False

        def fail_response_activation(source, target):
            nonlocal failed
            target = Path(target)
            if (
                not failed
                and source.name == OUTPUT_NAMES["response"]
                and target == paths["response"]
            ):
                failed = True
                raise OSError("injected activation failure")
            return original_replace(source, target)

        with mock.patch.object(
            Path,
            "replace",
            autospec=True,
            side_effect=fail_response_activation,
        ):
            with self.assertRaisesRegex(OSError, "injected activation failure"):
                write_audit_outputs(
                    self.output_dir,
                    overwrite=True,
                    **replacement_payload,
                )

        self.assertTrue(failed)
        self.assertEqual(
            {key: path.read_bytes() for key, path in paths.items()},
            old_contents,
        )
        self.assertFalse(
            any(".backup." in path.name for path in self.root.iterdir())
        )

    def test_writer_does_not_fail_after_successful_activation_if_cleanup_fails(self):
        paths = write_audit_outputs(self.output_dir, overwrite=False, **self.payload)
        replacement_payload = dict(self.payload)
        replacement_payload["metadata"] = {"seed": 1, "invalid_records": []}
        original_rmtree = shutil.rmtree
        cleanup_attempted = False

        def fail_backup_cleanup(path, *args, **kwargs):
            nonlocal cleanup_attempted
            if ".backup." in Path(path).name:
                cleanup_attempted = True
                raise OSError("injected backup cleanup failure")
            return original_rmtree(path, *args, **kwargs)

        with mock.patch.object(shutil, "rmtree", side_effect=fail_backup_cleanup):
            replaced_paths = write_audit_outputs(
                self.output_dir,
                overwrite=True,
                **replacement_payload,
            )

        self.assertTrue(cleanup_attempted)
        self.assertEqual(replaced_paths, paths)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["seed"], 1)
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            set(OUTPUT_NAMES.values()),
        )

    def test_writer_replaces_complete_existing_output_set(self):
        paths = write_audit_outputs(self.output_dir, overwrite=False, **self.payload)
        replacement_payload = dict(self.payload)
        replacement_payload["metadata"] = {"seed": 1, "invalid_records": []}

        replaced_paths = write_audit_outputs(
            self.output_dir,
            overwrite=True,
            **replacement_payload,
        )

        self.assertEqual(replaced_paths, paths)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["seed"], 1)
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            set(OUTPUT_NAMES.values()),
        )

    def test_markdown_renderer_is_available_without_writing_files(self):
        report = render_markdown_report(
            self.summary_rows,
            self.payload["coverage_rows"],
            self.payload["support_rows"],
            self.payload["confounding_rows"],
            self.payload["response_rows"],
            self.payload["metadata"],
        )

        self.assertIn("B0.2", report)
        self.assertIn("不能证明反事实物理正确", report)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train_dir = self.root / "train"
        self.test_dir = self.root / "test"
        self.output_dir = self.root / "outputs"
        self.train_dir.mkdir()
        self.test_dir.mkdir()

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_h5(
        path: Path,
        *,
        material_code: int,
        index: int,
        include_dynamics: bool,
    ) -> None:
        cube = np.asarray(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
            ],
            dtype=np.float32,
        )
        frames = 25 if include_dynamics else 1
        displacement = 0.002 * (index + 1)
        x = np.stack(
            [cube + np.asarray([0, -displacement * frame, 0]) for frame in range(frames)]
        )
        with h5py.File(path, "w") as handle:
            handle["x"] = x
            handle["vol"] = np.ones(len(cube), dtype=np.float32) / len(cube)
            handle["E"] = np.asarray(10.0 ** (4.0 + 0.1 * index))
            handle["nu"] = np.asarray(0.05 + 0.05 * index)
            handle["mat_type"] = np.asarray(material_code)
            handle["gravity"] = np.asarray(1)
            handle["floor_height"] = np.asarray(-0.1)
            handle["drag_force"] = np.zeros((0, 3), dtype=np.float32)
            handle["drag_mask"] = np.zeros((0, len(cube)), dtype=np.float32)
            if include_dynamics:
                handle["v"] = np.zeros_like(x)
                eye = np.eye(3, dtype=np.float32)
                handle["F"] = np.broadcast_to(eye, (frames, len(cube), 3, 3))
                handle["C"] = np.zeros((frames, len(cube), 3, 3), dtype=np.float32)

    def _create_smoke_fixture(self) -> None:
        for material_code in range(3):
            for index in range(6):
                self._write_h5(
                    self.train_dir / f"train-{material_code}-{5 - index}.h5",
                    material_code=material_code,
                    index=index,
                    include_dynamics=True,
                )
            for index in range(2):
                self._write_h5(
                    self.test_dir / f"test-{material_code}-{1 - index}.h5",
                    material_code=material_code,
                    index=index,
                    include_dynamics=False,
                )
        self._write_h5(
            self.train_dir / "bad.h5",
            material_code=0,
            index=0,
            include_dynamics=False,
        )

    def test_runner_rejects_same_resolved_split_directory_before_reading(self):
        (self.train_dir / "shared.h5").touch()

        with mock.patch(
            "diagnose_material_identifiability._read_split_records",
            return_value=[],
        ) as reader:
            with self.assertRaisesRegex(ValueError, "same resolved directory"):
                run_material_identifiability_audit(
                    self.train_dir,
                    self.train_dir,
                    self.output_dir,
                    AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
                )

        reader.assert_not_called()

    def test_runner_rejects_overlapping_resolved_h5_files_before_reading(self):
        train_path = self.train_dir / "train-object.h5"
        test_alias = self.test_dir / "test-alias.h5"
        train_path.touch()
        os.link(train_path, test_alias)

        with mock.patch(
            "diagnose_material_identifiability._read_split_records",
            return_value=[],
        ) as reader:
            with self.assertRaisesRegex(ValueError, "overlapping resolved H5"):
                run_material_identifiability_audit(
                    self.train_dir,
                    self.test_dir,
                    self.output_dir,
                    AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
                )

        reader.assert_not_called()

    def test_runner_rejects_overlapping_object_filenames_before_reading(self):
        (self.train_dir / "same-object.h5").touch()
        (self.test_dir / "same-object.h5").touch()

        with mock.patch(
            "diagnose_material_identifiability._read_split_records",
            return_value=[],
        ) as reader:
            with self.assertRaisesRegex(
                ValueError,
                "object/model ID.*same-object.h5",
            ):
                run_material_identifiability_audit(
                    self.train_dir,
                    self.test_dir,
                    self.output_dir,
                    AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
                )

        reader.assert_not_called()

    def test_parser_uses_frozen_production_defaults(self):
        args = build_parser().parse_args(
            ["--train-dir", str(self.train_dir), "--test-dir", str(self.test_dir)]
        )
        reduced_resampling_args = build_parser().parse_args(
            [
                "--train-dir", str(self.train_dir),
                "--test-dir", str(self.test_dir),
                "--permutations", "2",
                "--bootstrap-samples", "4",
            ]
        )

        self.assertEqual(args.seed, 0)
        self.assertEqual(args.folds, 5)
        self.assertEqual(args.permutations, 500)
        self.assertEqual(args.bootstrap_samples, 1000)
        self.assertEqual(args.contact_band_raw, 0.08)
        self.assertEqual(reduced_resampling_args.permutations, 2)
        self.assertEqual(reduced_resampling_args.bootstrap_samples, 4)

    def test_parser_rejects_invalid_statistical_settings(self):
        parser = build_parser()
        base = ["--train-dir", str(self.train_dir), "--test-dir", str(self.test_dir)]
        invalid_options = (
            ("--seed", "-1"),
            ("--folds", "1"),
            ("--permutations", "0"),
            ("--bootstrap-samples", "0"),
            ("--contact-band-raw", "0"),
        )

        for option, value in invalid_options:
            with self.subTest(option=option):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args([*base, option, value])

    def test_runner_streams_sorted_h5_records_and_propagates_invalid_metadata(self):
        self._create_smoke_fixture()
        progress = StringIO()

        with redirect_stdout(progress):
            paths = run_material_identifiability_audit(
                self.train_dir,
                self.test_dir,
                self.output_dir,
                AuditSettings(
                    seed=0,
                    folds=2,
                    permutations=2,
                    bootstrap_samples=4,
                ),
            )

        self.assertEqual(set(paths), set(OUTPUT_NAMES))
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["train_valid_count"], 18)
        self.assertEqual(metadata["test_valid_count"], 6)
        self.assertEqual(len(metadata["invalid_records"]), 1)
        self.assertTrue(metadata["invalid_records"][0]["path"].endswith("bad.h5"))
        self.assertEqual(metadata.get("audit_integrity_status"), "train_invalid")

        with paths["records"].open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        train_models = [row["model"] for row in records if row["split"] == "train"]
        test_models = [row["model"] for row in records if row["split"] == "test"]
        self.assertEqual(train_models, sorted(train_models))
        self.assertEqual(test_models, sorted(test_models))
        with paths["summary"].open(newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        self.assertEqual({row["status"] for row in summary_rows}, {"invalid"})
        self.assertTrue(
            all("train_invalid_records" in row["reason_codes"] for row in summary_rows)
        )
        self.assertIn("train 1/19", progress.getvalue())
        self.assertIn("test 1/6", progress.getvalue())
        self.assertIn("response", progress.getvalue())

    def test_runner_aborts_before_statistics_when_a_material_has_too_few_train_records(self):
        for material_code in (0, 1):
            for index in range(2):
                self._write_h5(
                    self.train_dir / f"train-{material_code}-{index}.h5",
                    material_code=material_code,
                    index=index,
                    include_dynamics=True,
                )
        self._write_h5(
            self.test_dir / "test-0.h5",
            material_code=0,
            index=0,
            include_dynamics=False,
        )

        with mock.patch(
            "diagnose_material_identifiability.build_coverage_rows",
            side_effect=AssertionError("statistical fitting started"),
        ):
            with self.assertRaisesRegex(ValueError, "sand"):
                run_material_identifiability_audit(
                    self.train_dir,
                    self.test_dir,
                    self.output_dir,
                    AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
                )

        self.assertFalse(self.output_dir.exists())

    def test_runner_reports_invalid_train_reasons_when_material_count_fails(self):
        for material_code in range(3):
            for index in range(2):
                path = self.train_dir / f"train-{material_code}-{index}.h5"
                self._write_h5(
                    path,
                    material_code=material_code,
                    index=index,
                    include_dynamics=True,
                )
                if material_code == 0:
                    with h5py.File(path, "a") as handle:
                        del handle["F"]
                        handle["F"] = np.zeros((25, 8, 8), dtype=np.float32)
        for material_code in range(3):
            self._write_h5(
                self.test_dir / f"test-{material_code}.h5",
                material_code=material_code,
                index=0,
                include_dynamics=False,
            )

        with self.assertRaisesRegex(
            ValueError,
            r"elastic has 0 valid train records.*invalid train reasons:.*F.*\(2\)",
        ):
            run_material_identifiability_audit(
                self.train_dir,
                self.test_dir,
                self.output_dir,
                AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
            )

        self.assertFalse(self.output_dir.exists())

    def test_runner_rejects_missing_valid_test_material_before_statistics(self):
        for material_code in range(3):
            for index in range(2):
                self._write_h5(
                    self.train_dir / f"train-{material_code}-{index}.h5",
                    material_code=material_code,
                    index=index,
                    include_dynamics=True,
                )
        for material_code in (0, 1):
            self._write_h5(
                self.test_dir / f"test-{material_code}.h5",
                material_code=material_code,
                index=0,
                include_dynamics=False,
            )

        with mock.patch(
            "diagnose_material_identifiability.build_coverage_rows",
            side_effect=ValueError("statistical fitting started"),
        ) as statistics:
            with self.assertRaisesRegex(ValueError, "sand.*0 valid test"):
                run_material_identifiability_audit(
                    self.train_dir,
                    self.test_dir,
                    self.output_dir,
                    AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
                )

        statistics.assert_not_called()
        self.assertFalse(self.output_dir.exists())

    def test_runner_rejects_empty_test_h5_directory_before_statistics_or_outputs(self):
        for material_code in range(3):
            for index in range(2):
                self._write_h5(
                    self.train_dir / f"train-{material_code}-{index}.h5",
                    material_code=material_code,
                    index=index,
                    include_dynamics=True,
                )

        with self.assertRaisesRegex(ValueError, "test.*\\*.h5"):
            run_material_identifiability_audit(
                self.train_dir,
                self.test_dir,
                self.output_dir,
                AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
            )

        self.assertFalse(self.output_dir.exists() and any(self.output_dir.iterdir()))

    def test_runner_rejects_empty_train_h5_directory_before_statistics_or_outputs(self):
        for material_code in range(3):
            self._write_h5(
                self.test_dir / f"test-{material_code}.h5",
                material_code=material_code,
                index=0,
                include_dynamics=False,
            )

        with self.assertRaisesRegex(ValueError, "train.*\\*.h5"):
            run_material_identifiability_audit(
                self.train_dir,
                self.test_dir,
                self.output_dir,
                AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
            )

        self.assertFalse(self.output_dir.exists() and any(self.output_dir.iterdir()))

    def test_runner_requires_overwrite_before_replacing_existing_outputs(self):
        self._create_smoke_fixture()
        settings = AuditSettings(folds=2, permutations=2, bootstrap_samples=4)
        run_material_identifiability_audit(
            self.train_dir, self.test_dir, self.output_dir, settings
        )

        with self.assertRaises(FileExistsError):
            run_material_identifiability_audit(
                self.train_dir, self.test_dir, self.output_dir, settings
            )

        paths = run_material_identifiability_audit(
            self.train_dir,
            self.test_dir,
            self.output_dir,
            settings,
            overwrite=True,
        )
        self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_runner_preflights_existing_output_before_reading_records(self):
        (self.train_dir / "train-only.h5").touch()
        (self.test_dir / "test-only.h5").touch()
        self.output_dir.mkdir()
        existing_target = self.output_dir / OUTPUT_NAMES["response"]
        existing_target.write_text("existing\n", encoding="utf-8")

        with mock.patch(
            "diagnose_material_identifiability._read_split_records",
            side_effect=FileExistsError("record reading started"),
        ) as reader:
            with self.assertRaisesRegex(FileExistsError, "response"):
                run_material_identifiability_audit(
                    self.train_dir,
                    self.test_dir,
                    self.output_dir,
                    AuditSettings(folds=2, permutations=2, bootstrap_samples=4),
                )

        reader.assert_not_called()
        self.assertEqual(existing_target.read_text(encoding="utf-8"), "existing\n")


if __name__ == "__main__":
    unittest.main()
