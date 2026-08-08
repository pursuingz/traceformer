import tempfile
import unittest
from pathlib import Path

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
            {},
            parameter,
            folds,
            augmented_parameter=None,
        )
        response_prediction = material_identifiability._nested_cv_predictions(
            nuisance,
            {"log10_e": parameter, "nu": np.asarray([record["nu"] for record in records])},
            response,
            folds,
            augmented_parameter=None,
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


if __name__ == "__main__":
    unittest.main()
