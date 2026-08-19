import csv
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch

import diagnose_local_deformation_fidelity as diagnostic_runner
import utils.local_deformation_fidelity as local_deformation_module
from utils.material_condition_diagnostics import MaterialRecord

from utils.local_deformation_fidelity import (
    LOCAL_RESPONSE_NAMES,
    OUTPUT_NAMES,
    build_rest_neighborhood,
    build_local_response_rows,
    compare_estimated_to_true_f,
    estimate_local_deformation,
    evaluate_calibration_gate,
    preflight_local_deformation_outputs,
    summarize_local_deformation,
    write_local_deformation_outputs,
)


def volumetric_grid(side: int = 4) -> np.ndarray:
    axis = np.linspace(-0.75, 0.75, side)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def affine_trajectory(rest: np.ndarray, transforms: list[np.ndarray], translations=None) -> np.ndarray:
    if translations is None:
        translations = [np.zeros(3) for _ in transforms]
    return np.stack(
        [rest @ transform.T + translation for transform, translation in zip(transforms, translations)],
        axis=0,
    )


class LocalDeformationEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.rest = volumetric_grid()

    def test_rigid_translation_has_identity_deformation(self):
        translations = [np.array([0.2 * t, -0.1 * t, 0.05 * t]) for t in range(4)]
        trajectory = affine_trajectory(
            self.rest,
            [np.eye(3) for _ in translations],
            translations,
        )
        neighborhood = build_rest_neighborhood(
            self.rest,
            k=16,
            condition_threshold=1e6,
            regularization_scale=1e-6,
        )

        result = estimate_local_deformation(trajectory, neighborhood)

        self.assertGreater(result.valid.mean(), 0.95)
        expected_identity = np.broadcast_to(
            np.eye(3), result.f_hat[:, result.valid].shape
        )
        np.testing.assert_allclose(
            result.f_hat[:, result.valid],
            expected_identity,
            atol=2e-5,
            rtol=2e-5,
        )
        self.assertLess(np.nanmax(result.legacy_strain), 4e-5)
        self.assertLess(np.nanmax(np.abs(result.jacobian - 1.0)), 4e-5)

    def test_uniform_scale_recovers_jacobian_and_green_strain(self):
        scales = np.asarray([1.0, 1.05, 1.10])
        transforms = [np.eye(3) * scale for scale in scales]
        trajectory = affine_trajectory(self.rest, transforms)
        neighborhood = build_rest_neighborhood(self.rest, k=16)

        result = estimate_local_deformation(trajectory, neighborhood)
        summary = summarize_local_deformation(result)

        np.testing.assert_allclose(
            summary["jacobian_mean"], scales**3, atol=5e-5, rtol=5e-5
        )
        expected_green = np.sqrt(3.0) * 0.5 * np.abs(scales**2 - 1.0)
        np.testing.assert_allclose(
            summary["green_strain_norm_mean"], expected_green, atol=5e-5, rtol=5e-5
        )
        self.assertLess(np.nanmax(summary["deviatoric_strain_norm_mean"]), 5e-5)

    def test_simple_shear_recovers_affine_tensor_and_deviatoric_strain(self):
        shear = 0.2
        transform = np.array(
            [[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        trajectory = affine_trajectory(self.rest, [np.eye(3), transform])
        neighborhood = build_rest_neighborhood(self.rest, k=16)

        result = estimate_local_deformation(trajectory, neighborhood)

        recovered = np.nanmedian(result.f_hat[1, result.valid], axis=0)
        np.testing.assert_allclose(recovered, transform, atol=3e-5, rtol=3e-5)
        self.assertGreater(np.nanmean(result.deviatoric_strain_norm[1]), 0.1)
        self.assertAlmostEqual(float(np.nanmean(result.jacobian[1])), 1.0, places=4)

    def test_coplanar_neighborhoods_are_marked_invalid(self):
        axis = np.linspace(-1.0, 1.0, 6)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        rest = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=-1)

        neighborhood = build_rest_neighborhood(rest, k=8)
        result = estimate_local_deformation(np.stack([rest, rest]), neighborhood)

        self.assertEqual(int(neighborhood.valid.sum()), 0)
        self.assertTrue(np.isnan(result.f_hat).all())
        self.assertEqual(summarize_local_deformation(result)["valid_fraction"], 0.0)

    def test_duplicate_coordinate_edges_do_not_create_nonfinite_stretch(self):
        rest = self.rest.copy()
        rest[1] = rest[0]
        trajectory = affine_trajectory(rest, [np.eye(3), np.eye(3) * 1.05])
        neighborhood = build_rest_neighborhood(rest, k=16)

        result = estimate_local_deformation(trajectory, neighborhood)

        self.assertGreater(result.valid.mean(), 0.9)
        self.assertTrue(np.isfinite(result.edge_stretch[:, result.valid]).all())

    def test_validation_rejects_bad_shapes_nonfinite_values_and_k(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            build_rest_neighborhood(np.zeros((8, 2)), k=4)
        with self.assertRaisesRegex(ValueError, "finite"):
            build_rest_neighborhood(np.full((8, 3), np.nan), k=4)
        with self.assertRaisesRegex(ValueError, "k"):
            build_rest_neighborhood(self.rest, k=len(self.rest))

        neighborhood = build_rest_neighborhood(self.rest, k=8)
        with self.assertRaisesRegex(ValueError, "particle count"):
            estimate_local_deformation(np.zeros((2, 8, 3)), neighborhood)


class CalibrationAndResponseTests(unittest.TestCase):
    def setUp(self):
        self.rest = volumetric_grid()
        self.scales = np.asarray([1.0, 1.03, 1.07, 1.12])
        self.transforms = np.stack([np.eye(3) * scale for scale in self.scales])
        self.trajectory = affine_trajectory(self.rest, list(self.transforms))
        self.neighborhood = build_rest_neighborhood(self.rest, k=16)
        self.result = estimate_local_deformation(self.trajectory, self.neighborhood)

    def test_compare_estimated_to_true_f_recovers_affine_calibration(self):
        true_f = np.broadcast_to(
            self.transforms[:, None, :, :],
            (len(self.transforms), len(self.rest), 3, 3),
        ).copy()

        comparison = compare_estimated_to_true_f(self.result, true_f)

        self.assertGreater(comparison["valid_fraction"], 0.95)
        self.assertLess(comparison["f_relative_error"], 5e-5)
        self.assertLess(comparison["j_absolute_error"], 5e-5)
        self.assertAlmostEqual(
            comparison["true_volumetric_strain_trajectory"],
            float(np.mean(np.abs(self.scales[1:] ** 3 - 1.0))),
            places=6,
        )

    def test_compare_rejects_misaligned_true_f(self):
        with self.assertRaisesRegex(ValueError, "true_f"):
            compare_estimated_to_true_f(
                self.result,
                np.zeros((len(self.transforms), len(self.rest) - 1, 3, 3)),
            )

    def test_build_local_response_rows_pairs_the_frozen_schema(self):
        gt = self.result
        pred_trajectory = affine_trajectory(
            self.rest,
            [np.eye(3) * (1.0 + 0.5 * (scale - 1.0)) for scale in self.scales],
        )
        pred = estimate_local_deformation(pred_trajectory, self.neighborhood)
        model_row = {
            "model": "sand_00.h5",
            "mat_type": 2,
            "material": "sand",
            "log10_e": 4.5,
            "nu": 0.3,
            "checkpoint": "checkpoint.safetensors",
            "config": "eval.yaml",
            "seed": 0,
            "sample_scope": "frozen test",
        }

        rows = build_local_response_rows(model_row, gt=gt, pred=pred)

        self.assertEqual(len(rows), len(LOCAL_RESPONSE_NAMES))
        self.assertEqual({row["response"] for row in rows}, set(LOCAL_RESPONSE_NAMES))
        self.assertTrue(all(row["absolute_error"] >= 0.0 for row in rows))
        self.assertTrue(
            all(
                np.isclose(
                    row["signed_error"], row["pred_value"] - row["gt_value"]
                )
                for row in rows
            )
        )

    @staticmethod
    def _calibration_rows(*, low_valid: bool = False, reverse_primary: bool = False):
        rows = []
        for k in (8, 16, 32):
            for mat_type, material in enumerate(("elastic", "plasticine", "sand")):
                for index in range(20):
                    true_legacy = 0.01 + 0.002 * index + 0.001 * mat_type
                    true_volume = 0.02 + 0.003 * index + 0.001 * mat_type
                    estimated_legacy = true_legacy * (1.0 + 0.001 * (k - 16))
                    estimated_volume = true_volume * (1.0 + 0.001 * (k - 16))
                    if reverse_primary and k == 16:
                        estimated_legacy = -true_legacy
                        estimated_volume = -true_volume
                    rows.append(
                        {
                            "model": f"{material}_{index:02d}.h5",
                            "material": material,
                            "mat_type": mat_type,
                            "k": k,
                            "valid_fraction": 0.90 if low_valid and material == "sand" and k == 16 else 0.99,
                            "estimated_legacy_strain_trajectory": estimated_legacy,
                            "true_legacy_strain_trajectory": true_legacy,
                            "estimated_volumetric_strain_trajectory": estimated_volume,
                            "true_volumetric_strain_trajectory": true_volume,
                        }
                    )
        return rows

    def test_calibration_gate_passes_only_valid_correlated_robust_estimator(self):
        passed = evaluate_calibration_gate(self._calibration_rows())
        self.assertEqual(passed["status"], "passed")
        self.assertEqual(passed["reasons"], [])
        self.assertGreaterEqual(passed["legacy_spearman_k16"], 0.8)
        self.assertGreaterEqual(passed["volumetric_spearman_k16"], 0.8)

        low_valid = evaluate_calibration_gate(self._calibration_rows(low_valid=True))
        self.assertEqual(low_valid["status"], "failed")
        self.assertTrue(any("valid_fraction" in reason for reason in low_valid["reasons"]))

        reversed_rows = evaluate_calibration_gate(
            self._calibration_rows(reverse_primary=True)
        )
        self.assertEqual(reversed_rows["status"], "failed")
        self.assertTrue(any("Spearman" in reason for reason in reversed_rows["reasons"]))


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name) / "b04"
        self.calibration_rows = CalibrationAndResponseTests._calibration_rows()
        self.model_rows = [
            {
                "model": "sand_00.h5",
                "mat_type": 2,
                "material": "sand",
                "log10_e": 4.5,
                "nu": 0.3,
                "valid_fraction": 0.99,
                "condition_number_median": 10.0,
                "condition_number_p95": 20.0,
                "local_f_mse": 0.001,
                "local_j_mae": 0.002,
                "checkpoint": "checkpoint.safetensors",
                "config": "eval.yaml",
                "seed": 0,
                "sample_scope": "frozen test",
            }
        ]
        self.frame_rows = [
            {
                "model": "sand_00.h5",
                "material": "sand",
                "frame": 24,
                "gt_legacy_strain": 0.2,
                "pred_legacy_strain": 0.1,
            }
        ]
        self.response_rows = [
            {
                **self.model_rows[0],
                "response": "legacy_strain_f24",
                "response_tier": "primary",
                "gt_value": 0.2,
                "pred_value": 0.1,
                "signed_error": -0.1,
                "absolute_error": 0.1,
            }
        ]
        self.fidelity_rows = [
            {
                "analysis": "prediction_gt_fidelity",
                "group": "sand",
                "response": "legacy_strain_f24",
                "n": 14,
                "mae": 0.1,
            }
        ]
        self.metadata = {
            "schema_version": "1.0",
            "checkpoint": "checkpoint.safetensors",
            "config": "eval.yaml",
            "seed": 0,
            "split": "frozen 41-model test",
            "model_counts": {"elastic": 13, "plasticine": 14, "sand": 14},
            "calibration_status": "passed",
            "calibration_gate": evaluate_calibration_gate(self.calibration_rows),
            "k_primary": 16,
            "k_sensitivity": [8, 32],
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write(self, **changes):
        arguments = {
            "output_dir": self.output_dir,
            "calibration_rows": self.calibration_rows,
            "model_rows": self.model_rows,
            "frame_rows": self.frame_rows,
            "response_rows": self.response_rows,
            "fidelity_rows": self.fidelity_rows,
            "metadata": self.metadata,
            "overwrite": False,
        }
        arguments.update(changes)
        return write_local_deformation_outputs(**arguments)

    def test_writer_emits_exact_outputs_and_chinese_interpretation_gate(self):
        paths = self._write()

        self.assertEqual(set(paths), set(OUTPUT_NAMES))
        self.assertEqual({path.name for path in paths.values()}, set(OUTPUT_NAMES.values()))
        report = paths["report"].read_text(encoding="utf-8")
        self.assertIn("局部形变响应保真度", report)
        self.assertIn("校准通过", report)
        self.assertIn("41-model test", report)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["calibration_gate"]["status"], "passed")
        with paths["calibration"].open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), len(self.calibration_rows))

    def test_failed_calibration_report_forbids_constitutive_interpretation(self):
        failed_metadata = dict(self.metadata)
        failed_metadata["calibration_gate"] = evaluate_calibration_gate(
            CalibrationAndResponseTests._calibration_rows(low_valid=True)
        )
        failed_metadata["calibration_status"] = "failed"

        paths = self._write(metadata=failed_metadata)

        report = paths["report"].read_text(encoding="utf-8")
        self.assertIn("校准失败", report)
        self.assertIn("禁止", report)
        self.assertIn("本构", report)

    def test_preflight_refuses_existing_target_unless_overwrite(self):
        self.output_dir.mkdir()
        existing = self.output_dir / OUTPUT_NAMES["models"]
        existing.write_text("old", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            preflight_local_deformation_outputs(self.output_dir, overwrite=False)

        targets = preflight_local_deformation_outputs(self.output_dir, overwrite=True)
        self.assertEqual(targets["models"], existing)

    def test_writer_rejects_empty_payload_before_creating_final_files(self):
        with self.assertRaisesRegex(ValueError, "model_rows"):
            self._write(model_rows=[])

        self.assertFalse(self.output_dir.exists())

    def test_overwrite_backup_failure_restores_every_old_output(self):
        first_paths = self._write()
        old_contents = {name: path.read_bytes() for name, path in first_paths.items()}
        real_replace = local_deformation_module.os.replace
        backup_moves = 0

        def fail_second_backup(source, target):
            nonlocal backup_moves
            source_path = Path(source)
            if source_path.parent == self.output_dir:
                backup_moves += 1
                if backup_moves == 2:
                    raise OSError("injected backup failure")
            return real_replace(source, target)

        with mock.patch.object(
            local_deformation_module.os, "replace", side_effect=fail_second_backup
        ):
            with self.assertRaisesRegex(OSError, "injected backup failure"):
                self._write(overwrite=True)

        self.assertEqual(
            {name: path.read_bytes() for name, path in first_paths.items()},
            old_contents,
        )


class DiagnosticRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train_dir = self.root / "mm3_train"
        self.test_dir = self.root / "mm3_test"
        self.train_dir.mkdir()
        self.test_dir.mkdir()
        self.checkpoint = self.root / "checkpoint.safetensors"
        self.checkpoint.write_bytes(b"checkpoint")
        self.output_dir = self.root / "output"
        self.records = []
        self.trajectories = {}
        rest = volumetric_grid()

        for mat_type, count in ((0, 13), (1, 14), (2, 14)):
            for index in range(count):
                model = f"material_{mat_type}_{index:02d}.h5"
                amplitude = 0.02 + 0.001 * index + 0.005 * mat_type
                scales = np.linspace(1.0, 1.0 + amplitude, 25)
                trajectory = affine_trajectory(
                    rest, [np.eye(3) * scale for scale in scales]
                )
                self.trajectories[model] = torch.from_numpy(trajectory).unsqueeze(0).float()
                self.records.append(
                    MaterialRecord(
                        model=model,
                        mat_type=mat_type,
                        log10_e=4.0 + 0.05 * index,
                        nu=0.10 + 0.01 * index,
                    )
                )

        for mat_type in range(3):
            amplitude = 0.04 + 0.03 * mat_type
            scales = np.linspace(1.0, 1.0 + amplitude, 25)
            trajectory = affine_trajectory(
                rest, [np.eye(3) * scale for scale in scales]
            )
            true_f = np.broadcast_to(
                np.stack([np.eye(3) * scale for scale in scales])[:, None],
                (25, len(rest), 3, 3),
            ).copy()
            path = self.train_dir / f"train_{mat_type}.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("x", data=trajectory)
                handle.create_dataset("F", data=true_f.reshape(25, len(rest), 9))
                handle.create_dataset("E", data=10.0 ** (4.0 + mat_type))
                handle.create_dataset("nu", data=0.15 + 0.05 * mat_type)
                handle.create_dataset("mat_type", data=mat_type)

        self.args = SimpleNamespace(
            pc_size=len(rest),
            input_frames=5,
            output_frames=1,
            eval_batch_size=1,
            dataloader_num_workers=0,
            num_inference_steps=1,
            seed=0,
            device="cpu",
            resume=str(self.checkpoint),
            model_config=SimpleNamespace(),
            train_dataset=SimpleNamespace(
                dataset_path=str(self.test_dir),
                input_frames=5,
                output_frames=1,
                norm_fac=5.0,
            ),
        )
        self.batches = [
            {
                "model": [record.model],
                "start_idx": torch.tensor([0]),
                "points_src": self.trajectories[record.model][:, :5].clone(),
                "E": torch.tensor([record.log10_e]),
                "nu": torch.tensor([record.nu]),
                "mat_type": torch.tensor([record.mat_type]),
            }
            for record in self.records
        ]
        self.rollout_calls = []

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _runtime(self):
        outer = self

        class FakeDataset:
            def __init__(self, split, config):
                self.split_lst_save = [record.model for record in outer.records]
                self.batches = outer.batches

        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def to(self, device):
                return self

            def load_state_dict(self, state, strict):
                if strict is not True:
                    raise AssertionError("strict load required")

            def eval(self):
                return self

            def requires_grad_(self, enabled):
                return self

        class FakePipeline:
            def __init__(self, *, model, scheduler):
                if scheduler is not None:
                    raise AssertionError("B0.4 requires deterministic inference")

        return diagnostic_runner.RuntimeComponents(
            dataset_cls=FakeDataset,
            model_cls=FakeModel,
            pipeline_cls=FakePipeline,
            checkpoint_loader=lambda path, device: {"weight": torch.tensor(1.0)},
            dataloader_cls=lambda dataset, **kwargs: [
                (batch, {}) for batch in dataset.batches
            ],
            compile_model=lambda model: model,
        )

    def _rollout(self, pipeline, batch, args, e_value, nu_value, mat_type):
        model = batch["model"][0]
        self.rollout_calls.append(model)
        return self.trajectories[model].clone()

    def test_full_runner_calibrates_then_evaluates_frozen_41_models(self):
        with (
            mock.patch.object(diagnostic_runner, "_validate_b0_identity"),
            mock.patch.object(
                diagnostic_runner, "load_material_records", return_value=self.records
            ),
            mock.patch.object(diagnostic_runner, "_validate_normal_material_condition"),
            mock.patch.object(diagnostic_runner, "rollout_condition", self._rollout),
            mock.patch.object(diagnostic_runner, "reset_inference_seed"),
            mock.patch.object(
                diagnostic_runner,
                "_build_raw_reference",
                side_effect=lambda batch, dataset: self.trajectories[batch["model"][0]].clone(),
            ),
            mock.patch.object(
                diagnostic_runner, "_rollout_autocast_context", return_value=nullcontext()
            ),
        ):
            paths = diagnostic_runner.run_local_deformation_fidelity(
                self.args,
                checkpoint=self.checkpoint,
                config_path=Path("configs/eval_mm3_contact_cond.yaml"),
                train_dir=self.train_dir,
                output_dir=self.output_dir,
                seed=0,
                bootstrap_samples=10,
                calibration_per_material=1,
                overwrite=False,
                runtime=self._runtime(),
            )

        self.assertEqual(len(self.rollout_calls), 41)
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["calibration_gate"]["status"], "passed")
        self.assertEqual(metadata["calibration_status"], "passed")
        with paths["models"].open(newline="", encoding="utf-8") as handle:
            model_reader = csv.DictReader(handle)
            model_rows = list(model_reader)
            self.assertEqual(len(model_rows), 41)
            self.assertIn("local_f_mse_short", model_reader.fieldnames)
            self.assertIn("local_f_mse_mid", model_reader.fieldnames)
            self.assertIn("local_f_mse_long", model_reader.fieldnames)
        with paths["frames"].open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 41 * 25)
        with paths["responses"].open(newline="", encoding="utf-8") as handle:
            self.assertEqual(
                len(list(csv.DictReader(handle))), 41 * len(LOCAL_RESPONSE_NAMES)
            )
        with paths["fidelity"].open(newline="", encoding="utf-8") as handle:
            fidelity_reader = csv.DictReader(handle)
            list(fidelity_reader)
            self.assertIn("gt_ordinary_rho", fidelity_reader.fieldnames)
            self.assertIn("gt_partial_rho", fidelity_reader.fieldnames)

    def test_calibration_selection_is_seeded_stratified_and_parser_is_frozen(self):
        selected = diagnostic_runner.select_calibration_paths(
            self.train_dir, per_material=1, seed=0
        )
        self.assertEqual({mat_type for mat_type, _ in selected}, {0, 1, 2})
        self.assertEqual(
            selected,
            diagnostic_runner.select_calibration_paths(
                self.train_dir, per_material=1, seed=0
            ),
        )

        parser = diagnostic_runner.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        parsed = parser.parse_args(
            [
                "--config",
                "eval.yaml",
                "--checkpoint",
                "checkpoint.safetensors",
                "--train-dir",
                "mm3_train",
            ]
        )
        self.assertEqual(parsed.seed, 0)
        self.assertEqual(parsed.calibration_per_material, 20)
        self.assertEqual(parsed.bootstrap_samples, 10000)

    def test_checkpoint_identity_fails_before_calibration(self):
        missing = self.root / "missing.safetensors"
        self.args.resume = str(missing)
        with (
            mock.patch.object(diagnostic_runner, "_validate_b0_identity"),
            mock.patch.object(diagnostic_runner, "run_calibration") as calibration,
        ):
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
                diagnostic_runner.run_local_deformation_fidelity(
                    self.args,
                    checkpoint=missing,
                    config_path=Path("configs/eval_mm3_contact_cond.yaml"),
                    train_dir=self.train_dir,
                    output_dir=self.output_dir,
                    seed=0,
                    bootstrap_samples=10,
                    calibration_per_material=1,
                    overwrite=False,
                    runtime=self._runtime(),
                )

        calibration.assert_not_called()


if __name__ == "__main__":
    unittest.main()
