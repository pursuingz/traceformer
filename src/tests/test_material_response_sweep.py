import math
import csv
import io
import tempfile
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


class SweepDefinitionTests(unittest.TestCase):
    def test_conditions_are_fixed_and_hold_the_other_parameter_constant(self):
        from src.utils.material_response_sweep import (
            SWEEP_CONDITIONS,
            build_sweep_conditions,
        )

        self.assertEqual(
            SWEEP_CONDITIONS,
            (
                "normal",
                "e_low",
                "e_mid",
                "e_high",
                "nu_low",
                "nu_mid",
                "nu_high",
            ),
        )
        record = SimpleNamespace(log10_e=5.8, nu=0.32, mat_type=2)
        conditions = build_sweep_conditions(record)
        self.assertEqual(tuple(conditions), SWEEP_CONDITIONS)
        self.assertEqual(
            [conditions[name].log10_e for name in ("e_low", "e_mid", "e_high")],
            [4.5, 5.5, 6.5],
        )
        self.assertTrue(
            all(conditions[name].nu == 0.32 for name in ("e_low", "e_mid", "e_high"))
        )
        self.assertEqual(
            [conditions[name].nu for name in ("nu_low", "nu_mid", "nu_high")],
            [0.10, 0.25, 0.40],
        )
        self.assertTrue(
            all(
                conditions[name].log10_e == 5.8
                for name in ("nu_low", "nu_mid", "nu_high")
            )
        )
        self.assertTrue(all(item.mat_type == 2 for item in conditions.values()))

    def test_condition_builder_rejects_nonfinite_or_unsupported_materials(self):
        from src.utils.material_response_sweep import build_sweep_conditions

        for record in (
            SimpleNamespace(log10_e=math.nan, nu=0.2, mat_type=0),
            SimpleNamespace(log10_e=5.0, nu=math.inf, mat_type=0),
            SimpleNamespace(log10_e=5.0, nu=0.2, mat_type=3),
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                build_sweep_conditions(record)


class SweepMetricTests(unittest.TestCase):
    @staticmethod
    def _reference_cloud():
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def test_state_metrics_separate_motion_shape_volume_and_penetration(self):
        from src.utils.material_response_sweep import trajectory_state_metrics

        reference = self._reference_cloud()
        prediction = torch.stack(
            [reference + torch.tensor([0.0, -0.25, 0.0]), reference * 2.0], dim=0
        )
        result = trajectory_state_metrics(
            prediction,
            reference,
            floor_height=-0.10,
        )

        self.assertGreater(result["motion_rms"], 0.0)
        self.assertGreater(result["f24_centroid_displacement"], 0.0)
        self.assertLess(result["f24_shape_deformation"], 1e-12)
        self.assertAlmostEqual(result["f24_volume_relative_change"], 7.0, places=6)
        self.assertGreater(result["penetration_rate"], 0.0)
        self.assertGreater(result["penetration_depth"], 0.0)

    def test_response_metrics_ignore_common_translation_in_shape_response(self):
        from src.utils.material_response_sweep import response_metrics

        reference = self._reference_cloud()
        normal = torch.stack([reference, reference + 0.1], dim=0)
        counterfactual = normal + torch.tensor([0.5, -0.25, 0.0])
        result = response_metrics(normal, counterfactual)

        self.assertGreater(result["prediction_response_mse"], 0.0)
        self.assertGreater(result["final_response_mse"], 0.0)
        self.assertGreater(result["f24_centroid_response"], 0.0)
        self.assertLess(result["f24_shape_response"], 1e-12)

    def test_monotonicity_reports_direction_and_ties(self):
        from src.utils.material_response_sweep import spearman_monotonicity

        decreasing = spearman_monotonicity(
            [4.5, 5.5, 6.5], [3.0, 2.0, 1.0], expected_direction="decreasing"
        )
        self.assertAlmostEqual(decreasing["rho"], -1.0)
        self.assertTrue(decreasing["strict_monotonic"])
        self.assertTrue(decreasing["weak_monotonic"])

        tied = spearman_monotonicity(
            [0.1, 0.25, 0.4], [3.0, 3.0, 2.0], expected_direction="decreasing"
        )
        self.assertFalse(tied["strict_monotonic"])
        self.assertTrue(tied["weak_monotonic"])

        with warnings.catch_warnings(record=True) as caught:
            constant = spearman_monotonicity(
                [0.1, 0.25, 0.4], [2.0, 2.0, 2.0], expected_direction="decreasing"
            )
        self.assertEqual(caught, [])
        self.assertEqual(constant["rho"], 0.0)

    def test_metrics_and_monotonicity_reject_invalid_inputs(self):
        from src.utils.material_response_sweep import (
            response_metrics,
            spearman_monotonicity,
            trajectory_state_metrics,
        )

        cloud = self._reference_cloud()
        with self.assertRaises(ValueError):
            trajectory_state_metrics(torch.zeros(2, 4, 2), cloud, -0.1)
        with self.assertRaises(ValueError):
            response_metrics(torch.zeros(2, 4, 3), torch.zeros(3, 4, 3))
        with self.assertRaises(ValueError):
            spearman_monotonicity([1.0, 2.0], [2.0, 1.0], "decreasing")


class SweepSummaryTests(unittest.TestCase):
    @staticmethod
    def _raw_rows():
        from src.utils.material_response_sweep import SWEEP_CONDITIONS

        rows = []
        material_counts = ((0, 13), (1, 14), (2, 14))
        model_index = 0
        for mat_type, count in material_counts:
            for _ in range(count):
                model = f"model_{model_index:02d}.h5"
                true_e = 5.8
                true_nu = 0.3
                for condition in SWEEP_CONDITIONS:
                    if condition.startswith("e_"):
                        level = {"e_low": 4.5, "e_mid": 5.5, "e_high": 6.5}[condition]
                        scanned_e, scanned_nu = level, true_nu
                        shape = {4.5: 3.0, 5.5: 2.0, 6.5: 1.0}[level]
                        volume = 0.2
                        response = 0.04
                    elif condition.startswith("nu_"):
                        level = {"nu_low": 0.1, "nu_mid": 0.25, "nu_high": 0.4}[condition]
                        scanned_e, scanned_nu = true_e, level
                        shape = 2.0
                        volume = {0.1: 3.0, 0.25: 2.0, 0.4: 1.0}[level]
                        response = 0.03
                    else:
                        scanned_e, scanned_nu = true_e, true_nu
                        shape, volume, response = 2.0, 0.2, 0.0
                    rows.append(
                        {
                            "checkpoint": "checkpoint.safetensors",
                            "config": "eval.yaml",
                            "seed": 0,
                            "sample_scope": "frozen 41-model start0 B2",
                            "model": model,
                            "mat_type": mat_type,
                            "true_log10_e": true_e,
                            "true_nu": true_nu,
                            "condition": condition,
                            "scanned_log10_e": scanned_e,
                            "scanned_nu": scanned_nu,
                            "motion_rms": 1.0,
                            "f24_centroid_displacement": 0.5,
                            "f24_shape_deformation": shape,
                            "f24_volume_relative_change": volume,
                            "penetration_rate": 0.0,
                            "penetration_depth": 0.0,
                            "prediction_response_mse": response,
                            "final_response_mse": response,
                            "f24_centroid_response": response,
                            "f24_shape_response": response,
                        }
                    )
                model_index += 1
        return rows

    def test_validation_and_summaries_enforce_frozen_protocol(self):
        from src.utils.material_response_sweep import (
            build_group_summaries,
            build_model_summaries,
            validate_raw_rows,
        )

        rows = self._raw_rows()
        validate_raw_rows(rows)
        model_rows = build_model_summaries(rows)
        self.assertEqual(len(model_rows), 41)
        self.assertTrue(all(row["e_shape_weak_monotonic"] for row in model_rows))
        self.assertTrue(all(row["nu_volume_weak_monotonic"] for row in model_rows))
        self.assertTrue(all(row["classification"] == "directionally_plausible" for row in model_rows))

        summary = build_group_summaries(model_rows)
        self.assertEqual([row["group"] for row in summary], ["overall", "elastic", "plasticine", "sand"])
        self.assertEqual([row["n"] for row in summary], [41, 13, 14, 14])
        self.assertTrue(all(row["directionally_plausible_fraction"] == 1.0 for row in summary))

        with self.assertRaises(ValueError):
            validate_raw_rows(rows[:-1])
        duplicated = list(rows)
        duplicated[-1] = dict(duplicated[0])
        with self.assertRaises(ValueError):
            validate_raw_rows(duplicated)
        inconsistent = [dict(row) for row in rows]
        inconsistent[0]["checkpoint"] = "other.safetensors"
        with self.assertRaises(ValueError):
            validate_raw_rows(inconsistent)

    def test_writers_emit_fixed_csv_and_chinese_limitations(self):
        from src.utils.material_response_sweep import (
            build_group_summaries,
            build_model_summaries,
            write_sweep_outputs,
        )

        raw_rows = self._raw_rows()
        model_rows = build_model_summaries(raw_rows)
        summary_rows = build_group_summaries(model_rows)
        metadata = {
            "checkpoint": "checkpoint.safetensors",
            "config": "eval.yaml",
            "seed": 0,
            "profile": "b3a45",
            "sample_scope": "frozen 41-model start0 B2",
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = write_sweep_outputs(
                directory, raw_rows, model_rows, summary_rows, metadata
            )
            self.assertEqual(set(paths), {"raw", "model_summary", "summary", "report"})
            with Path(paths["raw"]).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 287)
            with Path(paths["model_summary"]).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 41)
            report = Path(paths["report"]).read_text(encoding="utf-8")
            self.assertIn("profile: `b3a45`", report)
            self.assertIn("没有 counterfactual GT", report)
            self.assertIn("不能判断反事实轨迹的准确率", report)
            self.assertIn("elastic", report)
            self.assertIn("directionally_plausible", report)


class SweepCliTests(unittest.TestCase):
    @staticmethod
    def _reference_cloud():
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @classmethod
    def _batch(cls, model="model_00.h5", mat_type=1):
        points_src = cls._reference_cloud().view(1, 1, 4, 3).repeat(1, 5, 1, 1)
        return {
            "model": [model],
            "start_idx": torch.tensor([0]),
            "points_src": points_src,
            "floor_height": torch.tensor([-1.0]),
            "E": torch.tensor([5.8]),
            "nu": torch.tensor([0.3]),
            "mat_type": torch.tensor([mat_type]),
        }

    @staticmethod
    def _args(dataset_path="mm3_data/mm3_test"):
        return SimpleNamespace(
            pc_size=2048,
            eval_batch_size=1,
            dataloader_num_workers=0,
            seed=0,
            device="cuda",
            input_frames=5,
            output_frames=1,
            use_diffusion=False,
            model_config=SimpleNamespace(),
            train_dataset=SimpleNamespace(
                dataset_path=dataset_path,
                input_frames=5,
                output_frames=1,
            ),
        )

    def test_parser_requires_frozen_config_and_checkpoint(self):
        from src.diagnose_material_response_sweep import build_parser

        parsed = build_parser().parse_args(
            [
                "--config",
                "src/configs/eval_mm3_contact_cond.yaml",
                "--checkpoint",
                "outputs/mm3_contact_cond_8L/checkpoint-90000/model.safetensors",
            ]
        )
        self.assertEqual(parsed.seed, 0)
        self.assertEqual(parsed.output_dir, "results/material_response_sweep_b2")
        self.assertEqual(parsed.profile, "contact_cond90")
        b3 = build_parser().parse_args(
            [
                "--profile",
                "b3a45",
                "--config",
                "src/configs/eval_mm3_b3a_material_state_adapter_45k.yaml",
                "--checkpoint",
                "outputs/mm3_b3a_material_state_adapter_8L/checkpoint-45000/model.safetensors",
            ]
        )
        self.assertEqual(b3.profile, "b3a45")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["--config", "src/configs/eval_mm3_contact_cond.yaml"]
            )

    def test_b3_profile_is_strict_and_does_not_weaken_contact_profile(self):
        from omegaconf import OmegaConf
        from src.diagnose_material_condition import _validate_b0_identity
        from src.options import TestingConfig

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "eval_mm3_b3a_material_state_adapter_45k.yaml"
        )
        args = OmegaConf.merge(
            OmegaConf.structured(TestingConfig), OmegaConf.load(config_path)
        )
        _validate_b0_identity(args, profile="b3a45")
        with self.assertRaisesRegex(ValueError, "unexpected model_config"):
            _validate_b0_identity(args, profile="contact_cond90")

        args.model_config.material_state_rank = 32
        with self.assertRaisesRegex(ValueError, "material_state_rank"):
            _validate_b0_identity(args, profile="b3a45")

    def test_one_model_runs_seven_paired_conditions_with_identical_seed(self):
        import src.diagnose_material_response_sweep as diagnostic

        batch = self._batch()
        record = SimpleNamespace(
            model="model_00.h5", mat_type=1, log10_e=5.8, nu=0.3
        )
        args = self._args()
        full_trajectory = batch["points_src"][:, -1:].repeat(1, 25, 1, 1)
        observed = []

        def fake_rollout(pipeline, batch_arg, args_arg, e_value, nu_value, mat_type):
            observed.append((e_value, nu_value, mat_type))
            return full_trajectory.clone()

        metadata = {
            "checkpoint": "checkpoint.safetensors",
            "config": "eval.yaml",
            "seed": 7,
            "sample_scope": diagnostic.B2_SAMPLE_SCOPE,
        }
        with (
            mock.patch.object(
                diagnostic, "rollout_condition", side_effect=fake_rollout
            ),
            mock.patch.object(diagnostic, "reset_inference_seed") as reset_seed,
            mock.patch.object(
                diagnostic, "_validate_normal_material_condition"
            ) as validate_material,
        ):
            rows = diagnostic.evaluate_sweep_conditions(
                object(), batch, args, record, seed=7, metadata=metadata
            )

        self.assertEqual([row["condition"] for row in rows], list(diagnostic.SWEEP_CONDITIONS))
        self.assertEqual(len(observed), 7)
        self.assertEqual(observed[1:4], [(4.5, 0.3, 1), (5.5, 0.3, 1), (6.5, 0.3, 1)])
        self.assertEqual(observed[4:], [(5.8, 0.1, 1), (5.8, 0.25, 1), (5.8, 0.4, 1)])
        self.assertEqual(reset_seed.call_count, 7)
        self.assertEqual(
            reset_seed.call_args_list,
            [mock.call(7, torch.device("cuda"))] * 7,
        )
        validate_material.assert_called_once_with(batch, record)
        self.assertEqual(rows[0]["prediction_response_mse"], 0.0)

    def test_material_mismatch_fails_before_any_rollout(self):
        import src.diagnose_material_response_sweep as diagnostic

        with (
            mock.patch.object(
                diagnostic,
                "_validate_normal_material_condition",
                side_effect=ValueError("batch/HDF5 E mismatch"),
            ),
            mock.patch.object(diagnostic, "rollout_condition") as rollout,
        ):
            with self.assertRaisesRegex(ValueError, "batch/HDF5 E mismatch"):
                diagnostic.evaluate_sweep_conditions(
                    object(),
                    self._batch(),
                    self._args(),
                    SimpleNamespace(
                        model="model_00.h5", mat_type=1, log10_e=5.8, nu=0.3
                    ),
                    seed=0,
                    metadata={
                        "checkpoint": "checkpoint.safetensors",
                        "config": "eval.yaml",
                        "seed": 0,
                        "sample_scope": diagnostic.B2_SAMPLE_SCOPE,
                    },
                )
        rollout.assert_not_called()

    def test_full_run_loads_once_and_executes_exactly_287_rollouts(self):
        import src.diagnose_material_response_sweep as diagnostic
        from src.diagnose_material_condition import EXPECTED_MODEL_NAMES

        model_names = sorted(EXPECTED_MODEL_NAMES)
        records = [
            SimpleNamespace(
                model=name,
                mat_type=int(Path(name).stem.rsplit("_", 1)[1]),
                log10_e=5.8,
                nu=0.3,
            )
            for name in model_names
        ]
        records_by_name = {record.model: record for record in records}

        class FakeDataset:
            def __init__(self, split, config):
                self.split_lst_save = list(model_names)

        class FakeModel:
            def __init__(self):
                self.load_calls = 0

            def to(self, device):
                return self

            def load_state_dict(self, checkpoint, strict):
                self.load_calls += 1
                self.strict = strict

            def eval(self):
                return self

            def requires_grad_(self, enabled):
                return self

        fake_model = FakeModel()
        model_constructor = mock.Mock(return_value=fake_model)
        pipeline_constructor = mock.Mock(return_value=object())
        checkpoint_loader = mock.Mock(return_value={"weight": 1})
        compile_model = mock.Mock(side_effect=lambda model: model)
        dataloader = [
            (self._batch(name, records_by_name[name].mat_type), {})
            for name in model_names
        ]
        dataloader_constructor = mock.Mock(return_value=dataloader)
        runtime = diagnostic.RuntimeComponents(
            dataset_cls=FakeDataset,
            model_cls=model_constructor,
            pipeline_cls=pipeline_constructor,
            checkpoint_loader=checkpoint_loader,
            dataloader_cls=dataloader_constructor,
            compile_model=compile_model,
        )
        args = self._args()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "mm3_data" / "mm3_test"
            dataset_root.mkdir(parents=True)
            checkpoint = (
                root
                / "outputs"
                / "mm3_contact_cond_8L"
                / "checkpoint-90000"
                / "model.safetensors"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            args.train_dataset.dataset_path = str(dataset_root)
            full_trajectory = self._batch()["points_src"][:, -1:].repeat(
                1, 25, 1, 1
            )
            validation_stages = []
            with (
                mock.patch.object(
                    diagnostic,
                    "_validate_b0_identity",
                    side_effect=lambda checked_args, **kwargs: validation_stages.append(
                        hasattr(checked_args.model_config, "cond_frames")
                    ),
                ) as validate_identity,
                mock.patch.object(
                    diagnostic, "load_material_records", return_value=records
                ) as load_records,
                mock.patch.object(
                    diagnostic,
                    "rollout_condition",
                    return_value=full_trajectory,
                ) as rollout,
                mock.patch.object(diagnostic, "reset_inference_seed") as reset_seed,
                redirect_stdout(io.StringIO()),
            ):
                rows = diagnostic.run_material_response_sweep(
                    args,
                    checkpoint=checkpoint,
                    output_dir=root / "reports",
                    config_path=Path("src/configs/eval_mm3_contact_cond.yaml"),
                    seed=0,
                    runtime=runtime,
                )

            self.assertEqual(len(rows), 287)
            self.assertEqual(model_constructor.call_count, 1)
            self.assertEqual(checkpoint_loader.call_count, 1)
            self.assertEqual(fake_model.load_calls, 1)
            self.assertEqual(compile_model.call_count, 1)
            self.assertEqual(pipeline_constructor.call_count, 1)
            self.assertEqual(rollout.call_count, 287)
            self.assertEqual(reset_seed.call_count, 287)
            self.assertEqual(load_records.call_count, 1)
            self.assertEqual(validate_identity.call_count, 2)
            self.assertEqual(validation_stages, [False, False])
            self.assertEqual(args.model_config.cond_frames, 5)
            self.assertTrue(
                (root / "reports" / "material_response_sweep_b2_raw.csv").is_file()
            )
            self.assertTrue(
                (root / "reports" / "material_response_sweep_b2.md").is_file()
            )


if __name__ == "__main__":
    unittest.main()
