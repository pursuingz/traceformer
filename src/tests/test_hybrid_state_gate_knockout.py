import csv
import io
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import torch

from utils.eval_metrics import per_window_metrics
from utils.hybrid_state_gate_knockout import (
    KNOCKOUT_CONDITIONS,
    KNOCKOUT_METRICS,
    build_paired_rows,
    dynamic_gate_verdict,
    masked_feedback_gates,
    paired_delta_summary,
    summarize_paired_rows,
    trajectory_knockout_metrics,
    validate_raw_rows,
)


class WriterTests(unittest.TestCase):
    @staticmethod
    def _metadata():
        return {
            "checkpoint": "outputs/run/checkpoint-90000/model.safetensors",
            "config": "src/configs/eval.yaml",
            "seed": 0,
            "sample_scope": "frozen 41-model start_idx=0 full-horizon B1b test",
            "bootstrap_samples": 20,
            "bootstrap_seed": 7,
        }

    def test_output_paths_use_fixed_b1b_90k_names(self):
        from src.diagnose_hybrid_state_gate_knockout import _output_paths

        self.assertEqual(
            _output_paths(Path("results/hst-knockout")),
            (
                Path(
                    "results/hst-knockout/"
                    "hybrid_state_gate_knockout_b1b_90k_raw.csv"
                ),
                Path(
                    "results/hst-knockout/"
                    "hybrid_state_gate_knockout_b1b_90k_paired.csv"
                ),
                Path(
                    "results/hst-knockout/"
                    "hybrid_state_gate_knockout_b1b_90k.md"
                ),
            ),
        )

    def test_writers_emit_complete_csv_provenance_and_chinese_report(self):
        from utils.hybrid_state_gate_knockout import (
            write_knockout_report,
            write_paired_csv,
            write_raw_csv,
        )

        metadata = self._metadata()
        raw_rows = [
            {**row, **metadata}
            for row in PairedStatisticTests._raw_rows()
        ]
        paired_rows = [
            {**row, **metadata}
            for row in build_paired_rows(raw_rows)
        ]
        summary_rows = summarize_paired_rows(
            paired_rows,
            bootstrap_samples=metadata["bootstrap_samples"],
            bootstrap_seed=metadata["bootstrap_seed"],
        )
        verdict = dynamic_gate_verdict(summary_rows)

        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.csv"
            paired_path = Path(directory) / "paired.csv"
            report_path = Path(directory) / "report.md"
            write_raw_csv(raw_path, raw_rows)
            write_paired_csv(paired_path, paired_rows)
            write_knockout_report(
                report_path,
                raw_rows,
                summary_rows,
                metadata,
                (0.039516, 0.015344, -0.013035, -0.001890),
                verdict,
            )

            with raw_path.open(newline="", encoding="utf-8") as handle:
                raw_csv = list(csv.DictReader(handle))
            with paired_path.open(newline="", encoding="utf-8") as handle:
                paired_csv = list(csv.DictReader(handle))

            self.assertEqual(len(raw_csv), 205)
            self.assertEqual(len(paired_csv), 164)
            for csv_rows in (raw_csv, paired_csv):
                for row in csv_rows:
                    self.assertEqual(row["checkpoint"], metadata["checkpoint"])
                    self.assertEqual(row["config"], metadata["config"])
                    self.assertEqual(row["seed"], "0")
                    self.assertEqual(row["sample_scope"], metadata["sample_scope"])
            self.assertEqual(raw_csv[0]["condition"], "normal")
            self.assertEqual(paired_csv[0]["condition"], "all_off")

            report = report_path.read_text(encoding="utf-8")
            for token in (
                "# B1b HST Gate Knockout 诊断",
                metadata["checkpoint"],
                metadata["config"],
                metadata["sample_scope"],
                "+0.039516",
                "-0.013035",
                "205 / 205",
                "164 / 164",
                "overall",
                "elastic",
                "plasticine",
                "sand",
                "low_E",
                "high_E",
                "负值表示改善",
                "close",
            ):
                self.assertIn(token, report)

    def test_writers_reject_missing_provenance(self):
        from utils.hybrid_state_gate_knockout import (
            write_knockout_report,
            write_paired_csv,
            write_raw_csv,
        )

        metadata = self._metadata()
        raw_rows = [
            {**row, **metadata}
            for row in PairedStatisticTests._raw_rows()
        ]
        paired_rows = [
            {**row, **metadata}
            for row in build_paired_rows(raw_rows)
        ]
        del raw_rows[0]["sample_scope"]
        del paired_rows[0]["config"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "sample_scope"):
                write_raw_csv(root / "raw.csv", raw_rows)
            with self.assertRaisesRegex(ValueError, "config"):
                write_paired_csv(root / "paired.csv", paired_rows)
            incomplete_metadata = dict(metadata)
            del incomplete_metadata["seed"]
            with self.assertRaisesRegex(ValueError, "seed"):
                write_knockout_report(
                    root / "report.md",
                    PairedStatisticTests._raw_rows(),
                    [],
                    incomplete_metadata,
                    (0.1, 0.2, 0.3, 0.4),
                    {"proceed_dynamic_gate": False, "reasons": ()},
                )

    def test_paired_writer_rejects_missing_relative_change_field(self):
        from utils.hybrid_state_gate_knockout import write_paired_csv

        metadata = self._metadata()
        paired_rows = [
            {**row, **metadata}
            for row in build_paired_rows(PairedStatisticTests._raw_rows())
        ]
        del paired_rows[0]["relative_change_pct_fde"]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "relative_change_pct_fde"
            ):
                write_paired_csv(Path(directory) / "paired.csv", paired_rows)


def _diagnostic_args(dataset_path="mm3_data/mm3_test"):
    return SimpleNamespace(
        pc_size=2048,
        eval_batch_size=1,
        dataloader_num_workers=4,
        seed=0,
        use_diffusion=False,
        num_inference_steps=1,
        floor_projection=False,
        input_frames=5,
        output_frames=1,
        pred_offset=True,
        model_type="dit_st",
        model_config=SimpleNamespace(
            n_layers=8,
            latent_dim=256,
            frame_cond=True,
            point_embed=True,
            mask_cond=True,
            pred_offset=True,
            num_neighbors=-1,
            floor_cond=True,
            max_num_forces=1,
            force_as_token=False,
            force_as_latent=False,
            gravity_emb=True,
            coeff_cond=False,
            num_mat=4,
            class_token=True,
            transformer_block="SpatialTemporalTransformerBlockv11a",
            hybrid_state_dim=64,
            hybrid_state_heads=4,
            hybrid_state_interval=2,
            contact_particle_cond=True,
            contact_feature_sigma=0.04,
        ),
        train_dataset=SimpleNamespace(
            category="hf-objaverse-v1",
            dataset_path=dataset_path,
            dataset_list="DATASET_ITEM_LIST",
            has_gravity=True,
            max_num_forces=1,
            norm_fac=5,
            stage="deform",
            mode="diff",
            pc_size=2048,
            repeat=1,
            seed=0,
            n_sample_pro_model=300,
            n_frames_interval=1,
            n_training_frames=24,
            batch_size=20,
            overfit=False,
            input_frames=5,
            output_frames=1,
        ),
    )


class GateKnockoutCliTests(unittest.TestCase):
    def test_parser_fixes_required_paths_and_bootstrap_protocol(self):
        from src.diagnose_hybrid_state_gate_knockout import build_parser

        parser = build_parser()
        parsed = parser.parse_args(
            [
                "--config",
                "src/configs/eval.yaml",
                "--checkpoint",
                "outputs/run/checkpoint-90000/model.safetensors",
                "--output-dir",
                "results/knockout",
            ]
        )
        self.assertEqual(parsed.bootstrap_samples, 10000)
        self.assertEqual(parsed.bootstrap_seed, 0)
        for option, value in (
            ("--bootstrap-samples", "0"),
            ("--bootstrap-samples", "-1"),
            ("--bootstrap-seed", "-1"),
        ):
            with self.subTest(option=option, value=value):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(
                            [
                                "--config",
                                "src/configs/eval.yaml",
                                "--checkpoint",
                                "outputs/run/checkpoint-90000/model.safetensors",
                                "--output-dir",
                                "results/knockout",
                                option,
                                value,
                            ]
                        )

    def test_main_merges_testing_config_and_overrides_resume(self):
        import src.diagnose_hybrid_state_gate_knockout as diagnostic

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "eval_mm3_v11a_contact_cond_8L_45k.yaml"
        )
        checkpoint = Path(
            "outputs/mm3_v11a_contact_cond_8L/"
            "checkpoint-90000/model.safetensors"
        )
        output_dir = Path("results/hst-knockout")
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "diagnose_hybrid_state_gate_knockout.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir),
                    "--bootstrap-samples",
                    "123",
                    "--bootstrap-seed",
                    "9",
                ],
            ),
            mock.patch.object(
                diagnostic, "run_gate_knockout_diagnostics"
            ) as run,
        ):
            diagnostic.main()

        args, passed_checkpoint, passed_output_dir, samples, seed = run.call_args.args
        self.assertEqual(args.resume, str(checkpoint))
        self.assertEqual(passed_checkpoint, checkpoint)
        self.assertEqual(passed_output_dir, output_dir)
        self.assertEqual(samples, 123)
        self.assertEqual(seed, 9)
        self.assertEqual(run.call_args.kwargs["config_path"], config_path)

    def test_run_loads_once_resets_each_condition_and_restores_checkpoint_gates(self):
        import src.diagnose_hybrid_state_gate_knockout as diagnostic

        manifest = diagnostic.load_frozen_test_manifest()
        model_names = [name for names in manifest.values() for name in names]
        records = [
            SimpleNamespace(
                model=name,
                mat_type=0 if index < 13 else 1 if index < 27 else 2,
                log10_e=3.0 + index / 100.0,
                nu=0.2,
            )
            for index, name in enumerate(model_names)
        ]

        class FakeDataset:
            def __init__(self, split, config):
                self.split_lst_save = list(model_names)
                self.models = [
                    {"model": model_name, "start_idx": start_idx}
                    for model_name in model_names
                    for start_idx in (0, 5, 10, 15)
                ]

        class FakeModel:
            def __init__(self):
                self.dit = SimpleNamespace(
                    hybrid_state_exchange=SimpleNamespace(
                        feedback_gates=torch.nn.Parameter(
                            torch.tensor([0.04, 0.015, -0.013, -0.002])
                        )
                    )
                )
                self.load_calls = 0

            def to(self, device):
                self.device = device
                return self

            def load_state_dict(self, checkpoint, strict):
                self.load_calls += 1
                self.strict = strict

            def eval(self):
                return self

            def requires_grad_(self, enabled):
                self.requires_grad = enabled
                return self

        fake_model = FakeModel()
        original_gates = fake_model.dit.hybrid_state_exchange.feedback_gates.detach().clone()
        dataset_module = ModuleType("dataset.traj_dataset")
        dataset_module.TrajDataset = FakeDataset
        dataset_package = ModuleType("dataset")
        dataset_package.__path__ = []
        model_module = ModuleType("model.spacetime")
        model_module.MDM_ST = mock.Mock(return_value=fake_model)
        model_package = ModuleType("model")
        model_package.__path__ = []
        pipeline_module = ModuleType("pipeline_traj")
        pipeline_module.TrajPipeline = mock.Mock(return_value=object())
        safetensors_package = ModuleType("safetensors")
        safetensors_package.__path__ = []
        safetensors_torch_module = ModuleType("safetensors.torch")
        safetensors_torch_module.load_file = mock.Mock(return_value={"weight": 1})
        safetensors_package.torch = safetensors_torch_module
        imported_modules = {
            "dataset": dataset_package,
            "dataset.traj_dataset": dataset_module,
            "model": model_package,
            "model.spacetime": model_module,
            "pipeline_traj": pipeline_module,
            "safetensors": safetensors_package,
            "safetensors.torch": safetensors_torch_module,
        }

        points = torch.tensor(
            [
                [-1.0, -0.5, 0.0],
                [0.0, -0.5, 0.25],
                [1.0, -0.25, 0.5],
                [-0.75, 0.5, 1.0],
                [0.25, 0.75, 1.5],
                [1.25, 0.25, 1.75],
                [-0.5, 1.25, 2.0],
                [0.75, 1.5, 2.5],
            ]
        )
        reference = points.view(1, 1, 8, 3).repeat(1, 25, 1, 1)
        reference[:, :, :, 0] += torch.arange(25).view(1, 25, 1) * 0.01
        observed = []

        def fake_rollout(pipeline, batch, args, log10_e, nu, mat_type):
            gates = (
                fake_model.dit.hybrid_state_exchange.feedback_gates
                .detach()
                .clone()
            )
            observed.append((batch["model"][0], gates))
            pred = reference.clone()
            pred[:, 5:] += gates.sum().item() * 0.01
            return pred

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = (
                root
                / "outputs"
                / "mm3_v11a_contact_cond_8L"
                / "checkpoint-90000"
                / "model.safetensors"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            dataloader = [
                (
                    {
                        "model": [name],
                        "start_idx": torch.tensor([0]),
                        "floor_height": torch.tensor([-10.0]),
                    },
                    {},
                )
                for name in reversed(model_names)
            ]
            args = _diagnostic_args()
            with (
                mock.patch.dict(sys.modules, imported_modules),
                mock.patch.object(
                    diagnostic.torch.utils.data,
                    "DataLoader",
                    return_value=dataloader,
                ) as dataloader_constructor,
                mock.patch.object(
                    diagnostic, "load_material_records", return_value=records
                ),
                mock.patch.object(
                    diagnostic, "_build_raw_reference", return_value=reference
                ),
                mock.patch.object(
                    diagnostic, "rollout_condition", side_effect=fake_rollout
                ) as rollout,
                mock.patch.object(
                    diagnostic, "reset_inference_seed"
                ) as reset_seed,
                mock.patch.object(
                    diagnostic.Path, "is_dir", return_value=True
                ),
                mock.patch.object(
                    diagnostic.torch, "autocast", return_value=nullcontext()
                ),
                mock.patch.object(
                    diagnostic.torch,
                    "compile",
                    side_effect=AssertionError("must remain eager"),
                ) as compile_model,
                redirect_stdout(io.StringIO()),
            ):
                rows = diagnostic.run_gate_knockout_diagnostics(
                    args,
                    checkpoint,
                    root / "reports",
                    10,
                    7,
                )

            self.assertEqual(model_module.MDM_ST.call_count, 1)
            self.assertEqual(safetensors_torch_module.load_file.call_count, 1)
            self.assertEqual(fake_model.load_calls, 1)
            self.assertEqual(pipeline_module.TrajPipeline.call_count, 1)
            self.assertEqual(rollout.call_count, 41 * 5)
            self.assertEqual(reset_seed.call_count, 41 * 5)
            self.assertEqual(
                reset_seed.call_args_list,
                [mock.call(args.seed, torch.device("cuda"))] * (41 * 5),
            )
            compile_model.assert_not_called()
            selected_dataset = dataloader_constructor.call_args.args[0]
            self.assertEqual(selected_dataset.indices, list(range(0, 41 * 4, 4)))

            condition_names = [name for name, _ in KNOCKOUT_CONDITIONS]
            masks = [mask for _, mask in KNOCKOUT_CONDITIONS]
            for model_index, model_name in enumerate(reversed(model_names)):
                start = model_index * len(KNOCKOUT_CONDITIONS)
                model_observations = observed[start : start + len(KNOCKOUT_CONDITIONS)]
                self.assertEqual(
                    [name for name, _ in model_observations],
                    [model_name] * len(KNOCKOUT_CONDITIONS),
                )
                for (_, gates), mask in zip(model_observations, masks):
                    self.assertTrue(
                        torch.equal(gates, original_gates * torch.tensor(mask))
                    )
            self.assertEqual(
                [row["condition"] for row in rows[:5]], condition_names
            )
            self.assertTrue(
                torch.equal(
                    fake_model.dit.hybrid_state_exchange.feedback_gates.detach(),
                    original_gates,
                )
            )
            self.assertEqual(len(rows), 205)
            self.assertTrue(all(row["seed"] == args.seed for row in rows))
            raw_path, paired_path, report_path = diagnostic._output_paths(
                root / "reports"
            )
            self.assertTrue(raw_path.is_file())
            self.assertTrue(paired_path.is_file())
            self.assertTrue(report_path.is_file())
            with raw_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 205)
            with paired_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 164)


class GateMaskTests(unittest.TestCase):
    def test_conditions_are_pre_registered_and_ordered(self):
        self.assertEqual(
            KNOCKOUT_CONDITIONS,
            (
                ("normal", (1, 1, 1, 1)),
                ("all_off", (0, 0, 0, 0)),
                ("stage0_off", (0, 1, 1, 1)),
                ("stage1_off", (1, 0, 1, 1)),
                ("stage2_off", (1, 1, 0, 1)),
            ),
        )

    def test_mask_multiplies_trained_values_and_restores_them(self):
        exchange = SimpleNamespace(
            feedback_gates=torch.nn.Parameter(
                torch.tensor([0.04, 0.015, -0.013, -0.002])
            )
        )
        original = exchange.feedback_gates.detach().clone()
        with masked_feedback_gates(exchange, (0, 1, 0, 1)) as applied:
            torch.testing.assert_close(
                exchange.feedback_gates,
                torch.tensor([0.0, 0.015, 0.0, -0.002]),
            )
            torch.testing.assert_close(applied, exchange.feedback_gates)
        self.assertTrue(torch.equal(exchange.feedback_gates.detach(), original))

    def test_mask_rejects_invalid_length_and_values(self):
        exchange = SimpleNamespace(
            feedback_gates=torch.nn.Parameter(torch.ones(4))
        )
        for invalid_mask in ((0, 1, 0), (0, 1, 2, 1)):
            with self.subTest(invalid_mask=invalid_mask):
                with self.assertRaisesRegex(ValueError, "gate mask"):
                    with masked_feedback_gates(exchange, invalid_mask):
                        pass

    def test_gate_values_must_be_four_finite_values(self):
        for gates in (torch.ones(3), torch.tensor([1.0, 2.0, 3.0, float("nan")])):
            with self.subTest(gates=gates):
                exchange = SimpleNamespace(feedback_gates=torch.nn.Parameter(gates))
                with self.assertRaisesRegex(ValueError, "feedback_gates"):
                    with masked_feedback_gates(exchange, (1, 1, 1, 1)):
                        pass

    def test_mask_restores_original_gates_when_rollout_raises(self):
        exchange = SimpleNamespace(
            feedback_gates=torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        )
        with self.assertRaisesRegex(RuntimeError, "rollout failed"):
            with masked_feedback_gates(exchange, (0, 1, 1, 1)):
                raise RuntimeError("rollout failed")
        torch.testing.assert_close(
            exchange.feedback_gates,
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

    @mock.patch("utils.hybrid_state_gate_knockout.torch.cuda.manual_seed_all")
    @mock.patch("utils.hybrid_state_gate_knockout.torch.manual_seed")
    def test_reset_seed_resets_cpu_and_cuda_for_each_condition(
        self, cpu_seed, cuda_seed
    ):
        from utils.hybrid_state_gate_knockout import reset_inference_seed

        reset_inference_seed(7, torch.device("cuda"))
        cpu_seed.assert_called_once_with(7)
        cuda_seed.assert_called_once_with(7)

    def test_reset_seed_rejects_negative_non_integer_seed(self):
        from utils.hybrid_state_gate_knockout import reset_inference_seed

        for invalid_seed in (-1, True, 1.5):
            with self.subTest(invalid_seed=invalid_seed):
                with self.assertRaisesRegex(ValueError, "seed"):
                    reset_inference_seed(invalid_seed, torch.device("cpu"))


class TrajectoryMetricTests(unittest.TestCase):
    def test_trajectory_metrics_match_gm_fde_and_frame24_procrustes(self):
        points = torch.tensor(
            [
                [-1.0, -0.5, 0.0],
                [0.0, -0.5, 0.25],
                [1.0, -0.25, 0.5],
                [-0.75, 0.5, 1.0],
                [0.25, 0.75, 1.5],
                [1.25, 0.25, 1.75],
                [-0.5, 1.25, 2.0],
                [0.75, 1.5, 2.5],
            ]
        )
        gt = points.unsqueeze(0).repeat(25, 1, 1)
        gt[:, :, 0] += torch.arange(25, dtype=torch.float32).view(-1, 1) * 0.02
        gt[:, :, 2] += torch.arange(25, dtype=torch.float32).view(-1, 1) * 0.03

        angle = torch.tensor(0.35)
        rotation = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle), 0.0],
                [torch.sin(angle), torch.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        pred = gt.clone()
        pred[5:] = 1.15 * (gt[5:] @ rotation.T) + torch.tensor([0.4, -0.2, 0.3])
        pred[5:] += torch.arange(20, dtype=torch.float32).view(-1, 1, 1) * 0.01

        metrics = trajectory_knockout_metrics(
            pred, gt, input_frames=5, floor_height=-10.0
        )
        base = per_window_metrics(pred.float(), gt.float(), input_frames=5, k=7)
        frame_mse = (pred[5:] - gt[5:]).square().mean((1, 2))
        expected_gm = torch.exp(torch.log(frame_mse.clamp_min(1e-30)).mean()).item()
        expected_centroid, _, _, expected_shape = base["proc"][24]

        self.assertAlmostEqual(metrics["gm_mse"], expected_gm)
        self.assertAlmostEqual(metrics["fde"], base["fde"])
        self.assertAlmostEqual(metrics["f24_centroid_error"], expected_centroid)
        self.assertAlmostEqual(metrics["f24_shape_residual_mse"], expected_shape)

    def test_trajectory_metrics_exclude_history_and_use_absolute_frame_segments(self):
        gt = torch.zeros(25, 8, 3)
        gt[:, :, 0] = torch.arange(8, dtype=torch.float32)
        pred = gt.clone()
        pred[5:11] += 1.0
        pred[11:18] += 2.0
        pred[18:25] += 3.0
        pred[24, :2, 1] = -0.5

        metrics = trajectory_knockout_metrics(
            pred, gt, input_frames=5, floor_height=0.0
        )

        self.assertAlmostEqual(
            metrics["short_mse"], torch.mean((pred[5:11] - gt[5:11]) ** 2).item()
        )
        self.assertAlmostEqual(
            metrics["mid_mse"], torch.mean((pred[11:18] - gt[11:18]) ** 2).item()
        )
        self.assertAlmostEqual(
            metrics["long_mse"], torch.mean((pred[18:25] - gt[18:25]) ** 2).item()
        )
        self.assertAlmostEqual(
            metrics["full_rollout_mse"], torch.mean((pred[5:] - gt[5:]) ** 2).item()
        )
        self.assertAlmostEqual(metrics["penetration_rate"], 2.0 / (20 * 8))
        self.assertAlmostEqual(metrics["penetration_depth"], 1.0 / (20 * 8))

    def test_trajectory_metrics_reject_invalid_inputs(self):
        pred = torch.zeros(25, 8, 3)
        gt = torch.zeros_like(pred)
        cases = (
            (torch.zeros(24, 8, 3), gt, 5, 0.0),
            (pred, torch.zeros(25, 7, 3), 5, 0.0),
            (pred, gt, 4, 0.0),
            (pred, gt, 5, float("nan")),
            (pred, gt, 5, torch.tensor([0.0, 1.0])),
        )
        for invalid_pred, invalid_gt, input_frames, floor in cases:
            with self.subTest(input_frames=input_frames, floor=floor):
                with self.assertRaises(ValueError):
                    trajectory_knockout_metrics(
                        invalid_pred, invalid_gt, input_frames, floor
                    )

    def test_trajectory_metrics_reject_non_finite_trajectories(self):
        pred = torch.zeros(25, 8, 3)
        gt = torch.zeros_like(pred)
        pred[5, 0, 0] = float("inf")

        with self.assertRaisesRegex(ValueError, "finite"):
            trajectory_knockout_metrics(pred, gt, input_frames=5, floor_height=0.0)


class PairedStatisticTests(unittest.TestCase):
    @staticmethod
    def _raw_rows(knockout_factor=0.9):
        rows = []
        material_counts = ((0, 13), (1, 14), (2, 14))
        model_index = 0
        for mat_type, count in material_counts:
            for material_index in range(count):
                metadata = {
                    "model": f"model-{model_index:02d}",
                    "mat_type": mat_type,
                    "log10_e": float(2 + material_index),
                    "nu": 0.3,
                }
                normal_metrics = {
                    metric: 1.0 + metric_index / 100.0 + model_index / 1000.0
                    for metric_index, metric in enumerate(KNOCKOUT_METRICS)
                }
                for condition, _ in KNOCKOUT_CONDITIONS:
                    factor = 1.0 if condition == "normal" else knockout_factor
                    rows.append(
                        {
                            **metadata,
                            "condition": condition,
                            **{
                                metric: value * factor
                                for metric, value in normal_metrics.items()
                            },
                        }
                    )
                model_index += 1
        return rows

    def test_build_paired_rows_is_complete_and_uses_knockout_minus_normal(self):
        raw_rows = self._raw_rows()

        paired = build_paired_rows(raw_rows)

        self.assertEqual(len(paired), 41 * 4)
        row = paired[0]
        self.assertLess(row["delta_full_rollout_mse"], 0.0)
        self.assertAlmostEqual(row["relative_change_pct_full_rollout_mse"], -10.0)

    def test_validate_raw_rows_rejects_incomplete_or_inconsistent_protocol(self):
        cases = []

        missing_model = self._raw_rows()[:-5]
        cases.append((missing_model, "205"))

        duplicate_condition = self._raw_rows()
        duplicate_condition[1]["condition"] = "normal"
        cases.append((duplicate_condition, "condition"))

        invalid_condition = self._raw_rows()
        invalid_condition[1]["condition"] = "stage3_off"
        cases.append((invalid_condition, "condition"))

        changed_metadata = self._raw_rows()
        changed_metadata[1]["nu"] = 0.4
        cases.append((changed_metadata, "metadata"))

        nonfinite_metric = self._raw_rows()
        nonfinite_metric[0]["fde"] = float("nan")
        cases.append((nonfinite_metric, "finite"))

        negative_metric = self._raw_rows()
        negative_metric[0]["fde"] = -0.1
        cases.append((negative_metric, "non-negative"))

        for rows, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_raw_rows(rows)

    def test_build_paired_rows_keeps_zero_baseline_percentage_well_defined(self):
        rows = self._raw_rows()
        for row in rows:
            if row["condition"] == "normal":
                row["penetration_rate"] = 0.0
                row["penetration_depth"] = 0.0
            elif row["model"] == "model-00":
                row["penetration_rate"] = 0.1
                row["penetration_depth"] = 0.2
            else:
                row["penetration_rate"] = 0.0
                row["penetration_depth"] = 0.0

        paired = build_paired_rows(rows)
        zero_to_zero = next(
            row
            for row in paired
            if row["model"] == "model-01" and row["condition"] == "stage0_off"
        )
        zero_to_positive = next(
            row
            for row in paired
            if row["model"] == "model-00" and row["condition"] == "stage0_off"
        )

        self.assertEqual(zero_to_zero["relative_change_pct_penetration_rate"], 0.0)
        self.assertIsNone(
            zero_to_positive["relative_change_pct_penetration_rate"]
        )

    def test_summary_groups_materials_and_stratifies_e_within_material(self):
        rows = self._raw_rows()
        for row in rows:
            if row["mat_type"] == 1:
                row["log10_e"] = float(int(row["model"].split("-")[1]) % 14 // 2)
        paired = build_paired_rows(rows)

        summary = summarize_paired_rows(
            paired, bootstrap_samples=200, bootstrap_seed=11
        )

        groups = {(row["group"], row["condition"], row["metric"]) for row in summary}
        self.assertIn(("plasticine", "stage0_off", "fde"), groups)
        self.assertIn(("plasticine_low_E", "stage0_off", "fde"), groups)
        self.assertIn(("plasticine_high_E", "stage0_off", "fde"), groups)
        low_e = next(
            row
            for row in summary
            if row["group"] == "plasticine_low_E"
            and row["condition"] == "stage0_off"
            and row["metric"] == "fde"
        )
        high_e = next(
            row
            for row in summary
            if row["group"] == "plasticine_high_E"
            and row["condition"] == "stage0_off"
            and row["metric"] == "fde"
        )
        self.assertEqual(low_e["n_models"], 8)
        self.assertEqual(high_e["n_models"], 6)
        self.assertEqual(
            set(low_e),
            {
                "group",
                "condition",
                "metric",
                "n_models",
                "normal_mean",
                "knockout_mean",
                "mean_delta",
                "median_delta",
                "relative_change_pct",
                "improved_count",
                "degraded_count",
                "ci_low",
                "ci_high",
            },
        )

    def test_delta_bootstrap_handles_zero_normal_without_relative_fabrication(self):
        degraded = paired_delta_summary(
            normal=np.array([0.0, 0.0]),
            knockout=np.array([0.0, 0.4]),
            samples=200,
            seed=11,
        )
        unchanged = paired_delta_summary(
            normal=np.zeros(3),
            knockout=np.zeros(3),
            samples=200,
            seed=11,
        )

        self.assertIsNone(degraded["relative_change_pct"])
        self.assertAlmostEqual(degraded["mean_delta"], 0.2)
        self.assertTrue(np.isfinite(degraded["ci_low"]))
        self.assertTrue(np.isfinite(degraded["ci_high"]))
        self.assertEqual(unchanged["relative_change_pct"], 0.0)

    def test_delta_summary_counts_only_strictly_positive_degradations(self):
        normal = np.ones(14)
        knockout = np.concatenate((np.full(7, 1.1), np.ones(7)))

        stats = paired_delta_summary(normal, knockout, samples=200, seed=11)

        self.assertEqual(stats["improved_count"], 0)
        self.assertEqual(stats["degraded_count"], 7)

    def test_summary_retains_strict_degraded_count(self):
        rows = self._raw_rows()
        sand_models = sorted(
            {row["model"] for row in rows if row["mat_type"] == 2}
        )
        normal_fde = {
            row["model"]: row["fde"]
            for row in rows
            if row["condition"] == "normal" and row["mat_type"] == 2
        }
        for index, model in enumerate(sand_models):
            row = next(
                row
                for row in rows
                if row["model"] == model and row["condition"] == "stage0_off"
            )
            row["fde"] = normal_fde[model] * (1.1 if index < 7 else 1.0)

        summary = summarize_paired_rows(
            build_paired_rows(rows), bootstrap_samples=200, bootstrap_seed=11
        )
        sand_fde = next(
            row
            for row in summary
            if row["group"] == "sand"
            and row["condition"] == "stage0_off"
            and row["metric"] == "fde"
        )

        self.assertEqual(sand_fde["degraded_count"], 7)

    def test_summary_retains_zero_baseline_penetration_delta_and_ci(self):
        rows = self._raw_rows()
        for row in rows:
            row["penetration_rate"] = 0.0
            row["penetration_depth"] = 0.0
        rows[1]["penetration_rate"] = 0.2
        rows[1]["penetration_depth"] = 0.4
        paired = build_paired_rows(rows)

        summary = summarize_paired_rows(
            paired, bootstrap_samples=200, bootstrap_seed=11
        )
        rate = next(
            row
            for row in summary
            if row["group"] == "overall"
            and row["condition"] == "all_off"
            and row["metric"] == "penetration_rate"
        )

        self.assertIsNone(rate["relative_change_pct"])
        self.assertGreater(rate["mean_delta"], 0.0)
        self.assertTrue(np.isfinite(rate["ci_low"]))
        self.assertTrue(np.isfinite(rate["ci_high"]))

    @staticmethod
    def _verdict_row(
        group,
        condition,
        metric,
        relative_change_pct=0.0,
        improved_count=0,
        degraded_count=0,
        median_delta=0.0,
        n_models=14,
        normal_mean=1.0,
        knockout_mean=1.0,
    ):
        return {
            "group": group,
            "condition": condition,
            "metric": metric,
            "n_models": n_models,
            "normal_mean": normal_mean,
            "knockout_mean": knockout_mean,
            "mean_delta": median_delta,
            "median_delta": median_delta,
            "relative_change_pct": relative_change_pct,
            "improved_count": improved_count,
            "degraded_count": degraded_count,
            "ci_low": median_delta,
            "ci_high": median_delta,
        }

    @classmethod
    def _passing_verdict_summary(cls):
        rows = []
        condition = "stage0_off"
        for metric in ("long_mse", "fde"):
            rows.append(
                cls._verdict_row(
                    "plasticine",
                    condition,
                    metric,
                    relative_change_pct=-6.0,
                    improved_count=8,
                    median_delta=-0.1,
                )
            )
        rows.append(
            cls._verdict_row(
                "plasticine", condition, "f24_centroid_error"
            )
        )
        rows.append(
            cls._verdict_row(
                "sand",
                condition,
                "fde",
                relative_change_pct=6.0,
                improved_count=6,
                degraded_count=8,
                median_delta=0.1,
            )
        )
        for metric in ("full_rollout_mse", "fde"):
            rows.append(
                cls._verdict_row(
                    "overall",
                    condition,
                    metric,
                    relative_change_pct=5.0,
                    n_models=41,
                )
            )
        for metric in ("penetration_rate", "penetration_depth"):
            rows.append(
                cls._verdict_row(
                    "overall",
                    condition,
                    metric,
                    normal_mean=0.0,
                    knockout_mean=0.0,
                    n_models=41,
                )
            )
        return rows

    def test_dynamic_gate_verdict_requires_pre_registered_paired_evidence(self):
        verdict = dynamic_gate_verdict(self._passing_verdict_summary())

        self.assertTrue(verdict["proceed_dynamic_gate"])
        self.assertEqual(verdict["qualifying_stage"], "stage0_off")
        self.assertEqual(verdict["plasticine_metrics"], ("long_mse", "fde"))
        self.assertEqual(verdict["sand_opposite_metrics"], ("fde",))

    def test_dynamic_gate_verdict_fails_for_insufficient_wins_or_unregistered_stage(self):
        insufficient_wins = self._passing_verdict_summary()
        for row in insufficient_wins:
            if row["group"] == "plasticine" and row["metric"] == "fde":
                row["improved_count"] = 7
        self.assertFalse(dynamic_gate_verdict(insufficient_wins)["proceed_dynamic_gate"])

        stage1_only = self._passing_verdict_summary()
        for row in stage1_only:
            row["condition"] = "stage1_off"
        verdict = dynamic_gate_verdict(stage1_only)
        self.assertFalse(verdict["proceed_dynamic_gate"])
        self.assertIsNone(verdict["qualifying_stage"])

    def test_dynamic_gate_verdict_rejects_sand_zeros_as_non_degradations(self):
        summary = self._passing_verdict_summary()
        sand = next(
            row
            for row in summary
            if row["group"] == "sand" and row["metric"] == "fde"
        )
        sand.update(
            paired_delta_summary(
                np.ones(14),
                np.concatenate((np.full(7, 1.1), np.ones(7))),
                samples=200,
                seed=11,
            )
        )
        sand["degraded_count"] = 7

        self.assertEqual(sand["degraded_count"], 7)
        self.assertFalse(dynamic_gate_verdict(summary)["proceed_dynamic_gate"])


if __name__ == "__main__":
    unittest.main()
