import csv
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from types import ModuleType, SimpleNamespace
from unittest import mock
from unittest.mock import patch

import numpy as np
import torch

from model.hybrid_state import HybridStateExchange
from utils.hybrid_state_diagnostics import (
    HybridStateFeedbackRecorder,
    aggregate_feedback_rows,
    decompose_feedback,
    feedback_correlations,
    horizon_bucket,
    write_feedback_csv,
    write_feedback_report,
)
from utils.eval_metrics import per_window_metrics
from src.diagnose_hybrid_state_feedback import (
    _output_paths,
    build_parser,
    main,
    run_feedback_diagnostics,
    select_start_zero_indices,
    trajectory_diagnostic_fields,
    validate_diagnostic_config,
)


def _diagnostic_args(dataset_path="mm3_data/mm3_test"):
    return SimpleNamespace(
        pc_size=2048,
        eval_batch_size=1,
        dataloader_num_workers=0,
        seed=0,
        use_diffusion=False,
        num_inference_steps=1,
        input_frames=5,
        output_frames=1,
        model_config=SimpleNamespace(
            transformer_block="SpatialTemporalTransformerBlockv11a",
            contact_particle_cond=True,
        ),
        train_dataset=SimpleNamespace(dataset_path=dataset_path),
    )


class FeedbackDiagnosticCliTests(unittest.TestCase):
    def test_main_uses_contact_config_overrides_resume_and_passes_metadata_explicitly(self):
        import src.diagnose_hybrid_state_feedback as diagnostic

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "eval_mm3_v11a_contact_cond_8L_45k.yaml"
        )
        checkpoint = Path(
            "outputs/mm3_v11a_contact_cond_8L/checkpoint-90000/model.safetensors"
        )
        output_dir = Path("results/hst-feedback")
        with (
            patch.object(
                sys,
                "argv",
                [
                    "diagnose_hybrid_state_feedback.py",
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output_dir),
                ],
            ),
            patch.object(diagnostic, "run_feedback_diagnostics") as run,
        ):
            main()

        args, passed_checkpoint, passed_output_dir = run.call_args.args
        self.assertEqual(args.resume, str(checkpoint))
        self.assertNotIn("diagnostic_config_path", args)
        self.assertEqual(passed_checkpoint, checkpoint)
        self.assertEqual(passed_output_dir, output_dir)
        self.assertEqual(run.call_args.kwargs["config_path"], config_path)

    def test_select_start_zero_indices_keeps_one_window_per_model(self):
        dataset = SimpleNamespace(
            models=[
                {"model": "elastic.h5", "start_idx": 0},
                {"model": "elastic.h5", "start_idx": 5},
                {"model": "elastic.h5", "start_idx": 10},
                {"model": "plasticine.h5", "start_idx": 0},
                {"model": "plasticine.h5", "start_idx": 5},
                {"model": "plasticine.h5", "start_idx": 10},
                {"model": "sand.h5", "start_idx": 0},
                {"model": "sand.h5", "start_idx": 5},
                {"model": "sand.h5", "start_idx": 10},
            ],
            split_lst_save=["elastic.h5", "plasticine.h5", "sand.h5"],
        )

        self.assertEqual(select_start_zero_indices(dataset), [0, 3, 6])

    def test_validate_diagnostic_config_rejects_non_b1b_runtime_shapes(self):
        checkpoint = Path("outputs/mm3_v11a_mc_hst_8L/checkpoint-90000/model.safetensors")
        cases = (
            ("transformer_block", "SpatialTemporalTransformerBlock", "transformer_block"),
            ("contact_particle_cond", False, "contact_particle_cond"),
            ("input_frames", 4, "input_frames"),
            ("output_frames", 2, "output_frames"),
            ("use_diffusion", True, "use_diffusion"),
            ("eval_batch_size", 2, "eval_batch_size"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                args = _diagnostic_args()
                target = args.model_config if hasattr(args.model_config, field) else args
                setattr(target, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_diagnostic_config(args, checkpoint)

        with self.assertRaisesRegex(ValueError, "checkpoint-90000"):
            validate_diagnostic_config(
                _diagnostic_args(),
                Path("outputs/mm3_v11a_mc_hst_8L/checkpoint-45000/model.safetensors"),
            )

    def test_trajectory_diagnostic_fields_uses_full_mse_fde_and_frame24_proc(self):
        gt = torch.arange(27, dtype=torch.float32).view(1, 9, 3).repeat(25, 1, 1)
        pred = gt.clone()
        pred[5:] += 2.0
        pred[24, :, 0] += torch.arange(9, dtype=torch.float32)

        fields = trajectory_diagnostic_fields(pred, gt, input_frames=5)

        self.assertAlmostEqual(
            fields["full_rollout_mse"],
            torch.mean((pred - gt).square()).item(),
        )
        self.assertAlmostEqual(
            fields["fde"],
            (pred[24] - gt[24]).norm(dim=-1).mean().item(),
        )
        expected_proc = per_window_metrics(pred, gt, input_frames=5)["proc"][24]
        self.assertAlmostEqual(fields["f24_centroid_error"], expected_proc[0])
        self.assertAlmostEqual(fields["f24_shape_residual_mse"], expected_proc[3])

    def test_cli_parser_and_output_paths_use_fixed_b1b_90k_names(self):
        parsed = build_parser().parse_args(
            [
                "--config",
                "configs/eval_mm3_v11a_contact_cond_8L_45k.yaml",
                "--checkpoint",
                "outputs/mm3_v11a_mc_hst_8L/checkpoint-90000/model.safetensors",
                "--output-dir",
                "results/hst",
            ]
        )
        self.assertEqual(
            parsed.config,
            "configs/eval_mm3_v11a_contact_cond_8L_45k.yaml",
        )
        self.assertEqual(
            _output_paths(Path("results/hst")),
            (
                Path("results/hst/hybrid_state_feedback_b1b_90k.csv"),
                Path("results/hst/hybrid_state_feedback_b1b_90k.md"),
            ),
        )

    def test_run_feedback_diagnostics_writes_3280_rows_without_compile(self):
        import src.diagnose_hybrid_state_feedback as diagnostic

        model_names = [
            *(f"elastic_{index:02d}.h5" for index in range(13)),
            *(f"plasticine_{index:02d}.h5" for index in range(14)),
            *(f"sand_{index:02d}.h5" for index in range(14)),
        ]
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
            def to(self, device):
                self.device = device
                self.dit = SimpleNamespace(hybrid_state_exchange=object())
                return self

            def load_state_dict(self, checkpoint, strict):
                self.strict = strict

            def eval(self):
                return self

            def requires_grad_(self, enabled):
                self.requires_grad = enabled
                return self

        class FakeRecorder:
            def __init__(self, exchange):
                self.exchange = exchange

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def finalize(self, expected_rollout_steps):
                self.expected_rollout_steps = expected_rollout_steps
                return [
                    {
                        "stage": stage,
                        "rollout_step": rollout_step,
                        "absolute_frame": 5 + rollout_step,
                        "gate": 0.25,
                        "feedback_rms": 1.0,
                        "global_rms": 0.5,
                        "deform_rms": 0.5,
                        "global_energy_fraction": 0.25,
                    }
                    for rollout_step in range(20)
                    for stage in range(4)
                ]

        fake_model = FakeModel()
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

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-90000" / "model.safetensors"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            args = _diagnostic_args(dataset_path=str(root))
            dataloader = [
                (
                    {
                        "model": [name],
                        "start_idx": torch.tensor([0]),
                    },
                    {},
                )
                for name in reversed(model_names)
            ]
            reference = (
                torch.arange(9, dtype=torch.float32)
                .view(1, 1, 9, 1)
                .repeat(1, 25, 1, 3)
            )
            expected_pred = reference.clone()
            with (
                mock.patch.dict(sys.modules, imported_modules),
                mock.patch.object(
                    diagnostic.torch.utils.data,
                    "DataLoader",
                    return_value=dataloader,
                ) as dataloader_constructor,
                mock.patch.object(diagnostic, "load_material_records", return_value=records),
                mock.patch.object(
                    diagnostic,
                    "_build_raw_reference",
                    return_value=reference,
                ),
                mock.patch.object(diagnostic, "rollout_condition", return_value=expected_pred) as rollout,
                mock.patch.object(diagnostic, "HybridStateFeedbackRecorder", FakeRecorder),
                mock.patch.object(diagnostic.torch, "compile", side_effect=AssertionError("must be eager")) as compile,
                io.StringIO() as stdout,
                redirect_stdout(stdout),
            ):
                rows = run_feedback_diagnostics(
                    args,
                    checkpoint,
                    root / "reports",
                    config_path=Path("configs/eval_mm3_v11a_contact_cond_8L_45k.yaml"),
                )
                completion_output = stdout.getvalue()

            csv_path, markdown_path = _output_paths(root / "reports")
            self.assertEqual(len(rows), 3280)
            self.assertEqual(rollout.call_count, 41)
            compile.assert_not_called()
            selected_dataset = dataloader_constructor.call_args.args[0]
            self.assertEqual(selected_dataset.indices, list(range(0, 41 * 4, 4)))
            self.assertTrue(csv_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertIn(str(checkpoint.resolve()), markdown_path.read_text(encoding="utf-8"))
            self.assertIn("models=41", completion_output)
            self.assertIn("rows=3280", completion_output)


class FeedbackDecompositionTests(unittest.TestCase):
    def test_decompose_feedback_separates_particle_mean_and_centered_energy(self):
        feedback = torch.tensor([[[1.0, 3.0], [3.0, 1.0]]])
        stats = decompose_feedback(feedback, gate=torch.tensor(0.5))

        expected_delta = feedback * 0.5
        expected_global = expected_delta.mean(dim=1)
        expected_centered = expected_delta - expected_global[:, None]
        torch.testing.assert_close(
            stats["feedback_rms"],
            expected_delta.square().mean((1, 2)).sqrt(),
        )
        torch.testing.assert_close(
            stats["global_rms"],
            expected_global.square().mean(1).sqrt(),
        )
        torch.testing.assert_close(
            stats["deform_rms"],
            expected_centered.square().mean((1, 2)).sqrt(),
        )
        torch.testing.assert_close(
            stats["feedback_energy"],
            stats["global_energy"] + stats["deform_energy"],
        )

    def test_decompose_feedback_requires_bnc_shape(self):
        with self.assertRaisesRegex(ValueError, "shape \(B,N,C\)"):
            decompose_feedback(torch.ones(2, 3), gate=1.0)

    def test_decompose_feedback_rejects_nonfinite_gate(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            decompose_feedback(torch.ones(1, 2, 3), gate=float("nan"))

    def test_decompose_feedback_requires_scalar_gate(self):
        with self.assertRaisesRegex(ValueError, "scalar"):
            decompose_feedback(torch.ones(1, 2, 3), gate=torch.ones(2))

    def test_decompose_feedback_returns_zero_fraction_for_zero_energy(self):
        stats = decompose_feedback(torch.zeros(2, 3, 4), gate=2.0)

        torch.testing.assert_close(
            stats["global_energy_fraction"],
            torch.zeros(2),
        )

    def test_decompose_feedback_zeroes_all_nonfinite_feedback_values(self):
        feedback = torch.tensor([[[float("nan"), float("inf"), float("-inf")]]])

        stats = decompose_feedback(feedback, gate=1.0)

        for key in stats:
            torch.testing.assert_close(stats[key], torch.zeros(1))


class HybridStateFeedbackRecorderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.exchange = HybridStateExchange(
            particle_dim=8,
            state_dim=8,
            num_heads=2,
            num_stages=4,
        )
        with torch.no_grad():
            self.exchange.feedback_gates.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        self.hidden = torch.randn(1, 7, 3, 8)
        self.explicit = torch.randn(1, 5, 18)
        self.material = torch.randn(1, 2)

    def _forward(self, stage_index, hidden=None, batch_size=None):
        hidden = self.hidden if hidden is None else hidden
        if batch_size is not None:
            hidden = hidden[:1].expand(batch_size, -1, -1, -1).clone()
        return self.exchange(
            hidden_states=hidden,
            state_tokens=None,
            explicit_frame_state=self.explicit[: hidden.shape[0]],
            material_values=self.material[: hidden.shape[0]],
            history_start=1,
            prediction_index=6,
            stage_index=stage_index,
        )

    def test_records_one_complete_rollout_in_stage_order(self):
        hidden = self.hidden
        with HybridStateFeedbackRecorder(self.exchange) as recorder:
            for stage in range(4):
                state, hidden = self._forward(stage, hidden=hidden)

        records = recorder.finalize(expected_rollout_steps=1)
        self.assertEqual([row["stage"] for row in records], [0, 1, 2, 3])
        self.assertEqual([row["rollout_step"] for row in records], [0, 0, 0, 0])
        self.assertEqual([row["absolute_frame"] for row in records], [5, 5, 5, 5])

    def test_records_multiple_rollouts_with_stage_order(self):
        hidden = self.hidden
        with HybridStateFeedbackRecorder(self.exchange) as recorder:
            for _ in range(2):
                for stage in range(4):
                    state, hidden = self._forward(stage, hidden=hidden)

        records = recorder.finalize(expected_rollout_steps=2)
        self.assertEqual(
            [row["stage"] for row in records],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )
        self.assertEqual(
            [row["rollout_step"] for row in records],
            [0, 0, 0, 0, 1, 1, 1, 1],
        )
        self.assertEqual(
            [row["absolute_frame"] for row in records],
            [5, 5, 5, 5, 6, 6, 6, 6],
        )

    def test_finalize_rejects_missing_stage(self):
        with HybridStateFeedbackRecorder(self.exchange) as recorder:
            self._forward(0)

        with self.assertRaisesRegex(ValueError, "expected 4 records"):
            recorder.finalize(expected_rollout_steps=1)

    def test_finalize_rejects_duplicate_or_out_of_order_stage(self):
        with HybridStateFeedbackRecorder(self.exchange) as recorder:
            self._forward(0)
            self._forward(0)
            self._forward(2)
            self._forward(3)

        with self.assertRaisesRegex(ValueError, "stage order"):
            recorder.finalize(expected_rollout_steps=1)

    def test_forward_outside_context_is_not_recorded(self):
        self._forward(0)

        recorder = HybridStateFeedbackRecorder(self.exchange)
        self.assertEqual(recorder.finalize(expected_rollout_steps=0), [])

    def test_reset_discards_records(self):
        with HybridStateFeedbackRecorder(self.exchange) as recorder:
            self._forward(0)
            recorder.reset()

        self.assertEqual(recorder.finalize(expected_rollout_steps=0), [])

    def test_context_exit_removes_hooks(self):
        recorder = HybridStateFeedbackRecorder(self.exchange)
        with recorder:
            self._forward(0)
        self._forward(1)

        with self.assertRaisesRegex(ValueError, "expected 4 records"):
            recorder.finalize(expected_rollout_steps=1)

    def test_recorder_rejects_batch_larger_than_one(self):
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            with HybridStateFeedbackRecorder(self.exchange):
                self._forward(0, batch_size=2)

    def test_feedback_without_captured_stage_fails(self):
        recorder = HybridStateFeedbackRecorder(self.exchange)
        with recorder:
            with self.assertRaisesRegex(RuntimeError, "without exchange stage"):
                self.exchange.feedback_attention(
                    torch.randn(1, 3, 8),
                    encoder_hidden_states=torch.randn(1, 5, 8),
                )

    def test_recorder_zeroes_nonfinite_feedback_from_real_hook(self):
        nonfinite_feedback = torch.tensor(
            [[[float("nan"), float("inf"), float("-inf"), 0.0, 0.0, 0.0, 0.0, 0.0]]]
        )
        hidden = self.hidden
        with patch.object(
            self.exchange.feedback_attention,
            "forward",
            return_value=nonfinite_feedback,
        ):
            with HybridStateFeedbackRecorder(self.exchange) as recorder:
                for stage in range(4):
                    state, hidden = self._forward(stage, hidden=hidden)

        records = recorder.finalize(expected_rollout_steps=1)
        for row in records:
            self.assertEqual(row["feedback_energy"], 0.0)
            self.assertEqual(row["global_energy_fraction"], 0.0)


class FeedbackSummaryTests(unittest.TestCase):
    @staticmethod
    def _rows():
        rows = []
        trajectory_by_model = {
            "model-a": {
                "full_rollout_mse": 1.0,
                "fde": 2.0,
                "f24_centroid_error": 3.0,
                "f24_shape_residual_mse": 4.0,
            },
            "model-b": {
                "full_rollout_mse": 2.0,
                "fde": 4.0,
                "f24_centroid_error": 6.0,
                "f24_shape_residual_mse": 8.0,
            },
        }
        for model_index, (model, trajectory) in enumerate(trajectory_by_model.items()):
            for absolute_frame in (5, 11, 18):
                for stage in range(4):
                    value = float(model_index + 1)
                    rows.append(
                        {
                            "model": model,
                            "mat_type": model_index,
                            "log10_e": 1.0 + model_index,
                            "nu": 0.2 + model_index,
                            "rollout_step": absolute_frame - 5,
                            "absolute_frame": absolute_frame,
                            "stage": stage,
                            "gate": 0.1 * (stage + 1),
                            "feedback_rms": value,
                            "global_rms": value * 2.0,
                            "deform_rms": value * 3.0,
                            "global_energy_fraction": 0.1 * value,
                            **trajectory,
                        }
                    )
        return rows

    def test_horizon_bucket_uses_fixed_inclusive_boundaries(self):
        self.assertEqual(horizon_bucket(5), "short")
        self.assertEqual(horizon_bucket(10), "short")
        self.assertEqual(horizon_bucket(11), "mid")
        self.assertEqual(horizon_bucket(17), "mid")
        self.assertEqual(horizon_bucket(18), "long")
        self.assertEqual(horizon_bucket(24), "long")

    def test_aggregate_feedback_rows_reports_unique_model_counts(self):
        summary = aggregate_feedback_rows(self._rows())

        stage_row = next(
            row
            for row in summary
            if row["group"] == "overall" and row["stage"] == 0
        )
        horizon_row = next(
            row
            for row in summary
            if row["group"] == "overall" and row["horizon"] == "short"
        )
        self.assertEqual(stage_row["n_models"], 2)
        self.assertAlmostEqual(stage_row["feedback_rms"], 1.5)
        self.assertEqual(horizon_row["n_models"], 2)
        self.assertAlmostEqual(horizon_row["global_rms"], 3.0)
        self.assertEqual(
            {row["group"] for row in summary},
            {"overall", "elastic", "plasticine"},
        )

    def test_aggregate_feedback_rows_rejects_empty_missing_and_nonfinite_input(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            aggregate_feedback_rows([])

        missing = self._rows()
        del missing[0]["feedback_rms"]
        with self.assertRaisesRegex(ValueError, "feedback_rms"):
            aggregate_feedback_rows(missing)

        nonfinite = self._rows()
        nonfinite[0]["deform_rms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregate_feedback_rows(nonfinite)

    def test_feedback_correlations_aggregate_feedback_at_model_level(self):
        rows = []
        specs = (
            ("model-a", 0, 1.0, 1.0, 1),
            ("model-b", 1, 2.0, 4.0, 2),
            ("model-c", 2, 4.0, 2.0, 5),
        )
        for model, mat_type, feedback_value, trajectory_value, multiplicity in specs:
            for repetition in range(multiplicity):
                rows.append(
                    {
                        "model": model,
                        "mat_type": mat_type,
                        "absolute_frame": 5 + repetition,
                        "stage": repetition % 4,
                        "feedback_rms": feedback_value,
                        "global_rms": feedback_value,
                        "deform_rms": feedback_value,
                        "global_energy_fraction": feedback_value / 10.0,
                        "full_rollout_mse": trajectory_value,
                        "fde": trajectory_value,
                        "f24_centroid_error": trajectory_value,
                        "f24_shape_residual_mse": trajectory_value,
                    }
                )

        correlations = feedback_correlations(rows)
        direct = next(
            row
            for row in correlations
            if row["group"] == "overall"
            and row["feedback_metric"] == "feedback_rms"
            and row["trajectory_metric"] == "full_rollout_mse"
        )
        inverse = next(
            row
            for row in correlations
            if row["group"] == "overall"
            and row["feedback_metric"] == "feedback_rms"
            and row["trajectory_metric"] == "fde"
        )
        self.assertEqual(direct["n_models"], 3)
        self.assertAlmostEqual(direct["pearson"], 1.0 / 7.0)
        self.assertAlmostEqual(direct["spearman"], 0.5)
        self.assertAlmostEqual(inverse["pearson"], 1.0 / 7.0)
        self.assertAlmostEqual(inverse["spearman"], 0.5)
        row_level_pearson = np.corrcoef(
            np.repeat([1.0, 2.0, 4.0], [1, 2, 5]),
            np.repeat([1.0, 4.0, 2.0], [1, 2, 5]),
        )[0, 1]
        self.assertNotAlmostEqual(row_level_pearson, direct["pearson"], places=6)

    def test_feedback_correlations_returns_nan_for_constant_or_single_model_groups(self):
        rows = self._rows()
        rows = [row for row in rows if row["model"] == "model-a"]
        correlations = feedback_correlations(rows)
        result = next(
            row
            for row in correlations
            if row["group"] == "overall"
            and row["feedback_metric"] == "feedback_rms"
            and row["trajectory_metric"] == "full_rollout_mse"
        )
        self.assertTrue(math.isnan(result["pearson"]))
        self.assertTrue(math.isnan(result["spearman"]))

    def test_feedback_correlations_rejects_missing_or_nonfinite_input(self):
        missing = self._rows()
        del missing[0]["fde"]
        with self.assertRaisesRegex(ValueError, "fde"):
            feedback_correlations(missing)

        nonfinite = self._rows()
        nonfinite[0]["full_rollout_mse"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            feedback_correlations(nonfinite)

    def test_feedback_writers_sort_csv_and_render_report_tables(self):
        rows = self._rows()
        rows.reverse()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "feedback.csv"
            report_path = Path(directory) / "feedback.md"
            write_feedback_csv(csv_path, rows)
            write_feedback_report(
                report_path,
                rows,
                {
                    "checkpoint": "checkpoint-90000/model.safetensors",
                    "config": "configs/eval.yaml",
                    "windows": "41-window start_idx=0",
                },
            )

            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]["model"], "model-a")
            self.assertEqual(csv_rows[0]["rollout_step"], "0")
            self.assertEqual(csv_rows[0]["stage"], "0")

            report = report_path.read_text(encoding="utf-8")
            for token in (
                "checkpoint-90000/model.safetensors",
                "configs/eval.yaml",
                "41-window",
                "overall",
                "elastic",
                "plasticine",
                "sand",
                "stage",
                "horizon",
                "correlation",
            ):
                self.assertIn(token, report)
            self.assertNotIn("n=13/14", report)
            self.assertIn("elastic=1", report)
            self.assertIn("plasticine=1", report)
            self.assertNotIn("nan", report.lower())

    def test_feedback_report_requires_checkpoint_and_config_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "feedback.md"
            cases = (
                ({}, "checkpoint"),
                ({"checkpoint": "checkpoint.safetensors"}, "config"),
                ({"checkpoint": "", "config": "configs/eval.yaml"}, "checkpoint"),
                ({"checkpoint": "checkpoint.safetensors", "config": "  "}, "config"),
            )
            for metadata, missing_field in cases:
                with self.subTest(missing_field=missing_field, metadata=metadata):
                    with self.assertRaisesRegex(ValueError, missing_field):
                        write_feedback_report(report_path, self._rows(), metadata)

    def test_feedback_report_counts_material_models_from_raw_rows_with_partial_horizons(self):
        rows = [
            row
            for row in self._rows()
            if row["model"] == "model-a" or row["absolute_frame"] == 5
        ]
        for row in rows:
            row["mat_type"] = 0

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "feedback.md"
            write_feedback_report(
                report_path,
                rows,
                {
                    "checkpoint": "checkpoint.safetensors",
                    "config": "configs/eval.yaml",
                },
            )

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Material model counts: elastic=2.", report)
            self.assertIn(
                "| elastic | feedback_rms | full_rollout_mse | 2 |",
                report,
            )


if __name__ == "__main__":
    unittest.main()
