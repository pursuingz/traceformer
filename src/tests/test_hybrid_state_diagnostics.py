import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        rows = self._rows()
        for row in rows:
            model_value = 1.0 if row["model"] == "model-a" else 2.0
            row["feedback_rms"] = model_value
            row["full_rollout_mse"] = model_value
            row["fde"] = 3.0 - model_value

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
        self.assertEqual(direct["n_models"], 2)
        self.assertEqual(direct["pearson"], 1.0)
        self.assertEqual(direct["spearman"], 1.0)
        self.assertEqual(inverse["pearson"], -1.0)
        self.assertEqual(inverse["spearman"], -1.0)

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
        correlations = feedback_correlations(rows)
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
            self.assertNotIn("nan", report.lower())


if __name__ == "__main__":
    unittest.main()
