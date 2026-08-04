import unittest
from types import SimpleNamespace
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
