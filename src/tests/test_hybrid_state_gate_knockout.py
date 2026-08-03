import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from utils.eval_metrics import per_window_metrics
from utils.hybrid_state_gate_knockout import (
    KNOCKOUT_CONDITIONS,
    masked_feedback_gates,
    trajectory_knockout_metrics,
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


if __name__ == "__main__":
    unittest.main()
