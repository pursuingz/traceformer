import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from utils.hybrid_state_gate_knockout import (
    KNOCKOUT_CONDITIONS,
    masked_feedback_gates,
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


if __name__ == "__main__":
    unittest.main()
