import unittest
from unittest.mock import patch

import torch

from model.hybrid_state import HybridStateExchange
from utils.hybrid_state_diagnostics import (
    HybridStateFeedbackRecorder,
    decompose_feedback,
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


if __name__ == "__main__":
    unittest.main()
