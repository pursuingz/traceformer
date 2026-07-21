import unittest

import torch

from model.hybrid_state import compute_explicit_frame_state


class HybridFrameStateTests(unittest.TestCase):
    def setUp(self):
        base = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        self.offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
            ]
        )
        self.points = base[None, None] + self.offsets[None, :, None]

    def test_returns_one_18_value_state_per_frame(self):
        state = compute_explicit_frame_state(self.points)

        self.assertEqual(state.shape, (1, 3, 18))

    def test_relative_center_occupies_first_three_values(self):
        state = compute_explicit_frame_state(self.points)

        torch.testing.assert_close(state[0, :, :3], self.offsets)

    def test_adjacent_center_velocity_occupies_values_three_to_six(self):
        expected_velocity = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [2.0, 0.0, -2.0],
            ]
        )

        state = compute_explicit_frame_state(self.points)

        torch.testing.assert_close(state[0, :, 3:6], expected_velocity)

    def test_covariance_and_delta_use_upper_triangle_layout(self):
        expected_covariance = torch.tensor([0.5, 0.0, 0.0, 2.0, 0.0, 0.0])

        state = compute_explicit_frame_state(self.points)

        torch.testing.assert_close(
            state[0, :, 6:12],
            expected_covariance.expand(3, -1),
        )
        torch.testing.assert_close(state[0, :, 12:18], torch.zeros(3, 6))

    def test_requires_rank_four_input(self):
        for invalid in (torch.zeros(3, 4, 3), torch.zeros(1, 2, 3, 4, 3)):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError):
                    compute_explicit_frame_state(invalid)

    def test_requires_xyz_last_dimension(self):
        with self.assertRaises(ValueError):
            compute_explicit_frame_state(torch.zeros(1, 2, 4, 2))

    def test_requires_at_least_one_frame_and_particle(self):
        for invalid in (torch.zeros(1, 0, 4, 3), torch.zeros(1, 2, 0, 3)):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError):
                    compute_explicit_frame_state(invalid)

    def test_preserves_autograd(self):
        points = self.points.clone().requires_grad_(True)

        compute_explicit_frame_state(points).square().sum().backward()

        self.assertIsNotNone(points.grad)


if __name__ == "__main__":
    unittest.main()
