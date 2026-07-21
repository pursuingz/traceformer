import unittest

import torch

from model.hybrid_state import compute_explicit_frame_state


class HybridFrameStateTests(unittest.TestCase):
    def setUp(self):
        base = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
                [2.0, 1.0, -2.0],
                [-2.0, -1.0, 2.0],
            ]
        )
        frame_scales = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.5, 0.5, 1.5],
                [1.0, 1.25, 0.5],
            ]
        )
        self.offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
            ]
        )
        deformed = base[None] * frame_scales[:, None]
        self.points = deformed[None] + self.offsets[None, :, None]

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
        expected_frames = []
        for frame in self.points[0]:
            centered = frame - frame.mean(dim=0)
            x, y, z = centered.unbind(dim=-1)
            expected_frames.append(
                torch.stack(
                    (
                        (x * x).mean(),
                        (x * y).mean(),
                        (x * z).mean(),
                        (y * y).mean(),
                        (y * z).mean(),
                        (z * z).mean(),
                    )
                )
            )
        expected_covariance = torch.stack(expected_frames).unsqueeze(0)
        expected_delta = torch.cat(
            (
                torch.zeros_like(expected_covariance[:, :1]),
                expected_covariance[:, 1:] - expected_covariance[:, :-1],
            ),
            dim=1,
        )

        self.assertTrue(torch.all(expected_covariance[..., [1, 2, 4]] != 0))
        self.assertTrue(torch.all(expected_delta[:, 1:] != 0))

        state = compute_explicit_frame_state(self.points)

        self.assertTrue(torch.all(state[0, 1:, 12:18] != 0))
        torch.testing.assert_close(
            state[0, :, 6:12],
            expected_covariance[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state[0, :, 12:18],
            expected_delta[0],
            rtol=0,
            atol=0,
        )

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
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(torch.count_nonzero(points.grad).item(), 0)


if __name__ == "__main__":
    unittest.main()
