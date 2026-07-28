import unittest

import torch

from contact_feature_diagnostics import (
    collect_grouped_features,
    load_contact_projection,
)


class ContactFeatureDiagnosticTests(unittest.TestCase):
    def test_loads_separate_contact_projection_and_bias(self):
        state = {
            "model.contact_encoder.weight": torch.arange(
                12, dtype=torch.float32
            ).reshape(4, 3),
            "model.contact_encoder.bias": torch.arange(
                4, dtype=torch.float32
            ),
        }

        weight, bias = load_contact_projection(
            state,
            injection_mode="separate",
            feature_dim=3,
        )

        torch.testing.assert_close(
            weight,
            state["model.contact_encoder.weight"],
        )
        torch.testing.assert_close(
            bias,
            state["model.contact_encoder.bias"],
        )

    def test_loads_shared_contact_columns_without_shared_bias(self):
        full_weight = torch.arange(
            4 * 102, dtype=torch.float32
        ).reshape(4, 102)
        state = {
            "model.input_encoder.mlp.weight": full_weight,
            "model.input_encoder.mlp.bias": torch.ones(4),
        }

        weight, bias = load_contact_projection(
            state,
            injection_mode="shared",
            feature_dim=3,
        )

        torch.testing.assert_close(weight, full_weight[:, 99:102])
        self.assertIsNone(bias)

    def test_collects_tuple_dataloader_batches_by_material(self):
        points = torch.zeros(2, 2, 1, 3)
        points[0, :, 0, 1] = torch.tensor([0.1, 0.2])
        points[1, :, 0, 1] = torch.tensor([0.3, 0.5])
        batch = {
            "points_src": points,
            "floor_height": torch.zeros(2, 1),
            "start_vel": torch.zeros(2, 1, 3),
            "mat_type": torch.tensor([0, 2]),
        }
        loader = [(batch, {"unused": True})]

        grouped = collect_grouped_features(loader, sigma=0.04)

        self.assertEqual(set(grouped), {0, 2})
        self.assertEqual(grouped[0][0].shape, (2, 3))
        self.assertEqual(grouped[2][0].shape, (2, 3))
        torch.testing.assert_close(
            grouped[0][0][:, 0],
            torch.tensor([0.1, 0.2]),
        )

    def test_collects_xyz_displacement_features(self):
        points = torch.tensor(
            [[
                [[0.0, 0.1, 0.0]],
                [[0.2, 0.3, -0.4]],
            ]]
        )
        batch = {
            "points_src": points,
            "floor_height": torch.zeros(1, 1),
            "start_vel": torch.tensor([[[0.5, -0.2, 0.1]]]),
            "mat_type": torch.tensor([1]),
        }

        grouped = collect_grouped_features(
            [(batch, {})],
            sigma=0.04,
            velocity_mode="xyz",
        )

        self.assertEqual(grouped[1][0].shape, (2, 5))
        torch.testing.assert_close(
            grouped[1][0][0, 1:4],
            torch.tensor([0.5, -0.2, 0.1]),
        )
        torch.testing.assert_close(
            grouped[1][0][1, 1:4],
            torch.tensor([0.2, 0.2, -0.4]),
        )


if __name__ == "__main__":
    unittest.main()
