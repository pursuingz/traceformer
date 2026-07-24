import unittest

import torch

from contact_feature_diagnostics import collect_grouped_features


class ContactFeatureDiagnosticTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
