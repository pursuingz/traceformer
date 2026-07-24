import unittest

import torch

from utils.eval_metrics import append_result_tag, contact_region_metrics


class ContactRegionMetricTests(unittest.TestCase):
    def test_result_tag_isolates_artifacts_without_changing_default_path(self):
        self.assertEqual(append_result_tag("vis_results", None), "vis_results")
        self.assertEqual(append_result_tag("vis_results", ""), "vis_results")
        self.assertEqual(
            append_result_tag("vis_results", "no_velocity"),
            "vis_results_no_velocity",
        )

    def test_separates_contact_and_free_space_position_errors(self):
        gt = torch.zeros(1, 1, 2, 3)
        gt[0, 0, :, 1] = torch.tensor([0.02, 1.0])
        pred = gt.clone()
        pred[0, 0, 0, 0] = 3.0
        pred[0, 0, 1, 0] = 6.0

        metrics = contact_region_metrics(
            pred,
            gt,
            previous_frame=torch.zeros(1, 2, 3),
            floor_height=torch.zeros(1),
            margin=0.04,
        )

        self.assertEqual(metrics["contact_count"].item(), 1)
        self.assertEqual(metrics["free_count"].item(), 1)
        self.assertAlmostEqual(metrics["contact_mse_sum"].item(), 3.0)
        self.assertAlmostEqual(metrics["free_mse_sum"].item(), 12.0)

    def test_normal_velocity_uses_previous_frame(self):
        previous = torch.zeros(1, 1, 3)
        previous[..., 1] = 0.03
        gt = torch.zeros(1, 1, 1, 3)
        gt[..., 1] = 0.02
        pred = gt.clone()
        pred[..., 1] = 0.04

        metrics = contact_region_metrics(
            pred,
            gt,
            previous_frame=previous,
            floor_height=torch.zeros(1),
            margin=0.04,
        )

        # GT delta-y=-0.01, pred delta-y=+0.01, squared error=0.0004.
        self.assertEqual(metrics["normal_velocity_count"].item(), 1)
        self.assertAlmostEqual(
            metrics["normal_velocity_mse_sum"].item(),
            4e-4,
            places=8,
        )

    def test_contact_precision_recall_use_particle_time_counts(self):
        gt = torch.zeros(1, 1, 3, 3)
        pred = torch.zeros_like(gt)
        gt[0, 0, :, 1] = torch.tensor([0.01, 0.02, 1.0])
        pred[0, 0, :, 1] = torch.tensor([0.01, 1.0, 0.03])

        metrics = contact_region_metrics(
            pred,
            gt,
            previous_frame=torch.zeros(1, 3, 3),
            floor_height=torch.zeros(1),
            margin=0.04,
        )

        self.assertEqual(metrics["true_positive_count"].item(), 1)
        self.assertEqual(metrics["pred_contact_count"].item(), 2)
        self.assertEqual(metrics["gt_contact_count"].item(), 2)

    def test_empty_regions_return_zero_sums_and_counts(self):
        gt = torch.zeros(1, 1, 1, 3)
        pred = gt.clone()
        gt[..., 1] = 1.0
        pred[..., 1] = 1.0

        metrics = contact_region_metrics(
            pred,
            gt,
            previous_frame=torch.zeros(1, 1, 3),
            floor_height=torch.zeros(1),
            margin=0.04,
        )

        self.assertEqual(metrics["contact_count"].item(), 0)
        self.assertEqual(metrics["contact_mse_sum"].item(), 0.0)
        self.assertEqual(metrics["normal_velocity_count"].item(), 0)
        self.assertEqual(metrics["normal_velocity_mse_sum"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
