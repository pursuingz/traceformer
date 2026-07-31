import unittest
from unittest import mock

import numpy as np
import torch

from src.utils.material_condition_diagnostics import (
    MaterialRecord,
    build_parameter_derangement,
    condition_response_metrics,
    rotate_material_type,
    trajectory_metrics,
)


class MaterialConditionDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _trajectory():
        gt = np.zeros((25, 2, 3), dtype=np.float64)
        pred = np.zeros_like(gt)
        for frame in range(5, 25):
            pred[frame, :, :] = frame - 4
        return pred, gt

    def test_trajectory_metrics_match_full_rollout_and_prediction_horizon(self):
        pred, gt = self._trajectory()

        metrics = trajectory_metrics(pred, gt, input_frames=5)

        expected_gm = float(np.exp(np.mean(np.log(np.arange(1, 21, dtype=float) ** 2))))
        self.assertAlmostEqual(metrics["full_rollout_mse"], 114.8)
        self.assertAlmostEqual(metrics["gm_mse"], expected_gm)
        self.assertAlmostEqual(metrics["long_seg_mse"], 293.0)
        self.assertAlmostEqual(metrics["fde"], 20.0 * np.sqrt(3.0))

    def test_condition_response_metrics_excludes_identical_condition_frames(self):
        normal = np.zeros((25, 2, 3), dtype=np.float64)
        counterfactual = np.zeros_like(normal)
        counterfactual[:5] = 100.0
        for frame in range(5, 25):
            counterfactual[frame, :, :] = frame - 4

        response = condition_response_metrics(normal, counterfactual, input_frames=5)

        self.assertAlmostEqual(response["prediction_mse"], 143.5)
        self.assertAlmostEqual(response["final_prediction_mse"], 400.0)

    def test_trajectory_metrics_detaches_cpu_tensor_before_numpy_conversion(self):
        pred, gt = self._trajectory()
        pred_tensor = torch.tensor(pred, requires_grad=True)
        gt_tensor = torch.tensor(gt, requires_grad=True)

        metrics = trajectory_metrics(pred_tensor, gt_tensor, input_frames=5)

        self.assertAlmostEqual(metrics["full_rollout_mse"], 114.8)

    def test_torch_conversion_path_does_not_pass_original_tensor_to_numpy(self):
        pred, gt = self._trajectory()

        class GuardedTensor:
            def __init__(self, value):
                self.value = value
                self.calls = []

            def detach(self):
                self.calls.append("detach")
                return self

            def cpu(self):
                self.calls.append("cpu")
                return self

            def numpy(self):
                self.calls.append("numpy")
                return self.value

        guarded_pred = GuardedTensor(pred)
        guarded_gt = GuardedTensor(gt)
        original_asarray = np.asarray

        def assert_numpy_input(value, *args, **kwargs):
            self.assertIsNot(value, guarded_pred)
            self.assertIsNot(value, guarded_gt)
            return original_asarray(value, *args, **kwargs)

        with mock.patch(
            "src.utils.material_condition_diagnostics.torch.is_tensor",
            side_effect=lambda value: value in (guarded_pred, guarded_gt),
        ), mock.patch(
            "src.utils.material_condition_diagnostics.np.asarray",
            side_effect=assert_numpy_input,
        ):
            trajectory_metrics(guarded_pred, guarded_gt, input_frames=5)

        self.assertEqual(guarded_pred.calls, ["detach", "cpu", "numpy"])
        self.assertEqual(guarded_gt.calls, ["detach", "cpu", "numpy"])

    def test_trajectory_metrics_rejects_non_three_dimensional_inputs(self):
        with self.assertRaisesRegex(ValueError, r"shape \(T,N,3\)"):
            trajectory_metrics(np.zeros((1, 2, 3, 1)), np.zeros((1, 2, 3, 1)))

    def test_trajectory_metrics_rejects_invalid_input_frame_count(self):
        pred, gt = self._trajectory()
        for input_frames in (0, 25):
            with self.subTest(input_frames=input_frames):
                with self.assertRaisesRegex(ValueError, "input_frames"):
                    trajectory_metrics(pred, gt, input_frames=input_frames)

    def test_derangement_is_within_material_one_to_one_and_reproducible(self):
        records = [
            MaterialRecord("e0.h5", 0, 4.0, 0.1),
            MaterialRecord("e1.h5", 0, 5.0, 0.2),
            MaterialRecord("e2.h5", 0, 6.0, 0.3),
            MaterialRecord("s0.h5", 2, 4.5, 0.15),
            MaterialRecord("s1.h5", 2, 6.5, 0.35),
        ]
        first = build_parameter_derangement(records, seed=7)
        second = build_parameter_derangement(records, seed=7)
        self.assertEqual(first, second)
        for group in ({"e0.h5", "e1.h5", "e2.h5"}, {"s0.h5", "s1.h5"}):
            source = {(r.log10_e, r.nu) for r in records if r.model in group}
            assigned = {first[name] for name in group}
            self.assertEqual(source, assigned)
            for record in (r for r in records if r.model in group):
                self.assertNotEqual(first[record.model], (record.log10_e, record.nu))

    def test_rotates_supported_material_classes(self):
        self.assertEqual([rotate_material_type(i) for i in (0, 1, 2)], [1, 2, 0])

    def test_rejects_unsupported_material_class(self):
        with self.assertRaisesRegex(ValueError, "expected one of"):
            rotate_material_type(3)

    def test_rejects_unsupported_material_class_in_records(self):
        records = [
            MaterialRecord("e0.h5", 0, 4.0, 0.1),
            MaterialRecord("x0.h5", 3, 5.0, 0.2),
        ]
        with self.assertRaisesRegex(ValueError, "expected one of"):
            build_parameter_derangement(records, seed=7)

    def test_rejects_singleton_material_group(self):
        records = [MaterialRecord("e0.h5", 0, 4.0, 0.1)]
        with self.assertRaisesRegex(ValueError, "at least two records"):
            build_parameter_derangement(records, seed=7)


if __name__ == "__main__":
    unittest.main()
