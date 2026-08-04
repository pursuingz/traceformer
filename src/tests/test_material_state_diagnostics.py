import math
import unittest

import torch

from count_params import build_mm3
from utils.material_state_diagnostics import summarize_material_state_adapter


class MaterialStateDiagnosticsTest(unittest.TestCase):
    def test_rejects_model_without_adapter(self):
        model = build_mm3("SpatialTemporalTransformerBlock")
        with self.assertRaisesRegex(ValueError, "material-state adapter"):
            summarize_material_state_adapter(model)

    def test_reports_frozen_b3_structure_and_finite_norms(self):
        model = build_mm3(
            "SpatialTemporalTransformerBlock",
            material_state_adapter=True,
            material_state_rank=64,
            material_state_interval=2,
        )

        report = summarize_material_state_adapter(model)

        self.assertEqual(report["parameter_count"], 34_052)
        self.assertEqual(report["rank"], 64)
        self.assertEqual(report["interval"], 2)
        self.assertEqual(report["num_stages"], 4)
        self.assertEqual(report["stage_scales"], [1.0, 1.0, 1.0, 1.0])
        for field in (
            "state_projection_norm",
            "material_projection_norm",
            "output_projection_norm",
        ):
            self.assertTrue(math.isfinite(report[field]))

    def test_rejects_nonfinite_checkpoint_parameters(self):
        model = build_mm3(
            "SpatialTemporalTransformerBlock",
            material_state_adapter=True,
        )
        with torch.no_grad():
            model.dit.material_state_exchange.stage_scales[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            summarize_material_state_adapter(model)


if __name__ == "__main__":
    unittest.main()
