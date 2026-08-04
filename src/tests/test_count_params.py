import subprocess
import sys
import unittest
from pathlib import Path

import count_params


SRC_DIR = Path(__file__).resolve().parents[1]


class CountParamsImportTests(unittest.TestCase):
    def test_import_has_no_model_report_side_effects(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import count_params"],
            cwd=SRC_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")


class ExchangeComputeEstimateTests(unittest.TestCase):
    def test_parts_sum_and_calls_scale_known_production_shape(self):
        parts, per_call_macs, total_macs = count_params.estimate_exchange_compute(
            batch_size=1,
            n_points=2048,
            history_frames=5,
            particle_dim=256,
            state_dim=64,
            exchange_calls=4,
        )

        self.assertEqual(sum(parts.values()), per_call_macs)
        self.assertEqual(per_call_macs, 74_105_728)
        self.assertEqual(total_macs, 4 * per_call_macs)
        self.assertEqual(total_macs, 296_422_912)


class V11aParameterValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = count_params.build_mm3("SpatialTemporalTransformerBlock")
        cls.v11a = count_params.build_mm3("SpatialTemporalTransformerBlockv11a")

    def test_actual_models_return_expected_signed_delta_and_percentage(self):
        report = count_params.validate_v11a_parameter_budget(
            self.baseline,
            self.v11a,
        )

        self.assertEqual(report["signed_delta"], 160_773)
        self.assertAlmostEqual(report["delta_percent"], 0.998933, places=6)
        self.assertEqual(report["block_count"], 8)
        self.assertEqual(report["exchange_count"], 1)
        self.assertEqual(report["exchange_stages"], 4)
        self.assertEqual(report["exchange_calls"], 4)

    def test_frozen_exchange_fails_validation_explicitly(self):
        exchange = self.v11a.dit.hybrid_state_exchange
        for parameter in exchange.parameters():
            parameter.requires_grad_(False)
        try:
            with self.assertRaisesRegex(RuntimeError, "v11a total.*trainable"):
                count_params.validate_v11a_parameter_budget(
                    self.baseline,
                    self.v11a,
                )
        finally:
            for parameter in exchange.parameters():
                parameter.requires_grad_(True)


class MaterialStateParameterValidationTests(unittest.TestCase):
    def test_b3_exact_parameter_delta(self):
        baseline = count_params.build_mm3(
            "SpatialTemporalTransformerBlock",
            contact_particle_cond=True,
            contact_feature_sigma=0.04,
            contact_injection_mode="separate",
        )
        candidate = count_params.build_mm3(
            "SpatialTemporalTransformerBlock",
            contact_particle_cond=True,
            contact_feature_sigma=0.04,
            contact_injection_mode="separate",
            material_state_adapter=True,
            material_state_rank=64,
            material_state_interval=2,
        )

        report = count_params.validate_material_state_parameter_budget(
            baseline, candidate
        )

        self.assertEqual(report["adapter_params"], 34_052)
        self.assertEqual(report["signed_delta"], 34_052)
        self.assertLess(report["delta_percent"], 0.3)
        self.assertEqual(report["block_count"], 8)
        self.assertEqual(report["stage_count"], 4)
        self.assertEqual(report["rank"], 64)

    def test_b3_common_parameters_share_exact_initialization(self):
        import torch

        torch.manual_seed(0)
        baseline = count_params.build_mm3(
            "SpatialTemporalTransformerBlock",
            contact_particle_cond=True,
            contact_injection_mode="separate",
        )
        torch.manual_seed(0)
        candidate = count_params.build_mm3(
            "SpatialTemporalTransformerBlock",
            contact_particle_cond=True,
            contact_injection_mode="separate",
            material_state_adapter=True,
        )

        base_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        common = set(base_state) & set(candidate_state)
        self.assertIn("dit.norm_final.weight", common)
        self.assertIn("dit.proj_out.weight", common)
        for name in common:
            self.assertTrue(torch.equal(base_state[name], candidate_state[name]), name)


if __name__ == "__main__":
    unittest.main()
