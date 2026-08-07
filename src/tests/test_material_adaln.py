import unittest

import torch

import count_params
from model.material_adaln import ContinuousMaterialConditioner


class ContinuousMaterialConditionerTest(unittest.TestCase):
    def test_normalizes_log_e_and_nu_jointly(self):
        module = ContinuousMaterialConditioner(output_dim=8, hidden_dim=4)
        values = torch.tensor([[4.5, 0.10], [6.5, 0.40]])
        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        torch.testing.assert_close(
            module.normalize_material_values(values), expected
        )

    def test_rejects_invalid_constructor_scales_and_dims(self):
        for kwargs in (
            {"output_dim": 0},
            {"output_dim": 8, "hidden_dim": 0},
            {"output_dim": 8, "e_scale": 0.0},
            {"output_dim": 8, "nu_scale": 0.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ContinuousMaterialConditioner(**kwargs)

    def test_rejects_wrong_shape_and_non_finite_values(self):
        module = ContinuousMaterialConditioner(output_dim=8, hidden_dim=4)
        with self.assertRaisesRegex(ValueError, "shape"):
            module(torch.zeros(2, 1))
        with self.assertRaisesRegex(ValueError, "finite"):
            module(torch.tensor([[5.5, float("nan")]]))

    def test_zero_init_outputs_exact_zero_and_budget_is_exact(self):
        module = ContinuousMaterialConditioner(output_dim=256, hidden_dim=64)
        output = module(torch.tensor([[5.5, 0.25], [6.0, 0.40]]))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertEqual(sum(p.numel() for p in module.parameters()), 16832)

    def test_both_continuous_inputs_receive_gradient_after_output_path_is_active(
        self,
    ):
        module = ContinuousMaterialConditioner(output_dim=8, hidden_dim=4)
        with torch.no_grad():
            module.output_proj.weight.fill_(0.1)
        values = torch.tensor([[5.0, 0.20]], requires_grad=True)
        module(values).sum().backward()
        self.assertGreater(values.grad[0, 0].abs().item(), 0.0)
        self.assertGreater(values.grad[0, 1].abs().item(), 0.0)

    def test_b4_model_has_exact_conditioner_parameter_budget(self):
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
            material_adaln_cond=True,
        )

        report = count_params.validate_material_adaln_parameter_budget(
            baseline, candidate
        )

        self.assertEqual(report["signed_delta"], 16_832)
        self.assertEqual(report["conditioner_params"], 16_832)
        self.assertEqual(report["conditioner_count"], 1)
        self.assertEqual(report["block_count"], 8)
        self.assertEqual(
            report["block_types"],
            ["SpatialTemporalTransformerBlock"] * 8,
        )


if __name__ == "__main__":
    unittest.main()
