import unittest

import torch

from model.material_state import FactorizedMaterialStateAdapter


class MaterialStageGateMathTest(unittest.TestCase):
    @staticmethod
    def make_adapter(enabled: bool) -> FactorizedMaterialStateAdapter:
        adapter = FactorizedMaterialStateAdapter(
            particle_dim=16,
            rank=4,
            num_materials=4,
            num_stages=4,
            material_stage_gate=enabled,
            gated_materials=3,
            gate_max=2.0,
        )
        with torch.no_grad():
            adapter.output_proj.weight.fill_(0.1)
            adapter.output_proj.bias.zero_()
        return adapter

    def test_disabled_adapter_has_no_gate_parameter(self):
        adapter = self.make_adapter(False)
        self.assertNotIn("gate_logits", dict(adapter.named_parameters()))

    def test_enabled_gate_is_identity_and_bounded(self):
        adapter = self.make_adapter(True)
        gates = adapter.material_stage_gates()
        self.assertEqual(tuple(gates.shape), (3, 4))
        self.assertTrue(torch.equal(gates, torch.ones_like(gates)))

        with torch.no_grad():
            adapter.gate_logits[0, 0] = -100.0
            adapter.gate_logits[1, 0] = 100.0
        gates = adapter.material_stage_gates()
        self.assertGreaterEqual(gates.min().item(), 0.0)
        self.assertLessEqual(gates.max().item(), 2.0)

    def test_rigid_material_uses_identity_gate(self):
        adapter = self.make_adapter(True)
        with torch.no_grad():
            adapter.gate_logits.fill_(-5.0)
        labels = torch.tensor([0, 1, 2, 3])
        gates = adapter.gate_for(labels, stage_index=2, dtype=torch.float32)
        self.assertLess(gates[:3].max().item(), 1.0)
        self.assertEqual(gates[3].item(), 1.0)

    def test_only_selected_material_row_receives_gradient(self):
        torch.manual_seed(3)
        adapter = self.make_adapter(True)
        hidden = torch.randn(2, 6, 5, 16)
        material_values = torch.tensor([[5.0, 0.20], [6.0, 0.35]])
        material_labels = torch.tensor([1, 1])

        adapter(
            hidden,
            material_values,
            material_labels,
            stage_index=0,
        ).sum().backward()

        grad = adapter.gate_logits.grad
        self.assertIsNotNone(grad)
        self.assertEqual(torch.count_nonzero(grad[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(grad[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(grad[2]).item(), 0)

    def test_constructor_rejects_invalid_gate_settings(self):
        with self.assertRaisesRegex(ValueError, "gated_materials"):
            FactorizedMaterialStateAdapter(
                16,
                4,
                4,
                4,
                material_stage_gate=True,
                gated_materials=4,
            )
        with self.assertRaisesRegex(ValueError, "gate_max"):
            FactorizedMaterialStateAdapter(
                16,
                4,
                4,
                4,
                material_stage_gate=True,
                gate_max=0.0,
            )


if __name__ == "__main__":
    unittest.main()
