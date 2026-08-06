import unittest

import torch
from omegaconf import OmegaConf

from model.material_state import FactorizedMaterialStateAdapter
from model.spacetime import MDM_ST


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


class MaterialStageGateIntegrationTest(unittest.TestCase):
    @staticmethod
    def small_config(*, adapter: bool = True, gate: bool = False):
        return OmegaConf.create(
            {
                "n_layers": 2,
                "latent_dim": 64,
                "frame_cond": True,
                "cond_frames": 5,
                "point_embed": True,
                "mask_cond": True,
                "pred_offset": True,
                "num_neighbors": -1,
                "floor_cond": False,
                "max_num_forces": 1,
                "force_as_token": False,
                "force_as_latent": False,
                "gravity_emb": False,
                "coeff_cond": False,
                "num_mat": 4,
                "class_token": True,
                "class_dropout_prob": 0.0,
                "transformer_block": "SpatialTemporalTransformerBlock",
                "material_state_adapter": adapter,
                "material_state_rank": 16,
                "material_state_interval": 1,
                "material_stage_gate": gate,
                "material_stage_gate_max": 2.0,
            }
        )

    @staticmethod
    def small_inputs():
        return {
            "x": torch.randn(1, 1, 2, 3),
            "timesteps": torch.zeros(1, dtype=torch.long),
            "init_pc": torch.randn(1, 5, 2, 3),
            "force": torch.randn(1, 3),
            "E": torch.tensor([[5.5]]),
            "nu": torch.tensor([[0.25]]),
            "drag_mask": torch.zeros(1, 1, 2, 1),
            "drag_point": torch.zeros(1, 4),
            "floor_height": None,
            "y": torch.tensor([1]),
        }

    def test_disabled_b3a_state_dict_has_no_gate_parameter(self):
        model = MDM_ST(2, 1, 3, self.small_config(gate=False))
        self.assertNotIn(
            "dit.material_state_exchange.gate_logits",
            model.state_dict(),
        )

    def test_b3a_checkpoint_loads_with_only_gate_missing(self):
        b3a = MDM_ST(2, 1, 3, self.small_config(gate=False)).eval()
        b3b = MDM_ST(2, 1, 3, self.small_config(gate=True)).eval()

        incompatible = b3b.load_state_dict(b3a.state_dict(), strict=False)

        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(
            incompatible.missing_keys,
            ["dit.material_state_exchange.gate_logits"],
        )

    def test_identity_gate_matches_nonzero_b3a_forward(self):
        torch.manual_seed(17)
        b3a = MDM_ST(2, 1, 3, self.small_config(gate=False)).eval()
        with torch.no_grad():
            b3a.dit.material_state_exchange.output_proj.weight.fill_(0.01)
            b3a.dit.material_state_exchange.output_proj.bias.fill_(0.001)
        b3b = MDM_ST(2, 1, 3, self.small_config(gate=True)).eval()
        incompatible = b3b.load_state_dict(b3a.state_dict(), strict=False)
        self.assertEqual(
            incompatible.missing_keys,
            ["dit.material_state_exchange.gate_logits"],
        )

        inputs = self.small_inputs()
        with torch.no_grad():
            expected = b3a(**inputs)
            actual = b3b(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)

    def test_gate_requires_material_state_adapter(self):
        with self.assertRaisesRegex(ValueError, "material-state adapter"):
            MDM_ST(2, 1, 3, self.small_config(adapter=False, gate=True))


if __name__ == "__main__":
    unittest.main()
