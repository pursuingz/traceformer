import unittest

import torch
from omegaconf import OmegaConf

from model.material_state import FactorizedMaterialStateAdapter
from model.spacetime import MDM_ST
from count_params import build_mm3


class FactorizedMaterialStateAdapterTest(unittest.TestCase):
    @staticmethod
    def make_adapter():
        return FactorizedMaterialStateAdapter(
            particle_dim=256,
            rank=64,
            num_materials=4,
            num_stages=4,
        )

    def test_rejects_invalid_constructor_values(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            FactorizedMaterialStateAdapter(256, 0, 4, 4)
        with self.assertRaisesRegex(ValueError, "e_scale"):
            FactorizedMaterialStateAdapter(256, 64, 4, 4, e_scale=0.0)
        with self.assertRaisesRegex(ValueError, "nu_scale"):
            FactorizedMaterialStateAdapter(256, 64, 4, 4, nu_scale=0.0)

    def test_rejects_bad_labels_and_stage(self):
        adapter = self.make_adapter()
        hidden = torch.randn(2, 6, 8, 256)
        values = torch.tensor([[5.0, 0.2], [6.0, 0.4]])
        with self.assertRaisesRegex(ValueError, "material_labels"):
            adapter(hidden, values, torch.tensor([0, 4]), stage_index=0)
        with self.assertRaisesRegex(IndexError, "stage_index"):
            adapter(hidden, values, torch.tensor([0, 2]), stage_index=4)

    def test_zero_initialized_output_is_exact_identity(self):
        torch.manual_seed(0)
        adapter = self.make_adapter().eval()
        hidden = torch.randn(2, 6, 8, 256)
        values = torch.tensor([[4.5, 0.1], [6.5, 0.4]])
        labels = torch.tensor([0, 2])

        output = adapter(hidden, values, labels, stage_index=0)

        self.assertTrue(torch.equal(output, hidden))
        self.assertNotEqual(output.data_ptr(), hidden.data_ptr())


class MaterialStateIntegrationTest(unittest.TestCase):
    make_adapter = staticmethod(FactorizedMaterialStateAdapterTest.make_adapter)

    @staticmethod
    def small_config(enabled=True):
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
                "material_state_adapter": enabled,
                "material_state_rank": 16,
                "material_state_interval": 1,
            }
        )

    @staticmethod
    def small_inputs(batch_size=1, point_count=2):
        return {
            "x": torch.randn(batch_size, 1, point_count, 3),
            "timesteps": torch.zeros(batch_size, dtype=torch.long),
            "init_pc": torch.randn(batch_size, 5, point_count, 3),
            "force": torch.randn(batch_size, 3),
            "E": torch.tensor([[5.5]]).repeat(batch_size, 1),
            "nu": torch.tensor([[0.25]]).repeat(batch_size, 1),
            "drag_mask": torch.zeros(batch_size, 1, point_count, 1),
            "drag_point": torch.zeros(batch_size, 4),
            "floor_height": None,
            "y": torch.tensor([1] * batch_size),
        }

    def test_requires_layers_divisible_by_interval(self):
        with self.assertRaisesRegex(ValueError, "material_state_interval"):
            build_mm3(
                "SpatialTemporalTransformerBlock",
                material_state_adapter=True,
                n_layers=8,
                material_state_interval=3,
            )

    def test_requires_frozen_frame_layout(self):
        with self.assertRaisesRegex(ValueError, "mask pseudo-frame"):
            build_mm3(
                "SpatialTemporalTransformerBlock",
                material_state_adapter=True,
                mask_cond=False,
            )
        with self.assertRaisesRegex(ValueError, "history"):
            build_mm3(
                "SpatialTemporalTransformerBlock",
                material_state_adapter=True,
                cond_frames=4,
            )

    def test_requires_four_material_classes(self):
        with self.assertRaisesRegex(ValueError, "num_materials"):
            build_mm3(
                "SpatialTemporalTransformerBlock",
                material_state_adapter=True,
                num_mat=3,
            )

    def test_disabled_baseline_has_no_material_state_adapter(self):
        model = build_mm3("SpatialTemporalTransformerBlock")
        self.assertFalse(hasattr(model.dit, "material_state_exchange"))

    def test_enabled_model_has_shared_four_stage_adapter(self):
        model = build_mm3(
            "SpatialTemporalTransformerBlock",
            material_state_adapter=True,
            material_state_rank=64,
            material_state_interval=2,
        )
        self.assertEqual(len(model.dit.transformer_blocks), 8)
        self.assertEqual(model.dit.material_state_exchange.num_stages, 4)
        self.assertEqual(model.dit.material_state_exchange.rank, 64)

    def test_shared_initialization_and_zero_init_forward_are_exact(self):
        torch.manual_seed(10)
        baseline = MDM_ST(2, 1, 3, self.small_config(False)).eval()
        torch.manual_seed(10)
        candidate = MDM_ST(2, 1, 3, self.small_config(True)).eval()

        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        common = set(baseline_state) & set(candidate_state)
        for name in common:
            self.assertTrue(
                torch.equal(baseline_state[name], candidate_state[name]), name
            )

        incompatible = candidate.load_state_dict(baseline_state, strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(
            all(
                key.startswith("dit.material_state_exchange.")
                for key in incompatible.missing_keys
            )
        )

        inputs = self.small_inputs()
        calls = []

        def record_call(module, args, kwargs):
            calls.append((kwargs["stage_index"], tuple(args[0].shape)))

        handle = candidate.dit.material_state_exchange.register_forward_pre_hook(
            record_call, with_kwargs=True
        )
        try:
            with torch.no_grad():
                baseline_output = baseline(**inputs)
                candidate_output = candidate(**inputs)
        finally:
            handle.remove()

        self.assertTrue(torch.equal(candidate_output, baseline_output))
        self.assertEqual(calls, [(0, (1, 6, 2, 64)), (1, (1, 6, 2, 64))])

    def test_active_adapter_backward_is_finite_with_checkpointing(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                torch.manual_seed(20 + int(checkpointing))
                model = MDM_ST(2, 1, 3, self.small_config(True)).train()
                if checkpointing:
                    model.enable_gradient_checkpointing()
                with torch.no_grad():
                    model.dit.material_state_exchange.output_proj.weight.fill_(1e-3)
                inputs = self.small_inputs()
                inputs["x"].requires_grad_(True)

                model(**inputs).square().mean().backward()

                adapter = model.dit.material_state_exchange
                for parameter in (
                    adapter.state_proj.weight,
                    adapter.material_proj.weight,
                    adapter.output_proj.weight,
                    adapter.stage_scales,
                ):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_nonzero_projection_depends_on_state_and_material(self):
        torch.manual_seed(1)
        adapter = self.make_adapter().eval()
        with torch.no_grad():
            adapter.output_proj.weight.fill_(0.01)
        hidden = torch.randn(2, 6, 8, 256)
        values = torch.tensor([[4.5, 0.1], [6.5, 0.4]])
        labels = torch.tensor([0, 2])

        delta = adapter(hidden, values, labels, stage_index=1) - hidden

        self.assertFalse(torch.equal(delta[0], delta[1]))
        self.assertFalse(torch.equal(delta[:, :, 0], delta[:, :, 1]))

    def test_exact_parameter_budget(self):
        adapter = self.make_adapter()
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 34_052)

    def test_zero_init_first_backward_trains_output_projection(self):
        adapter = self.make_adapter()
        hidden = torch.randn(2, 6, 8, 256, requires_grad=True)
        values = torch.tensor([[5.0, 0.2], [6.0, 0.3]])
        labels = torch.tensor([0, 1])

        adapter(hidden, values, labels, 0).square().mean().backward()

        self.assertGreater(adapter.output_proj.weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_runtime_zero_short_circuits_to_exact_clone(self):
        adapter = self.make_adapter()
        with torch.no_grad():
            adapter.output_proj.weight.fill_(float("nan"))
        hidden = torch.randn(1, 6, 4, 256)
        values = torch.tensor([[5.5, 0.25]])

        output = adapter(
            hidden,
            values,
            torch.tensor([1]),
            stage_index=0,
            runtime_scale=0.0,
        )

        self.assertTrue(torch.equal(output, hidden))
        self.assertNotEqual(output.data_ptr(), hidden.data_ptr())


if __name__ == "__main__":
    unittest.main()
