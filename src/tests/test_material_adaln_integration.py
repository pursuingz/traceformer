import unittest

import torch
from omegaconf import OmegaConf

from model.spacetime import MDM_ST


def small_config(material_adaln=False, **overrides):
    config = {
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
        "material_state_adapter": False,
        "material_adaln_cond": material_adaln,
    }
    config.update(overrides)
    return OmegaConf.create(config)


def small_inputs(batch_size=1, point_count=2):
    return {
        "x": torch.full((batch_size, 1, point_count, 3), 0.125),
        "timesteps": torch.zeros(batch_size, dtype=torch.long),
        "init_pc": torch.full((batch_size, 5, point_count, 3), -0.25),
        "force": torch.full((batch_size, 3), 0.5),
        "E": torch.tensor([[5.5]]).repeat(batch_size, 1),
        "nu": torch.tensor([[0.25]]).repeat(batch_size, 1),
        "drag_mask": torch.zeros(batch_size, 1, point_count, 1),
        "drag_point": torch.zeros(batch_size, 4),
        "floor_height": None,
        "y": torch.tensor([1] * batch_size),
    }


class ContinuousMaterialAdaLNIntegrationTest(unittest.TestCase):
    def test_disabled_config_has_no_material_adaln_parameters(self):
        model = MDM_ST(2, 1, 3, small_config(material_adaln=False))

        self.assertFalse(
            any("material_conditioner" in key for key in model.state_dict())
        )

    def test_zero_initialized_b4_matches_baseline_forward(self):
        torch.manual_seed(17)
        baseline = MDM_ST(2, 1, 3, small_config(material_adaln=False)).eval()
        torch.manual_seed(17)
        candidate = MDM_ST(2, 1, 3, small_config(material_adaln=True)).eval()
        incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(
            all("material_conditioner" in key for key in incompatible.missing_keys)
        )
        inputs = small_inputs()
        with torch.no_grad():
            expected = baseline(**inputs)
            actual = candidate(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-7)

    def test_runtime_zero_disables_nonzero_conditioner(self):
        torch.manual_seed(23)
        baseline = MDM_ST(2, 1, 3, small_config(material_adaln=False)).eval()
        torch.manual_seed(23)
        model = MDM_ST(2, 1, 3, small_config(material_adaln=True)).eval()
        model.load_state_dict(baseline.state_dict(), strict=False)
        with torch.no_grad():
            model.dit.material_conditioner.output_proj.weight.fill_(0.01)
        inputs = small_inputs()
        model.dit.material_adaln_runtime_scale = 1.0
        enabled = model(**inputs)
        model.dit.material_adaln_runtime_scale = 0.0
        disabled = model(**inputs)
        self.assertFalse(torch.equal(enabled, disabled))
        torch.testing.assert_close(
            disabled, baseline(**inputs), rtol=0.0, atol=1e-7
        )

    def test_rejects_b3_and_b4_combination(self):
        config = small_config(material_adaln=True)
        config.material_state_adapter = True

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            MDM_ST(2, 1, 3, config)

    def test_active_b4_requires_material_values(self):
        model = MDM_ST(2, 1, 3, small_config(material_adaln=True)).eval()
        with self.assertRaisesRegex(
            ValueError, "continuous material AdaLN requires material_values"
        ):
            model.dit(
                hidden_states=torch.zeros(1, 7, 2, 64),
                encoder_hidden_states=torch.zeros(1, 5, 64),
                timestep=torch.zeros(1, dtype=torch.long),
                class_labels=torch.ones(1, dtype=torch.long),
            )

    def test_active_b4_accepts_continuous_conditions_without_mat_type(self):
        torch.manual_seed(29)
        model = MDM_ST(
            2,
            1,
            3,
            small_config(material_adaln=True, num_mat=0, class_token=False),
        ).eval()
        with torch.no_grad():
            model.dit.material_conditioner.output_proj.weight.fill_(0.01)

        inputs = small_inputs()
        inputs.pop("y")
        changed_e = {key: value.clone() if torch.is_tensor(value) else value for key, value in inputs.items()}
        changed_e["E"] = torch.tensor([[6.5]])
        changed_nu = {key: value.clone() if torch.is_tensor(value) else value for key, value in inputs.items()}
        changed_nu["nu"] = torch.tensor([[0.40]])

        with torch.no_grad():
            baseline = model(**inputs)
            e_output = model(**changed_e)
            nu_output = model(**changed_nu)

        self.assertFalse(torch.equal(e_output, baseline))
        self.assertFalse(torch.equal(nu_output, baseline))

    def test_b4_initialization_preserves_all_shared_state_tensors(self):
        torch.manual_seed(31)
        baseline = MDM_ST(2, 1, 3, small_config(material_adaln=False))
        torch.manual_seed(31)
        candidate = MDM_ST(2, 1, 3, small_config(material_adaln=True))

        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        common = set(baseline_state) & set(candidate_state)
        self.assertTrue(common)
        for name in common:
            self.assertTrue(torch.equal(candidate_state[name], baseline_state[name]), name)


if __name__ == "__main__":
    unittest.main()
