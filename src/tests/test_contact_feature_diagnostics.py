import math
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

import contact_feature_diagnostics
from contact_feature_diagnostics import (
    collect_grouped_features,
    load_contact_projection,
)
from model.contact_adapter import FactorizedContactAdapter


class ContactFeatureDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _factorized_state():
        return {
            "model.contact_adapter.boundary_encoder.weight": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]]
            ),
            "model.contact_adapter.normal_encoder.weight": torch.tensor(
                [[5.0], [6.0]]
            ),
            "model.contact_adapter.tangential_encoder.weight": torch.tensor(
                [[7.0, 8.0], [9.0, 10.0]]
            ),
            "model.contact_adapter.shared_bias": torch.tensor([11.0, 12.0]),
            "model.contact_adapter.tangential_gate": torch.atanh(
                torch.tensor(0.5)
            ),
        }

    def test_loads_separate_contact_projection_and_bias(self):
        state = {
            "model.contact_encoder.weight": torch.arange(
                12, dtype=torch.float32
            ).reshape(4, 3),
            "model.contact_encoder.bias": torch.arange(
                4, dtype=torch.float32
            ),
        }

        weight, bias = load_contact_projection(
            state,
            injection_mode="separate",
            feature_dim=3,
        )

        torch.testing.assert_close(
            weight,
            state["model.contact_encoder.weight"],
        )
        torch.testing.assert_close(
            bias,
            state["model.contact_encoder.bias"],
        )

    def test_loads_shared_contact_columns_without_shared_bias(self):
        full_weight = torch.arange(
            4 * 102, dtype=torch.float32
        ).reshape(4, 102)
        state = {
            "model.input_encoder.mlp.weight": full_weight,
            "model.input_encoder.mlp.bias": torch.ones(4),
        }

        weight, bias = load_contact_projection(
            state,
            injection_mode="shared",
            feature_dim=3,
        )

        torch.testing.assert_close(weight, full_weight[:, 99:102])
        self.assertIsNone(bias)

    def test_reconstructs_factorized_projection_and_shared_bias(self):
        state = self._factorized_state()

        weight, bias = load_contact_projection(
            state,
            injection_mode="factorized",
            feature_dim=5,
        )

        torch.testing.assert_close(
            weight,
            torch.tensor(
                [
                    [1.0, 3.5, 5.0, 4.0, 2.0],
                    [3.0, 4.5, 6.0, 5.0, 4.0],
                ]
            ),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            bias,
            state["model.contact_adapter.shared_bias"],
            rtol=0,
            atol=0,
        )

    def test_loads_projection_from_real_factorized_adapter_state(self):
        adapter = FactorizedContactAdapter(latent_dim=2)
        state = {
            f"model.contact_adapter.{key}": value
            for key, value in adapter.state_dict().items()
        }

        weight, bias = load_contact_projection(
            state,
            injection_mode="factorized",
            feature_dim=5,
        )

        torch.testing.assert_close(weight, torch.zeros(2, 5))
        torch.testing.assert_close(bias, torch.zeros(2))

    def test_factorized_projection_requires_five_features(self):
        with self.assertRaisesRegex(
            ValueError,
            "factorized.*feature_dim=5",
        ):
            load_contact_projection(
                {},
                injection_mode="factorized",
                feature_dim=3,
            )

    def test_invalid_injection_mode_lists_factorized(self):
        with self.assertRaisesRegex(ValueError, "factorized"):
            load_contact_projection(
                {},
                injection_mode="unknown",
                feature_dim=5,
            )

    def test_loads_factorized_gate_and_parameter_norms(self):
        stats = contact_feature_diagnostics.load_factorized_contact_stats(
            self._factorized_state()
        )

        self.assertAlmostEqual(stats["effective_gate"], 0.5)
        self.assertAlmostEqual(
            stats["boundary_weight_norm"],
            math.sqrt(30.0),
            places=5,
        )
        self.assertAlmostEqual(
            stats["normal_weight_norm"],
            math.sqrt(61.0),
            places=5,
        )
        self.assertAlmostEqual(
            stats["tangential_weight_norm"],
            math.sqrt(294.0),
            places=5,
        )
        self.assertAlmostEqual(
            stats["shared_bias_norm"],
            math.sqrt(265.0),
            places=5,
        )

    def test_computes_factorized_branch_hidden_norms(self):
        features = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        norms = contact_feature_diagnostics.factorized_branch_hidden_norms(
            features,
            self._factorized_state(),
        )

        self.assertAlmostEqual(
            norms["boundary"],
            (math.sqrt(650.0) + math.sqrt(320.0)) / 2.0,
            places=5,
        )
        self.assertAlmostEqual(
            norms["normal"],
            (math.sqrt(549.0) + math.sqrt(244.0)) / 2.0,
            places=5,
        )
        self.assertAlmostEqual(
            norms["tangential"],
            (math.sqrt(1370.0) + math.sqrt(620.5)) / 2.0,
            places=5,
        )

    def test_factorized_branch_hidden_norms_do_not_call_linear(self):
        state = self._factorized_state()
        state["model.contact_adapter.tangential_gate"] = torch.atanh(
            torch.tensor(-0.5)
        )
        features = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        with patch(
            "torch.nn.functional.linear",
            side_effect=AssertionError("F.linear must not be called"),
        ):
            norms = (
                contact_feature_diagnostics.factorized_branch_hidden_norms(
                    features,
                    state,
                )
            )

        self.assertAlmostEqual(
            norms["tangential"],
            (math.sqrt(1370.0) + math.sqrt(620.5)) / 2.0,
            places=5,
        )

    def test_factorized_branch_hidden_norms_require_token_matrix(self):
        state = self._factorized_state()
        invalid_features = (
            torch.zeros(5),
            torch.zeros(2, 4),
            torch.zeros(1, 2, 5),
        )

        for features in invalid_features:
            with self.subTest(shape=tuple(features.shape)):
                with self.assertRaisesRegex(
                    ValueError,
                    r"shape \(tokens, 5\)",
                ):
                    contact_feature_diagnostics.factorized_branch_hidden_norms(
                        features,
                        state,
                    )

    def test_main_prints_factorized_parameter_and_branch_stats(self):
        state = self._factorized_state()
        features = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
        cfg = OmegaConf.create(
            {
                "input_frames": 5,
                "output_frames": 1,
                "train_dataset": {},
                "eval_batch_size": 1,
                "dataloader_num_workers": 0,
                "resume": "factorized.safetensors",
                "model_config": {
                    "contact_feature_sigma": 0.04,
                    "contact_velocity_mode": "xyz",
                    "contact_injection_mode": "factorized",
                },
            }
        )
        output = StringIO()

        with (
            patch.object(contact_feature_diagnostics, "TrajDataset"),
            patch.object(
                contact_feature_diagnostics.torch.utils.data,
                "DataLoader",
                return_value=[],
            ),
            patch.object(
                contact_feature_diagnostics,
                "load_file",
                return_value=state,
            ),
            patch.object(
                contact_feature_diagnostics,
                "collect_grouped_features",
                return_value={0: [features]},
            ),
            redirect_stdout(output),
        ):
            contact_feature_diagnostics.main(cfg)

        text = output.getvalue()
        self.assertIn("factorized effective gate: 0.5", text)
        self.assertIn(
            "factorized parameter norms: "
            "boundary=5.47723, normal=7.81025, "
            "tangential=17.1464, shared_bias=16.2788",
            text,
        )
        self.assertIn(
            "factorized branch hidden norms: "
            "boundary=21.6918, normal=19.5256, tangential=30.9617",
            text,
        )

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

    def test_collects_xyz_displacement_features(self):
        points = torch.tensor(
            [[
                [[0.0, 0.1, 0.0]],
                [[0.2, 0.3, -0.4]],
            ]]
        )
        batch = {
            "points_src": points,
            "floor_height": torch.zeros(1, 1),
            "start_vel": torch.tensor([[[0.5, -0.2, 0.1]]]),
            "mat_type": torch.tensor([1]),
        }

        grouped = collect_grouped_features(
            [(batch, {})],
            sigma=0.04,
            velocity_mode="xyz",
        )

        self.assertEqual(grouped[1][0].shape, (2, 5))
        torch.testing.assert_close(
            grouped[1][0][0, 1:4],
            torch.tensor([0.5, -0.2, 0.1]),
        )
        torch.testing.assert_close(
            grouped[1][0][1, 1:4],
            torch.tensor([0.2, 0.2, -0.4]),
        )


if __name__ == "__main__":
    unittest.main()
