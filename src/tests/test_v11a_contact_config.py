import copy
import sys
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from options import TestingConfig, TrainingConfig
from count_params import build_mm3, validate_v11a_parameter_budget
from model.hybrid_state import HybridStateExchange
from model.spacetime import MDM_ST, SpatialTemporalTransformerBlock


CONFIG_DIR = SRC_DIR / "configs"
TRAIN_BASE = CONFIG_DIR / "config_mm3_contact_cond.yaml"
TRAIN_ARM = CONFIG_DIR / "config_mm3_v11a_contact_cond_8L.yaml"
EVAL_BASE = CONFIG_DIR / "eval_mm3_contact_cond_45k.yaml"
EVAL_ARM = CONFIG_DIR / "eval_mm3_v11a_contact_cond_8L_45k.yaml"

HYBRID_CONFIG = {
    "hybrid_state_dim": 64,
    "hybrid_state_heads": 4,
    "hybrid_state_interval": 2,
}


def load_plain(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


class V11aContactConfigTests(unittest.TestCase):
    def assert_config_exists(self, path):
        if not path.is_file():
            raise AssertionError(f"missing config: {path}")

    def assert_isolated_arm(self, baseline_path, arm_path, allowed_top_level):
        self.assert_config_exists(arm_path)
        baseline = copy.deepcopy(load_plain(baseline_path))
        arm = copy.deepcopy(load_plain(arm_path))

        for key, expected in allowed_top_level.items():
            self.assertEqual(baseline.pop(key, None), expected[0])
            self.assertEqual(arm.pop(key, None), expected[1])

        baseline_model = baseline["model_config"]
        arm_model = arm["model_config"]
        self.assertEqual(
            baseline_model.pop("transformer_block"),
            "SpatialTemporalTransformerBlock",
        )
        self.assertEqual(
            arm_model.pop("transformer_block"),
            "SpatialTemporalTransformerBlockv11a",
        )
        for key, value in HYBRID_CONFIG.items():
            self.assertNotIn(key, baseline_model)
            self.assertEqual(arm_model.pop(key), value)

        self.assertEqual(arm, baseline)

    def test_training_arm_changes_only_registered_fields(self):
        self.assert_isolated_arm(
            TRAIN_BASE,
            TRAIN_ARM,
            {
                "output_dir": (
                    "./outputs/mm3_contact_cond_8L",
                    "./outputs/mm3_v11a_contact_cond_8L",
                ),
                "stop_after_steps": (None, 45000),
            },
        )

    def test_eval_arm_changes_only_architecture_and_artifact_paths(self):
        self.assert_isolated_arm(
            EVAL_BASE,
            EVAL_ARM,
            {
                "resume": (
                    "outputs/mm3_contact_cond_8L/checkpoint-45000/model.safetensors",
                    "outputs/mm3_v11a_contact_cond_8L/checkpoint-45000/model.safetensors",
                ),
                "vis_dir": (
                    "vis_results_mm3_contact_cond_45k",
                    "vis_results_mm3_v11a_contact_cond_8L_45k",
                ),
            },
        )

    def test_missing_config_is_rejected_before_loading(self):
        missing_path = CONFIG_DIR / "__missing_v11a_contact_config__.yaml"
        with self.assertRaisesRegex(AssertionError, r"^missing config:"):
            self.assert_config_exists(missing_path)

    def test_configs_merge_and_keep_screening_contract(self):
        self.assert_config_exists(TRAIN_ARM)
        self.assert_config_exists(EVAL_ARM)
        train = OmegaConf.merge(
            OmegaConf.structured(TrainingConfig), OmegaConf.load(TRAIN_ARM)
        )
        evaluation = OmegaConf.merge(
            OmegaConf.structured(TestingConfig), OmegaConf.load(EVAL_ARM)
        )

        self.assertEqual(train.max_train_steps, 90000)
        self.assertEqual(train.stop_after_steps, 45000)
        self.assertEqual(train.checkpointing_steps, 2500)
        self.assertEqual(train.seed, 0)
        self.assertEqual(train.model_config.n_layers, 8)
        self.assertEqual(train.model_config.latent_dim, 256)
        self.assertTrue(train.model_config.contact_particle_cond)
        self.assertEqual(train.model_config.contact_feature_sigma, 0.04)
        self.assertEqual(train.model_config.hybrid_state_interval, 2)
        self.assertFalse(evaluation.use_diffusion)
        self.assertEqual(evaluation.num_inference_steps, 1)
        self.assertEqual(evaluation.output_frames, 1)
        self.assertTrue(evaluation.model_config.contact_particle_cond)


class V11aContactIntegrationTests(unittest.TestCase):
    @staticmethod
    def build_from_config(transformer_block):
        cfg = OmegaConf.load(TRAIN_ARM)
        cfg.model_config.cond_frames = cfg.get("input_frames", 5)
        cfg.model_config.transformer_block = transformer_block
        if transformer_block == "SpatialTemporalTransformerBlock":
            for key in HYBRID_CONFIG:
                cfg.model_config.pop(key, None)
        return MDM_ST(
            n_points=8,
            n_frame=1,
            n_feats=3,
            model_config=cfg.model_config,
        )

    @staticmethod
    def inputs(batch_size=1, point_count=8):
        return {
            "x": torch.randn(batch_size, 1, point_count, 3),
            "timesteps": torch.zeros(batch_size, dtype=torch.long),
            "init_pc": torch.randn(batch_size, 5, point_count, 3),
            "force": torch.randn(batch_size, 3),
            "E": torch.full((batch_size, 1), 6.0),
            "nu": torch.full((batch_size, 1), 0.35),
            "drag_mask": torch.zeros(batch_size, 1, point_count, 1),
            "drag_point": torch.zeros(batch_size, 4),
            "floor_height": torch.full((batch_size, 1), -2.0),
            "gravity_label": torch.ones(batch_size, 1, dtype=torch.long),
            "y": torch.ones(batch_size, dtype=torch.long),
            "start_vel": torch.zeros(batch_size, point_count, 3),
        }

    def test_combined_model_keeps_original_serial_blocks_and_both_modules(self):
        model = self.build_from_config("SpatialTemporalTransformerBlockv11a")

        self.assertTrue(model.contact_particle_cond)
        self.assertEqual(model.contact_injection_mode, "separate")
        self.assertEqual(model.contact_encoder.in_features, 3)
        self.assertEqual(model.contact_encoder.out_features, 256)
        self.assertTrue(
            all(
                type(block) is SpatialTemporalTransformerBlock
                for block in model.dit.transformer_blocks
            )
        )
        exchanges = [
            module for module in model.modules()
            if isinstance(module, HybridStateExchange)
        ]
        self.assertEqual(exchanges, [model.dit.hybrid_state_exchange])
        self.assertEqual(model.dit.hybrid_state_interval, 2)

    def test_zero_gate_load_from_contact_anchor_preserves_output_bits(self):
        torch.manual_seed(101)
        contact = self.build_from_config("SpatialTemporalTransformerBlock").eval()
        torch.manual_seed(202)
        combined = self.build_from_config(
            "SpatialTemporalTransformerBlockv11a"
        ).eval()

        incompatible = combined.load_state_dict(contact.state_dict(), strict=False)
        expected_missing = {
            key for key in combined.state_dict()
            if key.startswith("dit.hybrid_state_exchange.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(
            torch.equal(
                combined.dit.hybrid_state_exchange.feedback_gates,
                torch.zeros(4),
            )
        )

        inputs = self.inputs()
        with torch.no_grad():
            contact_output = contact(**inputs)
            combined_output = combined(**inputs)
        torch.testing.assert_close(combined_output, contact_output, rtol=0, atol=0)

    def test_contact_encoder_and_exchange_gate_receive_gradients(self):
        torch.manual_seed(303)
        model = self.build_from_config(
            "SpatialTemporalTransformerBlockv11a"
        ).train()
        output = model(**self.inputs())
        output.square().mean().backward()

        contact_grad = model.contact_encoder.weight.grad
        gate_grad = model.dit.hybrid_state_exchange.feedback_gates.grad
        self.assertIsNotNone(contact_grad)
        self.assertIsNotNone(gate_grad)
        self.assertTrue(torch.isfinite(contact_grad).all())
        self.assertTrue(torch.isfinite(gate_grad).all())
        self.assertGreater(torch.count_nonzero(contact_grad).item(), 0)
        self.assertTrue(torch.all(gate_grad != 0))

    def test_combination_adds_only_the_existing_v11a_exchange_budget(self):
        contact = build_mm3(
            "SpatialTemporalTransformerBlock",
            contact_particle_cond=True,
            contact_feature_sigma=0.04,
        )
        combined = build_mm3(
            "SpatialTemporalTransformerBlockv11a",
            contact_particle_cond=True,
            contact_feature_sigma=0.04,
        )
        report = validate_v11a_parameter_budget(contact, combined)

        self.assertEqual(report["signed_delta"], 160_773)
        self.assertLess(report["delta_percent"], 1.0)
        self.assertEqual(report["block_count"], 8)
        self.assertEqual(report["exchange_count"], 1)
        self.assertEqual(report["exchange_calls"], 4)


if __name__ == "__main__":
    unittest.main()
