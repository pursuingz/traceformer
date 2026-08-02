import copy
import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from options import TestingConfig, TrainingConfig


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


if __name__ == "__main__":
    unittest.main()
