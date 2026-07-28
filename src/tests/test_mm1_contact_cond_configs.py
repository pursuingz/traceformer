import copy
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from model.spacetime import MDM_ST
from options import TrainingConfig


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_structured(name, schema):
    return OmegaConf.merge(
        OmegaConf.structured(schema),
        OmegaConf.load(CONFIG_DIR / name),
    )


def _without_paths(cfg, paths):
    data = copy.deepcopy(OmegaConf.to_container(cfg, resolve=False))
    for path in paths:
        cursor = data
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor.pop(parts[-1], None)
    return data


def _parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


class Mm1ContactCondConfigTests(unittest.TestCase):
    def test_training_arm_changes_only_contact_and_screening_fields(self):
        baseline = _load_structured(
            "config_diffE2048_singleframe_geom_deform_d0001.yaml",
            TrainingConfig,
        )
        arm = _load_structured(
            "config_mm1_contact_cond.yaml",
            TrainingConfig,
        )
        ignored = {
            "output_dir",
            "stop_after_steps",
            "model_config.contact_particle_cond",
            "model_config.contact_injection_mode",
            "model_config.contact_velocity_mode",
            "model_config.contact_feature_sigma",
        }

        self.assertEqual(arm.max_train_steps, 90000)
        self.assertEqual(arm.stop_after_steps, 45000)
        self.assertEqual(
            arm.train_dataset.dataset_path,
            "diff_E_2048_data/2048_data/2048_train",
        )
        self.assertTrue(arm.model_config.contact_particle_cond)
        self.assertEqual(arm.model_config.contact_injection_mode, "separate")
        self.assertEqual(arm.model_config.contact_velocity_mode, "vertical")
        self.assertEqual(arm.model_config.contact_feature_sigma, 0.04)
        self.assertFalse(arm.model_config.get("class_token", False))
        self.assertFalse(arm.model_config.get("gravity_emb", False))
        self.assertFalse(arm.get("geom_elastic_only", False))
        self.assertEqual(
            _without_paths(arm, ignored),
            _without_paths(baseline, ignored),
        )

        baseline.model_config.cond_frames = baseline.get("input_frames", 5)
        arm.model_config.cond_frames = arm.get("input_frames", 5)
        baseline_model = MDM_ST(
            8,
            baseline.output_frames,
            n_feats=3,
            model_config=baseline.model_config,
        )
        arm_model = MDM_ST(
            8,
            arm.output_frames,
            n_feats=3,
            model_config=arm.model_config,
        )
        self.assertEqual(
            _parameter_count(arm_model) - _parameter_count(baseline_model),
            1024,
        )
        self.assertEqual(arm_model.contact_encoder.in_features, 3)
        self.assertEqual(arm_model.contact_encoder.out_features, 256)


if __name__ == "__main__":
    unittest.main()
