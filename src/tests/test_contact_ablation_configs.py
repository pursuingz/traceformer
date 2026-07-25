import copy
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from model.spacetime import MDM_ST
from options import (
    TestingConfig,
    TrainingConfig,
    resolve_training_stop_step,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

TRAIN_ARMS = {
    "config_mm3_contact_no_velocity.yaml": [1.0, 0.0, 1.0],
    "config_mm3_contact_no_proximity.yaml": [1.0, 1.0, 0.0],
    "config_mm3_contact_no_gap.yaml": [0.0, 1.0, 1.0],
}

EVAL_ARMS = {
    "eval_mm3_contact_no_velocity_45k.yaml": (
        [1.0, 0.0, 1.0],
        "outputs/mm3_contact_no_velocity_8L/checkpoint-45000/model.safetensors",
    ),
    "eval_mm3_contact_no_proximity_45k.yaml": (
        [1.0, 1.0, 0.0],
        "outputs/mm3_contact_no_proximity_8L/checkpoint-45000/model.safetensors",
    ),
    "eval_mm3_contact_no_gap_45k.yaml": (
        [0.0, 1.0, 1.0],
        "outputs/mm3_contact_no_gap_8L/checkpoint-45000/model.safetensors",
    ),
}


def _load_structured(name, schema):
    return OmegaConf.merge(
        OmegaConf.structured(schema),
        OmegaConf.load(CONFIG_DIR / name),
    )


def _without_paths(cfg, paths):
    data = copy.deepcopy(OmegaConf.to_container(cfg, resolve=False))
    for path in paths:
        parts = path.split(".")
        cursor = data
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor.pop(parts[-1], None)
    return data


class ContactAblationConfigTests(unittest.TestCase):
    def test_screening_stop_preserves_scheduler_horizon_and_checkpoint(self):
        self.assertEqual(resolve_training_stop_step(90000, None, 2500), 90000)
        self.assertEqual(resolve_training_stop_step(90000, 45000, 2500), 45000)
        with self.assertRaisesRegex(ValueError, "must be in"):
            resolve_training_stop_step(90000, 0, 2500)
        with self.assertRaisesRegex(ValueError, "align"):
            resolve_training_stop_step(90000, 45001, 2500)

    def test_training_arms_change_only_declared_screening_fields(self):
        baseline = _load_structured(
            "config_mm3_contact_cond.yaml",
            TrainingConfig,
        )
        ignored = {
            "output_dir",
            "stop_after_steps",
            "model_config.contact_feature_mask",
        }

        for filename, expected_mask in TRAIN_ARMS.items():
            with self.subTest(filename=filename):
                arm = _load_structured(filename, TrainingConfig)
                self.assertEqual(arm.max_train_steps, 90000)
                self.assertEqual(arm.stop_after_steps, 45000)
                self.assertEqual(
                    list(arm.model_config.contact_feature_mask),
                    expected_mask,
                )
                self.assertEqual(
                    _without_paths(arm, ignored),
                    _without_paths(baseline, ignored),
                )

                arm.model_config.cond_frames = arm.get("input_frames", 5)
                model = MDM_ST(
                    8,
                    arm.output_frames,
                    n_feats=3,
                    model_config=arm.model_config,
                )
                self.assertEqual(model.contact_encoder.in_features, 3)
                self.assertEqual(model.contact_encoder.out_features, 256)

    def test_eval_arms_point_to_matching_45k_checkpoints(self):
        baseline = OmegaConf.load(CONFIG_DIR / "eval_mm3_contact_cond.yaml")
        ignored = {
            "resume",
            "vis_dir",
            "model_config.contact_feature_mask",
        }
        for filename, (expected_mask, expected_resume) in EVAL_ARMS.items():
            with self.subTest(filename=filename):
                cfg = _load_structured(filename, TestingConfig)
                self.assertEqual(
                    list(cfg.model_config.contact_feature_mask),
                    expected_mask,
                )
                self.assertEqual(cfg.resume, expected_resume)
                plain = OmegaConf.load(CONFIG_DIR / filename)
                self.assertEqual(
                    _without_paths(plain, ignored),
                    _without_paths(baseline, ignored),
                )

    def test_anchor_evals_use_matched_45k_checkpoints(self):
        contact = _load_structured(
            "eval_mm3_contact_cond_45k.yaml",
            TestingConfig,
        )
        baseline = _load_structured(
            "eval_mm3_baseline_45k.yaml",
            TestingConfig,
        )

        self.assertEqual(
            contact.resume,
            "outputs/mm3_contact_cond_8L/checkpoint-45000/model.safetensors",
        )
        self.assertEqual(
            baseline.resume,
            "outputs/mm3_singleframe_geom_deform_d0001_8L/"
            "checkpoint-45000/model.safetensors",
        )
        self.assertEqual(
            _without_paths(
                OmegaConf.load(CONFIG_DIR / "eval_mm3_contact_cond_45k.yaml"),
                {"resume", "vis_dir"},
            ),
            _without_paths(
                OmegaConf.load(CONFIG_DIR / "eval_mm3_contact_cond.yaml"),
                {"resume", "vis_dir"},
            ),
        )
        self.assertEqual(
            _without_paths(
                OmegaConf.load(CONFIG_DIR / "eval_mm3_baseline_45k.yaml"),
                {"resume", "vis_dir"},
            ),
            _without_paths(
                OmegaConf.load(
                    CONFIG_DIR / "eval_mm3_singleframe_geom_deform_d0001.yaml"
                ),
                {"resume", "vis_dir"},
            ),
        )

    def test_vxyz_arm_changes_only_velocity_representation_and_screening_fields(self):
        baseline = _load_structured(
            "config_mm3_contact_cond.yaml",
            TrainingConfig,
        )
        arm = _load_structured(
            "config_mm3_contact_vxyz.yaml",
            TrainingConfig,
        )
        ignored = {
            "output_dir",
            "stop_after_steps",
            "model_config.contact_velocity_mode",
        }

        self.assertEqual(arm.max_train_steps, 90000)
        self.assertEqual(arm.stop_after_steps, 45000)
        self.assertEqual(arm.model_config.contact_velocity_mode, "xyz")
        self.assertEqual(
            _without_paths(arm, ignored),
            _without_paths(baseline, ignored),
        )

        arm.model_config.cond_frames = arm.get("input_frames", 5)
        model = MDM_ST(
            8,
            arm.output_frames,
            n_feats=3,
            model_config=arm.model_config,
        )
        self.assertEqual(model.contact_encoder.in_features, 5)
        self.assertEqual(model.contact_encoder.out_features, 256)

    def test_vxyz_eval_mirrors_training_model_and_45k_checkpoint(self):
        baseline = OmegaConf.load(CONFIG_DIR / "eval_mm3_contact_cond_45k.yaml")
        cfg = _load_structured(
            "eval_mm3_contact_vxyz_45k.yaml",
            TestingConfig,
        )
        plain = OmegaConf.load(CONFIG_DIR / "eval_mm3_contact_vxyz_45k.yaml")
        ignored = {
            "resume",
            "vis_dir",
            "model_config.contact_velocity_mode",
        }

        self.assertEqual(cfg.model_config.contact_velocity_mode, "xyz")
        self.assertEqual(
            cfg.resume,
            "outputs/mm3_contact_vxyz_8L/checkpoint-45000/model.safetensors",
        )
        self.assertEqual(
            _without_paths(plain, ignored),
            _without_paths(baseline, ignored),
        )


if __name__ == "__main__":
    unittest.main()
