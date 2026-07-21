import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from options import TestingConfig, TrainingConfig


CONFIG_DIR = SRC_DIR / "configs"
TRAIN_BASELINE = CONFIG_DIR / "config_mm3_singleframe_geom_deform_d0001.yaml"
TRAIN_CANDIDATE = CONFIG_DIR / "config_mm3_v11a_mc_hst_8L.yaml"
EVAL_BASELINE = CONFIG_DIR / "eval_mm3_singleframe_geom_deform_d0001.yaml"
EVAL_CANDIDATE = CONFIG_DIR / "eval_mm3_v11a_mc_hst_8L.yaml"

HYBRID_CONFIG = {
    "hybrid_state_dim": 64,
    "hybrid_state_heads": 4,
    "hybrid_state_interval": 2,
}


def load_plain_yaml(path):
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


class V11aConfigIsolationTests(unittest.TestCase):
    def assert_isolated_candidate(
        self,
        baseline_path,
        candidate_path,
        allowed_top_level_changes,
    ):
        baseline = load_plain_yaml(baseline_path)
        candidate = load_plain_yaml(candidate_path)

        for key, (baseline_value, candidate_value) in allowed_top_level_changes.items():
            self.assertEqual(baseline.pop(key), baseline_value)
            self.assertEqual(candidate.pop(key), candidate_value)

        baseline_model = baseline["model_config"]
        candidate_model = candidate["model_config"]
        self.assertEqual(
            baseline_model.pop("transformer_block"),
            "SpatialTemporalTransformerBlock",
        )
        self.assertEqual(
            candidate_model.pop("transformer_block"),
            "SpatialTemporalTransformerBlockv11a",
        )
        for key, value in HYBRID_CONFIG.items():
            self.assertNotIn(key, baseline_model)
            self.assertEqual(candidate_model.pop(key), value)

        self.assertEqual(candidate, baseline)

    def test_train_candidate_changes_only_v11a_architecture_and_output_dir(self):
        self.assert_isolated_candidate(
            TRAIN_BASELINE,
            TRAIN_CANDIDATE,
            {
                "output_dir": (
                    "./outputs/mm3_singleframe_geom_deform_d0001_8L",
                    "./outputs/mm3_v11a_mc_hst_8L",
                ),
            },
        )

    def test_eval_candidate_changes_only_v11a_architecture_and_artifact_paths(self):
        self.assert_isolated_candidate(
            EVAL_BASELINE,
            EVAL_CANDIDATE,
            {
                "resume": (
                    "outputs/mm3_singleframe_geom_deform_d0001_8L/checkpoint-90000/model.safetensors",
                    "outputs/mm3_v11a_mc_hst_8L/checkpoint-90000/model.safetensors",
                ),
                "vis_dir": (
                    "vis_results_mm3_singleframe_geom_deform_d0001",
                    "vis_results_mm3_v11a_mc_hst_8L",
                ),
            },
        )

    def test_candidates_merge_with_structured_configs(self):
        train = OmegaConf.merge(
            OmegaConf.structured(TrainingConfig), OmegaConf.load(TRAIN_CANDIDATE)
        )
        evaluation = OmegaConf.merge(
            OmegaConf.structured(TestingConfig), OmegaConf.load(EVAL_CANDIDATE)
        )

        self.assertEqual(
            train.model_config.transformer_block,
            "SpatialTemporalTransformerBlockv11a",
        )
        self.assertEqual(evaluation.model_config.hybrid_state_interval, 2)

    def test_frozen_training_and_evaluation_fields_remain_explicit(self):
        train = OmegaConf.load(TRAIN_CANDIDATE)
        evaluation = OmegaConf.load(EVAL_CANDIDATE)

        self.assertEqual(train.output_frames, 1)
        self.assertEqual(train.train_dataset.dataset_path, "mm3_data/mm3_train")
        self.assertEqual(train.seed, 0)
        self.assertEqual(train.train_dataset.seed, 0)
        self.assertTrue(train.rollout_random_window)
        self.assertEqual(train.windows_per_model, 20)
        self.assertEqual(train.max_train_steps, 90000)
        self.assertEqual(train.train_batch_size, 1)
        self.assertEqual(train.gradient_accumulation_steps, 8)
        self.assertEqual(train.learning_rate, 1e-4)
        self.assertEqual(
            {
                "lambda_vel": train.lambda_vel,
                "lambda_mask": train.lambda_mask,
                "lambda_momentum": train.lambda_momentum,
                "lambda_deform": train.lambda_deform,
                "lambda_laplacian": train.lambda_laplacian,
                "lambda_collision": train.lambda_collision,
                "lambda_edge": train.lambda_edge,
                "lambda_floor": train.lambda_floor,
            },
            {
                "lambda_vel": 1.0,
                "lambda_mask": 0.0,
                "lambda_momentum": 0.0,
                "lambda_deform": 0.001,
                "lambda_laplacian": 0.5,
                "lambda_collision": 0.1,
                "lambda_edge": 1.0,
                "lambda_floor": 0.1,
            },
        )

        self.assertEqual(evaluation.output_frames, 1)
        self.assertEqual(evaluation.train_dataset.dataset_path, "mm3_data/mm3_test")
        self.assertEqual(evaluation.seed, 0)
        self.assertEqual(evaluation.train_dataset.seed, 0)
        self.assertFalse(evaluation.use_diffusion)
        self.assertEqual(evaluation.num_inference_steps, 1)


if __name__ == "__main__":
    unittest.main()
