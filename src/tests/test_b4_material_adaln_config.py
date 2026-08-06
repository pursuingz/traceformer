import unittest
from pathlib import Path

from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def flatten(value, prefix=""):
    plain = OmegaConf.to_container(value, resolve=False)
    result = {}
    for key, item in plain.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(flatten(OmegaConf.create(item), path))
        else:
            result[path] = item
    return result


class B4MaterialAdaLNConfigTest(unittest.TestCase):
    def load(self, name):
        path = CONFIG_DIR / name
        self.assertTrue(path.exists(), path)
        return OmegaConf.load(path)

    def test_training_config_changes_only_declared_b4_fields(self):
        baseline = flatten(self.load("config_mm3_contact_cond.yaml"))
        candidate = self.load("config_mm3_b4_material_adaln.yaml")
        candidate_flat = flatten(candidate)
        differing = {
            key
            for key in set(baseline) | set(candidate_flat)
            if baseline.get(key) != candidate_flat.get(key)
        }
        self.assertEqual(
            differing,
            {
                "output_dir",
                "stop_after_steps",
                "model_config.material_adaln_cond",
                "model_config.material_adaln_hidden_dim",
                "model_config.material_adaln_e_center",
                "model_config.material_adaln_e_scale",
                "model_config.material_adaln_nu_center",
                "model_config.material_adaln_nu_scale",
                "model_config.material_adaln_runtime_scale",
            },
        )
        self.assertEqual(candidate.max_train_steps, 90000)
        self.assertEqual(candidate.stop_after_steps, 45000)
        self.assertEqual(candidate.model_config.n_layers, 8)
        self.assertEqual(
            candidate.model_config.transformer_block,
            "SpatialTemporalTransformerBlock",
        )

    def test_on_eval_mirrors_training_model_and_uses_test_45k_checkpoint(self):
        train = self.load("config_mm3_b4_material_adaln.yaml")
        evaluate = self.load("eval_mm3_b4_material_adaln_45k.yaml")
        self.assertEqual(
            OmegaConf.to_container(train.model_config, resolve=True),
            OmegaConf.to_container(evaluate.model_config, resolve=True),
        )
        self.assertEqual(
            evaluate.resume,
            "outputs/mm3_b4_material_adaln_8L/"
            "checkpoint-45000/model.safetensors",
        )
        self.assertEqual(evaluate.train_dataset.dataset_path, "mm3_data/mm3_test")
        self.assertEqual(evaluate.train_dataset.input_frames, 5)
        self.assertEqual(evaluate.train_dataset.output_frames, 1)
        self.assertFalse(evaluate.use_diffusion)
        self.assertEqual(evaluate.num_inference_steps, 1)

    def test_off_eval_changes_only_runtime_scale_and_vis_dir(self):
        enabled = flatten(self.load("eval_mm3_b4_material_adaln_45k.yaml"))
        disabled = flatten(
            self.load("eval_mm3_b4_material_adaln_45k_off.yaml")
        )
        differing = {
            key
            for key in set(enabled) | set(disabled)
            if enabled.get(key) != disabled.get(key)
        }
        self.assertEqual(
            differing,
            {"vis_dir", "model_config.material_adaln_runtime_scale"},
        )
        self.assertEqual(
            disabled["model_config.material_adaln_runtime_scale"], 0.0
        )
        self.assertEqual(disabled["resume"], enabled["resume"])


if __name__ == "__main__":
    unittest.main()
