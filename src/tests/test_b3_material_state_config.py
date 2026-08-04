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


class B3MaterialStateConfigTest(unittest.TestCase):
    def load(self, name):
        path = CONFIG_DIR / name
        self.assertTrue(path.exists(), path)
        return OmegaConf.load(path)

    def test_train_config_changes_only_b3_fields(self):
        baseline = flatten(self.load("config_mm3_contact_cond.yaml"))
        candidate = flatten(
            self.load("config_mm3_b3a_material_state_adapter.yaml")
        )
        differing = {
            key
            for key in set(baseline) | set(candidate)
            if baseline.get(key) != candidate.get(key)
        }
        self.assertEqual(
            differing,
            {
                "output_dir",
                "model_config.material_state_adapter",
                "model_config.material_state_rank",
                "model_config.material_state_interval",
                "model_config.material_state_e_center",
                "model_config.material_state_e_scale",
                "model_config.material_state_nu_center",
                "model_config.material_state_nu_scale",
                "model_config.material_state_runtime_scale",
            },
        )
        self.assertEqual(
            candidate["model_config.transformer_block"],
            "SpatialTemporalTransformerBlock",
        )
        self.assertEqual(candidate["model_config.n_layers"], 8)

    def test_eval_mirrors_training_model_and_uses_45k_checkpoint(self):
        train = self.load("config_mm3_b3a_material_state_adapter.yaml")
        evaluate = self.load("eval_mm3_b3a_material_state_adapter_45k.yaml")
        self.assertEqual(
            OmegaConf.to_container(train.model_config, resolve=True),
            OmegaConf.to_container(evaluate.model_config, resolve=True),
        )
        self.assertTrue(evaluate.resume.endswith("checkpoint-45000/model.safetensors"))
        self.assertEqual(evaluate.train_dataset.dataset_path, "mm3_data/mm3_test")
        self.assertEqual(evaluate.train_dataset.input_frames, 5)
        self.assertEqual(evaluate.train_dataset.output_frames, 1)

    def test_off_eval_changes_only_runtime_and_output_labels(self):
        enabled = flatten(
            self.load("eval_mm3_b3a_material_state_adapter_45k.yaml")
        )
        disabled = flatten(
            self.load("eval_mm3_b3a_material_state_adapter_45k_off.yaml")
        )
        differing = {
            key
            for key in set(enabled) | set(disabled)
            if enabled.get(key) != disabled.get(key)
        }
        self.assertEqual(
            differing,
            {"vis_dir", "model_config.material_state_runtime_scale"},
        )
        self.assertTrue(disabled["model_config.material_state_adapter"])
        self.assertEqual(disabled["model_config.material_state_runtime_scale"], 0.0)
        self.assertEqual(disabled["resume"], enabled["resume"])


if __name__ == "__main__":
    unittest.main()
