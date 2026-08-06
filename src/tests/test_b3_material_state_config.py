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

    def test_eval_configs_mirror_training_model_and_use_registered_checkpoints(self):
        train = self.load("config_mm3_b3a_material_state_adapter.yaml")
        for step in (45000, 90000):
            with self.subTest(step=step):
                evaluate = self.load(
                    f"eval_mm3_b3a_material_state_adapter_{step // 1000}k.yaml"
                )
                self.assertEqual(
                    OmegaConf.to_container(train.model_config, resolve=True),
                    OmegaConf.to_container(evaluate.model_config, resolve=True),
                )
                self.assertTrue(
                    evaluate.resume.endswith(
                        f"checkpoint-{step}/model.safetensors"
                    )
                )
                self.assertEqual(
                    evaluate.train_dataset.dataset_path, "mm3_data/mm3_test"
                )
                self.assertEqual(evaluate.train_dataset.input_frames, 5)
                self.assertEqual(evaluate.train_dataset.output_frames, 1)

    def test_off_eval_changes_only_runtime_and_output_labels(self):
        for label in ("45k", "90k"):
            with self.subTest(label=label):
                enabled = flatten(
                    self.load(f"eval_mm3_b3a_material_state_adapter_{label}.yaml")
                )
                disabled = flatten(
                    self.load(
                        f"eval_mm3_b3a_material_state_adapter_{label}_off.yaml"
                    )
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
                self.assertEqual(
                    disabled["model_config.material_state_runtime_scale"], 0.0
                )
                self.assertEqual(disabled["resume"], enabled["resume"])

    def test_adapter_diagnostic_parser_accepts_explicit_90k_profile(self):
        from src.diagnose_material_state_adapter import build_parser

        parsed = build_parser().parse_args(
            [
                "--profile",
                "b3a90",
                "--config",
                "configs/eval_mm3_b3a_material_state_adapter_90k.yaml",
                "--checkpoint",
                "outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors",
                "--output",
                "results/b3a90_adapter",
            ]
        )
        self.assertEqual(parsed.profile, "b3a90")

    def test_b3b_screen_changes_only_material_stage_gate_model_fields(self):
        b3a = flatten(
            self.load("config_mm3_b3a_material_state_adapter.yaml").model_config
        )
        b3b_config = self.load("config_mm3_b3b_material_stage_gate_screen.yaml")
        b3b = flatten(b3b_config.model_config)
        differing = {
            key
            for key in set(b3a) | set(b3b)
            if b3a.get(key) != b3b.get(key)
        }
        self.assertEqual(
            differing,
            {"material_stage_gate", "material_stage_gate_max"},
        )
        self.assertEqual(b3b_config.base_checkpoint,
            "outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors")
        self.assertEqual(b3b_config.train_dataset.dataset_path, "mm3_data/mm3_train")
        self.assertEqual(b3b_config.gate_batch_size, 1)
        self.assertEqual(b3b_config.gate_start0_probability, 0.5)
        self.assertEqual(b3b_config.gate_max_rollout_steps, 20)
        self.assertEqual(b3b_config.gate_updates_per_material, 200)

    def test_b3b_eval_mirrors_screen_model_and_uses_dev_test_once(self):
        train = self.load("config_mm3_b3b_material_stage_gate_screen.yaml")
        evaluate = self.load("eval_mm3_b3b_material_stage_gate_screen.yaml")
        self.assertEqual(
            OmegaConf.to_container(train.model_config, resolve=True),
            OmegaConf.to_container(evaluate.model_config, resolve=True),
        )
        self.assertEqual(evaluate.train_dataset.dataset_path, "mm3_data/mm3_test")
        self.assertEqual(evaluate.train_dataset.input_frames, 5)
        self.assertEqual(evaluate.train_dataset.output_frames, 1)
        self.assertFalse(evaluate.use_diffusion)
        self.assertEqual(evaluate.num_inference_steps, 1)
        self.assertEqual(
            evaluate.resume,
            "outputs/mm3_b3b_material_stage_gate_screen/checkpoint-best/model.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
