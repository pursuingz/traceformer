import unittest
import json
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

from model.material_state import FactorizedMaterialStateAdapter
from model.spacetime import MDM_ST
from utils.material_state_stage_diagnostics import (
    adapter_parameter_groups,
    mean_named_gradients,
    snapshot_adapter_gradients,
    summarize_material_gradient_conflict,
)


class MaterialStateGradientConflictTests(unittest.TestCase):
    @staticmethod
    def make_adapter():
        return FactorizedMaterialStateAdapter(
            particle_dim=4,
            rank=2,
            num_materials=3,
            num_stages=4,
        )

    def test_parameter_groups_cover_every_parameter_once_in_named_order(self):
        adapter = self.make_adapter()
        groups = adapter_parameter_groups(adapter)
        expected_groups = {
            "all_adapter",
            "state_norm",
            "state_proj",
            "material_proj",
            "output_proj",
            "stage_scales",
        }
        self.assertEqual(set(groups), expected_groups)

        all_names = tuple(name for name, _ in adapter.named_parameters())
        leaf_names = tuple(
            name
            for group, names in groups.items()
            if group != "all_adapter"
            for name in names
        )
        self.assertEqual(groups["all_adapter"], all_names)
        self.assertEqual(leaf_names, all_names)
        self.assertEqual(len(leaf_names), len(set(leaf_names)))

    def test_snapshot_uses_cpu_float64_and_rejects_missing_or_nonfinite_gradients(self):
        adapter = self.make_adapter()
        for _, parameter in adapter.named_parameters():
            parameter.grad = torch.ones_like(parameter)

        snapshot = snapshot_adapter_gradients(adapter)
        self.assertEqual(tuple(snapshot), tuple(name for name, _ in adapter.named_parameters()))
        self.assertTrue(all(value.device.type == "cpu" for value in snapshot.values()))
        self.assertTrue(all(value.dtype == torch.float64 for value in snapshot.values()))

        first_parameter = next(adapter.parameters())
        first_parameter.grad = None
        with self.assertRaisesRegex(ValueError, "missing gradient"):
            snapshot_adapter_gradients(adapter)

        first_parameter.grad = torch.full_like(first_parameter, float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite gradient"):
            snapshot_adapter_gradients(adapter)

    def test_mean_named_gradients_divides_sums_and_validates_inputs(self):
        sums = {
            "weight": torch.tensor([4.0, -2.0], dtype=torch.float64),
            "bias": torch.tensor([6.0], dtype=torch.float64),
        }
        means = mean_named_gradients(sums, sample_count=2)
        self.assertTrue(torch.equal(means["weight"], torch.tensor([2.0, -1.0], dtype=torch.float64)))
        self.assertTrue(torch.equal(means["bias"], torch.tensor([3.0], dtype=torch.float64)))

        with self.assertRaisesRegex(ValueError, "positive integer"):
            mean_named_gradients(sums, sample_count=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            mean_named_gradients({"weight": torch.tensor([float("inf")])}, sample_count=1)

    def test_summary_reports_known_cosines_norms_and_signed_stage_gradients(self):
        adapter = self.make_adapter()
        named = {
            name: torch.ones_like(parameter, dtype=torch.float64)
            for name, parameter in adapter.named_parameters()
        }

        def scaled(scale):
            values = {name: value * scale for name, value in named.items()}
            values["stage_scales"] = torch.tensor(
                [1.0, -2.0, 3.0, -4.0], dtype=torch.float64
            ) * scale
            return values

        material_gradients = {
            "elastic": {"sample_count": 52, "named_gradients": scaled(1.0)},
            "plasticine": {"sample_count": 56, "named_gradients": scaled(2.0)},
            "sand": {"sample_count": 56, "named_gradients": scaled(-1.0)},
        }
        summary = summarize_material_gradient_conflict(material_gradients)

        self.assertEqual(
            summary["sample_counts"],
            {"elastic": 52, "plasticine": 56, "sand": 56},
        )
        for group in (
            "all_adapter",
            "state_norm",
            "state_proj",
            "material_proj",
            "output_proj",
            "stage_scales",
        ):
            cosines = summary["groups"][group]["pairwise_cosine"]
            self.assertAlmostEqual(cosines["elastic__plasticine"], 1.0)
            self.assertAlmostEqual(cosines["elastic__sand"], -1.0)
            self.assertAlmostEqual(cosines["plasticine__sand"], -1.0)
            self.assertGreater(
                summary["groups"][group]["gradient_norms"]["elastic"], 0.0
            )
        self.assertEqual(
            summary["stage_scale_gradients"]["elastic"],
            [1.0, -2.0, 3.0, -4.0],
        )

    def test_summary_uses_none_for_zero_norm_cosine(self):
        adapter = self.make_adapter()
        zeros = {
            name: torch.zeros_like(parameter, dtype=torch.float64)
            for name, parameter in adapter.named_parameters()
        }
        ones = {name: torch.ones_like(value) for name, value in zeros.items()}
        summary = summarize_material_gradient_conflict(
            {
                "elastic": {"sample_count": 52, "named_gradients": zeros},
                "plasticine": {"sample_count": 56, "named_gradients": ones},
                "sand": {"sample_count": 56, "named_gradients": ones},
            }
        )
        self.assertIsNone(
            summary["groups"]["all_adapter"]["pairwise_cosine"][
                "elastic__plasticine"
            ]
        )

    def test_summary_rejects_noninteger_frozen_sample_count(self):
        adapter = self.make_adapter()
        gradients = {
            name: torch.ones_like(parameter, dtype=torch.float64)
            for name, parameter in adapter.named_parameters()
        }
        with self.assertRaisesRegex(ValueError, "sample_count"):
            summarize_material_gradient_conflict(
                {
                    "elastic": {
                        "sample_count": 52.0,
                        "named_gradients": gradients,
                    },
                    "plasticine": {
                        "sample_count": 56,
                        "named_gradients": gradients,
                    },
                    "sand": {
                        "sample_count": 56,
                        "named_gradients": gradients,
                    },
                }
            )


class FixedGradientWindowTests(unittest.TestCase):
    @staticmethod
    def manifest():
        return {
            "elastic": tuple(f"elastic_{index:02d}.h5" for index in range(13)),
            "plasticine": tuple(
                f"plasticine_{index:02d}.h5" for index in range(14)
            ),
            "sand": tuple(f"sand_{index:02d}.h5" for index in range(14)),
        }

    @staticmethod
    def dataset(manifest, *, point_count_overrides=None):
        names = [name for material in manifest.values() for name in material]
        entries = [
            {"model": name, "start_idx": start_idx}
            for name in names
            for start_idx in (0, 5, 10, 15)
        ]
        point_counts = {name: 8 for name in names}
        point_counts.update(point_count_overrides or {})

        class SyntheticDataset:
            split_lst_save = names
            models = entries

            def __getitem__(self, index):
                entry = self.models[index]
                point_count = point_counts[entry["model"]]
                return {
                    "points_src": torch.zeros(5, point_count, 3),
                    "points_tgt": torch.zeros(1, point_count, 3),
                }, {"model": entry["model"]}

        return SyntheticDataset()

    def test_fixed_windows_follow_manifest_order_and_four_registered_starts(self):
        from src.diagnose_material_state_gradient_conflict import (
            validate_fixed_gradient_windows,
        )

        manifest = self.manifest()
        dataset = self.dataset(manifest)
        selected = validate_fixed_gradient_windows(dataset, manifest)

        self.assertEqual(len(selected), 164)
        selected_entries = [dataset.models[index] for index in selected]
        expected_names = [
            name for material in manifest.values() for name in material
        ]
        self.assertEqual(
            [(entry["model"], entry["start_idx"]) for entry in selected_entries],
            [
                (model, start_idx)
                for model in expected_names
                for start_idx in (0, 5, 10, 15)
            ],
        )

    def test_fixed_windows_reject_protocol_mismatches_before_gpu_work(self):
        from src.diagnose_material_state_gradient_conflict import (
            validate_fixed_gradient_windows,
        )

        manifest = self.manifest()

        duplicate = self.dataset(manifest)
        duplicate.models[-1] = dict(duplicate.models[-2])
        with self.assertRaisesRegex(ValueError, "duplicate|starts"):
            validate_fixed_gradient_windows(duplicate, manifest)

        unknown = self.dataset(manifest)
        unknown.split_lst_save[-1] = "unknown.h5"
        with self.assertRaisesRegex(ValueError, "model set"):
            validate_fixed_gradient_windows(unknown, manifest)

        wrong_counts = dict(manifest)
        wrong_counts["elastic"] = wrong_counts["elastic"][:-1]
        with self.assertRaisesRegex(ValueError, "13/14/14"):
            validate_fixed_gradient_windows(self.dataset(wrong_counts), wrong_counts)

        first_sand = manifest["sand"][0]
        mixed_points = self.dataset(
            manifest, point_count_overrides={first_sand: 7}
        )
        with self.assertRaisesRegex(ValueError, "point count"):
            validate_fixed_gradient_windows(mixed_points, manifest)


class TeacherForcedGradientTests(unittest.TestCase):
    @staticmethod
    def model():
        config = OmegaConf.create(
            {
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
                "material_state_adapter": True,
                "material_state_rank": 16,
                "material_state_interval": 1,
            }
        )
        model = MDM_ST(2, 1, 3, config).eval()
        with torch.no_grad():
            model.dit.material_state_exchange.output_proj.weight.fill_(1e-3)
        return model

    @staticmethod
    def batch():
        return {
            "points_src": torch.randn(1, 5, 2, 3),
            "points_tgt": torch.randn(1, 1, 2, 3),
            "force": torch.randn(1, 3),
            "E": torch.tensor([[5.5]]),
            "nu": torch.tensor([[0.25]]),
            "mask": torch.zeros(1, 1, 2, 1),
            "drag_point": torch.zeros(1, 4),
            "floor_height": None,
            "gravity": None,
            "base_drag_coeff": None,
            "mat_type": torch.tensor([1]),
            "start_vel": torch.zeros(1, 2, 3),
            "points_rest": torch.randn(1, 2, 3),
        }

    def test_teacher_forced_forward_matches_input_contract_and_is_deterministic(self):
        from src.diagnose_material_state_gradient_conflict import (
            teacher_forced_one_step,
        )

        model = self.model()
        batch = self.batch()
        captured_inputs = []

        def capture_input(module, args, kwargs):
            captured_inputs.append(args[0].detach().clone())

        handle = model.register_forward_pre_hook(capture_input, with_kwargs=True)
        try:
            pred_a, target_a = teacher_forced_one_step(
                model, batch, torch.device("cpu"), seed=17
            )
            pred_b, target_b = teacher_forced_one_step(
                model, batch, torch.device("cpu"), seed=17
            )
        finally:
            handle.remove()

        torch.manual_seed(17)
        expected_input = batch["points_src"][:, -1:].clone()
        expected_input = expected_input + torch.randn_like(expected_input) * 0.02
        self.assertEqual(tuple(pred_a.shape), (1, 1, 2, 3))
        self.assertTrue(torch.equal(target_a, batch["points_tgt"]))
        self.assertTrue(torch.equal(target_b, batch["points_tgt"]))
        self.assertTrue(torch.equal(captured_inputs[0], expected_input))
        self.assertTrue(torch.equal(captured_inputs[0], captured_inputs[1]))
        self.assertTrue(torch.equal(pred_a, pred_b))

    def test_backward_only_populates_finite_adapter_gradients(self):
        from src.diagnose_material_state_gradient_conflict import (
            teacher_forced_one_step,
        )

        model = self.model()
        model.requires_grad_(False)
        adapter = model.dit.material_state_exchange
        adapter.requires_grad_(True)
        pred, target = teacher_forced_one_step(
            model, self.batch(), torch.device("cpu"), seed=23
        )
        torch.nn.functional.mse_loss(pred.float(), target.float()).backward()

        adapter_ids = {id(parameter) for parameter in adapter.parameters()}
        for parameter in adapter.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters():
            if id(parameter) not in adapter_ids:
                self.assertIsNone(parameter.grad)


class GradientConflictCliTests(unittest.TestCase):
    @staticmethod
    def payload():
        pairwise = {
            "elastic__plasticine": 0.5,
            "elastic__sand": -0.25,
            "plasticine__sand": 0.1,
        }
        groups = {
            group: {
                "gradient_norms": {
                    "elastic": 1.0,
                    "plasticine": 2.0,
                    "sand": 3.0,
                },
                "pairwise_cosine": dict(pairwise),
            }
            for group in (
                "all_adapter",
                "state_norm",
                "state_proj",
                "material_proj",
                "output_proj",
                "stage_scales",
            )
        }
        return {
            "checkpoint": "outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors",
            "config": "configs/eval_mm3_b3a_material_state_adapter_90k.yaml",
            "seed": 0,
            "sample_scope": "frozen 41-model x 4 fixed windows B3a90",
            "protocol": "164-window teacher-forced one-step coordinate MSE",
            "sample_counts": {"elastic": 52, "plasticine": 56, "sand": 56},
            "loss_means": {
                "overall": 0.4,
                "elastic": 0.2,
                "plasticine": 0.3,
                "sand": 0.6,
            },
            "groups": groups,
            "stage_scale_gradients": {
                "elastic": [0.1, -0.2, 0.3, -0.4],
                "plasticine": [0.2, -0.1, 0.4, -0.3],
                "sand": [-0.1, 0.2, -0.3, 0.4],
            },
        }

    def test_parser_requires_frozen_inputs_and_output_prefix(self):
        from src.diagnose_material_state_gradient_conflict import build_parser

        parsed = build_parser().parse_args(
            [
                "--config",
                "configs/eval_mm3_b3a_material_state_adapter_90k.yaml",
                "--checkpoint",
                "outputs/mm3_b3a_material_state_adapter_8L/checkpoint-90000/model.safetensors",
                "--output",
                "results/material_state_gradient_conflict_b3a90",
            ]
        )
        self.assertEqual(
            parsed.output, "results/material_state_gradient_conflict_b3a90"
        )

    def test_writer_emits_valid_json_and_chinese_protocol_report(self):
        from src.diagnose_material_state_gradient_conflict import (
            write_gradient_conflict_outputs,
        )

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "material_state_gradient_conflict_b3a90"
            paths = write_gradient_conflict_outputs(prefix, self.payload())
            self.assertEqual(
                {name: Path(path).name for name, path in paths.items()},
                {
                    "json": "material_state_gradient_conflict_b3a90.json",
                    "report": "material_state_gradient_conflict_b3a90.md",
                },
            )
            payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["sample_counts"]["elastic"], 52)
            report = Path(paths["report"]).read_text(encoding="utf-8")
            self.assertIn("164-window teacher-forced", report)
            self.assertIn("单步坐标 MSE", report)
            self.assertIn("不能证明 material experts 会改善 rollout", report)

    def test_writer_rejects_incomplete_payload_before_creating_outputs(self):
        from src.diagnose_material_state_gradient_conflict import (
            write_gradient_conflict_outputs,
        )

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "material_state_gradient_conflict_b3a90"
            payload = self.payload()
            payload["sample_counts"]["elastic"] = 51
            with self.assertRaisesRegex(ValueError, "52/56/56"):
                write_gradient_conflict_outputs(prefix, payload)
            self.assertFalse(prefix.with_suffix(".json").exists())
            self.assertFalse(prefix.with_suffix(".md").exists())


if __name__ == "__main__":
    unittest.main()
