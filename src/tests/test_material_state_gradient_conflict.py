import unittest

import torch

from model.material_state import FactorizedMaterialStateAdapter
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


if __name__ == "__main__":
    unittest.main()
