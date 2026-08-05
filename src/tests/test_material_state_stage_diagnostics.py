import unittest

import torch

from model.material_state import FactorizedMaterialStateAdapter
from utils.material_state_stage_diagnostics import (
    STAGE_KNOCKOUT_CONDITIONS,
    MaterialStateActivityCollector,
    masked_material_state_stages,
)


class MaterialStateStageDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def make_adapter():
        adapter = FactorizedMaterialStateAdapter(
            particle_dim=8,
            rank=4,
            num_materials=3,
            num_stages=4,
        )
        with torch.no_grad():
            adapter.output_proj.weight.fill_(0.01)
        return adapter

    @staticmethod
    def inputs():
        return (
            torch.randn(1, 2, 3, 8),
            torch.tensor([[5.0, 0.25]]),
            torch.tensor([1]),
        )

    def forward_stage(self, adapter, stage_index):
        hidden, values, labels = self.inputs()
        return adapter(
            hidden,
            values,
            labels,
            stage_index=stage_index,
        )

    def test_stage_conditions_cover_normal_all_and_four_single_knockouts(self):
        self.assertEqual(
            STAGE_KNOCKOUT_CONDITIONS,
            (
                ("normal", (1, 1, 1, 1)),
                ("all_off", (0, 0, 0, 0)),
                ("stage0_off", (0, 1, 1, 1)),
                ("stage1_off", (1, 0, 1, 1)),
                ("stage2_off", (1, 1, 0, 1)),
                ("stage3_off", (1, 1, 1, 0)),
            ),
        )

    def test_mask_multiplies_checkpoint_scales_and_restores_exactly(self):
        adapter = self.make_adapter()
        with torch.no_grad():
            adapter.stage_scales.copy_(torch.tensor([1.1, 0.9, 0.7, 0.5]))
        original = adapter.stage_scales.detach().clone()

        with masked_material_state_stages(adapter, (1, 0, 1, 0)):
            self.assertTrue(
                torch.equal(
                    adapter.stage_scales.detach(),
                    torch.tensor([1.1, 0.0, 0.7, 0.0]),
                )
            )

        self.assertTrue(torch.equal(adapter.stage_scales.detach(), original))

    def test_mask_rejects_nonbinary_or_wrong_length_masks(self):
        adapter = self.make_adapter()

        with self.assertRaisesRegex(ValueError, "four values"):
            with masked_material_state_stages(adapter, (1, 0, 1)):
                pass
        with self.assertRaisesRegex(ValueError, "binary"):
            with masked_material_state_stages(adapter, (1, 0, 0.5, 1)):
                pass

    def test_collector_records_two_calls_per_stage_with_nonzero_activity(self):
        adapter = self.make_adapter()

        with MaterialStateActivityCollector(adapter) as collector:
            with collector.capture("model-a", "elastic", 2):
                for _ in range(2):
                    for stage_index in range(4):
                        self.forward_stage(adapter, stage_index)

        rows = collector.rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["stage_index"] for row in rows}, {0, 1, 2, 3})
        self.assertTrue(all(row["call_count"] == 2 for row in rows))
        self.assertTrue(all(row["delta_rms"] > 0 for row in rows))
        self.assertTrue(all(row["hidden_rms"] > 0 for row in rows))
        self.assertTrue(all(row["relative_rms"] > 0 for row in rows))
        self.assertTrue(all(row["model"] == "model-a" for row in rows))
        self.assertTrue(all(row["mat_type"] == "elastic" for row in rows))

    def test_rows_returns_a_defensive_copy(self):
        adapter = self.make_adapter()
        with MaterialStateActivityCollector(adapter) as collector:
            with collector.capture("model-a", 0, 1):
                self.forward_stage(adapter, 0)
                self.forward_stage(adapter, 1)
                self.forward_stage(adapter, 2)
                self.forward_stage(adapter, 3)

        rows = collector.rows()
        rows[0]["delta_rms"] = -1.0
        self.assertGreater(collector.rows()[0]["delta_rms"], 0.0)

    def test_capture_rejects_nested_model_contexts(self):
        adapter = self.make_adapter()
        with MaterialStateActivityCollector(adapter) as collector:
            with collector.capture("model-a", 0, 1):
                with self.assertRaisesRegex(ValueError, "nested"):
                    with collector.capture("model-b", 1, 1):
                        pass
                for stage_index in range(4):
                    self.forward_stage(adapter, stage_index)

    def test_capture_rejects_missing_stages_on_exit(self):
        adapter = self.make_adapter()
        with MaterialStateActivityCollector(adapter) as collector:
            with self.assertRaisesRegex(ValueError, "stage"):
                with collector.capture("model-a", 0, 1):
                    self.forward_stage(adapter, 0)

        self.assertEqual(collector.rows(), [])

    def test_capture_rejects_unexpected_stage_call_count_on_exit(self):
        adapter = self.make_adapter()
        with MaterialStateActivityCollector(adapter) as collector:
            with self.assertRaisesRegex(ValueError, "call count"):
                with collector.capture("model-a", 0, 2):
                    for stage_index in range(4):
                        self.forward_stage(adapter, stage_index)

        self.assertEqual(collector.rows(), [])

    def test_capture_rejects_nonfinite_outputs(self):
        adapter = self.make_adapter()
        with torch.no_grad():
            adapter.output_proj.weight.fill_(float("nan"))

        with MaterialStateActivityCollector(adapter) as collector:
            with self.assertRaisesRegex(ValueError, "finite"):
                with collector.capture("model-a", 0, 1):
                    self.forward_stage(adapter, 0)

        self.assertEqual(collector.rows(), [])


if __name__ == "__main__":
    unittest.main()
