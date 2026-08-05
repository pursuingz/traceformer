import math
import csv
import tempfile
import unittest
from pathlib import Path

import torch

from model.material_state import FactorizedMaterialStateAdapter
from utils.material_state_stage_diagnostics import (
    STAGE_METRICS,
    STAGE_KNOCKOUT_CONDITIONS,
    MaterialStateActivityCollector,
    build_stage_paired_rows,
    masked_material_state_stages,
    validate_stage_raw_rows,
)


class MaterialStateStageDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def stage_raw_rows():
        rows = []
        model_index = 0
        for mat_type, count in ((0, 13), (1, 14), (2, 14)):
            for _ in range(count):
                model = f"model_{model_index:02d}.h5"
                for condition_index, (condition, _) in enumerate(
                    STAGE_KNOCKOUT_CONDITIONS
                ):
                    row = {
                        "model": model,
                        "mat_type": mat_type,
                        "log10_e": 5.8,
                        "nu": 0.3,
                        "condition": condition,
                    }
                    row.update(
                        {
                            metric: float(model_index + condition_index + 1)
                            for metric in STAGE_METRICS
                        }
                    )
                    rows.append(row)
                model_index += 1
        return rows

    def test_stage_pairing_enforces_frozen_cardinality_and_direction(self):
        raw_rows = self.stage_raw_rows()

        self.assertEqual(len(validate_stage_raw_rows(raw_rows)), 246)
        paired = build_stage_paired_rows(raw_rows)

        self.assertEqual(len(paired), 205)
        self.assertEqual(
            paired[0]["delta_long_mse"],
            paired[0]["knockout_long_mse"] - paired[0]["normal_long_mse"],
        )

    def test_stage_validation_rejects_invalid_pairing_inputs(self):
        raw_rows = self.stage_raw_rows()
        duplicate = [dict(row) for row in raw_rows]
        duplicate[-1]["condition"] = "all_off"
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            validate_stage_raw_rows(duplicate)

        missing_stage3 = [
            dict(row) for row in raw_rows if row["condition"] != "stage3_off"
        ]
        with self.assertRaisesRegex(ValueError, "246"):
            validate_stage_raw_rows(missing_stage3)

        wrong_material_counts = [dict(row) for row in raw_rows]
        for row in wrong_material_counts:
            if row["model"] == "model_00.h5":
                row["mat_type"] = 1
        with self.assertRaisesRegex(ValueError, "material counts"):
            validate_stage_raw_rows(wrong_material_counts)

        nonfinite = [dict(row) for row in raw_rows]
        nonfinite[0]["long_mse"] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_stage_raw_rows(nonfinite)

        mixed_provenance = [
            {**row, "checkpoint": "checkpoint-a.safetensors"} for row in raw_rows
        ]
        mixed_provenance[1]["checkpoint"] = "checkpoint-b.safetensors"
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_stage_raw_rows(mixed_provenance)

        zero_baseline = [dict(row) for row in raw_rows]
        for row in zero_baseline:
            if row["model"] == "model_00.h5" and row["condition"] == "normal":
                row["penetration_rate"] = 0.0
            elif row["model"] == "model_00.h5" and row["condition"] == "all_off":
                row["penetration_rate"] = 0.2
        paired = build_stage_paired_rows(zero_baseline)
        row = next(
            item
            for item in paired
            if item["model"] == "model_00.h5" and item["condition"] == "all_off"
        )
        self.assertIsNone(row["relative_change_pct_penetration_rate"])

    def test_stage_writers_emit_complete_registered_outputs(self):
        from utils.material_state_stage_diagnostics import (
            summarize_stage_paired_rows,
            write_stage_outputs,
        )

        metadata = {
            "checkpoint": "checkpoint-90000/model.safetensors",
            "config": "configs/eval_mm3_b3a_material_state_adapter_90k.yaml",
            "seed": 0,
            "sample_scope": "frozen 41-model start_idx=0 B3a90",
            "bootstrap_samples": 10,
            "bootstrap_seed": 0,
        }
        raw_rows = [{**row, **metadata} for row in self.stage_raw_rows()]
        paired_rows = build_stage_paired_rows(raw_rows)
        summary_rows = summarize_stage_paired_rows(
            paired_rows, bootstrap_samples=10, bootstrap_seed=0
        )
        activity_rows = [
            {
                **metadata,
                "model": f"model_{model_index:02d}.h5",
                "mat_type": mat_type,
                "stage_index": stage_index,
                "call_count": 20,
                "delta_rms": 0.1,
                "hidden_rms": 1.0,
                "relative_rms": 0.1,
            }
            for mat_type, count, start in ((0, 13, 0), (1, 14, 13), (2, 14, 27))
            for model_index in range(start, start + count)
            for stage_index in range(4)
        ]

        with tempfile.TemporaryDirectory() as directory:
            paths = write_stage_outputs(
                directory, raw_rows, paired_rows, activity_rows, summary_rows, metadata
            )
            self.assertEqual(
                {name: Path(path).name for name, path in paths.items()},
                {
                    "raw": "material_state_stage_knockout_b3a90_raw.csv",
                    "paired": "material_state_stage_knockout_b3a90_paired.csv",
                    "activity": "material_state_stage_activity_b3a90.csv",
                    "report": "material_state_stage_knockout_b3a90.md",
                },
            )
            for name, expected_rows in (("raw", 246), ("paired", 205), ("activity", 164)):
                with Path(paths[name]).open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), expected_rows)
            report = Path(paths["report"]).read_text(encoding="utf-8")
            self.assertIn("knockout - normal", report)
            self.assertIn("all_off", report)
            self.assertIn("不是独立训练的 baseline", report)


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

    def test_nested_collector_contexts_are_rejected_without_leaking_hook(self):
        adapter = self.make_adapter()
        outer = MaterialStateActivityCollector(adapter)

        with outer:
            with self.assertRaisesRegex(ValueError, "nested collector"):
                with MaterialStateActivityCollector(adapter):
                    pass
            with outer.capture("model-a", 0, 1):
                for stage_index in range(4):
                    self.forward_stage(adapter, stage_index)

        self.assertEqual(len(adapter._forward_hooks), 0)

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


class MaterialStateStageKnockoutCliTests(unittest.TestCase):
    def test_parser_uses_registered_b3a90_defaults(self):
        from src.diagnose_material_state_stage_knockout import build_parser

        parsed = build_parser().parse_args(
            [
                "--config",
                "configs/eval_mm3_b3a_material_state_adapter_90k.yaml",
                "--checkpoint",
                "outputs/mm3_b3a_material_state_adapter_8L/"
                "checkpoint-90000/model.safetensors",
                "--output-dir",
                "results/b3a-stage",
            ]
        )

        self.assertEqual(parsed.bootstrap_samples, 10000)
        self.assertEqual(parsed.bootstrap_seed, 0)

    def test_b3a90_identity_rejects_wrong_checkpoint_and_frozen_fields(self):
        from omegaconf import OmegaConf
        from src.diagnose_material_state_stage_knockout import _validate_b0_identity
        from src.options import TestingConfig

        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "eval_mm3_b3a_material_state_adapter_90k.yaml"
        )

        def loaded_args():
            return OmegaConf.merge(
                OmegaConf.structured(TestingConfig), OmegaConf.load(config_path)
            )

        args = loaded_args()
        _validate_b0_identity(args, profile="b3a90")
        for path, value, error in (
            ("resume", "outputs/mm3_b3a_material_state_adapter_8L/checkpoint-45000/model.safetensors", "checkpoint mismatch"),
            ("model_config.material_state_rank", 32, "material_state_rank"),
            ("model_config.material_state_interval", 1, "material_state_interval"),
            ("model_config.material_state_adapter", False, "material_state_adapter"),
            ("train_dataset.dataset_path", "mm3_data/mm3_train", "dataset mismatch"),
        ):
            with self.subTest(path=path):
                args = loaded_args()
                target, field = path.rsplit(".", 1) if "." in path else (None, path)
                section = args if target is None else getattr(args, target)
                setattr(section, field, value)
                with self.assertRaisesRegex(ValueError, error):
                    _validate_b0_identity(args, profile="b3a90")

if __name__ == "__main__":
    unittest.main()
