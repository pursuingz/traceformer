import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.stats import rankdata

from utils import material_response_fidelity
from utils.material_response_fidelity import (
    OUTPUT_NAMES,
    RESPONSE_NAMES,
    build_alignment_summary,
    build_fidelity_summary,
    build_response_rows,
    classify_alignment,
    extract_position_responses,
    partial_spearman,
    preflight_fidelity_outputs,
    write_fidelity_outputs,
)


def translating_scaling_box(
    *, frames: int, scale_step: float, dy: float
) -> np.ndarray:
    box = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    time = np.arange(frames, dtype=np.float64)[:, None, None]
    scale = 1.0 + scale_step * time
    center = box.mean(axis=0)
    translation = np.concatenate(
        [
            np.zeros_like(time),
            dy * time,
            np.zeros_like(time),
        ],
        axis=-1,
    )
    return center + (box[None, :, :] - center) * scale + translation


def fake_model_row(
    model: str,
    *,
    mat_type: int,
    log10_e: float,
    nu: float,
) -> dict[str, object]:
    return {
        "model": model,
        "mat_type": mat_type,
        "log10_e": log10_e,
        "nu": nu,
        "checkpoint": "checkpoint-90000/model.safetensors",
        "config": "configs/eval_mm3_contact_cond.yaml",
        "seed": 0,
        "sample_scope": "test_start_idx_0",
        "full_rollout_mse": 0.01,
        "gm_mse": 0.02,
        "long_seg_mse": 0.03,
        "fde": 0.04,
    }


def synthetic_trajectory(*, amplitude: float) -> np.ndarray:
    return translating_scaling_box(
        frames=25,
        scale_step=0.002 * amplitude,
        dy=-0.004 * amplitude,
    )


def complete_response_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = {0: 13, 1: 14, 2: 14}
    for mat_type, count in counts.items():
        for index in range(count):
            parameter = float(index) / max(count - 1, 1)
            gt = synthetic_trajectory(amplitude=0.5 + parameter)
            pred = synthetic_trajectory(amplitude=0.5 + 0.75 * parameter)
            rows.extend(
                build_response_rows(
                    fake_model_row(
                        f"material_{mat_type}_{index:02d}.h5",
                        mat_type=mat_type,
                        log10_e=4.0 + 3.0 * parameter,
                        nu=0.05 + 0.40 * (1.0 - parameter),
                    ),
                    gt=gt,
                    pred=pred,
                    floor_height=-2.0,
                    contact_band_raw=0.08,
                )
            )
    return rows


class PositionResponseTests(unittest.TestCase):
    def test_extract_position_responses_matches_closed_form_motion_and_volume(self):
        trajectory = translating_scaling_box(frames=25, scale_step=0.01, dy=-0.02)
        result = extract_position_responses(
            trajectory,
            floor_height=-10.0,
            contact_band_raw=0.08,
        )
        expected_velocity = np.sqrt(np.mean(np.diff(trajectory, axis=0) ** 2))
        expected_acceleration = np.sqrt(
            np.mean(np.diff(trajectory, n=2, axis=0) ** 2)
        )

        self.assertAlmostEqual(
            result["position_velocity_rms_trajectory"], expected_velocity
        )
        self.assertAlmostEqual(
            result["position_acceleration_rms_trajectory"], expected_acceleration
        )
        self.assertAlmostEqual(result["centroid_displacement_f24"], 0.48)
        self.assertGreater(result["hull_volume_relative_change_f24"], 0.0)
        self.assertEqual(set(result), set(RESPONSE_NAMES))

    def test_extract_position_responses_reports_contact_and_signed_extents(self):
        trajectory = translating_scaling_box(frames=25, scale_step=0.01, dy=-0.02)

        result = extract_position_responses(
            trajectory,
            floor_height=-0.10,
            contact_band_raw=0.08,
        )

        self.assertAlmostEqual(result["future_contact_fraction"], 0.5)
        self.assertGreater(result["extent_change_x_f24"], 0.0)
        self.assertGreater(result["extent_change_y_f24"], 0.0)
        self.assertGreater(result["extent_change_z_f24"], 0.0)

    def test_extract_position_responses_rejects_invalid_inputs(self):
        valid = translating_scaling_box(frames=25, scale_step=0.01, dy=-0.02)

        cases = (
            (valid[:24], -0.10, 0.08, "25 frames"),
            (valid[:, :3], -0.10, 0.08, "at least 4 particles"),
            (valid.copy(), -0.10, -0.01, "contact_band_raw"),
        )

        for trajectory, floor_height, contact_band_raw, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    extract_position_responses(
                        trajectory,
                        floor_height=floor_height,
                        contact_band_raw=contact_band_raw,
                    )

        nonfinite = valid.copy()
        nonfinite[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            extract_position_responses(
                nonfinite,
                floor_height=-0.10,
                contact_band_raw=0.08,
            )

    def test_extract_position_responses_rejects_degenerate_hull(self):
        degenerate = np.zeros((25, 8, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "convex hull"):
            extract_position_responses(degenerate, floor_height=-0.10)


class ResponseStatisticsTests(unittest.TestCase):
    def test_build_response_rows_pairs_identical_schema(self):
        rows = build_response_rows(
            model_row=fake_model_row(
                "elastic_00.h5", mat_type=0, log10_e=5.0, nu=0.2
            ),
            gt=synthetic_trajectory(amplitude=2.0),
            pred=synthetic_trajectory(amplitude=1.0),
            floor_height=-2.0,
            contact_band_raw=0.08,
        )

        self.assertEqual(len(rows), len(RESPONSE_NAMES))
        self.assertEqual({row["response"] for row in rows}, set(RESPONSE_NAMES))
        self.assertTrue(all(row["absolute_error"] >= 0.0 for row in rows))
        self.assertTrue(
            all(
                np.isclose(
                    row["signed_error"], row["pred_value"] - row["gt_value"]
                )
                for row in rows
            )
        )

    def test_partial_spearman_removes_other_parameter_rank_effect(self):
        control = np.arange(20, dtype=float)
        x = control + np.tile([0.0, 1.0], 10)
        y = control.copy()

        rho = partial_spearman(x, y, control)

        self.assertIsNotNone(rho)
        self.assertLess(abs(rho), 0.2)

    def test_partial_spearman_uses_pearson_correlation_of_rank_residuals(self):
        x = np.asarray([4.0, 2.0, 1.0, 0.0, 3.0])
        y = np.asarray([3.0, 0.0, 4.0, 2.0, 1.0])
        control = np.asarray([1.0, 2.0, 0.0, 3.0, 4.0])
        rank_x, rank_y, rank_control = (rankdata(values) for values in (x, y, control))
        design = np.column_stack((np.ones_like(rank_control), rank_control))
        residual_x = rank_x - design @ np.linalg.lstsq(design, rank_x, rcond=None)[0]
        residual_y = rank_y - design @ np.linalg.lstsq(design, rank_y, rcond=None)[0]
        expected = float(np.corrcoef(residual_x, residual_y)[0, 1])

        actual = partial_spearman(x, y, control)

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertNotAlmostEqual(actual, -0.7, places=6)

    def test_alignment_labels_are_frozen_at_boundaries(self):
        self.assertEqual(classify_alignment(-0.60, -0.40), "aligned")
        self.assertEqual(classify_alignment(-0.60, -0.20), "attenuated")
        self.assertEqual(classify_alignment(-0.60, +0.30), "reversed")
        self.assertEqual(classify_alignment(-0.60, +0.10), "reversed")
        self.assertEqual(classify_alignment(-0.60, -0.10), "attenuated")
        self.assertEqual(classify_alignment(-0.19, -0.19), "weak_or_unresolved")

    def test_fidelity_bootstrap_is_model_paired_and_seed_reproducible(self):
        rows = complete_response_rows()

        first = build_fidelity_summary(rows, bootstrap_samples=100, seed=7)
        second = build_fidelity_summary(rows, bootstrap_samples=100, seed=7)

        self.assertEqual(first, second)
        overall = next(
            row
            for row in first
            if row["group"] == "overall"
            and row["response"] == "centroid_displacement_f24"
        )
        self.assertEqual(overall["n"], 41)
        self.assertLessEqual(overall["spearman_ci_low"], overall["spearman"])
        self.assertGreaterEqual(overall["spearman_ci_high"], overall["spearman"])

    def test_alignment_freezes_13_14_14_groups_and_weak_magnitude_ratio(self):
        rows = complete_response_rows()
        alignment = build_alignment_summary(rows, bootstrap_samples=100, seed=11)

        elastic = next(
            row
            for row in alignment
            if row["material"] == "elastic"
            and row["parameter"] == "log10_e"
            and row["response"] == "future_contact_fraction"
        )
        self.assertEqual(elastic["n"], 13)
        self.assertEqual(elastic["status"], "constant_response")
        self.assertIsNone(elastic["gt_ordinary_rho"])
        self.assertIsNone(elastic["magnitude_ratio"])

    def test_alignment_rejects_nonfrozen_material_counts(self):
        rows = complete_response_rows()
        bad = [row for row in rows if row["model"] != "material_2_13.h5"]

        with self.assertRaisesRegex(ValueError, "elastic=13, plasticine=14, sand=14"):
            build_alignment_summary(bad, bootstrap_samples=10, seed=0)

    def test_alignment_bootstrap_resamples_model_triples_and_hides_weak_ratio(self):
        rows = complete_response_rows()
        altered = []
        for row in rows:
            copied = dict(row)
            if (
                copied["mat_type"] == 0
                and copied["response"] == "extent_change_x_f24"
            ):
                index = int(str(copied["model"]).split("_")[-1].split(".")[0])
                copied["gt_value"] = float(index % 2)
                copied["pred_value"] = float(index % 2) + 0.05
                copied["signed_error"] = 0.05
                copied["absolute_error"] = 0.05
            altered.append(copied)

        first = build_alignment_summary(altered, bootstrap_samples=100, seed=23)
        second = build_alignment_summary(altered, bootstrap_samples=100, seed=23)
        row = next(
            item
            for item in first
            if item["material"] == "elastic"
            and item["parameter"] == "log10_e"
            and item["response"] == "extent_change_x_f24"
        )

        self.assertEqual(first, second)
        self.assertLess(abs(row["gt_ordinary_rho"]), 0.05)
        self.assertIsNone(row["magnitude_ratio"])
        self.assertIsNotNone(row["gt_partial_ci_low"])


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary_directory.name) / "outputs"
        self.response_rows = complete_response_rows()
        self.model_rows = []
        seen = set()
        for row in self.response_rows:
            if row["model"] in seen:
                continue
            seen.add(row["model"])
            self.model_rows.append(
                {
                    key: row[key]
                    for key in (
                        "model",
                        "mat_type",
                        "material",
                        "log10_e",
                        "nu",
                        "checkpoint",
                        "config",
                        "seed",
                        "sample_scope",
                        "full_rollout_mse",
                        "gm_mse",
                        "long_seg_mse",
                        "fde",
                    )
                }
            )
        self.fidelity_rows = build_fidelity_summary(
            self.response_rows, bootstrap_samples=20, seed=0
        )
        self.alignment_rows = build_alignment_summary(
            self.response_rows, bootstrap_samples=20, seed=0
        )
        self.metadata = {
            "schema_version": "1.0",
            "checkpoint": "checkpoint-90000/model.safetensors",
            "config": "configs/eval_mm3_contact_cond.yaml",
            "seed": 0,
            "split": "test_start_idx_0",
            "model_counts": {"elastic": 13, "plasticine": 14, "sand": 14},
            "response_schema": list(RESPONSE_NAMES),
            "bootstrap_samples": 20,
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write(self, *, overwrite=False, **changes):
        payload = {
            "model_rows": self.model_rows,
            "response_rows": self.response_rows,
            "fidelity_rows": self.fidelity_rows,
            "alignment_rows": self.alignment_rows,
            "metadata": self.metadata,
        }
        payload.update(changes)
        return write_fidelity_outputs(self.output_dir, overwrite=overwrite, **payload)

    def test_writer_emits_exact_six_outputs_and_chinese_report(self):
        paths = self._write()

        self.assertEqual(set(paths), set(OUTPUT_NAMES))
        self.assertEqual(
            {path.name for path in paths.values()}, set(OUTPUT_NAMES.values())
        )
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()}, set(OUTPUT_NAMES.values())
        )
        report = paths["report"].read_text(encoding="utf-8")
        self.assertIn("位置可观测响应", report)
        self.assertIn("不能证明 counterfactual", report)
        for material in ("elastic", "plasticine", "sand"):
            self.assertIn(material, report)
        with paths["responses"].open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 41 * len(RESPONSE_NAMES))
        written_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.assertEqual(written_metadata["model_counts"]["elastic"], 13)

    def test_preflight_rejects_any_existing_target_without_overwrite(self):
        self.output_dir.mkdir()
        existing = self.output_dir / OUTPUT_NAMES["models"]
        existing.write_text("old", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            preflight_fidelity_outputs(self.output_dir, overwrite=False)

    def test_writer_rejects_invalid_metadata_model_counts_and_response_schema(self):
        missing_metadata = dict(self.metadata)
        del missing_metadata["response_schema"]
        with self.assertRaisesRegex(ValueError, "metadata is missing"):
            self._write(metadata=missing_metadata)

        with self.assertRaisesRegex(ValueError, "exactly 41"):
            self._write(model_rows=self.model_rows[:-1])

        incomplete = self.response_rows[:-1]
        with self.assertRaisesRegex(ValueError, "response schema"):
            self._write(response_rows=incomplete)

        self.assertFalse(self.output_dir.exists())

    def test_writer_rejects_cross_file_provenance_mismatch(self):
        mismatched_responses = [dict(row) for row in self.response_rows]
        mismatched_responses[0]["checkpoint"] = "other/model.safetensors"
        with self.assertRaisesRegex(ValueError, "provenance"):
            self._write(response_rows=mismatched_responses)

        mismatched_metadata = dict(self.metadata, seed=1)
        with self.assertRaisesRegex(ValueError, "metadata and model_rows"):
            self._write(metadata=mismatched_metadata)

        self.assertFalse(self.output_dir.exists())

    def test_writer_rejects_response_row_missing_provenance_field(self):
        missing_provenance = [dict(row) for row in self.response_rows]
        del missing_provenance[0]["checkpoint"]

        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self._write(response_rows=missing_provenance)

        self.assertFalse(self.output_dir.exists())

    def test_writer_rejects_41_models_with_nonfrozen_material_counts(self):
        wrong_counts = [dict(row) for row in self.model_rows]
        plasticine_row = next(row for row in wrong_counts if row["mat_type"] == 1)
        plasticine_row["mat_type"] = 0
        plasticine_row["material"] = "elastic"

        self.assertEqual(len({row["model"] for row in wrong_counts}), 41)
        self.assertEqual(
            {row["mat_type"] for row in wrong_counts}.intersection({0, 1, 2}),
            {0, 1, 2},
        )
        with self.assertRaisesRegex(ValueError, "material counts"):
            self._write(model_rows=wrong_counts)

        self.assertFalse(self.output_dir.exists())

    def test_writer_leaves_no_final_outputs_when_staged_rendering_fails(self):
        with mock.patch.object(
            material_response_fidelity,
            "_write_csv",
            side_effect=OSError("injected staged write failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected staged write failure"):
                self._write()

        self.assertFalse(self.output_dir.exists())

    def test_overwrite_replaces_complete_old_output_set(self):
        paths = self._write()
        old_contents = {key: path.read_bytes() for key, path in paths.items()}
        replacement_metadata = dict(self.metadata, schema_version="1.1")

        replaced = self._write(overwrite=True, metadata=replacement_metadata)

        self.assertEqual(replaced, paths)
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()}, set(OUTPUT_NAMES.values())
        )
        self.assertNotEqual(replaced["metadata"].read_bytes(), old_contents["metadata"])
        self.assertFalse(any(".backup." in path.name for path in self.output_dir.parent.iterdir()))

    def test_overwrite_rolls_back_all_old_outputs_after_mid_activation_failure(self):
        paths = self._write()
        old_contents = {key: path.read_bytes() for key, path in paths.items()}
        original_replace = Path.replace
        failed = False

        def fail_fidelity_activation(source, target):
            nonlocal failed
            target = Path(target)
            if (
                not failed
                and Path(source).name == OUTPUT_NAMES["fidelity"]
                and target == paths["fidelity"]
            ):
                failed = True
                raise OSError("injected activation failure")
            return original_replace(source, target)

        with mock.patch.object(
            Path, "replace", autospec=True, side_effect=fail_fidelity_activation
        ):
            with self.assertRaisesRegex(OSError, "injected activation failure"):
                self._write(overwrite=True, metadata=dict(self.metadata, schema_version="1.1"))

        self.assertTrue(failed)
        self.assertEqual({key: path.read_bytes() for key, path in paths.items()}, old_contents)
        self.assertFalse(any(".backup." in path.name for path in self.output_dir.parent.iterdir()))


if __name__ == "__main__":
    unittest.main()
