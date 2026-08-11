import unittest

import numpy as np
from scipy.stats import rankdata

from utils.material_response_fidelity import (
    RESPONSE_NAMES,
    build_alignment_summary,
    build_fidelity_summary,
    build_response_rows,
    classify_alignment,
    extract_position_responses,
    partial_spearman,
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


if __name__ == "__main__":
    unittest.main()
