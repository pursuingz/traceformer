import unittest

import numpy as np

from utils.material_response_fidelity import (
    RESPONSE_NAMES,
    extract_position_responses,
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


if __name__ == "__main__":
    unittest.main()
