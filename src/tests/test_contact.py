import unittest
import os
import tempfile

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from dataset.traj_dataset import TrajDataset
from model.spacetime import MDM_ST
from options import TestingConfig, TrainingConfig
from utils.contact import (
    apply_contact_feature_mask,
    build_contact_features,
    contact_channel_contributions,
    contact_weighted_losses,
    find_contact_window_starts,
)


def _model_config(contact_particle_cond=False):
    return OmegaConf.create({
        "n_layers": 1,
        "latent_dim": 64,
        "frame_cond": True,
        "cond_frames": 2,
        "point_embed": True,
        "mask_cond": False,
        "pred_offset": True,
        "num_neighbors": -1,
        "floor_cond": True,
        "max_num_forces": 1,
        "force_as_token": False,
        "force_as_latent": False,
        "gravity_emb": False,
        "coeff_cond": False,
        "num_mat": 0,
        "class_token": False,
        "class_dropout_prob": 0.0,
        "transformer_block": "SpatialTemporalTransformerBlock",
        "contact_particle_cond": contact_particle_cond,
        "contact_feature_sigma": 0.04,
    })


class ContactFeatureTests(unittest.TestCase):
    def test_dataset_can_reserve_random_windows_for_contact_targets(self):
        with tempfile.TemporaryDirectory(suffix="_test") as dataset_path:
            h5_path = os.path.join(dataset_path, "sample_001.h5")
            points = np.zeros((25, 2, 3), dtype=np.float32)
            points[:, :, 1] = 4.0
            points[11:, :, 1] = 1.05
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("x", data=points)
                h5.create_dataset("floor_height", data=1.0)

            cfg = OmegaConf.create({
                "dataset_path": dataset_path,
                "dataset_list": "MISSING_DATASET_LIST",
                "stage": "deform",
                "mode": "diff",
                "repeat": 1,
                "seed": 0,
                "pc_size": 2,
                "n_sample_pro_model": 1,
                "n_frames_interval": 1,
                "n_training_frames": 24,
                "input_frames": 5,
                "output_frames": 1,
                "rollout_unroll_steps": 1,
                "rollout_random_window": True,
                "rollout_force_start0": False,
                "windows_per_model": 4,
                "train_extra_random_windows": 0,
                "contact_window_ratio": 1.0,
                "contact_margin": 0.04,
                "batch_size": 1,
                "has_gravity": True,
                "max_num_forces": 1,
                "overfit": False,
            })

            dataset = TrajDataset("train", cfg)

            self.assertEqual(len(dataset.models), 4)
            self.assertTrue(all(item["start_idx"] == -2 for item in dataset.models))
            self.assertTrue(
                all(item["contact_starts"] == [4, 5, 6, 7, 8] for item in dataset.models)
            )

    def test_training_config_contact_defaults_preserve_existing_behavior(self):
        fields = TrainingConfig.__dataclass_fields__
        self.assertEqual(fields["lambda_contact_pos"].default, 0.0)
        self.assertEqual(fields["lambda_contact_vel"].default, 0.0)
        self.assertEqual(fields["contact_margin"].default, 0.04)
        self.assertEqual(fields["contact_temperature"].default, 0.01)
        self.assertEqual(fields["contact_window_ratio"].default, 0.0)
        self.assertEqual(fields["contact_frame_radius"].default, 2)

    def test_eval_contact_margin_defaults_to_training_contact_band(self):
        fields = TestingConfig.__dataclass_fields__

        self.assertEqual(fields["contact_eval_margin"].default, 0.04)

    def test_contact_sampling_falls_back_to_uniform_without_floor_metadata(self):
        with tempfile.TemporaryDirectory(suffix="_test") as dataset_path:
            h5_path = os.path.join(dataset_path, "sample_001.h5")
            points = np.zeros((25, 2, 3), dtype=np.float32)
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("x", data=points)

            cfg = OmegaConf.create({
                "dataset_path": dataset_path,
                "dataset_list": "MISSING_DATASET_LIST",
                "stage": "deform",
                "mode": "diff",
                "repeat": 1,
                "seed": 0,
                "pc_size": 2,
                "n_sample_pro_model": 1,
                "n_frames_interval": 1,
                "n_training_frames": 24,
                "input_frames": 5,
                "output_frames": 1,
                "rollout_unroll_steps": 1,
                "rollout_random_window": True,
                "rollout_force_start0": False,
                "windows_per_model": 4,
                "train_extra_random_windows": 0,
                "contact_window_ratio": 1.0,
                "contact_margin": 0.04,
                "batch_size": 1,
                "has_gravity": True,
                "max_num_forces": 1,
                "overfit": False,
            })

            dataset = TrajDataset("train", cfg)

            self.assertTrue(all(item["start_idx"] == -1 for item in dataset.models))

    def test_contact_encoder_is_opt_in_and_zero_initialized(self):
        baseline = MDM_ST(8, 1, n_feats=3, model_config=_model_config(False))
        contact_model = MDM_ST(8, 1, n_feats=3, model_config=_model_config(True))

        self.assertFalse(hasattr(baseline, "contact_encoder"))
        self.assertTrue(hasattr(contact_model, "contact_encoder"))
        self.assertEqual(contact_model.contact_encoder.in_features, 3)
        self.assertEqual(contact_model.contact_encoder.out_features, 64)
        self.assertEqual(torch.count_nonzero(contact_model.contact_encoder.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(contact_model.contact_encoder.bias).item(), 0)
        self.assertEqual(contact_model.contact_feature_mask, (1.0, 1.0, 1.0))
        self.assertEqual(contact_model.contact_bias_scale, 1.0)

    def test_contact_feature_mask_disables_only_selected_channels(self):
        features = torch.tensor([[[[1.0, 2.0, 3.0]]]])

        masked = apply_contact_feature_mask(features, [1.0, 0.0, 1.0])

        torch.testing.assert_close(
            masked,
            torch.tensor([[[[1.0, 0.0, 3.0]]]]),
        )

    def test_contact_feature_mask_rejects_invalid_length(self):
        features = torch.zeros(1, 1, 1, 3)

        with self.assertRaisesRegex(ValueError, "must contain 3"):
            apply_contact_feature_mask(features, [1.0, 0.0])

    def test_contact_channel_contributions_match_linear_column_norms(self):
        features = torch.tensor(
            [[[[1.0, -2.0, 3.0], [-3.0, 4.0, -5.0]]]]
        )
        weight = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 12.0],
            ]
        )

        contributions = contact_channel_contributions(features, weight)

        # mean |feature| = [2, 3, 4], column norms = [5, 0, 12].
        torch.testing.assert_close(
            contributions,
            torch.tensor([10.0, 0.0, 48.0]),
        )

    def test_contact_model_rejects_invalid_feature_mask(self):
        cfg = _model_config(True)
        cfg.contact_feature_mask = [1.0, 0.0]

        with self.assertRaisesRegex(ValueError, "must contain 3"):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_contact_model_accepts_ablation_controls_without_shape_change(self):
        cfg = _model_config(True)
        cfg.contact_feature_mask = [0.0, 1.0, 1.0]
        cfg.contact_bias_scale = 0.0

        model = MDM_ST(8, 1, n_feats=3, model_config=cfg)

        self.assertEqual(model.contact_feature_mask, (0.0, 1.0, 1.0))
        self.assertEqual(model.contact_bias_scale, 0.0)
        self.assertEqual(model.contact_encoder.in_features, 3)
        self.assertEqual(model.contact_encoder.out_features, 64)

    def test_ablation_controls_do_not_change_checkpoint_keys_or_shapes(self):
        baseline = MDM_ST(8, 1, n_feats=3, model_config=_model_config(True))
        cfg = _model_config(True)
        cfg.contact_feature_mask = [1.0, 0.0, 1.0]
        cfg.contact_bias_scale = 0.0
        ablation = MDM_ST(8, 1, n_feats=3, model_config=cfg)

        baseline_state = baseline.state_dict()
        ablation_state = ablation.state_dict()
        self.assertEqual(baseline_state.keys(), ablation_state.keys())
        self.assertEqual(
            {key: value.shape for key, value in baseline_state.items()},
            {key: value.shape for key, value in ablation_state.items()},
        )

    def test_default_controls_match_linear_layer_exactly(self):
        torch.manual_seed(0)
        features = torch.randn(2, 3, 4, 3)
        encoder = torch.nn.Linear(3, 8)

        expected = encoder(features)
        masked = apply_contact_feature_mask(features)
        actual = F.linear(masked, encoder.weight, encoder.bias * 1.0)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_initialized_contact_conditioning_preserves_forward(self):
        torch.manual_seed(0)
        model = MDM_ST(4, 1, n_feats=3, model_config=_model_config(True)).eval()
        batch_size = 1
        x = torch.randn(batch_size, 1, 4, 3)
        init_pc = torch.randn(batch_size, 2, 4, 3)
        timesteps = torch.zeros(batch_size, dtype=torch.long)
        force = torch.zeros(batch_size, 1, 3)
        E = torch.ones(batch_size, 1)
        nu = torch.ones(batch_size, 1)
        drag_mask = torch.zeros(batch_size, 4, 1)
        drag_point = torch.zeros(batch_size, 1, 4)
        floor = torch.zeros(batch_size, 1)
        gravity = torch.zeros(batch_size, 1, dtype=torch.long)
        start_vel = torch.randn(batch_size, 4, 3)

        with torch.no_grad():
            contact_output = model(
                x, timesteps, init_pc, force, E, nu, drag_mask, drag_point,
                floor, gravity, start_vel=start_vel,
            )
            model.contact_particle_cond = False
            baseline_output = model(
                x, timesteps, init_pc, force, E, nu, drag_mask, drag_point,
                floor, gravity, start_vel=start_vel,
            )

        self.assertEqual(contact_output.shape, (batch_size, 1, 4, 3))
        torch.testing.assert_close(contact_output, baseline_output, rtol=0, atol=0)

    def test_features_encode_gap_velocity_and_proximity(self):
        points = torch.zeros(1, 3, 2, 3)
        points[0, :, 0, 1] = torch.tensor([0.0, 0.1, 0.15])
        points[0, :, 1, 1] = torch.tensor([1.0, 0.8, 0.5])
        floor = torch.tensor([[0.0]])
        start_velocity = torch.zeros(1, 2, 3)
        start_velocity[0, :, 1] = torch.tensor([-0.2, 0.3])

        features = build_contact_features(
            points, floor, start_velocity=start_velocity, sigma=1.0
        )

        self.assertEqual(features.shape, (1, 3, 2, 3))
        torch.testing.assert_close(features[..., 0], points[..., 1])
        torch.testing.assert_close(
            features[0, :, 0, 1], torch.tensor([-0.2, 0.1, 0.05])
        )
        torch.testing.assert_close(
            features[0, :, 1, 1], torch.tensor([0.3, -0.2, -0.3])
        )
        self.assertAlmostEqual(features[0, 0, 0, 2].item(), 1.0, places=6)
        self.assertAlmostEqual(
            features[0, 0, 1, 2].item(), float(np.exp(-1.0)), places=6
        )

    def test_weighted_losses_focus_on_near_floor_particles(self):
        target = torch.zeros(1, 1, 2, 3)
        target[0, 0, :, 1] = torch.tensor([0.01, 1.0])
        last_source = target.clone()
        floor = torch.tensor([[0.0]])

        near_error = target.clone()
        near_error[0, 0, 0, 0] = 1.0
        near_pos, _, near_fraction = contact_weighted_losses(
            near_error,
            target,
            last_source,
            floor,
            margin=0.1,
            temperature=0.01,
        )

        far_error = target.clone()
        far_error[0, 0, 1, 0] = 1.0
        far_pos, _, _ = contact_weighted_losses(
            far_error,
            target,
            last_source,
            floor,
            margin=0.1,
            temperature=0.01,
        )

        self.assertGreater(near_pos.item(), 0.3)
        self.assertLess(far_pos.item(), 1e-4)
        self.assertGreater(near_fraction.item(), 0.49)
        self.assertLess(near_fraction.item(), 0.51)

    def test_contact_window_starts_target_contact_frames(self):
        min_y = np.full(25, 4.0, dtype=np.float32)
        min_y[11:] = 1.05

        starts = find_contact_window_starts(
            min_y,
            floor_height=1.0,
            max_start=19,
            input_frames=5,
            output_frames=1,
            frame_interval=1,
            margin=0.04,
        )

        self.assertEqual(starts, [4, 5, 6, 7, 8])


if __name__ == "__main__":
    unittest.main()
