import unittest
import os
import tempfile

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import model.contact_adapter as contact_adapter_module
from dataset.traj_dataset import TrajDataset
from model.contact_adapter import FactorizedContactAdapter
from model.spacetime import MDM_ST, PointEmbed
from options import TestingConfig, TrainingConfig
from utils.contact import (
    apply_contact_feature_mask,
    build_contact_features,
    contact_channel_contributions,
    contact_weighted_losses,
    find_contact_window_starts,
)


class FactorizedContactAdapterTests(unittest.TestCase):
    def test_declares_feature_order_and_branch_indices(self):
        self.assertEqual(
            getattr(contact_adapter_module, "CONTACT_FEATURE_ORDER", None),
            (
                "signed_gap",
                "displacement_x",
                "displacement_y",
                "displacement_z",
                "proximity",
            ),
        )
        self.assertEqual(
            getattr(contact_adapter_module, "BOUNDARY_FEATURE_INDICES", None),
            (0, 4),
        )
        self.assertEqual(
            getattr(contact_adapter_module, "NORMAL_FEATURE_INDICES", None),
            (2,),
        )
        self.assertEqual(
            getattr(contact_adapter_module, "TANGENTIAL_FEATURE_INDICES", None),
            (1, 3),
        )

    def test_rejects_nonpositive_latent_dim(self):
        for latent_dim in (0, -1):
            with self.subTest(latent_dim=latent_dim):
                with self.assertRaisesRegex(
                    ValueError,
                    "latent_dim must be positive",
                ):
                    FactorizedContactAdapter(latent_dim=latent_dim)

    def test_fixed_feature_groups_sum_with_gated_tangential_branch(self):
        adapter = FactorizedContactAdapter(latent_dim=2)
        features = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]])

        with torch.no_grad():
            adapter.boundary.weight.copy_(
                torch.tensor([[1.0, 10.0], [100.0, 1000.0]])
            )
            adapter.normal.weight.copy_(torch.tensor([[2.0], [3.0]]))
            adapter.tangential.weight.copy_(
                torch.tensor([[4.0, 5.0], [6.0, 7.0]])
            )
            adapter.shared_bias.copy_(torch.tensor([8.0, 9.0]))
            adapter.tangential_gate.copy_(torch.atanh(torch.tensor(0.5)))

        actual = adapter(features)
        expected = torch.tensor([[[[79.0, 5138.0]]]])

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bias_scale_applies_only_to_shared_bias(self):
        adapter = FactorizedContactAdapter(latent_dim=2)
        features = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]])

        with torch.no_grad():
            adapter.boundary.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            )
            adapter.normal.weight.copy_(torch.tensor([[5.0], [6.0]]))
            adapter.shared_bias.copy_(torch.tensor([2.0, -4.0]))

        without_bias = adapter(features, bias_scale=0.0)
        with_half_bias = adapter(features, bias_scale=0.5)

        torch.testing.assert_close(
            without_bias,
            torch.tensor([[[[26.0, 41.0]]]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            with_half_bias - without_bias,
            0.5 * adapter.shared_bias.detach().reshape(1, 1, 1, -1),
            rtol=0,
            atol=0,
        )

    def test_cpu_bfloat16_autocast_preserves_output_dtype_and_gradients(self):
        adapter = FactorizedContactAdapter(latent_dim=4)
        features = torch.ones(2, 3, 4, 5)
        with torch.no_grad():
            adapter.tangential.weight.fill_(1.0)
            adapter.shared_bias.fill_(1.0)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = adapter(features)
        output.sum().backward()

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertIsNotNone(adapter.tangential_gate.grad)
        self.assertIsNotNone(adapter.shared_bias.grad)

    def test_latent_256_has_expected_parameterization(self):
        adapter = FactorizedContactAdapter(latent_dim=256)

        self.assertEqual(adapter.boundary.in_features, 2)
        self.assertEqual(adapter.boundary.out_features, 256)
        self.assertIsNone(adapter.boundary.bias)
        self.assertEqual(adapter.normal.in_features, 1)
        self.assertEqual(adapter.normal.out_features, 256)
        self.assertIsNone(adapter.normal.bias)
        self.assertEqual(adapter.tangential.in_features, 2)
        self.assertEqual(adapter.tangential.out_features, 256)
        self.assertIsNone(adapter.tangential.bias)
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.parameters()),
            1537,
        )

    def test_initialization_preserves_zero_output_and_tangential_weights(self):
        torch.manual_seed(0)
        adapter = FactorizedContactAdapter(latent_dim=16)
        features = torch.randn(2, 3, 4, 5)

        self.assertEqual(torch.count_nonzero(adapter.boundary.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.normal.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.shared_bias).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.tangential_gate).item(), 0)
        self.assertGreater(
            torch.count_nonzero(adapter.tangential.weight).item(),
            0,
        )
        torch.testing.assert_close(
            adapter(features),
            torch.zeros(2, 3, 4, 16),
            rtol=0,
            atol=0,
        )

    def test_zero_gate_blocks_tangential_weight_gradient_but_learns_gate(self):
        adapter = FactorizedContactAdapter(latent_dim=4)
        features = torch.ones(2, 3, 4, 5)
        with torch.no_grad():
            adapter.tangential.weight.fill_(1.0)

        adapter(features).sum().backward()

        self.assertGreater(
            torch.count_nonzero(adapter.boundary.weight.grad).item(),
            0,
        )
        self.assertGreater(
            torch.count_nonzero(adapter.normal.weight.grad).item(),
            0,
        )
        self.assertEqual(
            torch.count_nonzero(adapter.tangential.weight.grad).item(),
            0,
        )
        self.assertNotEqual(adapter.tangential_gate.grad.item(), 0.0)

    def test_nonzero_gate_enables_tangential_weight_gradient(self):
        adapter = FactorizedContactAdapter(latent_dim=4)
        features = torch.ones(2, 3, 4, 5)
        with torch.no_grad():
            adapter.tangential_gate.fill_(1.0)

        adapter(features).sum().backward()

        self.assertGreater(
            torch.count_nonzero(adapter.tangential.weight.grad).item(),
            0,
        )

    def test_rejects_non_bfn5_inputs(self):
        adapter = FactorizedContactAdapter(latent_dim=4)
        invalid_shapes = (
            (2, 3, 5),
            (2, 3, 4, 4),
            (2, 3, 4, 5, 1),
        )

        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, r"\(B, F, N, 5\)"):
                    adapter(torch.zeros(shape))


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
        "contact_injection_mode": "separate",
        "contact_feature_sigma": 0.04,
    })


def _small_model_batch(batch_size=1, n_frames=1, n_points=4, cond_frames=2):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(
        batch_size,
        n_frames,
        n_points,
        3,
        generator=generator,
    )
    init_pc = torch.zeros(batch_size, cond_frames, n_points, 3)
    for frame_index in range(cond_frames):
        init_pc[:, frame_index, :, 1] = 0.01 * (frame_index + 1)
    return {
        "x": x,
        "timesteps": torch.zeros(batch_size, dtype=torch.long),
        "init_pc": init_pc,
        "force": torch.zeros(batch_size, 1, 3),
        "E": torch.ones(batch_size, 1),
        "nu": torch.ones(batch_size, 1),
        "drag_mask": torch.zeros(batch_size, n_points, 1),
        "drag_point": torch.zeros(batch_size, 1, 4),
        "floor_height": torch.zeros(batch_size, 1),
        "gravity_label": torch.zeros(batch_size, 1, dtype=torch.long),
        "start_vel": torch.full((batch_size, n_points, 3), 0.1),
    }


class PointEmbedTests(unittest.TestCase):
    def test_extra_feature_columns_are_zero_initialized(self):
        encoder = PointEmbed(hidden_dim=96, dim=64, extra_feature_dim=3)

        self.assertEqual(encoder.mlp.in_features, 102)
        self.assertEqual(
            torch.count_nonzero(encoder.mlp.weight[:, -3:]).item(),
            0,
        )

    def test_extra_features_preserve_legacy_initialization_and_rng(self):
        seed = 1234
        torch.manual_seed(seed)
        legacy_encoder = PointEmbed(
            hidden_dim=96,
            dim=64,
            extra_feature_dim=0,
        )
        legacy_rng_state = torch.random.get_rng_state()

        torch.manual_seed(seed)
        shared_encoder = PointEmbed(
            hidden_dim=96,
            dim=64,
            extra_feature_dim=3,
        )
        shared_rng_state = torch.random.get_rng_state()

        torch.testing.assert_close(
            shared_encoder.mlp.weight[:, :99],
            legacy_encoder.mlp.weight,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            shared_encoder.mlp.bias,
            legacy_encoder.mlp.bias,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            torch.count_nonzero(shared_encoder.mlp.weight[:, 99:]).item(),
            0,
        )
        torch.testing.assert_close(
            shared_rng_state,
            legacy_rng_state,
            rtol=0,
            atol=0,
        )

        points = torch.linspace(-1.0, 1.0, steps=24).reshape(2, 4, 3)
        extra_features = torch.zeros(2, 4, 3)
        torch.testing.assert_close(
            shared_encoder(points, extra_features=extra_features),
            legacy_encoder(points),
            rtol=0,
            atol=0,
        )

    def test_zero_initialized_extra_features_preserve_output(self):
        torch.manual_seed(0)
        encoder = PointEmbed(hidden_dim=96, dim=64, extra_feature_dim=3)
        points = torch.randn(2, 4, 3)
        contact_a = torch.randn(2, 4, 3)
        contact_b = torch.randn(2, 4, 3)

        output_a = encoder(points, extra_features=contact_a)
        output_b = encoder(points, extra_features=contact_b)

        torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)

    def test_configured_extra_features_are_required(self):
        encoder = PointEmbed(hidden_dim=96, dim=64, extra_feature_dim=3)
        points = torch.randn(2, 4, 3)

        with self.assertRaisesRegex(ValueError, "extra_features"):
            encoder(points)

    def test_extra_features_reject_wrong_last_dimension(self):
        encoder = PointEmbed(hidden_dim=96, dim=64, extra_feature_dim=3)
        points = torch.randn(2, 4, 3)
        extra_features = torch.randn(2, 4, 2)

        with self.assertRaisesRegex(ValueError, "last dimension"):
            encoder(points, extra_features=extra_features)


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

    def test_contact_injection_mode_defaults_to_separate(self):
        default_cfg = _model_config(True)
        del default_cfg.contact_injection_mode
        separate_cfg = _model_config(True)
        default_model = MDM_ST(
            4,
            1,
            n_feats=3,
            model_config=default_cfg,
        ).eval()
        separate_model = MDM_ST(
            4,
            1,
            n_feats=3,
            model_config=separate_cfg,
        ).eval()

        default_state = default_model.state_dict()
        separate_state = separate_model.state_dict()
        self.assertEqual(default_state.keys(), separate_state.keys())
        self.assertEqual(
            {key: value.shape for key, value in default_state.items()},
            {key: value.shape for key, value in separate_state.items()},
        )

        reference_state = {
            key: value.clone()
            for key, value in separate_state.items()
        }
        default_model.load_state_dict(reference_state)
        separate_model.load_state_dict(reference_state)
        batch = _small_model_batch()

        with torch.no_grad():
            default_output = default_model(**batch)
            separate_output = separate_model(**batch)

        self.assertTrue(hasattr(default_model, "contact_encoder"))
        self.assertTrue(hasattr(separate_model, "contact_encoder"))
        torch.testing.assert_close(
            default_output,
            separate_output,
            rtol=0,
            atol=0,
        )

    def test_shared_contact_uses_input_encoder_without_contact_encoder(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "shared"

        model = MDM_ST(8, 1, n_feats=3, model_config=cfg)

        self.assertFalse(hasattr(model, "contact_encoder"))
        self.assertEqual(model.input_encoder.extra_feature_dim, 3)
        self.assertEqual(model.input_encoder.mlp.in_features, 102)

    def test_shared_contact_forwards_contact_features_only_for_condition_frames(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "shared"
        model = MDM_ST(4, 1, n_feats=3, model_config=cfg).eval()
        captured = {}
        original_forward = model.input_encoder.forward

        def capture_forward(points, extra_features=None):
            captured["extra_features"] = (
                None
                if extra_features is None
                else extra_features.detach().clone()
            )
            if extra_features is None:
                return original_forward(points)
            return original_forward(points, extra_features=extra_features)

        model.input_encoder.forward = capture_forward
        batch = _small_model_batch()
        batch_size, n_frames, n_points, _ = batch["x"].shape

        with torch.no_grad():
            output = model(**batch)

        self.assertEqual(output.shape, (batch_size, n_frames, n_points, 3))
        extra_features = captured["extra_features"]
        self.assertIsNotNone(extra_features)
        n_frame_slots = cfg.cond_frames + n_frames
        self.assertEqual(
            extra_features.shape,
            (batch_size * n_frame_slots, n_points, 3),
        )
        frame_features = extra_features.reshape(
            batch_size,
            n_frame_slots,
            n_points,
            3,
        )
        expected_contact = build_contact_features(
            batch["init_pc"],
            batch["floor_height"],
            start_velocity=batch["start_vel"],
            sigma=cfg.contact_feature_sigma,
            velocity_mode="vertical",
        )
        expected_contact = apply_contact_feature_mask(
            expected_contact,
            model.contact_feature_mask,
        )
        torch.testing.assert_close(
            frame_features[:, :cfg.cond_frames],
            expected_contact,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            frame_features[:, -1],
            torch.zeros_like(frame_features[:, -1]),
            rtol=0,
            atol=0,
        )

    def test_shared_contact_rejects_xyz_velocity_mode(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "shared"
        cfg.contact_velocity_mode = "xyz"

        with self.assertRaisesRegex(ValueError, "vertical"):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_shared_contact_rejects_force_as_latent(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "shared"
        cfg.force_as_latent = True

        with self.assertRaisesRegex(ValueError, "force_as_latent"):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_shared_contact_rejects_disabled_point_embedding(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "shared"
        cfg.point_embed = False

        with self.assertRaisesRegex(ValueError, "point_embed"):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_shared_contact_rejects_disabled_frame_conditioning(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "shared"
        cfg.frame_cond = False

        with self.assertRaisesRegex(ValueError, "frame_cond"):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_contact_model_rejects_unknown_injection_mode(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "unknown"

        with self.assertRaisesRegex(
            ValueError,
            "contact_injection_mode must be one of "
            "'separate', 'shared', or 'factorized'; got 'unknown'",
        ):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_factorized_contact_requires_particle_conditioning(self):
        cfg = _model_config(False)
        cfg.contact_injection_mode = "factorized"
        cfg.contact_velocity_mode = "xyz"

        with self.assertRaisesRegex(
            ValueError,
            "contact_injection_mode='factorized' requires "
            "contact_particle_cond=true",
        ):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_factorized_contact_requires_xyz_velocity_mode(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "factorized"

        with self.assertRaisesRegex(
            ValueError,
            "contact_injection_mode='factorized' requires "
            "contact_velocity_mode='xyz'",
        ):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_factorized_contact_uses_adapter_without_contact_encoder(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "factorized"
        cfg.contact_velocity_mode = "xyz"

        model = MDM_ST(8, 1, n_feats=3, model_config=cfg)

        self.assertIsInstance(model.contact_adapter, FactorizedContactAdapter)
        self.assertFalse(hasattr(model, "contact_encoder"))

    def test_factorized_contact_preserves_dit_initialization_rng(self):
        separate_cfg = _model_config(True)
        separate_cfg.contact_velocity_mode = "xyz"
        factorized_cfg = _model_config(True)
        factorized_cfg.contact_injection_mode = "factorized"
        factorized_cfg.contact_velocity_mode = "xyz"

        torch.manual_seed(1234)
        separate_model = MDM_ST(
            8,
            1,
            n_feats=3,
            model_config=separate_cfg,
        )
        separate_state = {
            key: value.detach().clone()
            for key, value in separate_model.dit.state_dict().items()
        }

        torch.manual_seed(1234)
        factorized_model = MDM_ST(
            8,
            1,
            n_feats=3,
            model_config=factorized_cfg,
        )
        factorized_state = factorized_model.dit.state_dict()

        self.assertEqual(separate_state.keys(), factorized_state.keys())
        for key in separate_state:
            with self.subTest(key=key):
                torch.testing.assert_close(
                    factorized_state[key],
                    separate_state[key],
                    rtol=0,
                    atol=0,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_factorized_contact_preserves_dit_rng_on_default_cuda(self):
        separate_cfg = _model_config(True)
        separate_cfg.contact_velocity_mode = "xyz"
        factorized_cfg = _model_config(True)
        factorized_cfg.contact_injection_mode = "factorized"
        factorized_cfg.contact_velocity_mode = "xyz"
        original_default_device = torch.get_default_device()
        cuda_device = torch.device("cuda", torch.cuda.current_device())

        try:
            torch.set_default_device(cuda_device)
            torch.manual_seed(5678)
            separate_model = MDM_ST(
                8,
                1,
                n_feats=3,
                model_config=separate_cfg,
            )
            separate_state = {
                key: value.detach().clone()
                for key, value in separate_model.dit.state_dict().items()
            }

            torch.manual_seed(5678)
            factorized_model = MDM_ST(
                8,
                1,
                n_feats=3,
                model_config=factorized_cfg,
            )
            factorized_state = factorized_model.dit.state_dict()
        finally:
            torch.set_default_device(original_default_device)

        self.assertEqual(separate_state.keys(), factorized_state.keys())
        for key in separate_state:
            with self.subTest(key=key):
                torch.testing.assert_close(
                    factorized_state[key],
                    separate_state[key],
                    rtol=0,
                    atol=0,
                )

    def test_factorized_step_zero_matches_separate_xyz_exactly(self):
        separate_cfg = _model_config(True)
        separate_cfg.contact_velocity_mode = "xyz"
        factorized_cfg = _model_config(True)
        factorized_cfg.contact_injection_mode = "factorized"
        factorized_cfg.contact_velocity_mode = "xyz"

        torch.manual_seed(4321)
        separate_model = MDM_ST(
            4,
            1,
            n_feats=3,
            model_config=separate_cfg,
        ).eval()
        torch.manual_seed(4321)
        factorized_model = MDM_ST(
            4,
            1,
            n_feats=3,
            model_config=factorized_cfg,
        ).eval()
        batch = _small_model_batch()

        with torch.no_grad():
            separate_output = separate_model(**batch)
            factorized_output = factorized_model(**batch)

        torch.testing.assert_close(
            factorized_output,
            separate_output,
            rtol=0,
            atol=0,
        )

    def test_factorized_contact_casts_fp32_features_for_bfloat16_model(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "factorized"
        cfg.contact_velocity_mode = "xyz"
        cfg.frame_cond = False
        cfg.cond_frames = 0
        cfg.pred_offset = False
        model = MDM_ST(4, 1, n_feats=3, model_config=cfg).to(
            torch.bfloat16
        ).eval()
        batch = _small_model_batch(cond_frames=1)
        for key in (
            "x",
            "force",
            "E",
            "nu",
            "drag_mask",
            "drag_point",
            "floor_height",
        ):
            batch[key] = batch[key].to(torch.bfloat16)
        captured_adapter_spec = []

        def capture_adapter_features(_module, inputs):
            captured_adapter_spec.append(
                (inputs[0].dtype, inputs[0].device)
            )

        hook = model.contact_adapter.register_forward_pre_hook(
            capture_adapter_features
        )
        try:
            with torch.no_grad():
                output = model(**batch)
        finally:
            hook.remove()

        adapter_parameter = model.contact_adapter.boundary.weight
        self.assertEqual(batch["init_pc"].dtype, torch.float32)
        self.assertEqual(
            captured_adapter_spec,
            [(adapter_parameter.dtype, adapter_parameter.device)],
        )
        self.assertEqual(output.dtype, torch.bfloat16)

    def test_factorized_contact_injects_only_real_condition_frames(self):
        cfg = _model_config(True)
        cfg.contact_injection_mode = "factorized"
        cfg.contact_velocity_mode = "xyz"
        cfg.contact_feature_mask = [1.0, 0.0, 1.0, 0.0, 1.0]
        cfg.contact_bias_scale = 0.25
        cfg.mask_cond = True
        model = MDM_ST(4, 1, n_feats=3, model_config=cfg).eval()
        batch = _small_model_batch()
        batch["drag_mask"] = batch["drag_mask"].unsqueeze(1)
        batch["start_vel"].zero_()
        batch["start_vel"][..., 0] = 0.25
        batch["start_vel"][..., 1] = 0.1
        batch["start_vel"][..., 2] = -0.5
        captured_hidden = []
        captured_adapter_features = []

        with torch.no_grad():
            model.contact_adapter.boundary.weight.fill_(1.0)
            model.contact_adapter.normal.weight.fill_(2.0)
            model.contact_adapter.tangential.weight.fill_(3.0)
            model.contact_adapter.tangential_gate.fill_(1.0)
            model.contact_adapter.shared_bias.fill_(4.0)
            model.start_vel_encoder.weight.zero_()
            model.start_vel_encoder.bias.zero_()

        def capture_hidden(_module, inputs):
            captured_hidden.append(inputs[0].detach().clone())

        def capture_adapter_features(_module, inputs):
            captured_adapter_features.append(inputs[0].detach().clone())

        dit_hook = model.dit.register_forward_pre_hook(capture_hidden)
        adapter_hook = model.contact_adapter.register_forward_pre_hook(
            capture_adapter_features
        )
        try:
            with torch.no_grad():
                model(**batch)
                model.contact_particle_cond = False
                model(**batch)
        finally:
            adapter_hook.remove()
            dit_hook.remove()

        unmasked_contact_features = build_contact_features(
            batch["init_pc"],
            batch["floor_height"].unsqueeze(1),
            start_velocity=batch["start_vel"],
            sigma=cfg.contact_feature_sigma,
            velocity_mode="xyz",
        )
        self.assertGreater(
            torch.count_nonzero(unmasked_contact_features[..., 1]).item(),
            0,
        )
        self.assertGreater(
            torch.count_nonzero(unmasked_contact_features[..., 3]).item(),
            0,
        )
        contact_features = apply_contact_feature_mask(
            unmasked_contact_features,
            cfg.contact_feature_mask,
        )
        self.assertEqual(len(captured_adapter_features), 1)
        torch.testing.assert_close(
            captured_adapter_features[0],
            contact_features,
            rtol=0,
            atol=0,
        )
        for channel_index in (1, 3):
            with self.subTest(masked_channel=channel_index):
                self.assertEqual(
                    torch.count_nonzero(
                        captured_adapter_features[0][..., channel_index]
                    ).item(),
                    0,
                )
        for channel_index in (0, 2, 4):
            with self.subTest(retained_channel=channel_index):
                torch.testing.assert_close(
                    captured_adapter_features[0][..., channel_index],
                    unmasked_contact_features[..., channel_index],
                    rtol=0,
                    atol=0,
                )
        expected_contact = model.contact_adapter(
            contact_features,
            bias_scale=cfg.contact_bias_scale,
        ).to(captured_hidden[0].dtype)

        self.assertEqual(captured_hidden[0].shape[1], 4)
        torch.testing.assert_close(
            captured_hidden[0][:, 0],
            captured_hidden[1][:, 0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            captured_hidden[0][:, 1:3],
            captured_hidden[1][:, 1:3] + expected_contact,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            captured_hidden[0][:, 3],
            captured_hidden[1][:, 3],
            rtol=0,
            atol=0,
        )

    def test_xyz_contact_encoder_uses_five_features(self):
        cfg = _model_config(True)
        cfg.contact_velocity_mode = "xyz"

        model = MDM_ST(8, 1, n_feats=3, model_config=cfg)

        self.assertEqual(model.contact_encoder.in_features, 5)
        self.assertEqual(model.contact_encoder.out_features, 64)
        self.assertEqual(model.contact_feature_mask, (1.0,) * 5)
        self.assertEqual(model.contact_velocity_mode, "xyz")
        self.assertEqual(
            model.contact_feature_names,
            (
                "signed_gap",
                "displacement_x",
                "displacement_y",
                "displacement_z",
                "proximity",
            ),
        )

    def test_contact_feature_mask_disables_only_selected_channels(self):
        features = torch.tensor([[[[1.0, 2.0, 3.0]]]])

        masked = apply_contact_feature_mask(features, [1.0, 0.0, 1.0])

        torch.testing.assert_close(
            masked,
            torch.tensor([[[[1.0, 0.0, 3.0]]]]),
        )

    def test_contact_feature_mask_supports_five_channels(self):
        features = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0]]]])

        masked = apply_contact_feature_mask(
            features,
            [1.0, 0.0, 1.0, 0.0, 1.0],
        )

        torch.testing.assert_close(
            masked,
            torch.tensor([[[[1.0, 0.0, 3.0, 0.0, 5.0]]]]),
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

    def test_contact_channel_contributions_support_five_channels(self):
        features = torch.ones(1, 1, 2, 5)
        weight = torch.eye(5)

        contributions = contact_channel_contributions(features, weight)

        torch.testing.assert_close(contributions, torch.ones(5))

    def test_contact_model_rejects_invalid_feature_mask(self):
        cfg = _model_config(True)
        cfg.contact_feature_mask = [1.0, 0.0]

        with self.assertRaisesRegex(ValueError, "must contain 3"):
            MDM_ST(8, 1, n_feats=3, model_config=cfg)

    def test_xyz_contact_model_rejects_three_channel_feature_mask(self):
        cfg = _model_config(True)
        cfg.contact_velocity_mode = "xyz"
        cfg.contact_feature_mask = [1.0, 1.0, 1.0]

        with self.assertRaisesRegex(ValueError, "must contain 5"):
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

    def test_zero_initialized_xyz_contact_conditioning_preserves_forward(self):
        torch.manual_seed(0)
        cfg = _model_config(True)
        cfg.contact_velocity_mode = "xyz"
        model = MDM_ST(4, 1, n_feats=3, model_config=cfg).eval()
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

    def test_features_encode_full_xyz_displacement(self):
        points = torch.tensor(
            [[
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                [[0.1, 0.2, 0.3], [0.8, 0.9, 1.4]],
                [[0.4, 0.1, 0.5], [0.7, 0.5, 1.2]],
            ]]
        )
        floor = torch.tensor([[0.0]])
        start_velocity = torch.tensor(
            [[[0.5, -0.2, 0.1], [-0.4, 0.3, 0.2]]]
        )

        features = build_contact_features(
            points,
            floor,
            start_velocity=start_velocity,
            sigma=1.0,
            velocity_mode="xyz",
        )

        self.assertEqual(features.shape, (1, 3, 2, 5))
        torch.testing.assert_close(features[..., 0], points[..., 1])
        torch.testing.assert_close(features[0, 0, :, 1:4], start_velocity[0])
        torch.testing.assert_close(
            features[:, 1:, :, 1:4],
            points[:, 1:, :, :3] - points[:, :-1, :, :3],
        )
        torch.testing.assert_close(
            features[..., 4],
            torch.exp(-(torch.relu(points[..., 1]) ** 2)),
        )

    def test_features_reject_unknown_velocity_mode(self):
        points = torch.zeros(1, 2, 2, 3)
        floor = torch.tensor([[0.0]])

        with self.assertRaisesRegex(ValueError, "contact_velocity_mode"):
            build_contact_features(points, floor, velocity_mode="bad")

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
