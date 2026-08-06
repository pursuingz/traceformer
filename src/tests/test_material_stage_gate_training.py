import os
import json
import tempfile
import unittest

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file
from torch import nn

from train_material_stage_gates import create_gate_model
from model.spacetime import MDM_ST
from utils.material_stage_gate_training import (
    calibrate_material_rows,
    frame_loss_weights,
    freeze_for_gate_training,
    gate_objective,
    gate_rollout_loss,
    load_b3a_into_b3b,
    MaterialBestRowTracker,
    save_gate_artifacts,
)


class TinyGateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.gate_logits = nn.Parameter(torch.zeros(3, 4))

    def material_stage_gates(self):
        return 2.0 * torch.sigmoid(self.gate_logits)


class GateCheckpointTest(unittest.TestCase):
    def test_b3a_checkpoint_may_miss_only_gate_logits(self):
        with tempfile.TemporaryDirectory() as root:
            source = TinyGateModel()
            with torch.no_grad():
                source.backbone.weight.fill_(2.0)
                source.backbone.bias.fill_(3.0)
            state = {
                key: value.detach().clone()
                for key, value in source.state_dict().items()
                if key != "gate_logits"
            }
            checkpoint = os.path.join(root, "model.safetensors")
            save_file(state, checkpoint)

            target = TinyGateModel()
            load_b3a_into_b3b(target, checkpoint)

            torch.testing.assert_close(
                target.backbone.weight,
                torch.full_like(target.backbone.weight, 2.0),
            )
            torch.testing.assert_close(
                target.backbone.bias,
                torch.full_like(target.backbone.bias, 3.0),
            )
            torch.testing.assert_close(
                target.gate_logits,
                torch.zeros_like(target.gate_logits),
            )

    def test_checkpoint_rejects_any_additional_missing_key(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, "model.safetensors")
            save_file({}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "backbone.weight"):
                load_b3a_into_b3b(TinyGateModel(), checkpoint)

    def test_checkpoint_rejects_unexpected_key(self):
        with tempfile.TemporaryDirectory() as root:
            source = TinyGateModel()
            state = {
                key: value.detach().clone()
                for key, value in source.state_dict().items()
                if key != "gate_logits"
            }
            state["unexpected"] = torch.zeros(1)
            checkpoint = os.path.join(root, "model.safetensors")
            save_file(state, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                load_b3a_into_b3b(TinyGateModel(), checkpoint)


class GateFreezeAndLossTest(unittest.TestCase):
    def test_freeze_leaves_exactly_twelve_gate_parameters_trainable(self):
        model = TinyGateModel()
        gate = freeze_for_gate_training(model)

        trainable = [
            (name, parameter.numel())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertIs(gate, model.gate_logits)
        self.assertEqual(trainable, [("gate_logits", 12)])

    def test_frame_weights_encode_full_and_last_third_means(self):
        weights = frame_loss_weights(6, long_weight=0.5)
        torch.testing.assert_close(
            weights,
            torch.tensor([1 / 6, 1 / 6, 1 / 6, 1 / 6, 5 / 12, 5 / 12]),
        )

    def test_objective_matches_hand_computed_loss_and_identity_regularizer(self):
        frame_losses = torch.arange(1.0, 7.0)
        gates = torch.ones(3, 4)
        result = gate_objective(
            frame_losses,
            gates,
            material_id=1,
            long_weight=0.5,
            reg_weight=1.0e-3,
        )

        self.assertAlmostEqual(result["full_mse"].item(), 3.5)
        self.assertAlmostEqual(result["long_mse"].item(), 5.5)
        self.assertAlmostEqual(result["regularizer"].item(), 0.0)
        self.assertAlmostEqual(result["objective"].item(), 6.25)


class RecordingRolloutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.zeros(3, 4))
        self.windows = []
        self.start_velocities = []
        self.window_requires_grad = []

    def material_stage_gates(self):
        return 2.0 * torch.sigmoid(self.gate_logits)

    def forward(
        self,
        x,
        timesteps,
        init_pc,
        force,
        E,
        nu,
        drag_mask,
        drag_point,
        floor_height,
        gravity_label=None,
        coeff=None,
        y=None,
        null_emb=None,
        start_vel=None,
        points_rest=None,
    ):
        self.windows.append(init_pc.detach().clone())
        self.start_velocities.append(start_vel.detach().clone())
        self.window_requires_grad.append(init_pc.requires_grad)
        gates = self.material_stage_gates()[y.long(), 0]
        return init_pc[:, -1:] + 1.0 + (gates - 1.0).view(-1, 1, 1, 1)


class BackwardRecorder:
    def __init__(self):
        self.calls = 0

    def backward(self, loss):
        self.calls += 1
        loss.backward()


def rollout_sample():
    points = torch.zeros(1, 5, 2, 3)
    points[0, :, :, 0] = torch.arange(5.0).view(5, 1)
    return {
        "points_src": points,
        "future_gt": torch.zeros(1, 3, 2, 3),
        "start_vel": torch.full((1, 2, 3), 0.25),
        "force": torch.zeros(1, 1, 3),
        "E": torch.full((1, 1), 6.0),
        "nu": torch.full((1, 1), 0.3),
        "mask": torch.zeros(1, 1, 2, 1),
        "drag_point": torch.zeros(1, 1, 4),
        "floor_height": torch.zeros(1, 1),
        "gravity": torch.ones(1, 1, dtype=torch.long),
        "base_drag_coeff": torch.zeros(1, 1, 1),
        "mat_type": torch.ones(1, dtype=torch.long),
        "points_rest": torch.zeros(1, 2, 3),
    }


class GateRolloutTest(unittest.TestCase):
    def test_rollout_slides_window_and_detaches_each_prediction(self):
        model = RecordingRolloutModel()
        accelerator = BackwardRecorder()

        result = gate_rollout_loss(
            model,
            rollout_sample(),
            material_id=1,
            long_weight=0.5,
            reg_weight=1.0e-3,
            accelerator=accelerator,
            backward=True,
            noise_seed=0,
        )

        self.assertEqual(len(model.windows), 3)
        torch.testing.assert_close(
            model.windows[1][0, :, 0, 0],
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        )
        torch.testing.assert_close(
            model.windows[2][0, :, 0, 0],
            torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        self.assertEqual(model.window_requires_grad, [False, False, False])
        self.assertEqual(tuple(result["frame_mse"].shape), (3,))
        self.assertEqual(accelerator.calls, 4)

    def test_rollout_uses_dataset_velocity_then_window_forward_difference(self):
        model = RecordingRolloutModel()
        gate_rollout_loss(
            model,
            rollout_sample(),
            material_id=1,
            long_weight=0.5,
            reg_weight=0.0,
            backward=False,
            noise_seed=0,
        )

        torch.testing.assert_close(
            model.start_velocities[0],
            torch.full((1, 2, 3), 0.25),
        )
        torch.testing.assert_close(
            model.start_velocities[1][..., 0],
            torch.ones(1, 2),
        )
        torch.testing.assert_close(
            model.start_velocities[2][..., 0],
            torch.ones(1, 2),
        )

    def test_rollout_gradient_is_isolated_to_selected_material_row(self):
        model = RecordingRolloutModel()
        gate_rollout_loss(
            model,
            rollout_sample(),
            material_id=1,
            long_weight=0.5,
            reg_weight=1.0e-3,
            backward=True,
            noise_seed=0,
        )

        self.assertEqual(torch.count_nonzero(model.gate_logits.grad[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.gate_logits.grad[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(model.gate_logits.grad[2]).item(), 0)


class MaterialBestRowTrackerTest(unittest.TestCase):
    def test_tracks_each_material_independently_and_stops_after_three_misses(self):
        tracker = MaterialBestRowTracker(
            num_materials=3,
            num_stages=4,
            validation_interval=25,
            patience=3,
        )
        self.assertFalse(tracker.should_validate(24))
        self.assertTrue(tracker.should_validate(25))

        tracker.record_identity(0, 6.0, torch.zeros(4))
        self.assertEqual(tracker.best_update[0], 0)
        self.assertFalse(tracker.observe(0, 25, 5.0, torch.full((4,), 0.25)))
        self.assertFalse(tracker.observe(0, 50, 4.0, torch.full((4,), 0.50)))
        self.assertFalse(tracker.observe(0, 75, 4.1, torch.full((4,), 0.75)))
        self.assertFalse(tracker.observe(0, 100, 4.2, torch.full((4,), 1.00)))
        self.assertTrue(tracker.observe(0, 125, 4.3, torch.full((4,), 1.25)))
        self.assertEqual(tracker.best_update[0], 50)
        torch.testing.assert_close(
            tracker.best_rows[0],
            torch.full((4,), 0.50),
        )

        tracker.record_identity(1, 3.0, torch.zeros(4))
        self.assertFalse(tracker.observe(1, 25, 2.0, torch.full((4,), -0.25)))
        self.assertEqual(tracker.best_update[1], 25)
        torch.testing.assert_close(
            tracker.best_rows[1],
            torch.full((4,), -0.25),
        )

    def test_rejects_observations_outside_validation_schedule(self):
        tracker = MaterialBestRowTracker(3, 4, validation_interval=25, patience=3)
        tracker.record_identity(0, 2.0, torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "validation interval"):
            tracker.observe(0, 24, 1.0, torch.zeros(4))

    def test_requires_identity_baseline_before_post_update_observation(self):
        tracker = MaterialBestRowTracker(3, 4, validation_interval=25, patience=3)
        with self.assertRaisesRegex(RuntimeError, "identity"):
            tracker.observe(0, 25, 1.0, torch.zeros(4))


class GateArtifactTest(unittest.TestCase):
    def test_saves_full_checkpoint_and_required_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            model = TinyGateModel()
            with torch.no_grad():
                model.gate_logits.copy_(torch.arange(12.0).reshape(3, 4) / 10.0)
            manifest = {
                "seed": 0,
                "train_fraction": 0.8,
                "materials": {
                    "elastic": {"train": ["a.h5"], "val": ["b.h5"]}
                },
            }
            history = [
                {
                    "material": "elastic",
                    "update": 25,
                    "split": "val",
                    "full_mse": 0.1,
                    "long_mse": 0.2,
                    "score": 0.2,
                }
            ]

            save_gate_artifacts(
                model,
                root,
                split_manifest=manifest,
                training_history=history,
                metadata={
                    "base_checkpoint": "base/model.safetensors",
                    "identity_scores": [0.3, 0.2, 0.1],
                    "best_updates": [25, 0, 50],
                    "best_scores": [0.2, 0.2, 0.05],
                    "seed": 0,
                },
            )

            expected = [
                "split_manifest.json",
                "training_history.csv",
                "best_gates.json",
                os.path.join("checkpoint-best", "model.safetensors"),
                os.path.join("checkpoint-best", "gate_metadata.json"),
            ]
            for relative_path in expected:
                self.assertTrue(os.path.isfile(os.path.join(root, relative_path)))
            restored = load_file(
                os.path.join(root, "checkpoint-best", "model.safetensors")
            )
            self.assertEqual(set(restored), set(model.state_dict()))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(restored[key], value)
            del restored
            with open(os.path.join(root, "best_gates.json"), encoding="utf-8") as handle:
                gate_audit = json.load(handle)
            self.assertEqual(gate_audit["identity_scores"], [0.3, 0.2, 0.1])
            self.assertEqual(gate_audit["best_updates"], [25, 0, 50])
            self.assertEqual(gate_audit["seed"], 0)


class CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Parameter(torch.tensor(1.0))
        self.gate_logits = nn.Parameter(torch.zeros(3, 4))

    def material_stage_gates(self):
        return 2.0 * torch.sigmoid(self.gate_logits)

    def forward(
        self,
        x,
        timesteps,
        init_pc,
        force,
        E,
        nu,
        drag_mask,
        drag_point,
        floor_height,
        gravity_label=None,
        coeff=None,
        y=None,
        null_emb=None,
        start_vel=None,
        points_rest=None,
    ):
        gate = self.material_stage_gates()[y.long()].mean(dim=1)
        return gate.view(-1, 1, 1, 1).expand(-1, 1, init_pc.shape[2], 3)


def calibration_sample(material_id, target):
    sample = rollout_sample()
    sample["mat_type"] = torch.tensor([material_id], dtype=torch.long)
    sample["future_gt"] = torch.full_like(sample["future_gt"][:, :2], target)
    return sample


class CalibrationOrchestrationTest(unittest.TestCase):
    def test_reports_material_update_and_validation_progress(self):
        class RecordingProgress:
            def __init__(self, **kwargs):
                self.total = kwargs["total"]
                self.description = kwargs["desc"]
                self.updates = 0
                self.postfixes = []
                self.closed = False

            def update(self, amount=1):
                self.updates += amount

            def set_description_str(self, description):
                self.description = description

            def set_postfix(self, values, refresh=True):
                self.postfixes.append(dict(values))

            def close(self):
                self.closed = True

        bars = []

        def progress_factory(**kwargs):
            bar = RecordingProgress(**kwargs)
            bars.append(bar)
            return bar

        model = CalibrationModel()
        train_loaders = {
            0: [calibration_sample(0, 0.2)],
            1: [calibration_sample(1, 1.0)],
            2: [calibration_sample(2, 1.8)],
        }
        result = calibrate_material_rows(
            model,
            train_loaders,
            {material_id: list(samples) for material_id, samples in train_loaders.items()},
            max_updates=2,
            validation_interval=1,
            patience=2,
            learning_rate=0.1,
            long_weight=0.5,
            reg_weight=1.0e-3,
            noise_seed=0,
            progress_factory=progress_factory,
        )

        self.assertEqual(result["best_updates"], [2, 0, 2])
        self.assertEqual(len(bars), 3)
        self.assertEqual([bar.total for bar in bars], [2, 2, 2])
        self.assertEqual([bar.updates for bar in bars], [2, 2, 2])
        self.assertEqual(
            [bar.description for bar in bars],
            ["elastic gate", "plasticine gate", "sand gate"],
        )
        self.assertTrue(all(bar.closed for bar in bars))
        self.assertTrue(all("train" in bar.postfixes[-1] for bar in bars))
        self.assertTrue(all("val" in bar.postfixes[-1] for bar in bars))
        self.assertTrue(all("best" in bar.postfixes[-1] for bar in bars))

    def test_calibrates_rows_independently_and_restores_each_best_row(self):
        model = CalibrationModel()
        train_loaders = {
            0: [calibration_sample(0, 0.2)],
            1: [calibration_sample(1, 1.0)],
            2: [calibration_sample(2, 1.8)],
        }
        val_loaders = {
            material_id: list(samples)
            for material_id, samples in train_loaders.items()
        }

        result = calibrate_material_rows(
            model,
            train_loaders,
            val_loaders,
            max_updates=2,
            validation_interval=1,
            patience=2,
            learning_rate=0.1,
            long_weight=0.5,
            reg_weight=1.0e-3,
            noise_seed=0,
        )

        self.assertLess(model.gate_logits[0].mean().item(), 0.0)
        self.assertAlmostEqual(model.gate_logits[1].mean().item(), 0.0, places=6)
        self.assertGreater(model.gate_logits[2].mean().item(), 0.0)
        self.assertEqual(result["best_updates"], [2, 0, 2])
        self.assertEqual(len(result["identity_scores"]), 3)
        self.assertEqual(len(result["history"]), 15)
        self.assertFalse(model.backbone.requires_grad)
        self.assertTrue(model.gate_logits.requires_grad)


class GateTrainerCliTest(unittest.TestCase):
    @staticmethod
    def small_config():
        return OmegaConf.create(
            {
                "pc_size": 2,
                "input_frames": 5,
                "output_frames": 1,
                "use_diffusion": False,
                "train_dataset": {"input_frames": 1, "output_frames": 5},
                "model_config": {
                    "n_layers": 4,
                    "latent_dim": 64,
                    "frame_cond": True,
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
                    "material_stage_gate": True,
                    "material_stage_gate_max": 2.0,
                },
            }
        )

    def test_create_model_synchronizes_single_frame_dataset_and_history(self):
        config = self.small_config()
        model = create_gate_model(config)

        self.assertEqual(config.train_dataset.input_frames, 5)
        self.assertEqual(config.train_dataset.output_frames, 1)
        self.assertEqual(config.model_config.cond_frames, 5)
        self.assertEqual(tuple(model.dit.material_state_exchange.gate_logits.shape), (3, 4))

    def test_create_model_rejects_diffusion_or_disabled_gate(self):
        diffusion = self.small_config()
        diffusion.use_diffusion = True
        with self.assertRaisesRegex(ValueError, "deterministic"):
            create_gate_model(diffusion)

        disabled = self.small_config()
        disabled.model_config.material_stage_gate = False
        with self.assertRaisesRegex(ValueError, "material_stage_gate"):
            create_gate_model(disabled)


class RealModelGateCalibrationSmokeTest(unittest.TestCase):
    def test_load_rollout_update_save_and_reload_changes_only_gate(self):
        with tempfile.TemporaryDirectory() as root:
            config = GateTrainerCliTest.small_config()
            b3a_config = OmegaConf.create(
                OmegaConf.to_container(config.model_config, resolve=True)
            )
            b3a_config.cond_frames = 5
            b3a_config.material_stage_gate = False
            b3a = MDM_ST(2, 1, 3, b3a_config).eval()
            with torch.no_grad():
                b3a.dit.material_state_exchange.output_proj.weight.fill_(0.01)
                b3a.dit.material_state_exchange.output_proj.bias.fill_(0.001)
            base_checkpoint = os.path.join(root, "b3a.safetensors")
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in b3a.state_dict().items()
                },
                base_checkpoint,
            )

            b3b = create_gate_model(config).eval()
            load_b3a_into_b3b(b3b, base_checkpoint)
            gate = freeze_for_gate_training(b3b)
            frozen_before = {
                key: value.detach().clone()
                for key, value in b3b.state_dict().items()
                if not key.endswith("gate_logits")
            }
            sample = rollout_sample()
            sample["mat_type"] = torch.tensor([1], dtype=torch.long)
            sample["future_gt"] = torch.randn(1, 2, 2, 3)
            optimizer = torch.optim.Adam([gate], lr=1.0e-2)
            optimizer.zero_grad(set_to_none=True)
            gate_rollout_loss(
                b3b,
                sample,
                material_id=1,
                backward=True,
                noise_seed=0,
            )
            optimizer.step()

            self.assertGreater(torch.count_nonzero(gate).item(), 0)
            for key, expected in frozen_before.items():
                torch.testing.assert_close(
                    b3b.state_dict()[key],
                    expected,
                    rtol=0.0,
                    atol=0.0,
                )

            output_dir = os.path.join(root, "artifacts")
            save_gate_artifacts(
                b3b,
                output_dir,
                split_manifest={"seed": 0, "train_fraction": 0.8, "materials": {}},
                training_history=[],
                metadata={"base_checkpoint": base_checkpoint},
            )
            restored_state = load_file(
                os.path.join(output_dir, "checkpoint-best", "model.safetensors")
            )
            restored = create_gate_model(GateTrainerCliTest.small_config()).eval()
            incompatible = restored.load_state_dict(restored_state, strict=True)
            self.assertEqual(incompatible.missing_keys, [])
            self.assertEqual(incompatible.unexpected_keys, [])
            torch.testing.assert_close(
                restored.dit.material_state_exchange.gate_logits,
                b3b.dit.material_state_exchange.gate_logits,
            )
            del restored_state


if __name__ == "__main__":
    unittest.main()
