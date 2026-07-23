import time
import unittest

import torch
from omegaconf import OmegaConf

from model.hybrid_state import HybridStateExchange, compute_explicit_frame_state
from model.spacetime import MDM_ST, SpatialTemporalTransformerBlock


class HybridFrameStateTests(unittest.TestCase):
    def setUp(self):
        base = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-1.0, -2.0, -3.0],
                [2.0, 1.0, -2.0],
                [-2.0, -1.0, 2.0],
            ]
        )
        frame_scales = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.5, 0.5, 1.5],
                [1.0, 1.25, 0.5],
            ]
        )
        self.offsets = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
            ]
        )
        deformed = base[None] * frame_scales[:, None]
        self.points = deformed[None] + self.offsets[None, :, None]

    def test_returns_one_18_value_state_per_frame(self):
        state = compute_explicit_frame_state(self.points)

        self.assertEqual(state.shape, (1, 3, 18))

    def test_relative_center_occupies_first_three_values(self):
        state = compute_explicit_frame_state(self.points)

        torch.testing.assert_close(state[0, :, :3], self.offsets)

    def test_adjacent_center_velocity_occupies_values_three_to_six(self):
        expected_velocity = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [2.0, 0.0, -2.0],
            ]
        )

        state = compute_explicit_frame_state(self.points)

        torch.testing.assert_close(state[0, :, 3:6], expected_velocity)

    def test_covariance_and_delta_use_upper_triangle_layout(self):
        expected_frames = []
        for frame in self.points[0]:
            centered = frame - frame.mean(dim=0)
            x, y, z = centered.unbind(dim=-1)
            expected_frames.append(
                torch.stack(
                    (
                        (x * x).mean(),
                        (x * y).mean(),
                        (x * z).mean(),
                        (y * y).mean(),
                        (y * z).mean(),
                        (z * z).mean(),
                    )
                )
            )
        expected_covariance = torch.stack(expected_frames).unsqueeze(0)
        expected_delta = torch.cat(
            (
                torch.zeros_like(expected_covariance[:, :1]),
                expected_covariance[:, 1:] - expected_covariance[:, :-1],
            ),
            dim=1,
        )

        self.assertTrue(torch.all(expected_covariance[..., [1, 2, 4]] != 0))
        self.assertTrue(torch.all(expected_delta[:, 1:] != 0))

        state = compute_explicit_frame_state(self.points)

        self.assertTrue(torch.all(state[0, 1:, 12:18] != 0))
        torch.testing.assert_close(
            state[0, :, 6:12],
            expected_covariance[0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state[0, :, 12:18],
            expected_delta[0],
            rtol=0,
            atol=0,
        )

    def test_requires_rank_four_input(self):
        for invalid in (torch.zeros(3, 4, 3), torch.zeros(1, 2, 3, 4, 3)):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError):
                    compute_explicit_frame_state(invalid)

    def test_requires_xyz_last_dimension(self):
        with self.assertRaises(ValueError):
            compute_explicit_frame_state(torch.zeros(1, 2, 4, 2))

    def test_requires_at_least_one_frame_and_particle(self):
        for invalid in (torch.zeros(1, 0, 4, 3), torch.zeros(1, 2, 0, 3)):
            with self.subTest(shape=tuple(invalid.shape)):
                with self.assertRaises(ValueError):
                    compute_explicit_frame_state(invalid)

    def test_preserves_autograd(self):
        points = self.points.clone().requires_grad_(True)

        compute_explicit_frame_state(points).square().sum().backward()

        self.assertIsNotNone(points.grad)
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(torch.count_nonzero(points.grad).item(), 0)


class HybridStateExchangeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 2
        self.frames = 7
        self.particles = 4
        self.particle_dim = 8
        self.state_dim = 4
        self.history_start = 1
        self.prediction_index = 6
        self.exchange = HybridStateExchange(
            particle_dim=self.particle_dim,
            state_dim=self.state_dim,
            num_heads=2,
            history_frames=5,
            num_stages=3,
        )
        self.hidden = torch.randn(
            self.batch_size,
            self.frames,
            self.particles,
            self.particle_dim,
        )
        self.explicit = torch.randn(self.batch_size, 5, 18)
        self.material = torch.randn(self.batch_size, 2)

    def _forward(self, **overrides):
        arguments = {
            "hidden_states": self.hidden,
            "state_tokens": None,
            "explicit_frame_state": self.explicit,
            "material_values": self.material,
            "history_start": self.history_start,
            "prediction_index": self.prediction_index,
            "stage_index": 0,
        }
        arguments.update(overrides)
        return self.exchange(**arguments)

    def test_constructor_creates_zero_initialized_stage_gates(self):
        self.assertEqual(self.exchange.feedback_gates.shape, (3,))
        self.assertTrue(torch.equal(self.exchange.feedback_gates, torch.zeros(3)))
        self.assertEqual(self.exchange.frame_embeddings.shape, (5, self.state_dim))
        self.assertEqual(self.exchange.stage_embeddings.shape, (3, self.state_dim))

    def test_v11a_configuration_stays_within_parameter_budget(self):
        exchange = HybridStateExchange(
            particle_dim=256,
            state_dim=64,
            num_heads=4,
            history_frames=5,
            num_stages=4,
        )

        parameter_count = sum(
            parameter.numel()
            for parameter in exchange.parameters()
            if parameter.requires_grad
        )

        self.assertLessEqual(parameter_count, 161_000)

    def test_constructor_rejects_invalid_dimensions(self):
        invalid_arguments = (
            ({"particle_dim": 0}, "particle_dim"),
            ({"particle_dim": -1}, "particle_dim"),
            ({"particle_dim": 8, "state_dim": 0}, "state_dim"),
            ({"particle_dim": 8, "state_dim": -1}, "state_dim"),
            ({"particle_dim": 8, "num_heads": 0}, "num_heads"),
            ({"particle_dim": 8, "num_heads": -1}, "num_heads"),
            ({"particle_dim": 8, "state_dim": 5, "num_heads": 2}, "state_dim"),
            ({"particle_dim": 8, "history_frames": 4}, "history_frames"),
            ({"particle_dim": 8, "num_stages": 0}, "num_stages"),
        )

        for arguments, message in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    HybridStateExchange(**arguments)

    def test_zero_gate_preserves_hidden_states_exactly(self):
        state_tokens, updated_hidden = self._forward()

        self.assertEqual(state_tokens.shape, (self.batch_size, 5, self.state_dim))
        self.assertTrue(torch.equal(updated_hidden, self.hidden))

    def test_zero_gate_preserves_output_and_receives_startup_gradient(self):
        _, updated_hidden = self._forward(stage_index=0)

        self.assertTrue(torch.equal(updated_hidden, self.hidden))

        updated_hidden.square().sum().backward()
        active_gate_grad = self.exchange.feedback_gates.grad[0]

        self.assertTrue(torch.isfinite(active_gate_grad))
        self.assertNotEqual(active_gate_grad.item(), 0.0)
        self.assertEqual(self.exchange.feedback_gates[0].item(), 0.0)

    def test_zero_gate_sanitizes_non_finite_feedback_and_keeps_finite_gradient(self):
        hidden = self.hidden.clone().requires_grad_(True)
        explicit = torch.full_like(self.explicit, torch.finfo(self.explicit.dtype).max)
        with torch.no_grad():
            explicit_layers = (
                self.exchange.explicit_encoder[0],
                self.exchange.explicit_encoder[2],
            )
            for layer in explicit_layers:
                layer.weight.fill_(1)
                layer.bias.zero_()
        raw_feedback = []

        def capture_feedback(module, args, output):
            raw_feedback.append(output.detach())

        handle = self.exchange.feedback_attention.register_forward_hook(capture_feedback)
        try:
            _, updated_hidden = self._forward(
                hidden_states=hidden,
                explicit_frame_state=explicit,
                stage_index=0,
            )
            updated_hidden.sum().backward()
        finally:
            handle.remove()

        self.assertTrue(torch.isfinite(explicit).all())
        self.assertEqual(len(raw_feedback), 1)
        self.assertFalse(torch.isfinite(raw_feedback[0]).all())
        self.assertTrue(torch.isfinite(updated_hidden).all())
        self.assertTrue(torch.equal(updated_hidden, hidden))
        self.assertIsNotNone(self.exchange.feedback_gates.grad)
        active_gate_grad = self.exchange.feedback_gates.grad[0]
        self.assertTrue(torch.isfinite(active_gate_grad))

    def test_mask_and_prediction_frames_do_not_affect_state_tokens(self):
        baseline_state, _ = self._forward()
        changed_hidden = self.hidden.clone()
        changed_hidden[:, 0] = torch.randn_like(changed_hidden[:, 0]) * 1000
        changed_hidden[:, self.prediction_index] = (
            torch.randn_like(changed_hidden[:, self.prediction_index]) * 1000
        )

        changed_state, _ = self._forward(hidden_states=changed_hidden)

        torch.testing.assert_close(changed_state, baseline_state, rtol=0, atol=0)

    def test_open_gate_changes_only_prediction_frame(self):
        with torch.no_grad():
            self.exchange.feedback_gates[0] = 1

        _, updated_hidden = self._forward()

        self.assertTrue(
            torch.equal(
                updated_hidden[:, : self.prediction_index],
                self.hidden[:, : self.prediction_index],
            )
        )
        self.assertFalse(
            torch.equal(
                updated_hidden[:, self.prediction_index],
                self.hidden[:, self.prediction_index],
            )
        )

    def test_state_is_retained_and_refined_across_stages(self):
        stage_one, _ = self._forward(stage_index=0)
        stage_two, _ = self._forward(state_tokens=stage_one, stage_index=1)
        stage_two_without_retained_state, _ = self._forward(
            state_tokens=torch.zeros_like(stage_one),
            stage_index=1,
        )

        self.assertEqual(stage_one.shape, (self.batch_size, 5, self.state_dim))
        self.assertEqual(stage_two.shape, stage_one.shape)
        self.assertFalse(torch.equal(stage_two, stage_one))
        self.assertFalse(torch.equal(stage_two, stage_two_without_retained_state))

    def test_material_values_condition_state_and_feedback(self):
        zero_material = torch.zeros_like(self.material)
        other_material = torch.full_like(self.material, 2.0)
        zero_state, _ = self._forward(material_values=zero_material)
        other_state, _ = self._forward(material_values=other_material)

        self.assertFalse(torch.equal(zero_state, other_state))

        with torch.no_grad():
            self.exchange.feedback_gates[0] = 1
        _, zero_hidden = self._forward(material_values=zero_material)
        _, other_hidden = self._forward(material_values=other_material)
        self.assertFalse(
            torch.equal(
                zero_hidden[:, self.prediction_index],
                other_hidden[:, self.prediction_index],
            )
        )

    def test_rejects_invalid_forward_shapes(self):
        invalid_cases = (
            (
                {"hidden_states": self.hidden[:, 0]},
                r"hidden_states.*\(B, F, N, C\)",
            ),
            ({"hidden_states": self.hidden[..., :-1]}, "particle_dim"),
            ({"explicit_frame_state": self.explicit[:, :4]}, "explicit_frame_state"),
            ({"material_values": self.material[:, :1]}, "material_values"),
            (
                {"state_tokens": torch.zeros(self.batch_size, 4, self.state_dim)},
                "state_tokens",
            ),
        )

        for overrides, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._forward(**overrides)

    def test_rejects_non_floating_hidden_states(self):
        integer_hidden = torch.ones_like(self.hidden, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "floating"):
            self._forward(hidden_states=integer_hidden)

    def test_rejects_invalid_history_prediction_and_stage_layout(self):
        invalid_cases = (
            ({"history_start": -1}, "history_start"),
            ({"prediction_index": 5}, "immediately follow"),
            (
                {
                    "hidden_states": torch.cat(
                        (self.hidden, torch.zeros_like(self.hidden[:, :1])), dim=1
                    )
                },
                "exactly one prediction frame",
            ),
            ({"stage_index": -1}, "stage_index"),
            ({"stage_index": 3}, "stage_index"),
        )

        for overrides, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self._forward(**overrides)

    def test_mixed_dtype_forward_and_autograd(self):
        hidden = self.hidden.double().requires_grad_(True)
        explicit = self.explicit.double().requires_grad_(True)
        material = self.material.double().requires_grad_(True)
        with torch.no_grad():
            self.exchange.feedback_gates[0] = 1

        state_tokens, updated_hidden = self._forward(
            hidden_states=hidden,
            explicit_frame_state=explicit,
            material_values=material,
        )
        (state_tokens.square().mean() + updated_hidden.square().mean()).backward()

        self.assertEqual(state_tokens.dtype, self.exchange.frame_embeddings.dtype)
        self.assertEqual(updated_hidden.dtype, hidden.dtype)
        for value in (hidden, explicit, material):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        parameter_grads = [
            parameter.grad
            for parameter in self.exchange.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(parameter_grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in parameter_grads))


class V11aIntegrationTests(unittest.TestCase):
    @staticmethod
    def _model_config(transformer_block, n_layers=2, latent_dim=64):
        return OmegaConf.create(
            {
                "n_layers": n_layers,
                "latent_dim": latent_dim,
                "frame_cond": True,
                "cond_frames": 5,
                "point_embed": False,
                "mask_cond": True,
                "pred_offset": False,
                "num_neighbors": -1,
                "floor_cond": False,
                "max_num_forces": 1,
                "force_as_token": False,
                "force_as_latent": False,
                "gravity_emb": False,
                "coeff_cond": False,
                "num_mat": 0,
                "class_token": False,
                "class_dropout_prob": 0.0,
                "transformer_block": transformer_block,
            }
        )

    @staticmethod
    def _inputs(batch_size=1, point_count=2):
        return (
            torch.randn(batch_size, 1, point_count, 3),
            torch.zeros(batch_size, dtype=torch.long),
            torch.randn(batch_size, 5, point_count, 3),
            torch.randn(batch_size, 3),
            torch.tensor([[2.0]]).repeat(batch_size, 1),
            torch.tensor([[0.3]]).repeat(batch_size, 1),
            torch.zeros(batch_size, 1, point_count, 1),
            torch.zeros(batch_size, 4),
            None,
        )

    def _model(
        self,
        transformer_block,
        n_layers=2,
        output_frames=1,
        point_count=2,
        latent_dim=64,
    ):
        return MDM_ST(
            n_points=point_count,
            n_frame=output_frames,
            n_feats=3,
            model_config=self._model_config(
                transformer_block,
                n_layers,
                latent_dim,
            ),
        )

    def test_eight_layer_v11a_uses_one_exchange_four_times(self):
        model = self._model("SpatialTemporalTransformerBlockv11a", n_layers=8).eval()

        self.assertEqual(len(model.dit.transformer_blocks), 8)
        self.assertTrue(
            all(
                type(block) is SpatialTemporalTransformerBlock
                for block in model.dit.transformer_blocks
            )
        )
        exchanges = [
            module
            for module in model.modules()
            if isinstance(module, HybridStateExchange)
        ]
        self.assertEqual(exchanges, [model.dit.hybrid_state_exchange])
        self.assertEqual(model.dit.hybrid_state_exchange.num_stages, 4)

        calls = []

        def record_call(module, args, kwargs):
            calls.append(
                (
                    id(module),
                    kwargs["stage_index"],
                    kwargs["history_start"],
                    kwargs["prediction_index"],
                )
            )

        handle = model.dit.hybrid_state_exchange.register_forward_pre_hook(
            record_call,
            with_kwargs=True,
        )
        try:
            with torch.no_grad():
                output = model(*self._inputs())
        finally:
            handle.remove()

        self.assertEqual(output.shape, (1, 1, 2, 3))
        self.assertEqual(
            calls,
            [(id(exchanges[0]), stage, 1, 6) for stage in range(4)],
        )

    def test_baseline_state_dict_loads_and_zero_gates_preserve_bits(self):
        torch.manual_seed(10)
        baseline = self._model("SpatialTemporalTransformerBlock").eval()
        torch.manual_seed(20)
        v11a = self._model("SpatialTemporalTransformerBlockv11a").eval()

        incompatible = v11a.load_state_dict(baseline.state_dict(), strict=False)
        expected_missing = {
            key
            for key in v11a.state_dict()
            if key.startswith("dit.hybrid_state_exchange.")
        }
        self.assertEqual(set(incompatible.missing_keys), expected_missing)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertTrue(
            torch.equal(
                v11a.dit.hybrid_state_exchange.feedback_gates,
                torch.zeros_like(v11a.dit.hybrid_state_exchange.feedback_gates),
            )
        )

        inputs = self._inputs()
        with torch.no_grad():
            baseline_output = baseline(*inputs)
            v11a_output = v11a(*inputs)

        self.assertTrue(torch.equal(v11a_output, baseline_output))

    def test_feedback_gates_receive_finite_gradients_with_and_without_checkpointing(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                torch.manual_seed(30 + int(checkpointing))
                model = self._model(
                    "SpatialTemporalTransformerBlockv11a",
                    n_layers=8,
                ).train()
                if checkpointing:
                    model.enable_gradient_checkpointing()
                inputs = list(self._inputs())
                inputs[0].requires_grad_(True)

                model(*inputs).square().mean().backward()

                gate_grad = model.dit.hybrid_state_exchange.feedback_gates.grad
                self.assertIsNotNone(gate_grad)
                self.assertEqual(gate_grad.shape, (4,))
                self.assertTrue(torch.isfinite(gate_grad).all())
                self.assertTrue(torch.all(gate_grad != 0))

    def test_nonzero_gates_train_pooling_state_and_cross_attention_across_stages(self):
        torch.manual_seed(40)
        model = self._model(
            "SpatialTemporalTransformerBlockv11a",
            n_layers=8,
        ).train()
        exchange = model.dit.hybrid_state_exchange
        with torch.no_grad():
            exchange.feedback_gates.fill_(1e-3)
        inputs = list(self._inputs())
        inputs[0].requires_grad_(True)

        model(*inputs).square().mean().backward()

        representative_parameters = {
            "pooling": exchange.pool_score.weight,
            "state_attention": exchange.state_attention.to_q.weight,
            "cross_attention": exchange.feedback_attention.to_q.weight,
        }
        for name, parameter in representative_parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)

    def test_production_shape_v11a_forward_smoke(self):
        torch.manual_seed(50)
        point_count = 2048
        model = self._model(
            "SpatialTemporalTransformerBlockv11a",
            n_layers=8,
            point_count=point_count,
            latent_dim=256,
        ).eval()
        inputs = self._inputs(point_count=point_count)

        started_at = time.perf_counter()
        with torch.no_grad():
            output = model(*inputs)
        elapsed = time.perf_counter() - started_at

        print(f"v11a production-shape forward: {elapsed:.3f}s")
        self.assertEqual(output.shape, (1, 1, point_count, 3))

    def test_explicit_state_depends_only_on_physical_init_frames(self):
        model = self._model("SpatialTemporalTransformerBlockv11a").eval()
        inputs = list(self._inputs())
        captured = []

        def capture_explicit(module, args, kwargs):
            captured.append(kwargs["explicit_frame_state"].detach().clone())

        handle = model.dit.hybrid_state_exchange.register_forward_pre_hook(
            capture_explicit,
            with_kwargs=True,
        )
        try:
            with torch.no_grad():
                model(*inputs)
                inputs[0] = torch.randn_like(inputs[0]) * 1000
                model(*inputs)
        finally:
            handle.remove()

        expected = compute_explicit_frame_state(inputs[2])
        self.assertEqual(len(captured), 2)
        torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
        torch.testing.assert_close(captured[1], expected, rtol=0, atol=0)

    def test_v11a_rejects_more_than_one_output_frame(self):
        with self.assertRaisesRegex(ValueError, "exactly one output frame"):
            self._model(
                "SpatialTemporalTransformerBlockv11a",
                output_frames=2,
            )

    def test_baseline_selector_still_runs_basic_forward(self):
        model = self._model(
            "SpatialTemporalTransformerBlock",
            n_layers=1,
        ).eval()

        with torch.no_grad():
            output = model(*self._inputs())

        self.assertEqual(output.shape, (1, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
