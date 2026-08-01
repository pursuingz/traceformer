import csv
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.material_condition_diagnostics import (
    MaterialRecord,
    build_parameter_derangement,
    condition_response_metrics,
    dependency_label,
    paired_bootstrap,
    rotate_material_type,
    summarize_rows,
    trajectory_metrics,
)


EXPECTED_B0_MODEL_NAMES = (
    "39478_000.h5",
    "45767_002.h5",
    "39999_000.h5",
    "43624_001.h5",
    "47781_002.h5",
    "42107_001.h5",
    "45583_002.h5",
    "39973_000.h5",
    "39511_000.h5",
    "43739_001.h5",
    "39726_000.h5",
    "39659_000.h5",
    "39959_000.h5",
    "40481_001.h5",
    "41341_001.h5",
    "48721_002.h5",
    "44431_001.h5",
    "47458_002.h5",
    "49860_002.h5",
    "47570_002.h5",
    "42798_001.h5",
    "39746_000.h5",
    "46809_002.h5",
    "39586_000.h5",
    "49196_002.h5",
    "39764_000.h5",
    "39653_000.h5",
    "47956_002.h5",
    "39714_000.h5",
    "45752_002.h5",
    "43014_001.h5",
    "43941_001.h5",
    "45790_002.h5",
    "43393_001.h5",
    "44732_001.h5",
    "46081_002.h5",
    "46213_002.h5",
    "42473_001.h5",
    "39560_000.h5",
    "44774_001.h5",
    "44503_001.h5",
)


class NamespaceConfig(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


class MaterialConditionDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _trajectory():
        gt = np.zeros((25, 2, 3), dtype=np.float64)
        pred = np.zeros_like(gt)
        for frame in range(5, 25):
            pred[frame, :, :] = frame - 4
        return pred, gt

    @staticmethod
    def _b0_args():
        return SimpleNamespace(
            resume=(
                r"D:\runs\outputs\mm3_contact_cond_8L\checkpoint-90000\model.safetensors"
            ),
            input_frames=5,
            output_frames=1,
            use_diffusion=False,
            num_inference_steps=1,
            eval_batch_size=1,
            dataloader_num_workers=0,
            floor_projection=False,
            pred_offset=True,
            seed=0,
            pc_size=2048,
            model_type="dit_st",
            vis_dir="vis_results_mm3_contact_cond",
            per_model_csv=True,
            train_dataset=NamespaceConfig(
                category="hf-objaverse-v1",
                dataset_path=r"D:\data\mm3_data\mm3_test",
                dataset_list="DATASET_ITEM_LIST",
                has_gravity=True,
                max_num_forces=1,
                norm_fac=5,
                stage="deform",
                mode="diff",
                pc_size=2048,
                repeat=1,
                seed=0,
                n_sample_pro_model=300,
                n_frames_interval=1,
                n_training_frames=24,
                batch_size=20,
                overfit=False,
                input_frames=5,
                output_frames=1,
            ),
            model_config=NamespaceConfig(
                n_layers=8,
                latent_dim=256,
                frame_cond=True,
                point_embed=True,
                mask_cond=True,
                pred_offset=True,
                num_neighbors=-1,
                floor_cond=True,
                max_num_forces=1,
                force_as_token=False,
                force_as_latent=False,
                gravity_emb=True,
                coeff_cond=False,
                num_mat=4,
                class_token=True,
                transformer_block="SpatialTemporalTransformerBlock",
                contact_particle_cond=True,
                contact_feature_sigma=0.04,
            ),
        )

    def _factorized_b0_args(self):
        args = self._b0_args()
        args.resume = (
            "outputs/mm3_contact_vxyz_factorized_8L/"
            "checkpoint-90000/model.safetensors"
        )
        args.model_config.contact_velocity_mode = "xyz"
        args.model_config.contact_injection_mode = "factorized"
        args.model_config.contact_feature_mask = [1, 1, 1, 1, 1]
        return args

    @staticmethod
    def _b0_records():
        return [
            MaterialRecord(
                model_name,
                int(Path(model_name).stem.rsplit("_", 1)[1]),
                3.0,
                0.2,
            )
            for model_name in EXPECTED_B0_MODEL_NAMES
        ]

    @staticmethod
    def _summary_rows(model_names=("elastic.h5", "plasticine.h5", "sand.h5")):
        rows = []
        for mat_type, model_name in enumerate(model_names):
            row = {"model": model_name, "mat_type": mat_type}
            for prefix, value in (
                ("normal", 2.0),
                ("shuffle_params", 3.0),
                ("shuffle_class", 4.0),
            ):
                for metric in ("full_rollout_mse", "gm_mse", "long_seg_mse", "fde"):
                    row[f"{prefix}_{metric}"] = value
            row["shuffle_params_prediction_mse"] = 1.0
            row["shuffle_params_final_prediction_mse"] = 1.25
            row["shuffle_class_prediction_mse"] = 1.5
            row["shuffle_class_final_prediction_mse"] = 1.75
            row["shuffle_e_prediction_mse"] = 0.5
            row["shuffle_nu_prediction_mse"] = 0.75
            for metric in ("full_rollout_mse", "gm_mse", "long_seg_mse", "fde"):
                row[f"shuffle_e_{metric}"] = row[f"normal_{metric}"] * 1.10
                row[f"shuffle_nu_{metric}"] = row[f"normal_{metric}"] * 1.20
            row.update(
                {
                    "true_log10_e": 3.0 + mat_type,
                    "true_nu": 0.2 + mat_type * 0.05,
                    "shuffled_log10_e": 3.5 + mat_type,
                    "shuffled_nu": 0.25 + mat_type * 0.05,
                    "shuffled_mat_type": (mat_type + 1) % 3,
                }
            )
            rows.append(row)
        return rows

    def test_trajectory_metrics_match_full_rollout_and_prediction_horizon(self):
        pred, gt = self._trajectory()

        metrics = trajectory_metrics(pred, gt, input_frames=5)

        expected_gm = float(np.exp(np.mean(np.log(np.arange(1, 21, dtype=float) ** 2))))
        self.assertAlmostEqual(metrics["full_rollout_mse"], 114.8)
        self.assertAlmostEqual(metrics["gm_mse"], expected_gm)
        self.assertAlmostEqual(metrics["long_seg_mse"], 293.0)
        self.assertAlmostEqual(metrics["fde"], 20.0 * np.sqrt(3.0))

    def test_condition_response_metrics_excludes_identical_condition_frames(self):
        normal = np.zeros((25, 2, 3), dtype=np.float64)
        counterfactual = np.zeros_like(normal)
        counterfactual[:5] = 100.0
        for frame in range(5, 25):
            counterfactual[frame, :, :] = frame - 4

        response = condition_response_metrics(normal, counterfactual, input_frames=5)

        self.assertAlmostEqual(response["prediction_mse"], 143.5)
        self.assertAlmostEqual(response["final_prediction_mse"], 400.0)

    def test_trajectory_metrics_detaches_cpu_tensor_before_numpy_conversion(self):
        pred, gt = self._trajectory()
        pred_tensor = torch.tensor(pred, requires_grad=True)
        gt_tensor = torch.tensor(gt, requires_grad=True)

        metrics = trajectory_metrics(pred_tensor, gt_tensor, input_frames=5)

        self.assertAlmostEqual(metrics["full_rollout_mse"], 114.8)

    def test_trajectory_metrics_supports_torch_bfloat16(self):
        gt = torch.zeros((25, 2, 3), dtype=torch.bfloat16)
        pred = gt.clone()
        pred[5:] = 1.0

        metrics = trajectory_metrics(pred, gt, input_frames=5)

        self.assertAlmostEqual(metrics["full_rollout_mse"], 0.8)

    def test_torch_conversion_path_does_not_pass_original_tensor_to_numpy(self):
        pred, gt = self._trajectory()

        class GuardedTensor:
            def __init__(self, value):
                self.value = value
                self.calls = []

            def detach(self):
                self.calls.append("detach")
                return self

            def cpu(self):
                self.calls.append("cpu")
                return self

            def float(self):
                self.calls.append("float")
                return self

            def numpy(self):
                self.calls.append("numpy")
                return self.value

        guarded_pred = GuardedTensor(pred)
        guarded_gt = GuardedTensor(gt)
        original_asarray = np.asarray

        def assert_numpy_input(value, *args, **kwargs):
            self.assertIsNot(value, guarded_pred)
            self.assertIsNot(value, guarded_gt)
            return original_asarray(value, *args, **kwargs)

        with mock.patch(
            "src.utils.material_condition_diagnostics.torch.is_tensor",
            side_effect=lambda value: value in (guarded_pred, guarded_gt),
        ), mock.patch(
            "src.utils.material_condition_diagnostics.np.asarray",
            side_effect=assert_numpy_input,
        ):
            trajectory_metrics(guarded_pred, guarded_gt, input_frames=5)

        self.assertEqual(guarded_pred.calls, ["detach", "cpu", "float", "numpy"])
        self.assertEqual(guarded_gt.calls, ["detach", "cpu", "float", "numpy"])

    def test_trajectory_metrics_rejects_non_three_dimensional_inputs(self):
        with self.assertRaisesRegex(ValueError, r"shape \(T,N,3\)"):
            trajectory_metrics(np.zeros((1, 2, 3, 1)), np.zeros((1, 2, 3, 1)))

    def test_trajectory_metrics_rejects_invalid_input_frame_count(self):
        pred, gt = self._trajectory()
        for input_frames in (0, 25):
            with self.subTest(input_frames=input_frames):
                with self.assertRaisesRegex(ValueError, "input_frames"):
                    trajectory_metrics(pred, gt, input_frames=input_frames)

    def test_derangement_is_within_material_one_to_one_and_reproducible(self):
        records = [
            MaterialRecord("e0.h5", 0, 4.0, 0.1),
            MaterialRecord("e1.h5", 0, 5.0, 0.2),
            MaterialRecord("e2.h5", 0, 6.0, 0.3),
            MaterialRecord("s0.h5", 2, 4.5, 0.15),
            MaterialRecord("s1.h5", 2, 6.5, 0.35),
        ]
        first = build_parameter_derangement(records, seed=7)
        second = build_parameter_derangement(records, seed=7)
        self.assertEqual(first, second)
        for group in ({"e0.h5", "e1.h5", "e2.h5"}, {"s0.h5", "s1.h5"}):
            source = {(r.log10_e, r.nu) for r in records if r.model in group}
            assigned = {first[name] for name in group}
            self.assertEqual(source, assigned)
            for record in (r for r in records if r.model in group):
                self.assertNotEqual(first[record.model], (record.log10_e, record.nu))

    def test_component_interventions_share_one_same_material_derangement(self):
        from src.utils.material_condition_diagnostics import (
            MaterialRecord,
            build_parameter_derangement,
            build_parameter_donor_mapping,
        )

        records = [
            MaterialRecord(f"elastic_{i}.h5", 0, 4.0 + i, 0.10 + i * 0.01)
            for i in range(4)
        ] + [
            MaterialRecord(f"sand_{i}.h5", 2, 5.0 + i, 0.20 + i * 0.01)
            for i in range(4)
        ]
        donors = build_parameter_donor_mapping(records, seed=7)
        pairs = build_parameter_derangement(records, seed=7)
        by_name = {record.model: record for record in records}

        self.assertEqual(set(donors), set(by_name))
        for model, donor in donors.items():
            self.assertNotEqual(model, donor.model)
            self.assertEqual(by_name[model].mat_type, donor.mat_type)
            self.assertEqual(pairs[model], (donor.log10_e, donor.nu))

    def test_parameter_donor_mapping_is_deterministic_per_seed(self):
        from src.utils.material_condition_diagnostics import (
            MaterialRecord,
            build_parameter_donor_mapping,
        )

        records = [
            MaterialRecord(f"elastic_{i}.h5", 0, 4.0 + i, 0.10 + i * 0.01)
            for i in range(4)
        ] + [
            MaterialRecord(f"sand_{i}.h5", 2, 5.0 + i, 0.20 + i * 0.01)
            for i in range(4)
        ]

        first = build_parameter_donor_mapping(records, seed=7)
        repeated = build_parameter_donor_mapping(records, seed=7)
        different_seed = build_parameter_donor_mapping(records, seed=8)

        self.assertEqual(first, repeated)
        self.assertTrue(
            any(first[model].model != different_seed[model].model for model in first)
        )

    def test_rotates_supported_material_classes(self):
        self.assertEqual([rotate_material_type(i) for i in (0, 1, 2)], [1, 2, 0])

    def test_rejects_unsupported_material_class(self):
        with self.assertRaisesRegex(ValueError, "expected one of"):
            rotate_material_type(3)

    def test_rejects_unsupported_material_class_in_records(self):
        records = [
            MaterialRecord("e0.h5", 0, 4.0, 0.1),
            MaterialRecord("x0.h5", 3, 5.0, 0.2),
        ]
        with self.assertRaisesRegex(ValueError, "expected one of"):
            build_parameter_derangement(records, seed=7)

    def test_rejects_singleton_material_group(self):
        records = [MaterialRecord("e0.h5", 0, 4.0, 0.1)]
        with self.assertRaisesRegex(ValueError, "at least two records"):
            build_parameter_derangement(records, seed=7)

    def test_paired_bootstrap_preserves_pairing_and_is_reproducible(self):
        normal = np.array([1.0, 3.0, 5.0])
        counterfactual = normal + 2.0

        first = paired_bootstrap(normal, counterfactual, samples=100, seed=11)
        second = paired_bootstrap(normal, counterfactual, samples=100, seed=11)

        self.assertEqual(first, second)
        self.assertEqual(first["normal_mean"], 3.0)
        self.assertEqual(first["counterfactual_mean"], 5.0)
        self.assertEqual(first["mean_delta"], 2.0)
        self.assertAlmostEqual(first["relative_change_pct"], 200.0 / 3.0)
        self.assertEqual(first["ci_low"], 2.0)
        self.assertEqual(first["ci_high"], 2.0)

    def test_paired_bootstrap_rejects_invalid_inputs(self):
        for normal, counterfactual, samples in (
            ([], [], 10),
            ([1.0], [1.0, 2.0], 10),
            ([0.0, 0.0], [1.0, 1.0], 10),
            ([1.0], [2.0], 0),
        ):
            with self.subTest(normal=normal, counterfactual=counterfactual, samples=samples):
                with self.assertRaises(ValueError):
                    paired_bootstrap(normal, counterfactual, samples=samples, seed=0)

    def test_dependency_label_applies_effect_ci_and_response_thresholds(self):
        self.assertEqual(dependency_label(5.0, 0.01, 0.2, 10.0), "used")
        self.assertEqual(dependency_label(5.0, 0.01, 0.2, 0.0), "used")
        self.assertEqual(dependency_label(1.9, -0.1, 0.1, 1.9), "ignored")
        self.assertEqual(dependency_label(1.9, 0.1, 0.2, 1.9), "ambiguous")
        self.assertEqual(dependency_label(1.9, -0.1, 0.1, 2.0), "ignored")
        self.assertEqual(dependency_label(4.0, 0.01, 0.2, 5.0), "ambiguous")
        self.assertEqual(dependency_label(-6.0, -0.3, -0.1, 10.0), "ambiguous")

    def test_summarize_rows_groups_metrics_and_uses_group_response_ratio(self):
        rows = []
        for mat_type, normal_mse, counterfactual_mse, prediction_mse in (
            (0, 2.0, 4.0, 1.0),
            (1, 4.0, 8.0, 4.0),
            (2, 8.0, 16.0, 12.0),
        ):
            row = {"model": f"model-{mat_type}", "mat_type": mat_type}
            for metric in ("full_rollout_mse", "gm_mse", "long_seg_mse", "fde"):
                row[f"normal_{metric}"] = normal_mse
                row[f"shuffle_params_{metric}"] = counterfactual_mse
            row["shuffle_params_prediction_mse"] = prediction_mse
            rows.append(row)

        summary = summarize_rows(rows, "shuffle_params", samples=100, seed=3)

        self.assertEqual(set(summary), {"overall", "elastic", "plasticine", "sand"})
        self.assertEqual(
            set(summary["overall"]),
            {"full_rollout_mse", "gm_mse", "long_seg_mse", "fde"},
        )
        elastic = summary["elastic"]["full_rollout_mse"]
        self.assertEqual(elastic["normal_mean"], 2.0)
        self.assertEqual(elastic["counterfactual_mean"], 4.0)
        self.assertEqual(elastic["mean_delta"], 2.0)
        self.assertEqual(elastic["relative_change_pct"], 100.0)
        self.assertEqual(elastic["response_ratio_pct"], 50.0)
        self.assertEqual(elastic["label"], "used")
        self.assertAlmostEqual(
            summary["overall"]["gm_mse"]["response_ratio_pct"], 17.0 / 14.0 * 100.0
        )

    def test_summarize_rows_supports_component_interventions(self):
        rows = self._summary_rows()

        for intervention, counterfactual_mean, response_ratio_pct in (
            ("shuffle_e", 2.2, 25.0),
            ("shuffle_nu", 2.4, 37.5),
        ):
            with self.subTest(intervention=intervention):
                summary = summarize_rows(rows, intervention, samples=100, seed=3)
                overall = summary["overall"]["full_rollout_mse"]
                self.assertAlmostEqual(overall["counterfactual_mean"], counterfactual_mean)
                self.assertAlmostEqual(overall["response_ratio_pct"], response_ratio_pct)

    def test_load_material_records_uses_basename_and_dataset_log10_e(self):
        from src.diagnose_material_condition import load_material_records

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, e_value, nu_value, mat_type in (
                ("elastic.h5", 1_000.0, 0.2, 0),
                ("sand.h5", 100_000.0, 0.35, 2),
            ):
                with h5py.File(root / name, "w") as handle:
                    handle["E"] = e_value
                    handle["nu"] = nu_value
                    handle["mat_type"] = mat_type

            records = load_material_records(root, ["elastic.h5", "nested/sand.h5"])

        self.assertEqual([record.model for record in records], ["elastic.h5", "sand.h5"])
        self.assertEqual([record.mat_type for record in records], [0, 2])
        self.assertAlmostEqual(records[0].log10_e, 3.0)
        self.assertAlmostEqual(records[1].log10_e, 5.0)
        self.assertAlmostEqual(records[0].nu, 0.2)

    def test_load_material_records_reports_model_for_invalid_metadata(self):
        from src.diagnose_material_condition import load_material_records

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with h5py.File(root / "missing-field.h5", "w") as handle:
                handle["E"] = 1_000.0
                handle["nu"] = 0.2
            with h5py.File(root / "duplicate.h5", "w") as handle:
                handle["E"] = 1_000.0
                handle["nu"] = 0.2
                handle["mat_type"] = 0

            with self.assertRaisesRegex(ValueError, "missing-field.h5"):
                load_material_records(root, ["missing-field.h5"])
            with self.assertRaisesRegex(ValueError, "missing-file.h5"):
                load_material_records(root, ["missing-file.h5"])
            with self.assertRaisesRegex(ValueError, r"duplicate.h5: duplicate model record"):
                load_material_records(root, ["duplicate.h5", "other/duplicate.h5"])

    def test_b0_identity_requires_all_registered_config_fields(self):
        from src.diagnose_material_condition import (
            EXPECTED_MODEL_CONFIG,
            EXPECTED_MODEL_DEFAULT_VALUES,
            EXPECTED_TOP_LEVEL_CONFIG,
            EXPECTED_TRAIN_DATASET_CONFIG,
            _validate_b0_identity,
        )

        records = self._b0_records()
        _validate_b0_identity(self._b0_args(), records)

        mismatches = {
            "input_frames": 4,
            "output_frames": 5,
            "use_diffusion": True,
            "num_inference_steps": 2,
            "eval_batch_size": 2,
            "floor_projection": True,
            "pred_offset": False,
            "seed": 1,
            "pc_size": 1024,
            "model_type": "mdm",
            "train_dataset.category": "other-category",
            "train_dataset.dataset_list": "OTHER_LIST",
            "train_dataset.has_gravity": False,
            "train_dataset.max_num_forces": 2,
            "train_dataset.norm_fac": 4,
            "train_dataset.stage": "shape",
            "train_dataset.mode": "ae",
            "train_dataset.pc_size": 1024,
            "train_dataset.repeat": 2,
            "train_dataset.seed": 1,
            "train_dataset.n_sample_pro_model": 200,
            "train_dataset.n_frames_interval": 2,
            "train_dataset.n_training_frames": 23,
            "train_dataset.batch_size": 19,
            "train_dataset.overfit": True,
            "train_dataset.input_frames": 4,
            "train_dataset.output_frames": 5,
            "model_config.n_layers": 5,
            "model_config.latent_dim": 128,
            "model_config.frame_cond": False,
            "model_config.point_embed": False,
            "model_config.mask_cond": False,
            "model_config.pred_offset": False,
            "model_config.num_neighbors": 8,
            "model_config.floor_cond": False,
            "model_config.max_num_forces": 2,
            "model_config.force_as_token": True,
            "model_config.force_as_latent": True,
            "model_config.gravity_emb": False,
            "model_config.coeff_cond": True,
            "model_config.num_mat": 3,
            "model_config.class_token": False,
            "model_config.transformer_block": "SpatialTemporalTransformerBlockv3",
            "model_config.contact_particle_cond": False,
            "model_config.contact_feature_sigma": 0.08,
            "model_config.contact_injection_mode": "shared",
            "model_config.contact_velocity_mode": "xyz",
            "model_config.contact_feature_mask": [1, 0, 1],
            "model_config.contact_bias_scale": 0.5,
        }
        expected_fields = set(EXPECTED_TOP_LEVEL_CONFIG)
        expected_fields.update(
            f"model_config.{field}" for field in EXPECTED_MODEL_CONFIG
        )
        expected_fields.update(
            f"model_config.{field}" for field in EXPECTED_MODEL_DEFAULT_VALUES
        )
        expected_fields.update(
            f"train_dataset.{field}" for field in EXPECTED_TRAIN_DATASET_CONFIG
        )
        self.assertEqual(set(mismatches), expected_fields)
        for field, value in mismatches.items():
            with self.subTest(field=field):
                args = self._b0_args()
                owner = args
                parts = field.split(".")
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                setattr(owner, parts[-1], value)
                with self.assertRaisesRegex(ValueError, field.replace(".", r"\.")):
                    _validate_b0_identity(args, records)

        explicit_defaults = self._b0_args()
        explicit_defaults.model_config.contact_injection_mode = "separate"
        explicit_defaults.model_config.contact_velocity_mode = "vertical"
        explicit_defaults.model_config.contact_feature_mask = [1, 1, 1]
        explicit_defaults.model_config.contact_bias_scale = 1.0
        _validate_b0_identity(explicit_defaults, records)

    def test_b0_identity_accepts_legacy_and_registered_profiles(self):
        from src.diagnose_material_condition import _validate_b0_identity

        _validate_b0_identity(self._b0_args(), self._b0_records(), profile=None)
        _validate_b0_identity(
            self._b0_args(), self._b0_records(), profile="contact_cond90"
        )
        _validate_b0_identity(
            self._factorized_b0_args(),
            self._b0_records(),
            profile="factorized90",
        )

    def test_b0_identity_rejects_crossed_and_unknown_profiles(self):
        from src.diagnose_material_condition import _validate_b0_identity

        contact_args = self._b0_args()
        factorized_args = self._factorized_b0_args()

        with self.assertRaisesRegex(ValueError, "mismatch"):
            _validate_b0_identity(
                contact_args, self._b0_records(), profile="factorized90"
            )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            _validate_b0_identity(
                factorized_args, self._b0_records(), profile="contact_cond90"
            )
        with self.assertRaisesRegex(ValueError, "unknown.*profile"):
            _validate_b0_identity(
                contact_args, self._b0_records(), profile="not-registered"
            )

    def test_b0_identity_requires_checkpoint_suffix_component_boundary(self):
        from src.diagnose_material_condition import _validate_b0_identity

        cases = (
            (
                "contact_cond90",
                self._b0_args(),
                "attackeroutputs/mm3_contact_cond_8L/"
                "checkpoint-90000/model.safetensors",
            ),
            (
                "factorized90",
                self._factorized_b0_args(),
                "attackeroutputs/mm3_contact_vxyz_factorized_8L/"
                "checkpoint-90000/model.safetensors",
            ),
        )
        for profile, args, malicious_resume in cases:
            with self.subTest(profile=profile):
                args.resume = malicious_resume
                with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                    _validate_b0_identity(
                        args, self._b0_records(), profile=profile
                    )

    def test_factorized_profile_requires_each_locked_model_default(self):
        from src.diagnose_material_condition import _validate_b0_identity

        for field, value in (
            ("contact_injection_mode", "separate"),
            ("contact_velocity_mode", "vertical"),
            ("contact_feature_mask", [1, 1, 1]),
            ("contact_bias_scale", 0.5),
        ):
            with self.subTest(field=field):
                args = self._factorized_b0_args()
                setattr(args.model_config, field, value)

                with self.assertRaisesRegex(ValueError, field):
                    _validate_b0_identity(
                        args, self._b0_records(), profile="factorized90"
                    )

    def test_b0_identity_allows_only_declared_nonsemantic_runtime_fields(self):
        from src.diagnose_material_condition import _validate_b0_identity

        args = self._b0_args()
        args.dataloader_num_workers = 12
        args.vis_dir = "another-visualization-directory"
        args.per_model_csv = False
        args.contact_eval_margin = 123.0

        _validate_b0_identity(args, self._b0_records())

    def test_b0_identity_rejects_unregistered_config_fields(self):
        from src.diagnose_material_condition import _validate_b0_identity

        for section, field in (
            ("top-level", "unexpected_semantic"),
            ("model_config", "pred_velocity"),
            ("train_dataset", "rollout_random_window"),
        ):
            with self.subTest(section=section, field=field):
                args = self._b0_args()
                target = args if section == "top-level" else getattr(args, section)
                setattr(target, field, True)

                with self.assertRaisesRegex(ValueError, rf"unexpected.*{field}"):
                    _validate_b0_identity(args, self._b0_records())

    def test_b0_identity_requires_expected_paths_counts_and_unique_metadata_set(self):
        from src.diagnose_material_condition import _validate_b0_identity

        records = self._b0_records()
        args = self._b0_args()

        args.resume = "outputs/mm3_contact_full_8L/checkpoint-90000/model.safetensors"
        with self.assertRaisesRegex(ValueError, r"actual=.*expected=.*mm3_contact_cond_8L"):
            _validate_b0_identity(args, records)
        args.resume = r"D:\runs\outputs\mm3_contact_cond_8L\checkpoint-90000\model.safetensors"
        args.train_dataset.dataset_path = "mm3_data/not_the_b0_test"
        with self.assertRaisesRegex(ValueError, r"actual=.*expected=.*mm3_data/mm3_test"):
            _validate_b0_identity(args, records)
        args.train_dataset.dataset_path = r"D:\data\mm3_data\mm3_test"
        with self.assertRaisesRegex(ValueError, r"actual=.*expected=.*\{0: 13, 1: 14, 2: 14\}"):
            _validate_b0_identity(args, records[:-1])
        duplicate_records = list(records)
        duplicate_records[-1] = MaterialRecord(
            records[0].model,
            duplicate_records[-1].mat_type,
            duplicate_records[-1].log10_e,
            duplicate_records[-1].nu,
        )
        with self.assertRaisesRegex(ValueError, "duplicate material metadata model"):
            _validate_b0_identity(args, duplicate_records)

    def test_b0_identity_freezes_exact_model_set_with_actionable_difference(self):
        from src.diagnose_material_condition import (
            EXPECTED_MODEL_NAMES,
            _validate_b0_identity,
        )

        self.assertEqual(EXPECTED_MODEL_NAMES, frozenset(EXPECTED_B0_MODEL_NAMES))
        records = self._b0_records()
        replaced = records[0]
        records[0] = MaterialRecord(
            "39998_000.h5",
            replaced.mat_type,
            replaced.log10_e,
            replaced.nu,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"missing=.*39478_000\.h5.*unexpected=.*39998_000\.h5",
        ):
            _validate_b0_identity(self._b0_args(), records)

    def test_build_raw_reference_uses_interval_point_indices_and_normalization(self):
        from src.diagnose_material_condition import _build_raw_reference

        class DatasetConfig(SimpleNamespace):
            def get(self, name, default=None):
                return getattr(self, name, default)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.arange(50 * 4 * 3, dtype=np.float32).reshape(50, 4, 3)
            with h5py.File(root / "model.h5", "w") as handle:
                handle["x"] = raw
            batch = {
                "model": ["nested/model.h5"],
                "point_indices": torch.tensor([[3, 1]]),
            }
            config = DatasetConfig(
                dataset_path=str(root), n_frames_interval=2, norm_fac=10.0
            )

            reference = _build_raw_reference(batch, config)

        expected = torch.from_numpy(raw[np.arange(25) * 2][:, [3, 1]]).float()
        expected = (expected - 10.0) / 2
        torch.testing.assert_close(reference, expected.unsqueeze(0))

    def test_normal_rollout_rejects_batch_material_mismatch_before_inference(self):
        from src.diagnose_material_condition import rollout_counterfactuals

        record = MaterialRecord("model.h5", 0, 3.0, 0.2)
        matching_batch = {
            "E": torch.tensor([[3.0 + 1e-6]]),
            "nu": torch.tensor([[0.2 + 1e-7]]),
            "mat_type": torch.tensor([0]),
        }
        mismatches = {
            "E": torch.tensor([[4.0]]),
            "nu": torch.tensor([[0.4]]),
            "mat_type": torch.tensor([2]),
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                batch = dict(matching_batch)
                batch[field] = value
                with mock.patch(
                    "src.diagnose_material_condition.rollout_condition"
                ) as rollout, mock.patch(
                    "src.diagnose_material_condition.torch.autocast"
                ) as autocast:
                    with self.assertRaisesRegex(ValueError, rf"model\.h5.*{field}"):
                        rollout_counterfactuals(
                            pipeline=object(),
                            batch=batch,
                            args=SimpleNamespace(),
                            record=record,
                            shuffled_parameters=(4.5, 0.35),
                        )
                rollout.assert_not_called()
                autocast.assert_not_called()

    def test_strict_checkpoint_loader_rejects_state_dict_mismatch(self):
        from src.diagnose_material_condition import _load_checkpoint_strict

        class Model:
            def __init__(self, error=None):
                self.calls = []
                self.error = error

            def load_state_dict(self, checkpoint, strict):
                self.calls.append((checkpoint, strict))
                if self.error is not None:
                    raise self.error

        loader = mock.Mock(return_value={"weights": torch.tensor(1.0)})
        model = Model()

        _load_checkpoint_strict(model, "checkpoint.safetensors", loader)

        loader.assert_called_once_with("checkpoint.safetensors", device="cpu")
        self.assertEqual(model.calls, [({"weights": torch.tensor(1.0)}, True)])
        with self.assertRaisesRegex(RuntimeError, "missing key"):
            _load_checkpoint_strict(Model(RuntimeError("missing key")), "bad.safetensors", loader)

    def test_three_rollouts_share_one_cuda_autocast_context(self):
        from src.diagnose_material_condition import rollout_counterfactuals

        events = []

        class RecordingAutocast:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, exc_type, exc_value, traceback):
                events.append("exit")

        record = MaterialRecord("model.h5", 0, 3.0, 0.2)
        batch = {
            "E": torch.tensor([[3.0]]),
            "nu": torch.tensor([[0.2]]),
            "mat_type": torch.tensor([0]),
        }

        def fake_rollout(pipeline, batch, args, e_value, nu_value, mat_type):
            events.append((e_value, nu_value, mat_type))
            return torch.tensor([mat_type])

        with mock.patch(
            "src.diagnose_material_condition.torch.autocast",
            return_value=RecordingAutocast(),
        ) as autocast, mock.patch(
            "src.diagnose_material_condition.rollout_condition",
            side_effect=fake_rollout,
        ):
            normal, shuffled_params, shuffled_class = rollout_counterfactuals(
                pipeline=object(),
                batch=batch,
                args=SimpleNamespace(),
                record=record,
                shuffled_parameters=(4.5, 0.35),
            )

        autocast.assert_called_once_with("cuda", dtype=torch.bfloat16)
        self.assertEqual(events, ["enter", (3.0, 0.2, 0), (4.5, 0.35, 0), (3.0, 0.2, 1), "exit"])
        torch.testing.assert_close(normal, torch.tensor([0]))
        torch.testing.assert_close(shuffled_params, torch.tensor([0]))
        torch.testing.assert_close(shuffled_class, torch.tensor([1]))

    def test_rollout_condition_isolates_intervention_without_mutating_batch(self):
        from src.diagnose_material_condition import rollout_condition

        class RecordingPipeline:
            def __init__(self):
                self.calls = []

            def __call__(
                self,
                init_pc,
                force,
                E,
                nu,
                mask,
                drag_point,
                floor_height,
                gravity,
                coeff,
                **kwargs,
            ):
                noise = torch.randn(
                    init_pc[:, -1:].shape,
                    generator=kwargs["generator"],
                    dtype=init_pc.dtype,
                )
                self.calls.append(
                    {
                        "force": force.clone(),
                        "E": E.clone(),
                        "nu": nu.clone(),
                        "mask": mask.clone(),
                        "drag_point": drag_point.clone(),
                        "floor_height": floor_height.clone(),
                        "gravity": gravity.clone(),
                        "coeff": coeff.clone(),
                        "y": kwargs["y"].clone(),
                        "start_vel": kwargs["start_vel"].clone(),
                        "points_rest": kwargs["points_rest"].clone(),
                        "seed": kwargs["generator"].initial_seed(),
                        "noise": noise.clone(),
                    }
                )
                return init_pc[:, -1:] + 1.0 + noise

        points = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1).repeat(1, 1, 1, 3)
        batch = {
            "points_src": points,
            "force": torch.tensor([[1.0, 2.0, 3.0]]),
            "E": torch.tensor([[3.0]]),
            "nu": torch.tensor([[0.1]]),
            "mask": torch.ones(1, 1, 1, 2),
            "drag_point": torch.tensor([[[4.0, 5.0, 6.0]]]),
            "floor_height": torch.tensor([0.25]),
            "gravity": torch.tensor([[0.0, -1.0, 0.0]]),
            "base_drag_coeff": torch.tensor([[0.4]]),
            "mat_type": torch.tensor([0]),
            "start_vel": torch.full((1, 1, 3), -7.0),
        }
        original = {key: value.clone() for key, value in batch.items()}
        args = SimpleNamespace(
            input_frames=5,
            output_frames=1,
            eval_batch_size=1,
            num_inference_steps=1,
            seed=17,
            device="cpu",
        )
        batch["points_rest"] = torch.tensor([[[7.0, 8.0, 9.0]]])
        original["points_rest"] = batch["points_rest"].clone()
        pipeline = RecordingPipeline()

        normal = rollout_condition(pipeline, batch, args, 3.0, 0.1, 0)
        shuffled_params = rollout_condition(pipeline, batch, args, 4.5, 0.33, 0)
        shuffled_class = rollout_condition(pipeline, batch, args, 3.0, 0.1, 2)

        for output in (normal, shuffled_params, shuffled_class):
            self.assertEqual(tuple(output.shape), (1, 25, 1, 3))
            torch.testing.assert_close(output[:, :5], points)
        self.assertEqual(len(pipeline.calls), 60)
        paths = [pipeline.calls[index : index + 20] for index in range(0, 60, 20)]
        for step in range(20):
            normal_call, params_call, class_call = (path[step] for path in paths)
            self.assertEqual(
                [call["seed"] for call in (normal_call, params_call, class_call)],
                [17, 17, 17],
            )
            torch.testing.assert_close(normal_call["noise"], params_call["noise"])
            torch.testing.assert_close(normal_call["noise"], class_call["noise"])
            for key in ("force", "mask", "drag_point", "floor_height", "gravity", "coeff", "points_rest"):
                expected_key = "base_drag_coeff" if key == "coeff" else key
                expected = original[expected_key]
                if key == "mask":
                    expected = expected[..., :1]
                for call in (normal_call, params_call, class_call):
                    torch.testing.assert_close(call[key], expected)
            torch.testing.assert_close(normal_call["E"], torch.tensor([[3.0]]))
            torch.testing.assert_close(normal_call["nu"], torch.tensor([[0.1]]))
            torch.testing.assert_close(normal_call["y"], torch.tensor([0]))
            torch.testing.assert_close(params_call["E"], torch.tensor([[4.5]]))
            torch.testing.assert_close(params_call["nu"], torch.tensor([[0.33]]))
            torch.testing.assert_close(params_call["y"], normal_call["y"])
            torch.testing.assert_close(class_call["E"], normal_call["E"])
            torch.testing.assert_close(class_call["nu"], normal_call["nu"])
            torch.testing.assert_close(class_call["y"], torch.tensor([2]))
        torch.testing.assert_close(paths[0][0]["start_vel"], original["start_vel"])
        torch.testing.assert_close(paths[0][1]["start_vel"], torch.ones(1, 1, 3))
        for key, value in batch.items():
            torch.testing.assert_close(value, original[key])

    def test_shuffle_class_markdown_explains_its_diagnostic_scope(self):
        from src.diagnose_material_condition import _write_markdown

        rows = self._summary_rows(("z.h5", "a.h5", "m.h5"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            _write_markdown(path, rows, samples=10, seed=0)
            markdown = path.read_text(encoding="utf-8")

        self.assertIn("## shuffle_class", markdown)
        self.assertIn("only measures class-condition dependence", markdown)
        self.assertIn("does not represent physical accuracy", markdown)
        self.assertIn("## Evaluated Models", markdown)
        self.assertLess(markdown.index("`a.h5`"), markdown.index("`m.h5`"))
        self.assertLess(markdown.index("`m.h5`"), markdown.index("`z.h5`"))

    def test_csv_contains_key_columns_and_sorts_models(self):
        from src.diagnose_material_condition import _write_csv

        rows = self._summary_rows(("z.h5", "a.h5", "m.h5"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.csv"
            _write_csv(path, rows)
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                written = list(reader)

        self.assertEqual([row["model"] for row in written], ["a.h5", "m.h5", "z.h5"])
        key_columns = {
            "model",
            "mat_type",
            "true_log10_e",
            "true_nu",
            "shuffled_log10_e",
            "shuffled_nu",
            "shuffled_mat_type",
            "normal_full_rollout_mse",
            "shuffle_params_full_rollout_mse",
            "shuffle_class_full_rollout_mse",
            "shuffle_params_prediction_mse",
            "shuffle_class_prediction_mse",
        }
        self.assertTrue(key_columns.issubset(reader.fieldnames))

    def test_run_diagnostics_orchestrates_start_zero_models_and_writes_reports(self):
        import src.diagnose_material_condition as diagnostic

        model_names = (
            "10000_000.h5",
            "10001_000.h5",
            "20000_001.h5",
            "20001_001.h5",
            "30000_002.h5",
            "30001_002.h5",
        )

        class FakeDataset:
            def __init__(self, split, config):
                self.split = split
                self.config = config
                self.split_lst_save = list(model_names)

        class FakeModel:
            def __init__(self):
                self.loaded_strict = None
                self.requires_grad = None

            def to(self, device):
                self.device = device
                return self

            def load_state_dict(self, checkpoint, strict):
                self.loaded_strict = strict

            def eval(self):
                return self

            def requires_grad_(self, enabled):
                self.requires_grad = enabled
                return self

        fake_model = FakeModel()
        model_constructor = mock.Mock(return_value=fake_model)
        fake_pipeline = object()
        pipeline_constructor = mock.Mock(return_value=fake_pipeline)
        dataset_module = ModuleType("dataset.traj_dataset")
        dataset_module.TrajDataset = FakeDataset
        dataset_package = ModuleType("dataset")
        dataset_package.__path__ = []
        model_module = ModuleType("model.spacetime")
        model_module.MDM_ST = model_constructor
        model_package = ModuleType("model")
        model_package.__path__ = []
        pipeline_module = ModuleType("pipeline_traj")
        pipeline_module.TrajPipeline = pipeline_constructor
        imported_modules = {
            "dataset": dataset_package,
            "dataset.traj_dataset": dataset_module,
            "model": model_package,
            "model.spacetime": model_module,
            "pipeline_traj": pipeline_module,
        }

        def fake_rollouts(pipeline, batch, args, record, shuffled_parameters):
            self.assertIs(pipeline, fake_pipeline)
            baseline = torch.full((1, 25, 1, 3), -2.5)
            outputs = []
            for delta in (0.1, 0.2, 0.3):
                trajectory = baseline.clone()
                trajectory[:, 5:] += delta
                outputs.append(trajectory)
            return tuple(outputs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = (
                root
                / "outputs"
                / "mm3_contact_cond_8L"
                / "checkpoint-90000"
                / "model.safetensors"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            dataset_root = root / "mm3_data" / "mm3_test"
            dataset_root.mkdir(parents=True)
            for index, model_name in enumerate(model_names):
                mat_type = int(Path(model_name).stem.rsplit("_", 1)[1])
                with h5py.File(dataset_root / model_name, "w") as handle:
                    handle["E"] = 10.0 ** (3.0 + index / 10.0)
                    handle["nu"] = 0.2 + index / 100.0
                    handle["mat_type"] = mat_type
                    handle["x"] = np.zeros((25, 1, 3), dtype=np.float32)

            nonzero_batch = {
                "model": [model_names[0]],
                "start_idx": torch.tensor([5]),
            }
            start_zero_batches = [
                {
                    "model": [model_name],
                    "start_idx": torch.tensor([0]),
                    "point_indices": torch.tensor([[0]]),
                }
                for model_name in reversed(model_names)
            ]
            dataloader = [
                (nonzero_batch, {}),
                *((batch, {}) for batch in start_zero_batches),
            ]
            from src.options import TestingConfig

            config_path = PROJECT_ROOT / "src" / "configs" / "eval_mm3_contact_cond.yaml"
            args = OmegaConf.merge(
                OmegaConf.structured(TestingConfig),
                OmegaConf.load(config_path),
            )
            args.resume = str(checkpoint)
            args.train_dataset.dataset_path = str(dataset_root)
            output_dir = root / "reports"

            with mock.patch.dict(sys.modules, imported_modules), mock.patch(
                "safetensors.torch.load_file", return_value={"weight": 1}
            ) as checkpoint_loader, mock.patch.object(
                diagnostic.torch, "compile", side_effect=lambda model: model
            ), mock.patch.object(
                diagnostic.torch.utils.data,
                "DataLoader",
                return_value=dataloader,
            ) as dataloader_constructor, mock.patch.object(
                diagnostic,
                "rollout_counterfactuals",
                side_effect=fake_rollouts,
            ) as rollout, mock.patch.object(
                diagnostic, "EXPECTED_MODEL_COUNT", len(model_names)
            ), mock.patch.object(
                diagnostic,
                "EXPECTED_MATERIAL_COUNTS",
                {0: 2, 1: 2, 2: 2},
            ), mock.patch.object(
                diagnostic, "EXPECTED_MODEL_NAMES", frozenset(model_names)
            ), io.StringIO() as progress, redirect_stdout(progress):
                rows = diagnostic.run_diagnostics(
                    args,
                    output_dir=output_dir,
                    permutation_seed=3,
                    bootstrap_samples=20,
                )

            csv_path = output_dir / "material_condition_b0_seed3.csv"
            markdown_path = output_dir / "material_condition_b0_seed3.md"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual([row["model"] for row in rows], sorted(model_names))
        self.assertEqual([row["model"] for row in csv_rows], sorted(model_names))
        self.assertEqual(rollout.call_count, len(model_names))
        self.assertEqual(
            {call.args[3].model for call in rollout.call_args_list},
            set(model_names),
        )
        self.assertAlmostEqual(rows[0]["normal_full_rollout_mse"], 0.008)
        self.assertAlmostEqual(rows[0]["shuffle_params_full_rollout_mse"], 0.032)
        self.assertAlmostEqual(rows[0]["shuffle_class_full_rollout_mse"], 0.072)
        self.assertAlmostEqual(rows[0]["shuffle_params_prediction_mse"], 0.01)
        self.assertIn("Count: 6", markdown)
        self.assertIn("only measures class-condition dependence", markdown)
        checkpoint_loader.assert_called_once_with(str(checkpoint), device="cpu")
        self.assertTrue(fake_model.loaded_strict)
        self.assertFalse(fake_model.requires_grad)
        model_constructor.assert_called_once_with(
            2048,
            1,
            n_feats=3,
            model_config=args.model_config,
        )
        pipeline_constructor.assert_called_once_with(model=fake_model, scheduler=None)
        dataloader_constructor.assert_called_once_with(
            mock.ANY,
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )

    def test_progress_and_completion_output_are_concise_and_include_artifacts(self):
        from src.diagnose_material_condition import (
            _print_completion,
            _print_model_progress,
        )

        rows = self._summary_rows()
        with tempfile.TemporaryDirectory() as directory, io.StringIO() as output:
            with redirect_stdout(output):
                _print_model_progress(2, 41, "model.h5")
                _print_completion(Path(directory), permutation_seed=7, rows=rows, samples=10)
            rendered = output.getvalue()

        self.assertIn("[2/41] model.h5 complete", rendered)
        self.assertIn("material_condition_b0_seed7.csv", rendered)
        self.assertIn("material_condition_b0_seed7.md", rendered)
        self.assertIn("Overall summary", rendered)
        self.assertIn("shuffle_params", rendered)
        self.assertIn("shuffle_class", rendered)

    def test_main_prints_completion_after_diagnostics_finish(self):
        import src.diagnose_material_condition as diagnostic

        cli_args = SimpleNamespace(
            config="registered.yaml",
            output_dir="results/b0",
            permutation_seed=7,
            bootstrap_samples=10,
        )
        config = self._b0_args()
        rows = self._summary_rows()
        omega_conf = mock.Mock()
        omega_conf.structured.return_value = object()
        omega_conf.load.return_value = object()
        omega_conf.merge.return_value = config
        omega_module = ModuleType("omegaconf")
        omega_module.OmegaConf = omega_conf
        options_module = ModuleType("options")
        options_module.TestingConfig = object

        with mock.patch.dict(
            sys.modules,
            {"omegaconf": omega_module, "options": options_module},
        ), mock.patch.object(
            diagnostic, "_parse_args", return_value=cli_args
        ), mock.patch.object(
            diagnostic, "run_diagnostics", return_value=rows
        ) as run, mock.patch.object(
            diagnostic, "_print_completion"
        ) as completion:
            diagnostic.main()

        run.assert_called_once_with(
            config,
            output_dir=Path("results/b0"),
            permutation_seed=7,
            bootstrap_samples=10,
        )
        completion.assert_called_once_with(
            Path("results/b0"),
            permutation_seed=7,
            rows=rows,
            samples=10,
        )

    def test_diagnose_cli_help_requires_no_runtime_inputs(self):
        repository = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "src/diagnose_material_condition.py", "--help"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--config", "--output-dir", "--permutation-seed", "--bootstrap-samples"):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
