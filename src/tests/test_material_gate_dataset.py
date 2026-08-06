import os
import tempfile
import unittest

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from dataset.material_gate_dataset import MaterialGateDataset, build_material_split
from dataset.traj_dataset import TrajDataset


def write_model(path, material, *, frames=25, particles=4):
    positions = np.zeros((frames, particles, 3), dtype=np.float32)
    for frame in range(frames):
        positions[frame, :, 0] = frame + np.arange(particles) * 0.1
        positions[frame, :, 1] = 2.0 - frame * 0.01
        positions[frame, :, 2] = np.arange(particles) * 0.2
    identities = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (frames, particles, 3, 3),
    ).copy()
    zeros = np.zeros_like(identities)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=positions)
        h5.create_dataset("vol", data=np.ones(particles, dtype=np.float32))
        h5.create_dataset("F", data=identities)
        h5.create_dataset("C", data=zeros)
        h5.create_dataset("drag_mask", data=np.zeros(particles, dtype=np.bool_))
        h5.create_dataset("drag_force", data=np.zeros((1, 3), dtype=np.float32))
        h5.create_dataset("base_drag_coeff", data=np.ones((1, 1), dtype=np.float32))
        h5.create_dataset("drag_point", data=np.array([5.0, 5.0, 5.0], dtype=np.float32))
        h5.create_dataset("floor_height", data=np.array(1.0, dtype=np.float32))
        h5.create_dataset("gravity", data=np.array(0, dtype=np.int64))
        h5.create_dataset("mat_type", data=np.array(material, dtype=np.int64))
        h5.create_dataset("E", data=np.array(1.0e6, dtype=np.float32))
        h5.create_dataset("nu", data=np.array(0.3, dtype=np.float32))


def make_dataset_config(dataset_path):
    return OmegaConf.create(
        {
            "dataset_path": dataset_path,
            "dataset_list": "MISSING_DATASET_LIST",
            "stage": "deform",
            "mode": "diff",
            "repeat": 1,
            "seed": 0,
            "pc_size": 4,
            "n_sample_pro_model": 1,
            "n_frames_interval": 1,
            "n_training_frames": 24,
            "input_frames": 5,
            "output_frames": 1,
            "rollout_unroll_steps": 1,
            "rollout_random_window": False,
            "rollout_force_start0": False,
            "windows_per_model": 4,
            "train_extra_random_windows": 0,
            "contact_window_ratio": 0.0,
            "contact_margin": 0.04,
            "batch_size": 1,
            "has_gravity": True,
            "max_num_forces": 1,
            "overfit": False,
            "norm_fac": 5.0,
        }
    )


class TrajDatasetExplicitSpecTest(unittest.TestCase):
    def test_explicit_model_spec_matches_index_lookup(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = os.path.join(root, "mm3_train")
            os.makedirs(dataset_path)
            write_model(os.path.join(dataset_path, "elastic_000.h5"), 0)
            dataset = TrajDataset("train", make_dataset_config(dataset_path))
            dataset.models = [{"model": "elastic_000.h5", "start_idx": 0}]

            torch.manual_seed(0)
            indexed_data, indexed_info = dataset.get_deform_diff(0)
            torch.manual_seed(0)
            explicit_data, explicit_info = dataset.get_deform_diff_from_spec(
                {"model": "elastic_000.h5", "start_idx": 0}
            )

            self.assertEqual(indexed_info["model"], explicit_info["model"])
            np.testing.assert_array_equal(
                indexed_info["indices"], explicit_info["indices"]
            )
            self.assertEqual(set(indexed_data), set(explicit_data))
            for key in indexed_data:
                if torch.is_tensor(indexed_data[key]):
                    torch.testing.assert_close(
                        indexed_data[key],
                        explicit_data[key],
                        rtol=0.0,
                        atol=0.0,
                    )
                else:
                    self.assertEqual(indexed_data[key], explicit_data[key])


class MaterialSplitTest(unittest.TestCase):
    def test_split_is_stratified_reproducible_and_excludes_rigid(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = os.path.join(root, "mm3_train")
            os.makedirs(dataset_path)
            expected = {0: set(), 1: set(), 2: set()}
            for material in range(3):
                for index in range(5):
                    name = f"material_{material}_{index:03d}.h5"
                    expected[material].add(name)
                    write_model(os.path.join(dataset_path, name), material)
            write_model(os.path.join(dataset_path, "rigid_000.h5"), 3)

            first = build_material_split(
                dataset_path,
                "MISSING_DATASET_LIST",
                train_fraction=0.8,
                seed=7,
            )
            repeated = build_material_split(
                dataset_path,
                "MISSING_DATASET_LIST",
                train_fraction=0.8,
                seed=7,
            )
            other_seed = build_material_split(
                dataset_path,
                "MISSING_DATASET_LIST",
                train_fraction=0.8,
                seed=8,
            )

            self.assertEqual(first, repeated)
            self.assertNotEqual(first, other_seed)
            for material, name in enumerate(("elastic", "plasticine", "sand")):
                train = first["materials"][name]["train"]
                val = first["materials"][name]["val"]
                self.assertTrue(train)
                self.assertTrue(val)
                self.assertTrue(set(train).isdisjoint(val))
                self.assertEqual(set(train) | set(val), expected[material])
                self.assertNotIn("rigid_000.h5", train + val)

    def test_split_rejects_test_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = os.path.join(root, "mm3_test")
            os.makedirs(dataset_path)
            with self.assertRaisesRegex(ValueError, "mm3_train"):
                build_material_split(
                    dataset_path,
                    "MISSING_DATASET_LIST",
                    train_fraction=0.8,
                    seed=0,
                )


class MaterialGateDatasetTest(unittest.TestCase):
    def test_fixed_and_random_specs_form_equal_sampling_pool(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = os.path.join(root, "mm3_train")
            os.makedirs(dataset_path)
            write_model(os.path.join(dataset_path, "elastic_000.h5"), 0)

            dataset = MaterialGateDataset(
                make_dataset_config(dataset_path),
                ["elastic_000.h5"],
                seed=0,
                max_rollout_steps=20,
            )

            self.assertEqual(
                dataset.sample_specs,
                [
                    {"model": "elastic_000.h5", "start_idx": 0},
                    {
                        "model": "elastic_000.h5",
                        "start_idx": -1,
                        "min_start": 1,
                        "max_start": 19,
                    },
                ],
            )

    def test_future_gt_is_aligned_normalized_and_uses_same_particles(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = os.path.join(root, "mm3_train")
            os.makedirs(dataset_path)
            model_name = "elastic_000.h5"
            write_model(os.path.join(dataset_path, model_name), 0)
            dataset = MaterialGateDataset(
                make_dataset_config(dataset_path),
                [model_name],
                seed=0,
                max_rollout_steps=20,
            )

            start0, _ = dataset.load_window(model_name, 0)
            start5, _ = dataset.load_window(model_name, 5)

            self.assertEqual(tuple(start0["points_src"].shape), (5, 4, 3))
            self.assertEqual(tuple(start0["future_gt"].shape), (20, 4, 3))
            self.assertEqual(tuple(start5["future_gt"].shape), (15, 4, 3))
            torch.testing.assert_close(
                start0["future_gt"][0, :, 0],
                torch.tensor([0.0, 0.05, 0.10, 0.15]),
            )
            torch.testing.assert_close(
                start5["future_gt"][0, :, 0],
                torch.tensor([2.5, 2.55, 2.60, 2.65]),
            )
            torch.testing.assert_close(
                start0["future_gt"][0],
                start0["points_tgt"][0],
            )
            torch.testing.assert_close(
                start0["point_indices"],
                start5["point_indices"],
            )

    def test_random_spec_never_draws_start_zero(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = os.path.join(root, "mm3_train")
            os.makedirs(dataset_path)
            model_name = "elastic_000.h5"
            write_model(os.path.join(dataset_path, model_name), 0)
            dataset = MaterialGateDataset(
                make_dataset_config(dataset_path),
                [model_name],
                seed=0,
                max_rollout_steps=20,
            )

            np.random.seed(0)
            starts = [int(dataset[1][0]["start_idx"]) for _ in range(50)]
            self.assertGreaterEqual(min(starts), 1)
            self.assertLessEqual(max(starts), 19)


if __name__ == "__main__":
    unittest.main()
