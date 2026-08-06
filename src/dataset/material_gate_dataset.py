import json
import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.traj_dataset import TrajDataset


MATERIAL_NAMES = {0: "elastic", 1: "plasticine", 2: "sand"}


def _model_names(dataset_path, dataset_list):
    if dataset_list and os.path.exists(dataset_list):
        with open(dataset_list, "r", encoding="utf-8") as handle:
            names = list(json.load(handle))
    else:
        names = sorted(name for name in os.listdir(dataset_path) if name.endswith(".h5"))

    dataset_root = os.path.realpath(dataset_path)
    seen = set()
    for name in names:
        if (
            not isinstance(name, str)
            or name != os.path.basename(name)
            or not name.endswith(".h5")
        ):
            raise ValueError("dataset_list entries must be plain H5 basenames")
        if name in seen:
            raise ValueError(f"dataset_list contains duplicate model {name!r}")
        seen.add(name)
    for name in names:
        resolved = os.path.realpath(os.path.join(dataset_root, name))
        if os.path.commonpath((dataset_root, resolved)) != dataset_root:
            raise ValueError("dataset_list entries must stay inside mm3_train")
        if not os.path.isfile(resolved):
            raise ValueError(f"dataset model does not exist: {name}")
    return names


def build_material_split(dataset_path, dataset_list, train_fraction=0.8, seed=0):
    if os.path.basename(os.path.normpath(dataset_path)) != "mm3_train":
        raise ValueError("material-gate calibration must read an mm3_train dataset")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")

    grouped = {material_id: [] for material_id in MATERIAL_NAMES}
    for model_name in _model_names(dataset_path, dataset_list):
        with h5py.File(os.path.join(dataset_path, model_name), "r") as model:
            material_id = (
                int(np.asarray(model["mat_type"]).reshape(-1)[0])
                if "mat_type" in model
                else 0
            )
        if material_id in grouped:
            grouped[material_id].append(model_name)

    manifest = {
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "materials": {},
    }
    for material_id, material_name in MATERIAL_NAMES.items():
        names = sorted(grouped[material_id])
        if len(names) < 2:
            raise ValueError(
                f"material {material_name} needs at least two models for train/val split"
            )
        random.Random(seed + material_id).shuffle(names)
        val_count = max(1, round(len(names) * (1.0 - train_fraction)))
        val_count = min(val_count, len(names) - 1)
        manifest["materials"][material_name] = {
            "train": names[val_count:],
            "val": names[:val_count],
        }
    return manifest


class MaterialGateDataset(Dataset):
    def __init__(
        self,
        dataset_cfg,
        model_names,
        seed=0,
        max_rollout_steps=20,
        resample_random_start=True,
    ):
        if os.path.basename(os.path.normpath(dataset_cfg.dataset_path)) != "mm3_train":
            raise ValueError("material-gate calibration must read an mm3_train dataset")
        if int(dataset_cfg.get("output_frames", 1)) != 1:
            raise ValueError("material-gate calibration requires output_frames=1")
        if max_rollout_steps < 1:
            raise ValueError("max_rollout_steps must be positive")

        self.base_dataset = TrajDataset("train", dataset_cfg)
        self.dataset_path = dataset_cfg.dataset_path
        self.input_frames = int(dataset_cfg.get("input_frames", 5))
        self.frame_interval = int(dataset_cfg.get("n_frames_interval", 1))
        self.norm_fac = float(dataset_cfg.norm_fac)
        self.max_rollout_steps = int(max_rollout_steps)
        self.seed = int(seed)
        random_start_generator = random.Random(self.seed)
        self.sample_specs = []

        for model_name in model_names:
            with h5py.File(os.path.join(self.dataset_path, model_name), "r") as model:
                frame_count = int(model["x"].shape[0])
            max_start = frame_count - 1 - self.input_frames * self.frame_interval
            if max_start < 1:
                raise ValueError(
                    f"{model_name} does not have enough frames for both start0 and random windows"
                )
            if resample_random_start:
                random_spec = {
                    "model": model_name,
                    "start_idx": -1,
                    "min_start": 1,
                    "max_start": max_start,
                }
            else:
                random_spec = {
                    "model": model_name,
                    "start_idx": random_start_generator.randint(1, max_start),
                }
            self.sample_specs.extend(
                [{"model": model_name, "start_idx": 0}, random_spec]
            )

    def __len__(self):
        return len(self.sample_specs)

    def __getitem__(self, index):
        return self._load_spec(self.sample_specs[index])

    def load_window(self, model_name, start_idx):
        return self._load_spec({"model": model_name, "start_idx": int(start_idx)})

    def _load_spec(self, spec):
        model_data, model_info = self.base_dataset.get_deform_diff_from_spec(spec)
        start_idx = int(model_data["start_idx"])
        first_future = start_idx + self.input_frames * self.frame_interval

        with h5py.File(os.path.join(self.dataset_path, spec["model"]), "r") as model:
            positions = np.asarray(model["x"])
        future_indices = np.arange(
            first_future,
            positions.shape[0],
            self.frame_interval,
            dtype=np.int64,
        )[: self.max_rollout_steps]
        point_indices = model_data["point_indices"].cpu().numpy()
        future_gt = torch.from_numpy(positions[future_indices][:, point_indices]).float()
        model_data["future_gt"] = (future_gt - self.norm_fac) / 2.0
        model_info["future_indices"] = future_indices
        return model_data, model_info
