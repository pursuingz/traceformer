import csv
import json
import os
from contextlib import nullcontext

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F


def _gate_parameter(model):
    matches = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.endswith("gate_logits")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one gate_logits parameter, found {[name for name, _ in matches]}"
        )
    name, parameter = matches[0]
    if tuple(parameter.shape) != (3, 4):
        raise RuntimeError(f"{name} must have shape (3, 4), got {tuple(parameter.shape)}")
    return name, parameter


def load_b3a_into_b3b(model, checkpoint_path):
    gate_name, _ = _gate_parameter(model)
    checkpoint = load_file(checkpoint_path, device="cpu")
    incompatible = model.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys != [gate_name] or incompatible.unexpected_keys:
        raise RuntimeError(
            "B3a checkpoint is incompatible with B3b: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )


def freeze_for_gate_training(model):
    _, gate = _gate_parameter(model)
    model.requires_grad_(False)
    gate.requires_grad_(True)
    return gate


def frame_loss_weights(num_frames, long_weight, *, device=None, dtype=torch.float32):
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if long_weight < 0:
        raise ValueError("long_weight must be non-negative")
    weights = torch.full(
        (num_frames,),
        1.0 / num_frames,
        device=device,
        dtype=dtype,
    )
    long_start = (2 * num_frames) // 3
    weights[long_start:] += long_weight / (num_frames - long_start)
    return weights


def gate_objective(
    frame_losses,
    gates,
    material_id,
    long_weight=0.5,
    reg_weight=1.0e-3,
):
    if frame_losses.ndim != 1 or frame_losses.numel() == 0:
        raise ValueError("frame_losses must be a non-empty 1D tensor")
    if tuple(gates.shape) != (3, 4):
        raise ValueError(f"gates must have shape (3, 4), got {tuple(gates.shape)}")
    if material_id not in (0, 1, 2):
        raise ValueError("material_id must be 0, 1, or 2")
    if reg_weight < 0:
        raise ValueError("reg_weight must be non-negative")

    long_start = (2 * frame_losses.numel()) // 3
    full_mse = frame_losses.mean()
    long_mse = frame_losses[long_start:].mean()
    regularizer = (gates[material_id] - 1.0).square().mean()
    objective = full_mse + long_weight * long_mse + reg_weight * regularizer
    return {
        "full_mse": full_mse,
        "long_mse": long_mse,
        "regularizer": regularizer,
        "objective": objective,
    }


def _gate_module(model, accelerator=None):
    unwrapped = (
        accelerator.unwrap_model(model)
        if accelerator is not None and hasattr(accelerator, "unwrap_model")
        else model
    )
    matches = [
        module
        for module in unwrapped.modules()
        if hasattr(module, "gate_logits")
        and callable(getattr(module, "material_stage_gates", None))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one material-stage gate module, found {len(matches)}")
    return matches[0]


def _move_sample(sample, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def _backward(loss, accelerator):
    if accelerator is None:
        loss.backward()
    else:
        accelerator.backward(loss)


def gate_rollout_loss(
    model,
    sample,
    material_id,
    long_weight=0.5,
    reg_weight=1.0e-3,
    accelerator=None,
    backward=True,
    noise_seed=0,
):
    gate_module = _gate_module(model, accelerator)
    device = gate_module.gate_logits.device
    batch = _move_sample(sample, device)
    current_input = batch["points_src"]
    future_gt = batch["future_gt"]
    if current_input.ndim != 4 or current_input.shape[1] < 2:
        raise ValueError("points_src must have shape (B,F,N,3) with at least two frames")
    if future_gt.ndim != 4 or future_gt.shape[1] < 1:
        raise ValueError("future_gt must have shape (B,T,N,3) with at least one frame")
    labels = batch["mat_type"].reshape(-1).long()
    if not torch.all(labels == material_id):
        raise ValueError("sample material labels do not match material_id")

    weights = frame_loss_weights(
        future_gt.shape[1],
        long_weight,
        device=device,
        dtype=torch.float32,
    )
    timestep = torch.zeros(current_input.shape[0], device=device, dtype=torch.long)
    null_emb = torch.ones(
        current_input.shape[0], 1, 1, device=device, dtype=current_input.dtype
    )
    frame_losses = []

    for step_index in range(future_gt.shape[1]):
        if step_index == 0:
            start_velocity = batch["start_vel"]
        else:
            start_velocity = current_input[:, 1] - current_input[:, 0]

        generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        noise = torch.randn(
            current_input[:, -1:].shape,
            generator=generator,
            dtype=current_input.dtype,
            device="cpu",
        ).to(device)
        model_input = current_input[:, -1:] + 0.02 * noise
        prediction = model(
            model_input,
            timestep,
            current_input,
            batch["force"],
            batch["E"],
            batch["nu"],
            batch["mask"][..., :1],
            batch["drag_point"],
            batch["floor_height"],
            batch["gravity"],
            batch["base_drag_coeff"],
            y=labels,
            null_emb=null_emb,
            start_vel=start_velocity,
            points_rest=batch.get("points_rest"),
        )
        frame_loss = F.mse_loss(
            prediction.float(),
            future_gt[:, step_index : step_index + 1].float(),
        )
        if backward:
            _backward(weights[step_index] * frame_loss, accelerator)
        frame_losses.append(frame_loss.detach())
        current_input = torch.cat([current_input, prediction.detach()], dim=1)[
            :, -current_input.shape[1] :
        ]

    gates = gate_module.material_stage_gates()
    regularizer = (gates[material_id] - 1.0).square().mean()
    if backward and reg_weight > 0:
        _backward(reg_weight * regularizer, accelerator)

    metrics = gate_objective(
        torch.stack(frame_losses),
        gates.detach(),
        material_id,
        long_weight=long_weight,
        reg_weight=reg_weight,
    )
    metrics["frame_mse"] = torch.stack(frame_losses)
    return metrics


class MaterialBestRowTracker:
    def __init__(
        self,
        num_materials,
        num_stages,
        validation_interval=25,
        patience=3,
    ):
        if num_materials < 1 or num_stages < 1:
            raise ValueError("num_materials and num_stages must be positive")
        if validation_interval < 1 or patience < 1:
            raise ValueError("validation_interval and patience must be positive")
        self.num_materials = int(num_materials)
        self.num_stages = int(num_stages)
        self.validation_interval = int(validation_interval)
        self.patience = int(patience)
        self.best_scores = [float("inf")] * self.num_materials
        self.identity_scores = [None] * self.num_materials
        self.best_update = [None] * self.num_materials
        self.best_rows = torch.zeros(self.num_materials, self.num_stages)
        self.bad_counts = [0] * self.num_materials

    def should_validate(self, update):
        return update > 0 and update % self.validation_interval == 0

    def record_identity(self, material_id, score, row):
        if not 0 <= material_id < self.num_materials:
            raise ValueError("material_id is out of range")
        if self.identity_scores[material_id] is not None:
            raise RuntimeError(f"identity baseline already recorded for material {material_id}")
        row = row.detach().cpu().reshape(-1)
        if row.numel() != self.num_stages:
            raise ValueError("gate row has the wrong number of stages")
        self.identity_scores[material_id] = float(score)
        self.best_scores[material_id] = float(score)
        self.best_update[material_id] = 0
        self.best_rows[material_id].copy_(row)
        self.bad_counts[material_id] = 0

    def observe(self, material_id, update, score, row):
        if not self.should_validate(update):
            raise ValueError("observation update must match the validation interval")
        if not 0 <= material_id < self.num_materials:
            raise ValueError("material_id is out of range")
        if self.identity_scores[material_id] is None:
            raise RuntimeError("identity baseline must be recorded before gate updates")
        row = row.detach().cpu().reshape(-1)
        if row.numel() != self.num_stages:
            raise ValueError("gate row has the wrong number of stages")

        if float(score) < self.best_scores[material_id]:
            self.best_scores[material_id] = float(score)
            self.best_update[material_id] = int(update)
            self.best_rows[material_id].copy_(row)
            self.bad_counts[material_id] = 0
        else:
            self.bad_counts[material_id] += 1
        return self.bad_counts[material_id] >= self.patience


def save_gate_artifacts(
    model,
    output_dir,
    split_manifest,
    training_history,
    metadata,
    accelerator=None,
):
    unwrapped = (
        accelerator.unwrap_model(model)
        if accelerator is not None and hasattr(accelerator, "unwrap_model")
        else model
    )
    gate_module = _gate_module(unwrapped)
    checkpoint_dir = os.path.join(output_dir, "checkpoint-best")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(output_dir, "split_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=False)

    history_path = os.path.join(output_dir, "training_history.csv")
    fieldnames = (
        list(training_history[0])
        if training_history
        else ["material", "update", "split", "full_mse", "long_mse", "score"]
    )
    with open(history_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_history)

    gate_logits = gate_module.gate_logits.detach().cpu()
    gates = gate_module.material_stage_gates().detach().cpu()
    gate_payload = {
        "gate_logits": gate_logits.tolist(),
        "gates": gates.tolist(),
    }
    gate_audit = dict(metadata)
    gate_audit.update(gate_payload)
    with open(os.path.join(output_dir, "best_gates.json"), "w", encoding="utf-8") as handle:
        json.dump(gate_audit, handle, indent=2, ensure_ascii=False)

    state = {
        key: value.detach().cpu().contiguous()
        for key, value in unwrapped.state_dict().items()
    }
    save_file(state, os.path.join(checkpoint_dir, "model.safetensors"))
    gate_metadata = gate_audit
    with open(
        os.path.join(checkpoint_dir, "gate_metadata.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(gate_metadata, handle, indent=2, ensure_ascii=False)


def _sample_dict(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _autocast_context(accelerator):
    return accelerator.autocast() if accelerator is not None else nullcontext()


def _validation_metrics(
    model,
    loader,
    material_id,
    long_weight,
    reg_weight,
    accelerator,
    noise_seed,
):
    totals = {"full_mse": 0.0, "long_mse": 0.0, "regularizer": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            with _autocast_context(accelerator):
                metrics = gate_rollout_loss(
                    model,
                    _sample_dict(batch),
                    material_id,
                    long_weight=long_weight,
                    reg_weight=reg_weight,
                    accelerator=accelerator,
                    backward=False,
                    noise_seed=noise_seed,
                )
            for key in totals:
                totals[key] += float(metrics[key])
            count += 1
    if count == 0:
        raise ValueError(f"validation loader for material {material_id} is empty")
    averages = {key: value / count for key, value in totals.items()}
    averages["score"] = averages["full_mse"] + long_weight * averages["long_mse"]
    return averages


def calibrate_material_rows(
    model,
    train_loaders,
    val_loaders,
    max_updates=200,
    validation_interval=25,
    patience=3,
    learning_rate=3.0e-3,
    long_weight=0.5,
    reg_weight=1.0e-3,
    accelerator=None,
    noise_seed=0,
):
    if max_updates < validation_interval:
        raise ValueError("max_updates must include at least one validation interval")
    model.eval()
    gate = freeze_for_gate_training(model)
    tracker = MaterialBestRowTracker(
        num_materials=3,
        num_stages=4,
        validation_interval=validation_interval,
        patience=patience,
    )
    history = []

    for material_id in range(3):
        if material_id not in train_loaders or material_id not in val_loaders:
            raise ValueError(f"missing train/val loader for material {material_id}")
        identity_metrics = _validation_metrics(
            model,
            val_loaders[material_id],
            material_id,
            long_weight,
            reg_weight,
            accelerator,
            noise_seed,
        )
        tracker.record_identity(
            material_id,
            identity_metrics["score"],
            gate[material_id],
        )
        history.append(
            {
                "material": material_id,
                "update": 0,
                "split": "val_identity",
                **identity_metrics,
            }
        )

        row_mask = torch.zeros_like(gate)
        row_mask[material_id] = 1.0
        hook = gate.register_hook(lambda gradient, mask=row_mask: gradient * mask)
        optimizer = torch.optim.Adam([gate], lr=learning_rate, weight_decay=0.0)
        loader = train_loaders[material_id]
        iterator = iter(loader)

        try:
            for update in range(1, max_updates + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    try:
                        batch = next(iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            f"training loader for material {material_id} is empty"
                        ) from exc

                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(accelerator):
                    train_metrics = gate_rollout_loss(
                        model,
                        _sample_dict(batch),
                        material_id,
                        long_weight=long_weight,
                        reg_weight=reg_weight,
                        accelerator=accelerator,
                        backward=True,
                        noise_seed=noise_seed,
                    )
                optimizer.step()
                history.append(
                    {
                        "material": material_id,
                        "update": update,
                        "split": "train",
                        "full_mse": float(train_metrics["full_mse"]),
                        "long_mse": float(train_metrics["long_mse"]),
                        "regularizer": float(train_metrics["regularizer"]),
                        "score": float(train_metrics["full_mse"])
                        + long_weight * float(train_metrics["long_mse"]),
                    }
                )

                if tracker.should_validate(update):
                    val_metrics = _validation_metrics(
                        model,
                        val_loaders[material_id],
                        material_id,
                        long_weight,
                        reg_weight,
                        accelerator,
                        noise_seed,
                    )
                    history.append(
                        {
                            "material": material_id,
                            "update": update,
                            "split": "val",
                            **val_metrics,
                        }
                    )
                    should_stop = tracker.observe(
                        material_id,
                        update,
                        val_metrics["score"],
                        gate[material_id],
                    )
                    if should_stop:
                        break
        finally:
            hook.remove()

        if tracker.best_update[material_id] is None:
            raise RuntimeError(f"material {material_id} was never validated")
        with torch.no_grad():
            gate[material_id].copy_(tracker.best_rows[material_id].to(gate))

    return {
        "history": history,
        "identity_scores": list(tracker.identity_scores),
        "best_updates": list(tracker.best_update),
        "best_scores": list(tracker.best_scores),
        "best_rows": tracker.best_rows.clone(),
    }
