from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..data import CachedSplitData
from ..models import PhoneWatchContrastiveModel
from .metrics import compute_classification_metrics


@dataclass(frozen=True)
class ProbeFeatureSet:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


class LinearProbeHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def load_contrastive_model_from_checkpoint(checkpoint: dict, device: torch.device) -> PhoneWatchContrastiveModel:
    metadata = checkpoint.get("metadata", {})
    config = metadata.get("config", {})
    model = PhoneWatchContrastiveModel(
        in_channels=6,
        projection_hidden_dim=int(config.get("projection_hidden_dim", 128)),
        projection_out_dim=int(config.get("projection_out_dim", 64)),
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Contrastive checkpoint is missing model_state_dict")
    model.load_state_dict(state_dict)
    freeze_module(model.phone_encoder)
    freeze_module(model.watch_encoder)
    freeze_module(model.phone_projector)
    freeze_module(model.watch_projector)
    return model


def extract_probe_features(
    model: PhoneWatchContrastiveModel,
    split_data: CachedSplitData,
    evaluation_mode: str,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> ProbeFeatureSet:
    if evaluation_mode not in {"pair", "phone", "watch"}:
        raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")

    phone = np.stack([sample.x_phone for sample in split_data.samples]).astype(np.float32)
    watch = np.stack([sample.x_watch for sample in split_data.samples]).astype(np.float32)
    labels = np.asarray([sample.label_idx for sample in split_data.samples], dtype=np.int64)
    subjects = np.asarray([sample.subject_id for sample in split_data.samples], dtype=np.int64)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(phone), torch.from_numpy(watch)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    feature_batches: list[np.ndarray] = []
    with torch.no_grad():
        for x_phone, x_watch in loader:
            x_phone = x_phone.to(device)
            x_watch = x_watch.to(device)
            h_phone = model.encode_phone(x_phone)
            h_watch = model.encode_watch(x_watch)
            if evaluation_mode == "pair":
                batch_features = torch.cat([h_phone, h_watch], dim=1)
            elif evaluation_mode == "phone":
                batch_features = h_phone
            else:
                batch_features = h_watch
            feature_batches.append(batch_features.cpu().numpy())

    x = np.concatenate(feature_batches, axis=0) if feature_batches else np.zeros((0, 0), dtype=np.float32)
    return ProbeFeatureSet(x=x.astype(np.float32), y=labels, subjects=subjects)


def to_loader(features: ProbeFeatureSet, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features.x), torch.from_numpy(features.y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_probe_epoch(
    head: LinearProbeHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    head.train()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = head(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(x_batch)
        total_count += len(x_batch)
    return total_loss / max(1, total_count)


def evaluate_probe(
    head: LinearProbeHead,
    loader: DataLoader,
    subjects: np.ndarray,
    loss_fn: nn.Module,
    index_to_label: dict[int, str],
    device: torch.device,
) -> dict:
    head.eval()
    total_loss = 0.0
    total_count = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = head(x_batch)
            loss = loss_fn(logits, y_batch)
            total_loss += float(loss.item()) * len(x_batch)
            total_count += len(x_batch)
            predictions = logits.argmax(dim=1)
            all_true.extend(y_batch.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())

    metrics = compute_classification_metrics(
        y_true=all_true,
        y_pred=all_pred,
        subject_ids=subjects.tolist(),
        index_to_label=index_to_label,
        loss=total_loss / max(1, total_count),
    )
    return {
        "loss": metrics.loss,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "per_class_f1": metrics.per_class_f1,
        "per_subject_accuracy": metrics.per_subject_accuracy,
        "confusion_matrix": metrics.confusion_matrix,
    }
