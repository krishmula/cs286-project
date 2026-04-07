from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import DEFAULTS
from .data import (
    CachedSplitData,
    ContrastiveAugmentationConfig,
    WindowRepository,
    apply_default_augmentations,
    limit_samples,
)
from .models import PhoneWatchContrastiveModel, symmetric_info_nce_loss
from .runtime import detect_device, resolve_project_paths, seed_everything
from .training import EarlyStopper, checkpoint_metadata, save_checkpoint


@dataclass(frozen=True)
class ContrastiveArrays:
    x_phone: np.ndarray
    x_watch: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Project root containing artifacts/windows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for contrastive checkpoints and metrics.",
    )
    parser.add_argument("--experiment-name", type=str, default="phone_watch_contrastive")
    parser.add_argument(
        "--description",
        type=str,
        default="Phone-watch contrastive pretraining using paired cached train windows.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULTS.contrastive_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.max_epochs)
    parser.add_argument("--patience", type=int, default=DEFAULTS.early_stopping_patience)
    parser.add_argument("--grad-clip", type=float, default=DEFAULTS.grad_clip)
    parser.add_argument("--num-workers", type=int, default=DEFAULTS.num_workers)
    parser.add_argument("--device", type=str, default=None, help="Force a device, e.g. cpu, cuda, or mps.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--projection-hidden-dim", type=int, default=128)
    parser.add_argument("--projection-out-dim", type=int, default=64)
    parser.add_argument("--jitter-std", type=float, default=0.01)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--mask-length", type=int, default=6)
    parser.add_argument("--disable-augmentations", action="store_true")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    return parser.parse_args()


def trim_split(split_data: CachedSplitData, max_samples: int | None) -> CachedSplitData:
    trimmed_samples = limit_samples(split_data.samples, max_samples)
    return CachedSplitData(split=split_data.split, samples=tuple(trimmed_samples))


def build_contrastive_arrays(split_data: CachedSplitData) -> ContrastiveArrays:
    return ContrastiveArrays(
        x_phone=np.stack([sample.x_phone for sample in split_data.samples]).astype(np.float32),
        x_watch=np.stack([sample.x_watch for sample in split_data.samples]).astype(np.float32),
        labels=np.asarray([sample.label_idx for sample in split_data.samples], dtype=np.int64),
        subjects=np.asarray([sample.subject_id for sample in split_data.samples], dtype=np.int64),
    )


def to_loader(arrays: ContrastiveArrays, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays.x_phone),
        torch.from_numpy(arrays.x_watch),
        torch.from_numpy(arrays.labels),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_one_epoch(
    model: PhoneWatchContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    augmentation_config: ContrastiveAugmentationConfig,
    temperature: float,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for x_phone, x_watch, _labels in loader:
        x_phone = x_phone.to(device)
        x_watch = x_watch.to(device)
        x_phone_aug = apply_default_augmentations(x_phone, augmentation_config)
        x_watch_aug = apply_default_augmentations(x_watch, augmentation_config)

        optimizer.zero_grad()
        outputs = model(x_phone_aug, x_watch_aug)
        loss_result = symmetric_info_nce_loss(outputs.z_phone, outputs.z_watch, temperature=temperature)
        loss_result.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss_result.loss.item()) * len(x_phone)
        total_count += len(x_phone)

    return total_loss / max(1, total_count)


def evaluate_epoch(
    model: PhoneWatchContrastiveModel,
    loader: DataLoader,
    temperature: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_phone_loss = 0.0
    total_watch_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x_phone, x_watch, _labels in loader:
            x_phone = x_phone.to(device)
            x_watch = x_watch.to(device)
            outputs = model(x_phone, x_watch)
            loss_result = symmetric_info_nce_loss(outputs.z_phone, outputs.z_watch, temperature=temperature)
            batch_size = len(x_phone)
            total_loss += float(loss_result.loss.item()) * batch_size
            total_phone_loss += float(loss_result.phone_to_watch_loss.item()) * batch_size
            total_watch_loss += float(loss_result.watch_to_phone_loss.item()) * batch_size
            total_count += batch_size

    return {
        "loss": total_loss / max(1, total_count),
        "phone_to_watch_loss": total_phone_loss / max(1, total_count),
        "watch_to_phone_loss": total_watch_loss / max(1, total_count),
    }


def main() -> None:
    args = parse_args()
    paths = resolve_project_paths(args.project_root)
    output_dir = (args.output_dir or (paths.artifacts_root / "contrastive")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repository = WindowRepository(paths.windows_root)
    manifest = repository.manifest
    label_to_index = manifest["label_to_index"]

    train_split = trim_split(repository.load_split("train"), args.max_train_windows)
    val_split = trim_split(repository.load_split("val"), args.max_val_windows)
    train_arrays = build_contrastive_arrays(train_split)
    val_arrays = build_contrastive_arrays(val_split)

    train_loader = to_loader(train_arrays, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = to_loader(val_arrays, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    seed_everything(args.seed)
    device = torch.device(args.device or detect_device())
    model = PhoneWatchContrastiveModel(
        in_channels=train_arrays.x_phone.shape[1],
        projection_hidden_dim=args.projection_hidden_dim,
        projection_out_dim=args.projection_out_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    augmentation_config = ContrastiveAugmentationConfig(
        jitter_std=args.jitter_std,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        mask_ratio=args.mask_ratio,
        mask_length=args.mask_length,
        enabled=not args.disable_augmentations,
    )

    early_stopper = EarlyStopper(patience=args.patience, mode="min")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            augmentation_config=augmentation_config,
            temperature=args.temperature,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate_epoch(model=model, loader=val_loader, temperature=args.temperature, device=device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_phone_to_watch_loss": val_metrics["phone_to_watch_loss"],
                "val_watch_to_phone_loss": val_metrics["watch_to_phone_loss"],
            }
        )
        print(
            f"[{args.experiment_name}] epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_p2w={val_metrics['phone_to_watch_loss']:.4f} "
            f"val_w2p={val_metrics['watch_to_phone_loss']:.4f}"
        )
        improved = early_stopper.update(val_metrics["loss"], epoch=epoch)
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if early_stopper.should_stop:
            print(f"[{args.experiment_name}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Contrastive training completed without producing a best model state.")

    checkpoint_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs_requested": args.epochs,
        "early_stopping_patience": args.patience,
        "temperature": args.temperature,
        "projection_hidden_dim": args.projection_hidden_dim,
        "projection_out_dim": args.projection_out_dim,
        "jitter_std": args.jitter_std,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "mask_ratio": args.mask_ratio,
        "mask_length": args.mask_length,
        "augmentations_enabled": not args.disable_augmentations,
        "train_windows": len(train_split),
        "val_windows": len(val_split),
        "seed": args.seed,
    }

    def build_checkpoint_payload(*, checkpoint_role: str, checkpoint_epoch: int | None) -> dict[str, object]:
        return {
            "model_state_dict": model.state_dict(),
            "phone_encoder_state_dict": model.phone_encoder.state_dict(),
            "watch_encoder_state_dict": model.watch_encoder.state_dict(),
            "phone_projector_state_dict": model.phone_projector.state_dict(),
            "watch_projector_state_dict": model.watch_projector.state_dict(),
            "label_to_index": label_to_index,
            "best_epoch": early_stopper.best_epoch,
            "metadata": checkpoint_metadata(
                experiment_name=args.experiment_name,
                stage="contrastive",
                config=checkpoint_config,
                manifest_path=str(paths.windows_root / "manifest.json"),
                extra={
                    "description": args.description,
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_epoch": checkpoint_epoch,
                },
            ),
        }

    last_epoch = int(history[-1]["epoch"])
    last_train_metrics = evaluate_epoch(model=model, loader=train_loader, temperature=args.temperature, device=device)
    last_val_metrics = evaluate_epoch(model=model, loader=val_loader, temperature=args.temperature, device=device)
    last_checkpoint_path = output_dir / f"contrastive_{args.experiment_name}_last_epoch{last_epoch:02d}_checkpoint.pt"
    save_checkpoint(
        last_checkpoint_path,
        build_checkpoint_payload(checkpoint_role="last", checkpoint_epoch=last_epoch),
    )

    model.load_state_dict(best_state)
    best_train_metrics = evaluate_epoch(model=model, loader=train_loader, temperature=args.temperature, device=device)
    best_val_metrics = evaluate_epoch(model=model, loader=val_loader, temperature=args.temperature, device=device)

    stem = f"contrastive_{args.experiment_name}_valloss{best_val_metrics['loss']:.4f}".replace(".", "p")
    checkpoint_path = output_dir / f"{stem}_checkpoint.pt"
    metrics_path = output_dir / f"{stem}_metrics.json"
    checkpoint_payload = build_checkpoint_payload(checkpoint_role="best", checkpoint_epoch=early_stopper.best_epoch)
    save_checkpoint(checkpoint_path, checkpoint_payload)

    metrics_payload = {
        "experiment_name": args.experiment_name,
        "description": args.description,
        "best_epoch": early_stopper.best_epoch,
        "last_epoch": last_epoch,
        "selection_metric": "val_loss",
        "config": checkpoint_payload["metadata"]["config"],
        "train": best_train_metrics,
        "val": best_val_metrics,
        "last_train": last_train_metrics,
        "last_val": last_val_metrics,
        "history": history,
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "last_checkpoint_path": str(last_checkpoint_path),
            "metrics_path": str(metrics_path),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print(
        json.dumps(
            {
                "metrics_path": str(metrics_path),
                "checkpoint_path": str(checkpoint_path),
                "last_checkpoint_path": str(last_checkpoint_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
