from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import DEFAULTS
from .data import CachedSplitData, WindowRepository, limit_samples, select_labeled_subset
from .models import SupervisedHARModel
from .runtime import detect_device, resolve_project_paths, seed_everything
from .training import EarlyStopper, checkpoint_metadata, compute_classification_metrics, save_checkpoint


def metric_token(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def load_pretrained_encoder(
    model: nn.Module,
    checkpoint_path: Path,
    model_type: str,
    device: torch.device,
) -> nn.Module:
    """Load pretrained encoder weights into model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_state = checkpoint.get("encoder_state_dict")
    
    if encoder_state is None:
        raise ValueError(f"Checkpoint {checkpoint_path} missing 'encoder_state_dict'")
    
    model.load_state_dict(encoder_state, strict=False)
    
    loaded_keys = list(encoder_state.keys())
    print(
        f"Loaded {len(loaded_keys)} encoder parameters from {checkpoint_path} "
        f"(source: {checkpoint.get('source_dataset', 'unknown')}, "
        f"classes={checkpoint.get('source_classes', '?')}, "
        f"channels={checkpoint.get('source_channels', '?')})"
    )
    return model


def apply_freeze_strategy(model: nn.Module, strategy: str, model_type: str) -> str:
    """Apply layer freezing strategy and return description."""
    if strategy == "none":
        for p in model.parameters():
            p.requires_grad = True
        return "No freezing — all parameters trainable"
    
    elif strategy == "all":
        if model_type == "cnn":
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif model_type == "lstm":
            for p in model.lstm.parameters():
                p.requires_grad = False
        return "Frozen entire encoder, training classifier only"
    
    elif strategy == "first_two":
        if model_type == "cnn":
            for i, layer in enumerate(model.encoder.features):
                freeze = i < 8
                for p in layer.parameters():
                    p.requires_grad = not freeze
            return "Frozen first two conv blocks, training conv3 + classifier"
        elif model_type == "lstm":
            for name, p in model.lstm.named_parameters():
                p.requires_grad = "l1" in name
            return "Frozen LSTM layer 1, training layer 2 + classifier"
    
    elif strategy == "progressive":
        if model_type == "cnn":
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif model_type == "lstm":
            for p in model.lstm.parameters():
                p.requires_grad = False
        return "Progressive: encoder frozen, will unfreeze at specified epoch"
    
    else:
        raise ValueError(f"Unknown freeze strategy: {strategy}")


def unfreeze_all_encoder(model: nn.Module, model_type: str) -> None:
    """Unfreeze all encoder layers."""
    if model_type == "cnn":
        for p in model.encoder.parameters():
            p.requires_grad = True
    elif model_type == "lstm":
        for p in model.lstm.parameters():
            p.requires_grad = True
    print(f"Unfrozen all {model_type.upper()} encoder layers")


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_count, total_count)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Project root containing artifacts/windows.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoint, metrics, confusion matrix, and per-subject outputs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override the experiment name written into artifacts.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional human-readable description for this supervised run.",
    )
    parser.add_argument("--channel-mode", choices=("fusion", "phone", "watch"), default="fusion")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULTS.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.max_epochs)
    parser.add_argument("--patience", type=int, default=DEFAULTS.early_stopping_patience)
    parser.add_argument("--grad-clip", type=float, default=DEFAULTS.grad_clip)
    parser.add_argument("--dropout", type=float, default=DEFAULTS.dropout)
    parser.add_argument("--num-workers", type=int, default=DEFAULTS.num_workers)
    parser.add_argument("--device", type=str, default=None, help="Force a device, e.g. cpu, cuda, or mps.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Overwrite output-dir/latest_checkpoint.pt every N epochs. Set to 0 to disable periodic saves.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume optimizer/model state from a previous latest checkpoint.",
    )
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    parser.add_argument(
        "--model-type",
        choices=("cnn", "lstm"),
        default="cnn",
        help="Model architecture type.",
    )
    # Transfer learning arguments
    parser.add_argument(
        "--pretrained-encoder-path",
        type=Path,
        default=None,
        help="Path to pretrained encoder checkpoint (e.g., from UCI HAR pretraining).",
    )
    parser.add_argument(
        "--freeze-strategy",
        choices=("none", "all", "first_two", "progressive"),
        default="none",
        help="Layer freezing strategy for transfer learning. 'none'=train all, 'all'=freeze encoder, "
             "'first_two'=freeze early layers, 'progressive'=freeze then unfreeze at --unfreeze-at-epoch.",
    )
    parser.add_argument(
        "--unfreeze-at-epoch",
        type=int,
        default=10,
        help="Epoch at which to unfreeze encoder for progressive strategy.",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning (used when --pretrained-encoder-path is set).",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class SplitArrays:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


def build_split_arrays(split_data: CachedSplitData, channel_mode: str) -> SplitArrays:
    if channel_mode == "fusion":
        x = np.stack([sample.x_fusion for sample in split_data.samples]).astype(np.float32)
    elif channel_mode == "phone":
        x = np.stack([sample.x_phone for sample in split_data.samples]).astype(np.float32)
    elif channel_mode == "watch":
        x = np.stack([sample.x_watch for sample in split_data.samples]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported channel_mode: {channel_mode}")

    y = np.asarray([sample.label_idx for sample in split_data.samples], dtype=np.int64)
    subjects = np.asarray([sample.subject_id for sample in split_data.samples], dtype=np.int64)
    return SplitArrays(x=x, y=y, subjects=subjects)


def to_loader(split_arrays: SplitArrays, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(split_arrays.x), torch.from_numpy(split_arrays.y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * len(x_batch)
        total_count += len(x_batch)
    return total_loss / max(1, total_count)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    subjects: np.ndarray,
    loss_fn: nn.Module,
    index_to_label: dict[int, str],
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
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


def plot_confusion_matrix(matrix: np.ndarray, index_to_label: dict[int, str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ticks = np.arange(len(index_to_label))
    labels = [index_to_label[index] for index in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_subject_accuracy(path: Path, per_subject_accuracy: dict[int, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "accuracy"])
        writer.writeheader()
        for subject_id, accuracy in per_subject_accuracy.items():
            writer.writerow({"subject_id": subject_id, "accuracy": accuracy})


def default_experiment_name(
    channel_mode: str,
    label_fraction: float,
    model_type: str = "cnn",
    freeze_strategy: str = "none",
    is_transfer: bool = False,
) -> str:
    if label_fraction >= 1.0:
        fraction_token = "full"
    else:
        fraction_token = f"{int(round(label_fraction * 100)):02d}pct"
    suffix = f"_{model_type}" if model_type != "cnn" else ""
    
    if is_transfer and freeze_strategy != "none":
        transfer_suffix = f"_transfer_{freeze_strategy}"
    else:
        transfer_suffix = ""
    
    return f"{channel_mode}_{fraction_token}{suffix}{transfer_suffix}"


def description_for_run(channel_mode: str, label_fraction: float) -> str:
    if channel_mode == "watch" and label_fraction >= 1.0:
        return "Primary watch-only supervised baseline using all labeled train windows."
    if channel_mode == "watch":
        return f"Watch-only supervised baseline trained on a {label_fraction:.0%} subject-balanced labeled subset."
    if channel_mode == "fusion":
        return f"Fusion supervised baseline trained on a {label_fraction:.0%} subject-balanced labeled subset."
    return "Phone-only supervised baseline using all labeled train windows."


def trim_split(split_data: CachedSplitData, max_samples: int | None) -> CachedSplitData:
    trimmed_samples = limit_samples(split_data.samples, max_samples)
    return CachedSplitData(split=split_data.split, samples=tuple(trimmed_samples))


def save_latest_checkpoint(
    latest_checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopper: EarlyStopper,
    best_state: dict[str, torch.Tensor] | None,
    history: list[dict[str, float | int]],
    label_to_index: dict[str, int],
    args: argparse.Namespace,
    experiment_name: str,
    description: str,
    paths,
    epoch: int,
) -> None:
    save_checkpoint(
        latest_checkpoint_path,
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_model_state_dict": best_state,
            "early_stopper_state_dict": early_stopper.state_dict(),
            "history": history,
            "label_to_index": label_to_index,
            "channel_mode": args.channel_mode,
            "label_fraction": args.label_fraction,
            "metadata": checkpoint_metadata(
                experiment_name=experiment_name,
                stage="supervised",
                config={
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "epochs_requested": args.epochs,
                    "early_stopping_patience": args.patience,
                    "channel_mode": args.channel_mode,
                    "label_fraction": args.label_fraction,
                    "seed": args.seed,
                    "save_every": args.save_every,
                },
                manifest_path=str(paths.windows_root / "manifest.json"),
                extra={
                    "description": description,
                    "checkpoint_role": "latest",
                    "checkpoint_epoch": epoch,
                },
            ),
        },
    )


def main() -> None:
    args = parse_args()
    is_transfer = args.pretrained_encoder_path is not None
    paths = resolve_project_paths(args.project_root)
    output_dir = (args.output_dir or paths.baseline_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repository = WindowRepository(paths.windows_root)
    manifest = repository.manifest
    label_to_index = manifest["label_to_index"]
    index_to_label = {index: label for label, index in label_to_index.items()}

    experiment_name = args.experiment_name or default_experiment_name(
        args.channel_mode,
        args.label_fraction,
        args.model_type,
        args.freeze_strategy,
        is_transfer,
    )
    description = args.description or description_for_run(args.channel_mode, args.label_fraction)
    latest_checkpoint_path = output_dir / "latest_checkpoint.pt"

    train_split = repository.load_split("train")
    val_split = repository.load_split("val")
    test_split = repository.load_split("test")

    selected_train_samples = select_labeled_subset(train_split.samples, args.label_fraction, args.seed)
    train_split = CachedSplitData(split="train", samples=tuple(selected_train_samples))

    train_split = trim_split(train_split, args.max_train_windows)
    val_split = trim_split(val_split, args.max_val_windows)
    test_split = trim_split(test_split, args.max_test_windows)

    train_arrays = build_split_arrays(train_split, args.channel_mode)
    val_arrays = build_split_arrays(val_split, args.channel_mode)
    test_arrays = build_split_arrays(test_split, args.channel_mode)

    seed_everything(args.seed)
    device = torch.device(args.device or detect_device())

    model = SupervisedHARModel(
        in_channels=train_arrays.x.shape[1],
        num_classes=len(label_to_index),
        dropout=args.dropout,
    ).to(device)

    # Transfer learning: load pretrained encoder if specified
    if is_transfer:
        model = load_pretrained_encoder(
            model=model,
            checkpoint_path=args.pretrained_encoder_path,
            model_type=args.model_type,
            device=device,
        )
        freeze_desc = apply_freeze_strategy(model, args.freeze_strategy, args.model_type)
        trainable, total = count_trainable_params(model)
        print(
            f"[{experiment_name}] Transfer learning enabled. {freeze_desc}"
        )
        print(
            f"[{experiment_name}] Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)"
        )

    loss_fn = nn.CrossEntropyLoss()

    # Use fine-tune LR for transfer learning, standard LR for from-scratch
    current_lr = args.fine_tune_lr if is_transfer else args.lr

    # Only pass trainable parameters to optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=current_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    train_loader = to_loader(train_arrays, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = to_loader(val_arrays, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = to_loader(test_arrays, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    early_stopper = EarlyStopper(patience=args.patience, mode="max")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    start_epoch = 1

    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        early_stopper = EarlyStopper.from_state_dict(checkpoint["early_stopper_state_dict"])
        best_state = checkpoint.get("best_model_state_dict")
        history = [dict(entry) for entry in checkpoint.get("history", [])]
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[{experiment_name}] resumed from {args.resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        # Progressive unfreeze check
        if (
            is_transfer
            and args.freeze_strategy == "progressive"
            and epoch == args.unfreeze_at_epoch
        ):
            unfreeze_all_encoder(model, args.model_type)
            # Re-create optimizer to include newly unfrozen parameters
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            new_lr = args.fine_tune_lr * 0.1  # Even lower LR after unfreeze
            optimizer = torch.optim.AdamW(trainable_params, lr=new_lr, weight_decay=args.weight_decay)
            print(
                f"[{experiment_name}] Progressive unfreeze at epoch {epoch}. "
                f"New LR: {new_lr:.1e}"
            )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            subjects=val_arrays.subjects,
            loss_fn=loss_fn,
            index_to_label=index_to_label,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )
        print(
            f"[{experiment_name}] epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        improved = early_stopper.update(float(val_metrics["macro_f1"]), epoch=epoch)
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            save_latest_checkpoint(
                latest_checkpoint_path,
                model=model,
                optimizer=optimizer,
                early_stopper=early_stopper,
                best_state=best_state,
                history=history,
                label_to_index=label_to_index,
                args=args,
                experiment_name=experiment_name,
                description=description,
                paths=paths,
                epoch=epoch,
            )
        if early_stopper.should_stop:
            print(f"[{experiment_name}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training completed without producing a best model state.")
    if not latest_checkpoint_path.exists():
        save_latest_checkpoint(
            latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            early_stopper=early_stopper,
            best_state=best_state,
            history=history,
            label_to_index=label_to_index,
            args=args,
            experiment_name=experiment_name,
            description=description,
            paths=paths,
            epoch=int(history[-1]["epoch"]),
        )

    model.load_state_dict(best_state)

    train_metrics = evaluate_model(
        model=model,
        loader=train_loader,
        subjects=train_arrays.subjects,
        loss_fn=loss_fn,
        index_to_label=index_to_label,
        device=device,
    )
    val_metrics = evaluate_model(
        model=model,
        loader=val_loader,
        subjects=val_arrays.subjects,
        loss_fn=loss_fn,
        index_to_label=index_to_label,
        device=device,
    )
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        subjects=test_arrays.subjects,
        loss_fn=loss_fn,
        index_to_label=index_to_label,
        device=device,
    )

    stem = (
        f"supervised_{args.channel_mode}_baseline_{experiment_name}_"
        f"acc{metric_token(float(test_metrics['accuracy']))}_macrof1{metric_token(float(test_metrics['macro_f1']))}"
    )
    checkpoint_path = output_dir / f"{stem}_checkpoint.pt"
    metrics_path = output_dir / f"{stem}_metrics.json"
    confusion_path = output_dir / f"{stem}_confusion_matrix.png"
    per_subject_path = output_dir / f"{stem}_per_subject_accuracy.csv"

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "label_to_index": label_to_index,
        "channel_mode": args.channel_mode,
        "label_fraction": args.label_fraction,
        "best_epoch": early_stopper.best_epoch,
        "metadata": checkpoint_metadata(
            experiment_name=experiment_name,
            stage="supervised",
            config={
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "epochs_requested": args.epochs,
                "early_stopping_patience": args.patience,
                "channel_mode": args.channel_mode,
                "input_channels": int(train_arrays.x.shape[1]),
                "label_fraction": args.label_fraction,
                "train_windows_available": len(repository.load_split("train")),
                "train_windows_used": len(train_split),
                "val_windows": len(val_split),
                "test_windows": len(test_split),
                "seed": args.seed,
                "transfer_learning": {
                    "enabled": is_transfer,
                    "pretrained_encoder_path": str(args.pretrained_encoder_path) if is_transfer else None,
                    "freeze_strategy": args.freeze_strategy if is_transfer else None,
                    "unfreeze_at_epoch": args.unfreeze_at_epoch if is_transfer else None,
                    "fine_tune_lr": args.fine_tune_lr if is_transfer else None,
                },
            },
            manifest_path=str(paths.windows_root / "manifest.json"),
            extra={"description": description},
        ),
    }
    save_checkpoint(checkpoint_path, checkpoint_payload)

    metrics_payload = {
        "experiment_name": experiment_name,
        "description": description,
        "best_epoch": early_stopper.best_epoch,
        "channel_mode": args.channel_mode,
        "label_fraction": args.label_fraction,
        "config": checkpoint_payload["metadata"]["config"],
        "train": {
            "loss": train_metrics["loss"],
            "accuracy": train_metrics["accuracy"],
            "macro_f1": train_metrics["macro_f1"],
        },
        "val": {
            "loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
        },
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "per_class_f1": test_metrics["per_class_f1"],
            "per_subject_accuracy": test_metrics["per_subject_accuracy"],
        },
        "history": history,
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "metrics_path": str(metrics_path),
            "confusion_path": str(confusion_path),
            "per_subject_accuracy_path": str(per_subject_path),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    plot_confusion_matrix(test_metrics["confusion_matrix"], index_to_label, experiment_name, confusion_path)
    write_subject_accuracy(per_subject_path, test_metrics["per_subject_accuracy"])

    print(json.dumps({"metrics_path": str(metrics_path), "checkpoint_path": str(checkpoint_path)}, indent=2))


if __name__ == "__main__":
    main()
