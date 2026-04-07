from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .config import DEFAULTS
from .data import CachedSplitData, WindowRepository, limit_samples, select_labeled_subset
from .runtime import detect_device, resolve_project_paths, seed_everything
from .training import (
    EarlyStopper,
    LinearProbeHead,
    checkpoint_metadata,
    evaluate_probe,
    extract_probe_features,
    load_checkpoint,
    load_contrastive_model_from_checkpoint,
    probe_to_loader,
    save_checkpoint,
    train_probe_epoch,
)


def metric_token(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Project root containing artifacts/windows.")
    parser.add_argument("--encoder-ckpt-path", type=Path, required=True, help="Contrastive checkpoint to probe.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for probe checkpoints, metrics, confusion matrices, and subject CSVs.",
    )
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--description", type=str, default=None)
    parser.add_argument("--evaluation-mode", choices=("pair", "phone", "watch"), default="pair")
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULTS.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.max_epochs)
    parser.add_argument("--patience", type=int, default=DEFAULTS.early_stopping_patience)
    parser.add_argument("--num-workers", type=int, default=DEFAULTS.num_workers)
    parser.add_argument("--device", type=str, default=None, help="Force a device, e.g. cpu, cuda, or mps.")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    return parser.parse_args()


def default_experiment_name(evaluation_mode: str, label_fraction: float) -> str:
    fraction_token = "100pct" if label_fraction >= 1.0 else f"{int(round(label_fraction * 100)):02d}pct"
    return f"contrastive_{evaluation_mode}_probe_{fraction_token}"


def default_description(evaluation_mode: str, label_fraction: float) -> str:
    return (
        f"Frozen linear probe over {evaluation_mode} contrastive embeddings "
        f"using {label_fraction:.0%} labeled train windows."
    )


def trim_split(split_data: CachedSplitData, max_samples: int | None) -> CachedSplitData:
    trimmed_samples = limit_samples(split_data.samples, max_samples)
    return CachedSplitData(split=split_data.split, samples=tuple(trimmed_samples))


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


def main() -> None:
    args = parse_args()
    paths = resolve_project_paths(args.project_root)
    output_dir = (args.output_dir or (paths.artifacts_root / "probes")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repository = WindowRepository(paths.windows_root)
    label_to_index = repository.manifest["label_to_index"]
    index_to_label = {index: label for label, index in label_to_index.items()}

    experiment_name = args.experiment_name or default_experiment_name(args.evaluation_mode, args.label_fraction)
    description = args.description or default_description(args.evaluation_mode, args.label_fraction)

    contrastive_checkpoint = load_checkpoint(args.encoder_ckpt_path)
    seed_everything(args.seed)
    device = torch.device(args.device or detect_device())
    encoder_model = load_contrastive_model_from_checkpoint(contrastive_checkpoint, device=device)

    train_split = repository.load_split("train")
    selected_train_samples = select_labeled_subset(train_split.samples, args.label_fraction, args.seed)
    train_split = CachedSplitData(split="train", samples=tuple(selected_train_samples))
    val_split = repository.load_split("val")
    test_split = repository.load_split("test")

    train_split = trim_split(train_split, args.max_train_windows)
    val_split = trim_split(val_split, args.max_val_windows)
    test_split = trim_split(test_split, args.max_test_windows)

    train_features = extract_probe_features(
        model=encoder_model,
        split_data=train_split,
        evaluation_mode=args.evaluation_mode,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    val_features = extract_probe_features(
        model=encoder_model,
        split_data=val_split,
        evaluation_mode=args.evaluation_mode,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )
    test_features = extract_probe_features(
        model=encoder_model,
        split_data=test_split,
        evaluation_mode=args.evaluation_mode,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
    )

    head = LinearProbeHead(in_dim=train_features.x.shape[1], num_classes=len(label_to_index)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = probe_to_loader(train_features, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = probe_to_loader(val_features, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = probe_to_loader(test_features, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    early_stopper = EarlyStopper(patience=args.patience, mode="max")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_probe_epoch(head=head, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
        val_metrics = evaluate_probe(
            head=head,
            loader=val_loader,
            subjects=val_features.subjects,
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
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        if early_stopper.should_stop:
            print(f"[{experiment_name}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Probe training completed without producing a best linear-head state.")

    head.load_state_dict(best_state)

    train_metrics = evaluate_probe(
        head=head,
        loader=train_loader,
        subjects=train_features.subjects,
        loss_fn=loss_fn,
        index_to_label=index_to_label,
        device=device,
    )
    val_metrics = evaluate_probe(
        head=head,
        loader=val_loader,
        subjects=val_features.subjects,
        loss_fn=loss_fn,
        index_to_label=index_to_label,
        device=device,
    )
    test_metrics = evaluate_probe(
        head=head,
        loader=test_loader,
        subjects=test_features.subjects,
        loss_fn=loss_fn,
        index_to_label=index_to_label,
        device=device,
    )

    stem = (
        f"probe_{experiment_name}_acc{metric_token(float(test_metrics['accuracy']))}"
        f"_macrof1{metric_token(float(test_metrics['macro_f1']))}"
    )
    checkpoint_path = output_dir / f"{stem}_checkpoint.pt"
    metrics_path = output_dir / f"{stem}_metrics.json"
    confusion_path = output_dir / f"{stem}_confusion_matrix.png"
    per_subject_path = output_dir / f"{stem}_per_subject_accuracy.csv"

    checkpoint_payload = {
        "linear_head_state_dict": head.state_dict(),
        "probe_mode": args.evaluation_mode,
        "label_fraction": args.label_fraction,
        "best_epoch": early_stopper.best_epoch,
        "encoder_checkpoint_path": str(args.encoder_ckpt_path.resolve()),
        "label_to_index": label_to_index,
        "metadata": checkpoint_metadata(
            experiment_name=experiment_name,
            stage="probe",
            config={
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "epochs_requested": args.epochs,
                "early_stopping_patience": args.patience,
                "evaluation_mode": args.evaluation_mode,
                "label_fraction": args.label_fraction,
                "input_dim": int(train_features.x.shape[1]),
                "train_windows_available": len(repository.load_split("train")),
                "train_windows_used": len(train_split),
                "val_windows": len(val_split),
                "test_windows": len(test_split),
                "seed": args.seed,
                "contrastive_experiment_name": contrastive_checkpoint.get("metadata", {}).get("experiment_name"),
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
        "probe_mode": args.evaluation_mode,
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
