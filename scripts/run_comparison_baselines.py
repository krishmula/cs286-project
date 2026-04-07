from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
DROPOUT = 0.2


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    channel_mode: str
    label_fraction: float
    description: str


EXPERIMENT_SPECS: Dict[str, ExperimentSpec] = {
    "phone_full": ExperimentSpec(
        name="phone_full",
        channel_mode="phone",
        label_fraction=1.0,
        description="Phone-only supervised baseline using all labeled train windows.",
    ),
    "watch_full": ExperimentSpec(
        name="watch_full",
        channel_mode="watch",
        label_fraction=1.0,
        description="Watch-only supervised baseline using all labeled train windows.",
    ),
    "fusion_10pct": ExperimentSpec(
        name="fusion_10pct",
        channel_mode="fusion",
        label_fraction=0.10,
        description="Fusion supervised baseline trained on a 10% subject-balanced labeled subset.",
    ),
}

CHANNEL_SLICES = {
    "fusion": slice(0, 12),
    "phone": slice(0, 6),
    "watch": slice(6, 12),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing artifacts/windows and artifacts/baseline.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=sorted(EXPERIMENT_SPECS.keys()),
        default=["phone_full", "watch_full", "fusion_10pct"],
        help="Experiments to run.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_metadata_rows(window_root: Path, split: str) -> List[dict]:
    metadata_path = window_root / split / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata for split {split}: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def read_window_payloads(window_root: Path, split: str) -> Dict[int, List[List[float]]]:
    split_dir = window_root / split
    chunk_paths = sorted(split_dir.glob("data_chunk_*.pkl"))
    if not chunk_paths:
        raise FileNotFoundError(f"No cached chunks found for split {split}: {split_dir}")

    import pickle

    payloads: Dict[int, List[List[float]]] = {}
    for chunk_path in chunk_paths:
        with chunk_path.open("rb") as handle:
            chunk_records = pickle.load(handle)
        for record in chunk_records:
            payloads[int(record["window_id"])] = record["x_fusion"]
    return payloads


def select_labeled_subset(rows: List[dict], label_fraction: float, seed: int) -> List[dict]:
    if label_fraction >= 1.0:
        return rows
    if label_fraction <= 0.0:
        raise ValueError("label_fraction must be positive")

    grouped: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for row in rows:
        key = (int(row["subject_id"]), int(row["label_idx"]))
        grouped[key].append(row)

    selected: List[dict] = []
    for group_index, key in enumerate(sorted(grouped)):
        group_rows = grouped[key]
        group_rng = random.Random(seed + (group_index * 9973))
        quota = max(1, math.ceil(len(group_rows) * label_fraction))
        if quota >= len(group_rows):
            selected.extend(group_rows)
            continue
        selected.extend(group_rng.sample(group_rows, quota))

    selected.sort(key=lambda row: int(row["window_id"]))
    return selected


def build_arrays(
    rows: List[dict],
    payloads: Dict[int, List[List[float]]],
    channel_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    channel_slice = CHANNEL_SLICES[channel_mode]
    x_list: List[List[List[float]]] = []
    y_list: List[int] = []
    subject_list: List[int] = []
    for row in rows:
        window_id = int(row["window_id"])
        x_list.append(payloads[window_id][channel_slice])
        y_list.append(int(row["label_idx"]))
        subject_list.append(int(row["subject_id"]))

    x = np.asarray(x_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    subjects = np.asarray(subject_list, dtype=np.int64)
    return x, y, subjects, rows


def to_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TimeSeriesEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).squeeze(-1)


class SupervisedHARModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.encoder = TimeSeriesEncoder(in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def confusion_matrix_from_predictions(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def macro_f1_from_confusion(matrix: np.ndarray, index_to_label: Dict[int, str]) -> Tuple[float, Dict[str, float]]:
    scores: List[float] = []
    per_class_f1: Dict[str, float] = {}
    for class_index in range(matrix.shape[0]):
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - tp)
        fn = float(matrix[class_index, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_f1[index_to_label[class_index]] = f1
        scores.append(f1)
    return float(sum(scores) / len(scores)), per_class_f1


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    subjects: np.ndarray,
    num_classes: int,
    index_to_label: Dict[int, str],
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            total_loss += float(loss.item()) * len(x_batch)
            predictions = logits.argmax(dim=1)
            all_true.extend(y_batch.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())

    matrix = confusion_matrix_from_predictions(all_true, all_pred, num_classes=num_classes)
    macro_f1, per_class_f1 = macro_f1_from_confusion(matrix, index_to_label)
    accuracy = float(matrix.diagonal().sum() / max(1, matrix.sum()))

    per_subject_counts: Dict[int, List[int]] = defaultdict(lambda: [0, 0])
    for subject_id, truth, pred in zip(subjects.tolist(), all_true, all_pred):
        per_subject_counts[subject_id][0] += int(truth == pred)
        per_subject_counts[subject_id][1] += 1

    per_subject_accuracy = {
        int(subject_id): correct / total
        for subject_id, (correct, total) in sorted(per_subject_counts.items())
    }

    return {
        "loss": total_loss / len(all_true),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": matrix,
        "per_subject_accuracy": per_subject_accuracy,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += float(loss.item()) * len(x_batch)
        total_count += len(x_batch)
    return total_loss / max(1, total_count)


def plot_confusion_matrix(
    matrix: np.ndarray,
    index_to_label: Dict[int, str],
    title: str,
    output_path: Path,
) -> None:
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


def metric_token(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def write_subject_accuracy(path: Path, per_subject_accuracy: Dict[int, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "accuracy"])
        writer.writeheader()
        for subject_id, accuracy in per_subject_accuracy.items():
            writer.writerow({"subject_id": subject_id, "accuracy": accuracy})


def run_experiment(
    spec: ExperimentSpec,
    manifest: dict,
    baseline_root: Path,
    window_root: Path,
    device: torch.device,
) -> dict:
    label_to_index = manifest["label_to_index"]
    index_to_label = {index: label for label, index in label_to_index.items()}

    train_rows = read_metadata_rows(window_root, "train")
    val_rows = read_metadata_rows(window_root, "val")
    test_rows = read_metadata_rows(window_root, "test")

    selected_train_rows = select_labeled_subset(train_rows, spec.label_fraction, seed=SEED)
    payloads_by_split = {
        split: read_window_payloads(window_root, split)
        for split in ("train", "val", "test")
    }

    x_train, y_train, subjects_train, _ = build_arrays(selected_train_rows, payloads_by_split["train"], spec.channel_mode)
    x_val, y_val, subjects_val, _ = build_arrays(val_rows, payloads_by_split["val"], spec.channel_mode)
    x_test, y_test, subjects_test, _ = build_arrays(test_rows, payloads_by_split["test"], spec.channel_mode)

    train_loader = to_loader(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = to_loader(x_val, y_val, BATCH_SIZE, shuffle=False)
    test_loader = to_loader(x_test, y_test, BATCH_SIZE, shuffle=False)

    seed_everything(SEED)
    model = SupervisedHARModel(
        in_channels=x_train.shape[1],
        num_classes=len(label_to_index),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_macro_f1 = -1.0
    best_state: Optional[dict] = None
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate_model(
            model,
            val_loader,
            loss_fn,
            subjects_val,
            len(label_to_index),
            index_to_label,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )
        print(
            f"[{spec.name}] epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"[{spec.name}] early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"Training completed without a checkpoint for {spec.name}")

    model.load_state_dict(best_state)
    train_metrics = evaluate_model(
        model,
        train_loader,
        loss_fn,
        subjects_train,
        len(label_to_index),
        index_to_label,
        device,
    )
    val_metrics = evaluate_model(
        model,
        val_loader,
        loss_fn,
        subjects_val,
        len(label_to_index),
        index_to_label,
        device,
    )
    test_metrics = evaluate_model(
        model,
        test_loader,
        loss_fn,
        subjects_test,
        len(label_to_index),
        index_to_label,
        device,
    )

    stem = (
        f"supervised_{spec.channel_mode}_baseline_{spec.name}_"
        f"acc{metric_token(test_metrics['accuracy'])}_macrof1{metric_token(test_metrics['macro_f1'])}"
    )
    checkpoint_path = baseline_root / f"{stem}_checkpoint.pt"
    metrics_path = baseline_root / f"{stem}_metrics.json"
    confusion_path = baseline_root / f"{stem}_confusion_matrix.png"
    per_subject_path = baseline_root / f"{stem}_per_subject_accuracy.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "label_to_index": label_to_index,
            "channel_mode": spec.channel_mode,
            "label_fraction": spec.label_fraction,
            "description": spec.description,
            "manifest_path": str(window_root / "manifest.json"),
            "experiment_name": spec.name,
        },
        checkpoint_path,
    )

    metrics_payload = {
        "experiment_name": spec.name,
        "description": spec.description,
        "best_epoch": best_epoch,
        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "epochs_requested": MAX_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "channel_mode": spec.channel_mode,
            "input_channels": int(x_train.shape[1]),
            "label_fraction": spec.label_fraction,
            "train_windows_available": len(train_rows),
            "train_windows_used": len(selected_train_rows),
            "val_windows": len(val_rows),
            "test_windows": len(test_rows),
        },
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

    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
    write_subject_accuracy(per_subject_path, test_metrics["per_subject_accuracy"])
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        index_to_label,
        title=f"Supervised {spec.channel_mode.title()} Baseline ({spec.name})",
        output_path=confusion_path,
    )

    result = {
        "experiment_name": spec.name,
        "channel_mode": spec.channel_mode,
        "label_fraction": spec.label_fraction,
        "best_epoch": best_epoch,
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "train_windows_used": len(selected_train_rows),
        "train_windows_available": len(train_rows),
        "metrics_path": str(metrics_path),
    }
    return result


def write_summary(summary_path: Path, results: Iterable[dict]) -> None:
    results_list = list(results)
    summary_payload = {
        "seed": SEED,
        "results": results_list,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = summary_path.with_suffix(".csv")
    if results_list:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results_list[0].keys()))
            writer.writeheader()
            writer.writerows(results_list)


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    project_root = args.project_root.resolve()
    window_root = project_root / "artifacts" / "windows"
    baseline_root = project_root / "artifacts" / "baseline"
    baseline_root.mkdir(parents=True, exist_ok=True)

    manifest_path = window_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Window cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    device = detect_device()
    print(f"Project root: {project_root}")
    print(f"Training device: {device}")
    print(f"Experiments: {args.experiments}")

    results = []
    for experiment_name in args.experiments:
        spec = EXPERIMENT_SPECS[experiment_name]
        result = run_experiment(spec, manifest, baseline_root, window_root, device)
        results.append(result)
        print(
            f"[{spec.name}] done: test_acc={result['test_accuracy']:.4f} "
            f"test_macro_f1={result['test_macro_f1']:.4f}"
        )

    summary_path = baseline_root / "comparison_summary.json"
    write_summary(summary_path, results)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
