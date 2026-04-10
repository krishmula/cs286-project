"""Generate watch-first report figures from supervised and probe artifacts.

This script rebuilds the deep-learning figures used in the report around the
current primary benchmark:

- supervised watch baseline (100% labels)
- contrastive watch probe (100% labels)

Outputs:
  confusion_matrix_supervised_watch.png
  confusion_matrix_contrastive_watch_probe.png
  per_class_f1_watch_vs_ssl.png
  roc_auc_watch_vs_ssl.png

Usage:
    python -m src.reporting.generate_report_figures \
      --baseline-dir artifacts/baseline \
      --probe-dir artifacts/probes \
      --output-dir figures
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from ..data import WindowRepository
from ..models import SupervisedHARModel
from ..runtime import detect_device, resolve_project_paths
from ..training import (
    LinearProbeHead,
    extract_probe_features,
    load_checkpoint,
    load_contrastive_model_from_checkpoint,
)
from .compare_runs import collect_metrics_payloads, parse_experiment_aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None, help="Project root used to resolve artifacts/windows.")
    parser.add_argument("--windows-root", type=Path, default=None, help="Optional override for the cached windows root.")
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Directory containing supervised *_metrics.json files.")
    parser.add_argument("--probe-dir", type=Path, required=True, help="Directory containing probe *_metrics.json files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where the report figures will be written.")
    parser.add_argument(
        "--experiment-alias",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Rename an experiment before lookup. Useful for reusing older watch artifacts.",
    )
    parser.add_argument("--watch-experiment", default="watch_full")
    parser.add_argument("--watch-probe-experiment", default="contrastive_watch_probe_100pct")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None, help="Force a device such as cpu, cuda, or mps.")
    return parser.parse_args()


def _resolve_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    paths = resolve_project_paths(args.project_root)
    windows_root = (args.windows_root or paths.windows_root).resolve()
    output_dir = (args.output_dir or (paths.project_root / "figures")).resolve()
    return windows_root, output_dir


def _require_payload(payloads: Mapping[str, dict], experiment_name: str) -> dict:
    try:
        return payloads[experiment_name]
    except KeyError as exc:
        raise FileNotFoundError(f"Missing metrics for experiment {experiment_name!r}.") from exc


def _artifact_path(payload_entry: Mapping[str, dict], key: str) -> Path:
    payload = payload_entry["payload"]
    artifacts = payload.get("artifacts", {})
    raw_path = artifacts.get(key)
    if not raw_path:
        raise FileNotFoundError(
            f"Metrics payload for {payload.get('experiment_name')!r} does not contain artifacts[{key!r}]."
        )
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {path}")
    return path


def _copy_confusion_matrix(payload_entry: Mapping[str, dict], destination: Path) -> Path:
    source = _artifact_path(payload_entry, "confusion_path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _softmax_inference(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(x.astype(np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def _linear_probe_softmax(
    head: LinearProbeHead,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(features.astype(np.float32))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = head(batch.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def _macro_ovr_roc(
    y_true_bin: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray, float]:
    all_fpr = np.linspace(0.0, 1.0, n_points)
    mean_tpr = np.zeros(n_points)
    for class_index in range(num_classes):
        fpr_class, tpr_class, _ = roc_curve(y_true_bin[:, class_index], y_proba[:, class_index])
        mean_tpr += np.interp(all_fpr, fpr_class, tpr_class)
    mean_tpr /= num_classes
    macro_auc = float(roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr"))
    return all_fpr, mean_tpr, macro_auc


def _plot_per_class_f1(
    supervised_f1: Mapping[str, float],
    probe_f1: Mapping[str, float],
    output_path: Path,
) -> None:
    labels = sorted(supervised_f1)
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(
        x - width / 2,
        [supervised_f1[label] for label in labels],
        width,
        label="Supervised watch",
        color="#2166AC",
    )
    ax.bar(
        x + width / 2,
        [probe_f1[label] for label in labels],
        width,
        label="Contrastive watch probe",
        color="#D6604D",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Activity class")
    ax.set_ylabel("F1 score")
    ax.set_ylim([0.0, 1.0])
    ax.set_title("Per-Class F1: Supervised Watch vs Contrastive Watch Probe")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_roc_comparison(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2166AC", "#D6604D"]
    for color, (label, (fpr, tpr, auc_value)) in zip(colors, curves.items()):
        ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{label} (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-Averaged One-vs-Rest ROC — Watch Models")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _supervised_probabilities(
    *,
    checkpoint_path: Path,
    x_watch: np.ndarray,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = load_checkpoint(checkpoint_path)
    model = SupervisedHARModel(in_channels=x_watch.shape[1], num_classes=num_classes, dropout=0.0).to(device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("best_model_state_dict")
    if state_dict is None:
        raise KeyError(f"No supervised model weights found in {checkpoint_path}")
    model.load_state_dict(state_dict)
    return _softmax_inference(model, x_watch, batch_size=batch_size, device=device)


def _probe_probabilities(
    *,
    probe_checkpoint_path: Path,
    test_split,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probe_checkpoint = load_checkpoint(probe_checkpoint_path)
    encoder_checkpoint_path = probe_checkpoint.get("encoder_checkpoint_path")
    if not encoder_checkpoint_path:
        raise KeyError(f"No encoder_checkpoint_path found in {probe_checkpoint_path}")
    contrastive_checkpoint = load_checkpoint(Path(encoder_checkpoint_path))
    contrastive_model = load_contrastive_model_from_checkpoint(contrastive_checkpoint, device=device)
    test_features = extract_probe_features(
        model=contrastive_model,
        split_data=test_split,
        evaluation_mode="watch",
        batch_size=batch_size,
        device=device,
    )
    probe_head = LinearProbeHead(in_dim=test_features.x.shape[1], num_classes=num_classes).to(device)
    head_state = probe_checkpoint.get("linear_head_state_dict") or probe_checkpoint.get("best_linear_head_state_dict")
    if head_state is None:
        raise KeyError(f"No probe head weights found in {probe_checkpoint_path}")
    probe_head.load_state_dict(head_state)
    return _linear_probe_softmax(probe_head, test_features.x, batch_size=batch_size, device=device)


def generate_report_figures(
    *,
    windows_root: Path,
    baseline_dir: Path,
    probe_dir: Path,
    output_dir: Path,
    watch_experiment: str,
    watch_probe_experiment: str,
    experiment_aliases: Mapping[str, str] | None = None,
    batch_size: int = 512,
    device: str | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = collect_metrics_payloads(
        baseline_dir,
        probe_dir,
        experiment_aliases=experiment_aliases,
    )
    watch_payload = _require_payload(payloads, watch_experiment)
    watch_probe_payload = _require_payload(payloads, watch_probe_experiment)

    confusion_watch_path = _copy_confusion_matrix(
        watch_payload,
        output_dir / "confusion_matrix_supervised_watch.png",
    )
    confusion_probe_path = _copy_confusion_matrix(
        watch_probe_payload,
        output_dir / "confusion_matrix_contrastive_watch_probe.png",
    )

    watch_test = watch_payload["payload"]["test"]
    watch_probe_test = watch_probe_payload["payload"]["test"]
    per_class_path = output_dir / "per_class_f1_watch_vs_ssl.png"
    _plot_per_class_f1(
        supervised_f1=watch_test["per_class_f1"],
        probe_f1=watch_probe_test["per_class_f1"],
        output_path=per_class_path,
    )

    repository = WindowRepository(windows_root)
    label_to_index = repository.manifest["label_to_index"]
    num_classes = len(label_to_index)
    classes = list(range(num_classes))
    test_split = repository.load_split("test")
    x_watch = np.stack([sample.x_watch for sample in test_split.samples], axis=0).astype(np.float32)
    y_true = np.asarray([sample.label_idx for sample in test_split.samples], dtype=np.int64)
    y_true_bin = label_binarize(y_true, classes=classes)

    torch_device = torch.device(device or detect_device())
    supervised_checkpoint_path = _artifact_path(watch_payload, "checkpoint_path")
    watch_probe_checkpoint_path = _artifact_path(watch_probe_payload, "checkpoint_path")
    y_proba_supervised = _supervised_probabilities(
        checkpoint_path=supervised_checkpoint_path,
        x_watch=x_watch,
        num_classes=num_classes,
        batch_size=batch_size,
        device=torch_device,
    )
    y_proba_probe = _probe_probabilities(
        probe_checkpoint_path=watch_probe_checkpoint_path,
        test_split=test_split,
        num_classes=num_classes,
        batch_size=batch_size,
        device=torch_device,
    )

    roc_watch = _macro_ovr_roc(y_true_bin, y_proba_supervised, num_classes)
    roc_probe = _macro_ovr_roc(y_true_bin, y_proba_probe, num_classes)
    roc_path = output_dir / "roc_auc_watch_vs_ssl.png"
    _plot_roc_comparison(
        {
            "Supervised watch": roc_watch,
            "Contrastive watch probe": roc_probe,
        },
        output_path=roc_path,
    )

    outputs = {
        "confusion_matrix_supervised_watch": str(confusion_watch_path),
        "confusion_matrix_contrastive_watch_probe": str(confusion_probe_path),
        "per_class_f1_watch_vs_ssl": str(per_class_path),
        "roc_auc_watch_vs_ssl": str(roc_path),
    }
    manifest_path = output_dir / "report_figures.json"
    manifest_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    outputs["manifest"] = str(manifest_path)
    return outputs


def main() -> None:
    args = parse_args()
    windows_root, output_dir = _resolve_roots(args)
    outputs = generate_report_figures(
        windows_root=windows_root,
        baseline_dir=args.baseline_dir.resolve(),
        probe_dir=args.probe_dir.resolve(),
        output_dir=output_dir,
        watch_experiment=args.watch_experiment,
        watch_probe_experiment=args.watch_probe_experiment,
        experiment_aliases=parse_experiment_aliases(args.experiment_alias),
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
