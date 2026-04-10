"""Generate ROC/AUC curves for the trained deep learning models.

Loads the best supervised CNN fusion checkpoint and the best contrastive paired probe
checkpoint, runs inference on the test set to collect softmax class probabilities, and
plots macro-averaged one-vs-rest ROC curves for both models on a single figure.

Outputs:
    artifacts/evaluation/roc_auc_comparison.png

Usage:
    python -m src.eval_roc_auc [--project-root PATH] [--output-dir PATH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from .data import WindowRepository
from .models import SupervisedHARModel
from .runtime import detect_device, resolve_project_paths
from .training.probe import (
    LinearProbeHead,
    extract_probe_features,
    load_contrastive_model_from_checkpoint,
)
from .training.utils import load_checkpoint


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_checkpoint(directory: Path, pattern: str) -> Path:
    """Return the first file matching glob pattern; raise if none found."""
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found matching '{pattern}' in {directory}"
        )
    return matches[0]


def _softmax_inference(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run batched inference and return softmax probabilities (n, num_classes)."""
    tensor  = torch.from_numpy(x.astype(np.float32))
    loader  = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    probas: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = model(batch.to(device))
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probas, axis=0)


def _linear_probe_softmax(
    head: LinearProbeHead,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run batched inference on pre-extracted features and return softmax probas."""
    tensor  = torch.from_numpy(features.astype(np.float32))
    loader  = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    probas: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for (batch,) in loader:
            logits = head(batch.to(device))
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probas, axis=0)


def _macro_ovr_roc(
    y_true_bin: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate and average per-class ROC curves to get macro OVR curve."""
    all_fpr  = np.linspace(0, 1, n_points)
    mean_tpr = np.zeros(n_points)
    for cls_idx in range(num_classes):
        fpr_c, tpr_c, _ = roc_curve(y_true_bin[:, cls_idx], y_proba[:, cls_idx])
        mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
    mean_tpr /= num_classes
    macro_auc = float(
        roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    )
    return all_fpr, mean_tpr, macro_auc


def _plot_roc_comparison(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
) -> None:
    """curves maps label → (fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2166AC", "#D6604D", "#4DAC26"]
    for color, (label, (fpr, tpr, auc)) in zip(colors, curves.items()):
        ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-Averaged One-vs-Rest ROC — Deep Learning Models")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir",   type=Path, default=None)
    parser.add_argument("--batch-size",   type=int,  default=512)
    return parser.parse_args()


def main() -> None:
    args  = parse_args()
    paths = resolve_project_paths(args.project_root)
    out_dir = (args.output_dir or paths.artifacts_root / "evaluation").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = detect_device()
    device     = torch.device(device_str)
    print(f"[eval_roc_auc] device={device}")

    # ── load manifest and test split ──────────────────────────────────────────
    repo           = WindowRepository(paths.windows_root)
    label_to_index = repo.manifest["label_to_index"]
    num_classes    = len(label_to_index)
    classes        = list(range(num_classes))

    print("[eval_roc_auc] loading test split ...")
    test_split = repo.load_split("test")

    x_fusion = np.stack(
        [s.x_fusion for s in test_split.samples], axis=0
    ).astype(np.float32)                         # (47790, 12, 60)
    y_true   = np.asarray(
        [s.label_idx for s in test_split.samples], dtype=np.int64
    )

    y_true_bin = label_binarize(y_true, classes=classes)  # (47790, 18)
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    # ── 1. Supervised CNN fusion baseline ─────────────────────────────────────
    print("[eval_roc_auc] loading supervised CNN fusion checkpoint ...")
    cnn_ckpt_path = _find_checkpoint(
        paths.baseline_root,
        "supervised_fusion_baseline_fusion_full_*_checkpoint.pt",
    )
    print(f"  using: {cnn_ckpt_path.name}")
    cnn_ckpt = load_checkpoint(cnn_ckpt_path)

    cnn_model = SupervisedHARModel(
        in_channels=12,
        num_classes=num_classes,
        dropout=0.0,  # eval mode — dropout is inactive regardless
    ).to(device)
    cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])

    print("[eval_roc_auc] running CNN inference ...")
    y_proba_cnn = _softmax_inference(cnn_model, x_fusion, args.batch_size, device)
    fpr, tpr, auc = _macro_ovr_roc(y_true_bin, y_proba_cnn, num_classes)
    curves["Supervised CNN (fusion)"] = (fpr, tpr, auc)
    print(f"  CNN macro AUC = {auc:.4f}")

    # ── 2. Contrastive paired probe ───────────────────────────────────────────
    contrastive_dir = paths.artifacts_root / "contrastive"
    probes_dir      = paths.artifacts_root / "probes"

    contrastive_ckpt_path = _find_checkpoint(
        contrastive_dir,
        "contrastive_phone_watch_contrastive_valloss*_checkpoint.pt",
    )
    probe_ckpt_path = _find_checkpoint(
        probes_dir,
        "probe_contrastive_pair_probe_100pct_*_checkpoint.pt",
    )
    print(f"[eval_roc_auc] contrastive ckpt : {contrastive_ckpt_path.name}")
    print(f"[eval_roc_auc] probe ckpt       : {probe_ckpt_path.name}")

    contrastive_ckpt = load_checkpoint(contrastive_ckpt_path)
    contrastive_model = load_contrastive_model_from_checkpoint(contrastive_ckpt, device)

    print("[eval_roc_auc] extracting paired probe features ...")
    probe_features = extract_probe_features(
        contrastive_model,
        test_split,
        evaluation_mode="pair",
        batch_size=args.batch_size,
        device=device,
    )  # x shape: (47790, 512)

    probe_ckpt = load_checkpoint(probe_ckpt_path)
    probe_head = LinearProbeHead(
        in_dim=probe_features.x.shape[1],
        num_classes=num_classes,
    ).to(device)
    # Probe checkpoints store the head state under 'linear_head_state_dict'
    head_state_key = "linear_head_state_dict" if "linear_head_state_dict" in probe_ckpt else "model_state_dict"
    probe_head.load_state_dict(probe_ckpt[head_state_key])

    print("[eval_roc_auc] running probe inference ...")
    y_proba_probe = _linear_probe_softmax(
        probe_head, probe_features.x, args.batch_size, device
    )
    fpr, tpr, auc = _macro_ovr_roc(y_true_bin, y_proba_probe, num_classes)
    curves["Contrastive Probe (pair)"] = (fpr, tpr, auc)
    print(f"  probe macro AUC = {auc:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    _plot_roc_comparison(curves, out_dir / "roc_auc_comparison.png")
    print("[eval_roc_auc] done.")


if __name__ == "__main__":
    main()
