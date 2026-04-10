"""Train classical ML baselines on handcrafted features and generate evaluation figures.

Requires features to be pre-extracted by src.extract_features.

Models trained:
  dt  — Decision Tree (max_depth=20)
  rf  — Random Forest (200 trees)
  nb  — Gaussian Naive Bayes
  svm — Linear SVM with Platt calibration (gives predict_proba)
  ada — AdaBoost (100 estimators)
  xgb — XGBoost or HistGradientBoosting (fallback)

All models are evaluated on the subject-disjoint test split (same protocol as the
deep learning experiments, which is more stringent than a random 80/20 split).

Outputs to artifacts/classical/:
  classical_results.json
  rf_feature_importance.png
  rf_confusion_matrix.png
  roc_curves_classical.png
  per_class_f1_comparison.png

Usage:
    python -m src.train_classical [--project-root PATH] [--classical-dir PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .data import WindowRepository
from .runtime import resolve_project_paths


# ── model registry ────────────────────────────────────────────────────────────

def _build_models() -> dict:
    models = {
        "dt":  DecisionTreeClassifier(max_depth=20, random_state=42),
        "rf":  RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "nb":  GaussianNB(),
        "svm": CalibratedClassifierCV(
            LinearSVC(C=0.1, max_iter=2000, random_state=42)
        ),
        "ada": AdaBoostClassifier(n_estimators=100, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            n_estimators=200,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        print("[train_classical] using XGBoost")
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["xgb"] = HistGradientBoostingClassifier(
            max_iter=200, random_state=42
        )
        print("[train_classical] XGBoost not found; using HistGradientBoostingClassifier")
    return models


MODEL_DISPLAY_NAMES = {
    "dt":  "Decision Tree",
    "rf":  "Random Forest",
    "nb":  "Gaussian NB",
    "svm": "Linear SVM",
    "ada": "AdaBoost",
    "xgb": "XGBoost / HGB",
}


# ── figure helpers ────────────────────────────────────────────────────────────

def _plot_rf_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    top_k: int = 20,
) -> None:
    order = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in order],
        importances[order],
        color="steelblue",
    )
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title(f"Random Forest — Top {top_k} Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def _plot_confusion_matrix(
    matrix: np.ndarray,
    index_to_label: dict[int, str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ticks  = np.arange(len(index_to_label))
    labels = [index_to_label[i] for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def _plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: Path,
) -> None:
    """roc_data maps model_key → (fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    for color, (key, (fpr, tpr, auc)) in zip(colors, roc_data.items()):
        label = f"{MODEL_DISPLAY_NAMES.get(key, key)} (AUC={auc:.3f})"
        ax.plot(fpr, tpr, color=color, lw=1.8, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-Averaged One-vs-Rest ROC Curves — Classical Models")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


def _plot_per_class_f1(
    rf_f1: dict[str, float],
    cnn_f1: dict[str, float],
    probe_f1: dict[str, float],
    out_path: Path,
) -> None:
    labels = sorted(rf_f1.keys())
    x      = np.arange(len(labels))
    width  = 0.26

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, [rf_f1[l] for l in labels],    width, label="Random Forest",          color="#4878CF")
    ax.bar(x,         [cnn_f1[l] for l in labels],   width, label="Supervised CNN (fusion)", color="#6ACC65")
    ax.bar(x + width, [probe_f1[l] for l in labels], width, label="Contrastive Probe (pair)",color="#D65F5F")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Activity class")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-Class F1 Comparison: Classical vs Supervised CNN vs Contrastive Probe")
    ax.legend()
    ax.set_ylim([0, 1.0])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── confusion matrix from raw predictions ─────────────────────────────────────

def _confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(t), int(p)] += 1
    return matrix


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",  type=Path, default=None)
    parser.add_argument("--classical-dir", type=Path, default=None,
                        help="Directory with features_*.npz files and where outputs are written.")
    return parser.parse_args()


def _load_features(classical_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    npz = np.load(classical_dir / f"features_{split}.npz", allow_pickle=True)
    X   = npz["X"].astype(np.float32)
    y   = npz["y"].astype(np.int64)
    feature_names = npz["feature_names"].tolist()
    return X, y, feature_names


def _load_per_class_f1_from_metrics(metrics_glob_pattern: Path) -> dict[str, float]:
    """Load per_class_f1 from the first matching *_metrics.json file."""
    matches = sorted(metrics_glob_pattern.parent.glob(metrics_glob_pattern.name))
    if not matches:
        raise FileNotFoundError(f"No file matches: {metrics_glob_pattern}")
    with matches[0].open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["test"]["per_class_f1"]


def main() -> None:
    args  = parse_args()
    paths = resolve_project_paths(args.project_root)
    classical_dir = (args.classical_dir or paths.artifacts_root / "classical").resolve()

    # ── load manifest for label mapping ───────────────────────────────────────
    repo          = WindowRepository(paths.windows_root)
    label_to_index = repo.manifest["label_to_index"]
    index_to_label = {v: k for k, v in label_to_index.items()}
    num_classes    = len(label_to_index)
    classes        = list(range(num_classes))

    # ── load features ─────────────────────────────────────────────────────────
    print("[train_classical] loading features ...")
    X_train, y_train, feature_names = _load_features(classical_dir, "train")
    X_test,  y_test,  _             = _load_features(classical_dir, "test")
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    # binarize labels for OVR ROC/AUC
    y_test_bin = label_binarize(y_test, classes=classes)

    # ── train and evaluate each model ─────────────────────────────────────────
    models     = _build_models()
    results    = {}
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for key, clf in models.items():
        display = MODEL_DISPLAY_NAMES.get(key, key)
        print(f"[train_classical] fitting {display} ...")
        clf.fit(X_train, y_train)

        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)   # (n_test, num_classes)

        acc      = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        per_class_f1 = {
            index_to_label[i]: float(f1_score(
                y_test == i, y_pred == i, average="binary", zero_division=0
            ))
            for i in range(num_classes)
        }
        macro_auc = float(
            roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
        )

        print(f"  acc={acc:.4f}  macro_f1={macro_f1:.4f}  macro_auc={macro_auc:.4f}")
        results[key] = {
            "display_name": display,
            "accuracy":     acc,
            "macro_f1":     macro_f1,
            "macro_roc_auc": macro_auc,
            "per_class_f1": per_class_f1,
        }

        # macro OVR ROC curve: average the per-class curves
        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        for cls_idx in range(num_classes):
            fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, cls_idx], y_proba[:, cls_idx])
            mean_tpr += np.interp(all_fpr, fpr_c, tpr_c)
        mean_tpr /= num_classes
        roc_data[key] = (all_fpr, mean_tpr, macro_auc)

    # ── save results JSON ──────────────────────────────────────────────────────
    results_path = classical_dir / "classical_results.json"
    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[train_classical] saved → {results_path}")

    # ── RF feature importance ──────────────────────────────────────────────────
    rf_clf = models["rf"]
    _plot_rf_feature_importance(
        rf_clf.feature_importances_,
        feature_names,
        classical_dir / "rf_feature_importance.png",
    )

    # ── RF confusion matrix ────────────────────────────────────────────────────
    rf_pred   = rf_clf.predict(X_test)
    rf_matrix = _confusion_matrix_np(y_test, rf_pred, num_classes)
    _plot_confusion_matrix(
        rf_matrix,
        index_to_label,
        "Random Forest — Confusion Matrix (test set)",
        classical_dir / "rf_confusion_matrix.png",
    )

    # ── ROC curves for all classical models ───────────────────────────────────
    _plot_roc_curves(roc_data, classical_dir / "roc_curves_classical.png")

    # ── per-class F1 comparison: RF vs CNN fusion vs contrastive probe ─────────
    cnn_metrics_pattern  = paths.baseline_root / "supervised_fusion_baseline_fusion_full_*_metrics.json"
    probe_metrics_pattern = paths.artifacts_root / "probes" / "probe_contrastive_pair_probe_100pct_*_metrics.json"

    try:
        cnn_f1   = _load_per_class_f1_from_metrics(cnn_metrics_pattern)
        probe_f1 = _load_per_class_f1_from_metrics(probe_metrics_pattern)
        rf_f1    = results["rf"]["per_class_f1"]
        _plot_per_class_f1(
            rf_f1, cnn_f1, probe_f1,
            classical_dir / "per_class_f1_comparison.png",
        )
    except FileNotFoundError as exc:
        print(f"[train_classical] skipping per-class F1 comparison: {exc}")

    print("[train_classical] done.")


if __name__ == "__main__":
    main()
