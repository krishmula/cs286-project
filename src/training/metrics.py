from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    confusion_matrix: np.ndarray
    per_class_f1: Dict[str, float]
    per_subject_accuracy: Dict[int, float]
    loss: Optional[float] = None

    def to_summary_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class_f1": dict(self.per_class_f1),
            "per_subject_accuracy": dict(self.per_subject_accuracy),
            "loss": self.loss,
        }


def confusion_matrix_from_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth), int(pred)] += 1
    return matrix


def macro_f1_from_confusion(
    matrix: np.ndarray,
    index_to_label: Mapping[int, str],
) -> tuple[float, Dict[str, float]]:
    scores = []
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
    return float(sum(scores) / len(scores)) if scores else 0.0, per_class_f1


def per_subject_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    subject_ids: Sequence[int],
) -> Dict[int, float]:
    counts: Dict[int, list[int]] = {}
    for subject_id, truth, pred in zip(subject_ids, y_true, y_pred):
        key = int(subject_id)
        if key not in counts:
            counts[key] = [0, 0]
        counts[key][0] += int(int(truth) == int(pred))
        counts[key][1] += 1
    return {
        subject_id: correct / total
        for subject_id, (correct, total) in sorted(counts.items())
        if total > 0
    }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    subject_ids: Sequence[int],
    index_to_label: Mapping[int, str],
    loss: Optional[float] = None,
) -> ClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) != len(subject_ids):
        raise ValueError("subject_ids must have the same length as y_true")
    if not index_to_label:
        raise ValueError("index_to_label must not be empty")

    num_classes = len(index_to_label)
    matrix = confusion_matrix_from_predictions(y_true, y_pred, num_classes=num_classes)
    accuracy = float(matrix.diagonal().sum() / max(1, matrix.sum()))
    macro_f1, per_class_scores = macro_f1_from_confusion(matrix, index_to_label)
    subject_scores = per_subject_accuracy(y_true, y_pred, subject_ids)
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion_matrix=matrix,
        per_class_f1=per_class_scores,
        per_subject_accuracy=subject_scores,
        loss=loss,
    )
