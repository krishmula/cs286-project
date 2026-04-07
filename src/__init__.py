"""Shared package for the phase 2 training and data-loading pipeline."""

from .config import DEFAULTS, ProjectPaths, RuntimeConfig
from .runtime import detect_device, resolve_project_paths, seed_everything
from .training import ClassificationMetrics, EarlyStopper, compute_classification_metrics

__all__ = [
    "DEFAULTS",
    "ProjectPaths",
    "RuntimeConfig",
    "ClassificationMetrics",
    "EarlyStopper",
    "compute_classification_metrics",
    "detect_device",
    "resolve_project_paths",
    "seed_everything",
]
