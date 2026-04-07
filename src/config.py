from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    batch_size: int = 256
    learning_rate: float = 1e-3
    contrastive_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 40
    early_stopping_patience: int = 7
    grad_clip: float = 1.0
    dropout: float = 0.2
    num_workers: int = 0


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    artifacts_root: Path
    windows_root: Path
    baseline_root: Path
    docs_root: Path


DEFAULTS = RuntimeConfig()
