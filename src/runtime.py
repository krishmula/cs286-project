from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np

from .config import ProjectPaths


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_project_paths(project_root: Optional[Path] = None) -> ProjectPaths:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    root = root.resolve()
    artifacts_root = root / "artifacts"
    return ProjectPaths(
        project_root=root,
        artifacts_root=artifacts_root,
        windows_root=artifacts_root / "windows",
        baseline_root=artifacts_root / "baseline",
        docs_root=root / "docs",
    )
