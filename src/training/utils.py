from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class EarlyStopper:
    patience: int
    mode: str = "max"
    min_delta: float = 0.0
    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs_without_improvement: int = 0
    should_stop: bool = False
    history: list[dict[str, float | int | bool]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")

    def update(self, score: float, epoch: int) -> bool:
        improved = self._is_improvement(score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            self.should_stop = False
        else:
            self.epochs_without_improvement += 1
            self.should_stop = self.epochs_without_improvement >= self.patience
        self.history.append(
            {
                "epoch": epoch,
                "score": score,
                "improved": improved,
                "should_stop": self.should_stop,
            }
        )
        return improved

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > (self.best_score + self.min_delta)
        return score < (self.best_score - self.min_delta)


def checkpoint_metadata(
    *,
    experiment_name: str,
    stage: str,
    config: Mapping[str, Any],
    manifest_path: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "experiment_name": experiment_name,
        "stage": stage,
        "config": dict(config),
        "manifest_path": manifest_path,
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch.save(dict(payload), path)
    except ImportError:
        with path.open("wb") as handle:
            pickle.dump(dict(payload), handle)
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint payload must be a dict, found {type(checkpoint)!r}")
    return checkpoint
