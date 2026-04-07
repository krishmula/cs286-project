from __future__ import annotations

import torch
from torch import nn


class TimeSeriesEncoder(nn.Module):
    output_dim = 256

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
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = TimeSeriesEncoder(in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.output_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))
