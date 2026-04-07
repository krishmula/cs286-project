from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import TimeSeriesEncoder


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = TimeSeriesEncoder.output_dim, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), dim=1)


@dataclass(frozen=True)
class ContrastiveForwardOutput:
    h_phone: torch.Tensor
    h_watch: torch.Tensor
    z_phone: torch.Tensor
    z_watch: torch.Tensor


class PhoneWatchContrastiveModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        encoder_cls: type[TimeSeriesEncoder] = TimeSeriesEncoder,
        projection_hidden_dim: int = 128,
        projection_out_dim: int = 64,
    ):
        super().__init__()
        self.phone_encoder = encoder_cls(in_channels=in_channels)
        self.watch_encoder = encoder_cls(in_channels=in_channels)
        self.phone_projector = ProjectionHead(
            in_dim=self.phone_encoder.output_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_out_dim,
        )
        self.watch_projector = ProjectionHead(
            in_dim=self.watch_encoder.output_dim,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_out_dim,
        )

    def encode_phone(self, x_phone: torch.Tensor) -> torch.Tensor:
        return self.phone_encoder(x_phone)

    def encode_watch(self, x_watch: torch.Tensor) -> torch.Tensor:
        return self.watch_encoder(x_watch)

    def project_phone(self, h_phone: torch.Tensor) -> torch.Tensor:
        return self.phone_projector(h_phone)

    def project_watch(self, h_watch: torch.Tensor) -> torch.Tensor:
        return self.watch_projector(h_watch)

    def forward(self, x_phone: torch.Tensor, x_watch: torch.Tensor) -> ContrastiveForwardOutput:
        h_phone = self.encode_phone(x_phone)
        h_watch = self.encode_watch(x_watch)
        z_phone = self.project_phone(h_phone)
        z_watch = self.project_watch(h_watch)
        return ContrastiveForwardOutput(
            h_phone=h_phone,
            h_watch=h_watch,
            z_phone=z_phone,
            z_watch=z_watch,
        )


@dataclass(frozen=True)
class InfoNCEResult:
    loss: torch.Tensor
    similarity_matrix: torch.Tensor
    phone_to_watch_loss: torch.Tensor
    watch_to_phone_loss: torch.Tensor


def symmetric_info_nce_loss(
    z_phone: torch.Tensor,
    z_watch: torch.Tensor,
    temperature: float = 0.2,
) -> InfoNCEResult:
    if z_phone.ndim != 2 or z_watch.ndim != 2:
        raise ValueError("z_phone and z_watch must be rank-2 tensors shaped [batch, dim]")
    if z_phone.shape != z_watch.shape:
        raise ValueError(
            f"z_phone and z_watch must have matching shapes, found {tuple(z_phone.shape)} and {tuple(z_watch.shape)}"
        )
    if z_phone.shape[0] < 2:
        raise ValueError("InfoNCE requires batch size of at least 2 to create negatives")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z_phone = F.normalize(z_phone, dim=1)
    z_watch = F.normalize(z_watch, dim=1)
    similarity_matrix = z_phone @ z_watch.T / temperature
    targets = torch.arange(z_phone.shape[0], device=z_phone.device)
    phone_to_watch_loss = F.cross_entropy(similarity_matrix, targets)
    watch_to_phone_loss = F.cross_entropy(similarity_matrix.T, targets)
    loss = 0.5 * (phone_to_watch_loss + watch_to_phone_loss)
    return InfoNCEResult(
        loss=loss,
        similarity_matrix=similarity_matrix,
        phone_to_watch_loss=phone_to_watch_loss,
        watch_to_phone_loss=watch_to_phone_loss,
    )
