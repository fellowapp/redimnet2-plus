"""Waveform / feature encoders producing fixed-size or sequence embeddings."""

from __future__ import annotations

import torch
from torch import nn


class IdentityEncoder(nn.Module):
    """Pass-through encoder (for Hydra defaults or feature-input pipelines)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvWaveformEncoder(nn.Module):
    """Lightweight 1-D CNN over raw waveform → global embedding [B, embed_dim].

    Expects input ``[B, 1, T]`` float waveform.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.extend(
                [
                    nn.Conv1d(c_in, c_out, kernel_size, stride=2, padding=kernel_size // 2),
                    nn.BatchNorm1d(c_out),
                    nn.ReLU(inplace=True),
                ]
            )
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(channels[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.backbone(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)
