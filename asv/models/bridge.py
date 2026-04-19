"""Pooling / projection between encoder output and classifier head."""

from __future__ import annotations

import torch
from torch import nn


class IdentityBridge(nn.Module):
    """No-op when encoder embedding dimension matches the classifier."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearBridge(nn.Module):
    """Linear projection ``[B, in_features] → [B, out_features]``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
