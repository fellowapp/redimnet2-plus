"""Cross-entropy variants for speaker / spoof classification."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingCrossEntropy(nn.Module):
    """Multi-class loss with label smoothing (same semantics as many ASV recipes)."""

    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(true_dist * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
