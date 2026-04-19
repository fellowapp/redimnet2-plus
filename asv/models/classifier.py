"""Speaker-ID / spoof heads and optional metric-learning pre-heads."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from asv.losses.spheface2 import SphereFace2


class LinearClassifier(nn.Module):
    """Dense logits for cross-entropy or margin losses operating on logits."""

    def __init__(
        self, in_features: int, num_classes: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        del labels
        return self.fc(x)


class IdentityClassifier(nn.Module):
    """Return embeddings unchanged (for embedding-only workflows)."""

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        del labels
        return x


class SphereFace2Head(nn.Module):
    """SphereFace2 head: training returns ``(logits, loss)``.

    For speaker embeddings, use
    :meth:`asv.models.assemble.ASVModel.forward_features` instead of the
    no-label forward here (which only exposes cosine logits for scoring).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 32.0,
        margin: float = 0.2,
        first_margin: float | None = None,
        last_margin: float | None = None,
        lanbuda: float = 0.7,
        t: int = 3,
        margin_type: str = "C",
    ) -> None:
        super().__init__()
        if (first_margin is None) != (last_margin is None):
            raise ValueError(
                "SphereFace2Head margin schedule requires both "
                "first_margin and last_margin"
            )
        initial_margin = (
            float(first_margin) if first_margin is not None else float(margin)
        )
        self.first_margin = float(first_margin) if first_margin is not None else None
        self.last_margin = float(last_margin) if last_margin is not None else None
        self.head = SphereFace2(
            in_features,
            num_classes,
            scale=scale,
            margin=initial_margin,
            lanbuda=lanbuda,
            t=t,
            margin_type=margin_type,
        )

    def has_margin_schedule(self) -> bool:
        return self.first_margin is not None and self.last_margin is not None

    @property
    def current_margin(self) -> float:
        return float(self.head.margin)

    def set_margin(self, margin: float) -> float:
        self.head.update(float(margin))
        return self.current_margin

    def margin_for_step(self, step: int, total_steps: int) -> float:
        if not self.has_margin_schedule():
            return self.current_margin
        if total_steps <= 1:
            return float(self.first_margin)
        clamped_step = min(max(int(step), 0), int(total_steps) - 1)
        progress = clamped_step / float(total_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        assert self.first_margin is not None
        assert self.last_margin is not None
        return float(
            self.last_margin + (self.first_margin - self.last_margin) * cosine
        )

    def set_margin_for_step(self, step: int, total_steps: int) -> float:
        return self.set_margin(self.margin_for_step(step, total_steps))

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if labels is None:
            cos = F.linear(F.normalize(x), F.normalize(self.head.weight))
            return self.head.scale * cos
        return self.head(x, labels)
