"""Training objectives; use :func:`build_loss` with Hydra configs."""

from asv.losses.cross_entropy import LabelSmoothingCrossEntropy
from asv.losses.integrated import LossComputedInModel
from asv.losses.registry import LOSSES, build_loss

__all__ = ["LOSSES", "LabelSmoothingCrossEntropy", "LossComputedInModel", "build_loss"]
