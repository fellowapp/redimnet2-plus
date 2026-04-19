"""Сборка функции потерь через Hydra + реестр имён для документации/поиска.

:func:`build_loss` — основной путь инстанцирования: принимает узел конфига с
``_target_`` (и опциональными kwargs) и возвращает уже сконструированный объект.
:data:`LOSSES` — короткие имена → классы лоссов (удобно для тестов и документации)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate

from torch.nn import CrossEntropyLoss

from asv.losses import cross_entropy as ce
from asv.losses.integrated import LossComputedInModel


def build_loss(cfg: DictConfig | dict[str, Any], *, _recursive_: bool = True) -> Any:
    """Инстанцирует функцию потерь из Hydra-конфига (``_target_`` + kwargs).

    Пример YAML::

        _target_: torch.nn.CrossEntropyLoss
        label_smoothing: 0.0

    Для ``SphereFace2``: лосс считается внутри модели (классификатор возвращает
    ``(logits, loss)``), а здесь инстанцируется плейсхолдер
    :class:`asv.losses.integrated.LossComputedInModel`."""
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    OmegaConf.resolve(cfg)
    return instantiate(cfg, _recursive_=_recursive_)


# Short names for documentation, tests, and optional non-YAML wiring.
LOSSES: dict[str, type] = {
    "cross_entropy": CrossEntropyLoss,
    "label_smoothing_cross_entropy": ce.LabelSmoothingCrossEntropy,
    "loss_computed_in_model": LossComputedInModel,
}
