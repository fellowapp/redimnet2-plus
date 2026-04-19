"""Сборка модели encoder → bridge → classifier через Hydra ``instantiate``.

Три поддерева в ``cfg.model`` (``encoder``, ``bridge``, ``classifier``) инстанцируются
независимо и собираются в :class:`ASVModel`. Это позволяет менять любую из частей
переопределением YAML без правки кода (например, подменить классификатор на
``SphereFace2Head``, а остальное оставить)."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

from hydra.utils import instantiate


class ASVModel(nn.Module):
    """Чейнит ``encoder`` → ``bridge`` → ``classifier``.

    :meth:`forward_features` возвращает эмбеддинги ``[B, D]`` — это то, что нужно для
    speaker verification (SV).

    :meth:`forward` передаёт ``labels`` в классификатор, если они заданы; margin-головы
    (например, :class:`asv.models.classifier.SphereFace2Head`) могут вернуть кортеж
    ``(logits, loss)`` — цикл трейна это учитывает."""

    def __init__(
        self,
        encoder: nn.Module,
        bridge: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bridge = bridge
        self.classifier = classifier

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.bridge(h)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z = self.forward_features(x)
        if return_embeddings:
            return z
        if labels is None:
            return self.classifier(z)
        return self.classifier(z, labels)


def build_components(
    cfg: DictConfig | dict[str, Any], *, _recursive_: bool = True
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Инстанцирует ``encoder``, ``bridge``, ``classifier`` из Hydra-конфига модели."""
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    OmegaConf.resolve(cfg)
    enc = instantiate(cfg.encoder, _recursive_=_recursive_)
    bridge = instantiate(cfg.bridge, _recursive_=_recursive_)
    clf = instantiate(cfg.classifier, _recursive_=_recursive_)
    return enc, bridge, clf


def build_model(cfg: DictConfig | dict[str, Any], *, _recursive_: bool = True) -> ASVModel:
    """Собирает :class:`ASVModel` из ``cfg.model`` (``encoder``, ``bridge``, ``classifier``).

    Каждое поддерево должно быть Hydra-style (``_target_`` + kwargs). Интерполяции
    резолвятся до инстанцирования — это важно, т.к. часть полей ссылается на
    ``${model.num_classes}`` и т.д."""
    enc, bridge, clf = build_components(cfg, _recursive_=_recursive_)
    return ASVModel(enc, bridge, clf)
