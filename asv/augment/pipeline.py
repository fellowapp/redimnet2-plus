"""Пайплайн аугментаций для waveform- и feature-доменов.

Структура: :class:`AudioAugmentPipeline` содержит список :class:`AugmentationLevel` —
каждый уровень — это группа связанных преобразований (например, ``filters`` или
``additive_noise``) с собственным гейтом ``p`` и флагом ``one_of`` (применить ровно
один из трансформов) или чейном (применить по очереди все). Сборка из YAML —
через :func:`build_audio_augmentor` + ``hydra.utils.instantiate`` на каждый трансформ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn


def _as_tensor_output(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict) and "samples" in x:
        return x["samples"]
    raise TypeError(f"unexpected transform output type: {type(x)!r}")


@dataclass
class AugmentationLevel:
    """Один уровень аугментаций: группа взаимозаменяемых/последовательных трансформов.

    Поля:
    - ``p`` — вероятность того, что уровень сработает вообще (при промахе — passthrough);
    - ``one_of`` — если ``True``, применяется ровно один случайный трансформ из ``transforms``;
      иначе все трансформы чейнятся последовательно;
    - ``shuffle`` — перетасовать ``transforms`` при чейне."""

    name: str
    transforms: list[nn.Module]
    p: float = 1.0
    one_of: bool = False
    shuffle: bool = False

    def apply_waveform(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.transforms or torch.rand(()) > self.p:
            return samples
        modules = list(self.transforms)
        if self.shuffle:
            order = torch.randperm(len(modules)).tolist()
            modules = [modules[i] for i in order]
        if self.one_of:
            idx = int(torch.randint(0, len(modules), (1,)).item())
            return _apply_waveform_module(modules[idx], samples, sample_rate)
        out = samples
        for module in modules:
            out = _apply_waveform_module(module, out, sample_rate)
        return out

    def apply_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.transforms or torch.rand(()) > self.p:
            return features
        modules = list(self.transforms)
        if self.shuffle:
            order = torch.randperm(len(modules)).tolist()
            modules = [modules[i] for i in order]
        if self.one_of:
            idx = int(torch.randint(0, len(modules), (1,)).item())
            return _apply_feature_module(modules[idx], features)
        out = features
        for module in modules:
            out = _apply_feature_module(module, out)
        return out


def _apply_waveform_module(module: nn.Module, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
    try:
        out = module(samples, sample_rate)
    except TypeError:
        out = module(samples=samples, sample_rate=sample_rate)
    return _as_tensor_output(out)


def _apply_feature_module(module: nn.Module, features: torch.Tensor) -> torch.Tensor:
    try:
        out = module(features)
    except TypeError:
        out = module(samples=features)
    return _as_tensor_output(out)


class AudioAugmentPipeline(nn.Module):
    """Применяет сконфигурированные уровни аугментаций к waveform / features.

    Запускается сразу после :func:`asv.data.decode.load_with_torchcodec`. Поддерживает
    любые формы ``[T]``, ``[C,T]`` или ``[B,C,T]`` — добавит/уберёт временные оси,
    чтобы на вход уровням всегда пришёл ``[B,C,T]``.

    Только ``waveform_levels`` реально используются в трейне (spec augment применяется
    в :func:`asv.data.collate.apply_spec_augment`, не здесь)."""

    def __init__(
        self,
        waveform_levels: list[AugmentationLevel] | None = None,
        feature_levels: list[AugmentationLevel] | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.waveform_levels = waveform_levels or []
        self.feature_levels = feature_levels or []
        self.enabled = enabled

    def apply_to_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.enabled:
            return waveform
        x = waveform
        squeezed = False
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
            squeezed = True
        elif x.ndim == 2:
            x = x.unsqueeze(0)
            squeezed = True
        elif x.ndim != 3:
            raise ValueError(f"expected waveform [T], [C,T], or [B,C,T], got {tuple(x.shape)}")
        for level in self.waveform_levels:
            x = level.apply_waveform(x, sample_rate)
        if squeezed:
            x = x.squeeze(0)
        return x

    def apply_to_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return features
        if features.ndim != 3:
            raise ValueError(f"expected features [B,F,T], got {tuple(features.shape)}")
        x = features
        for level in self.feature_levels:
            x = level.apply_features(x)
        return x

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.apply_to_waveform(waveform, sample_rate)


def _build_level(level_cfg: DictConfig | dict[str, Any]) -> AugmentationLevel:
    if not OmegaConf.is_config(level_cfg):
        level_cfg = OmegaConf.create(level_cfg)
    name = str(level_cfg.get("name", "unnamed"))
    p = float(level_cfg.get("p", 1.0))
    one_of = bool(level_cfg.get("one_of", False))
    shuffle = bool(level_cfg.get("shuffle", False))
    transform_cfgs = level_cfg.get("transforms", [])
    transforms = [instantiate(tcfg, _convert_="all") for tcfg in transform_cfgs]
    return AugmentationLevel(name=name, transforms=transforms, p=p, one_of=one_of, shuffle=shuffle)


def build_audio_augmentor(cfg: DictConfig | dict[str, Any]) -> AudioAugmentPipeline:
    """Собирает :class:`AudioAugmentPipeline` из Hydra-конфига ``cfg.augment``."""
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    enabled = bool(cfg.get("enabled", True))
    waveform_levels = [_build_level(level) for level in cfg.get("waveform_levels", [])]
    feature_levels = [_build_level(level) for level in cfg.get("feature_levels", [])]
    return AudioAugmentPipeline(
        waveform_levels=waveform_levels,
        feature_levels=feature_levels,
        enabled=enabled,
    )
