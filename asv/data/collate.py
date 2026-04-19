"""Хелперы сборки батча и pre-processing'а спектрограмм.

Содержит:
- ``pad_random`` / ``pad_repeat_to_length_if_shorter`` — нормировка длины аудио;
- ``collate_waveform_batch`` — собирает waveform-батч ``[B,1,T]`` + ``labels [B]``;
- ``compute_spectrogram_batch`` — применяет spec-frontend (eval + no_grad) и
  :func:`apply_spec_augment` одним вызовом (используется в трейн-цикле);
- ``SpecAugmentConfig`` — параметры ReDimNet-style спектральных аугментаций
  (freq mask / time mask / cutmix);
- ``SpectrogramCollator`` — альтернативный collate, если нужно считать mel
  прямо в DataLoader-воркерах (не применяется в основном трейне)."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def pad_random(x: torch.Tensor, max_len: int = 64400) -> torch.Tensor:
    """Подгоняет аудио ``x`` к длине ``max_len``: короче — repeat-pad, длиннее — случайный crop.

    Работает по последней оси, форму сохраняет (``[T]``, ``[C,T]``, ``[B,C,T]`` — любой случай).
    """
    if x.ndim == 0:
        raise ValueError("expected at least 1D audio tensor")
    x_len = x.shape[-1]
    if x_len <= 0:
        raise ValueError("audio tensor cannot have zero length along time axis")
    if x_len >= max_len:
        start = random.randint(0, x_len - max_len)
        return x[..., start : start + max_len]

    num_repeats = int(max_len / x_len) + 1
    repeat_dims = [1] * x.ndim
    repeat_dims[-1] = num_repeats
    padded_x = x.repeat(*repeat_dims)[..., :max_len]
    return padded_x


def pad_repeat_to_length_if_shorter(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Repeat-pad по последней оси до ``target_len``, если короче; длиннее — не режет."""
    if x.ndim == 0:
        raise ValueError("expected at least 1D audio tensor")
    x_len = x.shape[-1]
    if x_len <= 0:
        raise ValueError("audio tensor cannot have zero length along time axis")
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if x_len >= target_len:
        return x
    num_repeats = int(target_len / x_len) + 1
    repeat_dims = [1] * x.ndim
    repeat_dims[-1] = num_repeats
    padded_x = x.repeat(*repeat_dims)[..., :target_len]
    return padded_x


def collate_waveform_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Собрать батч фиксированной длины: ``inputs [B,1,T]`` или ``[B,C,T]`` + ``labels [B]``."""
    waveforms = [item["waveform"] for item in batch]
    first = waveforms[0]
    if first.ndim == 1:
        x = torch.stack([w.unsqueeze(0) for w in waveforms], dim=0)
    elif first.ndim == 2:
        x = torch.stack(waveforms, dim=0)
    else:
        raise ValueError(f"expected waveform shape [T] or [C,T], got {tuple(first.shape)}")
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    return {"inputs": x, "labels": labels}


def clone_frozen_frontend(spec_frontend: nn.Module, device: torch.device | str = "cpu") -> nn.Module:
    """Глубокая копия spec-frontend на заданное устройство с отключёнными градиентами (eval-режим)."""
    frontend = copy.deepcopy(spec_frontend).to(device=torch.device(device), dtype=torch.float32)
    frontend.eval()
    for parameter in frontend.parameters():
        parameter.requires_grad_(False)
    return frontend


def build_worker_safe_frontend(spec_frontend: nn.Module) -> nn.Module:
    """CPU-only eval-клон для DataLoader-воркеров.

    Accelerate/FSDP переносят оригинальную модель на CUDA уже после того, как
    коллатор пикнулся в воркеры. Приватная CPU-копия держит воркеры вне CUDA-стейта
    и не ломается на fork при ``num_workers > 0``."""
    return clone_frozen_frontend(spec_frontend, device="cpu")


def _prepare_waveform_batch_for_frontend(waveforms: torch.Tensor) -> torch.Tensor:
    """Normalize waveform batches to the ``[B, T]`` shape expected by the frontend."""
    if waveforms.ndim == 2:
        return waveforms
    if waveforms.ndim == 3 and waveforms.shape[1] == 1:
        return waveforms[:, 0, :]
    raise ValueError(f"expected waveform batch shape [B,T] or [B,1,T], got {tuple(waveforms.shape)}")


def compute_spectrogram_batch(
    waveforms: torch.Tensor,
    spec_frontend: nn.Module,
    augment_cfg: SpecAugmentConfig | None = None,
    *,
    training: bool,
) -> torch.Tensor:
    """Посчитать спектрограммы из waveform-батча и (опционально) применить spec augment.

    Фронтенд всегда гоняется в ``eval()`` + ``no_grad`` — чтобы BatchNorm/подобные
    слои не копили статистики и не пытались считать градиенты. По возвращении
    режим фронтенда восстанавливается. Аугментации применяются только если
    ``training=True``."""
    x = _prepare_waveform_batch_for_frontend(waveforms)
    was_training = spec_frontend.training
    spec_frontend.eval()
    try:
        with torch.no_grad():
            spec = spec_frontend(x)
    finally:
        spec_frontend.train(was_training)
    return apply_spec_augment(spec, augment_cfg=augment_cfg, training=training)


# ---------------------------------------------------------------------------
# ReDimNet-style spectrogram augmentations (FbankAug approach)
# ---------------------------------------------------------------------------

def _mask_along_axis(x: torch.Tensor, dim: int, max_width: int) -> torch.Tensor:
    """Zero-mask a random contiguous region along *dim* for each sample.

    Uses the same vectorised logic as the vendor ``FbankAug``.
    ``x`` has shape ``[B, F, T]``.
    """
    batch, fea, time = x.shape
    D = fea if dim == 1 else time

    if D <= 1 or max_width <= 0:
        return x

    mask_len = torch.randint(0, max_width + 1, (batch, 1), device=x.device).unsqueeze(2)
    mask_pos = torch.randint(0, max(1, D - mask_len.max().item()), (batch, 1), device=x.device).unsqueeze(2)
    arange = torch.arange(D, device=x.device).view(1, 1, -1)
    mask = (mask_pos <= arange) & (arange < (mask_pos + mask_len))
    mask = mask.any(dim=1)  # [B, D]

    if dim == 1:
        mask = mask.unsqueeze(2)  # [B, F, 1]
    else:
        mask = mask.unsqueeze(1)  # [B, 1, T]

    return x.masked_fill(mask, 0.0)


def _cutmix(x: torch.Tensor, min_frames: int, max_frames: int) -> torch.Tensor:
    """CutMix: replace a random time region with the same region from another sample.

    Each row *i* always takes its patch from a *different* row (cyclic shift by random
    ``k in [1, B-1]``). ``torch.randperm`` was wrong here: with e.g. ``B=2`` it often
    returned ``[0, 1]``, so row 0 copied from itself and nothing changed.
    """
    if x.shape[0] < 2:
        return x
    b, _, t = x.shape
    k = int(torch.randint(1, b, (1,), device=x.device).item())
    donors = (torch.arange(b, device=x.device) + k) % b
    min_w = min(min_frames, t)
    max_w = min(max_frames, t)
    out = x.clone()
    for i in range(out.shape[0]):
        width = int(torch.randint(min_w, max_w + 1, (1,), device=out.device).item())
        start = int(torch.randint(0, max(1, t - width + 1), (1,), device=out.device).item())
        d = int(donors[i].item())
        out[i, :, start : start + width] = x[d, :, start : start + width]
    return out


def apply_spec_augment(
    spec: torch.Tensor,
    augment_cfg: SpecAugmentConfig | None = None,
    *,
    training: bool,
) -> torch.Tensor:
    """ReDimNet-style спектральные аугментации для батчей ``[B, F, T]``.

    Есть верхний гейт (``level_p`` — вероятность применить *что-либо* на этом батче).
    Дальше либо ``one_of`` (равномерный выбор между freq mask / time mask / cutmix),
    либо каждый оп независимо по своему ``p``. В eval-режиме passthrough."""
    cfg = augment_cfg or SpecAugmentConfig()
    if not training:
        return spec
    if torch.rand(()).item() >= cfg.level_p:
        return spec

    if cfg.one_of:
        # Uniform over {freq mask, time mask, cutmix}. Per-op p values below are ignored in this mode.
        op = int(torch.randint(0, 3, (1,)).item())
        if op == 0:
            return _mask_along_axis(spec, dim=1, max_width=cfg.freq_mask_max_width)
        if op == 1:
            return _mask_along_axis(spec, dim=2, max_width=cfg.time_mask_max_width)
        return _cutmix(spec, cfg.cutmix_min_frames, cfg.cutmix_max_frames)

    if torch.rand(()).item() < cfg.freq_mask_p:
        spec = _mask_along_axis(spec, dim=1, max_width=cfg.freq_mask_max_width)
    if torch.rand(()).item() < cfg.time_mask_p:
        spec = _mask_along_axis(spec, dim=2, max_width=cfg.time_mask_max_width)
    if torch.rand(()).item() < cfg.cutmix_p:
        spec = _cutmix(spec, cfg.cutmix_min_frames, cfg.cutmix_max_frames)
    return spec


@dataclass
class SpecAugmentConfig:
    """Параметры спектральных аугментаций (freq mask / time mask / cutmix).

    ``level_p`` гейтит весь блок: с вероятностью ``level_p`` вообще применяется что-то,
    иначе passthrough. При ``one_of=True`` выбирается ровно один оп равномерно."""

    one_of: bool = False
    #: Probability of applying any spec-domain augmentation in this collate step (gates the whole block).
    level_p: float = 1.0
    time_mask_p: float = 0.5
    time_mask_max_width: int = 10
    freq_mask_p: float = 0.5
    freq_mask_max_width: int = 8
    cutmix_p: float = 0.3
    cutmix_min_frames: int = 5
    cutmix_max_frames: int = 30


class SpectrogramCollator:
    """Collate function that converts waveforms to spectrograms and applies augmentations.

    Designed as a drop-in replacement for :func:`collate_waveform_batch`
    when training with a model whose spectrogram frontend has been extracted
    (e.g. ``ReDimNet2Encoder.spec_frontend``).

    Parameters
    ----------
    spec_frontend:
        An ``nn.Module`` that maps ``[B, T]`` waveforms to ``[B, F, T_frames]``
        mel-spectrograms.  Typically ``TFMelBanks`` or ``MelBanks``.
    augment_cfg:
        Augmentation probabilities / params.  Pass ``None`` to disable augmentation.
    training:
        When ``False`` augmentations are skipped.
    """

    def __init__(
        self,
        spec_frontend: nn.Module,
        augment_cfg: SpecAugmentConfig | None = None,
        training: bool = True,
    ) -> None:
        self.spec_frontend = build_worker_safe_frontend(spec_frontend)
        self.augment_cfg = augment_cfg or SpecAugmentConfig()
        self.training = training

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        waveforms = [item["waveform"] for item in batch]
        labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
        x = torch.stack(waveforms, dim=0)
        spec = compute_spectrogram_batch(
            x,
            self.spec_frontend,
            augment_cfg=self.augment_cfg,
            training=self.training,
        )

        return {"inputs": spec, "labels": labels}

    def _augment(self, spec: torch.Tensor) -> torch.Tensor:
        return apply_spec_augment(spec, augment_cfg=self.augment_cfg, training=self.training)
