"""Custom augmentation modules not covered by torch_audiomentations."""

from __future__ import annotations

import importlib
from typing import Any, Sequence

import torch
import torchaudio.transforms as AT
from torch import nn
from asv.augment.rawboost import RawBoostCfg, process_rawboost_feature


class SpeedPerturbation(nn.Module):
    """Resample-based speed perturbation for ``[B, C, T]`` audio.

    Note: this changes tempo and pitch together (classic speed perturbation).
    By default, it uses 10 uniformly spaced factors in ``[min_rate, max_rate]`` so
    torchaudio can reuse a bounded set of resampling kernels across utterances.
    """

    def __init__(
        self,
        min_rate: float = 0.9,
        max_rate: float = 1.1,
        factors: Sequence[float] | None = None,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        if min_rate <= 0.0 or max_rate <= 0.0:
            raise ValueError("speed rates must be positive")
        if min_rate > max_rate:
            raise ValueError("min_rate must be <= max_rate")
        if factors is None:
            factors = self._default_factors(min_rate, max_rate)
        normalized_factors = tuple(float(rate) for rate in factors)
        if not normalized_factors:
            raise ValueError("factors must be non-empty")
        if any(rate <= 0.0 for rate in normalized_factors):
            raise ValueError("speed factors must be positive")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.factors = normalized_factors
        self.p = p
        self._sample_rate: int | None = None
        self._transform: AT.SpeedPerturbation | None = None

    _DEFAULT_NUM_FACTORS: int = 10

    @classmethod
    def _default_factors(cls, min_rate: float, max_rate: float) -> tuple[float, ...]:
        if min_rate == max_rate:
            return (float(min_rate),)
        n = cls._DEFAULT_NUM_FACTORS
        if n < 2:
            raise ValueError("default factor count must be >= 2 when min_rate != max_rate")
        span = max_rate - min_rate
        return tuple(float(min_rate + span * i / (n - 1)) for i in range(n))

    def _get_transform(self, sample_rate: int, device: torch.device) -> AT.SpeedPerturbation:
        if self._transform is None or self._sample_rate != sample_rate:
            self._transform = AT.SpeedPerturbation(orig_freq=sample_rate, factors=self.factors)
            self._sample_rate = sample_rate
        return self._transform.to(device)

    def forward(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.training or torch.rand(()) > self.p:
            return samples
        transform = self._get_transform(sample_rate, samples.device)
        out, _ = transform(samples)
        return out


class SpectrogramTimeMask(nn.Module):
    """Masks a contiguous time region in spectrogram domain ``[B, F, T]``."""

    def __init__(self, max_frames: int = 20, p: float = 0.5) -> None:
        super().__init__()
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")
        self.max_frames = max_frames
        self.p = p

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(()) > self.p:
            return features
        out = features.clone()
        _, _, t = out.shape
        width_max = min(self.max_frames, t)
        for i in range(out.shape[0]):
            width = int(torch.randint(1, width_max + 1, (1,), device=out.device).item())
            start = int(torch.randint(0, max(1, t - width + 1), (1,), device=out.device).item())
            out[i, :, start : start + width] = 0.0
        return out


class SpectrogramCutMix(nn.Module):
    """CutMix for spectrogram features ``[B, F, T]``."""

    def __init__(self, min_mix_frames: int = 5, max_mix_frames: int = 30, p: float = 0.5) -> None:
        super().__init__()
        if not (1 <= min_mix_frames <= max_mix_frames):
            raise ValueError("mix frames must satisfy 1 <= min <= max")
        self.min_mix_frames = min_mix_frames
        self.max_mix_frames = max_mix_frames
        self.p = p

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or features.shape[0] < 2 or torch.rand(()) > self.p:
            return features
        out = features.clone()
        perm = torch.randperm(out.shape[0], device=out.device)
        _, _, t = out.shape
        min_w = min(self.min_mix_frames, t)
        max_w = min(self.max_mix_frames, t)
        for i in range(out.shape[0]):
            width = int(torch.randint(min_w, max_w + 1, (1,), device=out.device).item())
            start = int(torch.randint(0, max(1, t - width + 1), (1,), device=out.device).item())
            out[i, :, start : start + width] = out[perm[i], :, start : start + width]
        return out


class RawBoostTransform(nn.Module):
    """RawBoost augmentation for waveform batches."""

    def __init__(
        self,
        algo: int = 5,
        cfg: dict[str, Any] | None = None,
        per_channel: bool = True,
        kwargs: dict[str, Any] | None = None,
        p: float = 0.5,
        target: str | None = None,
    ) -> None:
        super().__init__()
        self.algo = algo
        self.cfg = RawBoostCfg(**(cfg or {}))
        self.per_channel = per_channel
        self.kwargs = kwargs or {}
        self.p = p
        self.target = target
        self._callable: Any | None = None

    @staticmethod
    def _resolve_target(target: str) -> Any:
        module_name, attr = target.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr)

    def forward(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if not self.training or torch.rand(()) > self.p:
            return samples
        if self.target is not None:
            if self._callable is None:
                self._callable = self._resolve_target(self.target)
            fn = self._callable
            try:
                return fn(samples, sample_rate=sample_rate, **self.kwargs)
            except TypeError:
                return fn(samples, sample_rate_hz=sample_rate, **self.kwargs)
        out = samples.detach().cpu().clone()
        if out.ndim != 3:
            raise ValueError(f"RawBoostTransform expects [B,C,T], got {tuple(out.shape)}")
        if self.per_channel:
            for b in range(out.shape[0]):
                for c in range(out.shape[1]):
                    arr = out[b, c].numpy()
                    arr_aug = process_rawboost_feature(arr, sample_rate, self.cfg, self.algo)
                    out[b, c] = torch.from_numpy(arr_aug)
        else:
            for b in range(out.shape[0]):
                mono = out[b].mean(dim=0).numpy()
                mono_aug = process_rawboost_feature(mono, sample_rate, self.cfg, self.algo)
                aug = torch.from_numpy(mono_aug).unsqueeze(0).repeat(out.shape[1], 1)
                out[b] = aug
        out = out.to(device=samples.device, dtype=samples.dtype)
        return out
