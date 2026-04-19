"""Хелперы для декодирования аудио в датасете.

Основная функция — :func:`load_with_torchcodec`. На длинных файлах читается только
случайное окно ``window_num_samples``; это сильно дешевле по I/O и CPU, чем читать
полный файл и потом обрезать. Если ``torchcodec`` по каким-то причинам недоступен,
используется fallback на ``torchaudio.load_with_torchcodec`` + ресемпл через
``torchaudio.functional.resample``."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio.functional as AF


def ensure_torchaudio_io_compat() -> None:
    """Добивает отсутствующие torchaudio-хелперы для библиотек аугментаций.

    В некоторых окружениях есть только ``torchaudio.load_with_torchcodec``, а старые
    ``torchaudio.info``/``torchaudio.load`` отсутствуют. ``torch_audiomentations`` всё
    ещё обращается к ним за MUSAN/RIR-ассетами, так что мы подставляем эквиваленты
    через ``soundfile`` — это ничего не меняет в трейн-пайплайне, просто держит
    воркеры совместимыми."""
    import torchaudio

    if hasattr(torchaudio, "info") and hasattr(torchaudio, "load"):
        return

    def _sf_load(
        path: str,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        del normalize, kwargs
        start = int(frame_offset)
        frames = -1 if num_frames is None or num_frames < 0 else int(num_frames)
        array, sample_rate = sf.read(path, dtype="float32", always_2d=True, start=start, frames=frames)
        tensor = torch.from_numpy(array.T.copy())
        if not channels_first:
            tensor = tensor.transpose(0, 1)
        return tensor, int(sample_rate)

    class _Info:
        def __init__(self, sample_rate: int, num_frames: int, num_channels: int) -> None:
            self.sample_rate = int(sample_rate)
            self.num_frames = int(num_frames)
            self.num_channels = int(num_channels)

    def _info(path: str) -> _Info:
        with sf.SoundFile(str(path)) as audio_file:
            return _Info(
                sample_rate=int(audio_file.samplerate),
                num_frames=int(len(audio_file)),
                num_channels=int(audio_file.channels),
            )

    if not hasattr(torchaudio, "info"):
        setattr(torchaudio, "info", _info)
    if not hasattr(torchaudio, "load"):
        setattr(torchaudio, "load", _sf_load)
    if not hasattr(torchaudio, "load_with_torchcodec"):
        setattr(torchaudio, "load_with_torchcodec", _sf_load)


def load_with_torchcodec(
    path: str | Path,
    sample_rate: int | None = None,
    *,
    window_num_samples: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Прочитать аудио через ``torchcodec``, опционально сразу взять случайное окно.

    Если ``window_num_samples`` задан и файл длиннее этого окна — декодируем только
    нужный диапазон (не весь файл), это сильно дешевле. На ошибках torchcodec —
    fallback на ``torchaudio.load_with_torchcodec`` + ресемпл."""
    ensure_torchaudio_io_compat()

    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(
            str(path),
            sample_rate=int(sample_rate) if sample_rate is not None else None,
        )
        if window_num_samples is None or window_num_samples <= 0:
            samples = decoder.get_all_samples()
        else:
            out_sr = int(sample_rate or decoder.metadata.sample_rate)
            window_duration = float(window_num_samples) / float(out_sr)
            duration_seconds = float(decoder.metadata.duration_seconds or 0.0)
            if duration_seconds > window_duration:
                start_seconds = random.uniform(
                    0.0, duration_seconds - window_duration
                )
                samples = decoder.get_samples_played_in_range(
                    start_seconds=start_seconds,
                    stop_seconds=start_seconds + window_duration,
                )
            else:
                samples = decoder.get_all_samples()

        waveform = samples.data
        if waveform.ndim > 1:
            waveform = waveform.squeeze(0)
        waveform = waveform.to(dtype=torch.float32)
        return waveform, int(samples.sample_rate)
    except Exception:
        # Fall back to torchaudio's load_with_torchcodec path when direct
        # torchcodec bindings are unavailable in the runtime.
        import torchaudio

        waveform, src_sr = torchaudio.load_with_torchcodec(str(path))
        if waveform.ndim > 1:
            waveform = waveform.squeeze(0)
        waveform = waveform.to(dtype=torch.float32)
        out_sr = src_sr
        if sample_rate is not None and src_sr > 0 and src_sr != int(sample_rate):
            waveform = AF.resample(
                waveform.unsqueeze(0), src_sr, int(sample_rate)
            ).squeeze(0)
            out_sr = int(sample_rate)
        return waveform, out_sr
