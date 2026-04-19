"""Симуляция кодек-ухудшения: PCM → codec encode → codec decode → PCM.

Используется как lossy-аугментация в :class:`asv.data.parquet_dataset.ParquetAudioDataset`.
Поддерживает два бэкенда: ``ffmpeg`` (пайпы, быстрее) и ``pydub`` (через ``AudioSegment``;
нужен для aac/mp4 и пары других форматов, плохо работающих с пайпом). В режиме ``auto``
выбирается ffmpeg, если формат совместим с пайпами, иначе pydub.

Основной метод — :meth:`CodecPipeline.roundtrip`: принимает float32 ``[C,T]`` или ``[T]``,
возвращает ``(decoded_np, out_sample_rate)``. Тензоры torch поддерживаются ленивым
импортом (см. :meth:`CodecPipeline.forward`)."""

from __future__ import annotations

import io
import subprocess
from typing import Any

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError, CouldntEncodeError

from asv.codec.conversion import (
    float_pcm_to_s16le_bytes,
    float_pcm_to_segment,
    s16le_bytes_to_float_pcm,
    segment_to_float_pcm,
    soundfile_wav_segment_to_float,
    waveform_to_numpy_f32,
)
from asv.codec.registry import CodecPreset, get_preset

__all__ = [
    "CodecPipeline",
    "CodecUnavailableError",
]


class CodecUnavailableError(RuntimeError):
    """Бросается, когда preset помечен недоступным или encode/decode упал в рантайме."""


_AUTO_PIPE_FORMATS = frozenset(
    {
        "amr",
        "flac",
        "g722",
        "mp3",
        "ogg",
        "opus",
        "wav",
    }
)


class CodecPipeline:
    """Lossy codec simulation: float PCM -> codec encode -> decode -> float PCM.

    **Dataset usage (NumPy-only, tensor as late as possible):** call
    :meth:`roundtrip` with a float32 ``[C, T]`` (or ``[T]`` mono) array. It returns
    ``(decoded_np, sample_rate_hz)``. Build ``torch.Tensor`` in the collate
    function or training step, not here. :meth:`forward` exists only if you want
    a tensor in/tensor out path (imports ``torch`` on use).

    This is a plain class (not ``nn.Module``) so importing it does not require
    PyTorch; tensors are supported via a lazy import inside :meth:`roundtrip`.
    """

    def __init__(
        self,
        preset: str | CodecPreset,
        *,
        check_available: bool = False,
        backend: str = "auto",
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.preset: CodecPreset = get_preset(preset) if isinstance(preset, str) else preset
        if backend not in {"auto", "ffmpeg", "pydub"}:
            raise ValueError(f"unsupported codec backend {backend!r}")
        self.backend = backend
        self.ffmpeg_bin = ffmpeg_bin

        if check_available:
            from asv.codec.availability import check_preset_available

            if not check_preset_available(self.preset):
                raise CodecUnavailableError(
                    f"codec preset {self.preset.name!r} is not available on this system"
                )

    def roundtrip(
        self,
        waveform: np.ndarray | Any,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        """Encode and decode audio using the configured codec backend.

        Prefer passing **NumPy** so this path never imports ``torch``. If you pass
        a ``torch.Tensor``, it is copied to CPU NumPy (``detach()``; no grad).

        Parameters
        ----------
        waveform
            Float32 ``[C, T]`` or ``[T]`` (mono). Values nominally in ``[-1, 1]``.
        sample_rate
            Input sample rate in Hz.

        Returns
        -------
        decoded
            Float32 ``[C, T_out]`` **NumPy** array.
        out_sample_rate
            Sample rate after decode (usually ``normalize_frame_rate`` when set).
        """
        x = waveform_to_numpy_f32(waveform)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return self._roundtrip_prepared(x, sample_rate)

    def _roundtrip_prepared(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        backend = self.selected_backend()
        if backend == "ffmpeg":
            return self._roundtrip_ffmpeg(waveform, sample_rate)
        return self._roundtrip_pydub(waveform, sample_rate)

    def selected_backend(self) -> str:
        """Return the backend that will be used for the next roundtrip."""
        if self.backend != "auto":
            return self.backend
        if self._supports_ffmpeg_pipes():
            return "ffmpeg"
        return "pydub"

    def _supports_ffmpeg_pipes(self) -> bool:
        export_format = self.preset.export_format
        decode_format = self.preset.decode_format or export_format
        return export_format in _AUTO_PIPE_FORMATS and decode_format in _AUTO_PIPE_FORMATS

    def _roundtrip_pydub(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        seg_in = float_pcm_to_segment(waveform, sample_rate)
        if self.preset.target_frame_rate is not None:
            seg_in = seg_in.set_frame_rate(self.preset.target_frame_rate)

        buf = io.BytesIO()
        try:
            seg_in.export(buf, **self.preset.export_kwargs())
        except CouldntEncodeError as e:
            raise CodecUnavailableError(f"encode failed for {self.preset.name}: {e}") from e
        buf.seek(0)

        try:
            seg_out = AudioSegment.from_file(buf, **self.preset.decode_from_file_kwargs())
        except CouldntDecodeError as e:
            raise CodecUnavailableError(f"decode failed for {self.preset.name}: {e}") from e

        if self.preset.normalize_frame_rate is not None:
            seg_out = seg_out.set_frame_rate(self.preset.normalize_frame_rate)

        if self.preset.pcm_via_soundfile_wav:
            out, out_sr = soundfile_wav_segment_to_float(seg_out)
        else:
            out, out_sr = segment_to_float_pcm(seg_out)
        return out, out_sr

    def _roundtrip_ffmpeg(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        channels = int(waveform.shape[0])
        pcm_in = self._float_pcm_to_ffmpeg_input(waveform)
        encoded = self._encode_ffmpeg(pcm_in, sample_rate=sample_rate, channels=channels)
        out_sr = self._ffmpeg_output_sample_rate(sample_rate)
        decoded = self._decode_ffmpeg(encoded, sample_rate=out_sr, channels=channels)
        out = s16le_bytes_to_float_pcm(decoded, channels)
        return out, out_sr

    def _float_pcm_to_ffmpeg_input(self, waveform: np.ndarray) -> bytes:
        return float_pcm_to_s16le_bytes(waveform)

    def _encode_ffmpeg(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int,
        channels: int,
    ) -> bytes:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
        ]
        if self.preset.target_frame_rate is not None:
            cmd += ["-ar", str(self.preset.target_frame_rate)]
        if self.preset.export_codec is not None:
            cmd += ["-c:a", self.preset.export_codec]
        if self.preset.bitrate is not None:
            cmd += ["-b:a", self.preset.bitrate]
        cmd += list(self.preset.export_parameters)
        cmd += ["-f", self.preset.export_format, "pipe:1"]
        return self._run_ffmpeg(cmd, pcm_bytes, stage="encode")

    def _decode_ffmpeg(
        self,
        encoded_bytes: bytes,
        *,
        sample_rate: int,
        channels: int,
    ) -> bytes:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
        ]
        if self.preset.decode_format is not None:
            cmd += ["-f", self.preset.decode_format]
        if self.preset.decode_codec is not None:
            cmd += ["-c:a", self.preset.decode_codec]
        cmd += list(self.preset.decode_parameters)
        cmd += [
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "pipe:1",
        ]
        return self._run_ffmpeg(cmd, encoded_bytes, stage="decode")

    def _run_ffmpeg(self, cmd: list[str], payload: bytes, *, stage: str) -> bytes:
        try:
            completed = subprocess.run(
                cmd,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError as e:
            raise CodecUnavailableError(f"{stage} failed for {self.preset.name}: ffmpeg not found") from e
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace").strip()
            raise CodecUnavailableError(
                f"{stage} failed for {self.preset.name}: {stderr or 'ffmpeg returned non-zero exit'}"
            ) from e
        return completed.stdout

    def _ffmpeg_output_sample_rate(self, input_sample_rate: int) -> int:
        if self.preset.normalize_frame_rate is not None:
            return int(self.preset.normalize_frame_rate)
        if self.preset.target_frame_rate is not None:
            return int(self.preset.target_frame_rate)
        return int(input_sample_rate)

    def forward(
        self,
        waveform: Any,
        sample_rate: int,
    ) -> tuple[Any, int]:
        """Backward-compatible tensor adapter path."""
        from asv.codec.torch_adapter import TorchCodecPipelineAdapter

        return TorchCodecPipelineAdapter(self).forward(waveform, sample_rate)
