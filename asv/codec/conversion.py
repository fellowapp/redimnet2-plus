"""Waveform/AudioSegment and PCM conversion helpers."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from pydub import AudioSegment


def waveform_to_numpy_f32(waveform: np.ndarray | Any) -> np.ndarray:
    """Convert supported waveform types to float32 ``[C, T]`` (or 1-D)."""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(waveform, torch.Tensor):
        return waveform.detach().cpu().to(dtype=torch.float32).numpy()
    return np.asarray(waveform, dtype=np.float32)


def float_pcm_to_s16le_bytes(waveform: np.ndarray) -> bytes:
    """Convert float32 ``[C, T]`` waveform to interleaved signed 16-bit PCM bytes."""
    if waveform.ndim != 2:
        raise ValueError(f"expected waveform [C, T], got shape {waveform.shape}")
    c, _t = waveform.shape
    if c < 1 or c > 2:
        raise ValueError(f"only 1-2 channels supported, got {c}")
    clipped = np.clip(waveform.astype(np.float32, copy=False), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    if c == 1:
        return pcm[0].tobytes()
    interleaved = np.empty((pcm.shape[1], c), dtype=np.int16)
    interleaved[:, 0] = pcm[0]
    interleaved[:, 1] = pcm[1]
    return interleaved.tobytes()


def s16le_bytes_to_float_pcm(raw: bytes, channels: int) -> np.ndarray:
    """Convert interleaved signed 16-bit PCM bytes to float32 ``[C, T]``."""
    if channels < 1 or channels > 2:
        raise ValueError(f"only 1-2 channels supported, got {channels}")
    arr = np.frombuffer(raw, dtype=np.int16)
    if channels == 1:
        return (arr.astype(np.float32) / 32768.0)[np.newaxis, :]
    if arr.size % channels != 0:
        raise ValueError(f"PCM byte count {len(raw)} is not divisible by {channels} channels")
    return (arr.reshape(-1, channels).T.astype(np.float32) / 32768.0)


def float_pcm_to_segment(waveform: np.ndarray, sample_rate: int) -> AudioSegment:
    """Build ``AudioSegment`` from float32 ``[C, T]`` in [-1, 1]."""
    if waveform.ndim != 2:
        raise ValueError(f"expected waveform [C, T], got shape {waveform.shape}")
    c, _t = waveform.shape
    if c < 1 or c > 2:
        raise ValueError(f"only 1-2 channels supported, got {c}")
    raw = float_pcm_to_s16le_bytes(waveform)
    return AudioSegment(
        data=raw,
        sample_width=2,
        frame_rate=sample_rate,
        channels=c,
    )


def segment_to_float_pcm(seg: AudioSegment) -> tuple[np.ndarray, int]:
    """Convert ``AudioSegment`` to float32 ``[C, T]`` and output sample rate."""
    arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
    sw = seg.sample_width
    if sw == 2:
        arr /= 32768.0
    elif sw == 1:
        arr = (arr - 128.0) / 128.0
    elif sw == 3:
        # pydub exposes 24-bit PCM as signed integers in int32 container.
        arr /= 8388608.0
    elif sw == 4:
        arr /= 2147483648.0
    else:
        raise ValueError(f"unsupported sample_width {sw}")
    ch = seg.channels
    if ch == 1:
        out = arr[np.newaxis, :]
    else:
        out = arr.reshape(-1, ch).T
    return out, int(seg.frame_rate)


def soundfile_wav_segment_to_float(seg: AudioSegment) -> tuple[np.ndarray, int]:
    """Match the ``export(wav)`` -> ``soundfile.read`` path (optional)."""
    import soundfile as sf

    buf = io.BytesIO()
    seg.export(buf, format="wav")
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32", always_2d=True)
    return data.T.astype(np.float32, copy=False), int(sr)
