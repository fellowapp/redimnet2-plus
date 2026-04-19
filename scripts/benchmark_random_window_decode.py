#!/usr/bin/env python3
"""Compare random fixed-length window loading strategies (torchcodec range vs full decode).

Training uses :func:`asv.data.decode.load_with_torchcodec` with ``window_num_samples``
(see ``configs/data/default.yaml`` ``trim_num_samples``), which asks torchcodec for only
the time range instead of decoding the entire file when the clip is longer than the window.

This script times that path against decoding the full clip then cropping in Python.
"""

from __future__ import annotations

import argparse
import random
import statistics
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import torch

from asv.data.decode import load_with_torchcodec


def _write_synthetic_wav(path: Path, *, sample_rate: int, duration_sec: float) -> None:
    n = int(sample_rate * duration_sec)
    t = np.linspace(0.0, duration_sec, n, endpoint=False)
    mono = 0.2 * np.sin(2 * np.pi * 440.0 * t) + 0.15 * np.sin(2 * np.pi * 880.0 * t)
    pcm = np.clip(mono * 32_767.0, -32_768, 32_767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def bench_range_via_loader(
    path: Path,
    *,
    sample_rate: int,
    window_num_samples: int,
    iterations: int,
    seed: int,
) -> list[float]:
    times: list[float] = []

    for i in range(iterations):
        # :func:`load_with_torchcodec` uses the global ``random`` for the start offset.
        random.seed(seed + i * 1_009)
        t0 = time.perf_counter()
        w, sr = load_with_torchcodec(
            str(path),
            sample_rate=sample_rate,
            window_num_samples=window_num_samples,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)
        assert w.numel() == window_num_samples, (w.shape, window_num_samples)
        assert sr == sample_rate
    return times


def bench_full_torchcodec_then_crop(
    path: Path,
    *,
    sample_rate: int,
    window_num_samples: int,
    iterations: int,
    seed: int,
) -> list[float]:
    from torchcodec.decoders import AudioDecoder

    rng = random.Random(seed)
    times: list[float] = []

    for i in range(iterations):
        t0 = time.perf_counter()
        decoder = AudioDecoder(str(path), sample_rate=sample_rate)
        samples = decoder.get_all_samples()
        waveform = samples.data
        if waveform.ndim > 1:
            waveform = waveform.squeeze(0)
        waveform = waveform.to(dtype=torch.float32)
        total = int(waveform.numel())
        if total <= window_num_samples:
            out = waveform
        else:
            rng.seed(seed + i * 1_009)
            start = rng.randint(0, total - window_num_samples)
            out = waveform[start : start + window_num_samples]
        t1 = time.perf_counter()
        times.append(t1 - t0)
        assert out.numel() == window_num_samples
    return times


def _summarize(name: str, seconds: list[float]) -> str:
    ms = [s * 1000.0 for s in seconds]
    return (
        f"{name}: mean {statistics.mean(ms):.3f} ms  "
        f"stdev {statistics.pstdev(ms):.3f} ms  "
        f"min {min(ms):.3f} ms  max {max(ms):.3f} ms"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Path to a long audio file (wav/flac/etc.). If omitted, a synthetic WAV is used.",
    )
    p.add_argument("--sample-rate", type=int, default=16_000)
    p.add_argument(
        "--synthetic-seconds",
        type=float,
        default=120.0,
        help="Duration of synthetic WAV when --audio is not set.",
    )
    p.add_argument(
        "--window-samples",
        type=int,
        default=32_200,
        help="Window length in samples (default 32200 ≈ 2.01 s at 16 kHz, matches trim_num_samples).",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.audio is not None:
        audio_path = args.audio.expanduser().resolve()
        if not audio_path.is_file():
            raise SystemExit(f"not a file: {audio_path}")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        audio_path = Path(tmp.name)
        _write_synthetic_wav(
            audio_path,
            sample_rate=args.sample_rate,
            duration_sec=args.synthetic_seconds,
        )

    try:
        print(
            f"Audio: {audio_path}\n"
            f"sample_rate={args.sample_rate} window_samples={args.window_samples} "
            f"warmup={args.warmup} iterations={args.iterations}\n"
        )

        # Warmup (discard)
        for _ in range(args.warmup):
            random.seed(_)
            load_with_torchcodec(
                str(audio_path),
                sample_rate=args.sample_rate,
                window_num_samples=args.window_samples,
            )

        t_range = bench_range_via_loader(
            audio_path,
            sample_rate=args.sample_rate,
            window_num_samples=args.window_samples,
            iterations=args.iterations,
            seed=args.seed,
        )
        t_full = bench_full_torchcodec_then_crop(
            audio_path,
            sample_rate=args.sample_rate,
            window_num_samples=args.window_samples,
            iterations=args.iterations,
            seed=args.seed,
        )

        print(_summarize("load_with_torchcodec (random range decode)", t_range))
        print(_summarize("AudioDecoder.get_all_samples + random crop", t_full))
        speedup = statistics.mean(t_full) / statistics.mean(t_range)
        print(f"\nMean speedup (full / range): {speedup:.2f}x faster with range decode")
    finally:
        if args.audio is None and audio_path.is_file():
            audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
