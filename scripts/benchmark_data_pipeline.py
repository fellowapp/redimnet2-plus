#!/usr/bin/env python3
"""Benchmark dataset -> augment -> codec -> spec front-end preparation."""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import soundfile as sf
from omegaconf import OmegaConf

from asv.data import CodecAugmentProbabilities, ParquetAudioDataset, SpectrogramCollator, collate_waveform_batch
from asv.data.collate import clone_frozen_frontend, compute_spectrogram_batch
from asv.data.decode import ensure_torchaudio_io_compat
from asv.manifest.schema import SCHEMA as MANIFEST_SCHEMA
from asv.models.redimnet2 import ReDimNet2Encoder
from asv.train.spec_augment import build_spec_augment_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile ASV training input pipeline components.")
    parser.add_argument("--manifest", type=Path, default=None, help="Existing manifest path. If missing, create synthetic one.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/asv_benchmark_data_pipeline"), help="Where to place synthetic assets.")
    parser.add_argument("--num-utterances", type=int, default=1024, help="Synthetic dataset size when manifest is generated.")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--trim-num-samples", type=int, default=32_200, help="Fixed waveform length used by training collate input.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--iterations", type=int, default=256, help="How many dataset items to sample in item-level benchmark.")
    parser.add_argument("--num-batches", type=int, default=128, help="How many dataloader batches to time.")
    parser.add_argument("--phase", choices=("all", "dataset", "loader"), default="all")
    parser.add_argument("--codec-preset", type=str, default=None, help="e.g. mp3_16k / opus_16k / none")
    parser.add_argument(
        "--codec-backend",
        choices=("auto", "ffmpeg", "pydub"),
        default="auto",
        help="Codec roundtrip backend to benchmark.",
    )
    parser.add_argument("--keep-original", type=float, default=None, help="Override transform keep probability.")
    parser.add_argument("--codec-only", type=float, default=0.0)
    parser.add_argument("--augment-only", type=float, default=0.3)
    parser.add_argument("--codec-then-augment", type=float, default=0.0)
    parser.add_argument("--codec-min-duration-sec", type=float, default=None, help="Skip codec for waveforms shorter than this.")
    parser.add_argument("--waveform-speed-augment", action="store_true", help="Inject a waveform SpeedPerturbation level in synthetic config.")
    parser.add_argument("--spec-level-p", type=float, default=1.0, help="Spec-augment level_p passed to collator.")
    parser.add_argument(
        "--spec-path",
        choices=("cpu_collator", "gpu_preprocess", "compare"),
        default="cpu_collator",
        help="Benchmark current CPU collator path, GPU preprocessing path, or both.",
    )
    parser.add_argument(
        "--benchmark-device",
        type=str,
        default="cuda",
        help="Target device for ready-for-model batches (e.g. cuda, cuda:1, cpu).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--clean-work-dir", action="store_true", default=True)
    parser.add_argument("--codec-baseline", action="store_true", help="Run a codec-free baseline in the same script call.")
    return parser.parse_args()


def ms(values: list[float]) -> list[float]:
    return [x * 1000 for x in values]


def report_stage(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name}: no samples")
        return
    arr = ms(values)
    arr_sorted = sorted(arr)
    mean = statistics.mean(arr)
    med = statistics.median(arr)
    p95 = arr_sorted[int(0.95 * (len(arr_sorted) - 1))]
    p99 = arr_sorted[int(0.99 * (len(arr_sorted) - 1))]
    print(
        f"{name:<28} count={len(arr):>4} mean={mean:>8.3f} ms | "
        f"p50={med:>7.3f} p95={p95:>8.3f} p99={p99:>8.3f}"
    )


def synth_audio_path(path: Path, sample_rate: int, n_samples: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    f = 220.0 + 120.0 * rng.random()
    base = 0.04 * np.sin(2.0 * np.pi * f * t)
    noise = 0.01 * rng.normal(0.0, 1.0, size=(n_samples,))
    x = base + noise
    if channels == 1:
        return x.astype(np.float32)
    x2 = (0.7 * x + 0.3 * rng.normal(0.0, 1.0, size=n_samples)).astype(np.float32)
    return np.stack([x, x2], axis=-1).astype(np.float32)


def build_synthetic_manifest(work_dir: Path, n: int, sample_rate: int, seed: int) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = work_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    audio_paths: list[str] = []
    rel_paths: list[str] = []
    speaker_ids: list[str] = []
    video_ids: list[str] = []
    segment_ids: list[str] = []
    audio_ids: list[str] = []
    durations: list[float] = []

    for idx in range(n):
        sec = 0.5 + float(rng.uniform(0.5, 8.0))
        n_samples = int(sec * sample_rate)
        channels = 1 if idx % 3 else 2
        w = synth_audio_path(audio_dir / f"utt_{idx:06d}.wav", sample_rate, n_samples, channels, rng)
        path = audio_dir / f"utt_{idx:06d}.wav"
        sf.write(path, w, samplerate=sample_rate, subtype="PCM_16")
        audio_paths.append(str(path.resolve()))
        rel_paths.append(path.name)
        speaker_ids.append(f"spk_{idx % 64}")
        video_ids.append(f"vid_{idx % 32}")
        segment_ids.append(f"{idx:06d}")
        audio_ids.append(f"spk_{idx % 64}/vid_{idx % 32}/{idx:06d}")
        durations.append(float(n_samples) / sample_rate)

    manifest_path = work_dir / "manifest.parquet"
    table = pa.table(
        {
            "audio_path": audio_paths,
            "rel_path": rel_paths,
            "speaker_id": speaker_ids,
            "video_id": video_ids,
            "segment_id": segment_ids,
            "audio_id": audio_ids,
            "duration_sec": durations,
        },
        schema=MANIFEST_SCHEMA,
    )
    pq.write_table(table, manifest_path)
    return manifest_path


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def wrap_timed(
    name: str,
    obj: Any,
    attr: str,
    bucket: dict[str, list[float]],
    *,
    sync_device_for_timing: torch.device | None = None,
) -> Callable:
    method = getattr(obj, attr)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if sync_device_for_timing is not None:
            synchronize_device(sync_device_for_timing)
        t0 = time.perf_counter()
        out = method(*args, **kwargs)
        if sync_device_for_timing is not None:
            synchronize_device(sync_device_for_timing)
        bucket.setdefault(name, []).append(time.perf_counter() - t0)
        return out

    setattr(obj, attr, wrapped)
    return method


def restore_timed(obj: Any, attr: str, original: Callable) -> None:
    setattr(obj, attr, original)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device=device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def prepare_batch_for_model(
    batch: dict[str, torch.Tensor],
    *,
    spec_path: str,
    device: torch.device,
    timings: dict[str, list[float]],
    spec_frontend: torch.nn.Module | None,
    spec_cfg: Any,
) -> dict[str, torch.Tensor]:
    synchronize_device(device)
    t0 = time.perf_counter()
    device_batch = move_batch_to_device(batch, device)
    if spec_path == "gpu_preprocess":
        if spec_frontend is None:
            raise ValueError("spec_frontend is required for gpu_preprocess benchmarks")
        device_batch = {
            "inputs": compute_spectrogram_batch(
                device_batch["inputs"],
                spec_frontend,
                augment_cfg=spec_cfg,
                training=True,
            ),
            "labels": device_batch["labels"],
        }
    synchronize_device(device)
    timings.setdefault("ready_batch", []).append(time.perf_counter() - t0)
    return device_batch


def build_augment_cfg(speed_augment: bool) -> dict[str, Any]:
    if not speed_augment:
        return {
            "enabled": False,
            "_target_": "asv.augment.build_audio_augmentor",
            "waveform_levels": [],
            "feature_levels": [],
        }
    return {
        "enabled": True,
        "_target_": "asv.augment.build_audio_augmentor",
        "waveform_levels": [
            {
                "name": "speed",
                "p": 0.5,
                "one_of": False,
                "transforms": [
                    {
                        "_target_": "asv.augment.SpeedPerturbation",
                        "min_rate": 0.95,
                        "max_rate": 1.05,
                        "p": 1.0,
                    }
                ],
            }
        ],
        "feature_levels": [],
    }


def benchmark_dataset_items(dataset: ParquetAudioDataset, n: int) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = {}
    originals: list[Tuple[Any, str, Callable]] = []
    originals.append((dataset, "__getitem__", wrap_timed("getitem_total", dataset, "__getitem__", timings)))
    originals.append((dataset, "_read_row", wrap_timed("read_row", dataset, "_read_row", timings)))
    originals.append((dataset, "_decode", wrap_timed("decode", dataset, "_decode", timings)))
    originals.append((dataset, "_apply_codec", wrap_timed("codec", dataset, "_apply_codec", timings)))
    originals.append((dataset, "_trim_if_long", wrap_timed("trim_long", dataset, "_trim_if_long", timings)))
    originals.append((dataset, "_pad_or_trim", wrap_timed("pad_or_trim", dataset, "_pad_or_trim", timings)))
    originals.append((dataset, "_to_mono", wrap_timed("to_mono", dataset, "_to_mono", timings)))
    if dataset.codec is not None:
        codec_methods = [
            ("codec_roundtrip_total", "_roundtrip_prepared"),
            ("codec_ffmpeg_total", "_roundtrip_ffmpeg"),
            ("codec_pydub_total", "_roundtrip_pydub"),
            ("codec_ffmpeg_pcm_in", "_float_pcm_to_ffmpeg_input"),
            ("codec_ffmpeg_encode", "_encode_ffmpeg"),
            ("codec_ffmpeg_decode", "_decode_ffmpeg"),
        ]
        for name, attr in codec_methods:
            if hasattr(dataset.codec, attr):
                originals.append((dataset.codec, attr, wrap_timed(name, dataset.codec, attr, timings)))
    if hasattr(dataset.augmentor, "apply_to_waveform"):
        originals.append(
            (
                dataset.augmentor,
                "apply_to_waveform",
                wrap_timed("augment_waveform", dataset.augmentor, "apply_to_waveform", timings),
            )
        )
    idx = list(range(min(n, len(dataset))))
    # Keep index order stable for easier reproducibility.
    for i in idx:
        _ = dataset.__getitem__(i)

    for obj, attr, original in originals:
        restore_timed(obj, attr, original)

    return timings


def benchmark_dataloader(loader: torch.utils.data.DataLoader) -> dict[str, float]:
    start = time.perf_counter()
    seen = 0
    it = iter(loader)
    for _ in range(len(loader)):
        batch = next(it)
        seen += batch["labels"].shape[0]
    elapsed = time.perf_counter() - start
    return {
        "batch_count": len(loader),
        "sample_count": seen,
        "elapsed_s": elapsed,
    }


def run_pipeline_benchmark(
    manifest_path: Path,
    *,
    args: argparse.Namespace,
    augment_cfg: dict[str, Any],
    spec_cfg: Any,
    spec_path: str,
    benchmark_device: torch.device,
    codec_preset: str | None,
    probabilities: CodecAugmentProbabilities,
    codec_min_duration_sec: float | None,
) -> tuple[dict[str, list[float]] | None, dict[str, float] | None]:
    """Run one dataset/dataloader benchmark pass for a single codec/probability setup."""
    dataset = ParquetAudioDataset(
        manifest_path=manifest_path,
        sample_rate=args.sample_rate,
        trim_num_samples=args.trim_num_samples,
        codec_preset=codec_preset,
        codec_backend=args.codec_backend,
        augment_cfg=augment_cfg,
        probabilities=probabilities,
        codec_min_duration_sec=codec_min_duration_sec,
        fallback_on_decode_error=True,
    )
    if dataset.codec is not None:
        print(f"[INFO] Codec backend -> requested={args.codec_backend}, selected={dataset.codec.selected_backend()}")
    encoder = ReDimNet2Encoder(model_name="b0", pretrained=False)
    collator = None
    device_frontend = None
    collate_fn: Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]
    if spec_path == "cpu_collator":
        collator = SpectrogramCollator(spec_frontend=encoder.spec_frontend, augment_cfg=spec_cfg, training=True)
        collate_fn = collator
    elif spec_path == "gpu_preprocess":
        device_frontend = clone_frozen_frontend(encoder.spec_frontend, device=benchmark_device)
        collate_fn = collate_waveform_batch
    else:
        raise ValueError(f"unsupported spec_path={spec_path}")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    dataset_stats: dict[str, list[float]] | None = None
    loader_stats: dict[str, float] | None = None

    if args.phase in {"all", "dataset"}:
        dataset_stats = benchmark_dataset_items(dataset, min(args.iterations, len(dataset)))
        print("[RESULT] Dataset stage timings (mean/p50/p95/p99 ms, wall-clock over function calls)")
        for k in [
            "getitem_total",
            "read_row",
            "decode",
            "trim_long",
            "to_mono",
            "pad_or_trim",
            "codec",
            "codec_roundtrip_total",
            "codec_ffmpeg_total",
            "codec_ffmpeg_pcm_in",
            "codec_ffmpeg_encode",
            "codec_ffmpeg_decode",
            "codec_pydub_total",
            "augment_waveform",
        ]:
            report_stage(k, dataset_stats.get(k, []))

    if args.phase in {"all", "loader"}:
        print(f"[INFO] Running dataloader throughput benchmark ({spec_path})...")
        timings: dict[str, list[float]] = {}
        if args.num_workers > 0:
            if args.num_batches < len(loader):
                num_batches = args.num_batches
            else:
                num_batches = len(loader)
            if spec_path == "cpu_collator" and collator is not None:
                orig_frontend = wrap_timed(
                    "frontend",
                    collator.spec_frontend,
                    "forward",
                    timings,
                )
            elif spec_path == "gpu_preprocess" and device_frontend is not None:
                orig_frontend = wrap_timed(
                    "frontend",
                    device_frontend,
                    "forward",
                    timings,
                    sync_device_for_timing=benchmark_device,
                )
            else:
                orig_frontend = None
            start = time.perf_counter()
            it = iter(loader)
            count = 0
            for _ in range(num_batches):
                batch = next(it)
                _ = prepare_batch_for_model(
                    batch,
                    spec_path=spec_path,
                    device=benchmark_device,
                    timings=timings,
                    spec_frontend=device_frontend,
                    spec_cfg=spec_cfg,
                )
                count += args.batch_size
            elapsed = time.perf_counter() - start
            loader_stats = {
                "batch_count": num_batches,
                "sample_count": count,
                "elapsed_s": elapsed,
            }
            if orig_frontend is not None:
                target_frontend = collator.spec_frontend if collator is not None else device_frontend
                restore_timed(target_frontend, "forward", orig_frontend)
            print(
                f"[RESULT] Ready batch ({spec_path}) -> {count / elapsed:.2f} samples/s, "
                f"{elapsed / num_batches * 1000:.3f} ms/batch"
            )
            report_stage("ready_batch", timings.get("ready_batch", []))
            report_stage("frontend", timings.get("frontend", []))
        else:
            if spec_path == "cpu_collator" and collator is not None:
                orig_frontend = wrap_timed("frontend", collator.spec_frontend, "forward", timings)
            elif spec_path == "gpu_preprocess" and device_frontend is not None:
                orig_frontend = wrap_timed(
                    "frontend",
                    device_frontend,
                    "forward",
                    timings,
                    sync_device_for_timing=benchmark_device,
                )
            else:
                orig_frontend = None

            def timed_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
                start = time.perf_counter()
                out = collate_fn(batch)
                timings.setdefault("collate", []).append(time.perf_counter() - start)
                return out

            orig_collate_fn = loader.collate_fn
            loader.collate_fn = timed_collate
            start = time.perf_counter()
            seen = 0
            it = iter(loader)
            num_batches = min(args.num_batches, len(loader))
            for _ in range(num_batches):
                batch = next(it)
                batch = prepare_batch_for_model(
                    batch,
                    spec_path=spec_path,
                    device=benchmark_device,
                    timings=timings,
                    spec_frontend=device_frontend,
                    spec_cfg=spec_cfg,
                )
                seen += batch["labels"].shape[0]
            elapsed = time.perf_counter() - start
            loader_stats = {
                "batch_count": num_batches,
                "sample_count": seen,
                "elapsed_s": elapsed,
            }
            loader.collate_fn = orig_collate_fn
            if orig_frontend is not None:
                target_frontend = collator.spec_frontend if collator is not None else device_frontend
                restore_timed(target_frontend, "forward", orig_frontend)

            print(
                f"[RESULT] Ready batch ({spec_path}, num_workers=0) -> "
                f"{loader_stats['sample_count'] / loader_stats['elapsed_s']:.2f} samples/s, "
                f"{loader_stats['elapsed_s'] / max(1, loader_stats['batch_count']) * 1000:.3f} ms/batch"
            )
            report_stage("collate", timings.get("collate", []))
            report_stage("ready_batch", timings.get("ready_batch", []))
            report_stage("frontend", timings.get("frontend", []))
    return dataset_stats, loader_stats


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ensure_torchaudio_io_compat()

    benchmark_device = torch.device(args.benchmark_device)
    if benchmark_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA benchmark requested via --benchmark-device={args.benchmark_device}, but CUDA is unavailable")
    if args.spec_path == "compare" and args.codec_baseline:
        raise ValueError("--codec-baseline is not supported with --spec-path compare")

    if args.codec_preset is not None and args.codec_preset.lower() in {"none", "null", "disabled", ""}:
        args.codec_preset = None

    keep_original = args.keep_original
    if keep_original is None:
        keep_original = 1.0 - (args.codec_only + args.augment_only + args.codec_then_augment)
    if keep_original < 0.0:
        raise ValueError("Transform probabilities must sum to <= 1.0")

    augment_cfg = build_augment_cfg(args.waveform_speed_augment)

    work_dir = args.work_dir
    clean_up = False
    manifest_path = args.manifest
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    if manifest_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="asv_bench_", dir="/tmp")
        tmp = Path(temp_dir.name)
        manifest_path = build_synthetic_manifest(
            tmp,
            n=args.num_utterances,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )
        work_dir = tmp
        clean_up = True
    else:
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")

    omega = OmegaConf.create(
        {
            "augment": augment_cfg,
            "data": {
                "spec_augment": {
                    "level_p": args.spec_level_p,
                    "freq_mask": {"p": 0.1, "max_width": 8},
                    "time_mask": {"p": 0.1, "max_width": 10},
                    "cutmix": {"p": 0.0, "min_frames": 5, "max_frames": 25},
                }
            },
            "spec_augment": {},
        }
    )
    spec_cfg = build_spec_augment_cfg(omega)

    def run_for_spec_path(
        spec_path: str,
        *,
        codec_preset: str | None,
        probabilities: CodecAugmentProbabilities,
    ) -> tuple[dict[str, list[float]] | None, dict[str, float] | None]:
        print(f"[INFO] Running configuration for spec_path={spec_path}")
        return run_pipeline_benchmark(
            manifest_path=manifest_path,
            args=args,
            augment_cfg=augment_cfg,
            spec_cfg=spec_cfg,
            spec_path=spec_path,
            benchmark_device=benchmark_device,
            codec_preset=codec_preset,
            codec_min_duration_sec=args.codec_min_duration_sec,
            probabilities=probabilities,
        )

    baseline_run = None
    if args.phase in {"all", "dataset", "loader"} and args.codec_baseline and args.codec_preset is not None:
        print("[INFO] Running codec-disabled baseline (keep_all)")
        baseline_run = run_for_spec_path(
            args.spec_path,
            codec_preset=None,
            probabilities=CodecAugmentProbabilities(
                keep_original=1.0,
                codec_only=0.0,
                augment_only=0.0,
                codec_then_augment=0.0,
            ),
        )
        print("[INFO] Baseline complete")

    target_probabilities = CodecAugmentProbabilities(
        keep_original=keep_original,
        codec_only=args.codec_only,
        augment_only=args.augment_only,
        codec_then_augment=args.codec_then_augment,
    )
    if args.spec_path == "compare":
        cpu_run = run_for_spec_path("cpu_collator", codec_preset=args.codec_preset, probabilities=target_probabilities)
        gpu_run = run_for_spec_path("gpu_preprocess", codec_preset=args.codec_preset, probabilities=target_probabilities)
        if cpu_run[1] is not None and gpu_run[1] is not None and cpu_run[1]["elapsed_s"] > 0 and gpu_run[1]["elapsed_s"] > 0:
            cpu_loader = cpu_run[1]
            gpu_loader = gpu_run[1]
            cpu_sps = cpu_loader["sample_count"] / cpu_loader["elapsed_s"]
            gpu_sps = gpu_loader["sample_count"] / gpu_loader["elapsed_s"]
            cpu_ms = cpu_loader["elapsed_s"] / cpu_loader["batch_count"] * 1000
            gpu_ms = gpu_loader["elapsed_s"] / gpu_loader["batch_count"] * 1000
            print(f"[RESULT] GPU preprocess speedup: {gpu_sps:.2f}/{cpu_sps:.2f} samples/s = {gpu_sps / cpu_sps:.2f}x")
            print(f"[RESULT] GPU preprocess latency: {gpu_ms:.3f} vs {cpu_ms:.3f} ms/batch")
        target_run = gpu_run
    else:
        target_run = run_for_spec_path(args.spec_path, codec_preset=args.codec_preset, probabilities=target_probabilities)

    if args.codec_baseline and baseline_run[1] is not None and target_run[1] is not None:
        base_loader = baseline_run[1]
        target_loader = target_run[1]
        if base_loader["batch_count"] > 0 and base_loader["elapsed_s"] > 0 and target_loader["batch_count"] > 0:
            base_sps = base_loader["sample_count"] / base_loader["elapsed_s"]
            target_sps = target_loader["sample_count"] / target_loader["elapsed_s"]
            print(
                f"[RESULT] Baseline vs target throughput: "
                f"{target_sps:.2f}/{base_sps:.2f} samples/s = {target_sps / base_sps:.2f}x"
            )
            base_ms = base_loader["elapsed_s"] / base_loader["batch_count"] * 1000
            target_ms = target_loader["elapsed_s"] / target_loader["batch_count"] * 1000
            print(f"[RESULT] Baseline vs target latency: {target_ms:.3f} vs {base_ms:.3f} ms/batch")

    if clean_up and temp_dir is not None:
        if args.clean_work_dir:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
