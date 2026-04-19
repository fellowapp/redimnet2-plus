#!/usr/bin/env python3
"""Generate N augmented waveforms using the same ParquetAudioDataset pipeline as training.

Uses Hydra config (data + augment + train.fallback_on_decode_error) and the same
codec/augment probability policy as ``train.py``. Writes WAV files named with the
sampled mode: ``keep``, ``codec``, ``augment``, ``codec_augment`` (internal keys
from :meth:`CodecAugmentProbabilities.choose_mode`).
"""

from __future__ import annotations

import argparse
import random
from typing import Any
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from asv.data import CodecAugmentProbabilities, ParquetAudioDataset
from asv.manifest.schema import SCHEMA as MANIFEST_SCHEMA


def _ensure_torchaudio_compat() -> None:
    """Patch torchaudio.info/load when missing (torch_audiomentations uses them for RIR/MUSAN)."""
    if hasattr(torchaudio, "info"):
        return

    def _sf_load(
        path: str,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        start = int(frame_offset)
        frames = -1 if num_frames is None or num_frames < 0 else int(num_frames)
        arr, sr = sf.read(path, dtype="float32", always_2d=True, start=start, frames=frames)
        tensor = torch.from_numpy(arr.T.copy())
        return tensor, int(sr)

    class _Info:
        def __init__(self, sample_rate: int, num_frames: int, num_channels: int):
            self.sample_rate = int(sample_rate)
            self.num_frames = int(num_frames)
            self.num_channels = int(num_channels)

    def _info(path: str) -> _Info:
        arr, sr = sf.read(path, dtype="float32", always_2d=True)
        return _Info(sample_rate=sr, num_frames=arr.shape[0], num_channels=arr.shape[1])

    setattr(torchaudio, "info", _info)
    setattr(torchaudio, "load", _sf_load)
    setattr(torchaudio, "load_with_torchcodec", _sf_load)


def _paths_from_run_sh(run_sh: Path) -> tuple[str | None, str | None]:
    if not run_sh.is_file():
        return None, None
    text = run_sh.read_text(encoding="utf-8")
    musan_m = re.search(r'^MUSAN_PATH="([^"]+)"', text, flags=re.MULTILINE)
    rirs_m = re.search(r'^RIRS_PATH="([^"]+)"', text, flags=re.MULTILINE)
    return (
        musan_m.group(1) if musan_m else None,
        rirs_m.group(1) if rirs_m else None,
    )


def _write_single_row_manifest(
    audio_path: Path,
    out_path: Path,
) -> None:
    path_str = str(audio_path.expanduser().resolve())
    rel = audio_path.as_posix().split("/")[-4:]  # best-effort rel
    rel_path = "/".join(rel) if rel else audio_path.name
    speaker_id = audio_path.parent.parent.parent.name if len(audio_path.parts) >= 4 else "spk"
    video_id = audio_path.parent.name
    seg = audio_path.stem
    table = pa.table(
        {
            "audio_path": [path_str],
            "rel_path": [rel_path],
            "speaker_id": [speaker_id],
            "video_id": [video_id],
            "segment_id": [seg],
            "audio_id": [f"{speaker_id}/{video_id}/{seg}"],
            "duration_sec": [0.0],
        },
        schema=MANIFEST_SCHEMA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, row_group_size=1)


def _mode_to_label(mode: str) -> str:
    return {
        "keep": "keep_original",
        "codec": "codec_only",
        "augment": "augment_only",
        "codec_augment": "codec_then_augment",
    }.get(mode, mode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preview training-style augmentations (codec + waveform aug) from a single audio file."
    )
    p.add_argument("input_audio", type=Path, help="Path to one audio file (e.g. WAV).")
    p.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=8,
        help="How many augmented samples to draw (each draw samples mode + aug stochastically).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_augment_preview"),
        help="Directory under repo root for outputs.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible mode/aug draws.")
    p.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Optional Hydra overrides (e.g. data.codec_preset=amr_nb data.sample_rate=16000).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_torchaudio_compat()
    repo_root = Path(__file__).resolve().parent.parent
    input_audio = args.input_audio.expanduser().resolve()
    if not input_audio.is_file():
        print(f"Input audio not found: {input_audio}", file=sys.stderr)
        raise SystemExit(2)

    manifest_path = repo_root / "outputs" / "dataset_augment_preview_manifest.parquet"
    _write_single_row_manifest(input_audio, manifest_path)

    musan_path, rirs_path = _paths_from_run_sh(repo_root / "run.sh")

    overrides = [
        f"data.manifest_path={manifest_path}",
        "output_dir=/tmp/asv_dataset_augment_preview",
        *args.hydra_overrides,
    ]

    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs")):
        cfg = compose(config_name="config", overrides=overrides)

    OmegaConf.resolve(cfg)
    # Match run.sh paths; use plain dicts so Hydra instantiate gets real lists (not ListConfig).
    augment_cfg: dict = OmegaConf.to_container(cfg.augment, resolve=True)
    if musan_path:
        augment_cfg["waveform_levels"][1]["transforms"][1]["background_paths"] = [musan_path]
    if rirs_path:
        augment_cfg["waveform_levels"][2]["transforms"][0]["ir_paths"] = [rirs_path]
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    probabilities = CodecAugmentProbabilities(
        **OmegaConf.to_container(cfg.data.transform_probabilities, resolve=True)
    )
    dataset = ParquetAudioDataset(
        manifest_path=cfg.data.manifest_path,
        sample_rate=int(cfg.data.sample_rate),
        trim_num_samples=cfg.data.trim_num_samples,
        codec_preset=cfg.data.codec_preset,
        augment_cfg=augment_cfg,
        probabilities=probabilities,
        fallback_on_decode_error=bool(cfg.train.fallback_on_decode_error),
    )

    out_dir = (repo_root / args.output_dir).resolve() / input_audio.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_audio.stem
    sr = int(cfg.data.sample_rate)

    for i in range(int(args.num_samples)):
        item, mode = dataset.sample_with_mode(0)
        w = item["waveform"].detach().cpu().numpy().astype(np.float32)
        label = _mode_to_label(mode)
        fname = f"{stem}__train_pipeline__{i:04d}__{label}.wav"
        out_path = out_dir / fname
        sf.write(str(out_path), w, samplerate=sr, subtype="PCM_16")
        print(f"[OK] {i + 1}/{args.num_samples} mode={mode} ({label}) -> {out_path}")

    print(f"Done. output_dir={out_dir}")


if __name__ == "__main__":
    main()
