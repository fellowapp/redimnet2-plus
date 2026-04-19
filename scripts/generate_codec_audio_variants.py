#!/usr/bin/env python3
"""Generate one codec-processed output for each registered codec preset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydub import AudioSegment

from asv.codec.availability import check_preset_available
from asv.codec.conversion import segment_to_float_pcm
from asv.codec.pipeline import CodecPipeline, CodecUnavailableError
from asv.codec.registry import list_presets

_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".mp4",
    ".aac",
    ".amr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a single input audio file and write one output WAV per "
            "registered codec preset into output_dir."
        )
    )
    parser.add_argument(
        "input_audio",
        type=Path,
        help="Path to a single input audio file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory where codec-processed WAV files are saved.",
        default=Path("../outputs/codec_tests"),
    )
    return parser.parse_args()


def _load_waveform(path: Path) -> tuple[object, int]:
    seg = AudioSegment.from_file(path)
    waveform, sample_rate = segment_to_float_pcm(seg)
    return waveform, sample_rate


def _write_wav(path: Path, waveform, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(path, waveform.T, sample_rate, format="WAV", subtype="PCM_16")


def main() -> None:
    args = parse_args()
    input_audio = args.input_audio.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_audio.is_file():
        print(f"Input audio file does not exist: {input_audio}", file=sys.stderr)
        raise SystemExit(2)
    if input_audio.suffix.lower() not in _AUDIO_EXTENSIONS:
        print(f"Unsupported input audio extension: {input_audio.suffix}", file=sys.stderr)
        raise SystemExit(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    waveform, sample_rate = _load_waveform(input_audio)
    input_stem = input_audio.stem

    presets = list_presets()
    ok_count = 0
    skipped_count = 0
    failed_count = 0

    for preset_name in presets:
        output_path = output_dir / f"{input_stem}__{preset_name}.wav"
        if not check_preset_available(preset_name):
            skipped_count += 1
            print(f"[SKIP] {preset_name}: unavailable on this system")
            continue

        try:
            pipeline = CodecPipeline(preset_name, check_available=False)
            decoded, out_sr = pipeline.roundtrip(waveform, sample_rate)
            _write_wav(output_path, decoded, out_sr)
            ok_count += 1
            print(f"[OK]   {preset_name} -> {output_path.name}")
        except CodecUnavailableError as error:
            failed_count += 1
            print(f"[FAIL] {preset_name}: {error}")
        except Exception as error:  # pragma: no cover
            failed_count += 1
            print(f"[FAIL] {preset_name}: unexpected error: {error}")

    print(
        f"Done. presets={len(presets)} generated={ok_count} skipped={skipped_count} failed={failed_count}"
    )


if __name__ == "__main__":
    main()
