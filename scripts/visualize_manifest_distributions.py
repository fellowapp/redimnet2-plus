#!/usr/bin/env python3
"""Visualize speaker-count and duration distributions."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    print(
        "This script requires pyarrow. "
        "Install with: python -m pip install pyarrow",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
except ImportError as e:  # pragma: no cover
    print(
        "This script requires matplotlib. "
        "Install with: python -m pip install matplotlib",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create bar plots for speaker utterance counts and durations."
        )
    )
    parser.add_argument("manifest_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default: "
            "<manifest_stem>_speaker_duration_distributions.png"
        ),
    )
    parser.add_argument(
        "--speaker-column",
        default="speaker_id",
        help="Column containing the speaker ID (default: speaker_id).",
    )
    parser.add_argument(
        "--duration-column",
        default="duration_sec",
        help=(
            "Column containing audio duration in seconds "
            "(default: duration_sec)."
        ),
    )
    parser.add_argument(
        "--utterance-max",
        type=int,
        default=50,
        help="Upper bound for exact speaker-count bars (default: 50).",
    )
    parser.add_argument(
        "--duration-bin-width",
        type=float,
        default=1.0,
        help="Duration histogram bin width in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=0.0,
        help="Lower bound (inclusive) for duration plot bins in seconds (default: 0).",
    )
    parser.add_argument(
        "--duration-max",
        type=float,
        default=None,
        help=(
            "Upper bound (inclusive) for duration plot bins in seconds. "
            "Default: no limit."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for the saved PNG (default: 150).",
    )
    return parser.parse_args()


def _png_output_path(path: Path) -> Path:
    path = path.expanduser()
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    return path.resolve()


def _to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return str(value)


def _merge_value_counts_into(counter: Counter[str], arr: pa.Array) -> None:
    value_counts = pc.value_counts(arr)
    if len(value_counts) == 0:
        return
    values = value_counts.field(0)
    counts = value_counts.field(1)
    for value, raw_count in zip(values.to_pylist(), counts.to_pylist()):
        speaker_id = _to_text(value).strip()
        if speaker_id:
            counter[speaker_id] += int(raw_count)


def read_speaker_counts(manifest_path: Path, speaker_column: str) -> Counter[str]:
    parquet_file = pq.ParquetFile(manifest_path)
    if speaker_column not in parquet_file.schema.names:
        raise ValueError(f"Column not found: {speaker_column}")

    counter: Counter[str] = Counter()
    for row_group_id in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_id,
            columns=[speaker_column],
        )
        column = table.column(0)
        chunks = (
            column.chunks
            if isinstance(column, pa.ChunkedArray)
            else [column]
        )
        for chunk in chunks:
            _merge_value_counts_into(counter, chunk)
    return counter


def read_duration_histogram(
    manifest_path: Path,
    duration_column: str,
    bin_width: float,
    *,
    duration_min: float,
    duration_max: float | None,
) -> tuple[Counter[int], int, int]:
    parquet_file = pq.ParquetFile(manifest_path)
    if duration_column not in parquet_file.schema.names:
        raise ValueError(f"Column not found: {duration_column}")

    bins: Counter[int] = Counter()
    missing = 0
    present = 0
    for row_group_id in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group_id,
            columns=[duration_column],
        )
        for value in table.column(0).to_pylist():
            if value is None:
                missing += 1
                continue
            float_value = float(value)
            if float_value < duration_min:
                continue
            if duration_max is not None and float_value > duration_max:
                continue
            bucket = int(math.floor(float_value / bin_width))
            bins[bucket] += 1
            present += 1
    return bins, present, missing


def make_plots(
    speaker_counts: Counter[str],
    duration_bins: Counter[int],
    duration_present: int,
    duration_missing: int,
    output: Path,
    *,
    utterance_max: int,
    duration_bin_width: float,
    dpi: int,
) -> Path:
    fig = Figure(figsize=(16, 6))
    ax_utt = fig.add_subplot(1, 2, 1)
    ax_dur = fig.add_subplot(1, 2, 2)

    utterance_counts = list(speaker_counts.values())
    exact_buckets = [0] * utterance_max
    excluded = 0
    for count in utterance_counts:
        if 1 <= count <= utterance_max:
            exact_buckets[count - 1] += 1
        else:
            excluded += 1

    x_utt = list(range(1, utterance_max + 1))
    ax_utt.bar(x_utt, exact_buckets, width=0.9)
    ax_utt.set_title(
        f"Speakers by utterance count (exact counts in [1, {utterance_max}])"
    )
    ax_utt.set_xlabel("Utterances per speaker")
    ax_utt.set_ylabel("Number of speakers")
    ax_utt.set_xticks(list(range(1, utterance_max + 1, 5)))
    ax_utt.grid(axis="y", alpha=0.2)
    if excluded:
        ax_utt.text(
            0.99,
            0.98,
            f"Excluded speakers outside range: {excluded}",
            transform=ax_utt.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

    if duration_present > 0:
        bucket_ids = sorted(duration_bins)
        x_dur = [bucket * duration_bin_width for bucket in bucket_ids]
        y_dur = [duration_bins[bucket] for bucket in bucket_ids]
        ax_dur.bar(x_dur, y_dur, width=duration_bin_width * 0.95, align="edge")
        ax_dur.set_xlabel(
            f"Duration bin start (seconds, width={duration_bin_width:g}s)"
        )
        ax_dur.set_ylabel("Number of utterances")
        ax_dur.grid(axis="y", alpha=0.2)
        max_tick = int(max(x_dur) + duration_bin_width)
        ax_dur.set_xticks(list(range(0, max_tick + 1, 5)))
    else:
        ax_dur.text(
            0.5,
            0.5,
            "No non-null durations available",
            transform=ax_dur.transAxes,
            ha="center",
            va="center",
        )
        ax_dur.set_xlabel("Duration (seconds)")
        ax_dur.set_ylabel("Number of utterances")

    ax_dur.set_title("Audio duration distribution")
    ax_dur.text(
        0.99,
        0.98,
        "Using non-null durations only: "
        f"{duration_present}\nMissing durations: {duration_missing}",
        transform=ax_dur.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    output = _png_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, format="png")
    return output


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        raise SystemExit(2)
    if args.utterance_max < 1:
        print("--utterance-max must be >= 1", file=sys.stderr)
        raise SystemExit(2)
    if args.duration_bin_width <= 0:
        print("--duration-bin-width must be > 0", file=sys.stderr)
        raise SystemExit(2)
    if args.duration_min < 0:
        print("--duration-min must be >= 0", file=sys.stderr)
        raise SystemExit(2)
    if args.duration_max is not None and args.duration_max < args.duration_min:
        print(
            "--duration-max must be >= duration-min",
            file=sys.stderr,
        )
        raise SystemExit(2)

    speaker_counts = read_speaker_counts(manifest_path, args.speaker_column)
    duration_bins, duration_present, duration_missing = (
        read_duration_histogram(
            manifest_path,
            args.duration_column,
            args.duration_bin_width,
            duration_min=args.duration_min,
            duration_max=args.duration_max,
        )
    )

    if args.output is None:
        output = manifest_path.with_name(
            f"{manifest_path.stem}_speaker_duration_distributions.png"
        )
    else:
        output = _png_output_path(args.output)

    out_path = make_plots(
        speaker_counts,
        duration_bins,
        duration_present,
        duration_missing,
        output,
        utterance_max=args.utterance_max,
        duration_bin_width=args.duration_bin_width,
        dpi=args.dpi,
    )

    print(f"speakers={len(speaker_counts)}")
    print(f"duration_present={duration_present}")
    print(f"duration_missing={duration_missing}")
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
