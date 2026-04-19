#!/usr/bin/env python3
"""Visualize utterance count distribution per speaker from a manifest.

The script accepts an ASV parquet manifest and writes a **PNG** file with two
panels: a bar plot over utterance counts in ``[0, N)`` (default ``N=100``:
bins 0..99, each bin is an exact count; speakers with ``≥ N`` clips are
omitted so the tail does not swamp the figure). Unless disabled, a top-K
bar chart is included.
Uses the Agg backend only (no GUI); counts rows via
``pyarrow.compute.value_counts`` to limit RAM use on large manifests.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    print("This script requires pyarrow. Install with: python -m pip install pyarrow", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import matplotlib

    # Headless-safe: avoid Tk/Qt trying to open a display (hangs / fork storms on SSH).
    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    import numpy as np
except ImportError as e:  # pragma: no cover
    print(
        "This script requires matplotlib (and numpy, pulled in by matplotlib). "
        "Install with: python -m pip install matplotlib",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per-speaker utterance distribution from an ASV manifest."
    )
    p.add_argument("manifest_path", type=Path, help="Path to a parquet manifest.")
    p.add_argument(
        "--speaker-column",
        default="speaker_id",
        help="Column containing speaker identifier (default: speaker_id).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top speakers to show in ranking plot (default: 30).",
    )
    p.add_argument(
        "--min-utterances",
        type=int,
        default=1,
        help="Ignore speakers with fewer than this many utterances (default: 1).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .png path (extension forced to .png). Default: <manifest_stem>_speaker_distribution.png.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for saved output (default: 150).",
    )
    p.add_argument(
        "--no-show-top-k",
        action="store_true",
        help="Skip the top-k bar chart and only render the histogram.",
    )
    p.add_argument(
        "--utterance-hist-cap",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Half-open range [0, N): one bar per count k = 0 .. N-1 (exact). "
            "Speakers with ≥ N utterances are omitted from this panel (no tail). "
            "Example: N=100 → bins 0–99 only. Use 0 for a full-range histogram "
            "(see --hist-max-bins). Use N=101 if you need an exact bin for 100 "
            "but still omit counts ≥ 101."
        ),
    )
    p.add_argument(
        "--hist-max-bins",
        type=int,
        default=200,
        metavar="N",
        help=(
            "With --utterance-hist-cap 0: cap bin count when counts span a wide "
            "range (default: 200). Ignored when --utterance-hist-cap > 0 "
            "(capped / half-open mode)."
        ),
    )
    return p.parse_args()


def _png_output_path(path: Path) -> Path:
    """Always write PNG; avoid accidental non-PNG paths."""
    path = path.expanduser()
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    return path.resolve()


def _to_text_speaker_id(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    return str(value)


def _merge_value_counts_into(counter: Counter[str], arr: pa.Array) -> None:
    """Aggregate counts without ``to_pylist()`` on every manifest row (saves RAM)."""
    vc = pc.value_counts(arr)
    if len(vc) == 0:
        return
    vals = vc.field(0)
    cnts = vc.field(1)
    for value, raw_count in zip(vals.to_pylist(), cnts.to_pylist()):
        sid = _to_text_speaker_id(value).strip()
        if sid:
            counter[sid] += int(raw_count)


def read_speaker_counts(manifest_path: Path, speaker_column: str) -> Counter[str]:
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        raise SystemExit(2)

    parquet_file = pq.ParquetFile(manifest_path)
    if speaker_column not in parquet_file.schema.names:
        print(
            f"Column '{speaker_column}' not found in {manifest_path}.",
            file=sys.stderr,
        )
        print(f"Available columns: {', '.join(parquet_file.schema.names)}", file=sys.stderr)
        raise SystemExit(2)

    counter: Counter[str] = Counter()
    for row_group_id in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(row_group_id, columns=[speaker_column])
        col = table.column(0)
        chunks = col.chunks if isinstance(col, pa.ChunkedArray) else [col]
        for chunk in chunks:
            _merge_value_counts_into(counter, chunk)
    return counter


def _print_summary(utterance_counts: list[int]) -> None:
    sorted_counts = sorted(utterance_counts)
    total_speakers = len(sorted_counts)
    total_utterances = sum(sorted_counts)
    min_count = sorted_counts[0]
    max_count = sorted_counts[-1]
    p50 = statistics.median(sorted_counts)
    p90 = _percentile(sorted_counts, 0.90)
    p99 = _percentile(sorted_counts, 0.99)
    print(f"speakers: {total_speakers}")
    print(f"utterances: {total_utterances}")
    print(f"min/median/90p/99p/max utterances per speaker: {min_count}/{p50}/{p90:.2f}/{p99:.2f}/{max_count}")


def _percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    n = len(sorted_values)
    pos = q * (n - 1)
    left = int(math.floor(pos))
    right = min(left + 1, n - 1)
    frac = pos - left
    return float(sorted_values[left] * (1.0 - frac) + sorted_values[right] * frac)


def make_plots(
    speaker_counts: Counter[str],
    output: Path,
    top_k: int,
    dpi: int,
    hide_top_k: bool,
    hist_max_bins: int,
    utterance_hist_cap: int,
) -> Path:
    utterance_counts = sorted(speaker_counts.values())
    num_top = min(top_k, len(utterance_counts))
    top_k_items = speaker_counts.most_common(top_k)

    fig_width = 14 if not hide_top_k else 8
    n_cols = 1 if hide_top_k else 2
    fig = Figure(figsize=(fig_width, 6))
    ax_hist = fig.add_subplot(1, n_cols, 1)

    # Distribution: how many speakers have X utterances.
    arr = np.asarray(utterance_counts, dtype=np.int64)
    if utterance_hist_cap > 0:
        cap_n = int(utterance_hist_cap)
        if cap_n < 1:
            print("--utterance-hist-cap must be >= 1 when used.", file=sys.stderr)
            raise SystemExit(2)
        # [0, cap_n): bin k = exactly k utterances; ci >= cap_n omitted (no tail mass on the right).
        n_bins = cap_n
        buckets = np.zeros(n_bins, dtype=np.int64)
        excluded = 0
        for c in arr.flat:
            ci = int(c)
            if 0 <= ci < cap_n:
                buckets[ci] += 1
            else:
                excluded += 1
        x_plot = np.arange(n_bins, dtype=np.float64)
        ax_hist.bar(
            x_plot,
            buckets.astype(np.float64),
            width=1.0,
            align="edge",
            edgecolor="white",
            linewidth=0.4,
        )
        hi = int(buckets.max()) if n_bins else 0
        argmax = int(buckets.argmax()) if n_bins else 0
        print(
            f"Bar panel: counts in [0, {cap_n}) only; "
            f"excluded {excluded} speaker(s) with ≥{cap_n} utterances; "
            f"max bin height={hi} (at count={argmax}).",
            file=sys.stderr,
            flush=True,
        )
        ax_hist.set_title(
            f"Utterances per speaker (exact counts in [0, {cap_n}); tail excluded)"
        )
        ax_hist.set_xlabel("Utterances per speaker")
        ax_hist.set_xlim(0, cap_n)
        step = max(1, cap_n // 10)
        ticks = list(range(0, cap_n, step))
        if ticks[-1] != cap_n - 1:
            ticks.append(cap_n - 1)
        ax_hist.set_xticks(ticks)
        ax_hist.set_xticklabels([str(t) for t in ticks])
    else:
        x_min = int(arr.min())
        x_max = int(arr.max())
        span = x_max - x_min + 1
        cap = max(1, int(hist_max_bins))
        if span <= cap:
            bins = np.arange(x_min, x_max + 2, dtype=np.float64)
        else:
            bins = np.linspace(float(x_min), float(x_max + 1), num=cap + 1)
        ax_hist.hist(arr, bins=bins, edgecolor="white")
        ax_hist.set_title("Utterances per speaker")
        ax_hist.set_xlabel("Number of utterances")
        ax_hist.set_xlim(left=max(0, x_min - 1), right=x_max + 1)
    ax_hist.set_ylabel("Number of speakers")
    ax_hist.grid(alpha=0.2)

    if not hide_top_k:
        ax_top = fig.add_subplot(1, n_cols, 2)
        if num_top > 0:
            speaker_ids, counts = zip(*reversed(top_k_items))  # type: ignore[var-annotated]
            ax_top.barh(range(num_top), counts, color="#4c72b0")
            ax_top.set_yticks(range(num_top))
            ax_top.set_yticklabels(speaker_ids)
            ax_top.invert_yaxis()
            ax_top.set_title(f"Top {num_top} speakers by utterance count")
            ax_top.set_xlabel("Number of utterances")
            ax_top.grid(axis="x", alpha=0.2)
        else:
            ax_top.set_title("No speakers after filtering")

    output = _png_output_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        output,
        dpi=dpi,
        format="png",
        metadata={"Software": "ASV visualize_speaker_distribution"},
    )
    return output


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_path.expanduser().resolve()
    speaker_counts = read_speaker_counts(manifest_path, args.speaker_column)
    if not speaker_counts:
        print("No speaker entries were found after parsing the manifest.", file=sys.stderr)
        raise SystemExit(1)

    filtered = Counter(
        {
            speaker_id: count
            for speaker_id, count in speaker_counts.items()
            if count >= args.min_utterances and speaker_id
        }
    )
    utterance_counts = list(filtered.values())

    if not filtered:
        print(
            "No speakers remain after applying min-utterances filter: "
            f"{args.min_utterances}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    _print_summary(sorted(utterance_counts))

    if args.output is None:
        output = manifest_path.with_name(f"{manifest_path.stem}_speaker_distribution.png")
    else:
        output = _png_output_path(args.output)

    out_path = make_plots(
        filtered,
        output=output,
        top_k=args.top_k,
        dpi=args.dpi,
        hide_top_k=args.no_show_top_k,
        hist_max_bins=args.hist_max_bins,
        utterance_hist_cap=args.utterance_hist_cap,
    )
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
