#!/usr/bin/env python3
"""Concatenate ASV training manifests into one training-safe parquet.

By default the script fails when the input manifests share any ``speaker_id``
values, since training would then mix those rows into one class. Use
``--speaker-prefix-a`` / ``--speaker-prefix-b`` to namespace speakers from each
source manifest before writing the merged output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asv.manifest.concat import concat_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate two ASV training manifests. Fails on overlapping "
            "speaker_id values unless you prefix one or both inputs."
        )
    )
    parser.add_argument("manifest_a", type=Path)
    parser.add_argument("manifest_b", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--speaker-prefix-a",
        default="",
        help="Optional prefix added to all speaker_id values from manifest A.",
    )
    parser.add_argument(
        "--speaker-prefix-b",
        default="",
        help="Optional prefix added to all speaker_id values from manifest B.",
    )
    parser.add_argument(
        "--allow-speaker-overlap",
        action="store_true",
        help=(
            "Allow overlapping speaker_id values without prefixes. "
            "Use only if "
            "the overlaps intentionally represent the same speaker classes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_a = args.manifest_a.expanduser().resolve()
    manifest_b = args.manifest_b.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    total_rows, num_classes = concat_manifests(
        [
            (manifest_a, args.speaker_prefix_a),
            (manifest_b, args.speaker_prefix_b),
        ],
        output_path,
        allow_speaker_overlap=args.allow_speaker_overlap,
    )

    print(f"Wrote {output_path} ({total_rows} rows)")
    print(f"num_classes={num_classes}")


if __name__ == "__main__":
    main()
