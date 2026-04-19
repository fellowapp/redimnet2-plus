#!/usr/bin/env python3
"""Concatenate many training manifests with dataset-name speaker prefixes."""

from __future__ import annotations

import argparse
from pathlib import Path

from asv.manifest.concat import concat_manifests


def _parse_manifest_spec(spec: str) -> tuple[str, Path]:
    dataset_name, sep, raw_path = spec.partition("=")
    dataset_name = dataset_name.strip()
    raw_path = raw_path.strip()
    if not sep or not dataset_name or not raw_path:
        raise argparse.ArgumentTypeError(
            "Each --manifest must look like "
            "dataset_name=/path/to/manifest.parquet"
        )
    return dataset_name, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate multiple ASV manifests into one training manifest. "
            "Each source gets a dataset-name speaker prefix such as "
            "'cnceleb2:' or '3dspeaker:' so classes stay distinct."
        )
    )
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--manifest",
        dest="manifests",
        action="append",
        type=_parse_manifest_spec,
        required=True,
        help=(
            "Dataset spec in the form dataset_name=/path/to/manifest.parquet. "
            "Repeat once per input dataset."
        ),
    )
    parser.add_argument(
        "--raw-manifest",
        dest="raw_manifests",
        action="append",
        type=Path,
        default=[],
        help=(
            "Manifest parquet to include without adding a dataset prefix. "
            "Use this for an already-prefixed merged manifest."
        ),
    )
    parser.add_argument(
        "--prefix-separator",
        default=":",
        help=(
            "Separator used between the dataset name and original speaker_id."
        ),
    )
    parser.add_argument(
        "--allow-speaker-overlap",
        action="store_true",
        help=(
            "Allow overlapping speaker_id values after prefixing. This should "
            "rarely be needed."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path.expanduser().resolve()
    manifest_specs = [
        (path.expanduser().resolve(), "")
        for path in args.raw_manifests
    ]
    manifest_specs.extend(
        (
            path.expanduser().resolve(),
            f"{dataset_name}{args.prefix_separator}",
        )
        for dataset_name, path in args.manifests
    )

    total_rows, num_classes = concat_manifests(
        manifest_specs,
        output_path,
        allow_speaker_overlap=args.allow_speaker_overlap,
    )

    print(f"Wrote {output_path} ({total_rows} rows)")
    print(f"num_classes={num_classes}")


if __name__ == "__main__":
    main()
