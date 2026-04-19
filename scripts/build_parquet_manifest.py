#!/usr/bin/env python3
"""
Сборка Parquet-манифеста для VoxCeleb1/2, CN-Celeb, CN-Celeb2, CN-Celeb all splits,
3D-Speaker, Kryp, tidy или KeSpeech.

Схема выхода совпадает с :mod:`asv.manifest.schema`. Подробности по полям и
использованию — см. ``docs/MANIFESTS.md``.

Раскладки (как распаковывается каждый датасет):

VoxCeleb2 (типовая):
  <input_dir>/dev/aac/<idXXXXX>/<youtube_id>/<segment>.m4a
  (и/или dev/wav, test/*)
  Плоская <idXXXXX>/*.m4a под этими корнями тоже поддерживается.

VoxCeleb1 (типовая):
  <input_dir>/wav/<idXXXXX>/<youtube_id>/<segment>.wav

CN-Celeb / CN-Celeb2:
  <input_dir>/data/<idXXXXX>/<genre>-<session_id>-<utter_id>.flac

CN-Celeb all-splits:
  <input_dir>/data/<idXXXXX>/<genre>-<session_id>-<utter_id>.flac
  <input_dir>/eval/enroll/<idXXXXX>-enroll.flac
  <input_dir>/eval/test/<idXXXXX>-<genre>-<session_id>-<utter_id>.flac

3D-Speaker:
  <input_dir>/3D_SPK_<speaker_id>/<3D_SPK_<speaker_id>>_<utt>_<conditions>.wav

Kryp (speaker-id challenge):
  <input_dir>/train.csv   # columns: speaker_id, filepath
  <input_dir>/train/<speaker_id>/<NNNNN>.flac

tidy:
  <input_dir>/<speaker_id>/<lang>/<lang>_<utterance>.wav

KeSpeech:
  <input_dir>/Audio/<speaker_id>/<phase>/<utterance>.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pyarrow  # noqa: F401
except ImportError as e:  # pragma: no cover
    print(
        (
            "This script requires pyarrow. "
            "Install with: python -m pip install pyarrow"
        ),
        file=sys.stderr,
    )
    raise SystemExit(1) from e

try:
    import tqdm  # noqa: F401
except ImportError as e:  # pragma: no cover
    print(
        "This script requires tqdm. Install with: python -m pip install tqdm",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

from asv.manifest.schema import DEFAULT_DURATION_NAME
from asv.manifest.scan import (
    count_3d_speaker_speakers,
    count_cnceleb_all_speakers,
    count_cnceleb2_speakers,
    count_kespeech_speakers,
    count_kryp_speakers,
    count_tidy_speakers,
    count_voxceleb1_speakers,
    count_voxceleb2_speakers,
)
from asv.manifest.write import write_manifest



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build Parquet manifest for  VoxCeleb1/2, "
            "CN-Celeb, CN-Celeb2, CN-Celeb all splits, 3D-Speaker, "
            "Kryp, tidy, or KeSpeech."
        )
    )
    p.add_argument(
        "--dataset",
        choices=(
            "voxceleb1",
            "voxceleb2",
            "cnceleb",
            "cnceleb2",
            "cnceleb_all",
            "3dspeaker",
            "kryp",
            "tidy",
            "kespeech",
        ),
        required=True,
        help="Dataset layout to scan",
    )
    p.add_argument(
        "input_dir",
        type=Path,
        help=(
            "Dataset root (, VoxCeleb1 root containing wav/, "
            "VoxCeleb2 root containing dev/aac, CN-Celeb/CN-Celeb2 root "
            "containing data/, CN-Celeb all-splits root containing data/ "
            "and eval/, 3D-Speaker root containing 3D_SPK_* dirs, "
            "Kryp root with train.csv + train/, or tidy root containing "
            "<speaker>/<lang>/<lang>_<utt>.wav, or KeSpeech root "
            "containing Audio/<speaker>/<phase>/<utt>.wav)"
        ),
    )
    p.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write the manifest Parquet file",
    )
    p.add_argument(
        "--output-name",
        default=None,
        help=(
            "Output filename (defaults: "
            "voxceleb1_manifest.parquet / voxceleb2_manifest.parquet / "
            "cnceleb_manifest.parquet / cnceleb2_manifest.parquet / "
            "cnceleb_all_manifest.parquet / 3dspeaker_manifest.parquet / "
            "kryp_train_manifest.parquet / tidy_manifest.parquet / "
            "kespeech_manifest.parquet)"
        ),
    )
    p.add_argument(
        "--durations-parquet",
        type=Path,
        default=None,
        help="Path to durations parquet (path, duration). "
        f"<input_dir>/{DEFAULT_DURATION_NAME} if present",
    )
    p.add_argument(
        "--no-durations",
        action="store_true",
        help=(
            "Do not load or join duration column even if a default "
            "durations file exists"
        ),
    )
    p.add_argument(
        "--probe-durations",
        dest="kryp_probe_durations",
        action="store_true",
        default=None,
        help=(
            "Kryp only: fill duration_sec by reading each FLAC header via "
            "soundfile (default: on for --dataset kryp)."
        ),
    )
    p.add_argument(
        "--no-probe-durations",
        dest="kryp_probe_durations",
        action="store_false",
        help=(
            "Kryp only: skip header probing; use --durations-parquet only or "
            "leave duration_sec null."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    dataset = args.dataset

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        raise SystemExit(2)

    default_names = {
        "voxceleb1": "voxceleb1_manifest.parquet",
        "voxceleb2": "voxceleb2_manifest.parquet",
        "cnceleb": "cnceleb_manifest.parquet",
        "cnceleb2": "cnceleb2_manifest.parquet",
        "cnceleb_all": "cnceleb_all_manifest.parquet",
        "3dspeaker": "3dspeaker_manifest.parquet",
        "kryp": "kryp_train_manifest.parquet",
        "tidy": "tidy_manifest.parquet",
        "kespeech": "kespeech_manifest.parquet",
    }
    output_name = args.output_name or default_names[dataset]

    durations_path: Path | None = None
    if not args.no_durations:
        if args.durations_parquet is not None:
            durations_path = args.durations_parquet.expanduser().resolve()

    kryp_probe = False
    if dataset == "kryp":
        if args.no_durations:
            kryp_probe = False
        else:
            kryp_probe = (
                True
                if args.kryp_probe_durations is None
                else bool(args.kryp_probe_durations)
            )

    out, n_rows = write_manifest(
        input_dir,
        output_dir,
        durations_path,
        output_name,
        dataset=dataset,
        kryp_probe_durations=kryp_probe,
    )
    if dataset == "voxceleb1":
        num_classes = count_voxceleb1_speakers(input_dir)
    elif dataset == "voxceleb2":
        num_classes = count_voxceleb2_speakers(input_dir)
    elif dataset in {"cnceleb", "cnceleb2"}:
        num_classes = count_cnceleb2_speakers(input_dir)
    elif dataset == "cnceleb_all":
        num_classes = count_cnceleb_all_speakers(input_dir)
    elif dataset == "3dspeaker":
        num_classes = count_3d_speaker_speakers(input_dir)
    elif dataset == "tidy":
        num_classes = count_tidy_speakers(input_dir)
    elif dataset == "kespeech":
        num_classes = count_kespeech_speakers(input_dir)
    else:
        num_classes = count_kryp_speakers(input_dir)
    print(f"Wrote {out} ({n_rows} rows)")
    print(f"num_classes={num_classes}")
    print(f"ASV_NUM_CLASSES={num_classes}")


if __name__ == "__main__":
    main()