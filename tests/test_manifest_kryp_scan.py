from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from asv.data import ParquetAudioDataset
from asv.manifest.durations import kryp_rel_from_any_path
from asv.manifest.scan import count_kryp_speakers, iter_kryp_train_rows
from asv.manifest.write import write_manifest


def test_kryp_rel_from_any_path() -> None:
    assert (
        kryp_rel_from_any_path("/data/kryp/train/spk/00001.flac")
        == "train/spk/00001.flac"
    )
    assert kryp_rel_from_any_path("train/spk/00001.flac") == "train/spk/00001.flac"
    assert kryp_rel_from_any_path("/other/wav/x.wav") is None


def _write_flac(path: Path, frames: int = 3200, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.zeros(frames, dtype=np.float32)
    sf.write(str(path), x, sr, subtype="PCM_16", format="FLAC")


def test_kryp_manifest_roundtrip_and_dataset(tmp_path: Path) -> None:
    root = tmp_path / "kryp"
    train = root / "train"
    _write_flac(train / "spkA" / "00000.flac", frames=16000)
    _write_flac(train / "spkA" / "00001.flac", frames=32000)
    _write_flac(train / "spkB" / "00000.flac", frames=8000)

    csv_path = root / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "filepath"])
        w.writerow(["spkA", "train/spkA/00000.flac"])
        w.writerow(["spkA", "train/spkA/00001.flac"])
        w.writerow(["spkB", "train/spkB/00000.flac"])

    assert count_kryp_speakers(root) == 2

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "kryp_train_manifest.parquet",
        dataset="kryp",
        kryp_probe_durations=True,
    )
    assert n == 3
    assert out.is_file()

    table = pq.read_table(out)
    assert table.num_rows == 3
    durs = table.column("duration_sec").to_pylist()
    assert abs(durs[0] - 1.0) < 1e-3  # 16000 / 16000
    assert abs(durs[1] - 2.0) < 1e-3
    assert abs(durs[2] - 0.5) < 1e-3

    ds = ParquetAudioDataset(
        out,
        trim_num_samples=8000,
        fallback_on_decode_error=True,
    )
    assert len(ds) == 3
    assert ds.num_speakers == 2
    item = ds[0]
    assert "waveform" in item and "label" in item


def test_kryp_skips_missing_files(tmp_path: Path) -> None:
    root = tmp_path / "kryp"
    train = root / "train"
    _write_flac(train / "spkA" / "00000.flac", frames=1600)
    csv_path = root / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["speaker_id", "filepath"])
        w.writerow(["spkA", "train/spkA/00000.flac"])
        w.writerow(["spkA", "train/spkA/missing.flac"])

    rows = list(iter_kryp_train_rows(root, None, probe_durations=True))
    assert len(rows) == 1
