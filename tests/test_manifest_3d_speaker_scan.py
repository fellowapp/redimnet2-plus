"""3D-Speaker manifest row scanning."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from asv.manifest.durations import speaker3d_rel_from_any_path
from asv.manifest.scan import count_3d_speaker_speakers, iter_3d_speaker_rows
from asv.manifest.write import write_manifest


def test_speaker3d_rel_from_any_path() -> None:
    assert (
        speaker3d_rel_from_any_path(
            "/data/3d_speaker/3D_SPK_00001/"
            "3D_SPK_00001_001_Device01_Distance04_Dialect00.wav"
        )
        == "3D_SPK_00001/3D_SPK_00001_001_Device01_Distance04_Dialect00.wav"
    )
    assert (
        speaker3d_rel_from_any_path(
            "3D_SPK_00001/3D_SPK_00001_001_Device01_Distance04_Dialect00.wav"
        )
        == "3D_SPK_00001/3D_SPK_00001_001_Device01_Distance04_Dialect00.wav"
    )
    assert speaker3d_rel_from_any_path("/other/train/x.wav") is None


def test_iter_3d_speaker_rows_basic(tmp_path: Path) -> None:
    root = tmp_path / "3d_speaker"
    utt = root / "3D_SPK_00001"
    utt.mkdir(parents=True)
    wav = "3D_SPK_00001_001_Device01_Distance04_Dialect00.wav"
    (utt / wav).write_bytes(b"")

    rows = list(iter_3d_speaker_rows(root, None))
    assert len(rows) == 1
    audio_path, rel_path, spk, vid, seg, aid, dur = rows[0]
    assert spk == "3D_SPK_00001"
    assert vid == "001"
    assert seg == "Device01_Distance04_Dialect00"
    assert aid == "3D_SPK_00001/001/Device01_Distance04_Dialect00"
    assert rel_path == f"3D_SPK_00001/{wav}"
    assert audio_path.endswith(wav)
    assert dur is None


def test_count_3d_speaker_speakers(tmp_path: Path) -> None:
    root = tmp_path / "3d_speaker"
    for sid in ("3D_SPK_00001", "3D_SPK_00002"):
        d = root / sid
        d.mkdir(parents=True)
        (d / f"{sid}_001_Device01_Distance04_Dialect00.wav").write_bytes(b"")
    assert count_3d_speaker_speakers(root) == 2


def test_3d_speaker_manifest_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "3d_speaker"
    samples = (
        ("3D_SPK_00001", "3D_SPK_00001_001_Device01_Distance04_Dialect00.wav"),
        ("3D_SPK_00001", "3D_SPK_00001_002_Device02_Distance13_Dialect00.wav"),
        ("3D_SPK_00002", "3D_SPK_00002_001_Device05_Distance12_Dialect01.wav"),
    )
    for speaker, filename in samples:
        path = root / speaker / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "3dspeaker_manifest.parquet",
        dataset="3dspeaker",
    )
    assert n == 3
    assert out.is_file()

    table = pq.read_table(out)
    assert table.num_rows == 3
    assert table.column("speaker_id").to_pylist() == [
        "3D_SPK_00001",
        "3D_SPK_00001",
        "3D_SPK_00002",
    ]
    assert table.column("video_id").to_pylist() == ["001", "002", "001"]
    assert table.column("segment_id").to_pylist() == [
        "Device01_Distance04_Dialect00",
        "Device02_Distance13_Dialect00",
        "Device05_Distance12_Dialect01",
    ]
