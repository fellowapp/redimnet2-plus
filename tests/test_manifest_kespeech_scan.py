from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from asv.manifest.durations import kespeech_rel_from_any_path
from asv.manifest.scan import count_kespeech_speakers, iter_kespeech_rows
from asv.manifest.write import write_manifest


def test_kespeech_rel_from_any_path() -> None:
    path1 = "/data/KeSpeech/Audio/1000505/phase1/1000505_02f36f9a.wav"
    assert (
        kespeech_rel_from_any_path(path1)
        == "Audio/1000505/phase1/1000505_02f36f9a.wav"
    )
    path2 = "Audio/1000505/phase2/1000505_p2_8b9bef0b.wav"
    assert (
        kespeech_rel_from_any_path(path2)
        == "Audio/1000505/phase2/1000505_p2_8b9bef0b.wav"
    )
    assert kespeech_rel_from_any_path("/other/train/x.wav") is None


def test_iter_kespeech_rows_basic(tmp_path: Path) -> None:
    root = tmp_path / "KeSpeech"
    wav = root / "Audio" / "1000505" / "phase1" / "1000505_02f36f9a.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"")

    rows = list(iter_kespeech_rows(root, None))
    assert len(rows) == 1
    audio_path, rel_path, spk, vid, seg, aid, dur = rows[0]
    assert spk == "1000505"
    assert vid == "phase1"
    assert seg == "1000505_02f36f9a"
    assert aid == "1000505/phase1/1000505_02f36f9a"
    assert rel_path == "Audio/1000505/phase1/1000505_02f36f9a.wav"
    assert audio_path.endswith("1000505_02f36f9a.wav")
    assert dur is None


def test_iter_kespeech_rows_audio_subroot_rel_path(tmp_path: Path) -> None:
    root = tmp_path / "KeSpeech" / "Audio"
    wav = root / "1000557" / "phase2" / "1000557_p2_edc52ef4.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"")

    rows = list(iter_kespeech_rows(root, None))
    assert len(rows) == 1
    _ap, rel, spk, vid, seg, aid, _dur = rows[0]
    assert spk == "1000557"
    assert vid == "phase2"
    assert seg == "1000557_p2_edc52ef4"
    assert aid == "1000557/phase2/1000557_p2_edc52ef4"
    assert rel == "Audio/1000557/phase2/1000557_p2_edc52ef4.wav"


def test_count_kespeech_speakers(tmp_path: Path) -> None:
    root = tmp_path / "KeSpeech"
    for sid in ("1000505", "1000649"):
        d = root / "Audio" / sid / "phase1"
        d.mkdir(parents=True)
        (d / f"{sid}_00000000.wav").write_bytes(b"")
    assert count_kespeech_speakers(root) == 2


def test_kespeech_manifest_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "KeSpeech"
    samples = (
        ("1000505", "phase1", "1000505_02f36f9a.wav"),
        ("1000557", "phase2", "1000557_p2_edc52ef4.wav"),
        ("1000649", "phase2", "1000649_p2_10b5b03a.wav"),
    )
    for speaker, phase, filename in samples:
        path = root / "Audio" / speaker / phase / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "kespeech_manifest.parquet",
        dataset="kespeech",
    )
    assert n == 3
    assert out.is_file()

    table = pq.read_table(out)
    assert table.num_rows == 3
    assert table.column("speaker_id").to_pylist() == [
        "1000505",
        "1000557",
        "1000649",
    ]
    assert table.column("video_id").to_pylist() == [
        "phase1",
        "phase2",
        "phase2",
    ]
    assert table.column("segment_id").to_pylist() == [
        "1000505_02f36f9a",
        "1000557_p2_edc52ef4",
        "1000649_p2_10b5b03a",
    ]
