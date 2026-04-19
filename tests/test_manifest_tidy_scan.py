from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from asv.manifest.durations import tidy_rel_from_any_path
from asv.manifest.scan import count_tidy_speakers, iter_tidy_rows
from asv.manifest.write import write_manifest


def test_tidy_rel_from_any_path() -> None:
    assert (
        tidy_rel_from_any_path("/data/tidy/id020110/ru/ru_26234196.wav")
        == "id020110/ru/ru_26234196.wav"
    )
    assert (
        tidy_rel_from_any_path("id020110/en/en_26340124.wav")
        == "id020110/en/en_26340124.wav"
    )
    assert tidy_rel_from_any_path("/other/train/x.wav") is None


def test_iter_tidy_rows_basic(tmp_path: Path) -> None:
    root = tmp_path / "tidy"
    wav = root / "id020110" / "ru" / "ru_26234196.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"")

    rows = list(iter_tidy_rows(root, None))
    assert len(rows) == 1
    audio_path, rel_path, spk, vid, seg, aid, dur = rows[0]
    assert spk == "id020110"
    assert vid == "ru"
    assert seg == "ru_26234196"
    assert aid == "id020110/ru/ru_26234196"
    assert rel_path == "id020110/ru/ru_26234196.wav"
    assert audio_path.endswith("ru_26234196.wav")
    assert dur is None


def test_count_tidy_speakers(tmp_path: Path) -> None:
    root = tmp_path / "tidy"
    for sid in ("id020110", "id020111"):
        d = root / sid / "en"
        d.mkdir(parents=True)
        (d / "en_00001.wav").write_bytes(b"")
    assert count_tidy_speakers(root) == 2


def test_tidy_manifest_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "tidy"
    samples = (
        ("id020110", "en", "en_26340124.wav"),
        ("id020110", "ru", "ru_26234196.wav"),
        ("id020111", "cy", "cy_00000001.wav"),
    )
    for speaker, lang, filename in samples:
        path = root / speaker / lang / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "tidy_manifest.parquet",
        dataset="tidy",
    )
    assert n == 3
    assert out.is_file()

    table = pq.read_table(out)
    assert table.num_rows == 3
    assert table.column("speaker_id").to_pylist() == [
        "id020110",
        "id020110",
        "id020111",
    ]
    assert table.column("video_id").to_pylist() == ["en", "ru", "cy"]
    assert table.column("segment_id").to_pylist() == [
        "en_26340124",
        "ru_26234196",
        "cy_00000001",
    ]
