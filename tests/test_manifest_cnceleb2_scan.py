"""CN-Celeb2 manifest row scanning."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from asv.manifest.durations import (
    cnceleb2_rel_from_any_path,
    cnceleb_all_rel_from_any_path,
)
from asv.manifest.scan import (
    count_cnceleb2_speakers,
    count_cnceleb_all_speakers,
    iter_cnceleb2_rows,
    iter_cnceleb_all_rows,
)
from asv.manifest.write import write_manifest


def test_cnceleb2_rel_from_any_path() -> None:
    assert (
        cnceleb2_rel_from_any_path(
            "/data/CN-Celeb2_flac/data/id10001/drama-01-001.flac"
        )
        == "data/id10001/drama-01-001.flac"
    )
    assert (
        cnceleb2_rel_from_any_path("data/id10001/drama-01-001.flac")
        == "data/id10001/drama-01-001.flac"
    )
    assert cnceleb2_rel_from_any_path("/other/train/x.flac") is None


def test_cnceleb_all_rel_from_any_path() -> None:
    assert (
        cnceleb_all_rel_from_any_path(
            "/data/CN-Celeb_flac/eval/test/id00800-singing-01-001.flac"
        )
        == "eval/test/id00800-singing-01-001.flac"
    )
    assert (
        cnceleb_all_rel_from_any_path(
            "/data/CN-Celeb_flac/data/id00001/speech-01-001.flac"
        )
        == "data/id00001/speech-01-001.flac"
    )
    assert cnceleb_all_rel_from_any_path("/other/train/x.flac") is None


def test_iter_cnceleb2_rows_basic(tmp_path: Path) -> None:
    root = tmp_path / "CN-Celeb2"
    utt = root / "data" / "id10001"
    utt.mkdir(parents=True)
    (utt / "drama-01-001.flac").write_bytes(b"")

    rows = list(iter_cnceleb2_rows(root, None))
    assert len(rows) == 1
    audio_path, rel_path, spk, vid, seg, aid, dur = rows[0]
    assert spk == "id10001"
    assert vid == "drama-01"
    assert seg == "001"
    assert aid == "id10001/drama-01/001"
    assert rel_path.replace("\\", "/") == "data/id10001/drama-01-001.flac"
    assert audio_path.endswith("drama-01-001.flac")
    assert dur is None


def test_iter_cnceleb2_rows_data_subroot_rel_path(tmp_path: Path) -> None:
    root = tmp_path / "CN-Celeb2" / "data"
    utt = root / "id10001"
    utt.mkdir(parents=True)
    (utt / "live_broadcast-02-015.flac").write_bytes(b"")

    rows = list(iter_cnceleb2_rows(root, None))
    assert len(rows) == 1
    _ap, rel, spk, vid, seg, aid, _dur = rows[0]
    assert spk == "id10001"
    assert vid == "live_broadcast-02"
    assert seg == "015"
    assert aid == "id10001/live_broadcast-02/015"
    assert rel.replace("\\", "/") == "data/id10001/live_broadcast-02-015.flac"


def test_count_cnceleb2_speakers(tmp_path: Path) -> None:
    root = tmp_path / "CN-Celeb2"
    for sid in ("id10001", "id10002"):
        d = root / "data" / sid
        d.mkdir(parents=True)
        (d / "a-01-001.flac").write_bytes(b"")
    assert count_cnceleb2_speakers(root) == 2


def test_cnceleb2_manifest_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "CN-Celeb2"
    data = root / "data"
    for rel in (
        ("id10001", "drama-01-001.flac"),
        ("id10001", "drama-01-002.flac"),
        ("id10002", "speech-03-004.flac"),
    ):
        path = data / rel[0] / rel[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "cnceleb2_manifest.parquet",
        dataset="cnceleb2",
    )
    assert n == 3
    assert out.is_file()

    table = pq.read_table(out)
    assert table.num_rows == 3
    assert table.column("speaker_id").to_pylist() == [
        "id10001",
        "id10001",
        "id10002",
    ]
    assert table.column("video_id").to_pylist() == [
        "drama-01",
        "drama-01",
        "speech-03",
    ]
    assert table.column("segment_id").to_pylist() == ["001", "002", "004"]


def test_cnceleb_manifest_alias_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "CN-Celeb"
    data = root / "data" / "id00001"
    data.mkdir(parents=True)
    (data / "speech-01-001.flac").write_bytes(b"")

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "cnceleb_manifest.parquet",
        dataset="cnceleb",
    )
    assert n == 1
    assert out.is_file()

    table = pq.read_table(out)
    assert table.num_rows == 1
    row = table.to_pylist()[0]
    assert row["speaker_id"] == "id00001"
    assert row["video_id"] == "speech-01"
    assert row["segment_id"] == "001"


def test_iter_cnceleb_all_rows_and_count(tmp_path: Path) -> None:
    root = tmp_path / "CN-Celeb"
    data = root / "data" / "id00001"
    data.mkdir(parents=True)
    (data / "speech-01-001.flac").write_bytes(b"")

    enroll = root / "eval" / "enroll"
    enroll.mkdir(parents=True)
    (enroll / "id00001-enroll.flac").write_bytes(b"")

    test = root / "eval" / "test"
    test.mkdir(parents=True)
    (test / "id00001-speech-01-001.flac").write_bytes(b"")

    dev = root / "dev"
    dev.mkdir(parents=True)
    (dev / "dev.lst").write_text("id00001\n", encoding="utf-8")

    rows = list(iter_cnceleb_all_rows(root, None))
    assert len(rows) == 3
    assert [row[5] for row in rows] == [
        "id00001/speech-01/001",
        "id00001/eval-enroll/enroll",
        "id00001/eval-test-speech-01/001",
    ]
    assert count_cnceleb_all_speakers(root) == 1


def test_cnceleb_all_manifest_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "CN-Celeb"
    data = root / "data" / "id00001"
    data.mkdir(parents=True)
    (data / "speech-01-001.flac").write_bytes(b"")

    enroll = root / "eval" / "enroll"
    enroll.mkdir(parents=True)
    (enroll / "id00001-enroll.flac").write_bytes(b"")

    test = root / "eval" / "test"
    test.mkdir(parents=True)
    (test / "id00001-speech-01-001.flac").write_bytes(b"")

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "cnceleb_all_manifest.parquet",
        dataset="cnceleb_all",
    )
    assert n == 3
    assert out.is_file()

    table = pq.read_table(out)
    assert table.num_rows == 3
    assert table.column("rel_path").to_pylist() == [
        "data/id00001/speech-01-001.flac",
        "eval/enroll/id00001-enroll.flac",
        "eval/test/id00001-speech-01-001.flac",
    ]
    assert table.column("video_id").to_pylist() == [
        "speech-01",
        "eval-enroll",
        "eval-test-speech-01",
    ]
