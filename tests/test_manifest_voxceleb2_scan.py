"""VoxCeleb2 manifest row scanning (minimal layout)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from asv.manifest.scan import count_voxceleb2_speakers, iter_voxceleb2_rows
from asv.manifest.write import write_manifest


def test_iter_voxceleb2_rows_basic(tmp_path: Path) -> None:
    root = tmp_path / "VoxCeleb2"
    utt = root / "dev" / "aac" / "id10001"
    utt.mkdir(parents=True)
    (utt / "YtId_00001.m4a").write_bytes(b"")

    rows = list(iter_voxceleb2_rows(root, None))
    assert len(rows) == 1
    audio_path, rel_path, spk, vid, seg, aid, dur = rows[0]
    assert spk == "id10001"
    assert vid == "YtId"
    assert seg == "00001"
    assert aid == "id10001/YtId/00001"
    assert rel_path.replace("\\", "/") == "dev/aac/id10001/YtId_00001.m4a"
    assert audio_path.endswith("YtId_00001.m4a")
    assert dur is None


def test_count_voxceleb2_speakers(tmp_path: Path) -> None:
    root = tmp_path / "VoxCeleb2"
    for sid in ("id10001", "id10002"):
        d = root / "dev" / "aac" / sid
        d.mkdir(parents=True)
        (d / "a.m4a").write_bytes(b"")
    assert count_voxceleb2_speakers(root) == 2


def test_iter_voxceleb2_rows_dev_subroot_rel_path(tmp_path: Path) -> None:
    """Passing ``.../VoxCeleb2/dev`` must yield ``rel_path`` with ``dev/`` prefix (duration join)."""
    root = tmp_path / "VoxCeleb2" / "dev"
    utt = root / "aac" / "id10001"
    utt.mkdir(parents=True)
    (utt / "YtId_00001.m4a").write_bytes(b"")

    rows = list(iter_voxceleb2_rows(root, None))
    assert len(rows) == 1
    _ap, rel, spk, vid, seg, aid, _dur = rows[0]
    assert spk == "id10001"
    assert rel.replace("\\", "/") == "dev/aac/id10001/YtId_00001.m4a"
    assert aid == "id10001/YtId/00001"


def test_iter_voxceleb2_rows_nested_video_folder(tmp_path: Path) -> None:
    root = tmp_path / "VoxCeleb2"
    utt = root / "dev" / "aac" / "id00012" / "tqaasM16UxI"
    utt.mkdir(parents=True)
    (utt / "00156.m4a").write_bytes(b"")

    rows = list(iter_voxceleb2_rows(root, None))
    assert len(rows) == 1
    _ap, rel, spk, vid, seg, aid, _dur = rows[0]
    assert spk == "id00012"
    assert vid == "tqaasM16UxI"
    assert seg == "00156"
    assert aid == "id00012/tqaasM16UxI/00156"
    assert rel.replace("\\", "/") == "dev/aac/id00012/tqaasM16UxI/00156.m4a"


def test_write_voxceleb2_manifest_excludes_trial_utterances(tmp_path: Path) -> None:
    root = tmp_path / "VoxCeleb2"
    base = root / "dev" / "aac" / "id10001"
    base.mkdir(parents=True)
    (base / "keep_YtId_00001.m4a").write_bytes(b"")
    (base / "drop_YtId_00002.m4a").write_bytes(b"")

    trial = root / "voxceleb2_o.txt"
    trial.write_text(
        "1 id10001/drop_YtId_00002.m4a id10999/not_in_tree/y.m4a\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "manifest.parquet",
        dataset="voxceleb2",
        exclude_voxceleb_trial_txt=trial,
    )
    assert n == 1
    rels = pq.read_table(out, columns=["rel_path"]).column(0).to_pylist()
    assert rels[0].replace("\\", "/") == "dev/aac/id10001/keep_YtId_00001.m4a"
