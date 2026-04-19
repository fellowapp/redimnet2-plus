"""VoxCeleb1 manifest row scanning (minimal layout)."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from asv.manifest.scan import count_voxceleb1_speakers, iter_voxceleb1_rows
from asv.manifest.write import write_manifest


def test_iter_voxceleb1_rows_basic(tmp_path: Path) -> None:
    root = tmp_path / "VoxCeleb1"
    utt = root / "wav" / "id10001"
    utt.mkdir(parents=True)
    (utt / "YtId_00001.wav").write_bytes(b"")

    rows = list(iter_voxceleb1_rows(root, None))
    assert len(rows) == 1
    audio_path, rel_path, spk, vid, seg, aid, dur = rows[0]
    assert spk == "id10001"
    assert vid == "YtId"
    assert seg == "00001"
    assert aid == "id10001/YtId/00001"
    assert rel_path.replace("\\", "/") == "wav/id10001/YtId_00001.wav"
    assert audio_path.endswith("YtId_00001.wav")
    assert dur is None


def test_count_voxceleb1_speakers(tmp_path: Path) -> None:
    root = tmp_path / "VoxCeleb1"
    for sid in ("id10001", "id10002"):
        d = root / "wav" / sid
        d.mkdir(parents=True)
        (d / "a.wav").write_bytes(b"")
    assert count_voxceleb1_speakers(root) == 2


def test_iter_voxceleb1_rows_wav_subroot_rel_path(tmp_path: Path) -> None:
    """Passing ``.../VoxCeleb1/wav`` must yield ``rel_path`` with ``wav/`` prefix."""
    root = tmp_path / "VoxCeleb1" / "wav"
    utt = root / "id10001"
    utt.mkdir(parents=True)
    (utt / "YtId_00001.wav").write_bytes(b"")

    rows = list(iter_voxceleb1_rows(root, None))
    assert len(rows) == 1
    _ap, rel, spk, vid, seg, aid, _dur = rows[0]
    assert spk == "id10001"
    assert rel.replace("\\", "/") == "wav/id10001/YtId_00001.wav"
    assert aid == "id10001/YtId/00001"


def test_iter_voxceleb1_rows_nested_video_folder(tmp_path: Path) -> None:
    root = tmp_path / "VoxCeleb1"
    utt = root / "wav" / "id00012" / "tqaasM16UxI"
    utt.mkdir(parents=True)
    (utt / "00156.wav").write_bytes(b"")

    rows = list(iter_voxceleb1_rows(root, None))
    assert len(rows) == 1
    _ap, rel, spk, vid, seg, aid, _dur = rows[0]
    assert spk == "id00012"
    assert vid == "tqaasM16UxI"
    assert seg == "00156"
    assert aid == "id00012/tqaasM16UxI/00156"
    assert rel.replace("\\", "/") == "wav/id00012/tqaasM16UxI/00156.wav"


def test_write_voxceleb1_manifest_excludes_trial_utterances(tmp_path: Path) -> None:
    root = tmp_path / "VoxCeleb1"
    d = root / "wav" / "id10001"
    d.mkdir(parents=True)
    for stem in ("keep_YtId_00001.wav", "drop_YtId_00002.wav"):
        (d / stem).write_bytes(b"")

    trial = root / "voxceleb1_o.txt"
    # Second path is only a trial pairing placeholder (no file under this tree).
    trial.write_text(
        "1 id10001/drop_YtId_00002.wav id10999/not_in_tree/y.wav\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out, n = write_manifest(
        root,
        out_dir,
        None,
        "manifest.parquet",
        dataset="voxceleb1",
        exclude_voxceleb_trial_txt=trial,
    )
    assert n == 1
    rels = pq.read_table(out, columns=["rel_path"]).column(0).to_pylist()
    assert rels[0].replace("\\", "/") == "wav/id10001/keep_YtId_00001.wav"
