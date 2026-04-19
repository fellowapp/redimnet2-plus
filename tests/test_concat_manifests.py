from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from asv.data import ParquetAudioDataset
from asv.manifest.schema import SCHEMA


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _row(idx: int, speaker_id: str) -> dict[str, object]:
    return {
        "audio_path": f"/tmp/audio_{idx}.wav",
        "rel_path": f"rel/audio_{idx}.wav",
        "speaker_id": speaker_id,
        "video_id": f"vid_{idx}",
        "segment_id": f"seg_{idx}",
        "audio_id": f"{speaker_id}/vid_{idx}/seg_{idx}",
        "duration_sec": 1.0,
    }


def test_concat_manifests_rejects_overlapping_speakers(tmp_path: Path) -> None:
    manifest_a = tmp_path / "a.parquet"
    manifest_b = tmp_path / "b.parquet"
    output = tmp_path / "merged.parquet"
    _write_manifest(manifest_a, [_row(0, "spk_same")])
    _write_manifest(manifest_b, [_row(1, "spk_same")])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/concat_manifests.py",
            str(manifest_a),
            str(manifest_b),
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "overlapping speaker_id values" in result.stderr


def test_concat_manifests_prefixes_speakers_for_training(tmp_path: Path) -> None:
    manifest_a = tmp_path / "a.parquet"
    manifest_b = tmp_path / "b.parquet"
    output = tmp_path / "merged.parquet"
    _write_manifest(
        manifest_a,
        [_row(0, "spk_same"), _row(1, "spk_a")],
    )
    _write_manifest(manifest_b, [_row(2, "spk_same"), _row(3, "spk_b")])

    subprocess.run(
        [
            sys.executable,
            "scripts/concat_manifests.py",
            str(manifest_a),
            str(manifest_b),
            str(output),
            "--speaker-prefix-a",
            "a:",
            "--speaker-prefix-b",
            "b:",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    table = pq.read_table(output)
    assert table.num_rows == 4
    assert table.column("speaker_id").to_pylist() == [
        "a:spk_same",
        "a:spk_a",
        "b:spk_same",
        "b:spk_b",
    ]
    assert table.column("audio_id").to_pylist() == [
        "a:spk_same/vid_0/seg_0",
        "a:spk_a/vid_1/seg_1",
        "b:spk_same/vid_2/seg_2",
        "b:spk_b/vid_3/seg_3",
    ]

    dataset = ParquetAudioDataset(output)
    assert len(dataset) == 4
    assert dataset.num_speakers == 4
