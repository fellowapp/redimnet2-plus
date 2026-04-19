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


def test_concat_training_manifests_prefixes_by_dataset(tmp_path: Path) -> None:
    manifest_a = tmp_path / "3dspeaker.parquet"
    manifest_b = tmp_path / "cnceleb2.parquet"
    manifest_c = tmp_path / "kryp.parquet"
    output = tmp_path / "merged.parquet"
    _write_manifest(manifest_a, [_row(0, "spk_same"), _row(1, "spk_a")])
    _write_manifest(manifest_b, [_row(2, "spk_same")])
    _write_manifest(manifest_c, [_row(3, "spk_same"), _row(4, "spk_c")])

    subprocess.run(
        [
            sys.executable,
            "scripts/concat_training_manifests.py",
            str(output),
            "--manifest",
            f"3dspeaker={manifest_a}",
            "--manifest",
            f"cnceleb2={manifest_b}",
            "--manifest",
            f"kryp={manifest_c}",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    table = pq.read_table(output)
    assert table.num_rows == 5
    assert table.column("speaker_id").to_pylist() == [
        "3dspeaker:spk_same",
        "3dspeaker:spk_a",
        "cnceleb2:spk_same",
        "kryp:spk_same",
        "kryp:spk_c",
    ]
    assert table.column("audio_id").to_pylist() == [
        "3dspeaker:spk_same/vid_0/seg_0",
        "3dspeaker:spk_a/vid_1/seg_1",
        "cnceleb2:spk_same/vid_2/seg_2",
        "kryp:spk_same/vid_3/seg_3",
        "kryp:spk_c/vid_4/seg_4",
    ]

    dataset = ParquetAudioDataset(output)
    assert len(dataset) == 5
    assert dataset.num_speakers == 5


def test_concat_training_manifests_keeps_raw_manifest_unchanged(tmp_path: Path) -> None:
    merged_manifest = tmp_path / "merged_existing.parquet"
    tidy_manifest = tmp_path / "tidy.parquet"
    output = tmp_path / "merged_with_tidy.parquet"
    _write_manifest(
        merged_manifest,
        [_row(0, "3dspeaker:spk_a"), _row(1, "kryp:spk_b")],
    )
    _write_manifest(
        tidy_manifest,
        [_row(2, "id020110"), _row(3, "id020111")],
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/concat_training_manifests.py",
            str(output),
            "--raw-manifest",
            str(merged_manifest),
            "--manifest",
            f"tidy={tidy_manifest}",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )

    table = pq.read_table(output)
    assert table.num_rows == 4
    assert table.column("speaker_id").to_pylist() == [
        "3dspeaker:spk_a",
        "kryp:spk_b",
        "tidy:id020110",
        "tidy:id020111",
    ]
    assert table.column("audio_id").to_pylist() == [
        "3dspeaker:spk_a/vid_0/seg_0",
        "kryp:spk_b/vid_1/seg_1",
        "tidy:id020110/vid_2/seg_2",
        "tidy:id020111/vid_3/seg_3",
    ]
