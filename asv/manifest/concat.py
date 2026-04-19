"""Безопасная конкатенация ASV-манифестов для трейна.

Основное правило: если в разных манифестах встречаются одинаковые ``speaker_id``,
они будут слеплены в один класс, а это почти всегда ошибка. :func:`concat_manifests`
требует либо уникальных префиксов (``--speaker-prefix-a/--speaker-prefix-b``), либо
явного ``allow_speaker_overlap=True`` (для случаев, когда это действительно один и
тот же спикер)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from asv.manifest.schema import SCHEMA

REQUIRED_COLUMNS = [field.name for field in SCHEMA]


def validate_manifest_schema(path: Path) -> None:
    """Проверить, что Parquet-манифест содержит все обязательные колонки ASV-схемы."""
    parquet_file = pq.ParquetFile(path)
    missing = [
        name
        for name in REQUIRED_COLUMNS
        if name not in parquet_file.schema.names
    ]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def collect_prefixed_speakers(path: Path, prefix: str) -> set[str]:
    """Collect speaker IDs after applying a training namespace prefix."""
    parquet_file = pq.ParquetFile(path)
    speakers: set[str] = set()
    for row_group in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(row_group, columns=["speaker_id"])
        speakers.update(
            f"{prefix}{speaker}" for speaker in batch.column(0).to_pylist()
        )
    return speakers


def prefixed_audio_id(
    speaker_id: str,
    video_id: str,
    segment_id: str,
    original_audio_id: str,
) -> str:
    """Rebuild ``audio_id`` after namespacing the speaker ID."""
    if video_id and segment_id:
        return f"{speaker_id}/{video_id}/{segment_id}"
    if original_audio_id:
        tail = original_audio_id.split("/", 1)
        if len(tail) == 2:
            return f"{speaker_id}/{tail[1]}"
    return speaker_id


def rewrite_table(table: pa.Table, speaker_prefix: str) -> pa.Table:
    """Prefix ``speaker_id`` and rewrite ``audio_id`` accordingly."""
    if not speaker_prefix:
        return table.select(REQUIRED_COLUMNS).cast(SCHEMA, safe=False)

    data = {
        "audio_path": table.column("audio_path").to_pylist(),
        "rel_path": table.column("rel_path").to_pylist(),
        "speaker_id": [],
        "video_id": table.column("video_id").to_pylist(),
        "segment_id": table.column("segment_id").to_pylist(),
        "audio_id": [],
        "duration_sec": table.column("duration_sec").to_pylist(),
    }
    original_speakers = table.column("speaker_id").to_pylist()
    original_audio_ids = table.column("audio_id").to_pylist()
    for speaker_id, video_id, segment_id, audio_id in zip(
        original_speakers,
        data["video_id"],
        data["segment_id"],
        original_audio_ids,
    ):
        new_speaker = f"{speaker_prefix}{speaker_id}"
        data["speaker_id"].append(new_speaker)
        data["audio_id"].append(
            prefixed_audio_id(
                speaker_id=new_speaker,
                video_id=video_id,
                segment_id=segment_id,
                original_audio_id=audio_id,
            )
        )
    return pa.Table.from_pydict(data, schema=SCHEMA)


def write_manifest_rows(
    writer: pq.ParquetWriter,
    path: Path,
    speaker_prefix: str,
) -> int:
    """Append one manifest to an open writer."""
    parquet_file = pq.ParquetFile(path)
    written = 0
    for row_group in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(
            row_group,
            columns=REQUIRED_COLUMNS,
        )
        writer.write_table(rewrite_table(table, speaker_prefix))
        written += table.num_rows
    return written


def concat_manifests(
    manifest_specs: list[tuple[Path, str]],
    output_path: Path,
    *,
    allow_speaker_overlap: bool = False,
) -> tuple[int, int]:
    """Склеить манифесты и вернуть ``(кол-во строк, кол-во классов)``.

    ``manifest_specs`` — список пар ``(путь_к_манифесту, префикс_спикера)``. Префикс
    добавляется к ``speaker_id`` (и пересобирает ``audio_id``), что позволяет
    развести пересекающиеся ID из разных датасетов."""
    prefixed_speakers: list[set[str]] = []
    for path, prefix in manifest_specs:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}")
        validate_manifest_schema(path)
        prefixed_speakers.append(collect_prefixed_speakers(path, prefix))

    if not allow_speaker_overlap:
        seen: set[str] = set()
        overlap: set[str] = set()
        for speakers in prefixed_speakers:
            overlap.update(seen & speakers)
            seen.update(speakers)
        if overlap:
            preview = sorted(overlap)[:10]
            raise ValueError(
                "Merged manifest would contain overlapping speaker_id values. "
                "Pass prefixes that keep speakers distinct, or use "
                "--allow-speaker-overlap only when overlaps truly refer to "
                f"the same speakers. Examples: {preview}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    writer = pq.ParquetWriter(output_path, SCHEMA, compression="zstd")
    try:
        for path, prefix in manifest_specs:
            total_rows += write_manifest_rows(writer, path, prefix)
    finally:
        writer.close()

    num_classes = len(set().union(*prefixed_speakers))
    return total_rows, num_classes
