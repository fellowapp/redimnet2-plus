"""Parquet-схема и константы для ASV-манифестов.

Схема одинаковая для всех датасетов (VoxCeleb1/2, CN-Celeb, 3D-Speaker, Kryp, tidy,
KeSpeech) — это позволяет склеивать манифесты между собой (см.
:mod:`asv.manifest.concat`) и одинаково их читать в :class:`asv.data.ParquetAudioDataset`.

Колонки:
- ``audio_path`` — абсолютный путь к аудио-файлу;
- ``rel_path`` — относительный путь (для join'а с duration-parquet и триалов);
- ``speaker_id`` — метка спикера (класс для трейна);
- ``video_id`` / ``segment_id`` — ID видео и сегмента (для группировки/отладки);
- ``audio_id`` — стабильный идентификатор ``speaker/video/segment``;
- ``duration_sec`` — длительность в секундах (может быть null).

``BATCH_ROWS`` — размер батча при записи (число строк, накапливаемых перед flush).
``AUDIO_SUBDIR`` — каноническое имя аудио-поддиректории для OGG-корпусов."""

from __future__ import annotations

import pyarrow as pa

AUDIO_SUBDIR = "audio"
DEFAULT_DURATION_NAME = "VoxCeleb2.compressed.durations.parquet"
BATCH_ROWS = 10_000

SCHEMA = pa.schema(
    [
        ("audio_path", pa.string()),
        ("rel_path", pa.string()),
        ("speaker_id", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("audio_id", pa.string()),
        ("duration_sec", pa.float64()),
    ]
)
