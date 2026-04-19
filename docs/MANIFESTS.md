# Манифесты (Parquet)

Всё, что трогает трейн — идёт через один Parquet-манифест. Это единственная точка, определяющая «что вообще видит датасет».

## Схема

Файл: `asv/manifest/schema.py`. Колонки:

| Колонка         | Тип     | Назначение                                           |
|-----------------|---------|------------------------------------------------------|
| `audio_path`    | string  | Абсолютный путь к аудио (читает `torchcodec`)        |
| `rel_path`      | string  | Относительный путь от корня датасета                 |
| `speaker_id`    | string  | Метка спикера (определяет класс)                     |
| `video_id`      | string  | ID видео/сессии (для группировки)                    |
| `segment_id`    | string  | ID сегмента (чаще всего stem файла)                  |
| `audio_id`      | string  | Стабильный ID сэмпла `speaker/video/segment`         |
| `duration_sec`  | double  | Длина в секундах (опционально, может быть null)      |

Типы — жёстко проверяются при конкатенации (`asv/manifest/concat.py`). Сжатие — zstd.

## Сбор одного манифеста

Скрипт `scripts/build_parquet_manifest.py` с флагом `--dataset` сканирует нужную раскладку и пишет Parquet:

```bash
# VoxCeleb2
python scripts/build_parquet_manifest.py --dataset voxceleb2 \
  /path/to/VoxCeleb2 /path/to/output_dir

# VoxCeleb1 (1251 класс)
python scripts/build_parquet_manifest.py --dataset voxceleb1 \
  /path/to/VoxCeleb1 /path/to/output_dir

# CN-Celeb / CN-Celeb2 / CN-Celeb all splits
python scripts/build_parquet_manifest.py --dataset cnceleb2 \
  /path/to/CNCeleb2 /path/to/output_dir

# 3D-Speaker / tidy / KeSpeech
python scripts/build_parquet_manifest.py --dataset kespeech \
  /path/to/KeSpeech /path/to/output_dir

# Kryp (speaker-id challenge): читает train.csv
python scripts/build_parquet_manifest.py --dataset kryp \
  /path/to/Kryp /path/to/output_dir
```

На выходе печатает:
```
Wrote .../<dataset>_manifest.parquet (N rows)
num_classes=<K>
ASV_NUM_CLASSES=<K>
```

Значения `num_classes` / `ASV_NUM_CLASSES` нужны для `run.sh` (переменная `ASV_NUM_CLASSES`) и для `model.num_classes`.

Полный список `--dataset`: `voxceleb1`, `voxceleb2`, `cnceleb`, `cnceleb2`, `cnceleb_all`, `3dspeaker`, `kryp`, `tidy`, `kespeech`.

## Опции

- `--output-name NAME` — кастомное имя `.parquet` файла (по умолчанию `<dataset>_manifest.parquet`).
- `--durations-parquet PATH` — приджойнить длительности из отдельного Parquet (колонки `path`, `duration`).
- `--no-durations` — вообще не читать файлы длительностей (даже если по умолчанию лежит).
- `--probe-durations` / `--no-probe-durations` — только для `--dataset kryp`: читать длительности из FLAC-хедеров (`soundfile.info`). По умолчанию включено для Kryp.

## Длительности (зачем)

Колонка `duration_sec` нужна для `codec_min_duration_sec` в `configs/data/default.yaml`: если длина файла меньше порога, кодек-ветку скипаем (короткие файлы часто не кодируются или кодируются с артефактами). Если `duration_sec` пустая — порог не работает, кодек применяется всегда.

## Конкатенация нескольких манифестов

Если хочется тренировать на нескольких датасетах сразу (VoxCeleb2 + CN-Celeb2 + …) — склейте манифесты через `scripts/concat_manifests.py`:

```bash
python scripts/concat_manifests.py \
  /path/voxceleb2_manifest.parquet \
  /path/cnceleb2_manifest.parquet \
  /path/merged_manifest.parquet \
  --speaker-prefix-a "vox2_" \
  --speaker-prefix-b "cn2_"
```

Префиксы обязательны, если ID спикеров пересекаются (иначе скрипт ругнётся). Если пересечение — это намеренно один и тот же спикер: `--allow-speaker-overlap`.

`scripts/concat_training_manifests.py` — расширенная версия на N манифестов с теми же правилами.

## Отдельные длительности

`asv/manifest/durations.py` хранит helpers для маппинга полных путей в «короткие» ключи (`wav/...`, `dev/aac/...` и т.д.) — они используются при join'е с `--durations-parquet`. Если нужно собрать durations-parquet заранее:

```bash
# Базовый формат: колонки (path, duration). Сборка — вручную (не поставлен готовый скрипт).
# Примеры из asv/manifest/durations.py: *_rel_from_any_path функции показывают, какой ключ ожидается.
```

Для быстрого обхода без отдельного парсинга: `--no-durations` (колонка останется null), либо `--probe-durations` для `--dataset kryp`.

## Использование в трейне

В `run.sh`:

```bash
MANIFEST_PATH="/path/to/merged_manifest.parquet"
ASV_NUM_CLASSES=63934     # число уникальных speaker_id
```

`train.py` дальше поднимает `ParquetAudioDataset(manifest_path, ...)` (см. `asv/data/parquet_dataset.py`), который:

1. Индексирует `(row_group, row_index)` для O(1) доступа.
2. Строит `speaker_id -> label` маппинг по всему манифесту (детерминированно, сортировкой).
3. Кэширует по одному `row_group`, чтобы чтения не рандомились между файлами.

Валидировать схему руками (если сомнения):
```python
from asv.manifest.concat import validate_manifest_schema
from pathlib import Path
validate_manifest_schema(Path("/path/to/manifest.parquet"))
```
