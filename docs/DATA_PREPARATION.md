# Подготовка данных

## С чем работаем

Трейн строится на корпусах speaker verification / speaker id. Поддерживаемые типы (из `scripts/build_parquet_manifest.py --dataset`):

| `--dataset`     | Назначение                          | Краткий layout                                              |
|-----------------|-------------------------------------|-------------------------------------------------------------|
| `voxceleb1`     | VoxCeleb1 (1251 спикер)             | `<root>/wav/<idXXXXX>/<youtube_id>/<segment>.wav`           |
| `voxceleb2`     | VoxCeleb2                           | `<root>/dev/aac/<idXXXXX>/<youtube_id>/<segment>.m4a`       |
| `cnceleb`       | CN-Celeb (train split)              | `<root>/data/<idXXXXX>/<genre>-<session>-<utt>.flac`        |
| `cnceleb2`      | CN-Celeb2 (та же схема)             | то же                                                       |
| `cnceleb_all`   | CN-Celeb + eval/{enroll,test}       | `data/` + `eval/enroll`, `eval/test`                        |
| `3dspeaker`     | 3D-Speaker                          | `<root>/3D_SPK_<id>/...wav`                                 |
| `kryp`          | Kryp speaker-id challenge           | `<root>/train.csv` + `<root>/train/<spk>/<NNNNN>.flac`      |
| `tidy`          | TIDY-подобные дампы                 | `<root>/<spk>/<lang>/<lang>_<utt>.wav`                      |
| `kespeech`      | KeSpeech                            | `<root>/Audio/<spk>/<phase>/<utt>.wav`                      |

Форматы файлов: `.wav`, `.flac`, `.m4a`, `.mp4`, `.ogg` — декод идёт через `torchcodec` (см. `asv/data/decode.py`), fallback — `torchaudio.load_with_torchcodec`.

## Частота дискретизации

Целевая частота — 16 kHz. В датасете (`configs/data/default.yaml`) это `data.sample_rate: 16000`. `torchcodec` резамплит на лету, если аудио в другом SR.

## Нарезка

Трейн использует фиксированную длину `data.trim_num_samples` (по умолчанию 96000 семплов = 6 секунд при 16 kHz). Если файл длиннее — берётся случайное окно; если короче — repeat-padding до длины.

## Шумы и реверберация (MUSAN, RIRS)

Конфиг аугментаций (`configs/augment/default.yaml`) опирается на два корневых пути через переменные окружения:

- `MUSAN_PATH` — корень MUSAN (используется в `AddBackgroundNoise`)
- `RIRS_PATH` — корень RIRS (используется в `ApplyImpulseResponse`)

Скачать и распаковать:

```
MUSAN: https://www.openslr.org/17/
RIRS : https://www.openslr.org/28/
```

Оба раскладываются в любую директорию; достаточно, чтобы `torch-audiomentations` видел поддиректории с `.wav`. Путь передаём в `run.sh`:

```bash
MUSAN_PATH="/.../musan"
RIRS_PATH="/.../RIRS_NOISES"
```

Если MUSAN/RIRS нет — выключить аугментации через CLI-overrides: `augment.waveform_levels.1.p=0.0` и `augment.waveform_levels.2.p=0.0`. Либо поправить `configs/augment/default.yaml`.

## Валидация (VoxCeleb1)

Для валидации нужен VoxCeleb1 wav-tree и trial-файлы (`voxceleb1_o.txt`, `voxceleb_e.txt`, `voxceleb_h.txt`). Пути и список передаются через `run.sh`:

```bash
VOXCELEB_VAL_ROOT="/.../VoxCeleb"
VOXCELEB_TRIAL_FILES="[${VOXCELEB_VAL_ROOT}/voxceleb1_o.txt, ...]"
```

Trial-файл — стандартный формат: `<label> <utt1> <utt2>` (разделитель — пробел). Относительные пути внутри триала разрешаются относительно `validation.data_root` (обычно `.../VoxCeleb/wav`).

## Variant-валидация (аугментированные наборы)

`variant_validation` прогоняет валидацию по поддиректориям, где лежат те же файлы, но прогнанные через конкретный кодек/аугментацию. Полезно для оценки робастности.

```bash
VARIANT_VAL_ENABLED=true
VARIANT_VAL_SCAN_ROOT="/.../VoxCelebAug"
```

Собрать такие поддиректории можно `scripts/build_voxceleb_trial_audio_variants.py` или `scripts/build_voxceleb_hard_augmentation_dataset.py`.

## Кодек-аугментации

Если включены `codec_only` / `codec_then_augment` в `configs/data/default.yaml`, нужен `ffmpeg` с нужными кодеками (см. `INSTALLATION.md` §4). Проверка:

```bash
python scripts/generate_codec_audio_variants.py <sample.wav> --output_dir ./outputs/test_codecs
```

Если любой из нужных кодеков падает — трейн будет часто падать на кодек-аугментации (или молча скипать). Либо чинить `ffmpeg`, либо выключать кодек-ветку (`data.codec_preset=null`, нулевые вероятности `codec_only`/`codec_then_augment`).

## Далее

Подготовить `audio_path`-манифест в Parquet: см. [`MANIFESTS.md`](MANIFESTS.md).
