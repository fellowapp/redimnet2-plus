# ReDimNet2+

| Материал | Ссылка |
|----------|--------|
| Отчет | _добавить ссылку_ |
| Презентация | _добавить ссылку_ |

Минимальный набор информации по проекту: сначала быстрый `eval`, затем навигация по остальной документации, запуску обучения и устройству репозитория.

## Быстрый запуск eval

В этом репозитории запуск оценки через Docker сделан вокруг одного сервиса: `submission`.

Что зафиксировано в контейнере:
- `./weights` на хосте монтируется в `/workspace/weights`
- `./outputs` на хосте монтируется в `/workspace/outputs`
- рабочая директория контейнера: `/workspace`
- результат по умолчанию пишется в `./outputs/output.csv`

Для запуска нужен GPU и рабочий Docker с поддержкой NVIDIA.

### Подготовка

Перед запуском убедитесь, что:
- веса лежат в `weights/`
- папка `outputs/` существует
- аудио находится в отдельной директории на хосте, например `/home/kirill/mnt/ssd_1tb_1/kryp/test_public`

### Команда сборки

Эта команда собирает image и проверяет, что внутри контейнера видны примонтированные `weights` и `outputs`:

```bash
docker compose -f docker-compose.submission.yml run --rm --build \
  --entrypoint /bin/bash submission -lc \
  "ls -ld /workspace/weights /workspace/outputs"
```

Что делает команда:
- `--build` пересобирает image перед запуском
- `--entrypoint /bin/bash` временно отключает обычный entrypoint инференса
- `submission -lc "ls -ld ..."` запускает короткую проверку внутри контейнера
- `--rm` удаляет временный контейнер после завершения

Если команда отработала без ошибки, можно запускать eval.

### Команда запуска оценки

Это команда для запуска оценки:

```bash
docker compose -f docker-compose.submission.yml run --rm --build \
  -v /home/kirill/mnt/ssd_1tb_1/kryp/test_public:/data/test_public:ro submission \
  --audio-root /data/test_public \
  --k 10 \
  --num-workers 8 \
  --gpu-model-instances 3
```

Важное правило: путь в `--audio-root` должен совпадать с путем внутри контейнера, который указан во флаге `-v`.

Пример:
- на хосте: `/home/kirill/mnt/ssd_1tb_1/kryp/test_public`
- внутри контейнера: `/data/test_public`
- значит в аргументе надо передать `--audio-root /data/test_public`

Если image уже собран и код не менялся, `--build` в команде оценки можно убрать.

### Где лежит результат

По умолчанию результат сохраняется в:

```text
./outputs/output.csv
```

Внутри контейнера это путь:

```text
/workspace/outputs/output.csv
```

### Аргументы команды запуска

Ниже перечислены аргументы, которые можно указывать в команде оценки.

#### Аргументы `docker compose run`

Это аргументы уровня Docker, они идут до имени сервиса `submission`.

- `--rm`  
  Удалить временный контейнер после завершения.

- `--build`  
  Пересобрать image перед запуском.

- `-v HOST_PATH:CONTAINER_PATH:ro`  
  Примонтировать папку с аудио в контейнер в режиме read-only.

Пример:

```bash
-v /abs/path/to/audio:/data/test_public:ro
```

#### Аргументы `run_submission_unified.py`

Это аргументы самого скрипта. Они идут после имени сервиса `submission`.

- `--audio-root PATH`  
  Обязательный аргумент. Корень аудио внутри контейнера. Должен совпадать с путем из `-v`.

- `--output-csv PATH`  
  Путь для выходного CSV.  
  По умолчанию: `/workspace/outputs/output.csv`

- `--weights-dir PATH`  
  Папка с весами. Скрипт рекурсивно ищет в ней актуальный checkpoint.  
  По умолчанию: `/workspace/weights`

- `--model-config PATH`  
  Путь к конфигу модели.  
  По умолчанию: `/workspace/configs/model/redimnet2_b6.yaml`

- `--k INT`  
  Число соседей в итоговом ответе.  
  По умолчанию: `10`

- `--gpu-model-instances INT`  
  Сколько копий модели держать на GPU параллельно.  
  По умолчанию: `3`

- `--num-workers INT`  
  Единый параметр для DataLoader и CPU-части rerank.  
  По умолчанию: `8`

<details>
<summary>Дополнительные параметры rerank и декода</summary>

- `--mean-steps INT`  
  Число шагов mean-chain rerank.  
  По умолчанию: `40`

- `--n-iter INT`  
  Число итераций графового rerank.  
  По умолчанию: `3`

- `--w-rank FLOAT`  
  Вес rank-компоненты в rerank.  
  По умолчанию: `1.0`

- `--w-recip FLOAT`  
  Вес reciprocal-компоненты в rerank.  
  По умолчанию: `1.5`

- `--w-common FLOAT`  
  Вес common-neighbors компоненты в rerank.  
  По умолчанию: `2.0`

- `--w-hub FLOAT`  
  Вес hub penalty / bonus в rerank.  
  По умолчанию: `0.3`

- `--community-min INT`  
  Нижняя граница размера сообщества на postprocess-этапе.  
  По умолчанию: `2`

- `--community-max INT`  
  Верхняя граница размера сообщества на postprocess-этапе.  
  По умолчанию: `50`

- `--sample-rate INT`  
  Частота дискретизации для декода.  
  По умолчанию: `16000`

- `--max-duration-sec FLOAT`  
  Максимальная длина одного аудио в секундах после обрезки.  
  По умолчанию: `60.0`

- `--knn-chunk-size INT`  
  Размер чанка при расчете kNN / candidate pool.  
  По умолчанию: `1024`

- `--device STRING`  
  Устройство для инференса. Обычно `cuda` или `cpu`.  
  По умолчанию: `cuda`, если CUDA доступна, иначе `cpu`

</details>

### Что важно знать

- `candidate_pool_k` больше не передается отдельным аргументом.
- Внутри скрипта он всегда вычисляется автоматически как `k + 200`.
- `rerank_cpu_workers` больше не передается отдельным аргументом.
- Для decode и rerank теперь используется один параметр: `--num-workers`.

### Минимальный пример

```bash
docker compose -f docker-compose.submission.yml run --rm \
  -v /abs/path/to/audio:/data/audio:ro submission \
  --audio-root /data/audio
```

### Пример с дополнительными параметрами

```bash
docker compose -f docker-compose.submission.yml run --rm \
  -v /abs/path/to/audio:/data/audio:ro submission \
  --audio-root /data/audio \
  --k 20 \
  --num-workers 8 \
  --gpu-model-instances 3 \
  --mean-steps 50 \
  --n-iter 4 \
  --output-csv /workspace/outputs/custom_output.csv
```

## Документация

Минимальный набор документов, чтобы понять проект от установки до инференса:

| Документ | Про что |
|----------|---------|
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Валидация во время трейна, submission-пайплайн (PyTorch и TensorRT). |
| [`docs/INSTALLATION.md`](docs/INSTALLATION.md) | Conda + uv, ffmpeg / torchcodec, базовая проверка окружения. |
| [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) | С какими датасетами работаем (VoxCeleb1/2, CN-Celeb, 3D-Speaker, Kryp, KeSpeech, tidy), MUSAN/RIRS. |
| [`docs/MANIFESTS.md`](docs/MANIFESTS.md) | Схема Parquet-манифеста, `build_parquet_manifest.py`, конкатенация, длительности. |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Как запустить трейн (`run.sh`, Accelerate/FSDP, чекпоинты, ресюм, валидация). |
| [`docs/CONFIGS.md`](docs/CONFIGS.md) | Карта Hydra-конфигов: model / data / augment / loss / logging / accelerate. |
| [`docs/DATA_PREPROCESSING_PIPELINE.md`](docs/DATA_PREPROCESSING_PIPELINE.md) | Детальный граф обработки внутри датасета и collate. |
| [`docs/TENSORRT_CONVERSION_AND_BENCHMARKING.md`](docs/TENSORRT_CONVERSION_AND_BENCHMARKING.md) | Экспорт чекпоинта в `.plan` + бенч TensorRT. |
| [`docs/Docker (TensorRT).md`](<docs/Docker (TensorRT).md>) | Сборка TensorRT-образа, прогрев движка и инференс через `docker compose`. |

## Быстрая шпаргалка

```bash
# 1. Быстрый eval через Docker (см. секцию выше / docs/EVALUATION.md)
docker compose -f docker-compose.submission.yml run --rm --build \
  -v /path/to/audio:/data/audio:ro submission \
  --audio-root /data/audio

# 2. Собрать окружение (см. docs/INSTALLATION.md)
conda create -n asv python=3.10 -y && conda activate asv
pip install uv
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked

# 3. Собрать манифест (см. docs/MANIFESTS.md)
python scripts/build_parquet_manifest.py --dataset voxceleb2 \
  /path/VoxCeleb2 /path/output_dir

# 4. Настроить run.sh (пути, batch, epochs) и запустить (см. docs/TRAINING.md)
bash run.sh

# 5. Инференс на submission-директории (см. docs/EVALUATION.md):
# - PyTorch:
bash submission/reranking/run_embeddings.sh
# - TensorRT:
./submission/reranking/run_submission_embeddings_trt_pipeline.sh --audio-dir /path/to/audio
```

## Структура репозитория

- `asv/` — код проекта: `data/`, `augment/`, `codec/`, `models/`, `losses/`, `train/`, `validation/`, `manifest/`, `logging/`.
- `configs/` — Hydra-конфиги (группы `model/`, `data/`, `augment/`, `loss/`, `logging/`, плюс `accelerate/` для FSDP).
- `scripts/` — утилиты: сборка манифестов, бенчмарки датасета, генерация кодек-вариантов, визуализации.
- `benchmarking/` — экспорт в ONNX/TensorRT, бенч движков и сравнение вариантов.
- `submission/reranking/` — скрипты инференса (PyTorch и TRT) + mean-chain v2 rerank.
- `run.sh` — единая точка запуска трейна (Accelerate + FSDP + Hydra-overrides).
- `train.py` — Hydra-entrypoint (тонкий, делегирует в `asv/train/run.py`).
- `Dockerfile.trt`, `docker-compose.trt.yml`, `trt-runtime-entrypoint.sh` — образ с TensorRT-runtime для инференса.
