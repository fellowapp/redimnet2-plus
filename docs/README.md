# ASV — документация

Минимальный набор документов, чтобы понять проект от установки до инференса. Читать в таком порядке:

| Документ | Про что |
|----------|---------|
| [`INSTALLATION.md`](INSTALLATION.md) | Conda + uv, ffmpeg / torchcodec, базовая проверка окружения. |
| [`DATA_PREPARATION.md`](DATA_PREPARATION.md) | С какими датасетами работаем (VoxCeleb1/2, CN-Celeb, 3D-Speaker, Kryp, KeSpeech, tidy), MUSAN/RIRS. |
| [`MANIFESTS.md`](MANIFESTS.md) | Схема Parquet-манифеста, `build_parquet_manifest.py`, конкатенация, длительности. |
| [`TRAINING.md`](TRAINING.md) | Как запустить трейн (`run.sh`, Accelerate/FSDP, чекпоинты, ресюм, валидация). |
| [`CONFIGS.md`](CONFIGS.md) | Карта Hydra-конфигов: model / data / augment / loss / logging / accelerate. |
| [`EVALUATION.md`](EVALUATION.md) | Валидация во время трейна, submission-пайплайн (PyTorch и TensorRT). |
| [`DATA_PREPROCESSING_PIPELINE.md`](DATA_PREPROCESSING_PIPELINE.md) | Детальный граф обработки внутри датасета и collate. |
| [`TENSORRT_CONVERSION_AND_BENCHMARKING.md`](TENSORRT_CONVERSION_AND_BENCHMARKING.md) | Экспорт чекпоинта в `.plan` + бенч TensorRT. |
| [`DOCKER.md`](DOCKER.md) | Сборка TensorRT-образа, прогрев движка и инференс через `docker compose`. |

## Быстрая шпаргалка

```bash
# 1. Собрать окружение (см. INSTALLATION.md)
conda create -n asv python=3.10 -y && conda activate asv
pip install uv
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked

# 2. Собрать манифест (см. MANIFESTS.md)
python scripts/build_parquet_manifest.py --dataset voxceleb2 \
  /path/VoxCeleb2 /path/output_dir

# 3. Настроить run.sh (пути, batch, epochs) и запустить (см. TRAINING.md)
bash run.sh

# 4. Инференс на submission-директории (см. EVALUATION.md):
# — PyTorch:
bash submission/reranking/run_embeddings.sh
# — TensorRT:
./submission/reranking/run_submission_embeddings_trt_pipeline.sh --audio-dir /path/to/audio
```

## Структура репозитория (коротко)

- `asv/` — код проекта: `data/`, `augment/`, `codec/`, `models/`, `losses/`, `train/`, `validation/`, `manifest/`, `logging/`.
- `configs/` — Hydra-конфиги (группы `model/`, `data/`, `augment/`, `loss/`, `logging/`, плюс `accelerate/` для FSDP).
- `scripts/` — утилиты: сборка манифестов, бенчмарки датасета, генерация кодек-вариантов, визуализации.
- `benchmarking/` — экспорт в ONNX/TensorRT, бенч движков и сравнение вариантов.
- `submission/reranking/` — скрипты инференса (PyTorch и TRT) + mean-chain v2 rerank.
- `run.sh` — единая точка запуска трейна (Accelerate + FSDP + Hydra-overrides).
- `train.py` — Hydra-entrypoint (тонкий, делегирует в `asv/train/run.py`).
- `Dockerfile.trt`, `docker-compose.trt.yml`, `trt-runtime-entrypoint.sh` — образ с TensorRT-runtime для инференса.
