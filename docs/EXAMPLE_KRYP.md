# Пример: трейн и submission на Kryp (1 GPU)

Пошаговый walkthrough на одном датасете (Kryp speaker-id challenge) и одной видеокарте. Для других датасетов — см. [`DATA_PREPARATION.md`](DATA_PREPARATION.md) + [`MANIFESTS.md`](MANIFESTS.md); для multi-GPU — [`TRAINING.md`](TRAINING.md).

## 0. Что у нас есть

Пусть датасет распакован в `~/cryp_data/kryp/`:

```
~/cryp_data/kryp/
├── _download_mailru_public.py
├── _manifest_build.log
├── kryp_train_manifest.parquet    # уже собранный манифест (опционально)
├── train/
│   └── <speaker_id>/<NNNNN>.flac  # тренировочное аудио
├── train.csv                       # columns: speaker_id, filepath
├── train.tar.gz
├── test_public/
│   └── <file>.flac                 # аудио для submission
└── test_public.csv                 # список файлов для submission
```

Всё в 16 kHz моно FLAC.

## 1. Окружение

Ставится один раз, см. [`INSTALLATION.md`](INSTALLATION.md):

```bash
conda create -n asv python=3.10 --no-default-packages -y
conda activate asv
pip install uv
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked

# для torchcodec:
conda install -y ffmpeg
uv pip install nvidia-npp
cp activate_asv_cuda_libs.sh "$CONDA_PREFIX/etc/conda/activate.d/asv_cuda_libs.sh"
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/asv_cuda_libs.sh"
```

## 2. Манифест

### Если он не собран — собрать

```bash
conda activate asv
cd /path/to/ASV

python scripts/build_parquet_manifest.py \
  --dataset kryp \
  ~/cryp_data/kryp \
  ~/cryp_data/kryp
```

Скрипт читает `~/cryp_data/kryp/train.csv` (колонки `speaker_id`, `filepath`), обходит `train/<spk>/*.flac` и пишет `~/cryp_data/kryp/kryp_train_manifest.parquet`. В stdout будет:

```
Wrote .../kryp_train_manifest.parquet (N rows)
num_classes=<K>
ASV_NUM_CLASSES=<K>
```

Запомните `K` — это число уникальных спикеров, оно же `ASV_NUM_CLASSES`.

### Если манифест уже есть

Можно просто прочитать число классов руками:

```bash
python -c "
import pyarrow.parquet as pq
f = pq.ParquetFile('/home/kirill/cryp_data/kryp/kryp_train_manifest.parquet', memory_map=True)
speakers = set()
for rg in range(f.num_row_groups):
    speakers.update(str(s) for s in f.read_row_group(rg, columns=['speaker_id']).column('speaker_id').to_pylist())
print('num_classes =', len(speakers))
"
```

## 3. Трейн на 1 GPU

### Accelerate-конфиг

Уже есть: `configs/accelerate/single_gpu_bf16.yaml` (`distributed_type: NO`, 1 процесс, bf16). В отличие от FSDP-конфигов, он **не** шардит параметры — на одной GPU шардить нечего.

### run.sh — что поправить

Откройте `run.sh` и замените верхний блок под single-GPU:

```bash
# --- под Kryp + 1 GPU ---
MANIFEST_PATH="/home/kirill/cryp_data/kryp/kryp_train_manifest.parquet"
ASV_NUM_CLASSES=<число_спикеров_из_шага_2>
MUSAN_PATH="/path/to/musan"        
RIRS_PATH="/path/to/RIRS_NOISES"   

MODEL_CFG="redimnet2_b6_sphereface2.yaml"
LOSS_CFG="sphereface2"
BATCH_PER_GPU=8                  
WORKERS_PER_GPU=4
PIN_MEMORY=true
PREFETCH_FACTOR=2
TOTAL_EPOCHS=4
GRAD_ACCUM_STEPS=4              
MAX_GRAD_NORM=20.0
START_LR=3e-5
FINAL_LR=5e-6
LOG_EVERY_STEPS=1
INIT_WEIGHTS=""                    # если хотите старт из претрейна — укажите путь
INIT_MODE="full"
RESUME_FROM_CHECKPOINT=""

NUM_GPUS=1
MIXED_PRECISION="bf16"
ACCELERATE_CONFIG="configs/accelerate/single_gpu_bf16.yaml"

# Валидация на VoxCeleb1 — не обязательна для Kryp-only трейна:
VALIDATION_ENABLED=false
VARIANT_VAL_ENABLED=false
VALIDATE_BEFORE_FIRST_EPOCH=false
```

Важно: в `configs/config.yaml` есть `train.expect_fsdp: true` — это только флаг для лога, на 1 GPU без FSDP ничего не сломает, можно оставить. Если мешает warning — добавьте в `run.sh` в блок вызова `accelerate launch` параметр `train.expect_fsdp=false`.

Если выключаете MUSAN/RIRS (нет датасетов под рукой) — передайте пустые `MUSAN_PATH=""` и **плюс** CLI-overrides:

```bash
# добавить в команду accelerate launch:
augment.waveform_levels.1.p=0.0 \
augment.waveform_levels.2.p=0.0
```

(это обнулит «additive_noise» и «rir» уровни в `configs/augment/default.yaml`).

### Запуск

```bash
cd /path/to/ASV
conda activate asv
bash run.sh
```

`run.sh` отказывается стартовать, если есть uncommitted-изменения. Закоммитьте свои правки (или временно закомментируйте проверку в `run.sh`).

Чекпоинты пишутся в `outputs/<MLOP_RUN_NAME>/<date>/<time>/checkpoints/epoch_XXXX_step_XXXXXXXX/`. Последний фиксируется в `latest.json`.

## 4. Инференс на test_public: эмбеддинги

Нужен путь к тренированному чекпоинту из шага 3, например:
`outputs/trainin_b6_lmft_long/2026-04-20/12-00-00/checkpoints/epoch_0004_step_00008000/`.

### Вариант A — PyTorch (одна GPU, без TRT)

```bash
accelerate launch \
  --config_file configs/accelerate/single_gpu_bf16.yaml \
  --num_processes 1 \
  --mixed_precision no \
  --dynamo_backend no \
  submission/reranking/run_embeddings.py \
  --weights /path/to/checkpoints/epoch_0004_step_00008000 \
  --audio-root /home/kirill/cryp_data/kryp/test_public \
  --output-pkl /home/kirill/cryp_data/kryp/embeddings_test_public.pkl \
  --model-config configs/model/redimnet2_b6_sphereface2.yaml \
  --batch-size 4 \
  --num-workers 4 \
  --sample-rate 16000 \
  --trim-num-samples 96000
```

- `--trim-num-samples 96000` — центр-кроп до 6 секунд (совпадает с трейном). Хотите полную длину — `--trim-num-samples 0`.
- `--audio-root` сканирует директорию рекурсивно; `test_public.csv` сам по себе не читается, нам нужен список реальных файлов — они берутся с файловой системы.

На выходе — `embeddings_test_public.pkl` со словарём `{embeddings, audio_ids, trial_rel_paths, audio_paths}` (L2-нормализованные).

### Вариант B — TensorRT (быстрее)

Сначала экспорт `.plan` (см. [`TENSORRT_CONVERSION_AND_BENCHMARKING.md`](TENSORRT_CONVERSION_AND_BENCHMARKING.md)):

```bash
python benchmarking/export_redimnet2_to_tensorrt.py \
  --model_name b6 \
  --checkpoint /path/to/checkpoints/epoch_0004_step_00008000 \
  --model_config configs/model/redimnet2_b6.yaml \
  --engine_precision fp16 \
  --export_dtype fp16 \
  --batch_size 32 \
  --num_samples 96000 \
  --engine_path artifacts/kryp_b6_fp16_bs32_s96000.plan \
  --report_json artifacts/kryp_b6_fp16_bs32_s96000.report.json
```

Потом пайплайн (эмбеддинги + реранк одной командой):

```bash
./submission/reranking/run_submission_embeddings_trt_pipeline.sh \
  --audio-dir /home/kirill/cryp_data/kryp/test_public \
  --engine-path artifacts/kryp_b6_fp16_bs32_s96000.plan \
  --output-dir ~/cryp_data/kryp/submission_outputs \
  --num-workers 4 \
  --k 10 \
  -- \
  --neighbors-filepath-prefix test_public
```

Скрипт сам вытащит `batch_size=32`, `num_samples=96000` из имени `.plan` и положит результат в `~/cryp_data/kryp/submission_outputs/`:
- `embeddings_trt.pkl` — эмбеддинги
- `embeddings_trt_mean_chain_v2.neighbors.csv` — **это и есть submission-файл**

## 5. Собрать submission (реранк + CSV)

Если делали Вариант A (PyTorch), реранк надо запустить отдельно:

```bash
python submission/reranking/rerank_embeddings_mean_chain_v2_fast.py \
  /home/kirill/cryp_data/kryp/embeddings_test_public.pkl \
  --k 10 \
  --output-prefix /home/kirill/cryp_data/kryp/submission \
  --cpu-workers 4 \
  --neighbors-filepath-prefix test_public \
  --skip-metrics
```

На выходе:
- `submission.neighbors.csv` — `filepath,neighbours` на каждый файл из `test_public/` (10 соседей в кавычках через запятую). **Это submission.**
- `submission.metrics.json` — ничего полезного (метрик нет, только тайминг).

Колонка `filepath` будет вида `test_public/<name>.flac` — это задаёт `--neighbors-filepath-prefix test_public`. Нужен другой префикс — пропишите его.

## 6. Проверить формат

Submission-CSV должен выглядеть так:

```
filepath,neighbours
test_public/abc123.flac,"test_public/xxx.flac,test_public/yyy.flac,test_public/zzz.flac,..."
test_public/def456.flac,"test_public/aaa.flac,..."
```

- Одна строка на каждый файл из `test_public/`.
- 10 соседей в поле `neighbours` через запятую, в двойных кавычках.
- Первой строкой — заголовок `filepath,neighbours`.

Если формат submission-платформы строже (например, нужен другой префикс или другое число соседей) — меняйте `--neighbors-filepath-prefix` и `--k`.

## 7. Типичные вопросы

- **OOM на трейне:** уменьшите `BATCH_PER_GPU`, увеличьте `GRAD_ACCUM_STEPS`. Для b6 на 24GB GPU комфортно 8–12 при 6-секундных чанках.
- **OOM на инференсе:** для PyTorch — уменьшите `--batch-size`; для TRT — пересоберите `.plan` с меньшим `--batch_size`.
- **Долгий трейн:** 4 эпохи на одной GPU — это много часов. Для быстрого смоука включите `SINGLE_BATCH_TEST=true bash run.sh` — один батч, sanity-проверка пайплайна.
- **Трейн падает на кодек-аугментациях:** проверьте ffmpeg (`python scripts/generate_codec_audio_variants.py <file> --output_dir ./tmp`). Или выключите кодек-ветку через CLI: `data.transform_probabilities.codec_only=0.0 data.transform_probabilities.codec_then_augment=0.0` и соответственно поднимите `keep_original`/`augment_only` чтобы сумма = 1.0 (эта таблица промаркирована как эксперимент-лок в дефолтах; здесь вы её переопределяете осознанно).
- **Валидация VoxCeleb не нужна на Kryp-трейне:** выключите через `VALIDATION_ENABLED=false` и `VARIANT_VAL_ENABLED=false` в `run.sh`.
