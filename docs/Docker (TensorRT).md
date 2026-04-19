# Docker (TensorRT) и оценка

Рабочая директория в контейнере: **`/opt/asv`**.

---

**GPU НУЖЕН ВСЕГДА** (в `docker-compose.trt.yml` задано `gpus: all`): без видеокарты этот образ не имеет смысла — и пребейк TensorRT, и эмбеддинги, и smoke-тест идут на GPU.

**ПРИ ПЕРВОМ ЗАПУСКЕ КОНТЕЙНЕРА ENTRYPOINT ДОЛГО СОБИРАЕТ TENSORRT-ДВИЖОК: ЭКСПОРТ ONNX, КОМПИЛЯЦИЯ ПЛАНА — ЭТО МОЖЕТ ЗАНЯТЬ МНОГО МИНУТ. В ЭТО ВРЕМЯ GPU ЗАНЯТ — ТАК РАБОТАЕТ TENSORRT. ДОЖДИТЕСЬ ОКОНЧАНИЯ ЛОГОВ, ПОКА НЕ ЗАПУСКАТЕ ПАЙПЛАЙН ОЦЕНКИ.**

Папка **`./artifacts`** на хосте примонтирована в **`/opt/asv/artifacts`**: туда пишется экспорт TensorRT (`export_summary.json`, `.plan` и т.п.).

---

## Быстрый старт

1. Веса лежат в **`weights/`** в корне репозитория (в `docker-compose.trt.yml` зафиксировано `ASV_CHECKPOINT: weights`).

2. Собрать образ и поднять контейнер (одна команда):

```bash
docker compose up --build asv-trt-runtime
```

Когда **TensorRT-движок и smoke-тест** успешно завершатся, в логах контейнера будет что-то в таком духе (имя `.plan` зависит от чекпоинта и варианта):

```text
Wrote export summary to /opt/asv/artifacts/export_summary.json
{
  "engine_path": "/opt/asv/artifacts/redimnet2_b6_weights_frontend_fp32_model_fp16_bs32_s96000.plan",
  "batch_size": 32,
  "num_samples": 96000,
  "output_shape": [32, 192],
  "output_dtype": "torch.float16",
  "input_names": ["waveform"],
  "output_names": ["embeddings"],
  "gpu_id": 0
}
```

Строка **`Wrote export summary to .../export_summary.json`** и JSON smoke-теста с **`engine_path`** / **`batch_size`** / **`num_samples`** — признак, что пребейк прошёл и можно запускать пайплайн оценки.

3. **Смонтировать папку с аудио** и запустить скрипт одной командой — флаг **`-v` на хосте:путь внутри контейнера** (чтение только — `:ro`):

```bash
docker compose run --rm \
  -v /абсолютный/путь/к/аудио:/data/audio:ro \
  asv-trt-runtime \
  /opt/asv/submission/reranking/run_submission_embeddings_trt_pipeline.sh \
  --audio-dir /data/audio
```

Замените **`/абсолютный/путь/к/аудио`** на свой каталог. Внутри контейнера путь **`/data/audio`** может быть любым, главное — тот же в **`--audio-dir`**.

4. Зайти в оболочку (при необходимости тоже с аудио):

```bash
docker compose run --rm -v /абсолютный/путь/к/аудио:/data/audio:ro asv-trt-runtime bash
# далее внутри:
# ./submission/reranking/run_submission_embeddings_trt_pipeline.sh --audio-dir /data/audio
```

Результаты по умолчанию: **`./submission_outputs`** на хосте → **`/opt/asv/submission_outputs`** в контейнере.

---

## Подробности: build-args, переменные, движок

### Сборка и веса

- В **`docker-compose.trt.yml`** задано **`ASV_CHECKPOINT: weights`**: в образ копируется каталог **`weights/`**. Менять build-args можно, но обычно не нужно, если веса уже в `weights/`.
- Остальные build-args (`ASV_MODEL_NAME`, `ASV_MODEL_CONFIG`, `ASV_BATCH_SIZE`, `ASV_NUM_SAMPLES`, …) задают параметры вызова `export_checkpoint_to_tensorrt.py` при первом старте.

### Пребейк

- **`ASV_PREBAKE_ENGINES: "1"`** — при отсутствии сводки entrypoint запускает экспорт ONNX + сборку **`.plan`** и `trt_smoke_test.py`. Итог: **`/opt/asv/artifacts/export_summary.json`**.

### Пайплайн `run_submission_embeddings_trt_pipeline.sh`

- **`--audio-dir`** — обязательный аргумент (дерево `.wav` / `.flac` / `.mp3` …).
- Движок и размеры батча подставляются из **`ASV_EXPORT_SUMMARY`** после пребейка. **`--engine-path`** — только переопределение; без явных `--trt-batch-size` / `--trt-num-samples` можно взять из имени вида `..._bs32_s96000.plan`.
- **`ASV_TRT_NUM_WORKERS`** / **`--num-workers`** — воркеры декода и **`--cpu-workers`** rerank.
- Rerank внутри скрипта всегда с **`--skip-metrics`**.

### Где задано в compose

- Том **`./artifacts:/opt/asv/artifacts`** (экспорт TRT), **`./submission_outputs`**, **`ASV_OUTPUT_DIR`**, **`ASV_EXPORT_SUMMARY`**, **`ASV_DEFAULT_VARIANT`**.
