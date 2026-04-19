# Оценка и инференс

Два сценария:

1. **Валидация во время трейна** — VoxCeleb1 EER, запускается встроенно в цикл обучения.
2. **Submission-инференс** — считаем эмбеддинги по директории аудио и ре-ранжируем соседей (mean-chain v2). Два бэкенда: PyTorch (Accelerate) и TensorRT.

## 1. Валидация во время трейна

Всё ставится в `run.sh` (см. также [`TRAINING.md`](TRAINING.md) и `asv/validation/voxceleb.py`).

- `VALIDATION_ENABLED=true` — включает.
- `VOXCELEB_VAL_ROOT`, `VOXCELEB_TRIAL_FILES` — корень VoxCeleb и список trial-файлов.
- `VALIDATE_EVERY_N_STEPS=N` — валидация каждые N optimizer-шагов + в конце каждой эпохи.
- `VALIDATE_BEFORE_FIRST_EPOCH=true` — прогнать валидацию до первого шага как baseline.

Логи:
- `val/eer/<trial_stem>` — EER по каждому trial-файлу.
- `val/eer/pooled` — pooled EER по всем триалам.

Ещё в `validation_metrics.jsonl` (в Hydra runtime dir) — каждая строка одна валидация с тегом (`epoch_XXX_end`, `epoch_XXX_step_NNN`, …).

Variant-валидация (устойчивость к кодекам/аугментациям):
- `VARIANT_VAL_ENABLED=true`, `VARIANT_VAL_SCAN_ROOT=<дирректория с поддирами>`.
- На каждую поддиректорию — отдельный EER (`val/variants/eer_<name>`) и глобальный `val/eer/pooled_hard`.

## 2. Submission — PyTorch (`run_embeddings.py`)

Кастовый инференс без TRT, через Accelerate на N GPU.

Быстрый запуск — скрипт `submission/reranking/run_embeddings.sh` (поправьте пути внутри). По сути он вызывает:

```bash
accelerate launch \
  --multi_gpu --num_processes <N> --num_machines 1 \
  --mixed_precision no --dynamo_backend no \
  submission/reranking/run_embeddings.py \
  --weights /path/to/checkpoint \
  --manifest-path /path/to/voxceleb1_manifest.parquet \
  --voxceleb-root /path/to/VoxCeleb \
  --output-pkl /path/to/embeddings.pkl \
  --model-config configs/model/redimnet2_b6_sphereface2.yaml \
  --trial-files /path/to/voxceleb1_o.txt /path/to/voxceleb_e.txt /path/to/voxceleb_h.txt \
  --batch-size 1 \
  --num-workers 4 \
  --sample-rate 16000 \
  --trim-num-samples 0
```

- `--weights` — путь к чекпоинту: директория Accelerate (`epoch_*`) или отдельный файл (`.safetensors`, `.bin`, `.pt`). Скрипт сам распознаёт и снимает `module.` / `_orig_mod.` префиксы.
- `--trim-num-samples 0` — использовать полную длину файла (без обрезки). Для fixed-length используйте, например, `96000`.
- `--audio-root <dir>` вместо `--manifest-path` — рекурсивно сканирует директорию и считает эмбеддинги по всем аудио.

На выходе — pickle `{"embeddings": np.ndarray [N, D], "audio_ids": list[str], "trial_rel_paths": list[str], "audio_paths": list[str]}`.

## 3. Submission — TensorRT (`run_embeddings_trt.py`)

Путь быстрее и детерминированнее. Использует готовый `.plan` движок (см. [`TENSORRT_CONVERSION_AND_BENCHMARKING.md`](TENSORRT_CONVERSION_AND_BENCHMARKING.md) про экспорт).

Запуск одной командой через оболочку (`submission/reranking/run_submission_embeddings_trt_pipeline.sh`):

```bash
./submission/reranking/run_submission_embeddings_trt_pipeline.sh \
  --audio-dir /path/to/audio
```

Обязательно: `--audio-dir <dir>` — директория с аудио (flac/wav/mp3/…).

Необязательно:
- `--engine-path <plan>` — напрямую указать `.plan`; batch/num_samples можно прочитать из имени вида `..._bs32_s96000.plan` автоматически (или передать `--trt-batch-size`, `--trt-num-samples`).
- Либо `--summary-json` / `--variant <key>` — брать движок из `export_summary.json` (по умолчанию `artifacts/export_summary.json` + вариант `frontend_fp32_model_fp16`).
- `--output-dir <dir>` — куда класть pickle и CSV (по умолчанию `submission_pipeline/`).
- `--num-workers N` — воркеры декода (они же `--cpu-workers` у rerank).
- `--k N` — сколько соседей сохранить (по умолчанию `10`).

Что скрипт делает:
1. `run_embeddings_trt.py` → эмбеддинги в `<output-dir>/embeddings_trt.pkl`.
2. `rerank_embeddings_mean_chain_v2_fast.py` → ре-ранг соседей в CSV с суффиксом `.neighbors.csv`.

Для Docker-сценария (prebake TRT-движок внутри контейнера + монтирование аудио) — см. `docs/README.md` «Quick start» и `docker-compose.trt.yml`.

## 4. Ре-ранжирование соседей

`submission/reranking/rerank_embeddings_mean_chain_v2_fast.py` — mean-chain v2 с графовым ре-ранком. Входы:
- `embeddings.pkl` с ключами `embeddings`, `audio_ids`.
- `--k <N>` — сколько соседей сохранить на файл.
- `--output-prefix <prefix>` — куда класть результат (будут суффиксы `.neighbors.csv`, `.metrics.json` и т.п.).
- `--cpu-workers <N>` — параллелизм на CPU.
- `--device cuda|cpu` — для основных тензорных операций.
- `--skip-metrics` — не считать метрики, просто выдать топ-K.

CSV соседей имеет колонки `filepath,neighbours`, где `neighbours` — 10 (или сколько указали) относительных путей через запятую в кавычках.

## 5. TensorRT: экспорт и прогрев

Короткая выжимка (полная — в [`TENSORRT_CONVERSION_AND_BENCHMARKING.md`](TENSORRT_CONVERSION_AND_BENCHMARKING.md)):

```bash
python benchmarking/export_redimnet2_to_tensorrt.py \
  --model_name b6 \
  --checkpoint outputs/.../epoch_0005_step_00014800 \
  --model_config configs/model/redimnet2_b6.yaml \
  --engine_precision fp16 \
  --export_dtype fp16 \
  --batch_size 32 \
  --num_samples 96000 \
  --engine_path benchmarking/outputs/...plan \
  --report_json benchmarking/outputs/...report.json
```

Frontend (mel) всегда fp32, тело модели — fp16 / fp32 по выбору. Батч и `num_samples` запекаются в `.plan` и потом подсасываются из имени файла.

Bench на VoxCeleb trial'ах — `benchmarking/benchmark_onnx_speed.py` (пример запуска в том же файле `TENSORRT_CONVERSION_AND_BENCHMARKING.md`).

## 6. Частые грабли

- **`run.sh` отказывается стартовать** — не закоммитил изменения: `git status --porcelain` не пустой.
- **OOM в submission-инференсе** — уменьшите `--batch-size`, уберите `mixed_precision` из Accelerate или включите его (`bf16`), либо переключитесь на TRT-путь.
- **TRT: per-batch corruption после первой группы файлов** — это был баг sync между CUDA-стримами (fix уже в коде, см. коммит `98d0784`). Если собираете движок в другом окружении, проверьте `benchmarking/benchmark_onnx_speed.py::TrtRunner.submit` — там есть `self.stream.wait_stream(torch.cuda.default_stream(...))`.
- **Resume не обновляет веса** — `train.verify_checkpoint_load=true` упадёт на старте. Скорее всего неверный путь к чекпоинту или кривой `latest.json`.
