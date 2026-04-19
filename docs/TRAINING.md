# Трейн

Полный цикл: манифест → DataLoader (декод + codec/augment) → spec-frontend → model (Encoder→Bridge→Classifier) → loss → Accelerate/FSDP.

## Что происходит при запуске

Точка входа — `train.py` (Hydra + `asv/train/run.py`), дальше `asv/train/loop.py::train()`. Он:

1. Строит модель (`build_model(cfg.model)`) и loss (`build_loss(cfg.loss)`).
2. (Опционально) инициализирует веса из чекпоинта: `train.init_weights` + `train.init_mode`.
3. `torch.compile(mode=cfg.train.torch_compile_mode)`.
4. Поднимает `Accelerator` с `mixed_precision` и `gradient_accumulation_steps`, готовит DataLoader.
5. Может ресюмить Accelerate-чекпоинт (`train.resume_from_checkpoint`).
6. Основной цикл эпох: forward, loss, `accelerator.backward`, `clip_grad_norm`, cosine scheduler.
7. Валидация: каждые `validation.every_n_steps` шагов и в конце каждой эпохи.
8. Чекпоинт: каждые `train.save_every_epochs` эпох (или на валидации, если та запускается по шагам).

## Быстрый старт

Всё сведено в `run.sh`. Откройте, настройте верхний блок, запустите:

```bash
bash run.sh
```

Минимальный набор, который реально крутим каждый раз:

| Переменная в `run.sh`       | Что это                                                                 |
|-----------------------------|------------------------------------------------------------------------|
| `MANIFEST_PATH`             | Parquet-манифест (см. [`MANIFESTS.md`](MANIFESTS.md))                  |
| `ASV_NUM_CLASSES`           | Число уникальных `speaker_id` в манифесте                              |
| `MUSAN_PATH`, `RIRS_PATH`   | Корни MUSAN/RIRS для аугментаций                                       |
| `MODEL_CFG`                 | Имя YAML из `configs/model/` (без `.yaml`)                             |
| `LOSS_CFG`                  | Имя YAML из `configs/loss/`                                            |
| `BATCH_PER_GPU`             | Размер батча на одну GPU                                               |
| `WORKERS_PER_GPU`           | `num_workers` у DataLoader                                             |
| `TOTAL_EPOCHS`              | Сколько эпох гоняем                                                    |
| `GRAD_ACCUM_STEPS`          | Gradient accumulation                                                  |
| `START_LR`, `FINAL_LR`      | Cosine: от `START_LR` до `FINAL_LR` (по числу optimizer steps)         |
| `MAX_GRAD_NORM`             | Клиппинг градиентов                                                    |
| `INIT_WEIGHTS`, `INIT_MODE` | Fresh-start init (не ресюм!); `init_mode: full/encoder_only/encoder_bridge` |
| `RESUME_FROM_CHECKPOINT`    | Accelerate-ресюм; пустая строка = с нуля, `"latest"` = из `latest.json`|
| `NUM_GPUS`                  | Число процессов (GPU)                                                  |
| `ACCELERATE_CONFIG`         | Конфиг FSDP (см. `configs/accelerate/fsdp_*gpu_bf16.yaml`)             |

`run.sh` запрещает стартовать трейн, если в working tree есть неcommitted изменения (`git status --porcelain`). Закоммитить или пушнуть, либо временно закомментировать проверку.

## Валидация

Настраивается в `run.sh`:

```bash
VALIDATION_ENABLED=true
VOXCELEB_VAL_ROOT="/data/.../VoxCeleb"
VOXCELEB_TRIAL_FILES="[${VOXCELEB_VAL_ROOT}/voxceleb1_o.txt,${VOXCELEB_VAL_ROOT}/voxceleb_e.txt,${VOXCELEB_VAL_ROOT}/voxceleb_h.txt]"
VALIDATE_BEFORE_FIRST_EPOCH=true
VALIDATE_EVERY_N_STEPS=400
```

При `run_before_first_epoch=true` EER считается до первого шага — удобно как baseline.

Variant-валидация (устойчивость к кодекам/аугментациям):
```bash
VARIANT_VAL_ENABLED=true
VARIANT_VAL_SCAN_ROOT="/data/.../VoxCelebAug"
```
Каждая непустая поддиректория `VARIANT_VAL_SCAN_ROOT` → отдельный EER (`val/variants/eer_<subdir>`), плюс pooled-hard EER по всем.

## Mixed precision и FSDP

Конфиги Accelerate:
- `configs/accelerate/fsdp_2gpu_bf16.yaml`
- `configs/accelerate/fsdp_4gpu_bf16.yaml`
- `configs/accelerate/fsdp_6gpu_bf16.yaml`

Все три — FSDP FULL_SHARD, compute/param/reduce dtype = bf16, `state_dict_type: FULL_STATE_DICT`. Менять под другое число GPU — просто скопировать и поправить `gpu_ids` / `num_processes`. `main_process_port` стоит сделать уникальным на хост, если крутите несколько трейнов параллельно.

## Чекпоинты

- Корень: `${CHECKPOINT_DIR}` (по умолчанию `${output_dir}/checkpoints`, где `output_dir` — это Hydra runtime dir под `outputs/<run_name>/<date>/<time>/`).
- Один чекпоинт = поддиректория `epoch_XXXX_step_XXXXXXXX/`.
- `latest.json` в корне чекпоинтов указывает на самый свежий.
- `training_state.json` внутри чекпоинта хранит `epoch`, `global_step`, `optimizer_step`, `next_epoch` — нужно для корректного ресюма.

Ресюм:
```bash
RESUME_FROM_CHECKPOINT="latest"                   # из ${CHECKPOINT_DIR}/latest.json
RESUME_FROM_CHECKPOINT="/path/epoch_0005_step_00014800"  # абсолютный путь
```

Пост-ресюм проверка (`train.verify_checkpoint_load: true`) — сравнивает сигнатуру весов до/после `load_state` и падает, если ничего не изменилось (защита от кривого пути).

Fresh-start init (начинаем новую стадию из старых весов, без оптимизатора):
```bash
INIT_WEIGHTS="/path/epoch_0005_step_00014800"
INIT_MODE="full"   # full | encoder_only | encoder_bridge
```
`init_weights` и `resume_from_checkpoint` взаимоисключаются — трейн падает, если заданы оба.

## Single-batch test

Для быстрой sanity-проверки трейн-пайплайна (единственный батч, EER ≈ 0 на его же данных):

```bash
SINGLE_BATCH_TEST=true bash run.sh
```

Должен завершиться `val/single_batch/eer <= 1e-6`, иначе где-то развалилось (модель, loss, формы, аугментации).

## Сколько всё бежит (ориентир)

На `b6` + 6 GPU + FSDP bf16 + batch/gpu=24 — порядка 1.5 сэмплов/с на GPU без codec-аугментаций, меньше с кодеками. Конкретные числа сильно зависят от железа и доли кодек-ветки (`transform_probabilities.codec_only` + `codec_then_augment`). Логи показывают EMA loss и grad_norm на каждом шаге.

## Скрипты, которые могут пригодиться

- `scripts/benchmark_data_pipeline.py` — прогнать DataLoader без модели, понять где упёрлись (декод, кодек, аугментация).
- `scripts/preview_training_dataset_augmentations.py` — послушать/посмотреть, что именно отдаёт датасет после кодеков/аугментаций.
- `scripts/generate_codec_audio_variants.py` — проверить, что все нужные кодеки доступны в системе.
- `scripts/visualize_manifest_distributions.py`, `scripts/visualize_speaker_distribution.py` — статистика по манифесту.

## См. также

- [`CONFIGS.md`](CONFIGS.md) — карта Hydra-групп, что какой YAML контролирует.
- [`DATA_PREPROCESSING_PIPELINE.md`](DATA_PREPROCESSING_PIPELINE.md) — граф обработки внутри датасета/коллатора.
- [`EVALUATION.md`](EVALUATION.md) — что запускать после трейна (submission / TRT).
