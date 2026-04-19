# Конфиги (Hydra)

Корневой конфиг — `configs/config.yaml`. Он композирует группы через `defaults:`:

```yaml
defaults:
  - model: default
  - data: default
  - augment: default
  - loss: sphereface2
  - logging: default
  - _self_
  - override hydra/output: asv_run
```

`_self_` в конце значит: значения самого `config.yaml` применяются после всех групп. `hydra/output: asv_run` — правила для Hydra runtime dir (см. `configs/hydra/output/`).

Все параметры переопределяются из CLI: `python train.py data.batch_size=64 train.epochs=10`. `run.sh` ровно так и делает.

## Группы

### `model/`
Модель = Encoder → Bridge → Classifier. Сборка в `asv/models/assemble.py::build_model()` через `hydra.utils.instantiate` на каждое поддерево. Все поддеревья требуют `_target_`.

Что лежит в `configs/model/`:
- `default.yaml` — baseline (conv-энкодер, линейный классификатор), нужен в основном как skeleton.
- `redimnet2_b0.yaml` … `redimnet2_b6.yaml` — ReDimNet2 разных размеров; базовая комплектация без pretrained.
- `redimnet2_b0_pretrained.yaml`, `redimnet2_b6_pretrained.yaml` — подгружают pretrained веса (см. `asv.models.redimnet2.ReDimNet2Encoder`).
- `redimnet2_b0_sphereface2.yaml`, `redimnet2_b6_sphereface2.yaml` — pretrained + SphereFace2 head (используется с `loss=sphereface2`).
- `notlm_*` — варианты без language-model stage в энкодере.

Ключевой инвариант: `model.num_classes` должен совпадать с числом спикеров в манифесте. `run.sh` явно прокидывает `model.num_classes=${ASV_NUM_CLASSES}`. Либо можно включить `train.auto_num_classes=true`, тогда скрипт прочитает их из манифеста на старте (медленнее на огромных датасетах).

### `data/`
`configs/data/default.yaml` — всё про DataLoader/датасет. Ключевые поля:

- `manifest_path` — Parquet (обязательный, задаётся через CLI).
- `batch_size` — на одну GPU.
- `num_workers`, `pin_memory`, `prefetch_factor` — воркеры DataLoader.
- `sample_rate: 16000` — целевой SR; `torchcodec` ресемплит на лету.
- `trim_num_samples: 96000` — жёсткая длина после pad/trim (6 секунд при 16 kHz).
- `codec_preset` — если задан, все codec-ветки используют этот preset; если `null` — выбирается случайно из доступных.
- `transform_probabilities` — 4 режима на один сэмпл, сумма = 1.0:
  - `keep_original` — ничего не делаем;
  - `codec_only` — только кодек-roundtrip;
  - `augment_only` — только waveform-аугментации (фильтры/шум/RIR/…);
  - `codec_then_augment` — кодек, потом аугментации.
  Значения жёстко зафиксированы эксперимент-локом. Трогать **только явно и руками** (в `AGENTS.md` отмечено).

### `augment/`
`configs/augment/default.yaml` — описание аугментаций двух доменов:

- `waveform_levels` — список «уровней» (`filters`, `additive_noise`, `rir`, `perturb`). Каждый уровень: `p`, `one_of`, `transforms` (Hydra-инстанцируемые модули, обычно из `torch_audiomentations` или `asv.augment.*`).
- `feature_levels` — единственный элемент `spec_augment` с блоками `freq_mask`, `time_mask`, `cutmix` (читается через `asv.train.spec_augment.build_spec_augment_cfg` и применяется в collate, не через `AudioAugmentPipeline.apply_to_features`).

Параметры фильтров (cutoff_freq, bandwidth_fraction, …) промаркированы `DO NOT CHANGE IN ANY CASE`. Менять только руками, не через автоматизацию.

`AddBackgroundNoise` тянет файлы из `${oc.env:MUSAN_PATH}`, `ApplyImpulseResponse` — из `${oc.env:RIRS_PATH}`. Эти переменные экспортит `run.sh`.

### `loss/`
- `default.yaml` — `torch.nn.CrossEntropyLoss`.
- `label_smoothing.yaml` — label smoothing CE.
- `sphereface2.yaml` — заглушка `asv.losses.integrated.LossComputedInModel`: loss реально считается **внутри** `SphereFace2Head` (классификатор возвращает `(logits, loss)`). Используется парно с `model=redimnet2_b6_sphereface2`.

Собирается через `asv.losses.registry.build_loss` (`_target_` + kwargs).

### `logging/`
`configs/logging/default.yaml` — backend для логирования. Можно выключить: `logging.backend=none` (внутри `run.sh` — флаг `MLOP_ENABLED=false`).

### `accelerate/`
Не Hydra-группа, это конфиги для `accelerate launch --config_file ...`. Три пресета FSDP bf16 на 2/4/6 GPU. Выбирается в `run.sh` через `ACCELERATE_CONFIG`.

### `hydra/`
`configs/hydra/output/asv_run.yaml` — шаблон для Hydra runtime dir (`output_dir`). Обычно выглядит как `outputs/<MLOP_RUN>/<date>/<time>/` — туда пишутся логи, `config snapshot`, `validation_metrics.jsonl` и `checkpoints/`.

## Корневые поля `config.yaml`

Не входят ни в какую группу, задаются прямо в `config.yaml`:

- `seed` — глобальный seed (`accelerate.utils.set_seed`).
- `output_dir = ${hydra:runtime.output_dir}` — основная output-директория.
- `optim.*` — optimizer (AdamW; `lr`, `final_lr`, `weight_decay`, `beta1`, `beta2`, `eps`).
- `train.*` — параметры цикла (`epochs`, `grad_accum_steps`, `mixed_precision`, `max_grad_norm`, `save_every_epochs`, `torch_compile`, `init_weights`/`init_mode`, `resume_from_checkpoint`, `fallback_on_decode_error` и т.д.).
- `validation.*` — VoxCeleb1 EER (`enabled`, `data_root`, `trial_files`, `batch_size`, `num_workers`, `every_n_steps`, `run_before_first_epoch`).
- `variant_validation.*` — EER по аугментированным вариантам.

## Полезные CLI-overrides

```bash
# Руками через CLI, минуя run.sh
python train.py \
  model=redimnet2_b6_sphereface2 \
  loss=sphereface2 \
  data.manifest_path=/.../manifest.parquet \
  data.batch_size=24 \
  train.epochs=10 \
  train.grad_accum_steps=1 \
  train.mixed_precision=bf16 \
  optim.lr=3e-5 \
  validation.enabled=false
```

Hydra CLI-синтаксис: `+key=value` (новое поле), `key=value` (override), `~key` (удалить). Все группы переопределяются как `model=redimnet2_b0` или подставляются из другого файла.

## Как добавить свой YAML

Например, новый размер модели:
1. Скопировать `configs/model/redimnet2_b0.yaml` в `configs/model/my_variant.yaml`.
2. Поправить `encoder._target_` / поля.
3. Запустить: `python train.py model=my_variant ...` или `MODEL_CFG=my_variant.yaml` в `run.sh`.

Аналогично для `loss/`, `augment/`, и т.д.
