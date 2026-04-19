# Конвертация в TensorRT и бенчмаркинг

В этой заметке описано, как:

1. сконвертировать чекпоинт в TensorRT с фронтендом в `fp32` и моделью в `fp16`;
2. прогнать бенчмарк получившегося engine через `benchmarking/benchmark_onnx_speed.py` и получить скорость и EER.

Все команды ниже предполагают запуск из корня репозитория в conda-окружении `asv`.

## 1. Конвертация чекпоинта в TensorRT

Чекпоинт в примерах:

- каталог чекпоинта: `outputs/FINAL_LMFT/epoch_0005_step_00014800`;
- файл весов из этого каталога: `outputs/FINAL_LMFT/epoch_0005_step_00014800/pytorch_model_fsdp.bin`;
- конфиг модели: `configs/model/redimnet2_b6.yaml`.

### Сборка обоих вариантов за один раз

Этот скрипт собирает:

- фронтенд `fp32` + модель `fp16`.

Пример для батча `32`, фрагментов по 6 с при 16 кГц (`96000` сэмплов):

```bash

source "/home/kirill/miniconda3/etc/profile.d/conda.sh"
conda activate asv

python benchmarking/export_redimnet2_to_tensorrt.py \
  --model_name b6 \
  --checkpoint "outputs/FINAL_LMFT/epoch_0005_step_00014800" \
  --model_config "configs/model/redimnet2_b6.yaml" \
  --engine_precision fp16 \
  --export_dtype fp16 \
  --batch_size 32 \
  --num_samples 96000 \
  --engine_path "benchmarking/outputs/redimnet2_b6_fp16_frontendfp32_bs32_s96000.plan" \
  --report_json "benchmarking/outputs/redimnet2_b6_fp16_frontendfp32_bs32_s96000.report.json"
```

Примечания:

- `--engine_precision fp16` включает сборку TensorRT в FP16;
- `--export_dtype fp16` экспортирует тело модели в FP16;
- слои фронтенда экспортером принудительно остаются в FP32.

## 2. Бенчмарк TensorRT engine: скорость и EER

### Запуск бенчмарка на одном trial-файле VoxCeleb

Пример:

```bash
python benchmarking/benchmark_onnx_speed.py \
  --backend trt \
  --trt_engine_path "benchmarking/outputs/redimnet2_b6_fp16_frontendfp32_bs32_s96000.plan" \
  --utterance_list "/home/kirill/mnt/ssd_1tb_1/VoxCeleb/voxceleb1_o.txt" \
  --audio_root "/home/kirill/mnt/ssd_1tb_1/VoxCeleb/wav" \
  --extra_voxceleb_o_audio_root "/home/kirill/mnt/ssd_1tb_1/VoxCelebHard" \
  --list_format voxceleb_trials_pooled \
  --batch_size 32 \
  --num_workers 0 \
  --trt_inflight_batches 1 \
  --chunk_seconds 6.0 \
  --warmup_batches 1 \
  --report_json "benchmarking/outputs/benchmark_fp16_frontendfp32_bs32_pooled_hard.json"
```

В отчёт добавляется дополнительная секция:

- `verification.setups.voxceleb1_o_hard.eer`
