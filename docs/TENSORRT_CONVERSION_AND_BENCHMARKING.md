# TensorRT Conversion and Benchmarking

This note describes how to:

1. convert a checkpoint to TensorRT with the frontend in `fp32` and the model in `fp16`;
2. run a benchmark of the resulting engine via `benchmarking/benchmark_onnx_speed.py` and get speed and EER.

All commands below assume running from the repository root in the `asv` conda environment.

## 1. Converting a checkpoint to TensorRT

Checkpoint used in the examples:

- checkpoint directory: `outputs/FINAL_LMFT/epoch_0005_step_00014800`;
- weights file from this directory: `outputs/FINAL_LMFT/epoch_0005_step_00014800/pytorch_model_fsdp.bin`;
- model config: `configs/model/redimnet2_b6.yaml`.

### Building both variants at once

This script builds:

- `fp32` frontend + `fp16` model.

Example for batch size `32`, 6-second chunks at 16 kHz (`96000` samples):

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

Notes:

- `--engine_precision fp16` enables the TensorRT build in FP16;
- `--export_dtype fp16` exports the model body in FP16;
- frontend layers are forced to stay in FP32 by the exporter.

## 2. Benchmarking the TensorRT engine: speed and EER

### Running the benchmark on a single VoxCeleb trial file

Example:

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

An extra section is added to the report:

- `verification.setups.voxceleb1_o_hard.eer`
