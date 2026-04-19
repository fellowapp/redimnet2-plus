# TensorRT Conversion And Benchmarking

This note shows how to:

1. convert a checkpoint to TensorRT with frontend in `fp32` and model in `fp16`
2. benchmark the resulting engine with `benchmarking/benchmark_onnx_speed.py` and read speed + EER

All commands below assume you run them from the repo root in the `asv` conda environment.

## 1. Convert checkpoint to TensorRT

Checkpoint used in the examples:

- checkpoint dir: `outputs/FINAL_LMFT/epoch_0005_step_00014800`
- weights file resolved from that dir: `outputs/FINAL_LMFT/epoch_0005_step_00014800/pytorch_model_fsdp.bin`
- model config: `configs/model/redimnet2_b6.yaml`

### Build both variants at once

This helper builds:

- frontend `fp32` + model `fp16`

Example for batch `32`, 6-second chunks at 16 kHz (`96000` samples):


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

- `--engine_precision fp16` enables TensorRT FP16 build
- `--export_dtype fp16` exports the model body in FP16
- frontend layers are forced to FP32 by the exporter

## 2. Benchmark TensorRT engine and get speed + EER

### Run benchmark on one VoxCeleb trial file

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

This adds an extra setup in the report:

- `verification.setups.voxceleb1_o_hard.eer`