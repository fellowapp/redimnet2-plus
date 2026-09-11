# Evaluation and Inference

Two scenarios:

1. **Validation during training** — VoxCeleb1 EER, runs embedded in the training loop.
2. **Submission inference** — compute embeddings over an audio directory and rerank neighbors (mean-chain v2). Two backends: PyTorch (Accelerate) and TensorRT.

## 1. Validation during training

Everything is configured in `run.sh` (see also [`TRAINING.md`](TRAINING.md) and `asv/validation/voxceleb.py`).

- `VALIDATION_ENABLED=true` — enables it.
- `VOXCELEB_VAL_ROOT`, `VOXCELEB_TRIAL_FILES` — VoxCeleb root and the list of trial files.
- `VALIDATE_EVERY_N_STEPS=N` — validate every N optimizer steps, plus at the end of every epoch.
- `VALIDATE_BEFORE_FIRST_EPOCH=true` — run validation before the first step as a baseline.

Logs:
- `val/eer/<trial_stem>` — EER for each trial file.
- `val/eer/pooled` — pooled EER across all trials.

Also written to `validation_metrics.jsonl` (in the Hydra runtime dir) — each line is one validation run with a tag (`epoch_XXX_end`, `epoch_XXX_step_NNN`, …).

Variant validation (robustness to codecs/augmentations):
- `VARIANT_VAL_ENABLED=true`, `VARIANT_VAL_SCAN_ROOT=<directory containing subdirectories>`.
- Each subdirectory gets its own EER (`val/variants/eer_<name>`), plus a global `val/eer/pooled_hard`.

## 2. Submission — PyTorch (`run_embeddings.py`)

Custom inference without TRT, via Accelerate on N GPUs.

Quick start — the `submission/reranking/run_embeddings.sh` script (edit the paths inside it). Under the hood it calls:

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

- `--weights` — path to the checkpoint: an Accelerate directory (`epoch_*`) or a single file (`.safetensors`, `.bin`, `.pt`). The script auto-detects it and strips `module.` / `_orig_mod.` prefixes.
- `--trim-num-samples 0` — use the full length of the file (no trimming). For a fixed length, use e.g. `96000`.
- `--audio-root <dir>` instead of `--manifest-path` — recursively scans the directory and computes embeddings for all audio files.

Output — a pickle `{"embeddings": np.ndarray [N, D], "audio_ids": list[str], "trial_rel_paths": list[str], "audio_paths": list[str]}`.

## 3. Submission — TensorRT (`run_embeddings_trt.py`)

Faster and more deterministic path. Uses a pre-built `.plan` engine (see [`TENSORRT_CONVERSION_AND_BENCHMARKING.md`](TENSORRT_CONVERSION_AND_BENCHMARKING.md) for export details).

Run with a single shell command (`submission/reranking/run_submission_embeddings_trt_pipeline.sh`):

```bash
./submission/reranking/run_submission_embeddings_trt_pipeline.sh \
  --audio-dir /path/to/audio
```

Required: `--audio-dir <dir>` — directory with audio (flac/wav/mp3/…).

Optional:
- `--engine-path <plan>` — point directly at the `.plan`; batch/num_samples can be read automatically from a filename like `..._bs32_s96000.plan` (or pass `--trt-batch-size`, `--trt-num-samples`).
- Or `--summary-json` / `--variant <key>` — take the engine from `export_summary.json` (default `artifacts/export_summary.json` + the `frontend_fp32_model_fp16` variant).
- `--output-dir <dir>` — where to place the pickle and CSV (default `submission_pipeline/`).
- `--num-workers N` — decode workers (same as `--cpu-workers` for rerank).
- `--k N` — how many neighbors to keep (default `10`).

What the script does:
1. `run_embeddings_trt.py` → embeddings into `<output-dir>/embeddings_trt.pkl`.
2. `rerank_embeddings_mean_chain_v2_fast.py` → reranks neighbors into a CSV with the `.neighbors.csv` suffix.

For the Docker scenario (pre-baking the TRT engine inside the container + mounting audio) — see `docs/README.md` "Quick start" and `docker-compose.trt.yml`.

## 4. Neighbor reranking

`submission/reranking/rerank_embeddings_mean_chain_v2_fast.py` — mean-chain v2 with graph-based reranking. Inputs:
- `embeddings.pkl` with keys `embeddings`, `audio_ids`.
- `--k <N>` — how many neighbors to keep per file.
- `--output-prefix <prefix>` — where to place the result (suffixes `.neighbors.csv`, `.metrics.json`, etc. will be appended).
- `--cpu-workers <N>` — CPU parallelism.
- `--device cuda|cpu` — for the main tensor operations.
- `--skip-metrics` — skip metric computation, just output the top-K.

The neighbors CSV has columns `filepath,neighbours`, where `neighbours` is 10 (or however many were specified) comma-separated relative paths in quotes.

## 5. TensorRT: export and warm-up

Short summary (full version in [`TENSORRT_CONVERSION_AND_BENCHMARKING.md`](TENSORRT_CONVERSION_AND_BENCHMARKING.md)):

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

The frontend (mel) is always fp32; the model body is fp16 / fp32 by choice. Batch size and `num_samples` are baked into the `.plan` and later picked up from the filename.

Benchmarking on VoxCeleb trials — `benchmarking/benchmark_onnx_speed.py` (an example invocation is in the same file, `TENSORRT_CONVERSION_AND_BENCHMARKING.md`).

## 6. Common pitfalls

- **`run.sh` refuses to start** — uncommitted changes: `git status --porcelain` is not empty.
- **OOM during submission inference** — reduce `--batch-size`, remove `mixed_precision` from Accelerate or enable it (`bf16`), or switch to the TRT path.
- **TRT: per-batch corruption after the first group of files** — this was a CUDA-stream sync bug (the fix is already in the code, see commit `98d0784`). If you build the engine in a different environment, check `benchmarking/benchmark_onnx_speed.py::TrtRunner.submit` — it contains `self.stream.wait_stream(torch.cuda.default_stream(...))`.
- **Resume doesn't update the weights** — `train.verify_checkpoint_load=true` will fail at startup. Most likely an incorrect checkpoint path or a broken `latest.json`.
