# Example: training and submission on Kryp (1 GPU)

A step-by-step walkthrough on a single dataset (the Kryp speaker-id challenge) and a single GPU. For other datasets, see [`DATA_PREPARATION.md`](DATA_PREPARATION.md) + [`MANIFESTS.md`](MANIFESTS.md); for multi-GPU, see [`TRAINING.md`](TRAINING.md).

## 0. What we have

Assume the dataset is unpacked into `~/cryp_data/kryp/`:

```
~/cryp_data/kryp/
├── _download_mailru_public.py
├── _manifest_build.log
├── kryp_train_manifest.parquet    # already-built manifest (optional)
├── train/
│   └── <speaker_id>/<NNNNN>.flac  # training audio
├── train.csv                       # columns: speaker_id, filepath
├── train.tar.gz
├── test_public/
│   └── <file>.flac                 # audio for submission
└── test_public.csv                 # list of files for submission
```

Everything is 16 kHz mono FLAC.

## 1. Environment

Set up once, see [`INSTALLATION.md`](INSTALLATION.md):

```bash
conda create -n asv python=3.10 --no-default-packages -y
conda activate asv
pip install uv
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked

# for torchcodec:
conda install -y ffmpeg
uv pip install nvidia-npp
cp activate_asv_cuda_libs.sh "$CONDA_PREFIX/etc/conda/activate.d/asv_cuda_libs.sh"
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/asv_cuda_libs.sh"
```

## 2. Manifest

### If it isn't built yet — build it

```bash
conda activate asv
cd /path/to/ASV

python scripts/build_parquet_manifest.py \
  --dataset kryp \
  ~/cryp_data/kryp \
  ~/cryp_data/kryp
```

The script reads `~/cryp_data/kryp/train.csv` (columns `speaker_id`, `filepath`), walks `train/<spk>/*.flac`, and writes `~/cryp_data/kryp/kryp_train_manifest.parquet`. Stdout will show:

```
Wrote .../kryp_train_manifest.parquet (N rows)
num_classes=<K>
ASV_NUM_CLASSES=<K>
```

Remember `K` — this is the number of unique speakers, i.e. `ASV_NUM_CLASSES`.

### If the manifest already exists

You can just read the number of classes by hand:

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

## 3. Training on 1 GPU

### Accelerate config

Already provided: `configs/accelerate/single_gpu_bf16.yaml` (`distributed_type: NO`, 1 process, bf16). Unlike the FSDP configs, it does **not** shard parameters — on a single GPU there's nothing to shard.

### run.sh — what to change

Open `run.sh` and replace the top block for single-GPU:

```bash
# --- for Kryp + 1 GPU ---
MANIFEST_PATH="/home/kirill/cryp_data/kryp/kryp_train_manifest.parquet"
ASV_NUM_CLASSES=<number_of_speakers_from_step_2>
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
INIT_WEIGHTS=""                    # if you want to start from a pretrained checkpoint — specify the path
INIT_MODE="full"
RESUME_FROM_CHECKPOINT=""

NUM_GPUS=1
MIXED_PRECISION="bf16"
ACCELERATE_CONFIG="configs/accelerate/single_gpu_bf16.yaml"

# Validation on VoxCeleb1 — not required for Kryp-only training:
VALIDATION_ENABLED=false
VARIANT_VAL_ENABLED=false
VALIDATE_BEFORE_FIRST_EPOCH=false
```

Important: `configs/config.yaml` has `train.expect_fsdp: true` — this is just a logging flag, and on 1 GPU without FSDP it won't break anything, so it's fine to leave it. If the warning bothers you, add `train.expect_fsdp=false` to the `accelerate launch` invocation block in `run.sh`.

If you're disabling MUSAN/RIRS (no datasets on hand), pass empty `MUSAN_PATH=""` **plus** CLI overrides:

```bash
# add to the accelerate launch command:
augment.waveform_levels.1.p=0.0 \
augment.waveform_levels.2.p=0.0
```

(this zeroes out the "additive_noise" and "rir" levels in `configs/augment/default.yaml`).

### Launch

```bash
cd /path/to/ASV
conda activate asv
bash run.sh
```

`run.sh` refuses to start if there are uncommitted changes. Commit your edits (or temporarily comment out the check in `run.sh`).

Checkpoints are written to `outputs/<MLOP_RUN_NAME>/<date>/<time>/checkpoints/epoch_XXXX_step_XXXXXXXX/`. The latest one is recorded in `latest.json`.

## 4. Inference on test_public: embeddings

You need the path to the trained checkpoint from step 3, e.g.:
`outputs/trainin_b6_lmft_long/2026-04-20/12-00-00/checkpoints/epoch_0004_step_00008000/`.

### Option A — PyTorch (single GPU, no TRT)

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

- `--trim-num-samples 96000` — center-crop to 6 seconds (matches training). If you want full length, use `--trim-num-samples 0`.
- `--audio-root` scans the directory recursively; `test_public.csv` itself is not read — we need the list of actual files, and those come from the filesystem.

The output is `embeddings_test_public.pkl` containing a dict `{embeddings, audio_ids, trial_rel_paths, audio_paths}` (L2-normalized).

### Option B — TensorRT (faster)

First export the `.plan` (see [`TENSORRT_CONVERSION_AND_BENCHMARKING.md`](TENSORRT_CONVERSION_AND_BENCHMARKING.md)):

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

Then run the pipeline (embeddings + rerank in a single command):

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

The script automatically extracts `batch_size=32` and `num_samples=96000` from the `.plan` filename and places the result in `~/cryp_data/kryp/submission_outputs/`:
- `embeddings_trt.pkl` — embeddings
- `embeddings_trt_mean_chain_v2.neighbors.csv` — **this is the submission file**

## 5. Assemble the submission (rerank + CSV)

If you used Option A (PyTorch), rerank needs to be run separately:

```bash
python submission/reranking/rerank_embeddings_mean_chain_v2_fast.py \
  /home/kirill/cryp_data/kryp/embeddings_test_public.pkl \
  --k 10 \
  --output-prefix /home/kirill/cryp_data/kryp/submission \
  --cpu-workers 4 \
  --neighbors-filepath-prefix test_public \
  --skip-metrics
```

Output:
- `submission.neighbors.csv` — `filepath,neighbours` for each file in `test_public/` (10 neighbors, comma-separated, in quotes). **This is the submission.**
- `submission.metrics.json` — nothing useful (no metrics, just timing).

The `filepath` column will look like `test_public/<name>.flac` — this is set by `--neighbors-filepath-prefix test_public`. If you need a different prefix, specify it there.

## 6. Check the format

The submission CSV should look like this:

```
filepath,neighbours
test_public/abc123.flac,"test_public/xxx.flac,test_public/yyy.flac,test_public/zzz.flac,..."
test_public/def456.flac,"test_public/aaa.flac,..."
```

- One row per file in `test_public/`.
- 10 neighbors in the `neighbours` field, comma-separated, in double quotes.
- The first row is the header `filepath,neighbours`.

If the submission platform's format is stricter (e.g. it needs a different prefix or a different number of neighbors), adjust `--neighbors-filepath-prefix` and `--k`.

## 7. Frequently asked questions

- **OOM during training:** reduce `BATCH_PER_GPU`, increase `GRAD_ACCUM_STEPS`. For b6 on a 24GB GPU, 8–12 is comfortable with 6-second chunks.
- **OOM during inference:** for PyTorch, reduce `--batch-size`; for TRT, rebuild the `.plan` with a smaller `--batch_size`.
- **Training takes a long time:** 4 epochs on a single GPU is many hours. For a quick smoke test, use `SINGLE_BATCH_TEST=true bash run.sh` — a single batch, a sanity check of the pipeline.
- **Training fails on codec augmentations:** check ffmpeg (`python scripts/generate_codec_audio_variants.py <file> --output_dir ./tmp`). Or disable the codec branch via CLI: `data.transform_probabilities.codec_only=0.0 data.transform_probabilities.codec_then_augment=0.0`, and raise `keep_original`/`augment_only` accordingly so the sum is 1.0 (this table is marked as an experiment-lock in the defaults; here you're overriding it deliberately).
- **VoxCeleb validation isn't needed for Kryp-only training:** disable it via `VALIDATION_ENABLED=false` and `VARIANT_VAL_ENABLED=false` in `run.sh`.
