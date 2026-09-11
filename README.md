# ReDimNet2+

**ReDimNet2+ for Robust Speaker Verification under Channel and Codec Shift**

Kirill Borodin<sup>1,2,3</sup>, Vasiliy Kudryavtsev<sup>1,2,3</sup>, Grach Mkrchian<sup>1,3</sup>
<br><sup>1</sup>lab260, Moscow, Russia&nbsp;&nbsp;<sup>2</sup>BitmanagerAI, Dubai, UAE&nbsp;&nbsp;<sup>3</sup>MTUCI, Moscow, Russia
<br>Submitted to ICASSP 2027

| Material | Link |
|---|---|
| Paper (report) | [report.pdf](report.pdf) |
| Slides | [ReDimNet2+.pptx](ReDimNet2+.pptx) |
| Model weights | [huggingface.co/lab260/redimnet2-plus](https://huggingface.co/lab260/redimnet2-plus) |

## Abstract

Automatic speaker verification must remain reliable across devices, rooms, and compression pipelines. We present ReDimNet2+, a robustness-oriented adaptation of the compact ReDimNet2 backbone. Analysis of a VoxBlink2 subset reveals a shift in predicted spectral coloration, motivating codec and waveform augmentation. We combine these augmentations with multi-corpus adaptation, large-margin fine-tuning (LMFT), and graph-based retrieval reranking. With random 4-second evaluation windows for all models, ReDimNet2+ LMFT reduces pooled VoxCeleb1 EER from 2.42% to 0.82% and a 26-condition robustness stress-test EER from 7.21% to 1.99%. Under this shared local protocol, it reaches 0.35% EER on VoxCeleb1-O versus 0.787% for the best evaluated WeSpeaker checkpoint. On a VoxBlink2 retrieval subset, reranking improves the final model's Pr@k from 0.7413 to 0.7687.

## Method at a glance

- **Backbone.** ReDimNet2-B6 (12.3M params), initialized from its public checkpoint and kept architecturally unchanged; robustness comes from data, augmentation, schedule, and scoring.
- **Training pipeline.** Random-window decoding from a Parquet manifest (2.6× faster than decode-then-crop), GPU-side spectrogram extraction (1.53× throughput), and a fallback-on-decode-failure loader for multi-day runs on crawled corpora.
- **Augmentation.** Each example goes through one of keep / codec / waveform / codec+waveform (p = 0.5/0.2/0.2/0.1). Waveform augmentation covers filters, additive noise (colored, MUSAN, RawBoost-style), simulated RIRs, and speed perturbation. Codec augmentation simulates MP3, Opus, AAC, FLAC, Vorbis, G.723.1, IMA ADPCM, G.722, A-law, μ-law, Speex, and AMR-NB at 8/16 kHz, directly motivated by the coloration shift found in VoxBlink2.
- **Training recipe.** SphereFace2 head with one-vs-all objectives, staged multi-corpus adaptation across 7 corpora (~63.9k speakers, ~4.6M utterances, ~8,675 h), followed by large-margin fine-tuning (LMFT) on 6-second windows. Accelerate + FSDP on 6 GPUs, BFloat16, AdamW.
- **Retrieval reranking.** Two lightweight, training-free stages on cached L2-normalized embeddings: mean-chain reranking (locally consistent chain over candidates) followed by graph reranking (k-NN graph with reciprocal, common-neighbor, and hubness-penalty terms), inspired by k-reciprocal reranking and hubness correction.

## Results

All numbers below use random 4-second evaluation windows and cosine similarity on L2-normalized embeddings; checkpoints are selected on the VoxCeleb1 dev set. EER<sub>p</sub> pools VoxCeleb1-O/E/H scores; EER<sub>ph</sub> is a 26-condition codec/waveform robustness stress test; Pr@k is the VoxBlink2-subset retrieval metric.

### Development path (VoxCeleb1, EER %)

| System | EER<sub>o</sub> | EER<sub>e</sub> | EER<sub>h</sub> | EER<sub>p</sub> | EER<sub>ph</sub> | Pr@k |
|---|---:|---:|---:|---:|---:|---:|
| ReDimNet2 baseline | 1.601 | 1.725 | 3.039 | 2.420 | 7.214 | 0.5978 |
| + multi-domain adaptation (10 epochs) | 0.872 | 0.958 | 1.833 | 1.458 | 3.303 | 0.7005 |
| ReDimNet2+ pretrained | 0.792 | 0.956 | 1.811 | 1.424 | 3.229 | 0.7024 |
| ReDimNet2 (LMFT only, no pretraining) | 0.436 | 0.589 | 1.118 | 0.885 | 2.281 | 0.7221 |
| **ReDimNet2+ LMFT (final)** | **0.351** | **0.523** | **1.055** | **0.824** | **1.991** | **0.7413** |

Full 11-row ablation (all intermediate multi-corpus and margin-schedule runs) is in the paper, Table 1.

### Comparison under a shared local 4-second protocol (VoxCeleb1, EER %)

| System | Params | EER<sub>o</sub> | EER<sub>e</sub> | EER<sub>h</sub> |
|---|---:|---:|---:|---:|
| ReDimNet2-B6 (public ckpt.) | 12.3M | 1.601 | 1.725 | 3.039 |
| WeSpeaker ResNet34-LM | 6.6M | 0.814 | 0.933 | 1.679 |
| WeSpeaker ECAPA512-LM | 6.2M | 0.877 | 1.071 | 1.968 |
| WeSpeaker CAM++ | 7.2M | 0.787 | 0.928 | 1.824 |
| **ReDimNet2+ LMFT (ours)** | 12.3M | **0.351** | **0.523** | **1.055** |

For reference, the *published* full-utterance numbers (different protocol, not directly comparable) include ReDimNet2-B6 0.29/0.52/0.99%, ECAPA2 0.34/0.52/0.99%, and W2V-BERT 2.0 (587M params) 0.38/0.51/1.06% — see paper Table 2 for the complete list and caveats.

### Retrieval reranking ablation (ReDimNet2+ pretrained embeddings)

| Method | Pr@1 (%) | Pr@10 (%) | Pr@45 (%) | VoxBlink2 Pr@k |
|---|---:|---:|---:|---:|
| Baseline (no rerank) | 99.9577 | 99.8611 | 99.2404 | 0.7024 |
| Mean-chain | 99.9577 | 99.9075 | 99.5299 | 0.7220 |
| Graph rerank | 99.9492 | 99.8798 | 99.5671 | 0.7369 |
| **Mean-chain + graph** | 99.9511 | 99.9003 | **99.6992** | **0.7391** |

Applied to the final ReDimNet2+ LMFT model, the full reranking pipeline raises VoxBlink2 Pr@k from 0.7413 to 0.7687 (paper Table 4).

See [`report.pdf`](report.pdf) for the full paper, including the data/domain-mismatch analysis, all ablations, and limitations.

---

Minimal set of information about the project: start with a quick `eval`, then navigate the rest of the docs, training setup, and repository layout.

## Quick eval via Docker

Evaluation via Docker in this repository defaults to the `submission-cu129` service (`cuda129` profile, CUDA 12.9).

<details>
<summary>Alternative: CUDA 13 profile</summary>

You can also use the `submission` service from `docker-compose.submission.yml` (same `weights`/`outputs` mounts by default, different base image for CUDA 13; see the nested command blocks below).

</details>

What the container fixes:
- `./weights` on the host is mounted to `/workspace/weights`
- `./outputs` on the host is mounted to `/workspace/outputs`
- container working directory: `/workspace`
- the result is written by default to `./outputs/output.csv`

Running it requires a GPU and a working Docker install with NVIDIA support.

### Preparation

Before running, make sure that:
- weights are in `weights/` (the checkpoint `pytorch_model_fsdp.bin` already ships in this repo; it's also published at [huggingface.co/lab260/redimnet2-plus](https://huggingface.co/lab260/redimnet2-plus) and can be fetched with `hf download lab260/redimnet2-plus pytorch_model_fsdp.bin --local-dir weights`)
- the `outputs/` folder exists
- audio is in a separate directory on the host, e.g. `/home/kirill/mnt/ssd_1tb_1/kryp/test_public`

### Build command

#### `cuda12.9` (`cuda129`)

This command builds the image and checks that the mounted `weights` and `outputs` are visible inside the container:

```bash
docker compose -f docker-compose.submission.cuda129.yml run --rm --build \
  --entrypoint /bin/bash submission-cu129 -lc \
  "ls -ld /workspace/weights /workspace/outputs"
```

What the command does:
- `--build` rebuilds the image before running
- `--entrypoint /bin/bash` temporarily disables the normal inference entrypoint
- `submission-cu129 -lc "ls -ld ..."` runs a short check inside the container
- `--rm` removes the temporary container after it finishes

If the command completes without errors, you can run eval.

<details>
<summary>CUDA 13 profile (`submission`)</summary>

Same idea for `docker-compose.submission.yml`:

```bash
docker compose -f docker-compose.submission.yml run --rm --build \
  --entrypoint /bin/bash submission -lc \
  "ls -ld /workspace/weights /workspace/outputs"
```

</details>

### Evaluation run command

#### `cuda12.9` (`cuda129`)

Command to run evaluation (recommended after the first build, e.g. offline):

```bash
docker compose -f docker-compose.submission.cuda129.yml run --rm \
  -v /home/kirill/mnt/ssd_1tb_1/kryp/test_public:/workspace/inference_audio:ro submission-cu129 \
  --audio-root /workspace/inference_audio \
  --k 10 \
  --num-workers 16 \
  --gpu-model-instances 3
```

Command to run with an additional rebuild:

```bash
docker compose -f docker-compose.submission.cuda129.yml run --rm --build \
  -v /home/kirill/mnt/ssd_1tb_1/kryp/test_public:/workspace/inference_audio:ro submission-cu129 \
  --audio-root /workspace/inference_audio \
  --k 10 \
  --num-workers 16 \
  --gpu-model-instances 3
```

Important rule: the path in `--audio-root` must match the path inside the container given in the `-v` flag.

Example for `cuda129`:
- on host: `/home/kirill/mnt/ssd_1tb_1/kryp/test_public`
- inside container: `/workspace/inference_audio`
- so pass `--audio-root /workspace/inference_audio`

If the image is already built and the code hasn't changed, `--build` can be dropped from the evaluation command.

<details>
<summary>CUDA 13 profile (`submission`)</summary>

Equivalent command for `docker-compose.submission.yml`:

```bash
docker compose -f docker-compose.submission.yml run --rm --build \
  -v /home/kirill/mnt/ssd_1tb_1/kryp/test_public:/data/test_public:ro submission \
  --audio-root /data/test_public \
  --k 10 \
  --num-workers 8 \
  --gpu-model-instances 3
```

Same rule about `--audio-root` and `-v`. Example:
- on host: `/home/kirill/mnt/ssd_1tb_1/kryp/test_public`
- inside container: `/data/test_public`
- argument: `--audio-root /data/test_public`

</details>

### Where the result is written

By default the result is saved to:

```text
./outputs/output.csv
```

Inside the container this is:

```text
/workspace/outputs/output.csv
```

**Audio paths in the CSV.** The `filepath` column (and neighbor lists) stores a string of the form **the last path segment of `--audio-root` inside the container** plus **the file's path relative to that folder**, joined by `/`. The name is taken from the last segment of the `--audio-root` path INSIDE THE CONTAINER (e.g. for `/workspace/inference_audio` this is `inference_audio`, for `/data/test_public` it's `test_public`). Then comes the full relative path to the file inside that directory, e.g. `inference_audio/subdir/track.wav`. The absolute path to the audio root is not written to the CSV.

### Run command arguments

Below are the arguments that can be passed to the evaluation command.

#### `docker compose run` arguments

These are Docker-level arguments; they go before the service name (`submission-cu129` or `submission`, depending on the compose file).

- `--rm`
  Remove the temporary container after it finishes.

- `--build`
  Rebuild the image before running.

- `-v HOST_PATH:CONTAINER_PATH:ro`
  Mount the audio folder into the container read-only.

Example for `docker-compose.submission.cuda129.yml`:

```bash
-v /abs/path/to/audio:/workspace/inference_audio:ro
```

For `docker-compose.submission.yml` (CUDA 13), e.g. `/data/test_public` is commonly used inside the container — see the collapsible block in the evaluation-run section.

#### `run_submission_unified.py` arguments

These are arguments of the script itself. They go after the service name (`submission-cu129` or `submission`).

- `--audio-root PATH`
  Required. Audio root inside the container. Must match the path from `-v`.

- `--output-csv PATH`
  Path for the output CSV.
  Default: `/workspace/outputs/output.csv`

- `--weights-dir PATH`
  Weights folder. The script recursively searches it for the latest checkpoint.
  Default: `/workspace/weights`

- `--model-config PATH`
  Path to the model config.
  Default: `/workspace/configs/model/redimnet2_b6.yaml`

- `--k INT`
  Number of neighbors in the final answer.
  Default: `10`

- `--gpu-model-instances INT`
  How many model replicas to keep on the GPU in parallel. SINCE WE USE A BATCH SIZE OF ONE, 3 INSTANCES IS THE OPTIMAL SETTING WE RECOMMEND. PLEASE DO NOT CHANGE THIS PARAMETER.
  Default: `3`

- `--num-workers INT`
  Single parameter shared by the DataLoader and the CPU part of rerank.
  Default: `8`

<details>
<summary>Additional rerank and decode parameters</summary>

- `--mean-steps INT`
  Number of mean-chain rerank steps.
  Default: `40`

- `--n-iter INT`
  Number of graph-rerank iterations.
  Default: `3`

- `--w-rank FLOAT`
  Weight of the rank component in rerank.
  Default: `1.0`

- `--w-recip FLOAT`
  Weight of the reciprocal component in rerank.
  Default: `1.5`

- `--w-common FLOAT`
  Weight of the common-neighbors component in rerank.
  Default: `2.0`

- `--w-hub FLOAT`
  Weight of the hub penalty/bonus in rerank.
  Default: `0.3`

- `--community-min INT`
  Lower bound on community size at the postprocess stage.
  Default: `2`

- `--community-max INT`
  Upper bound on community size at the postprocess stage.
  Default: `50`

- `--sample-rate INT`
  Sample rate used for decoding.
  Default: `16000`

- `--max-duration-sec FLOAT`
  Maximum length of a single audio file in seconds after trimming.
  Default: `60.0`

- `--knn-chunk-size INT`
  Chunk size for kNN / candidate-pool computation.
  Default: `1024`

- `--device STRING`
  Inference device. Usually `cuda` or `cpu`.
  Default: `cuda` if available, else `cpu`

</details>

## Documentation

Minimal set of docs to understand the project from installation to inference:

| Document | About |
|---|---|
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Validation during training, submission pipeline (PyTorch and TensorRT). |
| [`docs/INSTALLATION.md`](docs/INSTALLATION.md) | Conda + uv, ffmpeg / torchcodec, basic environment check. |
| [`docs/DATA_PREPARATION.md`](docs/DATA_PREPARATION.md) | Datasets used (VoxCeleb1/2, CN-Celeb, 3D-Speaker, Kryp, KeSpeech, TidyVoice), MUSAN/RIRS. |
| [`docs/MANIFESTS.md`](docs/MANIFESTS.md) | Parquet manifest schema, `build_parquet_manifest.py`, concatenation, durations. |
| [`docs/TRAINING.md`](docs/TRAINING.md) | How to run training (`run.sh`, Accelerate/FSDP, checkpoints, resume, validation). |
| [`docs/CONFIGS.md`](docs/CONFIGS.md) | Map of Hydra configs: model / data / augment / loss / logging / accelerate. |
| [`docs/DATA_PREPROCESSING_PIPELINE.md`](docs/DATA_PREPROCESSING_PIPELINE.md) | Detailed processing graph inside the dataset and collate step. |
| [`docs/TENSORRT_CONVERSION_AND_BENCHMARKING.md`](docs/TENSORRT_CONVERSION_AND_BENCHMARKING.md) | Exporting a checkpoint to `.plan` + TensorRT benchmarking. |
| [`docs/Docker (TensorRT).md`](<docs/Docker (TensorRT).md>) | NOT the recommended default Docker image — performance is slightly worse. Building the TensorRT image, engine warm-up, and inference via `docker compose`. |
| [`docs/EXAMPLE_KRYP.md`](docs/EXAMPLE_KRYP.md) | Worked example on the Kryp dataset. |

## Quick cheat sheet

```bash
# 1. Quick eval via Docker (see the section above / docs/EVALUATION.md)
docker compose -f docker-compose.submission.cuda129.yml run --rm --build \
  -v /path/to/audio:/workspace/inference_audio:ro submission-cu129 \
  --audio-root /workspace/inference_audio

# 2. Set up the environment (see docs/INSTALLATION.md)
conda create -n asv python=3.10 -y && conda activate asv
pip install uv
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked

# 3. Build a manifest (see docs/MANIFESTS.md)
python scripts/build_parquet_manifest.py --dataset voxceleb2 \
  /path/VoxCeleb2 /path/output_dir

# 4. Configure run.sh (paths, batch size, epochs) and launch (see docs/TRAINING.md)
bash run.sh

# 5. Inference on a submission directory (see docs/EVALUATION.md):
# - PyTorch:
bash submission/reranking/run_embeddings.sh
# - TensorRT:
./submission/reranking/run_submission_embeddings_trt_pipeline.sh --audio-dir /path/to/audio
```

## Repository structure

- `asv/` — project code: `data/`, `augment/`, `codec/`, `models/`, `losses/`, `train/`, `validation/`, `manifest/`, `logging/`.
- `configs/` — Hydra configs (`model/`, `data/`, `augment/`, `loss/`, `logging/` groups, plus `accelerate/` for FSDP).
- `scripts/` — utilities: manifest building, dataset benchmarks, codec-variant generation, visualizations.
- `benchmarking/` — ONNX/TensorRT export, engine benchmarking, and variant comparison.
- `submission/reranking/` — inference scripts (PyTorch and TRT) + mean-chain v2 rerank.
- `tests/` — unit tests (manifests, augmentation, models, training utilities, reranking, etc.).
- `run.sh` — single entry point for training (Accelerate + FSDP + Hydra overrides).
- `train.py` — thin Hydra entrypoint, delegates to `asv/train/run.py`.
- `Dockerfile.submission`, `Dockerfile.submission.cuda129`, `docker-compose.submission*.yml` — submission/eval images (CUDA 13 and CUDA 12.9 profiles).
- `Dockerfile.trt`, `docker-compose.trt.yml`, `trt-runtime-entrypoint.sh` — TensorRT-runtime image for inference.

## Citation

If you use this code or the ReDimNet2+ recipe, please cite:

```bibtex
@unpublished{borodin2027redimnet2plus,
  title  = {{ReDimNet2+} for Robust Speaker Verification under Channel and Codec Shift},
  author = {Borodin, Kirill and Kudryavtsev, Vasiliy and Mkrchian, Grach},
  year   = {2026},
  note   = {Manuscript submitted to ICASSP 2027}
}
```
