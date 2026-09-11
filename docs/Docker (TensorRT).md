# Docker (TensorRT) and evaluation

Working directory inside the container: **`/opt/asv`**.

---

**A GPU IS ALWAYS REQUIRED** (`docker-compose.trt.yml` sets `gpus: all`): without a GPU this image is pointless — TensorRT pre-baking, embeddings, and the smoke test all run on the GPU.

**ON THE FIRST CONTAINER START, THE ENTRYPOINT SPENDS A LONG TIME BUILDING THE TENSORRT ENGINE: ONNX EXPORT, PLAN COMPILATION — THIS CAN TAKE MANY MINUTES. DURING THIS TIME THE GPU IS BUSY — THAT'S HOW TENSORRT WORKS. WAIT FOR THE LOGS TO FINISH BEFORE STARTING THE EVALUATION PIPELINE.**

The **`./artifacts`** folder on the host is mounted at **`/opt/asv/artifacts`**: this is where the TensorRT export is written (`export_summary.json`, `.plan`, etc.).

---

## Quick start

1. Weights live in **`weights/`** at the repository root (`docker-compose.trt.yml` has **`ASV_CHECKPOINT: weights`** fixed).

2. Build the image and bring up the container (one command):

```bash
docker compose up --build asv-trt-runtime
```

Once the **TensorRT engine build and smoke test** finish successfully, the container logs will show something like this (the `.plan` name depends on the checkpoint and variant):

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

The line **`Wrote export summary to .../export_summary.json`** and the smoke-test JSON with **`engine_path`** / **`batch_size`** / **`num_samples`** indicate that the pre-bake succeeded and you can start the evaluation pipeline.

3. **Mount the audio folder** and run the script with a single command — the **`-v` flag takes host:path-inside-container** (read-only — `:ro`):

```bash
docker compose run --rm \
  -v /absolute/path/to/audio:/data/audio:ro \
  asv-trt-runtime \
  /opt/asv/submission/reranking/run_submission_embeddings_trt_pipeline.sh \
  --audio-dir /data/audio
```

Replace **`/absolute/path/to/audio`** with your own directory. Inside the container, the path **`/data/audio`** can be anything, as long as it matches what's passed to **`--audio-dir`**.

4. Enter a shell (optionally also with audio mounted):

```bash
docker compose run --rm -v /absolute/path/to/audio:/data/audio:ro asv-trt-runtime bash
# then, inside the container:
# ./submission/reranking/run_submission_embeddings_trt_pipeline.sh --audio-dir /data/audio
```

Results by default: **`./submission_outputs`** on the host → **`/opt/asv/submission_outputs`** inside the container.

---

## Details: build args, environment variables, engine

### Build and weights

- **`docker-compose.trt.yml`** sets **`ASV_CHECKPOINT: weights`**: the **`weights/`** directory is copied into the image. Build args can be changed, but this is usually unnecessary if the weights are already in `weights/`.
- The remaining build args (`ASV_MODEL_NAME`, `ASV_MODEL_CONFIG`, `ASV_BATCH_SIZE`, `ASV_NUM_SAMPLES`, ...) set the parameters used to call `export_checkpoint_to_tensorrt.py` on first start.

### Pre-bake

- **`ASV_PREBAKE_ENGINES: "1"`** — if no summary is found, the entrypoint runs an ONNX export + builds the **`.plan`** and runs `trt_smoke_test.py`. Result: **`/opt/asv/artifacts/export_summary.json`**.

### `run_submission_embeddings_trt_pipeline.sh` pipeline

- **`--audio-dir`** — required argument (a tree of `.wav` / `.flac` / `.mp3` ...).
- The engine and batch sizes are substituted from **`ASV_EXPORT_SUMMARY`** after the pre-bake. **`--engine-path`** is only an override; without explicit `--trt-batch-size` / `--trt-num-samples`, these can be inferred from a name like `..._bs32_s96000.plan`.
- **`ASV_TRT_NUM_WORKERS`** / **`--num-workers`** — decode workers, and **`--cpu-workers`** for rerank.
- Rerank inside the script is always run with **`--skip-metrics`**.

### Where this is configured in compose

- Volume **`./artifacts:/opt/asv/artifacts`** (TRT export), **`./submission_outputs`**, **`ASV_OUTPUT_DIR`**, **`ASV_EXPORT_SUMMARY`**, **`ASV_DEFAULT_VARIANT`**.
