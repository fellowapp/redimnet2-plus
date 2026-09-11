# Installation

Minimal instructions: set up the `asv` environment, install dependencies, and fix `torchcodec`/`ffmpeg` if something breaks.

All commands are run from the repository root.

## 1. Conda + uv

We keep the environment in conda and pull dependencies from `uv.lock`. `uv` itself is installed via `pip install` inside the conda environment (not via `conda install`).

```bash
conda create -n asv python=3.10 --no-default-packages -y
conda activate asv
pip install uv
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked
```

Important: `UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"` is always required — otherwise `uv` will create a local `.venv` and leave the conda environment untouched.

To install a single library by hand (without touching `uv.lock`):

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" <package>
```

## 2. Verification

```bash
conda run -n asv python -c "import torch, torchaudio, torchcodec, pyarrow; print('ok')"
```

## 3. FFmpeg + NVIDIA NPP (if torchcodec fails)

Symptom: `torchcodec` complains about `libnppicc.so.13: cannot open shared object file` or a missing `libavutil.so.*`.

```bash
conda activate asv
conda install -y ffmpeg
uv pip install nvidia-npp

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cp activate_asv_cuda_libs.sh  "$CONDA_PREFIX/etc/conda/activate.d/asv_cuda_libs.sh"
cp deactivate_asv_cuda_libs.sh "$CONDA_PREFIX/etc/conda/deactivate.d/asv_cuda_libs.sh"
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/asv_cuda_libs.sh" \
         "$CONDA_PREFIX/etc/conda/deactivate.d/asv_cuda_libs.sh"
```

The activation hook adds the paths to `torch/lib` and `nvidia/*/lib` inside conda to `LD_LIBRARY_PATH`.

Quick check:
```bash
conda run -n asv python -c "import torchaudio; x,sr=torchaudio.load_with_torchcodec('/path/to/audio.wav'); print(x.shape, sr)"
```

## 4. FFmpeg with the required codecs (if AMR/Speex/… are missing)

The codecs are only needed if codec augmentations are enabled in the training pipeline. Check the available codec set:

```bash
python scripts/generate_codec_audio_variants.py <path.wav> --output_dir ./outputs/test_codecs
# expected: generated=17, skipped=0, failed=0
```

If some codecs are missing, rebuild `ffmpeg` into the conda prefix with the required flags (`--enable-libspeex`, `--enable-libopencore-amrnb`, `--enable-libopencore-amrwb`, `--enable-libmp3lame`). The full instructions are in the top-level `README.md` ("Expand FFmpeg build and codec support setup").

## 5. Docker (inference + TensorRT only)

If you only need inference with TRT, the conda path isn't required. See `docs/README.md` (Docker sections) and `docker-compose.trt.yml`: `docker compose up --build asv-trt-runtime` builds the image, warms up the `.plan` engine, and computes embeddings.
