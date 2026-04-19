# Установка

Минимальная инструкция: поднять окружение `asv`, поставить зависимости, починить `torchcodec`/`ffmpeg` если что-то отвалилось.

Все команды — из корня репозитория.

## 1. Conda + uv

Окружение держим в conda, зависимости берём из `uv.lock`. `uv` ставим `pip install` внутри conda-окружения (не через `conda install`).

```bash
conda create -n asv python=3.10 --no-default-packages -y
conda activate asv
pip install uv
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync --locked
```

Важно: `UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"` нужен всегда, иначе `uv` создаст локальный `.venv` и не тронет conda.

Поставить одну библиотеку руками (не трогая `uv.lock`):

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" <package>
```

## 2. Проверка

```bash
conda run -n asv python -c "import torch, torchaudio, torchcodec, pyarrow; print('ok')"
```

## 3. FFmpeg + NVIDIA NPP (если torchcodec падает)

Симптом: `torchcodec` ругается на `libnppicc.so.13: cannot open shared object file` или на отсутствие `libavutil.so.*`.

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

Активационный хук добавляет в `LD_LIBRARY_PATH` пути к `torch/lib` и `nvidia/*/lib` внутри conda.

Быстрая проверка:
```bash
conda run -n asv python -c "import torchaudio; x,sr=torchaudio.load_with_torchcodec('/path/to/audio.wav'); print(x.shape, sr)"
```

## 4. FFmpeg с нужными кодеками (если не хватает AMR/Speex/…)

Кодеки нужны только если в трейн-пайплайне включены codec-аугментации. Проверить набор кодеков:

```bash
python scripts/generate_codec_audio_variants.py <путь.wav> --output_dir ./outputs/test_codecs
# ожидаемо: generated=17, skipped=0, failed=0
```

Если каких-то кодеков не хватает — пересобрать `ffmpeg` в префикс conda с нужными флагами (`--enable-libspeex`, `--enable-libopencore-amrnb`, `--enable-libopencore-amrwb`, `--enable-libmp3lame`). Полная инструкция — в `README.md` на верхнем уровне («Expand FFmpeg build and codec support setup»).

## 5. Docker (только инференс + TensorRT)

Если нужен только инференс с TRT — conda-путь не нужен. См. `docs/README.md` (разделы по Docker) и `docker-compose.trt.yml`: `docker compose up --build asv-trt-runtime` собирает образ, прогревает `.plan` движок и считает эмбеддинги.
