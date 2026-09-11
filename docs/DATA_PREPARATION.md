# Data Preparation

## What we work with

Training is built on speaker verification / speaker ID corpora. Supported types (from `scripts/build_parquet_manifest.py --dataset`):

| `--dataset`     | Purpose                             | Short layout                                                 |
|-----------------|--------------------------------------|---------------------------------------------------------------|
| `voxceleb1`     | VoxCeleb1 (1251 speakers)           | `<root>/wav/<idXXXXX>/<youtube_id>/<segment>.wav`           |
| `voxceleb2`     | VoxCeleb2                           | `<root>/dev/aac/<idXXXXX>/<youtube_id>/<segment>.m4a`       |
| `cnceleb`       | CN-Celeb (train split)              | `<root>/data/<idXXXXX>/<genre>-<session>-<utt>.flac`        |
| `cnceleb2`      | CN-Celeb2 (same scheme)             | same                                                          |
| `cnceleb_all`   | CN-Celeb + eval/{enroll,test}       | `data/` + `eval/enroll`, `eval/test`                          |
| `3dspeaker`     | 3D-Speaker                          | `<root>/3D_SPK_<id>/...wav`                                  |
| `kryp`          | Kryp speaker-id challenge           | `<root>/train.csv` + `<root>/train/<spk>/<NNNNN>.flac`       |
| `tidy`          | TIDY-style dumps                    | `<root>/<spk>/<lang>/<lang>_<utt>.wav`                        |
| `kespeech`      | KeSpeech                            | `<root>/Audio/<spk>/<phase>/<utt>.wav`                        |

File formats: `.wav`, `.flac`, `.m4a`, `.mp4`, `.ogg` — decoding goes through `torchcodec` (see `asv/data/decode.py`), with `torchaudio.load_with_torchcodec` as a fallback.

## Sample rate

Target sample rate is 16 kHz. In the dataset config (`configs/data/default.yaml`) this is `data.sample_rate: 16000`. `torchcodec` resamples on the fly if the audio has a different sample rate.

## Trimming

Training uses a fixed length `data.trim_num_samples` (96000 samples by default = 6 seconds at 16 kHz). If a file is longer, a random window is taken; if it's shorter, it's repeat-padded to the target length.

## Noise and reverberation (MUSAN, RIRS)

The augmentation config (`configs/augment/default.yaml`) relies on two root paths supplied via environment variables:

- `MUSAN_PATH` — MUSAN root (used in `AddBackgroundNoise`)
- `RIRS_PATH` — RIRS root (used in `ApplyImpulseResponse`)

Download and unpack:

```
MUSAN: https://www.openslr.org/17/
RIRS : https://www.openslr.org/28/
```

Both can be unpacked into any directory; it's enough for `torch-audiomentations` to see subdirectories containing `.wav` files. Pass the path via `run.sh`:

```bash
MUSAN_PATH="/.../musan"
RIRS_PATH="/.../RIRS_NOISES"
```

If MUSAN/RIRS are not available, disable the augmentations via CLI overrides: `augment.waveform_levels.1.p=0.0` and `augment.waveform_levels.2.p=0.0`. Alternatively, edit `configs/augment/default.yaml`.

## Validation (VoxCeleb1)

Validation requires the VoxCeleb1 wav-tree and trial files (`voxceleb1_o.txt`, `voxceleb_e.txt`, `voxceleb_h.txt`). The paths and list are passed via `run.sh`:

```bash
VOXCELEB_VAL_ROOT="/.../VoxCeleb"
VOXCELEB_TRIAL_FILES="[${VOXCELEB_VAL_ROOT}/voxceleb1_o.txt, ...]"
```

The trial file uses the standard format: `<label> <utt1> <utt2>` (space-separated). Relative paths inside the trial file are resolved relative to `validation.data_root` (usually `.../VoxCeleb/wav`).

## Variant validation (augmented sets)

`variant_validation` runs validation over subdirectories that contain the same files, but passed through a specific codec/augmentation. Useful for assessing robustness.

```bash
VARIANT_VAL_ENABLED=true
VARIANT_VAL_SCAN_ROOT="/.../VoxCelebAug"
```

Such subdirectories can be built with `scripts/build_voxceleb_trial_audio_variants.py` or `scripts/build_voxceleb_hard_augmentation_dataset.py`.

## Codec augmentations

If `codec_only` / `codec_then_augment` are enabled in `configs/data/default.yaml`, you need `ffmpeg` with the required codecs (see `INSTALLATION.md` §4). Check:

```bash
python scripts/generate_codec_audio_variants.py <sample.wav> --output_dir ./outputs/test_codecs
```

If any of the required codecs fails, training will frequently fail on codec augmentation (or silently skip it). Either fix `ffmpeg`, or disable the codec branch (`data.codec_preset=null`, zero probabilities for `codec_only`/`codec_then_augment`).

## Next

Prepare an `audio_path` manifest in Parquet: see [`MANIFESTS.md`](MANIFESTS.md).
