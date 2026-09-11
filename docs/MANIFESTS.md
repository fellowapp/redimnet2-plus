# Manifests (Parquet)

Anything that touches training goes through a single Parquet manifest. It's the single source of truth defining "what the dataset actually sees."

## Schema

File: `asv/manifest/schema.py`. Columns:

| Column          | Type    | Purpose                                              |
|-----------------|---------|------------------------------------------------------|
| `audio_path`    | string  | Absolute path to the audio (read via `torchcodec`)   |
| `rel_path`      | string  | Path relative to the dataset root                    |
| `speaker_id`    | string  | Speaker label (defines the class)                    |
| `video_id`      | string  | Video/session ID (for grouping)                      |
| `segment_id`    | string  | Segment ID (usually the file stem)                   |
| `audio_id`      | string  | Stable sample ID `speaker/video/segment`              |
| `duration_sec`  | double  | Length in seconds (optional, may be null)             |

Types are strictly checked on concatenation (`asv/manifest/concat.py`). Compression: zstd.

## Building a single manifest

The `scripts/build_parquet_manifest.py` script with the `--dataset` flag scans the required layout and writes a Parquet file:

```bash
# VoxCeleb2
python scripts/build_parquet_manifest.py --dataset voxceleb2 \
  /path/to/VoxCeleb2 /path/to/output_dir

# VoxCeleb1 (1251 classes)
python scripts/build_parquet_manifest.py --dataset voxceleb1 \
  /path/to/VoxCeleb1 /path/to/output_dir

# CN-Celeb / CN-Celeb2 / CN-Celeb all splits
python scripts/build_parquet_manifest.py --dataset cnceleb2 \
  /path/to/CNCeleb2 /path/to/output_dir

# 3D-Speaker / tidy / KeSpeech
python scripts/build_parquet_manifest.py --dataset kespeech \
  /path/to/KeSpeech /path/to/output_dir

# Kryp (speaker-id challenge): reads train.csv
python scripts/build_parquet_manifest.py --dataset kryp \
  /path/to/Kryp /path/to/output_dir
```

Output:
```
Wrote .../<dataset>_manifest.parquet (N rows)
num_classes=<K>
ASV_NUM_CLASSES=<K>
```

The `num_classes` / `ASV_NUM_CLASSES` values are needed for `run.sh` (the `ASV_NUM_CLASSES` variable) and for `model.num_classes`.

Full list of `--dataset` values: `voxceleb1`, `voxceleb2`, `cnceleb`, `cnceleb2`, `cnceleb_all`, `3dspeaker`, `kryp`, `tidy`, `kespeech`.

## Options

- `--output-name NAME` — custom name for the `.parquet` file (default `<dataset>_manifest.parquet`).
- `--durations-parquet PATH` — join durations from a separate Parquet file (columns `path`, `duration`).
- `--no-durations` — don't read duration files at all (even if one is present by default).
- `--probe-durations` / `--no-probe-durations` — for `--dataset kryp` only: read durations from FLAC headers (`soundfile.info`). Enabled by default for Kryp.

## Durations (why they matter)

The `duration_sec` column is used for `codec_min_duration_sec` in `configs/data/default.yaml`: if a file's length is below the threshold, the codec branch is skipped (short files often don't encode well, or encode with artifacts). If `duration_sec` is empty, the threshold check doesn't apply and the codec is always applied.

## Concatenating multiple manifests

If you want to train on several datasets at once (VoxCeleb2 + CN-Celeb2 + …), merge the manifests with `scripts/concat_manifests.py`:

```bash
python scripts/concat_manifests.py \
  /path/voxceleb2_manifest.parquet \
  /path/cnceleb2_manifest.parquet \
  /path/merged_manifest.parquet \
  --speaker-prefix-a "vox2_" \
  --speaker-prefix-b "cn2_"
```

Prefixes are required if speaker IDs overlap between the manifests (otherwise the script will complain). If the overlap is intentional — i.e. it's genuinely the same speaker — pass `--allow-speaker-overlap`.

`scripts/concat_training_manifests.py` is an extended version for N manifests, with the same rules.

## Standalone durations

`asv/manifest/durations.py` holds helpers for mapping full paths to "short" keys (`wav/...`, `dev/aac/...`, etc.) — used when joining with `--durations-parquet`. If you need to build a durations-parquet file ahead of time:

```bash
# Basic format: columns (path, duration). Building it is manual (no ready-made script is provided).
# Examples in asv/manifest/durations.py: the *_rel_from_any_path functions show what key is expected.
```

For a quick pass without separate parsing: use `--no-durations` (the column stays null), or `--probe-durations` for `--dataset kryp`.

## Usage in training

In `run.sh`:

```bash
MANIFEST_PATH="/path/to/merged_manifest.parquet"
ASV_NUM_CLASSES=63934     # number of unique speaker_id values
```

`train.py` then spins up a `ParquetAudioDataset(manifest_path, ...)` (see `asv/data/parquet_dataset.py`), which:

1. Indexes `(row_group, row_index)` for O(1) access.
2. Builds a `speaker_id -> label` mapping over the whole manifest (deterministically, via sorting).
3. Caches one `row_group` at a time, so reads don't jump around between files.

To validate the schema manually (if in doubt):
```python
from asv.manifest.concat import validate_manifest_schema
from pathlib import Path
validate_manifest_schema(Path("/path/to/manifest.parquet"))
```
