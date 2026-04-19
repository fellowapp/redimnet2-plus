# ASV Training Data Preprocessing Pipeline — Detailed Graph and Gates

## 1) Entry point and control flow

Training starts from `asv/train/run.py` and enters `asv/train/loop.py`.

1. `run.py` validates `cfg.data.manifest_path`, optionally infers `cfg.model.num_classes` from manifest speaker IDs, builds model/loss, and calls `train(cfg, model, loss_fn)`.
2. `loop.py` calls `_build_train_loader(cfg)`:
   - builds a `ParquetAudioDataset` from `cfg.data` + `cfg.augment`
   - uses `torch.utils.data.DataLoader` with `collate_waveform_batch`
3. In the training loop, each raw batch (`inputs`, `labels`) is passed through:
   - `compute_spectrogram_batch(batch["inputs"], spec_frontend, augment_cfg=spec_augment_cfg, training=True)`
   - which internally runs the model frontend and spectrogram augmentations.

This means the effective runtime graph for training is:

```
Manifest -> ParquetAudioDataset.__getitem__ -> collate_waveform_batch -> compute_spectrogram_batch -> model
```

`spec_frontend` is extracted from the model at runtime (`model.encoder.spec_frontend`) and frozen per-worker/collator copy is prepared.

## 2) Data source layer (manifest creation)

All downstream preprocessing assumes the manifest schema:

- `audio_path`, `rel_path`, `speaker_id`, `video_id`, `segment_id`, `audio_id`, `duration_sec`.

The manifest is written by `asv/manifest/write.py` (`write_manifest`) and driven by the dataset scanners (`iter_ogg_rows`, `iter_voxceleb1_rows`, `iter_voxceleb2_rows`).  

`write_manifest`:
- streams rows into Arrow schema batches,
- optionally joins duration values from a duration parquet,
- writes a memory-friendly parquet file used by training dataset.

This is the very first gate: only records present in the manifest can ever enter the pipeline.

## 3) Dataset level: wave-level source and branch selection

Core dataset implementation: `asv/data/parquet_dataset.py` (`ParquetAudioDataset`).

### 3.1 Manifest and indexing
- Builds a row-index `(row_group, row_index)` for all rows for O(1) random access.
- Builds `speaker_id -> label_id` mapping in `__init__` and exposes `num_speakers`.
- Caches one parquet row-group at a time to reduce I/O overhead.

### 3.2 Decode path and compatibility
Each sample request goes through `_decode` -> `load_with_torchcodec` (`asv/data/decode.py`).

`load_with_torchcodec`:
- first tries `torchcodec.decoders.AudioDecoder`,
- when `window_num_samples` is set, it samples a random window for long files (deterministic to the sample’s duration gate behavior),
- falls back to `torchaudio.load_with_torchcodec` when needed and resamples to requested `sample_rate` if required.

`ensure_torchaudio_io_compat()` patches `torchaudio.info` / `torchaudio.load` on environments where these are missing so augmentation libs (`torch_audiomentations`) still work for side resources (MUSAN/RIRS).

### 3.3 Decode failure gate
`ParquetAudioDataset.__getitem__` wraps decode + transform in `_waveform_for_index`.  
If decode fails:
- when `fallback_on_decode_error=False` → fail immediately,
- when enabled → fallback scans previous indices, then later indices, and retries until a valid row is found.

This is a resilience gate designed to avoid worker crashes due to a single corrupt file.

### 3.4 Length normalization gates
- `_trim_if_long`: trims only when waveform is longer than `trim_num_samples`.
- `_pad_or_trim`: final normalization to exact `trim_num_samples` via `pad_random`:
  - if too long: random crop
  - if too short: repeating-padding then slice
- Output length invariant for downstream model/collator assumptions.

### 3.5 Mono conversion gate
- If multichannel, `mean(dim=0)` is applied.
- 1D inputs pass through unchanged.

### 3.6 Sample-level branching (`keep / codec / augment / codec_augment`)
The branching is controlled by `CodecAugmentProbabilities.choose_mode()`; probabilities are required to sum to `1.0`.

Modes:
- `keep`: no codec and no waveform augmentation.
- `codec`: codec roundtrip only.
- `augment`: waveform augmentation only.
- `codec_augment`: codec then waveform augmentation.

`codec_min_duration_sec` is an additional gate: when set, codec branches are skipped for short recordings.

## 4) Codec layer (optional) and its gates

Codec simulation class: `asv/codec/pipeline.py` (`CodecPipeline`).

### 4.1 Codec configuration path
- If `codec_preset` is set in config, one fixed preset pipeline is used.
- If not set and codec modes are enabled, dataset builds a random selectable list of available presets (`list_presets()` + availability check).
- For random mode each item can choose one preset from available ones (per-item random choice).

### 4.2 Backend gate
`CodecPipeline` supports:
- `auto` (default): prefers ffmpeg when both export/decode formats are in compatible format set,
- `ffmpeg` explicit,
- `pydub` explicit.

### 4.3 ffmpeg vs pydub specifics
- ffmpeg path:
  - converts tensor/array -> int16 PCM bytes,
  - encodes to format configured in preset,
  - decodes back to PCM,
  - returns normalized sample-rate output.
- pydub path:
  - builds `AudioSegment`,
  - exports/imports via preset codec format,
  - optional reframe + PCM conversion.

Both throw `CodecUnavailableError` on runtime failures (e.g., missing codec tool/path), enabling fail-fast diagnostics upstream.

## 5) Waveform augmentation pipeline layer

Implemented in `asv/augment/pipeline.py` (`AudioAugmentPipeline` + `AugmentationLevel`).

### 5.1 Build and config
`build_audio_augmentor` reads `cfg.augment` and instantiates:
- `waveform_levels` (actual applied in training dataset),
- `feature_levels` (currently intended for feature-domain pipeline usage, but in current training path spectrogram augment goes through `SpecAugment`).

### 5.2 Level semantics
Each `AugmentationLevel` has:
- `p`: level gate,
- `one_of`: apply exactly one transform vs chain all transforms,
- `shuffle`: optional per-level permutation.

`apply_waveform`:
- early-return if gate fails,
- handles `[T]`, `[C,T]`, `[B,C,T]` inputs by adding temporary batch/channel dims,
- returns waveform in the same semantic shape once transforms are applied.

### 5.3 Actual waveform transform groups in defaults
Configured in `configs/augment/default.yaml`:
- `filters`: band-pass / band-stop / high-pass / low-pass,
- `additive_noise`: colored noise, background noise, RawBoost,
- `rir`: impulse response application,
- `perturb`: speed perturbation.

Each group has its own level gate (`p`) and `one_of=true` by default.

## 6) Collation layer

`asv/data/collate.py` has two options:

### 6.1 `collate_waveform_batch` (active in training loop)
- Validates waveform shape is `[T]` or `[C, T]`.
- Produces:
  - `inputs`: `[B, 1, T]` for 1D waveforms or `[B, C, T]` for multichannel.
  - `labels`: `LongTensor`.

### 6.2 `compute_spectrogram_batch` / `SpectrogramCollator`
- `compute_spectrogram_batch` forces frontend to eval under no-grad, then applies `apply_spec_augment`.
- `SpectrogramCollator` exists as a drop-in alternative when you want collate-time spectrogram extraction.

## 7) Spectrogram-layer augmentation graph

Config extraction: `asv/train/spec_augment.py`

Source precedence:
1. `augment.feature_levels` entry named `spec_augment`,
2. fallback to `data.spec_augment`.

`SpecAugmentConfig` gates:
- `level_p`: whether to apply any spectrogram augmentation at all,
- optional `one_of` mode,
- per-op params: `freq_mask_p`, `time_mask_p`, `cutmix_p`,
- max widths / frame ranges.

`apply_spec_augment` behavior:
- if `not training`: passthrough.
- if `rand >= level_p`: passthrough.
- if `one_of=True`: chooses one of (freq mask, time mask, cutmix) uniformly.
- else each op runs independently by its own Bernoulli (`p`).

### CutMix specifics
- If batch size is 1, cutmix becomes no-op.
- Chooses random partner permutation, random time width within bounds, random start, and copies time slices from partner sample.

## 8) End-to-end training-time data graph (explicit)

Current default path (`input_mode: waveform` in config):

```mermaid
flowchart TD
    %% ---------------------------
    %% Graph from dataset init to ready batch
    %% ---------------------------

    subgraph T[Training graph startup]
        R[run(cfg) in asv/train/run.py]
        TR[train(cfg, model, loss_fn) in asv/train/loop.py]
        BL[_build_train_loader(cfg)]
        LF[build_spec_augment_cfg(cfg)]
        ST[loop over train_loader batches]
        PT[_preprocess_training_batch(batch, spec_frontend, spec_augment_cfg)]
        CF[_forward_and_loss(...)]
    end

    subgraph DS_INIT[ParquetAudioDataset construction]
        PD[ParquetAudioDataset.__init__]
        IDX[Build index: list[(row_group, row_idx)]]
        SM[_build_speaker_index() -> speaker_id->label]
        CA[Codec settings]
        RN[Random preset names via list_presets/check_preset_available]
        AUG[build_audio_augmentor(augment_cfg)]
        DL[DataLoader(dataset, collate_fn=collate_waveform_batch)]
    end

    subgraph DS_ITEM[Per-sample runtime (__getitem__)]
        GI[_waveform_for_index(idx)]
        RG[_read_row(row_group,row_idx) with row-group cache]
        DE[_decode(path)]
        TC[load_with_torchcodec(path, sample_rate, window_num_samples)]
        W[Waveform decoded to float32 tensor]
        DC{Decode succeeded?}
        FB[Fallback path on decode error if enabled]
        TI[_trim_if_long(max_len)]
        PM[CodecAugmentProbabilities.choose_mode()]
        GateMode{Mode gate}
        C1{Mode in {codec, codec_augment}?}
        C2{_should_apply_codec(row)?}
        AP[_apply_codec(waveform,sr)]
        CP[_get_codec_pipeline()]
        CPP{codec_preset set?}
        RP[CodecPipeline(preset/backend)]
        FE[CodecPipeline.selected_backend()]
        FF[ffmpeg roundtrip path]
        PY[pydub roundtrip path]
        RT[_apply_to_waveform(levels)]
        AGG[AudioAugmentPipeline.apply_to_waveform]
        A1[for each AugmentationLevel]
        A2{level p gate}
        A3{one_of? / shuffle}
        MONO[_to_mono(mean over channels)]
        PTN[_pad_or_trim(trim_num_samples)]
        PAD[pad_random: repeat-pad or random crop]
        OUT[({"waveform","label"} dict)]
    end

    subgraph COLLATE[Batch assembly]
        CW[collate_waveform_batch(batch)]
        STCK{first.shape[0]=1? first.ndim==1}
        WAVE[inputs: [B,1,T] or [B,C,T]]
        LAB[labels: LongTensor]
        CS[compute_spectrogram_batch(inputs, spec_frontend, augment_cfg, training=True)]
        PR[ _prepare_waveform_batch_for_frontend ]
        FE2[spec_frontend.eval() + no_grad]
        SA[apply_spec_augment]
        SP[SpecAugmentConfig gates]
        FM[freq mask]
        TM[time mask]
        CM[cutmix]
        RB[ready_batch={"inputs"=spec, "labels"}]
    end

    R --> TR
    TR --> BL
    TR --> LF
    TR --> ST
    ST --> PT --> CF
    CF --> CF_RET[{"inputs"=spectrogram batch, "labels"}]
    CF_RET --> TR

    BL --> PD --> IDX --> SM --> CA --> RN --> AUG --> DL
    DL --> GI
    GI --> RG --> DE --> DC
    DC -->|no| W --> TI --> PM --> GateMode
    DC -->|yes| FB --> GI

    GateMode -->|keep| MONO
    GateMode -->|codec| C1
    GateMode -->|augment| RT
    GateMode -->|codec_augment| C1

    C1 -->|no| RT
    C1 -->|yes| C2
    C2 -->|no| RT
    C2 -->|yes| AP --> CP --> CPP
    CPP -->|preset fixed| RP
    CPP -->|random from available set| RP
    CP --> FE
    FE -->|auto->ffmpeg| FF
    FE -->|auto->pydub| PY
    FF --> RT
    PY --> RT

    RT --> A1 --> A2 --> A3 --> MONO
    A1 -->|level skipped| MONO
    A2 -->|p fail| MONO
    A3 -->|one_of| A2
    A3 -->|chain all| A2
    MONO --> PTN --> PAD --> OUT

    OUT --> CW
    CW --> STCK
    STCK --> WAVE --> CS
    CS --> PR --> FE2 --> SA
    SA --> SP
    SP -->|one_of| FM
    SP -->|one_of| TM
    SP -->|one_of| CM
    SP -->|independent| FM
    SP -->|independent| TM
    SP -->|independent| CM
    FM --> RB
    TM --> RB
    CM --> RB
    RB --> PT
    RB --> CF_RET

    FE -->|selected backend| FF
    FE -->|selected backend| PY
```

Validation and utility paths reuse the same decode + spectrogram compute primitives via `asv/validation/voxceleb.py`, but with their own sampler and batch structure.

## 9) Config knobs that gate behavior

- `configs/data/default.yaml`
  - `trim_num_samples` (target length for pad/trim),
  - `codec_preset`, `sample_rate`, `num_workers`, etc.
  - `transform_probabilities` (DO NOT CHANGE the listed keys unless requested explicitly; these are treated as experiment-locked in this repo).
- `configs/augment/default.yaml`
  - waveform-level gating in `waveform_levels`,
  - spec augment control under `feature_levels` name `spec_augment` (read by `build_spec_augment_cfg`),
  - comments clarify spectrogram augmentation is applied in `collate` path, not by `AudioAugmentPipeline.apply_to_features`.
- Launch script (`run.sh`) wires CLI overrides for data/augment/model/training validation, including:
  - `data.manifest_path`, `data.batch_size`, `data.num_workers`,
  - model/loss/lr/resume/checkpoint args,
  - workspace/validation controls.

## 10) Practical interpretation of “layers”

Think of the pipeline as a DAG with explicit decision nodes:

- **Source-layer gate**: manifest row visibility.
- **Decode-layer gate**: decode success / fallback.
- **Length gate**: trim/pad to fixed tensor length.
- **Mode gate**: `CodecAugmentProbabilities` branch.
- **Codec gate**: codec preset availability + min-duration check.
- **Augmentation-layer gates**: per-augmentation level p, one_of, shuffle.
- **Batching gate**: shape expectations for waveform vs features.
- **Spectrogram gate**: `training` + `level_p`/per-op p and optional `one_of`.

This decomposition is useful when comparing experiments:
- disable codec with `codec_preset: null` to isolate waveform augment impact,
- set one of `transform_probabilities` keys to isolate augmentation-only behavior,
- adjust `spec_augment` levels for frontend/domain-level stress only.

## 11) Why this structure exists

The design keeps:
- expensive I/O and waveform distortion steps inside dataset workers,
- fixed-size tensor output before collation,
- frontend transformation in a controlled training-only branch,
- augmentation policy modular and config-driven with deterministic graph-level gates.

The result is a reproducible, stochastic but inspectable pipeline with explicit checkpoints and test coverage around failure fallbacks, gates, and augmentation shapes.
