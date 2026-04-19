# Agent Instructions

## Quick Repository Overview

- This repository trains and evaluates an automatic speaker verification (ASV) system from audio manifests.
- Training is Hydra-driven: configs define model, losses, data loading, codec simulation, and augmentation behavior.
- Core flow is parquet manifest -> audio decode -> optional codec/augment transforms -> batching -> model training.
- Parquet manifests for  VoxCeleb2 are built with `scripts/build_parquet_manifest.py` (`--dataset voxceleb2`); output schema is shared (`audio_path`, `speaker_id`, etc.).
- Current data pipeline is waveform-first and includes robust handling for decode failures and multi-channel raw audio.
- Keep changes pragmatic for research workflows: preserve config-driven extensibility, avoid unnecessary abstraction, and prefer reliable defaults.

## Main rules

- Execute all commands in the `asv` conda environment.
- Use `uv` to install any new package dependencies.
- Keep modules focused: one primary responsibility per file. Split files when they combine registry/config/data conversion/runtime logic.
- Prefer small public APIs. If a helper is used by only one function/class, keep it local/private.
- Keep orchestration code thin. Training entrypoints should call dedicated builder/runner components, not contain business logic.
- Keep configuration simple: avoid adding new config groups or indirection unless there are at least two real implementations to swap.
- Avoid global mutable state when possible. If caching is needed, isolate it behind a clear API and document invalidation behavior.
- When adding codec/model/loss options, prefer data-driven declarations in config files over growing hardcoded dictionaries in Python.
- Limit "fallback path" complexity: support one default path first; add optional branches only when a documented use case exists.
- Treat the following filter parameter keys in `configs/augment/default.yaml` as user-locked and NEVER modify them unless the user explicitly asks in the current chat:
  - `waveform_levels[0].transforms[0].min_center_frequency`
  - `waveform_levels[0].transforms[0].max_center_frequency`
  - `waveform_levels[0].transforms[0].min_bandwidth_fraction`
  - `waveform_levels[0].transforms[0].max_bandwidth_fraction`
  - `waveform_levels[0].transforms[1].min_center_frequency`
  - `waveform_levels[0].transforms[1].max_center_frequency`
  - `waveform_levels[0].transforms[1].min_bandwidth_fraction`
  - `waveform_levels[0].transforms[1].max_bandwidth_fraction`
  - `waveform_levels[0].transforms[2].min_cutoff_freq`
  - `waveform_levels[0].transforms[2].max_cutoff_freq`
  - `waveform_levels[0].transforms[3].min_cutoff_freq`
  - `waveform_levels[0].transforms[3].max_cutoff_freq`
- Treat the following probability keys in `configs/data/default.yaml` as user-locked and NEVER modify them unless the user explicitly asks in the current chat:
  - `transform_probabilities.keep_original`
  - `transform_probabilities.codec_only`
  - `transform_probabilities.augment_only`
  - `transform_probabilities.codec_then_augment`
