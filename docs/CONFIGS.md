# Configs (Hydra)

The root config is `configs/config.yaml`. It composes groups via `defaults:`:

```yaml
defaults:
  - model: default
  - data: default
  - augment: default
  - loss: sphereface2
  - logging: default
  - _self_
  - override hydra/output: asv_run
```

`_self_` at the end means: the values in `config.yaml` itself are applied after all the groups. `hydra/output: asv_run` sets the rules for the Hydra runtime dir (see `configs/hydra/output/`).

All parameters can be overridden from the CLI: `python train.py data.batch_size=64 train.epochs=10`. That's exactly what `run.sh` does.

## Groups

### `model/`
Model = Encoder → Bridge → Classifier. Assembled in `asv/models/assemble.py::build_model()` via `hydra.utils.instantiate` on each subtree. All subtrees require `_target_`.

What's in `configs/model/`:
- `default.yaml` — baseline (conv encoder, linear classifier), mostly needed as a skeleton.
- `redimnet2_b0.yaml` … `redimnet2_b6.yaml` — ReDimNet2 in various sizes; base configuration without pretrained weights.
- `redimnet2_b0_pretrained.yaml`, `redimnet2_b6_pretrained.yaml` — load pretrained weights (see `asv.models.redimnet2.ReDimNet2Encoder`).
- `redimnet2_b0_sphereface2.yaml`, `redimnet2_b6_sphereface2.yaml` — pretrained + SphereFace2 head (used together with `loss=sphereface2`).
- `notlm_*` — variants without the language-model stage in the encoder.

Key invariant: `model.num_classes` must match the number of speakers in the manifest. `run.sh` explicitly passes through `model.num_classes=${ASV_NUM_CLASSES}`. Alternatively, you can enable `train.auto_num_classes=true`, in which case the script reads the count from the manifest at startup (slower on very large datasets).

### `data/`
`configs/data/default.yaml` — everything about the DataLoader/dataset. Key fields:

- `manifest_path` — Parquet (required, set via CLI).
- `batch_size` — per GPU.
- `num_workers`, `pin_memory`, `prefetch_factor` — DataLoader workers.
- `sample_rate: 16000` — target SR; `torchcodec` resamples on the fly.
- `trim_num_samples: 96000` — fixed length after pad/trim (6 seconds at 16 kHz).
- `codec_preset` — if set, all codec branches use this preset; if `null`, one is chosen randomly from the available presets.
- `transform_probabilities` — 4 modes per sample, summing to 1.0:
  - `keep_original` — do nothing;
  - `codec_only` — codec roundtrip only;
  - `augment_only` — waveform augmentations only (filters/noise/RIR/…);
  - `codec_then_augment` — codec, then augmentations.
  These values are locked by the experiment lock. Change them **only explicitly and by hand** (noted in `AGENTS.md`).

### `augment/`
`configs/augment/default.yaml` — description of augmentations for two domains:

- `waveform_levels` — a list of "levels" (`filters`, `additive_noise`, `rir`, `perturb`). Each level: `p`, `one_of`, `transforms` (Hydra-instantiable modules, usually from `torch_audiomentations` or `asv.augment.*`).
- `feature_levels` — a single `spec_augment` element with `freq_mask`, `time_mask`, `cutmix` blocks (read via `asv.train.spec_augment.build_spec_augment_cfg` and applied in collate, not via `AudioAugmentPipeline.apply_to_features`).

Filter parameters (cutoff_freq, bandwidth_fraction, …) are marked `DO NOT CHANGE IN ANY CASE`. Change only by hand, never through automation.

`AddBackgroundNoise` pulls files from `${oc.env:MUSAN_PATH}`, `ApplyImpulseResponse` from `${oc.env:RIRS_PATH}`. These environment variables are exported by `run.sh`.

### `loss/`
- `default.yaml` — `torch.nn.CrossEntropyLoss`.
- `label_smoothing.yaml` — label smoothing CE.
- `sphereface2.yaml` — a stub, `asv.losses.integrated.LossComputedInModel`: the loss is actually computed **inside** `SphereFace2Head` (the classifier returns `(logits, loss)`). Used together with `model=redimnet2_b6_sphereface2`.

Assembled via `asv.losses.registry.build_loss` (`_target_` + kwargs).

### `logging/`
`configs/logging/default.yaml` — logging backend. Can be disabled: `logging.backend=none` (inside `run.sh` this is the `MLOP_ENABLED=false` flag).

### `accelerate/`
Not a Hydra group — these are configs for `accelerate launch --config_file ...`. Three FSDP bf16 presets for 2/4/6 GPUs. Selected in `run.sh` via `ACCELERATE_CONFIG`.

### `hydra/`
`configs/hydra/output/asv_run.yaml` — template for the Hydra runtime dir (`output_dir`). Typically looks like `outputs/<MLOP_RUN>/<date>/<time>/` — this is where logs, the `config snapshot`, `validation_metrics.jsonl`, and `checkpoints/` are written.

## Root-level fields in `config.yaml`

Not part of any group, set directly in `config.yaml`:

- `seed` — global seed (`accelerate.utils.set_seed`).
- `output_dir = ${hydra:runtime.output_dir}` — the main output directory.
- `optim.*` — optimizer (AdamW; `lr`, `final_lr`, `weight_decay`, `beta1`, `beta2`, `eps`).
- `train.*` — training loop parameters (`epochs`, `grad_accum_steps`, `mixed_precision`, `max_grad_norm`, `save_every_epochs`, `torch_compile`, `init_weights`/`init_mode`, `resume_from_checkpoint`, `fallback_on_decode_error`, etc.).
- `validation.*` — VoxCeleb1 EER (`enabled`, `data_root`, `trial_files`, `batch_size`, `num_workers`, `every_n_steps`, `run_before_first_epoch`).
- `variant_validation.*` — EER over augmented variants.

## Useful CLI overrides

```bash
# By hand via CLI, bypassing run.sh
python train.py \
  model=redimnet2_b6_sphereface2 \
  loss=sphereface2 \
  data.manifest_path=/.../manifest.parquet \
  data.batch_size=24 \
  train.epochs=10 \
  train.grad_accum_steps=1 \
  train.mixed_precision=bf16 \
  optim.lr=3e-5 \
  validation.enabled=false
```

Hydra CLI syntax: `+key=value` (add a new field), `key=value` (override), `~key` (remove). All groups can be overridden like `model=redimnet2_b0` or swapped for another file.

## How to add your own YAML

For example, a new model size:
1. Copy `configs/model/redimnet2_b0.yaml` to `configs/model/my_variant.yaml`.
2. Adjust `encoder._target_` / fields.
3. Run: `python train.py model=my_variant ...` or set `MODEL_CFG=my_variant.yaml` in `run.sh`.

Same approach applies to `loss/`, `augment/`, etc.
