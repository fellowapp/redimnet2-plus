# Training

Full pipeline: manifest → DataLoader (decode + codec/augment) → spec-frontend → model (Encoder→Bridge→Classifier) → loss → Accelerate/FSDP.

## What happens on launch

The entry point is `train.py` (Hydra + `asv/train/run.py`), which then calls `asv/train/loop.py::train()`. It:

1. Builds the model (`build_model(cfg.model)`) and loss (`build_loss(cfg.loss)`).
2. (Optionally) initializes weights from a checkpoint: `train.init_weights` + `train.init_mode`.
3. `torch.compile(mode=cfg.train.torch_compile_mode)`.
4. Spins up `Accelerator` with `mixed_precision` and `gradient_accumulation_steps`, prepares the DataLoader.
5. May resume from an Accelerate checkpoint (`train.resume_from_checkpoint`).
6. Main epoch loop: forward, loss, `accelerator.backward`, `clip_grad_norm`, cosine scheduler.
7. Validation: every `validation.every_n_steps` steps and at the end of each epoch.
8. Checkpointing: every `train.save_every_epochs` epochs (or on validation, if that runs by steps).

## Quick start

Everything is wired up in `run.sh`. Open it, configure the top block, and run:

```bash
bash run.sh
```

The minimal set of variables we actually touch every time:

| Variable in `run.sh`        | What it is                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| `MANIFEST_PATH`             | Parquet manifest (see [`MANIFESTS.md`](MANIFESTS.md))                  |
| `ASV_NUM_CLASSES`           | Number of unique `speaker_id` values in the manifest                    |
| `MUSAN_PATH`, `RIRS_PATH`   | MUSAN/RIRS roots for augmentation                                       |
| `MODEL_CFG`                 | YAML name from `configs/model/` (without `.yaml`)                       |
| `LOSS_CFG`                  | YAML name from `configs/loss/`                                          |
| `BATCH_PER_GPU`             | Batch size per GPU                                                      |
| `WORKERS_PER_GPU`           | DataLoader's `num_workers`                                              |
| `TOTAL_EPOCHS`              | How many epochs to run                                                  |
| `GRAD_ACCUM_STEPS`          | Gradient accumulation                                                   |
| `START_LR`, `FINAL_LR`      | Cosine schedule: from `START_LR` to `FINAL_LR` (over the number of optimizer steps) |
| `MAX_GRAD_NORM`             | Gradient clipping                                                       |
| `INIT_WEIGHTS`, `INIT_MODE` | Fresh-start init (not a resume!); `init_mode: full/encoder_only/encoder_bridge` |
| `RESUME_FROM_CHECKPOINT`    | Accelerate resume; empty string = from scratch, `"latest"` = from `latest.json`|
| `NUM_GPUS`                  | Number of processes (GPUs)                                              |
| `ACCELERATE_CONFIG`         | FSDP config (see `configs/accelerate/fsdp_*gpu_bf16.yaml`)              |

`run.sh` refuses to start training if the working tree has uncommitted changes (`git status --porcelain`). Commit or push, or temporarily comment out the check.

## Validation

Configured in `run.sh`:

```bash
VALIDATION_ENABLED=true
VOXCELEB_VAL_ROOT="/data/.../VoxCeleb"
VOXCELEB_TRIAL_FILES="[${VOXCELEB_VAL_ROOT}/voxceleb1_o.txt,${VOXCELEB_VAL_ROOT}/voxceleb_e.txt,${VOXCELEB_VAL_ROOT}/voxceleb_h.txt]"
VALIDATE_BEFORE_FIRST_EPOCH=true
VALIDATE_EVERY_N_STEPS=400
```

With `run_before_first_epoch=true`, EER is computed before the first step — handy as a baseline.

Variant validation (robustness to codecs/augmentations):
```bash
VARIANT_VAL_ENABLED=true
VARIANT_VAL_SCAN_ROOT="/data/.../VoxCelebAug"
```
Every non-empty subdirectory of `VARIANT_VAL_SCAN_ROOT` gets its own EER (`val/variants/eer_<subdir>`), plus a pooled-hard EER across all of them.

## Mixed precision and FSDP

Accelerate configs:
- `configs/accelerate/fsdp_2gpu_bf16.yaml`
- `configs/accelerate/fsdp_4gpu_bf16.yaml`
- `configs/accelerate/fsdp_6gpu_bf16.yaml`

All three use FSDP FULL_SHARD, compute/param/reduce dtype = bf16, `state_dict_type: FULL_STATE_DICT`. To target a different number of GPUs, just copy one and adjust `gpu_ids` / `num_processes`. It's worth making `main_process_port` unique per host if you're running several trainings in parallel.

## Checkpoints

- Root: `${CHECKPOINT_DIR}` (defaults to `${output_dir}/checkpoints`, where `output_dir` is the Hydra runtime dir under `outputs/<run_name>/<date>/<time>/`).
- One checkpoint = one subdirectory `epoch_XXXX_step_XXXXXXXX/`.
- `latest.json` at the checkpoint root points to the most recent one.
- `training_state.json` inside a checkpoint stores `epoch`, `global_step`, `optimizer_step`, `next_epoch` — needed for a correct resume.

Resume:
```bash
RESUME_FROM_CHECKPOINT="latest"                   # from ${CHECKPOINT_DIR}/latest.json
RESUME_FROM_CHECKPOINT="/path/epoch_0005_step_00014800"  # absolute path
```

Post-resume verification (`train.verify_checkpoint_load: true`) compares the weight signature before/after `load_state` and fails if nothing changed (protects against a bad path).

Fresh-start init (starting a new stage from old weights, without the optimizer):
```bash
INIT_WEIGHTS="/path/epoch_0005_step_00014800"
INIT_MODE="full"   # full | encoder_only | encoder_bridge
```
`init_weights` and `resume_from_checkpoint` are mutually exclusive — training fails if both are set.

## Single-batch test

For a quick sanity check of the training pipeline (a single batch, EER ≈ 0 on that same data):

```bash
SINGLE_BATCH_TEST=true bash run.sh
```

Should finish with `val/single_batch/eer <= 1e-6`; otherwise something broke (model, loss, shapes, augmentations).

## How long everything takes (rough guide)

On `b6` + 6 GPUs + FSDP bf16 + batch/gpu=24 — roughly 1.5 samples/s per GPU without codec augmentation, less with codecs. Actual numbers depend heavily on the hardware and the share of the codec branch (`transform_probabilities.codec_only` + `codec_then_augment`). Logs show EMA loss and grad_norm at every step.

## Scripts that may come in handy

- `scripts/benchmark_data_pipeline.py` — run the DataLoader without the model to see where the bottleneck is (decode, codec, augmentation).
- `scripts/preview_training_dataset_augmentations.py` — listen to/inspect exactly what the dataset produces after codecs/augmentations.
- `scripts/generate_codec_audio_variants.py` — check that all required codecs are available on the system.
- `scripts/visualize_manifest_distributions.py`, `scripts/visualize_speaker_distribution.py` — manifest statistics.

## See also

- [`CONFIGS.md`](CONFIGS.md) — map of Hydra groups and which YAML controls what.
- [`DATA_PREPROCESSING_PIPELINE.md`](DATA_PREPROCESSING_PIPELINE.md) — processing graph inside the dataset/collator.
- [`EVALUATION.md`](EVALUATION.md) — what to run after training (submission / TRT).
