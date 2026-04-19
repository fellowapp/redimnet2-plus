#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Key experiment parameters
# --------------------------
# MANIFEST_PATH="/data/additional/t1_1/VoxCeleb/voxceleb1_manifest.parquet"
MANIFEST_PATH="/data/additional/t1_2/all_datasets_training_manifest_with_tidy_plus_kespeech.parquet"
# (same as scripts/build_parquet_manifest.py prints as num_classes / ASV_NUM_CLASSES).
# Override in .env if you use a different manifest root.
ASV_NUM_CLASSES=63934 
MUSAN_PATH="/data/additional/DATASETS/audio/noises/musan"
RIRS_PATH="/data/additional/DATASETS/audio/RIRS_NOISES"

MODEL_CFG="redimnet2_b6_sphereface2.yaml"
LOSS_CFG="sphereface2"
BATCH_PER_GPU=24
WORKERS_PER_GPU=4 # try to encrease from 4 up to 6 current speed is approsimately 1.4902 elements per second, ~1.5585 after codecs changing, spectrogram on gpu computation
PIN_MEMORY=true
PREFETCH_FACTOR=2
TOTAL_EPOCHS=4
SINGLE_BATCH_TEST=false
GRAD_ACCUM_STEPS=10
MAX_GRAD_NORM=20.0
START_LR=3e-5
FINAL_LR=5e-6
LOG_EVERY_STEPS=1
# Optional: initialize a completely new training stage from previous model weights.
# Accepts a checkpoint directory, checkpoint root with latest.json, or model file.
INIT_WEIGHTS="/data/additional/borodin_sam/ASV/outputs/trainin_b6_lmft/2026-04-16/22-39-23/checkpoints/epoch_0005_step_00014800"
# full = load all matching weights; encoder_only = encoder only; encoder_bridge = encoder + bridge.
INIT_MODE="full"
# Optional: resume Accelerate checkpoint: explicit path to .../epoch_XXXX_step_XXXXXXXX, or "latest"
# (reads train.checkpoint_dir/latest.json — same Hydra run only; otherwise pass the full checkpoint path).
RESUME_FROM_CHECKPOINT=""

# Hydra run dir: outputs/<MLOP_RUN_NAME>/<date>/<time>/ (via MLOP_RUN). Use path-safe names (no /).
MLOP_RUN_NAME="trainin_b6_lmft_long"
MLOP_ENABLED=true
# MLOP_ENABLED=false
# Optional: set to override train.checkpoint_dir (default is ${hydra:runtime.output_dir}/checkpoints).
CHECKPOINT_DIR=""

NUM_GPUS=6
MIXED_PRECISION="bf16"
ACCELERATE_CONFIG="configs/accelerate/fsdp_6gpu_bf16.yaml"

VALIDATION_ENABLED=true
VOXCELEB_VAL_ROOT="/data/additional/t1_2/VoxCeleb"
VOXCELEB_TRIAL_FILES="[${VOXCELEB_VAL_ROOT}/voxceleb1_o.txt,${VOXCELEB_VAL_ROOT}/voxceleb_e.txt,${VOXCELEB_VAL_ROOT}/voxceleb_h.txt]"
# Per-augmentation / per-codec trial wav trees (non-hidden subdirs of this path are validated).
VARIANT_VAL_ENABLED=true
VARIANT_VAL_SCAN_ROOT="/data/additional/t1_2/VoxCelebAug"
VALIDATE_BEFORE_FIRST_EPOCH=true
VALIDATE_EVERY_N_STEPS=400
VAL_BATCH_PER_GPU=32
VAL_WORKERS_PER_GPU=4
VAL_PIN_MEMORY="${PIN_MEMORY}"
VAL_PREFETCH_FACTOR="${PREFETCH_FACTOR}"





if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  set -a
  source ".env"
  set +a
fi
run_sh_main() {
  export MLOP_RUN="${MLOP_RUN_NAME}"
  export MUSAN_PATH
  export RIRS_PATH
  export SINGLE_BATCH_TEST

  # Refuse to run if there are uncommitted changes.
  if [ -n "$(git status --porcelain)" ]; then
    echo "[run.sh] ERROR: working tree is not clean. Commit/stage all files before training."
    git status --short
    return 2
  fi

  GIT_COMMIT_HASH="$(git rev-parse HEAD)"

  NUM_CLASSES="${ASV_NUM_CLASSES}"
  LOGGING_BACKEND="mlop"
  case "${MLOP_ENABLED,,}" in
    0|false|no|off)
      LOGGING_BACKEND="none"
      ;;
  esac

  export HYDRA_FULL_ERROR=1

  if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/accelerate" ]; then
    echo "[run.sh] ERROR: accelerate not found in active environment (${CONDA_PREFIX:-unset})."
    return 2
  fi

  # Make pip-installed NVIDIA CUDA libs visible to torchcodec workers even before
  # the user re-activates the conda environment with persistent hooks installed.
  if [ -f "./activate_asv_cuda_libs.sh" ]; then
    # shellcheck disable=SC1091
    source "./activate_asv_cuda_libs.sh"
  fi

  RESUME_OVERRIDES=()
  if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    RESUME_OVERRIDES+=(train.resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}")
  fi

  INIT_OVERRIDES=()
  if [ -n "${INIT_WEIGHTS}" ]; then
    INIT_OVERRIDES+=(train.init_weights="${INIT_WEIGHTS}")
    INIT_OVERRIDES+=(train.init_mode="${INIT_MODE}")
  fi

  CHECKPOINT_OVERRIDES=()
  if [ -n "${CHECKPOINT_DIR}" ]; then
    CHECKPOINT_OVERRIDES+=(train.checkpoint_dir="${CHECKPOINT_DIR}")
  fi

  "${CONDA_PREFIX}/bin/accelerate" launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    train.py \
    model="${MODEL_CFG}" \
    loss="${LOSS_CFG}" \
    data.manifest_path="${MANIFEST_PATH}" \
    data.batch_size="${BATCH_PER_GPU}" \
    data.num_workers="${WORKERS_PER_GPU}" \
    data.pin_memory="${PIN_MEMORY}" \
    data.prefetch_factor="${PREFETCH_FACTOR}" \
    train.epochs="${TOTAL_EPOCHS}" \
    train.grad_accum_steps="${GRAD_ACCUM_STEPS}" \
    train.max_grad_norm="${MAX_GRAD_NORM}" \
    "${CHECKPOINT_OVERRIDES[@]}" \
    "${RESUME_OVERRIDES[@]}" \
    "${INIT_OVERRIDES[@]}" \
    train.log_every_steps="${LOG_EVERY_STEPS}" \
    train.git_commit_hash="${GIT_COMMIT_HASH}" \
    train.auto_num_classes=false \
    logging.backend="${LOGGING_BACKEND}" \
    model.num_classes="${NUM_CLASSES}" \
    optim.lr="${START_LR}" \
    optim.final_lr="${FINAL_LR}" \
    validation.enabled="${VALIDATION_ENABLED}" \
    validation.data_root="${VOXCELEB_VAL_ROOT}/wav" \
    validation.trial_files="${VOXCELEB_TRIAL_FILES}" \
    validation.run_before_first_epoch="${VALIDATE_BEFORE_FIRST_EPOCH}" \
    validation.every_n_steps="${VALIDATE_EVERY_N_STEPS}" \
    validation.batch_size="${VAL_BATCH_PER_GPU}" \
    validation.num_workers="${VAL_WORKERS_PER_GPU}" \
    validation.pin_memory="${VAL_PIN_MEMORY}" \
    validation.prefetch_factor="${VAL_PREFETCH_FACTOR}" \
    variant_validation.enabled="${VARIANT_VAL_ENABLED}" \
    variant_validation.scan_root="${VARIANT_VAL_SCAN_ROOT}"
}

# `set -e` can terminate the current process when a command fails. Wrap the
# main entry point so failures are printed and the script returns/finishes
# with a friendly non-fatal status, especially when sourced in a terminal.
if ! run_sh_main "$@"; then
  echo "[run.sh] Training did not start due prerequisite check failure."
  if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 0
  fi
fi