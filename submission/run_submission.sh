#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Submission evaluation args
# --------------------------
WEIGHTS="/data/additional/borodin_sam/ASV/outputs/trainin_b6_7kinds_of_data_resume/2026-04-16/08-35-34/checkpoints/epoch_0010_step_00031690"
TEST_CSV="/data/additional/t1_2/kryp/test_public.csv"
AUDIO_ROOT="/data/additional/t1_2/kryp/test_public"
OUTPUT_CSV="/data/additional/borodin_sam/ASV/redimnet2_b6_6kinds_of_data.csv"

MODEL_CONFIG="configs/model/redimnet2_b6_sphereface2.yaml"
TOP_K=40
BATCH_PER_GPU=1
WORKERS_PER_GPU=4
SAMPLE_RATE=16000
# If 0, variable-length audio is used and batch size must stay 1.
TRIM_NUM_SAMPLES=0

NUM_GPUS=6
MIXED_PRECISION="bf16"
ACCELERATE_CONFIG="configs/accelerate/fsdp_6gpu_bf16.yaml"

if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  set -a
  source ".env"
  set +a
fi

run_submission_sh_main() {
  if [ -z "${WEIGHTS}" ]; then
    echo "[run_submission.sh] ERROR: set WEIGHTS to a checkpoint file or checkpoint directory."
    return 2
  fi
  if [ -z "${TEST_CSV}" ]; then
    echo "[run_submission.sh] ERROR: set TEST_CSV to the submission CSV path."
    return 2
  fi
  if [ -z "${AUDIO_ROOT}" ]; then
    echo "[run_submission.sh] ERROR: set AUDIO_ROOT to the base audio directory."
    return 2
  fi
  if [ -z "${OUTPUT_CSV}" ]; then
    echo "[run_submission.sh] ERROR: set OUTPUT_CSV to the desired output file."
    return 2
  fi

  if [ ! -e "${WEIGHTS}" ]; then
    echo "[run_submission.sh] ERROR: WEIGHTS path does not exist: ${WEIGHTS}"
    return 2
  fi
  if [ ! -f "${TEST_CSV}" ]; then
    echo "[run_submission.sh] ERROR: TEST_CSV file does not exist: ${TEST_CSV}"
    return 2
  fi
  if [ ! -d "${AUDIO_ROOT}" ]; then
    echo "[run_submission.sh] ERROR: AUDIO_ROOT directory does not exist: ${AUDIO_ROOT}"
    return 2
  fi

  if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/accelerate" ]; then
    echo "[run_submission.sh] ERROR: accelerate not found in active environment (${CONDA_PREFIX:-unset})."
    return 2
  fi

  # Make pip-installed NVIDIA CUDA libs visible to torchcodec workers.
  if [ -f "./activate_asv_cuda_libs.sh" ]; then
    # shellcheck disable=SC1091
    source "./activate_asv_cuda_libs.sh"
  fi

  "${CONDA_PREFIX}/bin/accelerate" launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    submission/run_submission.py \
    --weights "${WEIGHTS}" \
    --test-csv "${TEST_CSV}" \
    --audio-root "${AUDIO_ROOT}" \
    --output-csv "${OUTPUT_CSV}" \
    --model-config "${MODEL_CONFIG}" \
    --top-k "${TOP_K}" \
    --batch-size "${BATCH_PER_GPU}" \
    --num-workers "${WORKERS_PER_GPU}" \
    --sample-rate "${SAMPLE_RATE}" \
    --trim-num-samples "${TRIM_NUM_SAMPLES}"
}

if ! run_submission_sh_main "$@"; then
  echo "[run_submission.sh] Submission run did not start due to prerequisite check failure."
  if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 0
  fi
fi
