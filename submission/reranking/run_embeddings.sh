#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="/home/kirill/ASV/outputs/weigts"
VOXCELEB_ROOT="/home/kirill/mnt/ssd_1tb_1/VoxCeleb"
MANIFEST_PATH="${VOXCELEB_ROOT}/voxceleb1_manifest.parquet"
OUTPUT_PKL="/home/kirill/ASV/submission/reranking/outputs/redimnet2_b6_voxceleb1_embeddings.pkl"

MODEL_CONFIG="configs/model/redimnet2_b6_sphereface2.yaml"
TRIAL_FILES=(
  "${VOXCELEB_ROOT}/voxceleb1_o.txt"
  "${VOXCELEB_ROOT}/voxceleb_e.txt"
  "${VOXCELEB_ROOT}/voxceleb_h.txt"
)

BATCH_PER_GPU=1
WORKERS_PER_GPU=4
SAMPLE_RATE=16000
TRIM_NUM_SAMPLES=0

NUM_GPUS=4
MIXED_PRECISION="no"

if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  set -a
  source ".env"
  set +a
fi

run_embeddings_sh_main() {
  if [ ! -e "${WEIGHTS}" ]; then
    echo "[run_embeddings.sh] ERROR: WEIGHTS path does not exist: ${WEIGHTS}"
    return 2
  fi
  if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "[run_embeddings.sh] ERROR: MANIFEST_PATH file does not exist: ${MANIFEST_PATH}"
    return 2
  fi
  if [ ! -d "${VOXCELEB_ROOT}" ]; then
    echo "[run_embeddings.sh] ERROR: VOXCELEB_ROOT directory does not exist: ${VOXCELEB_ROOT}"
    return 2
  fi
  if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "${CONDA_PREFIX}/bin/accelerate" ]; then
    echo "[run_embeddings.sh] ERROR: accelerate not found in active environment (${CONDA_PREFIX:-unset})."
    return 2
  fi

  mkdir -p "$(dirname "${OUTPUT_PKL}")"

  # Make pip-installed NVIDIA CUDA libs visible to torchcodec workers.
  if [ -f "./activate_asv_cuda_libs.sh" ]; then
    # shellcheck disable=SC1091
    source "./activate_asv_cuda_libs.sh"
  fi

  "${CONDA_PREFIX}/bin/accelerate" launch \
    --multi_gpu \
    --num_processes "${NUM_GPUS}" \
    --num_machines 1 \
    --mixed_precision "${MIXED_PRECISION}" \
    --dynamo_backend no \
    submission/reranking/run_embeddings.py \
    --weights "${WEIGHTS}" \
    --manifest-path "${MANIFEST_PATH}" \
    --voxceleb-root "${VOXCELEB_ROOT}" \
    --output-pkl "${OUTPUT_PKL}" \
    --model-config "${MODEL_CONFIG}" \
    --trial-files "${TRIAL_FILES[@]}" \
    --batch-size "${BATCH_PER_GPU}" \
    --num-workers "${WORKERS_PER_GPU}" \
    --sample-rate "${SAMPLE_RATE}" \
    --trim-num-samples "${TRIM_NUM_SAMPLES}"
}

if ! run_embeddings_sh_main "$@"; then
  echo "[run_embeddings.sh] Embedding run did not start due to prerequisite check failure."
  if [ "${BASH_SOURCE[0]}" != "$0" ]; then
    return 0
  fi
fi
