#!/usr/bin/env bash
# TensorRT embedding export (fixed length from export summary) + mean-chain rerank.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_submission_embeddings_trt_pipeline.sh [options] [-- RERANK_EXTRA_ARGS...]

Required:
  --audio-dir DIR          Directory tree to scan for audio (ASV_AUDIO_DIR)

Defaults:
  ASV_EXPORT_SUMMARY       export_summary.json (default .../artifacts/export_summary.json)
  ASV_DEFAULT_VARIANT      TensorRT variant key in that summary
  ASV_OUTPUT_DIR           Default <repo>/submission_pipeline
  ASV_RERANK_K             Top-K for reranker (default 10)

Optional:
  --summary-json PATH      Override export summary path
  --variant KEY            Override TRT variant (default: env ASV_DEFAULT_VARIANT)
  --engine-path PATH       Override .plan; batch/length read from name ..._bs32_s96000.plan if omitted
  --output-dir DIR
  --neighbors-filepath-prefix STR
  --gpu-id N               CUDA device (default ASV_GPU_ID or 0)
  --num-workers N          Workers for decode and for rerank --cpu-workers (ASV_TRT_NUM_WORKERS, default 4; 0 = no workers)

Embeddings stage always uses --skip-eer for --audio-dir (same as the PyTorch pipeline script).

After "--", remaining args are passed only to rerank_embeddings_mean_chain_v2_fast.py.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ASV_AUDIO_DIR="${ASV_AUDIO_DIR:-}"
ASV_EXPORT_SUMMARY="${ASV_EXPORT_SUMMARY:-${REPO_ROOT}/artifacts/export_summary.json}"
ASV_DEFAULT_VARIANT="${ASV_DEFAULT_VARIANT:-frontend_fp32_model_fp16}"
ASV_OUTPUT_DIR="${ASV_OUTPUT_DIR:-${REPO_ROOT}/submission_pipeline}"
ASV_RERANK_K="${ASV_RERANK_K:-10}"
ASV_GPU_ID="${ASV_GPU_ID:-0}"
ASV_TRT_NUM_WORKERS="${ASV_TRT_NUM_WORKERS:-4}"
ASV_NEIGHBORS_FILEPATH_PREFIX="${ASV_NEIGHBORS_FILEPATH_PREFIX:-}"
ASV_OUTPUT_EMBEDDINGS="${ASV_OUTPUT_EMBEDDINGS:-}"
ASV_RERANK_OUTPUT_PREFIX="${ASV_RERANK_OUTPUT_PREFIX:-}"
ASV_RERANK_DEVICE="${ASV_RERANK_DEVICE:-}"

SUMMARY_JSON_OVERRIDE=""
VARIANT_OVERRIDE=""
ENGINE_PATH_OVERRIDE=""
TRT_BATCH_SIZE=""
TRT_NUM_SAMPLES=""
RERANK_USER_ARGS=()

NEIGHBORS_PREFIX_FLAG=()
RERANK_DEVICE_FLAG=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --audio-dir)
      ASV_AUDIO_DIR="$2"
      shift 2
      ;;
    --output-dir)
      ASV_OUTPUT_DIR="$2"
      shift 2
      ;;
    --summary-json)
      SUMMARY_JSON_OVERRIDE="$2"
      shift 2
      ;;
    --variant)
      VARIANT_OVERRIDE="$2"
      shift 2
      ;;
    --engine-path)
      ENGINE_PATH_OVERRIDE="$2"
      shift 2
      ;;
    --trt-batch-size)
      TRT_BATCH_SIZE="$2"
      shift 2
      ;;
    --trt-num-samples)
      TRT_NUM_SAMPLES="$2"
      shift 2
      ;;
    --neighbors-filepath-prefix)
      ASV_NEIGHBORS_FILEPATH_PREFIX="$2"
      shift 2
      ;;
    --gpu-id)
      ASV_GPU_ID="$2"
      shift 2
      ;;
    --num-workers)
      ASV_TRT_NUM_WORKERS="$2"
      shift 2
      ;;
    --k)
      ASV_RERANK_K="$2"
      shift 2
      ;;
    --)
      shift
      RERANK_USER_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${ASV_AUDIO_DIR}" ]]; then
  echo "Error: set --audio-dir or ASV_AUDIO_DIR." >&2
  exit 2
fi
if [[ ! -d "${ASV_AUDIO_DIR}" ]]; then
  echo "Error: audio directory not found: ${ASV_AUDIO_DIR}" >&2
  exit 2
fi

if [[ -n "${ENGINE_PATH_OVERRIDE}" ]]; then
  if [[ -z "${TRT_BATCH_SIZE}" || -z "${TRT_NUM_SAMPLES}" ]]; then
    _eng_base="$(basename "${ENGINE_PATH_OVERRIDE}")"
    if [[ "${_eng_base}" =~ _bs([0-9]+)_s([0-9]+)\.plan$ ]]; then
      TRT_BATCH_SIZE="${BASH_REMATCH[1]}"
      TRT_NUM_SAMPLES="${BASH_REMATCH[2]}"
    fi
  fi
  if [[ -z "${TRT_BATCH_SIZE}" || -z "${TRT_NUM_SAMPLES}" ]]; then
    echo "Error: with --engine-path, pass --trt-batch-size and --trt-num-samples or use a .plan name like *_bs32_s96000.plan" >&2
    exit 2
  fi
else
  if [[ ! -f "${ASV_EXPORT_SUMMARY}" ]]; then
    echo "Error: export summary not found: ${ASV_EXPORT_SUMMARY}" >&2
    echo "Hint: start the container once so the entrypoint writes it, or pass --summary-json." >&2
    exit 2
  fi
fi

mkdir -p "${ASV_OUTPUT_DIR}"

if [[ -z "${ASV_OUTPUT_EMBEDDINGS}" ]]; then
  ASV_OUTPUT_EMBEDDINGS="${ASV_OUTPUT_DIR}/embeddings_trt.pkl"
fi
if [[ -z "${ASV_RERANK_OUTPUT_PREFIX}" ]]; then
  ASV_RERANK_OUTPUT_PREFIX="${ASV_OUTPUT_DIR}/$(basename "${ASV_OUTPUT_EMBEDDINGS}" .pkl)_mean_chain_v2"
fi

if [[ -n "${ASV_NEIGHBORS_FILEPATH_PREFIX}" ]]; then
  NEIGHBORS_PREFIX_FLAG=(--neighbors-filepath-prefix "${ASV_NEIGHBORS_FILEPATH_PREFIX}")
fi
if [[ -n "${ASV_RERANK_DEVICE}" ]]; then
  RERANK_DEVICE_FLAG=(--device "${ASV_RERANK_DEVICE}")
fi

EMBED_CMD=(
  python submission/reranking/run_embeddings_trt.py
  --audio-root "${ASV_AUDIO_DIR}"
  --output-pkl "${ASV_OUTPUT_EMBEDDINGS}"
  --skip-eer
  --gpu-id "${ASV_GPU_ID}"
)
if [[ -n "${SUMMARY_JSON_OVERRIDE}" ]]; then
  EMBED_CMD+=(--summary-json "${SUMMARY_JSON_OVERRIDE}")
fi
if [[ -n "${VARIANT_OVERRIDE}" ]]; then
  EMBED_CMD+=(--variant "${VARIANT_OVERRIDE}")
fi
if [[ -n "${ENGINE_PATH_OVERRIDE}" ]]; then
  EMBED_CMD+=(--engine-path "${ENGINE_PATH_OVERRIDE}")
  EMBED_CMD+=(--trt-batch-size "${TRT_BATCH_SIZE}" --trt-num-samples "${TRT_NUM_SAMPLES}")
fi
EMBED_CMD+=(--num-workers "${ASV_TRT_NUM_WORKERS}")

echo "[pipeline-trt] repo root: ${REPO_ROOT}"
echo "[pipeline-trt] audio root: ${ASV_AUDIO_DIR}"
echo "[pipeline-trt] embeddings -> ${ASV_OUTPUT_EMBEDDINGS}"
echo "[pipeline-trt] rerank k=${ASV_RERANK_K}"

"${EMBED_CMD[@]}"

python submission/reranking/rerank_embeddings_mean_chain_v2_fast.py \
  "${ASV_OUTPUT_EMBEDDINGS}" \
  --k "${ASV_RERANK_K}" \
  --output-prefix "${ASV_RERANK_OUTPUT_PREFIX}" \
  --cpu-workers "${ASV_TRT_NUM_WORKERS}" \
  "${RERANK_DEVICE_FLAG[@]}" \
  "${NEIGHBORS_PREFIX_FLAG[@]}" \
  "${RERANK_USER_ARGS[@]}" \
  --skip-metrics

echo "[pipeline-trt] done. Neighbors CSV: ${ASV_RERANK_OUTPUT_PREFIX}.neighbors.csv"
