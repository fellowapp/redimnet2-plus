#!/usr/bin/env bash
set -euo pipefail

if [ -z "${CONDA_PREFIX:-}" ]; then
  return 0 2>/dev/null || exit 0
fi

ASV_PYTHON_VERSION="$("${CONDA_PREFIX}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export _ASV_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export _ASV_OLD_PATH="${PATH:-}"

ASV_LIB_DIR_PATTERNS=(
  "${CONDA_PREFIX}/ffmpeg-custom/lib"
  "${CONDA_PREFIX}/lib"
  "${CONDA_PREFIX}/lib/python${ASV_PYTHON_VERSION}/site-packages/torch/lib"
  "${CONDA_PREFIX}/lib/python${ASV_PYTHON_VERSION}/site-packages/nvidia/cu13/lib"
  "${CONDA_PREFIX}/lib/python${ASV_PYTHON_VERSION}/site-packages/nvidia/cudnn/lib"
  "${CONDA_PREFIX}/lib/python${ASV_PYTHON_VERSION}/site-packages/nvidia/npp*/lib"
  "${CONDA_PREFIX}/lib/python${ASV_PYTHON_VERSION}/site-packages/nvidia/npp*/lib64"
  "${CONDA_PREFIX}/lib/python${ASV_PYTHON_VERSION}/site-packages/nvidia/cusparselt/lib"
)

shopt -s nullglob
for ASV_LIB_PATTERN in "${ASV_LIB_DIR_PATTERNS[@]}"; do
  for ASV_LIB_DIR in ${ASV_LIB_PATTERN}; do
    if [ -d "${ASV_LIB_DIR}" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${ASV_LIB_DIR}:"*) ;;
        *) export LD_LIBRARY_PATH="${ASV_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
      esac
    fi
  done
done
shopt -u nullglob

ASV_FFMPEG_BIN="${CONDA_PREFIX}/ffmpeg-custom/bin"
if [ -d "${ASV_FFMPEG_BIN}" ]; then
  case ":${PATH:-}:" in
    *":${ASV_FFMPEG_BIN}:"*) ;;
    *) export PATH="${ASV_FFMPEG_BIN}${PATH:+:${PATH}}" ;;
  esac
fi

if [ -z "${LD_LIBRARY_PATH:+x}" ]; then
  unset LD_LIBRARY_PATH
fi