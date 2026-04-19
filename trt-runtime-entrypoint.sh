#!/usr/bin/env bash
set -euo pipefail

# Torch/TorchCodec load CUDA shared libs from wheel-provided directories. Surface the common
# torch/NVIDIA package locations before any Python workers start so forked DataLoader workers
# inherit a usable LD_LIBRARY_PATH.
if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
  _asv_cuda_ld="$("${VIRTUAL_ENV}/bin/python" -c "
import pathlib, site, sys

patterns = (
    'torch/lib',
    'nvidia/cu13/lib',
    'nvidia/cudnn/lib',
    'nvidia/cusparselt/lib',
    'nvidia/npp*/lib',
    'nvidia/npp*/lib64',
)
roots = []
seen = set()
for raw in [*site.getsitepackages(), *sys.path]:
    if not raw:
        continue
    try:
        root = pathlib.Path(raw).resolve()
    except Exception:
        continue
    if not root.is_dir():
        continue
    key = str(root)
    if key in seen:
        continue
    seen.add(key)
    roots.append(root)

parts = []
added = set()
for root in roots:
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir():
                resolved = str(path.resolve())
                if resolved not in added:
                    added.add(resolved)
                    parts.append(resolved)
print(':'.join(parts))
" 2>/dev/null || true)"
  if [[ -n "${_asv_cuda_ld}" ]]; then
    export LD_LIBRARY_PATH="${_asv_cuda_ld}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

# Optional one-shot engine build + smoke test when the container starts. Docker build does not
# expose GPUs to RUN instructions; compose sets `gpus: all` for the running service only.

if [[ "${ASV_PREBAKE_ENGINES:-1}" != "1" ]]; then
  exec "$@"
fi

if [[ -f "${ASV_EXPORT_SUMMARY}" && "${ASV_FORCE_PREBAKE:-0}" != "1" ]]; then
  exec "$@"
fi

mkdir -p "${ASV_ARTIFACTS_DIR}"

python benchmarking/export_checkpoint_to_tensorrt.py \
  --checkpoint "${ASV_CHECKPOINT}" \
  --model_name "${ASV_MODEL_NAME}" \
  --model_config "${ASV_MODEL_CONFIG}" \
  --output_dir "${ASV_ARTIFACTS_DIR}" \
  --report_json "${ASV_EXPORT_SUMMARY}" \
  --batch_size "${ASV_BATCH_SIZE}" \
  --num_samples "${ASV_NUM_SAMPLES}" \
  --gpu_id "${ASV_GPU_ID}" \
  --seed "${ASV_SEED}" \
  --torch_reference_device "${ASV_TORCH_REFERENCE_DEVICE}" \
  --workspace_gib "${ASV_WORKSPACE_GIB}" \
  --builder_optimization_level "${ASV_BUILDER_OPT_LEVEL}" \
  --max_aux_streams "${ASV_MAX_AUX_STREAMS}"

python benchmarking/trt_smoke_test.py \
  --summary_json "${ASV_EXPORT_SUMMARY}" \
  --variant "${ASV_DEFAULT_VARIANT}" \
  --gpu_id "${ASV_GPU_ID}" \
  --batch_size "${ASV_BATCH_SIZE}" \
  --num_samples "${ASV_NUM_SAMPLES}"

exec "$@"
