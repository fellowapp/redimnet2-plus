#!/usr/bin/env bash
set -euo pipefail

_asv_python="${VIRTUAL_ENV:-/workspace/.venv}/bin/python"
if [[ ! -x "${_asv_python}" ]]; then
  _asv_python="$(command -v python)"
fi

# cu129 wheels install CUDA 12 user-space libraries under package-specific folders.
# Export all matching library roots before Python starts so workers inherit them too.
if [[ -n "${_asv_python:-}" ]] && [[ -x "${_asv_python}" ]]; then
  _asv_cuda_ld="$("${_asv_python}" -c "
import pathlib, site, sys

patterns = (
    'torch/lib',
    'nvidia/*/lib',
    'nvidia/*/lib64',
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

exec python submission/run_submission_unified.py \
  --weights-dir /workspace/weights \
  --output-csv /workspace/outputs/output.csv \
  "$@"
