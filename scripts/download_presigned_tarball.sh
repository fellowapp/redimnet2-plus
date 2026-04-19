#!/usr/bin/env bash
# Download a presigned object (e.g. R2/S3) to a .tar.gz without saving HTML error pages.
# Usage: download_presigned_tarball.sh 'https://...?...' [output.tar.gz]
set -euo pipefail

if [ "${#}" -lt 1 ] || [ "${#}" -gt 2 ]; then
  echo "Usage: ${0##*/} 'PRESIGNED_URL' [output_path]" >&2
  exit 2
fi

URL=$1
if [ "${#}" -eq 2 ]; then
  OUT=$2
else
  base=$(basename "${URL%%\?*}")
  OUT=${base:-download.tar.gz}
fi

PARTIAL="${OUT}.partial"
rm -f "${PARTIAL}"

# -f: fail on 4xx/5xx so we never grow a multi-GB HTML "success" file.
curl -fL --retry 5 --retry-delay 10 --connect-timeout 30 \
  -o "${PARTIAL}" \
  "${URL}"

magic=$(xxd -l 2 -p "${PARTIAL}" | tr -d '\n')
if [ "${magic}" != "1f8b" ]; then
  echo "[download_presigned_tarball] ERROR: response is not gzip (expected magic 1f8b, got ${magic})." >&2
  echo "[download_presigned_tarball] First 240 bytes of the response:" >&2
  head -c 240 "${PARTIAL}" | cat -v >&2
  echo >&2
  rm -f "${PARTIAL}"
  exit 1
fi

mv -f "${PARTIAL}" "${OUT}"
echo "[download_presigned_tarball] OK: ${OUT} ($(wc -c < "${OUT}" | tr -d ' ') bytes)"
