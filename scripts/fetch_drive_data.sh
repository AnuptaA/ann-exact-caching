#!/bin/sh
# Fetch TripClick and MedRAG vectors from the shared Drive folder (by file id, only the allowed files).
#   sh scripts/fetch_drive_data.sh --local [--dry-run]     TripClick, MedRAG small files (needs about 0.5 GB)
#   sh scripts/fetch_drive_data.sh --cluster [--dry-run]   adds the MedRAG corpus (about 74 GB; Linux cluster only)
# Needs gdown (a small venv: python3 -m venv .venv-fetch && .venv-fetch/bin/pip install gdown).
# Refuses to start when fewer than 10 GB would remain free. Data stays gitignored. Do not re-share it.
set -e
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${FETCH_PY:-.venv-fetch/bin/python}
[ -x "$PY" ] || PY=python3
"$PY" scripts/drive_fetch.py "$@"
for mode in "$@"; do
  [ "$mode" = "--dry-run" ] && exit 0
done
"$PY" scripts/verify_data.py tripclick --dir datasets/aker
"$PY" scripts/verify_data.py medrag
