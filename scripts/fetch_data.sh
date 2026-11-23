#!/bin/sh
# Fetch the evaluation datasets, one mode per dataset, then verify them.
#   sh scripts/fetch_data.sh --spacev --triviaqa --sphere [--dry-run]
#   sh scripts/fetch_data.sh --all [--no-corpus] [--dry-run]    (ESCI is never part of --all)
#   sh scripts/fetch_data.sh --list
# Needs gdown for Drive files (python3 -m venv .venv-fetch && .venv-fetch/bin/pip install gdown).
# Free space is checked before every download. Data stays gitignored. TripClick must not be shared.
set -e
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY=${FETCH_PY:-.venv-fetch/bin/python}
[ -x "$PY" ] || PY=python3
"$PY" -m scripts.data_fetch "$@"
for arg in "$@"; do
  case "$arg" in --dry-run|--list) exit 0 ;; esac
done
DIR=${AKER_DIR:-datasets/aker}
for arg in "$@"; do
  case "$arg" in
    --spacev|--all) "$PY" scripts/verify_data.py spacev --dir "$DIR" ;;
  esac
  case "$arg" in
    --sphere|--all) "$PY" scripts/verify_data.py sphere --dir "$DIR" ;;
  esac
  case "$arg" in
    --triviaqa|--all) "$PY" scripts/verify_data.py triviaqa --dir "$DIR" ;;
  esac
  case "$arg" in
    --tripclick|--all) "$PY" scripts/verify_data.py tripclick --dir "$DIR" ;;
  esac
  case "$arg" in
    --medrag|--all) "$PY" scripts/verify_data.py medrag --aker-dir "$DIR" ;;
  esac
done
