#!/usr/bin/env bash
# siftsmall for validating the QVCache artifact (its README expects it; the clone does not contain it).
# SIFT is NOT part of the experiments: this data is only used by scripts/validate_qvcache.py.
#   bash scripts/fetch_qvcache_validation_data.sh [--dry-run]
# Environment: TP (third-party prefix, default $HOME/third_party), QVCACHE_DIR (default $TP/qvcache).
set -euo pipefail
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
TP="${TP:-$HOME/third_party}"
ART="${QVCACHE_DIR:-$TP/qvcache}"
URLS=("ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" "http://corpus-texmex.irisa.fr/siftsmall.tar.gz")
WORK="$ART/data/.siftsmall_download"
DATA="$ART/data/siftsmall"
run() {
    if [ "$DRY" -eq 1 ]; then printf '[dry-run] %s\n' "$*"; else "$@"; fi
}
run mkdir -p "$WORK" "$DATA"
if [ ! -f "$WORK/siftsmall.tar.gz" ]; then
    for url in "${URLS[@]}"; do
        if run curl -fL --retry 3 -o "$WORK/siftsmall.tar.gz" "$url"; then break; fi
    done
fi
run tar -xzf "$WORK/siftsmall.tar.gz" -C "$WORK"
run "$ART/build/utils/fvecs_to_bin" float "$WORK/siftsmall/siftsmall_base.fvecs" "$DATA/siftsmall_base.bin"
run "$ART/build/utils/fvecs_to_bin" float "$WORK/siftsmall/siftsmall_query.fvecs" "$DATA/siftsmall_query.bin"
run "$ART/build/utils/ivecs_to_bin" "$WORK/siftsmall/siftsmall_groundtruth.ivecs" "$DATA/siftsmall_groundtruth.bin"
run bash "$ART/scripts/generate_noisy_queries.sh"
run bash "$ART/python/scripts/build_index/build_faiss_index.sh"
echo "siftsmall ready in $DATA (validation only; not an experiment dataset)"
