#!/bin/sh
# doc 9ah: hnswlib vs FAISS sanity benchmark, then full node-cache curves (no early stop) for ESCI and
# SPACEV real (recall 0.95), quiet machine; node_cache.json is not touched.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
S=simulations/approximate_hits; R=$S/results
PYTHONPATH=. .venv-core/bin/python -u $S/bench_hnsw_libs.py --out $R/backends/bench_hnsw_libs.json > ${RECAL_QUEUE_DIR:-/tmp/mx}/bench_hnsw.txt 2>&1
for P in esci1m spacev-real95; do
  S2=trace; [ $P = esci1m ] && S2=random
  .venv-ann/bin/python -u $S/node_cache_curve.py --prep $R/online/$P --stream $S2 --full --max-frac 0.1 --wait-quiet 30 \
    --out $R/backends/$P/node_cache_full.json > $R/backends/$P/logs/node_cache_full.txt 2>&1
done
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/BENCH_CURVES.DONE
