#!/bin/sh
# TriviaQA DiskANN study (doc 9r): base + pq768 + mips first, then r128 + mem
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=24
S=simulations/approximate_hits
L=$S/results/backends/index_study
mkdir -p $L
for v in base pq768 mips; do .venv-ann/bin/python -u $S/index_study.py --variant $v > $L/log_$v.txt 2>&1 & done
wait
for v in r128 mem; do .venv-ann/bin/python -u $S/index_study.py --variant $v > $L/log_$v.txt 2>&1 & done
wait
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/INDEX_STUDY.DONE
