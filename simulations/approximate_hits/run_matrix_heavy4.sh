#!/bin/sh
# heavy pool, reordered: every cell's seed 0 first (ours, then LRU), then seed 1
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
xargs -P 7 -I{} sh simulations/approximate_hits/mx_run_job.sh 70 {} < ${RECAL_QUEUE_DIR:-/tmp/mx}/H_4.txt
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/H_4.DONE
