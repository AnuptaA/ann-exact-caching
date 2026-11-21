#!/bin/sh
# Headline: extra cache sizes 3% and 0.3% (SPACEV Zipf traces, TriviaQA ip), after the timing batch
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export LD_LIBRARY_PATH=${RECAL_THIRD_PARTY:-$HOME/third_party}/aker/build/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/boost-install/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/faiss-install/lib64
while [ ! -f ${RECAL_QUEUE_DIR:-/tmp/mx}/TIMING3B.DONE ]; do sleep 300; done
( xargs -P 48 -I{} sh simulations/approximate_hits/mx_run_job.sh 0 {} < ${RECAL_QUEUE_DIR:-/tmp/mx}/HX.txt ) &
( xargs -P 16 -I{} sh simulations/approximate_hits/mx_run_job.sh 0 {} < ${RECAL_QUEUE_DIR:-/tmp/mx}/HXA.txt ) &
( xargs -P 6 -I{} sh simulations/approximate_hits/mx_run_job.sh 0 {} < ${RECAL_QUEUE_DIR:-/tmp/mx}/tstates_sizes.txt; touch ${RECAL_QUEUE_DIR:-/tmp/mx}/TSTATES_SIZES.DONE ) &
wait
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/HX.DONE
