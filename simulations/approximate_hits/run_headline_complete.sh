#!/bin/sh
# headline runs to completion (outputs checked), self-adjusting admission (mx_run_job v4)
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export LD_LIBRARY_PATH=${RECAL_THIRD_PARTY:-$HOME/third_party}/aker/build/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/boost-install/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/faiss-install/lib64
C=simulations/approximate_hits/mx_complete.sh
( sh $C ${RECAL_QUEUE_DIR:-/tmp/mx}/HR.txt ${RECAL_QUEUE_DIR:-/tmp/mx}/HR_C.DONE 24 ) &
( sh $C ${RECAL_QUEUE_DIR:-/tmp/mx}/HX.txt ${RECAL_QUEUE_DIR:-/tmp/mx}/HX_C.DONE 24 ) &
( sh $C ${RECAL_QUEUE_DIR:-/tmp/mx}/HXA.txt ${RECAL_QUEUE_DIR:-/tmp/mx}/HXA_C.DONE 8 ) &
wait
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/REQ.DONE
