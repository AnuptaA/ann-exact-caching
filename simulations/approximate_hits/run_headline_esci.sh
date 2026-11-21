#!/bin/sh
# ESCI headline: real Aker (stable grid + tuned, random stream) at 0.3/1/3/10%; extra sizes for the rest
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export LD_LIBRARY_PATH=${RECAL_THIRD_PARTY:-$HOME/third_party}/aker/build/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/boost-install/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/faiss-install/lib64
C=simulations/approximate_hits/mx_complete.sh
( sh $C ${RECAL_QUEUE_DIR:-/tmp/mx}/HEA.txt ${RECAL_QUEUE_DIR:-/tmp/mx}/HEA_C.DONE 12 ) &
( sh $C ${RECAL_QUEUE_DIR:-/tmp/mx}/HE.txt ${RECAL_QUEUE_DIR:-/tmp/mx}/HE_C.DONE 16 ) &
wait
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/HE.DONE
