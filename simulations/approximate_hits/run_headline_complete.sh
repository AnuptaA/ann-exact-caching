#!/bin/sh
# headline runs to completion (outputs checked), self-adjusting admission (mx_run_job v4)
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export LD_LIBRARY_PATH=/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:/home/jialind/third_party/faiss-install/lib64
C=simulations/approximate_hits/mx_complete.sh
( sh $C /tmp/mx/HR.txt /tmp/mx/HR_C.DONE 24 ) &
( sh $C /tmp/mx/HX.txt /tmp/mx/HX_C.DONE 24 ) &
( sh $C /tmp/mx/HXA.txt /tmp/mx/HXA_C.DONE 8 ) &
wait
touch /tmp/mx/REQ.DONE
