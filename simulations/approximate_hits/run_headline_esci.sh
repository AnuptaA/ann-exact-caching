#!/bin/sh
# ESCI headline: real Aker (stable grid + tuned, random stream) at 0.3/1/3/10%; extra sizes for the rest
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export LD_LIBRARY_PATH=/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:/home/jialind/third_party/faiss-install/lib64
C=simulations/approximate_hits/mx_complete.sh
( sh $C /tmp/mx/HEA.txt /tmp/mx/HEA_C.DONE 12 ) &
( sh $C /tmp/mx/HE.txt /tmp/mx/HE_C.DONE 16 ) &
wait
touch /tmp/mx/HE.DONE
