#!/bin/sh
# heavy pool, reordered: every cell's seed 0 first (ours, then LRU), then seed 1
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
xargs -P 7 -I{} sh simulations/approximate_hits/mx_run_job.sh 70 {} < /tmp/mx/H_4.txt
touch /tmp/mx/H_4.DONE
