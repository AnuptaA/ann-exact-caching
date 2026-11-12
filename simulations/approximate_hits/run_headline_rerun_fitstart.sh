#!/bin/sh
# headline: ours / LRU re-run with the fit-split starting threshold (doc 9p)
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
( xargs -P 40 -I{} sh simulations/approximate_hits/mx_run_job.sh 0 {} < /tmp/mx/HR.txt ) &
# the 0.3% timing states (rebuilt directly): flag when all three exist
( T=simulations/approximate_hits/results/timing_v3
  while [ ! -f $T/spacev-aker-0.3/state_b0.003.json ] || [ ! -f $T/spacev-aker-0.6/state_b0.003.json ] || [ ! -f $T/spacev-aker-0.99/state_b0.003.json ]; do sleep 120; done
  touch /tmp/mx/TSTATES_003.DONE ) &
wait
touch /tmp/mx/HR.DONE
