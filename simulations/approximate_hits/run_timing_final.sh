#!/bin/sh
# The rest of the final timing set (doc 9ad), after the composed-timing checks:
#   1. DiskANN-only arms (10 ms gap), one at a time on a quiet machine (the s50 ones; spacev-aker-0.99 exists);
#   2. every remaining technique job, composed, 12 at once (run_timing_composed.py).
# Restartable: finished JSONs are skipped.
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx
S=simulations/approximate_hits
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
REG=$M/registry; mkdir -p $REG
reg() { sh -c "$1" & j=$!; sleep 6; for c in $(pgrep -P $j); do echo "tf|-|timing_final timing 0" > $REG/$c; done; wait $j; }
while [ -f $M/TIMING_GATE ]; do sleep 60; done
.venv/bin/python $S/timing_jobs.py --set final | grep "eval_timing_backend.py" | grep -v "backend_s20" > $M/tf_arms.txt
echo "$(date +%T) $(grep -c . $M/tf_arms.txt) arm jobs"
while IFS= read -r l; do echo "$(date +%T) arm: $(echo "$l" | grep -o 'backend_s[0-9]*.json' | head -1) $(echo "$l" | grep -o 'online/[^ ]*' | head -1)"; reg "$l"; done < $M/tf_arms.txt
echo "$(date +%T) composed technique runs"
.venv/bin/python $S/run_timing_composed.py --workers 12
echo "$(date +%T) done"
touch $M/TIMING_FINAL.DONE
