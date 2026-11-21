#!/bin/sh
# Timing v5 (doc 9af): two-stage hit path, base node cache in every arm, audits timed apart.
#   0. stage-1 / stage-2 search lists of every state (run_stage1_all.sh, before the quiet arms);
#   1. DiskANN-only arms with the base node cache (10 ms gap), one at a time on a quiet machine;
#   2. every technique job, composed (charged the v5 baseline latencies), 12 at once.
# Restartable: finished JSONs are skipped.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
REG=$M/registry; mkdir -p $REG
reg() { sh -c "$1" & j=$!; sleep 6; for c in $(pgrep -P $j); do echo "v5|-|timing_v5 timing 0" > $REG/$c; done; wait $j; }
sh $S/run_stage1_all.sh        # stage-1 / stage-2 search lists of every state (skips finished ones)
.venv/bin/python $S/timing_jobs.py --set v5 | grep "eval_timing_backend.py" > $M/v5_arms.txt
echo "$(date +%T) $(grep -c . $M/v5_arms.txt) arm jobs"
while IFS= read -r l; do echo "$(date +%T) arm: $(echo "$l" | grep -o 'online/[^ ]*' | head -1)"; reg "$l"; done < $M/v5_arms.txt
echo "$(date +%T) composed technique runs"
.venv/bin/python $S/run_timing_composed.py --workers 12 --set v5 --version timing_v5
echo "$(date +%T) done"
touch $M/TIMING_V5.DONE
