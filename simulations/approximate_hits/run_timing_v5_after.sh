#!/bin/sh
# After run_timing_v5.sh: rerun the jobs that crashed (AIO limit before the fix, doc 9af), then plot + summary.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}; S=simulations/approximate_hits
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
while [ ! -f $M/TIMING_V5.DONE ]; do sleep 60; done
python3 $S/move_audit_bug.py          # runs with the double-subtracted audit charge (doc 9af)
echo "$(date +%T) rerun of missing / audit-bug jobs"
.venv/bin/python $S/run_timing_composed.py --workers 12 --set v5 --version timing_v5
TIMING_VERSION=timing_v5 .venv/bin/python $S/plot_timing.py
python3 $S/v5_summary.py > $S/results/latest/timing_v5_summary.txt
echo "$(date +%T) done"
touch $M/TIMING_V5_AFTER.DONE
