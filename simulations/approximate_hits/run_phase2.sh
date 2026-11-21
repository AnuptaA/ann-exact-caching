#!/bin/sh
# Phase 2 (doc 9ae), unattended, after the TripClick chain's timing finished:
#   (TripClick / MedRAG timing under the old setup was stopped and set aside: the setup is being changed)
#   2. node-cache curve per index (quiet, sequential): where the latency gain saturates;
#   3. adaptive stage-1 tuning (CPU only, 3 cells at a time) on the 10% states of a first set of cells.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
R=$S/results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
REG=$M/registry; mkdir -p $REG
reg() { sh -c "$1" & j=$!; sleep 6; for c in $(pgrep -P $j); do echo "p2|-|phase2 timing 0" > $REG/$c; done; wait $j; }
for P in spacev-aker-0.6 spacev-aker-0.3 spacev-real esci1m triviaqa-ip-0.6 tripclick-ip medrag-ip; do
  S2=trace; [ $P = esci1m ] && S2=random
  TR=0.95; [ $P = spacev-real ] && TR=0.9
  O=$R/backends/$P/node_cache.json
  mkdir -p $R/backends/$P/logs
  echo "$(date +%T) node cache curve $P"
  reg "[ -f $O ] || .venv-ann/bin/python -u $S/node_cache_curve.py --prep $R/online/$P --stream $S2 --target-recall $TR --wait-quiet 30 --out $O > $R/backends/$P/logs/node_cache.txt 2>&1"
done
echo "$(date +%T) stage-1 tuning"
for P in spacev-aker-0.6 triviaqa-ip-0.6 esci1m; do
  S2=trace; [ $P = esci1m ] && S2=random
  for B in 0.01 0.1; do
    O=$R/timing_v4/$P/stage1_b$B.json
    mkdir -p $R/timing_v4/$P/logs
    [ -f $O ] || .venv-ann/bin/python -u $S/stage1_tune.py --prep $R/online/$P --state $R/timing_v4/$P/state_b$B.npz --stream $S2 --out $O > $R/timing_v4/$P/logs/stage1_b$B.txt 2>&1 &
    sleep 2
  done
done
wait
touch $M/PHASE2.DONE
echo "$(date +%T) done"
