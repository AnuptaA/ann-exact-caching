#!/bin/sh
# Second composed-timing check (doc 9ad), SPACEV Zipf 0.99, 1%: the DiskANN-alone arm rerun with a 10 ms
# gap between searches (matching the spacing of a technique run's timed requests), then the six
# techniques composed, one at a time and 6 at once (4 cores each). Live results: timing_v4 (unchanged).
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
R=$S/results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
P=spacev-aker-0.99
REG=$M/registry; mkdir -p $REG
reg() { sh -c "$1" & j=$!; sleep 6; for c in $(pgrep -P $j); do echo "cc2|-|compose_check2 timing 0" > $REG/$c; done; wait $j; }
echo "$(date +%T) arm with gap"
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from simulations.approximate_hits import timing_jobs as tj
print('\n'.join(tj.arm_lines('$P', 50)))" > $M/cc2_arm.txt
while IFS= read -r l; do reg "$l"; done < $M/cc2_arm.txt
D=$R/timing_v4/$P
mk() {
  N=$R/timing_compose/$1/$P
  while IFS= read -r l; do
    printf '%s\n' "$l" | sed "s|$D/|$N/|g; s|$N/state_b|$D/state_b|g; s|--no-backend-arms|--no-backend-arms --compose-backend $D/backend_s50_lat.npy|"
  done < $M/cc_$P.txt
}
mk seq > $M/cc2_seq.txt; mk par > $M/cc2_par.txt
echo "$(date +%T) composed, sequential"
while IFS= read -r l; do reg "$l"; done < $M/cc2_seq.txt
echo "$(date +%T) composed, parallel (6 at once)"
i=0
while IFS= read -r l; do
  c0=$((8 + 4 * i)); i=$((i + 1))
  l2=$(printf '%s\n' "$l" | sed "s|--wait-quiet 30|--allow-busy|; s|\.venv-ann/bin/python|taskset -c $c0-$((c0 + 3)) .venv-ann/bin/python|")
  ( reg "$l2" ) &
done < $M/cc2_par.txt
wait
echo "$(date +%T) done"
touch $M/COMPOSE_CHECK2.DONE
