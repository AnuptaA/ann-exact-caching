#!/bin/sh
# Does composed timing (doc 9ad) reproduce live timing? On a QUIET machine (nothing else of ours running).
#   1. live      TriviaQA Zipf 0.6, 1%: the six techniques (SPACEV Zipf 0.99 1%: live results already in timing_v4)
#   2. seq       composed (--compose-backend), one at a time, quiet guard on
#   3. parallel  composed, all 12 jobs (2 cells x 6 techniques) at once, each pinned to 4 cores, guard off
# Outputs: timing_v4/<prep>/ (live), timing_compose/seq/<prep>/, timing_compose/par/<prep>/.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
R=$S/results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
PREPS="spacev-aker-0.99 triviaqa-ip-0.6"
REG=$M/registry; mkdir -p $REG
# the cells' commands (sample 1 in 50, 1%), from timing_jobs.lines
for P in $PREPS; do
  .venv/bin/python - $P > $M/cc_$P.txt <<'PY'
import sys
sys.path.insert(0, ".")
from simulations.approximate_hits import timing_jobs as tj
print("\n".join(tj.lines(sys.argv[1], 0.01, 50, techs=("ours", "ours_drift", "ours_online", "lru", "aker", "qvcache"))))
PY
done
# 1. live (the real results of these cells)
reg() { sh -c "$1" & j=$!; sleep 6; for c in $(pgrep -P $j); do echo "cc|-|compose_check timing 0" > $REG/$c; done; wait $j; }
echo "$(date +%T) live"
while IFS= read -r l; do reg "$l"; done < $M/cc_triviaqa-ip-0.6.txt
# composed variants
mk() {  # mk <mode> <prep>  -> job lines for the composed runs
  P=$2; D=$R/timing_v4/$P; N=$R/timing_compose/$1/$P
  while IFS= read -r l; do
    printf '%s\n' "$l" | sed "s|$D/|$N/|g; s|$N/state_b|$D/state_b|g; s|--no-backend-arms|--no-backend-arms --compose-backend $D/backend_s50_lat.npy|"
  done < $M/cc_$P.txt
}
: > $M/cc_seq.txt; : > $M/cc_par.txt
for P in $PREPS; do mk seq $P >> $M/cc_seq.txt; mk par $P >> $M/cc_par.txt; done
echo "$(date +%T) composed, sequential"
while IFS= read -r l; do reg "$l"; done < $M/cc_seq.txt
echo "$(date +%T) composed, parallel (12 at once, 4 cores each)"
i=0
while IFS= read -r l; do
  c0=$((8 + 4 * i)); i=$((i + 1))
  l2=$(printf '%s\n' "$l" | sed "s|--wait-quiet 30|--allow-busy|; s|\.venv-ann/bin/python|taskset -c $c0-$((c0 + 3)) .venv-ann/bin/python|")
  ( reg "$l2" ) &
done < $M/cc_par.txt
wait
echo "$(date +%T) done"
touch $M/COMPOSE_CHECK.DONE
