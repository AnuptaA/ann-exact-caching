#!/bin/sh
# doc 9ah milestone 3: online_sim timed mode (core, two-stage, composed DiskANN charges) vs the v5 harness
# runs of the same cells. After run_core_parity.sh.
cd /local/home/jialind/ann-exact-caching || exit 1
R=simulations/approximate_hits/results; O=/tmp/mx/tcheck
while [ ! -f /tmp/mx/CORE_PARITY.DONE ]; do sleep 60; done
mkdir -p $O
efs() { python3 -c "
import json; d=json.load(open('$R/timing_v4/$1/stage1_b$2.json'))
print(min(t['ef'] for t in d['stage1_trials'] if t['precision']=='fp32' and t['ok']), d['stage2']['ef'])"; }
run() {   # prep budget method tag extra...
  P=$1; B=$2; M=$3; TAG=$4; shift 4
  set -- $(efs $P $B) "$@"; E1=$1; E2=$2; shift 2
  PYTHONPATH=. .venv-core/bin/python -u simulations/approximate_hits/online_sim.py --prep $R/online/$P --method $M --stream trace \
    --budget $B --seed 0 --shell-scope balls --shell-balls 10 --backend-file $R/backends/$P/diskann.npz --backend-target 0.95 \
    --cache-impl core --core-threads 4 --core-ef1 $E1 --core-ef2 $E2 --timing-compose $R/timing_v5/$P/backend_s50_lat.npy \
    "$@" --out $O/${P}_${TAG}_b$B > $O/${P}_${TAG}_b$B.txt 2>&1 &
}
run spacev-aker-0.6 0.01 static ours --detect
run spacev-aker-0.6 0.01 lru lru --detect --k-lru 20
run tripclick-ip 0.1 lru lru --detect --k-lru 20
run tripclick-ip 0.1 ogreedy oursonline --detect --og-evict utility
run tripclick-ip 0.1 greedy oursdrift --detect score --response rebuild
wait
touch /tmp/mx/CORE_TIMING_CHECK.DONE
