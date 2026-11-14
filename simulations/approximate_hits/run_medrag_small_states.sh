#!/bin/sh
# MedRAG-Zipf timing at its small cache sizes (doc 9y): warm-up states for 0.01 / 0.03 / 0.1% after
# the timing chain's own states (TSTATES_medrag-ip.DONE); the "new" timing batch waits on
# /tmp/mx/TIMING_GATE_new meanwhile. Opens the gate and stops if the chain is gone without states.
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. MX_PRIORITY=plot
M=/tmp/mx
S=simulations/approximate_hits
R=$S/results
P=medrag-ip
B=$R/backends/$P
O=$R/timing_v4/$P
touch $M/TIMING_GATE_new
while [ ! -f $M/TSTATES_$P.DONE ]; do
  pgrep -f "run_timing_new_workload.sh $P" > /dev/null || { echo "$(date +%T) chain gone"; rm -f $M/TIMING_GATE_new; exit 1; }
  sleep 300
done
tgt=$(.venv/bin/python -c "import json; print(f\"{json.load(open('$B/choice.json'))['target']:g}\")")
MQ=$(.venv/bin/python -c "import numpy as np; n = len(np.load('$R/online/$P/streams.npz')['trace']); print(min(20000, n // 5) + 1000)")
: > $M/tstates_${P}_small.txt
for b in 0.0001 0.0003 0.001; do
  echo "mkdir -p $O/logs && [ -f $O/state_b$b.json ] || .venv/bin/python -u $S/online_sim.py --prep $R/online/$P --method greedy --detect --stream trace --budget $b --max-queries $MQ --shell-scope balls --shell-balls 10 --backend-file $B/diskann.npz --backend-target $tgt --save-state $O/state_b$b.npz --tag _state_b$b --out $O > $O/logs/state_b$b.txt 2>&1 && ln -sf greedy_state_b$b.json $O/state_b$b.json" >> $M/tstates_${P}_small.txt
done
sh $S/mx_complete.sh $M/tstates_${P}_small.txt $M/TSTATES_${P}_SMALL.DONE 3
rm -f $M/TIMING_GATE_new
echo "$(date +%T) small states done"
