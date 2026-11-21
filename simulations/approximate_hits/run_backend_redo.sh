#!/bin/sh
# Redo of the timing backend (doc 9u): DiskANN tuned per workload over search list x beam width.
#  1. quiet probe of every candidate index (after the Aker-style builds), 2. selection
#     (select_backend.py -> backends/<prep>/choice.json), 3. stored answers at exactly the chosen
#     setting (eval_real_backend.py --fixed, plot priority, old files archived as diskann_v1.*),
#  4. warm-up states for timing_v4, 5. the final timing batch (run_timing_batch2.sh final).
# Every step waits on its predecessor's outputs; jobs go through the runner (mx_complete.sh).
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
R=$S/results
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MX_PRIORITY=plot
log() { echo "$(date +%T) $*"; }

while [ ! -f $M/INDEX_AKER.DONE ]; do sleep 300; done
log "Aker-style builds done; quiet probe"
[ -f $M/TIMING_PROBE_V2.DONE ] || sh $S/run_timing_batch2.sh probe TIMING_PROBE_V2 INDEX_AKER.DONE > $M/timing_probe_v2.log 2>&1

# recall over each workload's own requests, per candidate index (doc 9v; latency stays per index)
.venv/bin/python - > $M/workload_recall.txt <<'PY'
import os
from simulations.approximate_hits.select_backend import CANDIDATES
R = "simulations/approximate_hits/results"
for prep, probes in CANDIDATES.items():
    for p in probes:
        if os.path.exists(f"{R}/latency_probe/{p}.json"):
            out = f"{R}/backends/{prep}/recall_{p}.json"
            print(f"mkdir -p {R}/backends/{prep}/logs && [ -f {out} ] || OMP_NUM_THREADS=24 .venv-ann/bin/python -u "
                  f"simulations/approximate_hits/workload_recall.py --prep {R}/online/{prep} --probe {p} "
                  f"> {R}/backends/{prep}/logs/recall_{p}.txt 2>&1")
PY
log "$(grep -c . $M/workload_recall.txt) workload-recall jobs"
sh $S/mx_complete.sh $M/workload_recall.txt $M/WORKLOAD_RECALL.DONE 3

log "selection"
.venv/bin/python $S/select_backend.py > $M/select_backend.log 2>&1
cat $M/select_backend.log

# stored answers at the chosen settings
: > $M/answers_v2.txt
for d in $R/backends/*/choice.json; do
  [ -f "$d" ] || continue
  prep=$(basename $(dirname $d))
  B=$R/backends/$prep
  args=$(.venv/bin/python -c "import json; c = json.load(open('$d')); print(f\"--index-dir {c['index_dir']} --beam {c['W']} --fixed {c['target']:g}:{c['L']}\")")
  if [ -f $B/diskann.json ] && ! grep -q '"beam"' $B/diskann.json; then
    for f in $B/diskann.json $B/diskann.npz; do [ -f $f ] && mv $f ${f%.*}_v1.${f##*.}; done
  fi
  mkdir -p $B/logs
  echo "mkdir -p $B/logs && [ -f $B/diskann.json ] || OMP_NUM_THREADS=24 .venv-ann/bin/python -u $S/eval_real_backend.py --prep $R/online/$prep --index diskann --out-name $prep --threads 24 $args > $B/logs/answers_v2.txt 2>&1" >> $M/answers_v2.txt
done
log "$(grep -c . $M/answers_v2.txt) answer jobs"
sh $S/mx_complete.sh $M/answers_v2.txt $M/ANSWERS_V2.DONE 3

# warm-up states (timing_v4) with the new answers
: > $M/tstates_v4.txt
for d in $R/backends/*/choice.json; do
  prep=$(basename $(dirname $d))
  tgt=$(.venv/bin/python -c "import json; print(f\"{json.load(open('$d'))['target']:g}\")")
  case $prep in
    spacev-aker-*|esci1m) budgets="0.003 0.01 0.03 0.1" ;;
    *) budgets="0.01 0.1" ;;
  esac
  stream=trace; [ $prep = esci1m ] && stream=random
  O=$R/timing_v4/$prep
  for b in $budgets; do
    echo "mkdir -p $O/logs && [ -f $O/state_b$b.json ] || .venv/bin/python -u $S/online_sim.py --prep $R/online/$prep --method greedy --detect --stream $stream --budget $b --max-queries 21000 --shell-scope balls --shell-balls 10 --backend-file $R/backends/$prep/diskann.npz --backend-target $tgt --save-state $O/state_b$b.npz --tag _state_b$b --out $O > $O/logs/state_b$b.txt 2>&1 && ln -sf greedy_state_b$b.json $O/state_b$b.json" >> $M/tstates_v4.txt
  done
done
log "$(grep -c . $M/tstates_v4.txt) state jobs"
sh $S/mx_complete.sh $M/tstates_v4.txt $M/TSTATES_V4.DONE 8

log "final timing"
sh $S/run_timing_batch2.sh final TIMING4 TSTATES_V4.DONE REQ.DONE HE.DONE > $M/timing4.log 2>&1
log "done"
