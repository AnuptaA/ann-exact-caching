#!/bin/sh
# Timing for a NEW workload (11-13: tripclick-ip, medrag-ip), the steps of run_backend_redo.sh for
# one prep: DiskANN build (degree 64, build list 100, 256-byte PQ) -> quiet latency probe (search
# list x beam width; after the main timing batch TIMING4, so it never competes with it) -> recall
# over the workload's requests -> select_backend.py (recall 0.95, else 0.9 if 0.95 needs > 100 ms)
# -> stored answers at the chosen setting -> warm-up states -> timing (timing_jobs.py --set new,
# after the workload's headline runs, which pick the Aker / QVCache settings).
#   run_timing_new_workload.sh <prep>
P=$1
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
R=$S/results
B=$R/backends/$P
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MX_PRIORITY=plot
W=$S/mx_run_job.sh
log() { echo "$(date +%T) $*"; }
case $P in tripclick-ip) PROBE=tripclick ;; medrag-ip) PROBE=medrag ;; *) echo "unknown $P"; exit 1 ;; esac

while [ ! -f $R/online/$P/top500.npy ]; do sleep 300; done
mkdir -p $B/logs
log "$P: DiskANN build"
sh $W 0 "[ -f $B/diskann/ann_disk.index ] || OMP_NUM_THREADS=24 .venv-ann/bin/python -u $S/eval_real_backend.py --prep $R/online/$P --index diskann --out-name $P --diskann-pq-bytes 256 --build-only --threads 24 --build-gb 96 > $B/logs/build.txt 2>&1"
[ -f $B/diskann/ann_disk.index ] || { log "build failed"; exit 1; }

log "$P: waiting for the main timing batch, then the quiet probe"
while [ ! -f $M/TIMING4.DONE ]; do sleep 300; done
sh $S/run_timing_batch2.sh probe TPROBE_$P > $M/tprobe_$P.log 2>&1
[ -f $R/latency_probe/$PROBE.json ] || { log "probe failed"; exit 1; }

sh $W 0 "[ -f $B/recall_$PROBE.json ] || OMP_NUM_THREADS=24 .venv-ann/bin/python -u $S/workload_recall.py --prep $R/online/$P --probe $PROBE > $B/logs/recall_$PROBE.txt 2>&1"
.venv/bin/python $S/select_backend.py > $M/select_backend_$P.log 2>&1
grep -A3 "^$P" $M/select_backend_$P.log
[ -f $B/choice.json ] || { log "no backend choice"; exit 1; }
args=$(.venv/bin/python -c "import json; c = json.load(open('$B/choice.json')); print(f\"--index-dir {c['index_dir']} --beam {c['W']} --fixed {c['target']:g}:{c['L']}\")")
tgt=$(.venv/bin/python -c "import json; print(f\"{json.load(open('$B/choice.json'))['target']:g}\")")
log "$P: answers ($args)"
sh $W 0 "[ -f $B/diskann.json ] || OMP_NUM_THREADS=24 .venv-ann/bin/python -u $S/eval_real_backend.py --prep $R/online/$P --index diskann --out-name $P --threads 24 --diskann-pq-bytes 256 $args > $B/logs/answers.txt 2>&1"
[ -f $B/diskann.npz ] || { log "answers failed"; exit 1; }

O=$R/timing_v4/$P
# warm-up as the simulators' default: min(20000, a fifth of the stream); the state run stops 1000 later
MQ=$(.venv/bin/python -c "import numpy as np; n = len(np.load('$R/online/$P/streams.npz')['trace']); print(min(20000, n // 5) + 1000)")
: > $M/tstates_$P.txt
for b in 0.003 0.01 0.03 0.1; do
  echo "mkdir -p $O/logs && [ -f $O/state_b$b.json ] || .venv/bin/python -u $S/online_sim.py --prep $R/online/$P --method greedy --detect --stream trace --budget $b --max-queries $MQ --shell-scope balls --shell-balls 10 --backend-file $B/diskann.npz --backend-target $tgt --save-state $O/state_b$b.npz --tag _state_b$b --out $O > $O/logs/state_b$b.txt 2>&1 && ln -sf greedy_state_b$b.json $O/state_b$b.json" >> $M/tstates_$P.txt
done
sh $S/mx_complete.sh $M/tstates_$P.txt $M/TSTATES_$P.DONE 4

while [ ! -f $M/HEADLINE_$P.DONE ]; do sleep 300; done
log "$P: timing"
sh $S/run_timing_batch2.sh new TIMING_NEW_$P > $M/timing_new_$P.log 2>&1
.venv/bin/python $S/plot_timing.py > $M/plot_timing_$P.log 2>&1
log "$P: done"
