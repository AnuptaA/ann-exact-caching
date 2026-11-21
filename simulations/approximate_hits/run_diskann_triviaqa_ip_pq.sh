#!/bin/sh
# TriviaQA ip DiskANN with 256-byte PQ (the 32-byte PQ index reached recall@10 .11 at search list
# 1000: 769 dims at 24 dims / byte; SPACEV is 100 dims at ~3 / byte). Build, then tune per trace
# to 0.95 / 0.9 with an extended effort grid.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
S=simulations/approximate_hits
B=$S/results/backends
P=$S/results/online
PY=.venv-ann/bin/python
idx=sphereip10m-pq256
# a build already running (relaunch): wait for it
while pgrep -f "eval_real_backend.py .*--out-name $idx" > /dev/null; do sleep 120; done
if [ ! -f $B/$idx/diskann/ann_disk.index ]; then
  OMP_NUM_THREADS=32 $PY -u $S/eval_real_backend.py --prep $P/triviaqa-ip-0.99 --index diskann --build-only \
    --out-name $idx --threads 32 --build-gb 96 --diskann-pq-bytes 256 > $B/log_build_$idx.txt 2>&1
fi
[ -f $B/$idx/diskann/ann_disk.index ] || { echo "$(date +%T) DiskANN build failed: $idx" >> ${RECAL_QUEUE_DIR:-/tmp/mx}/failures.log; exit 1; }
# tuning (24 threads, hours) after the timing batch, which needs a quiet machine
while [ ! -f ${RECAL_QUEUE_DIR:-/tmp/mx}/TIMING3B.DONE ]; do sleep 300; done
for z in 0.99 0.6 0.3; do
  mkdir -p $B/triviaqa-ip-$z
  [ -e $B/triviaqa-ip-$z/diskann ] || ln -s ../$idx/diskann $B/triviaqa-ip-$z/diskann
  [ -f $B/triviaqa-ip-$z/diskann.npz ] || OMP_NUM_THREADS=24 $PY -u $S/eval_real_backend.py --prep $P/triviaqa-ip-$z \
    --index diskann --targets 0.95 0.9 --threads 24 --diskann-pq-bytes 256 \
    --efforts 10 16 24 32 48 64 96 128 160 200 256 320 400 512 640 800 1000 1250 1600 2000 2500 3200 4000 \
    > $B/triviaqa-ip-$z/log_diskann.txt 2>&1
done
touch ${RECAL_QUEUE_DIR:-/tmp/mx}/DISKANN_TQA_IP_PQ.DONE
