#!/bin/sh
# DiskANN for TriviaQA (both variants), target recall@10 0.95 (decided with Jialin, 11-10).
# One index per database, shared by the three Zipf traces of a variant:
#   dpr: SPHERE 10M x 768 (sphere-10m)     -> results/backends/sphere10m/diskann
#   ip : SPHERE 10M x 769 (sphere-ip-10m)  -> results/backends/sphereip10m/diskann
# The build is memory-heavy, so it takes the same prep lock (/tmp/mx/prep.lock) and HOLD as the
# exact-neighbor preps of run_matrix_triviaqa.sh (one heavy step at a time). Then, per trace, the
# search effort is tuned to recall@10 0.95 (and 0.9) and the answers written
# (results/backends/<prep>/diskann.npz) for the inexact-backend runs and for timing.
# Not timing: index builds and answer files only.
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
B=$S/results/backends
P=$S/results/online
PY=.venv-ann/bin/python

lock() {
  while :; do
    if ! pgrep -f "online_prepare(_data)?.py .*triviaqa-(dpr|ip)" > /dev/null \
       && [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 140 ] \
       && mkdir $M/prep.lock 2>/dev/null; then return; fi
    sleep 120
  done
}
unlock() { rm -f $M/HOLD; rmdir $M/prep.lock 2>/dev/null; }

for v in dpr ip; do
  case $v in dpr) idx=sphere10m;; ip) idx=sphereip10m;; esac
  first=$P/triviaqa-$v-0.99
  while [ ! -f $first/PREP_DONE ]; do sleep 300; done          # needs the prep's q / top500
  if [ ! -f $B/$idx/diskann/ann_disk.index ]; then
    lock
    touch $M/HOLD; ( sleep 1800; rm -f $M/HOLD ) &
    OMP_NUM_THREADS=32 $PY -u $S/eval_real_backend.py --prep $first --index diskann --build-only \
      --out-name $idx --threads 32 --build-gb 96 > $B/log_build_$idx.txt 2>&1
    unlock
  fi
  [ -f $B/$idx/diskann/ann_disk.index ] || { echo "$(date +%T) DiskANN build failed: $idx" >> $M/failures.log; continue; }
  for z in 0.99 0.6 0.3; do
    pr=$P/triviaqa-$v-$z
    while [ ! -f $pr/PREP_DONE ]; do sleep 300; done
    mkdir -p $B/triviaqa-$v-$z
    [ -e $B/triviaqa-$v-$z/diskann ] || ln -s ../$idx/diskann $B/triviaqa-$v-$z/diskann
    [ -f $B/triviaqa-$v-$z/diskann.npz ] || OMP_NUM_THREADS=24 $PY -u $S/eval_real_backend.py --prep $pr \
      --index diskann --targets 0.95 0.9 --threads 24 > $B/triviaqa-$v-$z/log_diskann.txt 2>&1
  done
done
touch $M/DISKANN_TQA.DONE
