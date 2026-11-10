#!/bin/sh
# TriviaQA part of the coverage matrix (context doc 9a), on the re-embedded queries
# (workloads triviaqa-dpr-<z>: Aker's request sequences, DPR_web question encoder,
# SPHERE 10M base; Euclidean distance as in Aker). Waits for
# datasets/reembed_triviaqa_queries.py, then per trace, one memory-heavy prep step at a
# time (~65 GB each): streams + exact top-500 + multishift, then the data-change
# rankings (top-1000 union). Runs are queued as their inputs land, all through the
# memory-guarded mx_run_job.sh; ours / LRU use the matrix configurations (KEEP).
# Each prep step waits for any running TriviaQA prep, for 120 GB free, and holds new job
# starts (/tmp/mx/HOLD) for its first 15 min while it allocates ~65 GB.
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:/home/jialind/third_party/faiss-install/lib64:$LD_LIBRARY_PATH PYTHONPATH=.
M=/tmp/mx
S=simulations/approximate_hits
P=$S/results/online
W=$S/mx_run_job.sh
PY=.venv/bin/python
KEEP="/(static_default|greedy_none|greedy_score_rebuild|greedy_all_rebuild_oraclecal|lru_k10_none|lru_k20_none|lru_k20_all|static_static|greedy_slide_none|greedy_slide_score|greedy_noupkeep_both|greedy_oracle|greedy_known_reset|lru_k20_both)\.json"
OURS="--method (static|greedy|lru)"

prep_step() {   # run one memory-heavy prep command (args) with headroom and a HOLD during ramp-up
  while pgrep -f "online_prepare(_data)?.py .*triviaqa-dpr" > /dev/null; do sleep 60; done
  while [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -lt 120 ]; do sleep 60; done
  touch $M/HOLD
  ( sleep 900; rm -f $M/HOLD ) &
  "$@"
  rm -f $M/HOLD
}

queue() {   # $1 = job list file: ours / LRU (filtered to KEEP) and baselines, real Aker separately
  grep -E -- "$OURS" $1 | grep -E "$KEEP" > $1.H
  grep -vE -- "$OURS" $1 > $1.L
  ( xargs -P 4 -I{} sh $W 80 {} < $1.H ) &
  ( xargs -P 8 -I{} sh $W 50 {} < $1.L ) &
}

while [ ! -f datasets/aker/triviaqa-dpr-sim-100k-0.99.npy ] || [ ! -f datasets/aker/triviaqa-dpr-sim-100k-0.3.npy ] \
      || [ ! -f datasets/aker/triviaqa-dpr-sim-100k-0.6.npy ]; do sleep 120; done
sleep 60

for z in 0.6 0.99 0.3; do
  O=$P/triviaqa-dpr-$z
  [ -f $O/PREP_DONE ] && continue
  while pgrep -f "online_prepare(_data)?.py .*triviaqa-dpr" > /dev/null; do sleep 60; done   # orphan prep
  export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
  [ -f $O/top500.npy ] || prep_step sh -c "$PY -u $S/online_prepare.py --dataset trace --trace triviaqa-dpr-$z --out $O >> $P/prep_triviaqa-dpr-$z.log 2>&1"
  $PY -u $S/online_prepare.py --dataset trace --trace triviaqa-dpr-$z --out $O --add-multishift >> $P/prep_triviaqa-dpr-$z.log 2>&1
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  [ -f $O/top500.npy ] || { echo "$(date +%T) prep failed for $z" >> $M/failures.log; continue; }
  touch $O/PREP_DONE
  {
    $PY $S/online_round2_jobs.py --preps triviaqa-dpr-$z
    $PY $S/online_round2_jobs.py --multishift --preps triviaqa-dpr-$z
    $PY $S/online_round2_jobs.py --budget 0.1 --preps triviaqa-dpr-$z
    $PY $S/online_round2_jobs.py --multishift --budget 0.1 --preps triviaqa-dpr-$z
  } > $M/T_a_$z.txt
  queue $M/T_a_$z.txt
  $PY $S/aker_real/aker_real_jobs.py --preps triviaqa-dpr-$z --only stable shift > $M/TA_a_$z.txt
  ( xargs -P 4 -I{} sh $W 20 {} < $M/TA_a_$z.txt ) &
done
for z in 0.99 0.6 0.3; do
  O=$P/triviaqa-dpr-$z
  [ -f $O/PREP_DONE ] || continue
  export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
  for kind in churn drift; do
    [ -f $O/data_$kind.npz ] || prep_step sh -c "$PY -u $S/online_prepare_data.py --prep $O --kind $kind >> $O/prep_data_$kind.log 2>&1"
  done
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  [ -f $O/data_drift.npz ] || { echo "$(date +%T) data prep failed for $z" >> $M/failures.log; continue; }
  touch $O/DATA_PREP_DONE
  {
    $PY $S/online_data_jobs.py --round 3 --preps triviaqa-dpr-$z --budget 0.01
    $PY $S/online_data_jobs.py --round 3 --preps triviaqa-dpr-$z --budget 0.1
    $PY $S/online_backend_jobs.py --preps triviaqa-dpr-$z
  } > $M/T_b_$z.txt
  queue $M/T_b_$z.txt
  $PY $S/aker_real/aker_real_jobs.py --preps triviaqa-dpr-$z --only data backend > $M/TA_b_$z.txt
  ( xargs -P 4 -I{} sh $W 20 {} < $M/TA_b_$z.txt ) &
done
wait
{
  $PY $S/online_round2_jobs.py --best-baselines 1 2 --preps triviaqa-dpr-0.3 triviaqa-dpr-0.6 triviaqa-dpr-0.99
  $PY $S/online_round2_jobs.py --best-baselines 1 2 --budget 0.1 --preps triviaqa-dpr-0.3 triviaqa-dpr-0.6 triviaqa-dpr-0.99
} > $M/T_4.txt
xargs -P 8 -I{} sh $W 50 {} < $M/T_4.txt
touch $M/TQA.DONE
