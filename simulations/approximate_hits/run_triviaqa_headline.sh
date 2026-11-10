#!/bin/sh
# TriviaQA, headline plot only (decided 11-10: everything unrelated to the headline and timing plots
# is un-queued). Usage: run_triviaqa_headline.sh ip|dpr
# Per trace: exact top-500 prep (one memory-heavy step at a time, shared prep lock), then the
# stable-trace runs at 1% and 10% (ours / LRU: the matrix configurations; Aker / QVCache: full grids)
# and real Aker on the trace. No multishift streams, no data-change rankings.
PFX=${1:-ip}
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export LD_LIBRARY_PATH=/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:/home/jialind/third_party/faiss-install/lib64:$LD_LIBRARY_PATH
M=/tmp/mx
S=simulations/approximate_hits
P=$S/results/online
W=$S/mx_run_job.sh
PY=.venv/bin/python
KEEP="/(static_default|greedy_none|greedy_score_rebuild|greedy_all_rebuild_oraclecal|lru_k10_none|lru_k20_none|lru_k20_all)\.json"

# an orphaned prep of an earlier launcher holds the lock: wait for it, then release the lock
while pgrep -f "online_prepare(_data)?.py .*triviaqa-(dpr|ip)" > /dev/null; do sleep 60; done
rm -f $M/HOLD; rmdir $M/prep.lock 2>/dev/null

for z in 0.6 0.99 0.3; do
  O=$P/triviaqa-$PFX-$z
  if [ ! -f $O/top500.npy ]; then
    while :; do
      if ! pgrep -f "online_prepare(_data)?.py .*triviaqa-(dpr|ip)" > /dev/null \
         && [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 120 ] \
         && mkdir $M/prep.lock 2>/dev/null; then break; fi
      sleep 90
    done
    if [ ! -f $O/top500.npy ]; then
      touch $M/HOLD; ( sleep 900; rm -f $M/HOLD ) &
      OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 $PY -u $S/online_prepare.py \
        --dataset trace --trace triviaqa-$PFX-$z --out $O >> $P/prep_triviaqa-$PFX-$z.log 2>&1
      rm -f $M/HOLD
    fi
    rmdir $M/prep.lock 2>/dev/null
  fi
  [ -f $O/top500.npy ] || { echo "$(date +%T) prep failed: triviaqa-$PFX-$z" >> $M/failures.log; continue; }
  touch $O/PREP_DONE
  {
    $PY $S/online_round2_jobs.py --preps triviaqa-$PFX-$z
    $PY $S/online_round2_jobs.py --budget 0.1 --preps triviaqa-$PFX-$z
  } | grep -E -- "--stream trace" | awk -v keep="$KEEP" '!/--method (static|greedy|lru)/ || $0 ~ keep' > $M/H_tqa_${PFX}_$z.txt
  $PY $S/aker_real/aker_real_jobs.py --preps triviaqa-$PFX-$z --only stable > $M/HA_tqa_${PFX}_$z.txt
  ( xargs -P 20 -I{} sh $W 0 {} < $M/H_tqa_${PFX}_$z.txt ) &
  ( xargs -P 6 -I{} sh $W 0 {} < $M/HA_tqa_${PFX}_$z.txt ) &
done
wait
touch $M/HEADLINE_TQA_$PFX.DONE
