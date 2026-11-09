#!/bin/sh
# Coverage-matrix runs (context doc 9). Three pools: heavy simulator runs (ours / LRU at a
# 10% SPACEV budget, all TriviaQA runs; ~17 GB each), light simulator runs, real Aker.
# Phases: 1 now; 2 after the SPACEV Zipf 0.99 / 0.3 data prep; 3a after the TriviaQA
# online prep; 3b after the TriviaQA data prep; 4 fresh-seed re-runs of the best baselines.
cd /local/home/jialind/ann-exact-caching || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:/home/jialind/third_party/faiss-install/lib64:$LD_LIBRARY_PATH PYTHONPATH=.
M=/tmp/mx
S=simulations/approximate_hits
P=$S/results/online
PY=.venv/bin/python
TQA="triviaqa-aker-0.3 triviaqa-aker-0.6 triviaqa-aker-0.99"

waitfor() { for f in "$@"; do while [ ! -f "$f" ]; do sleep 120; done; done; }

# split a job list into heavy (H) / light (L): heavy = ours / LRU on TriviaQA, or at a 10% SPACEV budget
HEAVY="(triviaqa[^ ]* --stream [a-z]+ --budget [0-9.]+ .*--method (static|greedy|lru))|(spacev[^ ]* --stream [a-z]+ --budget 0\.1 .*--method (static|greedy|lru))"
split() {   # $1 = list, $2 = phase name
  grep -E "$HEAVY" "$1" > $M/H_$2.txt
  grep -vE "$HEAVY" "$1" > $M/L_$2.txt
}

run() {     # $1 = pool, $2 = phase, $3 = parallel
  [ -s $M/$1_$2.txt ] && xargs -P $3 -I{} sh -c '{}' < $M/$1_$2.txt
  touch $M/$1_$2.DONE
}

# ---- phase 1 lists
{
  $PY $S/online_backend_jobs.py --preps esci1m spacev-aker-0.6 spacev-real
  $PY $S/online_round2_jobs.py --budget 0.1 --preps spacev-aker-0.6 spacev-aker-0.3 spacev-aker-0.99 spacev-real
  $PY $S/online_round2_jobs.py --multishift --budget 0.1 --preps spacev-aker-0.6 spacev-aker-0.3 spacev-aker-0.99
  $PY $S/online_data_jobs.py --round 3 --preps spacev-aker-0.6 spacev-real --budget 0.1
  $PY $S/online_round2_jobs.py --budget 0.01 --preps esci1m
  $PY $S/online_round2_jobs.py --multishift --budget 0.01 --preps esci1m
  $PY $S/online_data_jobs.py --round 3 --preps esci1m --budget 0.01
} > $M/all_1.txt
split $M/all_1.txt 1
$PY $S/aker_real/aker_real_jobs.py --preps spacev-aker-0.3 spacev-aker-0.6 spacev-aker-0.99 spacev-real > $M/A_1.txt

pool_sim() {   # $1 = H or L, $2 = parallel
  run $1 1 $2
  waitfor $P/spacev-aker-0.99/DATA_PREP_DONE $P/spacev-aker-0.3/DATA_PREP_DONE
  waitfor $M/lists_2.ready
  run $1 2 $2
  waitfor $M/lists_3a.ready
  run $1 3a $2
  waitfor $M/lists_3b.ready
  run $1 3b $2
  waitfor $M/lists_4.ready
  run $1 4 $2
}

pool_aker() {
  run A 1 10
  waitfor $M/lists_2.ready
  run A 2 10
  waitfor $M/lists_3a.ready
  run A 3a 6
  waitfor $M/lists_3b.ready
  run A 3b 6
}

lists() {
  waitfor $P/spacev-aker-0.99/DATA_PREP_DONE $P/spacev-aker-0.3/DATA_PREP_DONE
  {
    $PY $S/online_data_jobs.py --round 3 --preps spacev-aker-0.99 spacev-aker-0.3 --budget 0.01
    $PY $S/online_data_jobs.py --round 3 --preps spacev-aker-0.99 spacev-aker-0.3 --budget 0.1
    $PY $S/online_backend_jobs.py --preps spacev-aker-0.99 spacev-aker-0.3
  } > $M/all_2.txt
  split $M/all_2.txt 2
  $PY $S/aker_real/aker_real_jobs.py --preps spacev-aker-0.99 spacev-aker-0.3 --only data backend > $M/A_2.txt
  touch $M/lists_2.ready
  waitfor $P/triviaqa-aker-0.3/PREP_DONE $P/triviaqa-aker-0.6/PREP_DONE $P/triviaqa-aker-0.99/PREP_DONE
  {
    $PY $S/online_round2_jobs.py --preps $TQA
    $PY $S/online_round2_jobs.py --multishift --preps $TQA
    $PY $S/online_round2_jobs.py --budget 0.1 --preps $TQA
    $PY $S/online_round2_jobs.py --multishift --budget 0.1 --preps $TQA
  } > $M/all_3a.txt
  split $M/all_3a.txt 3a
  $PY $S/aker_real/aker_real_jobs.py --preps $TQA --only stable shift > $M/A_3a.txt
  touch $M/lists_3a.ready
  waitfor $P/triviaqa-aker-0.3/DATA_PREP_DONE $P/triviaqa-aker-0.6/DATA_PREP_DONE $P/triviaqa-aker-0.99/DATA_PREP_DONE
  {
    $PY $S/online_data_jobs.py --round 3 --preps $TQA --budget 0.01
    $PY $S/online_data_jobs.py --round 3 --preps $TQA --budget 0.1
    $PY $S/online_backend_jobs.py --preps $TQA
  } > $M/all_3b.txt
  split $M/all_3b.txt 3b
  $PY $S/aker_real/aker_real_jobs.py --preps $TQA --only data backend > $M/A_3b.txt
  touch $M/lists_3b.ready
  # phase 4: once every simulator phase is through, re-run the best seed-0 baseline
  # settings of the new workload-shift / stable grids on seeds 1 and 2
  waitfor $M/H_3b.DONE $M/L_3b.DONE
  {
    $PY $S/online_round2_jobs.py --best-baselines 1 2 --budget 0.1 --preps spacev-aker-0.6 spacev-aker-0.3 spacev-aker-0.99 spacev-real $TQA
    $PY $S/online_round2_jobs.py --best-baselines 1 2 --budget 0.01 --preps esci1m
    $PY $S/online_round2_jobs.py --best-baselines 1 2 --preps $TQA
  } > $M/all_4.txt
  split $M/all_4.txt 4
  touch $M/lists_4.ready
}

lists &
pool_sim H 6 &
pool_sim L 24 &
pool_aker &
wait
touch $M/ALL_DONE
