#!/bin/sh
# Coverage-matrix runs (context doc 9, 9a), version 3: every job goes through
# mx_run_job.sh, which starts it only when enough memory is free, retries jobs
# killed without output (out-of-memory kills leave no traceback), and never
# duplicates a job that is still running (orphans of an earlier driver).
# Pools (max parallel; admission by free memory decides how many actually run):
#   H  ours / LRU at a 10% SPACEV budget (20-60 GB each)      8 slots, needs 70 GB free
#   L  everything else in the simulator: 1% / ESCI jobs 16 slots (20 GB free), jobs at
#      a 10% budget 16 slots (45 GB free; QVCache / Aker at 10%: 12-18 GB each)
#   A  real Aker                                               8 slots, needs 10 GB free
# Phase 1-2 lists (/tmp/mx/{H,L,A}_{1,2}.txt, D_jobs.txt, AD_jobs.txt) were written by
# the earlier driver; TriviaQA waits for its re-embedded queries (separate launcher).
# At the end: fresh-seed re-runs of the best seed-0 baseline settings (new grids).
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=${RECAL_THIRD_PARTY:-$HOME/third_party}/aker/build/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/boost-install/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/faiss-install/lib64:$LD_LIBRARY_PATH PYTHONPATH=.
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
W=$S/mx_run_job.sh
PY=.venv/bin/python
SPACEV="spacev-aker-0.6 spacev-aker-0.3 spacev-aker-0.99 spacev-real"

$PY $S/online_backend_jobs.py --only-index diskann --preps $SPACEV > $M/D_jobs.txt
$PY $S/aker_real/aker_real_jobs.py --preps $SPACEV --only backend --only-diskann > $M/AD_jobs.txt
cat $M/H_1.txt $M/H_2.txt > $M/H_3.txt
cat $M/L_1.txt $M/L_2.txt $M/D_jobs.txt > $M/L_3.txt
cat $M/A_1.txt $M/A_2.txt $M/AD_jobs.txt > $M/A_3.txt

# light jobs at a 10% budget need more headroom: split L by budget
grep -E -- "--budget 0\.1 " $M/L_3.txt > $M/L10_3.txt
grep -vE -- "--budget 0\.1 " $M/L_3.txt > $M/L1_3.txt

pool() {   # $1 = list, $2 = parallel, $3 = min free GB
  xargs -P $2 -I{} sh $W $3 {} < $1
}
( pool $M/H_3.txt 8 70; touch $M/H_3.DONE ) &
( pool $M/L1_3.txt 16 20; touch $M/L1_3.DONE ) &
( pool $M/L10_3.txt 16 45; touch $M/L10_3.DONE ) &
( pool $M/A_3.txt 8 10; touch $M/A_3.DONE ) &
wait
{
  $PY $S/online_round2_jobs.py --best-baselines 1 2 --budget 0.1 --preps $SPACEV
  $PY $S/online_round2_jobs.py --best-baselines 1 2 --budget 0.01 --preps esci1m
} > $M/all_4.txt
pool $M/all_4.txt 16 45
touch $M/ALL_3.DONE
