#!/bin/sh
# TriviaQA part of the coverage matrix (context doc 9a, 9b, 9c).
# Usage: run_matrix_triviaqa.sh [dpr|ip]
#   dpr  triviaqa-dpr-<z>: Aker's request sequences re-embedded with SPHERE's DPR_web question
#        encoder, SPHERE 10M base, Euclidean distance (as Aker)
#   ip   triviaqa-ip-<z>: the same after the inner-product -> Euclidean transform
#        (datasets/make_sphere_ip_transform.py; 769-d; Euclidean neighbors = inner-product neighbors)
# Waits for the query files, then per trace: streams + exact top-500 + multishift, then the
# data-change rankings (top-1000 union). Runs are queued as inputs land, all through the
# memory-guarded mx_run_job.sh; ours / LRU use the matrix configurations (KEEP).
# Memory: an exact-neighbor prep holds 65-80 GB, so all prep steps of BOTH launchers share
# one lock (/tmp/mx/prep.lock; a step also needs 120 GB free and holds new job starts,
# /tmp/mx/HOLD, for its first 15 min). The "already done?" check is made after taking the
# lock, so a prep that finished while waiting is not redone.
PFX=${1:-dpr}
cd "${RECAL_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=${RECAL_THIRD_PARTY:-$HOME/third_party}/aker/build/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/boost-install/lib:${RECAL_THIRD_PARTY:-$HOME/third_party}/faiss-install/lib64:$LD_LIBRARY_PATH PYTHONPATH=.
M=${RECAL_QUEUE_DIR:-/tmp/mx}
S=simulations/approximate_hits
P=$S/results/online
W=$S/mx_run_job.sh
PY=.venv/bin/python
KEEP="/(static_default|greedy_none|greedy_score_rebuild|greedy_all_rebuild_oraclecal|lru_k10_none|lru_k20_none|lru_k20_all|static_static|greedy_slide_none|greedy_slide_score|greedy_noupkeep_both|greedy_oracle|greedy_known_reset|lru_k20_both)\.json"
OURS="--method (static|greedy|lru)"
DONE=$M/TQA.DONE; [ "$PFX" = dpr ] || DONE=$M/TQA_$PFX.DONE

prep_step() {   # $1 = marker file whose existence means "done"; rest = command (run with 32 threads)
  marker=$1; shift
  [ -f "$marker" ] && return 0
  while :; do
    if ! pgrep -f "online_prepare(_data)?.py .*triviaqa-(dpr|ip)" > /dev/null \
       && [ "$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 120 ] \
       && mkdir $M/prep.lock 2>/dev/null; then break; fi
    sleep $((45 + $(od -An -N1 -tu1 /dev/urandom) % 30))
  done
  if [ -f "$marker" ]; then rmdir $M/prep.lock; return 0; fi
  touch $M/HOLD
  ( sleep 900; rm -f $M/HOLD ) &
  OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32 "$@"
  rm -f $M/HOLD
  rmdir $M/prep.lock
}

queue() {   # $1 = job list: ours / LRU (filtered to KEEP, heavy pool) and baselines (light pool)
  grep -E -- "$OURS" $1 | grep -E "$KEEP" > $1.H
  grep -vE -- "$OURS" $1 > $1.L
  ( xargs -P 3 -I{} sh $W 80 {} < $1.H ) &
  ( xargs -P 6 -I{} sh $W 50 {} < $1.L ) &
}

while [ ! -f datasets/aker/triviaqa-$PFX-sim-100k-0.99.npy ] || [ ! -f datasets/aker/triviaqa-$PFX-sim-100k-0.3.npy ] \
      || [ ! -f datasets/aker/triviaqa-$PFX-sim-100k-0.6.npy ] \
      || { [ "$PFX" = ip ] && [ ! -f datasets/aker/sphere-ip-10m.json ]; }; do sleep 120; done
sleep 60

for z in 0.6 0.99 0.3; do
  O=$P/triviaqa-$PFX-$z
  [ -f $O/PREP_DONE ] && continue
  while pgrep -f "online_prepare.py .*triviaqa-$PFX-$z " > /dev/null; do sleep 60; done   # orphan of an earlier launcher
  prep_step $O/top500.npy sh -c "$PY -u $S/online_prepare.py --dataset trace --trace triviaqa-$PFX-$z --out $O >> $P/prep_triviaqa-$PFX-$z.log 2>&1"
  [ -f $O/top500.npy ] || { echo "$(date +%T) prep failed for $PFX $z" >> $M/failures.log; continue; }
  $PY -u $S/online_prepare.py --dataset trace --trace triviaqa-$PFX-$z --out $O --add-multishift >> $P/prep_triviaqa-$PFX-$z.log 2>&1
  touch $O/PREP_DONE
  {
    $PY $S/online_round2_jobs.py --preps triviaqa-$PFX-$z
    $PY $S/online_round2_jobs.py --multishift --preps triviaqa-$PFX-$z
    $PY $S/online_round2_jobs.py --budget 0.1 --preps triviaqa-$PFX-$z
    $PY $S/online_round2_jobs.py --multishift --budget 0.1 --preps triviaqa-$PFX-$z
  } > $M/T_a_${PFX}_$z.txt
  queue $M/T_a_${PFX}_$z.txt
  $PY $S/aker_real/aker_real_jobs.py --preps triviaqa-$PFX-$z --only stable shift > $M/TA_a_${PFX}_$z.txt
  ( xargs -P 3 -I{} sh $W 20 {} < $M/TA_a_${PFX}_$z.txt ) &
done
for z in 0.99 0.6 0.3; do
  O=$P/triviaqa-$PFX-$z
  [ -f $O/PREP_DONE ] || continue
  [ -f $O/DATA_PREP_DONE ] && continue
  while pgrep -f "online_prepare_data.py .*triviaqa-$PFX-$z" > /dev/null; do sleep 60; done   # orphan
  for kind in churn drift; do
    prep_step $O/data_$kind.npz sh -c "$PY -u $S/online_prepare_data.py --prep $O --kind $kind >> $O/prep_data_$kind.log 2>&1"
  done
  [ -f $O/data_drift.npz ] || { echo "$(date +%T) data prep failed for $PFX $z" >> $M/failures.log; continue; }
  touch $O/DATA_PREP_DONE
  {
    $PY $S/online_data_jobs.py --round 3 --preps triviaqa-$PFX-$z --budget 0.01
    $PY $S/online_data_jobs.py --round 3 --preps triviaqa-$PFX-$z --budget 0.1
    $PY $S/online_backend_jobs.py --preps triviaqa-$PFX-$z
  } > $M/T_b_${PFX}_$z.txt
  queue $M/T_b_${PFX}_$z.txt
  $PY $S/aker_real/aker_real_jobs.py --preps triviaqa-$PFX-$z --only data backend > $M/TA_b_${PFX}_$z.txt
  ( xargs -P 3 -I{} sh $W 20 {} < $M/TA_b_${PFX}_$z.txt ) &
done
wait
{
  $PY $S/online_round2_jobs.py --best-baselines 1 2 --preps triviaqa-$PFX-0.3 triviaqa-$PFX-0.6 triviaqa-$PFX-0.99
  $PY $S/online_round2_jobs.py --best-baselines 1 2 --budget 0.1 --preps triviaqa-$PFX-0.3 triviaqa-$PFX-0.6 triviaqa-$PFX-0.99
} > $M/T_4_$PFX.txt
xargs -P 6 -I{} sh $W 50 {} < $M/T_4_$PFX.txt
touch $DONE
