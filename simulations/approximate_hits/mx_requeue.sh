#!/bin/sh
# Re-queue matrix jobs that STARTED (their log exists) but have no result, are not running and
# did not crash with a traceback (killed for memory, given up by mx_run_job.sh, or stopped by
# hand), at most twice per job overall (/tmp/mx/requeued.log). Job lines come from all queue
# lists of the drivers; jobs that never started are left to their own queues.
# Run by mx_watchdog.sh every hour; safe to run by hand.
cd /local/home/jialind/ann-exact-caching || exit 1
M=/tmp/mx
W=simulations/approximate_hits/mx_run_job.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.
export LD_LIBRARY_PATH=/home/jialind/third_party/aker/build/lib:/home/jialind/third_party/boost-install/lib:/home/jialind/third_party/faiss-install/lib64:$LD_LIBRARY_PATH
touch $M/requeued.log
OUT=$M/requeue_$(date +%H%M%S).txt
: > $OUT
LISTS="$M/H_3.txt $M/L1_3.txt $M/L10_3.txt $M/A_3.txt $M/H_4.txt $M/D_real.txt $M/AD_real.txt $M/D_jobs.txt"
LISTS="$LISTS $(ls $M/T_*.txt $M/TA_*.txt $M/probe_jobs.txt $M/probe2_jobs.txt $M/paper_check.txt 2>/dev/null | tr '\n' ' ')"
RUNNING=$(ps -eo args | grep -E "^[^ ]*python[^ ]* -u [^ ]*(online_sim|eval_aker_real)\.py" | grep -oE -- "--out [^ ]+|--tag [^ ]+" | tr '\n' ' ')
cat $LISTS 2>/dev/null | awk '!seen[$0]++' | while IFS= read -r line; do
  j=$(printf '%s' "$line" | grep -o "\[ -f [^ ]*json" | cut -d' ' -f3)
  l=$(printf '%s' "$line" | grep -oE "> [^ ]*logs/[^ ]*\.txt" | cut -c3-)
  [ -n "$j" ] && [ ! -f "$j" ] && [ -n "$l" ] && [ -f "$l" ] || continue
  grep -q "Traceback" "$l" 2>/dev/null && continue                 # real failures are not retried
  case "$RUNNING" in *"--out $j"*|*"--out $(dirname $j)"*"--tag _${j##*/}"*) continue;; esac
  t=$(basename "$j" .json); tag=_${t#*_}
  printf '%s' "$RUNNING" | grep -q -- "--tag $tag --out $(dirname $j)" && continue
  [ "$(grep -cxF "$j" $M/requeued.log)" -ge 2 ] && continue
  printf '%s\n' "$j" >> $M/requeued.log
  printf '%s\n' "$line" >> $OUT
done
[ -s $OUT ] || { rm -f $OUT; exit 0; }
echo "$(date +%T) re-queued $(grep -c '' $OUT) jobs ($OUT)" >> $M/watchdog.log
grep -E -- "--budget 0\.1 .*--method (static|greedy|lru)|triviaqa.*--method (static|greedy|lru)" $OUT > $OUT.H
grep -vE -- "--budget 0\.1 .*--method (static|greedy|lru)|triviaqa.*--method (static|greedy|lru)" $OUT > $OUT.L
( xargs -P 6 -I{} sh $W 0 {} < $OUT.H ) &
( xargs -P 12 -I{} sh $W 0 {} < $OUT.L ) &
wait
